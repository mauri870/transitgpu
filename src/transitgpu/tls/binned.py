"""An approximate template search of Transit Least Squares on phase bins.

TLS (transitleastsquares 2.0.1, MIT, Michael Hippke and Rene Heller, https://github.com/hippke/tls) slides a
limb-darkened template over the phase-sorted points of each period and scores each start by a chi square. With equal
errors the chi square is the baseline of the points minus the gain 2 k AR - k^2 A2, with
AR = sum_j p_j w (1 - y_(i+j)), A2 = sum_j w p_j^2, k = mean * overshoot / SIGNAL_DEPTH and mean the mean of 1 - y
over the window of the duration; windows shallower than the shallowest depth that is fit are skipped.
transitgpu.tls.window computes this exactly.

Here the points of each period are folded into B phase bins by transitgpu's integer accumulation (no sort), the
template of each duration is averaged over the bins, and AR, A2 and the window sums are circular correlations of the
binned sums and counts with the binned template, its square and a box of the duration, by Fourier transforms of
length B. The statistic is TLS's up to the smoothing of the data and the template over a bin. B is the smallest power of
two for which B times the shortest duration (as a fraction of the period) is at least `bins_per_shortest`, taken at
the longest period that shares the value of B; the periods of a search are split into such bands (`banded_power`), each
with its own B and its own binned templates. `binned_power` alone gives every period of its argument the same B.

The search needs equal errors. It returns, for each period, the chi square, the template row and the depth of the
best start, as transitgpu.tls.window does.
"""
import time

import numpy as np
import torch
from transitleastsquares.backends import SearchBackend, SearchResult
from transitleastsquares.grid import T14

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch operations are used instead
    triton = None

from .. import core
from .screen import SIGNAL_DEPTH
from .window import NO_TLS_DEVICE, allowed_widths

if triton is not None:

    @triton.jit
    def _best_start(ar, a2, prefix_sum, prefix_count, allowed, ints, floats, best, at, depth_min, weight, bins, durations, block: tl.constexpr):
        """Write, for each period (row), duration and tile of start bins, the highest gain and its first start bin.

        The gain is 2 k AR - k^2 A2 over the starts whose window holds points, is deeper than `depth_min` and has a
        duration that TLS allows at the period; k is the window mean times the overshoot factor. The window sums come
        from the float64 prefix sums of the bins, wrapped at the end of the phase, and AR and A2 from the correlations.
        """
        tile = tl.program_id(0)
        duration = tl.program_id(1)
        row = tl.program_id(2)
        window = tl.load(ints + duration)
        factor = tl.load(floats + duration)
        start = tile * block + tl.arange(0, block)
        inside = start < bins
        stop = start + window
        wrapped = stop > bins
        stop = tl.where(wrapped, stop - bins, stop)
        base = row * (bins + 1)
        low_sum = tl.load(prefix_sum + base + start, mask=inside, other=0.0)
        low_count = tl.load(prefix_count + base + start, mask=inside, other=0.0)
        high_sum = tl.load(prefix_sum + base + stop, mask=inside, other=0.0)
        high_count = tl.load(prefix_count + base + stop, mask=inside, other=0.0)
        total_sum = tl.load(prefix_sum + base + bins)
        total_count = tl.load(prefix_count + base + bins)
        window_sum = tl.where(wrapped, total_sum - low_sum + high_sum, high_sum - low_sum).to(tl.float32)
        window_count = tl.where(wrapped, total_count - low_count + high_count, high_count - low_count).to(tl.float32)
        mean = window_sum / tl.maximum(window_count, 0.5)
        k = mean * factor
        index = (row * durations + duration) * bins + start
        gain = 2.0 * k * weight * tl.load(ar + index, mask=inside, other=0.0) - k * k * weight * tl.load(a2 + index, mask=inside, other=0.0)
        usable = inside & (window_count > 0.5) & (mean > depth_min) & (tl.load(allowed + row * durations + duration) != 0)
        gain = tl.where(usable, gain, float("-inf"))
        top = tl.max(gain, axis=0)
        first = tl.min(tl.where(gain == top, start, bins), axis=0)
        tiles = tl.num_programs(0)
        tl.store(best + (row * durations + duration) * tiles + tile, top)
        tl.store(at + (row * durations + duration) * tiles + tile, first)


# Periods folded by one launch of the integer accumulation. The correlations are taken `batch` periods at a time, and
# a launch over that few periods leaves most of the GPU idle; the sums of a period do not depend on the grouping.
FOLD_PERIODS = 512
MAXIMUM_PHASE_BINS = 1 << 15


def fold_sums(t, residual, periods, bins, checked=False):
    """Return the sum of `residual` and the number of points in each of `bins` phase bins, for each period (rows).

    On a GPU the sums are transitgpu.core's integer accumulation, after a check that they cannot overflow (as in
    core._check), which copies the largest value to the host; a caller that folds the same residual for many batches
    checks it once and passes checked=True. On a CPU the fold is in float64.
    """
    if t.is_cuda:
        if not checked:
            core.check_integer_range(residual.abs().max().item(), residual.numel())
        quantized = torch.round(residual / core.QUANTUM).to(torch.int64)
        found = core.accumulate(t, periods, bins, quantized, torch.ones_like(quantized))
        if found is not None:
            return found[0].to(torch.float64) * core.QUANTUM, found[1]
    index = core.fold_index(t, periods, bins)
    sums = torch.zeros((len(periods), bins), dtype=torch.float64, device=t.device).scatter_add_(1, index, residual.expand(len(periods), -1))
    counts = torch.zeros((len(periods), bins), dtype=torch.int64, device=t.device).scatter_add_(1, index, torch.ones_like(index))
    return sums, counts


def resampled(values, bins):
    """Return the area-weighted average of the samples `values` over each of `bins` equal parts of their span."""
    edges = np.linspace(0.0, len(values), bins + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    return np.diff(np.interp(edges, np.arange(len(values) + 1), cumulative)) * (bins / len(values))


def binned_templates(lc_arr, overview, bins):
    """Return the binned templates of each distinct width of TLS's template cache, in ascending order of width.

    For each width: its first cache row, the template (1 - lc_arr) and its square averaged over the bins of the
    duration, the overshoot and the number of bins.
    """
    widths = [int(w) for w in np.unique(overview["width_in_samples"])]
    found = {"widths": widths, "rows": [], "profile": [], "square": [], "overshoot": [], "bins": []}
    for width in widths:
        row = int(np.flatnonzero(overview["width_in_samples"] == width)[0])
        shape = 1.0 - np.asarray(lc_arr[row], dtype=np.float64)
        count = max(1, int(round(overview["duration"][row] * bins)))
        found["rows"].append(row)
        found["profile"].append(resampled(shape, count))
        found["square"].append(resampled(shape**2, count))
        found["overshoot"].append(float(overview["overshoot"][row]))
        found["bins"].append(count)
    return found


def bins_of(periods, bins_per_shortest, r_star_min, m_star_min):
    """Return B, the smallest power of two for which B times the shortest duration is at least bins_per_shortest.

    The shortest duration as a fraction of the period falls with the period (T14 / P scales as P^(-2/3)), so it is
    taken at the longest period. TLS passes the periods in shuffled order, so B must not depend on their order.
    """
    shortest = T14(R_s=r_star_min, M_s=m_star_min, P=float(np.max(periods)), small=True)
    return int(min(MAXIMUM_PHASE_BINS, 1 << int(np.ceil(np.log2(max(2.0, bins_per_shortest / shortest))))))


def binned_power(t, y, dy, periods, lc_arr, overview, transit_depth_min, r_star_min, r_star_max, m_star_min, m_star_max, device="cpu", bins_per_shortest=12, batch=64, bins=None):
    """Return (chi2, rows, depths) for `periods`: the lowest chi square over the durations and start bins of each.

    The output is that of transitgpu.tls.window.window_power, for the binned statistic. `bins` is the number of phase
    bins of every period; None gives the smallest that `bins_of` allows for the longest period of `periods`.
    """
    t, y, dy = (np.asarray(x, dtype=np.float64) for x in (t, y, dy))
    periods = np.asarray(periods, dtype=np.float64)
    if np.ptp(dy) != 0.0:
        raise ValueError("the binned search needs equal errors: the weights are one number")
    weight = float(1.0 / dy[0] ** 2)
    points = len(t)
    order = np.argsort(periods, kind="stable")  # neighbouring periods allow nearly the same widths; the results are put back in the given order
    periods = periods[order]
    bins = bins_of(periods, bins_per_shortest, r_star_min, m_star_min) if bins is None else bins
    templates = binned_templates(lc_arr, overview, bins)
    widths = templates["widths"]
    allowed = allowed_widths(periods, widths, points, float(t.max() - t.min()), r_star_min, r_star_max, m_star_min, m_star_max)  # (widths, periods)
    spectra = {}
    for name in ("profile", "square"):
        spectra[name] = torch.stack([torch.fft.rfft(torch.as_tensor(np.pad(v, (0, bins - len(v))), device=device, dtype=torch.float32)).conj() for v in templates[name]])
    spectra["box"] = torch.stack([torch.fft.rfft(torch.as_tensor(np.pad(np.ones(count), (0, bins - count)), device=device, dtype=torch.float32)).conj() for count in templates["bins"]])
    overshoot = torch.as_tensor(templates["overshoot"], device=device, dtype=torch.float32)[None, :, None]
    first_row = torch.as_tensor(templates["rows"], device=device)
    window_bins = torch.as_tensor(templates["bins"], device=device, dtype=torch.int64)
    factors = (torch.as_tensor(templates["overshoot"], device=device, dtype=torch.float32) / SIGNAL_DEPTH).contiguous()
    time_t = torch.as_tensor(t, device=device)
    residual = torch.as_tensor(1.0 - y, device=device)
    baseline = float(weight * np.sum((1.0 - y) ** 2))
    allowed_all = torch.as_tensor(np.ascontiguousarray(allowed.T), device=device)  # (periods, widths), one copy for the whole grid
    allowed_int8 = allowed_all.to(torch.int8)
    any_allowed = allowed_all.any(dim=1)
    periods_device = torch.as_tensor(periods, device=device)
    on_cuda = device != "cpu" and torch.device(device).type == "cuda"
    if on_cuda:
        core.check_integer_range(float(np.max(np.abs(1.0 - y))), len(y))  # once, on the host: the residual is the same for every batch
    kept = []
    fold_batch = batch * max(1, FOLD_PERIODS // batch)  # periods folded per launch, a multiple of the batch
    for start in range(0, len(periods), batch):
        rows_here = len(periods_device[start:start + batch])
        if start % fold_batch == 0:
            folded = fold_sums(time_t, residual, periods_device[start:start + fold_batch], bins, checked=on_cuda)
        offset = start % fold_batch
        sums, counts = folded[0][offset:offset + rows_here], folded[1][offset:offset + rows_here]
        # The widths a period allows are a contiguous range; only the range that some period of the batch allows is
        # correlated, since the others can never win.
        used = np.flatnonzero(allowed[:, start:start + batch].any(axis=1))
        low, high = (int(used[0]), int(used[-1]) + 1) if used.size else (0, 1)
        sum_spectrum = torch.fft.rfft(sums.to(torch.float32))[:, None, :]
        count_spectrum = torch.fft.rfft(counts.to(torch.float32))[:, None, :]
        correlate = lambda spectrum, template: torch.fft.irfft(spectrum * template[None, low:high], n=bins)  # (periods, widths, bins): start s sees bins s, s + 1, ...
        ar = correlate(sum_spectrum, spectra["profile"])
        a2 = correlate(count_spectrum, spectra["square"])
        allowed_here = allowed_all[start:start + batch, low:high]
        count_here = high - low
        prefix_sum = torch.cat([torch.zeros((rows_here, 1), dtype=torch.float64, device=device), torch.cumsum(sums, dim=1)], dim=1).contiguous()
        prefix_count = torch.cat([torch.zeros((rows_here, 1), dtype=torch.float64, device=device), torch.cumsum(counts.to(torch.float64), dim=1)], dim=1).contiguous()
        if triton is not None and on_cuda:
            block = 256
            tiles = triton.cdiv(bins, block)
            top = torch.empty((rows_here, count_here, tiles), dtype=torch.float32, device=device)
            first = torch.empty((rows_here, count_here, tiles), dtype=torch.int64, device=device)
            with core.on_device(device):
                _best_start[(tiles, count_here, rows_here)](ar, a2, prefix_sum, prefix_count, allowed_int8[start:start + batch, low:high].contiguous(), window_bins[low:high], factors[low:high], top, first, transit_depth_min, weight, bins, count_here, block=block)
            best, tile_at = top.reshape(rows_here, -1).max(dim=1)
            duration_of_best = tile_at // tiles
            start_of_best = first.reshape(rows_here, -1).gather(1, tile_at[:, None])[:, 0]
            at = duration_of_best * bins + start_of_best
            stop = start_of_best + window_bins[low:high][duration_of_best]
            wrapped = stop > bins
            stop = torch.where(wrapped, stop - bins, stop)
            rows_index = torch.arange(rows_here, device=device)
            window_sum = torch.where(wrapped, prefix_sum[:, bins] - prefix_sum[rows_index, start_of_best] + prefix_sum[rows_index, stop], prefix_sum[rows_index, stop] - prefix_sum[rows_index, start_of_best])
            window_count = torch.where(wrapped, prefix_count[:, bins] - prefix_count[rows_index, start_of_best] + prefix_count[rows_index, stop], prefix_count[rows_index, stop] - prefix_count[rows_index, start_of_best])
            mean_at = (window_sum / window_count.clamp(min=0.5)).to(torch.float32)
        else:
            window_sum = correlate(sum_spectrum, spectra["box"])
            window_count = correlate(count_spectrum, spectra["box"])
            mean = window_sum / window_count.clamp(min=0.5)
            k = mean * (overshoot[:, low:high] / SIGNAL_DEPTH)
            gain = 2.0 * k * weight * ar - k * k * weight * a2
            usable = (window_count > 0.5) & (mean > transit_depth_min) & allowed_here[:, :, None]
            gain = torch.where(usable, gain, torch.full_like(gain, -float("inf")))
            best, at = gain.reshape(rows_here, -1).max(dim=1)
            mean_at = mean.reshape(rows_here, -1).gather(1, at[:, None])[:, 0]
        improved = best > 0.0
        chi = baseline - torch.where(improved, best, torch.zeros_like(best)).to(torch.float64)
        chi = torch.where(any_allowed[start:start + batch], chi, torch.full_like(chi, float("inf")))
        duration_of = at // bins + low
        row = torch.where(improved, first_row[duration_of], torch.zeros_like(duration_of))
        depth = torch.where(improved, 1.0 - mean_at.to(torch.float64) * overshoot[0, :, 0].to(torch.float64)[duration_of], torch.zeros_like(chi))
        kept.append((chi, row, depth))
    unsorted = np.empty_like(order)
    unsorted[order] = np.arange(order.size)
    return tuple(torch.cat([k[i] for k in kept]).cpu().numpy()[unsorted] for i in range(3))


def bin_counts(periods, bins_per_shortest, r_star_min, m_star_min):
    """Return, for each period, the number of bins `bins_of` gives for that period alone; it never falls with the period."""
    return np.array([bins_of(np.array([period]), bins_per_shortest, r_star_min, m_star_min) for period in np.asarray(periods, dtype=np.float64)])


def banded_power(t, y, dy, periods, lc_arr, overview, transit_depth_min, r_star_min, r_star_max, m_star_min, m_star_max, device="cpu", bins_per_shortest=12, batch=64):
    """Return (chi2, rows, depths) for `periods` as `binned_power` does, with the bins of each band of periods its own.

    A band is the periods that have the same `bin_counts`; each is searched by `binned_power` with that number of bins,
    so every period keeps at least `bins_per_shortest` bins across its shortest duration and a short period is not
    folded into the bins that the longest period of the grid needs.
    """
    periods = np.asarray(periods, dtype=np.float64)
    counts = bin_counts(periods, bins_per_shortest, r_star_min, m_star_min)
    out = [np.empty(len(periods), dtype=dtype) for dtype in (np.float64, np.int64, np.float64)]
    for bins in np.unique(counts):
        inside = np.flatnonzero(counts == bins)
        found = binned_power(t, y, dy, periods[inside], lc_arr, overview, transit_depth_min, r_star_min, r_star_max, m_star_min, m_star_max, device=device, bins_per_shortest=bins_per_shortest, batch=batch, bins=int(bins))
        for column, values in zip(out, found):
            column[inside] = values
    return tuple(out)


class BinnedBackend(SearchBackend):
    """TLS's search of the trial periods on phase bins on a GPU, as a TLS backend.

    `power(backend=BinnedBackend())` gives TLS's periodogram, statistics and uncertainties from the chi squares of
    `binned_power`. The light curve is searched as given: TLS's pre-binning in time would end bins at gaps and leave
    the errors unequal.
    """

    name = "torch-binned"
    exact = True  # TLS pre-bins in time when exact is False; this backend bins in phase itself

    def __init__(self, device=None, bins_per_shortest=12, batch=64, banded=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bins_per_shortest, self.batch, self.banded = bins_per_shortest, batch, banded

    def search(self, problem, periods, use_threads=1, progress=None):
        if torch.device(self.device).type == "cpu":
            raise RuntimeError(NO_TLS_DEVICE)
        started = time.perf_counter()
        chi2, rows, depths = (banded_power if self.banded else binned_power)(problem.t, problem.y, problem.dy, periods, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max,
                                          device=self.device, bins_per_shortest=self.bins_per_shortest, batch=self.batch)
        self.seconds = time.perf_counter() - started
        if progress is not None:
            progress(len(periods))
        return SearchResult.from_unsorted(periods, chi2, rows, depths)
