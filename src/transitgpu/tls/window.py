"""The exact search of a window of trial periods of Transit Least Squares, in PyTorch.

The statistic is TLS's (transitleastsquares 2.0.1, `core.search_period`, MIT, Michael Hippke and Rene Heller, 2019,
https://github.com/hippke/tls). The way the trial periods and durations are laid out on a GPU follows GTLS (Hu et al.
2026, arXiv:2607.00348, MIT, https://github.com/Farthing-0/GTLS, release gputls 0.4.4); the code is separate from
GTLS (which needs CuPy, CUDA and NVML) and runs on PyTorch with CUDA or ROCm.

For a start i the chi square of a template is the constant residual plus 2 k B_i + k^2 C_i, with
k_i = overshoot * (1 - mean_i) / SIGNAL_DEPTH, B_i = sum_j w_(i+j) (d_(i+j) - 1) (1 - s_j) and
C_i = sum_j w_(i+j) (1 - s_j)^2, where d is the folded and wrapped flux, w the inverse squared errors and s the
template. Expanding the squared residuals inside the template gives these terms; the term of the data alone cancels
with the out-of-transit residual that TLS adds and the edge-effect correction that it subtracts. B and C are
correlations of the template with the folded data, taken at the starts that TLS tests: every
`int(width * T0_search_margin)`-th, where the mean in the window is deeper than the shallowest depth that is fit.
With equal errors C is a constant of the template.

The inputs are the arrays of TLS's SearchProblem (transitleastsquares.backends) and the output is, for each period,
the chi square, the template row and the depth at the minimum, as in a backend's SearchResult. `WindowBackend` is
this search as a backend of TLS.
"""
import time

import numpy as np
import torch
import torch.nn.functional as functional
from transitleastsquares.backends import SearchBackend, SearchResult
from transitleastsquares.grid import T14

from .fold import counting_order
from .prefix import prefix_arrays
from .screen import SEED_COARSEN, SIGNAL_DEPTH, Screen

NO_TLS_DEVICE = "transitgpu.tls needs a GPU; on a CPU use transitleastsquares"


def allowed_widths(periods, widths, points, length, r_star_min, r_star_max, m_star_min, m_star_max):
    """Return a (widths x periods) mask of the template widths, in samples, that TLS allows at each period.

    The limits are TLS's: grid.T14 with the stellar radius and mass limits.
    """
    duration_max = np.array([T14(R_s=r_star_max, M_s=m_star_max, P=q, small=False) for q in periods])
    duration_min = np.array([T14(R_s=r_star_min, M_s=m_star_min, P=q, small=True) for q in periods])
    correction = (length / periods + 1.0) / (length / periods)
    min_samples = np.floor(duration_min * points).astype(int)
    max_samples = np.ceil(duration_max * points * correction).astype(int)
    return np.array([(w >= min_samples) & (w <= max_samples) for w in widths])


WINDOW_BATCH_ELEMENTS = 1 << 23  # points times periods of a batch: about 0.7 GiB of sums and orders
BATCH_LIMITS = (64, 256)  # fewest periods in a batch (to share the launch overhead of a batch) and most


def default_batch(points):
    """Return the number of periods per batch of the screened search for a light curve of `points` points.

    It is the largest power of two that keeps points times periods under WINDOW_BATCH_ELEMENTS, clipped to
    BATCH_LIMITS. The sums of a period do not depend on the batch (transitgpu.tls.prefix), so the batch changes only
    the time and the memory.
    """
    fit = 1 << max(0, (WINDOW_BATCH_ELEMENTS // max(points, 1)).bit_length() - 1)
    return min(max(fit, BATCH_LIMITS[0]), BATCH_LIMITS[1])


EXACT_BATCH = 256  # the most periods in a batch of the exact search; the same chi square as a batch of 64 was checked up to this size
EXACT_BYTES_PER_POINT = 200  # device memory per point and period of a batch: 170 at most in the peaks measured (3.1 GB for 256 periods of 72,000 points)
EXACT_MEMORY_SHARE = 0.4  # the share of the free device memory that a batch may use
OFFSET_LIMIT = 1 << 30  # periods times starts of a batch: the screening kernel indexes its mask with 32-bit integers


def exact_batch(points, free_bytes):
    """Return the number of periods per batch of the exact search for `points` points and `free_bytes` of free device memory.

    It is EXACT_BATCH when the memory allows, and falls by multiples of 64 to the lower limit of 64 otherwise.
    """
    fit = int(EXACT_MEMORY_SHARE * free_bytes // (EXACT_BYTES_PER_POINT * max(points, 1)))
    return min(max(fit // 64 * 64, BATCH_LIMITS[0]), EXACT_BATCH)


def check_offsets(periods, starts):
    """Refuse a batch whose mask of starts is indexed past 32 bits: `periods` rows of `starts` columns."""
    if periods * starts >= OFFSET_LIMIT:
        raise ValueError(f"a batch of {periods} periods with {starts} starts a period would index its mask of starts past 2^30 with 32-bit integers; use a smaller batch")


def fitting_batch(periods, allowed, widths, stride, size):
    """Return the most periods, at most `periods`, that `check_offsets` accepts for the first of the `allowed` columns (widths x periods).

    The starts a period tests are those of the widths that some period of the batch allows, so a smaller batch can allow fewer
    widths; the batch is halved (to a multiple of 64 while it has more than 64 periods) until its mask of starts fits.
    """
    while periods > 1:
        used = [w for index, w in enumerate(widths) if allowed[index, :periods].any()]
        if periods * sum(len(range(0, size - w + 1, stride[w])) for w in used) < OFFSET_LIMIT:
            break
        periods = max(periods // 2 // 64 * 64, 64) if periods > 64 else (periods + 1) // 2  # multiples of 64 while there are more than 64
    return periods


def window_power(t, y, dy, periods, lc_arr, overview, transit_depth_min, r_star_min, r_star_max, m_star_min, m_star_max, t0_search_margin, device="cpu", dtype=torch.float64, batch=8, correlate="direct", compute=torch.float32, report=None):
    """Return (chi2, rows, depths) for `periods`: the lowest chi square over the durations and starts of each period.

    This is TLS's search_period for a batch of periods. `correlate` chooses how B and C are computed:
      "direct": by convolution in `dtype`.
      "fft": by Fourier transform in `compute` (float32 is fast on a GPU); the chi square is then the constant
        residual in `dtype` plus a correction in `compute`, whose precision decides the argmin over the starts.
      "screen": only for the starts whose bound on the gain (Cauchy-Schwarz and TLS's piecewise-linear screening, see
        `transitgpu.tls.screen`) is above the best gain found so far, as float64 dot products. The result is that of
        TLS's exact backend; it needs equal errors and `dtype` float64. `report`, a dict, receives the number of
        starts tested and of starts that survived the screening.

    On a CPU this is the reference that the GPU kernels are tested against, and it is slow. The templates and their
    spectra, the duration limits of every period and the stride and overshoot of every width are computed once,
    before the loop over the batches.
    """
    t, y, dy = (np.asarray(x, dtype=np.float64) for x in (t, y, dy))
    periods = np.asarray(periods, dtype=np.float64)
    n = len(t)
    widths = [int(w) for w in np.unique(overview["width_in_samples"])]
    max_width = max(widths)
    max_width += max_width % 2
    first_row = {w: int(np.flatnonzero(overview["width_in_samples"] == w)[0]) for w in widths}
    overshoot = {w: float(overview["overshoot"][first_row[w]]) for w in widths}
    stride = {w: (max(int(w / (1 / t0_search_margin)), 1) if t0_search_margin > 0 and w > t0_search_margin else 1) for w in widths}
    time_t = torch.as_tensor(t, device=device)
    y_t, dy_t = torch.as_tensor(y, device=device, dtype=dtype), torch.as_tensor(dy, device=device, dtype=dtype)
    length = t.max() - t.min()
    uniform = bool(np.ptp(dy) == 0.0)  # equal errors: the weights are one number and the square term is a constant of the template
    weight_unit = float(1.0 / dy[0] ** 2)
    allowed_all = allowed_widths(periods, widths, n, length, r_star_min, r_star_max, m_star_min, m_star_max)
    templates = {w: torch.as_tensor(np.asarray(lc_arr[first_row[w]], dtype=np.float64), device=device, dtype=dtype) for w in widths} if correlate == "direct" else {}
    spectra = {}
    screen = None
    if correlate == "screen":
        if dtype != torch.float64:
            raise ValueError("the screened search is exact only in float64")
        screen = Screen(y, dy, lc_arr, overview, widths, device)
        if report is not None:
            report.update(starts=0, survivors=0)
    if correlate == "fft":
        size_fft = 1 << ((n + max_width) - 1).bit_length()  # tested starts end at size - width: no wrap-around reaches them
        for w in widths:
            signal = torch.as_tensor(np.asarray(lc_arr[first_row[w]], dtype=np.float64), device=device, dtype=compute)
            spectra[w] = (torch.fft.rfft(1.0 - signal, n=size_fft).conj(), torch.fft.rfft((1.0 - signal) ** 2, n=size_fft).conj(), float(((1.0 - signal) ** 2).sum()))
    inf = float("inf")
    kept = []
    start = 0
    while start < len(periods):
        b = min(batch, len(periods) - start)
        if screen is not None:
            b = fitting_batch(b, allowed_all[:, start:start + b], widths, stride, n + max_width)
        chunk = periods[start:start + b]
        here, start = start, start + b
        p = torch.as_tensor(chunk, device=device)
        order = counting_order(time_t, p)
        if screen is not None:
            allowed_batch = allowed_all[:, here:here + b]
            residual, cumulative, squares, double, constant = prefix_arrays(y_t, order, weight_unit, screen.mean_rw, max_width, residual=True)
            kept.append(_screened_batch(screen, report, widths, first_row, stride, overshoot, allowed_batch, torch.as_tensor(allowed_batch, device=device), residual, cumulative, squares, double, constant, transit_depth_min, device))
            continue
        flux = y_t[order]
        patched = torch.cat([flux, flux[:, :max_width]], dim=1)
        dys = dy_t[order]
        weight = 1.0 / torch.cat([dys, dys[:, :max_width]], dim=1) ** 2
        a = patched - 1.0
        wa = weight * a
        constant = torch.cumsum((1.0 - flux) ** 2 / dys**2, dim=1)[:, -1]  # not .sum(dim=1): its order of addition depends on the alignment of the row, so on the position of the period in its batch
        cumulative = torch.cat([torch.zeros(b, 1, device=device, dtype=dtype), torch.cumsum(patched, dim=1)], dim=1)
        best = torch.full((b,), inf, device=device, dtype=dtype)
        best_row = torch.zeros(b, dtype=torch.int64, device=device)
        best_depth = torch.zeros(b, device=device, dtype=dtype)
        size = patched.shape[1]
        allowed_batch = allowed_all[:, here:here + b]
        allowed_device = torch.as_tensor(allowed_batch, device=device)  # one copy of the (widths, batch) mask
        if correlate == "fft":
            spectrum_a = torch.fft.rfft(wa.to(compute), n=size_fft)
            spectrum_w = None if uniform else torch.fft.rfft(weight.to(compute), n=size_fft)
        for index, w in enumerate(widths):
            if not allowed_batch[index].any():  # on the host: no wait for the device
                continue
            row = first_row[w]
            step = stride[w]
            starts = slice(0, size - w + 1, step)
            mean = 1.0 - (cumulative[:, slice(w, size + 1, step)] - cumulative[:, starts]) / float(w)
            count = mean.shape[1]
            k = mean * (overshoot[w] / SIGNAL_DEPTH)
            if correlate == "fft":
                cross = torch.fft.irfft(spectrum_a * spectra[w][0], n=size_fft)[:, starts]
                square = weight_unit * spectra[w][2] if uniform else torch.fft.irfft(spectrum_w * spectra[w][1], n=size_fft)[:, starts]
                kc = k.to(compute)
                delta = (2.0 * kc * cross + kc * kc * square).to(dtype)
                chi = constant[:, None] + delta
            else:
                s = 1.0 - templates[w]
                cross = functional.conv1d(wa[:, None, :], s[None, None, :], stride=step)[:, 0, :count]
                square = functional.conv1d(weight[:, None, :], (s * s)[None, None, :], stride=step)[:, 0, :count]
                chi = constant[:, None] + 2.0 * k * cross + k * k * square
            chi.masked_fill_(mean <= transit_depth_min, inf)
            low, at = chi.min(dim=1)
            improved = low < constant
            residual = torch.where(improved, low, constant)
            depth = torch.where(improved, 1.0 - mean.gather(1, at[:, None])[:, 0] * overshoot[w], torch.zeros_like(low))
            rows = torch.where(improved, torch.full_like(at, row), torch.zeros_like(at))
            take = allowed_device[index] & (residual < best)
            best = torch.where(take, residual, best)
            best_row = torch.where(take, rows, best_row)
            best_depth = torch.where(take, depth, best_depth)
        kept.append((best, best_row, best_depth))
    return (torch.cat([k[0] for k in kept]).cpu().numpy(), torch.cat([k[1] for k in kept]).cpu().numpy(), torch.cat([k[2] for k in kept]).cpu().numpy())


def _screened_batch(screen, report, widths, first_row, stride, overshoot, allowed_batch, allowed_device, residual, cumulative, squares, double, constant, transit_depth_min, device):
    """Return (chi2, rows, depths) for one batch of folded periods by the screened search of `transitgpu.tls.screen`.

    `residual` is weight * (1 - flux) of the folded flux (the first array of `transitgpu.tls.prefix.prefix_arrays` with residual=True), `cumulative`, `squares` and `double` its prefix sums.
    """
    b, size = residual.shape
    used = [index for index, w in enumerate(widths) if allowed_batch[index].any()]
    used_widths = [widths[index] for index in used]
    count = len(used)
    width_index = torch.as_tensor(used, device=device)
    width_tensor = torch.as_tensor(used_widths, device=device)
    stride_tensor = torch.as_tensor([stride[w] for w in used_widths], device=device)
    factor = torch.as_tensor([overshoot[w] / SIGNAL_DEPTH for w in used_widths], device=device, dtype=torch.float64)
    square = torch.as_tensor([screen.square[w] for w in used_widths], device=device, dtype=torch.float64)
    template_rows = torch.as_tensor([first_row[w] for w in used_widths], device=device)
    allowed_used = allowed_device[used]
    counts = [len(range(0, size - w + 1, stride[w])) for w in used_widths]
    edges = np.concatenate([[0], np.cumsum(counts)])
    edges_tensor = torch.as_tensor(edges, device=device)
    nowhere = float(edges[-1])

    def gain_of(rows, starts, owner):
        """Return the mean of each box and its gain 2 k AR - k^2 A2 (minus infinity where it is not deep enough).

        `owner` is the position in `used` of the width of each start.
        """
        w = width_tensor[owner]
        mean = 1.0 - (cumulative[rows, starts + w] - cumulative[rows, starts]) / w.to(torch.float64)
        k = mean * factor[owner]
        gain = 2.0 * k * screen.correlations(residual, rows, starts, width_index[owner], residual=True) - k * k * square[owner]
        return mean, torch.where(mean > transit_depth_min, gain, torch.full_like(gain, -float("inf")))

    # The incumbent of each period: the best gain of the deepest box of each width, among every SEED_COARSEN-th
    # tested start.
    coarse = {w: stride[w] * SEED_COARSEN for w in used_widths}
    deepest, _ = screen.deepest(cumulative, used_widths, coarse)
    seed_rows = torch.arange(b, device=device).repeat(count)  # a copy: expand(count, b).reshape(-1) is a view with stride 0 when b is 1
    seed_owner = torch.arange(count, device=device).repeat_interleave(b)
    coarse_tensor = torch.as_tensor([coarse[w] for w in used_widths], device=device)
    _, seed_gain = gain_of(seed_rows, (deepest * coarse_tensor[:, None]).reshape(-1), seed_owner)
    seed_gain = seed_gain.reshape(count, b)
    cutoff = torch.where(allowed_used, seed_gain, torch.zeros_like(seed_gain)).clamp(min=0.0).amax(dim=0)

    check_offsets(b, int(edges[-1]))
    keep_all = torch.empty((b, int(edges[-1])), dtype=torch.int8, device=device)
    screen.starts_kept_all(used_widths, stride, overshoot, size, transit_depth_min, cumulative, squares, double, cutoff, allowed_used, keep_all, edges)
    batch_index, column = keep_all.nonzero(as_tuple=True)  # row-major: by period, then by width and start
    if report is not None:
        report["starts"] += sum(int(((1.0 - (cumulative[:, slice(w, size + 1, stride[w])] - cumulative[:, slice(0, size - w + 1, stride[w])]) / float(w)) > transit_depth_min).sum()) for w in used_widths)
        report["survivors"] += len(batch_index)
    best_gain = torch.zeros(b, dtype=torch.float64, device=device)
    best_row = torch.zeros(b, dtype=torch.int64, device=device)
    best_depth = torch.zeros(b, dtype=torch.float64, device=device)
    if len(batch_index):
        owner = torch.bucketize(column, edges_tensor[1:], right=True)  # the position in `used` of the width of each survivor
        start = (column - edges_tensor[owner]) * stride_tensor[owner]
        mean, gains = gain_of(batch_index, start, owner)
        top = torch.full((b,), -float("inf"), dtype=torch.float64, device=device).scatter_reduce(0, batch_index, gains, "amax")
        at_top = gains >= top[batch_index]
        column_value = column.to(torch.float64)
        earliest = torch.full((b,), nowhere, dtype=torch.float64, device=device).scatter_reduce(0, batch_index[at_top], column_value[at_top], "amin")
        winner = at_top & (column_value == earliest[batch_index])
        improved = top > 0.0
        best_gain = torch.where(improved, top, best_gain)
        best_row = torch.where(improved, torch.zeros_like(best_row).scatter(0, batch_index[winner], template_rows[owner[winner]]), best_row)
        winner_depth = torch.zeros_like(best_depth).scatter(0, batch_index[winner], 1.0 - mean[winner] * factor[owner[winner]] * SIGNAL_DEPTH)
        best_depth = torch.where(improved, winner_depth, best_depth)
    any_allowed = torch.as_tensor(allowed_batch.any(axis=0), device=device)
    chi = torch.where(any_allowed, constant - best_gain, torch.full_like(constant, float("inf")))
    return chi, best_row, best_depth


class WindowBackend(SearchBackend):
    """TLS's exact search of the trial periods on a GPU, as a TLS backend.

    `power(backend=WindowBackend())` gives TLS's own periodogram, statistics and uncertainties from the chi squares
    of `window_power`. With correlate="auto" it uses the screened search for equal errors and the direct correlation
    otherwise, and refuses a CPU device; a named correlation also runs on a CPU.
    """

    name = "torch-window"
    exact = True

    def __init__(self, device=None, dtype=torch.float64, batch=None, correlate="auto"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype, self.batch, self.correlate = dtype, batch, correlate

    def search(self, problem, periods, use_threads=1, progress=None):
        started = time.perf_counter()
        if self.correlate == "auto" and torch.device(self.device).type == "cpu":
            raise RuntimeError(NO_TLS_DEVICE)
        periods = np.ascontiguousarray(np.sort(np.asarray(periods))[::-1])  # a batch holds the periods next to each other: TLS hands them over in a random order by default, a batch then needs the template widths of the whole range
        correlate, batch = self.correlate, self.batch or exact_batch(len(problem.t), torch.cuda.mem_get_info(self.device)[0] if torch.device(self.device).type == "cuda" else 0)
        if correlate == "auto":  # the screened search needs equal errors; otherwise the direct correlation in float64
            equal = bool(np.ptp(problem.dy) == 0.0)
            correlate, batch = ("screen", batch) if equal else ("direct", min(batch, 8))
        chi2, rows, depths = window_power(problem.t, problem.y, problem.dy, periods, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max,
                                          problem.T0_search_margin, device=self.device, dtype=self.dtype, batch=batch, correlate=correlate)
        self.seconds = time.perf_counter() - started  # the time of the search of the periods, without TLS's statistics
        if progress is not None:
            progress(len(periods))
        return SearchResult.from_unsorted(periods, chi2, rows, depths)
