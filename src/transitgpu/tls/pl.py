"""TLS's piecewise-linear search (fused-pl, Hippke and Heller 2019, transitleastsquares 2.0.1) on the device, on the unbinned folded points: backend "gpu-pl".

For a start i of a template of width w the gain over the constant residual is 2 k AR - k^2 A2, with k = (box mean) x overshoot / SIGNAL_DEPTH, AR = sum_j p_j r_(i+j) w and A2 = w sum_j p_j^2 (equal errors: a constant of the
template). fused-pl takes AR from a piecewise-linear fit of the template, about 20 terms of the double prefix sum of r w (`pl_fit` of `core_fused.py`, tolerance 2e-2, for templates of at least 64 samples), and from the direct dot
product for shorter ones; the chi square of a period is the constant residual less the largest gain over the starts and widths, with no refinement of the winner, and its depth is the box mean of the winning start. This module does
the same on the device with the templates, tolerance and minimum length of TLS's own `pl_templates`: the fold and the stable phase order (transitgpu.tls.fold), the prefix arrays (transitgpu.tls.prefix), one launch for the
gains of every start of every width of a batch of periods, which keeps the best start of each block of starts, and a reduction to the winner of each period. It evaluates every start of every allowed width: TLS's coarser grid of
start phases with local refinement and its scout durations (ideas L5 and L8 of its source) are approximations of the search and are not copied. Equal errors only.
"""
import time

import numpy as np
import torch

from transitgpu.core import on_device
from transitleastsquares.core_fused import flatten_templates, pl_templates

from .fold import counting_order
from .prefix import prefix_arrays
from .screen import SIGNAL_DEPTH
from .window import NO_TLS_DEVICE, allowed_widths, default_batch

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch operations are then used
    triton = None


FIT_TOLERANCE = 2e-2  # the tolerance of the fit of TLS's "fused-pl" backend
MINIMUM_LENGTH = 64  # the shortest template that "fused-pl" fits; the others are correlated directly
BLOCK = 256
COMPACT_BELOW_POINTS = 50_000  # below this many points a launch of the live blocks only is faster than one of every block of every width, which the dead ones leave at once; measured at 20,000, 100,000 and 200,000 points
REFINE_BLOCK = 8  # lanes of the refinement: the fine starts within one coarse step, at most 2 * (coarsen - 1) + 1 <= 8
SCOUT_EVERY = 4  # every fourth allowed duration (and the longest) searches every start, as TLS's fused-pl does for light curves of SCOUT_MIN_POINTS or more points (its scout_every)
SCOUT_MIN_POINTS = 2000  # transitleastsquares.core_fused.SCOUT_MIN_N
COARSEN = 3  # the coarse grid of starts of TLS's fused-pl (its t0_coarsen)

if triton is not None:

    @triton.jit
    def _gain_at(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, row, index, rows, start, inside, row_cumulative, row_double, row_patched, block: tl.constexpr):
        """The gain 2 k AR - k^2 A2 of the starts `start` of the width `index` of one period (minus infinity where a start is not `inside`, is not deeper than the least fit depth or the period does not allow the width). AR is
        the sum of the terms of the fit of the template on the double prefix sum, or, for a template without terms, the dot product with the template."""
        width = tl.load(ints + index * 8)
        terms = tl.load(ints + index * 8 + 3)
        first_term = tl.load(ints + index * 8 + 4)
        tap_count = tl.load(ints + index * 8 + 5)
        first_tap = tl.load(ints + index * 8 + 6)
        factor = tl.load(floats + index * 8)
        a2 = tl.load(floats + index * 8 + 1)
        constant = tl.load(floats + index * 8 + 2)
        depth_min = tl.load(floats + index * 8 + 3)
        weight = tl.load(floats + index * 8 + 4)
        inverse_width = 1.0 / width.to(tl.float64)
        before = tl.load(cumulative + row * row_cumulative + start, mask=inside, other=0.0)
        after = tl.load(cumulative + row * row_cumulative + start + width, mask=inside, other=0.0)
        mean = 1.0 - (after - before) * inverse_width
        k = mean * factor
        live = inside & (mean > depth_min) & (tl.load(allowed + index * rows + row) != 0)
        gain = tl.full([block], float("-inf"), dtype=tl.float64)
        if tl.max(live.to(tl.int32), axis=0) > 0:
            ar = tl.zeros([block], dtype=tl.float64)
            if terms > 0:
                for t in range(terms):
                    ar += tl.load(coefficients + first_term + t) * tl.load(double + row * row_double + start + tl.load(positions + first_term + t), mask=live, other=0.0)
                ar += constant
            else:
                for j in range(tap_count):
                    ar += tl.load(taps + first_tap + j) * (1.0 - tl.load(patched + row * row_patched + start + j, mask=live, other=0.0)) * weight
            gain = tl.where(live, 2.0 * k * ar - k * k * a2, float("-inf"))
        return gain

    @triton.jit
    def _gain_kernel(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, programs, best_gain, best_start, widths, rows, row_cumulative, row_double, row_patched, row_best,
                     block: tl.constexpr, compact: tl.constexpr):
        """The largest gain of a block of starts of one width of one period, with its start (minus infinity and -1 where no start of the block has a gain)."""
        program = tl.program_id(0)
        if compact:
            row = tl.load(programs + program * 4)
            index = tl.load(programs + program * 4 + 1)
            tile = tl.load(programs + program * 4 + 2)
            tile_all = tl.load(programs + program * 4 + 3)
        else:
            tile_all = program
            row = tl.program_id(1)
            index = 0
            for other in range(1, widths):
                index += (tile_all >= tl.load(programs + other)).to(tl.int32)
            tile = tile_all - tl.load(programs + index)
        step = tl.load(ints + index * 8 + 1)
        count = tl.load(ints + index * 8 + 2)
        column = tile * block + tl.arange(0, block)
        start = column * step
        gain = tl.full([block], float("-inf"), dtype=tl.float64)
        if compact or tl.load(allowed + index * rows + row) != 0:
            gain = _gain_at(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, row, index, rows, start, column < count, row_cumulative, row_double, row_patched, block)
        top = tl.max(gain, axis=0)
        first = tl.min(tl.where(gain == top, start, 2147483647), axis=0)
        tl.store(best_gain + row * row_best + tile_all, top)
        tl.store(best_start + row * row_best + tile_all, tl.where(top > float("-inf"), first, -1))

    @triton.jit
    def _refine_kernel(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, centre, refined_gain, refined_start, widths, rows, size, row_cumulative, row_double, row_patched,
                       block: tl.constexpr):
        """For the width and the period of the program: the largest gain among the starts on the fine grid that lie within one coarse step of the best coarse start `centre` (the coarse start itself is not repeated),
        and its start (minus infinity and -1 where the width is not coarsened or has no coarse start)."""
        index = tl.program_id(0)
        row = tl.program_id(1)
        width = tl.load(ints + index * 8)
        step = tl.load(ints + index * 8 + 1)
        ratio = tl.load(ints + index * 8 + 7)
        middle = tl.load(centre + row * widths + index)
        fine = step // ratio
        lane = tl.arange(0, block) - block // 2
        start = middle + lane * fine
        inside = (middle >= 0) & (ratio > 1) & (lane != 0) & (lane > -ratio) & (lane < ratio) & (start >= 0) & (start + width <= size)
        gain = _gain_at(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, row, index, rows, start, inside, row_cumulative, row_double, row_patched, block)
        top = tl.max(gain, axis=0)
        first = tl.min(tl.where(gain == top, start, 2147483647), axis=0)
        tl.store(refined_gain + row * widths + index, top)
        tl.store(refined_start + row * widths + index, tl.where(top > float("-inf"), first, -1))


    @triton.jit
    def _window_kernel(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, candidates, found_gain, found_start, widths, slots, rows, size, points, row_cumulative, row_double,
                       row_patched, block: tl.constexpr):
        """For the width and the period of the program: the largest gain among the coarse starts of the windows of every candidate centre (the best centres of the scout durations of the period, -1 for none), each
        window from the centre less 1.5 widths to the centre plus 0.5 widths, wrapped around the end of the phase, and its start (minus infinity and -1 where no start has a gain). The windows are laid end to end
        over the lanes."""
        index = tl.program_id(0)
        row = tl.program_id(1)
        width = tl.load(ints + index * 8)
        step = tl.load(ints + index * 8 + 1)
        span = (2 * width + 1 + step - 1) // step + 1
        best_gain = tl.full([], float("-inf"), dtype=tl.float64)
        best_start = tl.full([], -1, dtype=tl.int64)
        for first in range(0, slots * span, block):
            lane = first + tl.arange(0, block)
            other = lane // span
            centre = tl.load(candidates + row * slots + other, mask=other < slots, other=-1)
            low = centre - width // 2 - width
            low = tl.where(low < 0, -((-low + step - 1) // step) * step, (low // step) * step)
            at = low + (lane - other * span) * step
            at = tl.where(at < 0, at + points, tl.where(at + width > size, at - points, at))
            at = (tl.maximum(at, 0) // step) * step
            inside = (centre >= 0) & (at + width <= size)
            gain = _gain_at(cumulative, double, patched, allowed, positions, coefficients, taps, ints, floats, row, index, rows, at, inside, row_cumulative, row_double, row_patched, block)
            top = tl.max(gain, axis=0)
            earliest = tl.min(tl.where(gain == top, at, 2147483647), axis=0)
            better = top > best_gain
            best_gain = tl.where(better, top, best_gain)
            best_start = tl.where(better, earliest, best_start)
        tl.store(found_gain + row * widths + index, best_gain)
        tl.store(found_start + row * widths + index, best_start.to(tl.int32))


def live_programs(scouts, tile_first, tile_count, pairs, programs):
    """The programs of the gain kernel of a batch, one for each block of starts of each scout width of each period, as rows of (period, width index, block in the width, block in the launch) on the device, so that
    the launch covers the live blocks only: the grid of every block of every width held 150,000 programs a batch at 20,000 points, nine tenths of which left at once. `scouts` is the mask (widths by periods) on the
    device, `tile_first` and `tile_count` the first block and the number of blocks of each width, and `pairs` and `programs` the number of live (period, width) pairs and of programs, which the caller knows on the
    host from the mask it made, so that nothing waits for the device."""
    period, width = torch.nonzero_static(scouts.T != 0, size=pairs).unbind(1)
    counts = tile_count[width]
    within = torch.arange(programs, device=scouts.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts, output_size=programs)
    width_of = torch.repeat_interleave(width, counts, output_size=programs)
    return torch.stack([torch.repeat_interleave(period, counts, output_size=programs), width_of, within, tile_first[width_of] + within], dim=1).to(torch.int32)


class PLTemplates:
    """The terms of the fit of every width's template and the constants of its gain, made once for a problem and kept on the device."""

    def __init__(self, y, dy, lc_arr, overview, widths, device, margin, eps=FIT_TOLERANCE, min_length=MINIMUM_LENGTH):
        if np.ptp(dy) != 0.0:
            raise ValueError("the piecewise-linear search needs equal errors: with weights that differ the square term of the gain is not a constant of the template")
        self.weight = float(1.0 / dy[0] ** 2)
        self.mean_rw = float(np.mean((1.0 - y) * self.weight))  # the centering of the double prefix sums (transitgpu.tls.prefix)
        flat = flatten_templates(lc_arr, overview)
        unique, self.template_rows, offsets, lengths, profile, self.overshoot, _ = flat
        if [int(w) for w in unique] != [int(w) for w in widths]:
            raise ValueError("the widths of the templates are not the widths of the search")
        n, offset, n2, offset2, positions, coefficients, plain_sum, _ = pl_templates(flat, margin, min_length, 1 << 62, eps)
        self.terms = [int(v) for v in n]
        self.first_term = [int(v) for v in offset]
        self.constant = [float(plain_sum[i]) * self.mean_rw for i in range(len(widths))]
        self.square = [self.weight * float(np.sum(profile[offsets[i]:offsets[i] + lengths[i]] ** 2)) for i in range(len(widths))]
        self.tap_count = [0 if self.terms[i] > 0 else int(lengths[i]) for i in range(len(widths))]
        self.first_tap = [int(offsets[i]) for i in range(len(widths))]
        self.positions = torch.as_tensor(positions if len(positions) else [0], dtype=torch.int64, device=device)
        self.coefficients = torch.as_tensor(coefficients if len(coefficients) else [0.0], dtype=torch.float64, device=device)
        self.taps = torch.as_tensor(profile, dtype=torch.float64, device=device)


def _tables(templates, widths, stride, size, depth_min, device, coarsen, margin):
    """The per-width integers and floats of the kernels, the index of the first block of each width and the blocks of each width as a padded index, on the device; they depend on the number of samples of a row
    only, so one set serves every batch. A width is searched on the grid of `coarsen` times its stride when int(coarsen * margin * width) >= 2, as TLS's coarse grid of starts is."""
    ints, floats, tiles = [], [], [0]
    for index, w in enumerate(widths):
        ratio = coarsen if coarsen > 1 and int(coarsen * margin * w) >= 2 else 1
        step = stride[w] * ratio
        count = len(range(0, size - w + 1, step))
        ints.append([w, step, count, templates.terms[index], templates.first_term[index], templates.tap_count[index], templates.first_tap[index], ratio])
        floats.append([float(templates.overshoot[index]) / SIGNAL_DEPTH, templates.square[index], templates.constant[index], depth_min, templates.weight, 0.0, 0.0, 0.0])
        tiles.append(tiles[-1] + triton.cdiv(count, BLOCK))
    longest = max(b - a for a, b in zip(tiles[:-1], tiles[1:]))
    gather = np.zeros((len(widths), longest), dtype=np.int64)
    valid = np.zeros((len(widths), longest), dtype=bool)
    for index in range(len(widths)):
        blocks = tiles[index + 1] - tiles[index]
        gather[index, :blocks] = np.arange(tiles[index], tiles[index + 1])
        valid[index, :blocks] = True
    return (torch.as_tensor(ints, dtype=torch.int64, device=device), torch.as_tensor(floats, dtype=torch.float64, device=device), torch.as_tensor(tiles, dtype=torch.int64, device=device), tiles[-1],
            torch.as_tensor(gather, device=device), torch.as_tensor(valid, device=device), torch.as_tensor(np.asarray(tiles[:-1]), device=device), torch.as_tensor(np.diff(tiles), device=device), np.diff(tiles))


def _gains(templates, tables, patched, cumulative, double, scouts, scouts_on_host, followers, slots, points, device):
    """(gain, start) of the best start of every width for each period of the batch (batch by widths). The scout durations (`scouts`, widths by periods) search the coarse grid of starts and the fine starts around
    the best coarse start; the others (`followers`, None when the batch has none) search windows around the best starts of the scouts, then the fine starts around the best coarse start of the window."""
    ints, floats, tiles, block_count, gather, valid, tile_first, tile_count, tile_count_on_host = tables
    rows, widths = cumulative.shape[0], len(ints)
    size = cumulative.shape[1] - 1
    arguments = (templates.positions, templates.coefficients, templates.taps, ints, floats)
    strides = (cumulative.stride(0), double.stride(0), patched.stride(0))
    compact = points < COMPACT_BELOW_POINTS
    best_gain = torch.full((rows, block_count), float("-inf"), dtype=torch.float64, device=device) if compact else torch.empty((rows, block_count), dtype=torch.float64, device=device)
    best_start = torch.full((rows, block_count), -1, dtype=torch.int32, device=device) if compact else torch.empty((rows, block_count), dtype=torch.int32, device=device)
    if compact:
        pairs = int(scouts_on_host.sum())
        count = int((scouts_on_host * tile_count_on_host[:, None]).sum())
        launch = (count,)
        listed = live_programs(scouts, tile_first, tile_count, pairs, count) if count else None
    else:
        launch, listed = (block_count, rows), tiles
    if launch[0]:
        with on_device(device):
            _gain_kernel[launch](cumulative, double, patched, scouts, *arguments, listed, best_gain, best_start, widths, rows, *strides, best_gain.stride(0), block=BLOCK, compact=compact)
    per_width = torch.where(valid[None], best_gain[:, gather], float("-inf"))  # rows by widths by blocks
    coarse_gain, at = per_width.max(dim=2)
    coarse_start = best_start.gather(1, gather[torch.arange(widths, device=device)[None, :], at])
    found = (coarse_gain, coarse_start)
    refined = _refine(cumulative, double, patched, scouts, arguments, strides, coarse_start, size, device)
    result = _better(found, refined)
    if followers is not None:
        centres = torch.where(result[1] >= 0, result[1] + (ints[:, 0] // 2)[None, :].to(torch.int32), -1).to(torch.int32)
        centres = centres.sort(dim=1, descending=True).values[:, :slots].contiguous()  # the centres of the scouts first
        window_gain = torch.empty((rows, widths), dtype=torch.float64, device=device)
        window_start = torch.empty((rows, widths), dtype=torch.int32, device=device)
        with on_device(device):
            _window_kernel[(widths, rows)](cumulative, double, patched, followers, *arguments, centres, window_gain, window_start, widths, slots, rows, size, points, *strides, block=BLOCK)
        result = _better(result, _better((window_gain, window_start), _refine(cumulative, double, patched, followers, arguments, strides, window_start, size, device)))
    return result


def _refine(cumulative, double, patched, allowed, arguments, strides, centre, size, device):
    rows, widths = centre.shape
    refined_gain = torch.empty((rows, widths), dtype=torch.float64, device=device)
    refined_start = torch.empty((rows, widths), dtype=torch.int32, device=device)
    with on_device(device):
        _refine_kernel[(widths, rows)](cumulative, double, patched, allowed, *arguments, centre.contiguous(), refined_gain, refined_start, widths, rows, size, *strides, block=REFINE_BLOCK)
    return refined_gain, refined_start


def _better(first, second):
    """The larger gain of two (gain, start) pairs of the same shape, the first where they are equal."""
    take = second[0] > first[0]
    return torch.where(take, second[0], first[0]), torch.where(take, second[1], first[1])


def pl_power(t, y, dy, periods, lc_arr, overview, transit_depth_min, r_star_min, r_star_max, m_star_min, m_star_max, t0_search_margin, device="cuda", batch=None, eps=FIT_TOLERANCE, min_length=MINIMUM_LENGTH, coarsen=COARSEN, scout_every=SCOUT_EVERY):
    """(chi2, rows, depths) for each period of `periods`, as transitgpu.tls.window.window_power gives them for the exact statistic, by the piecewise-linear gains of this module. `eps` and `min_length` are the
    tolerance of the fit and the shortest template that is fitted; eps 0 takes every sample of the template as a knot, which is the exact correlation (for the tests). `coarsen` is the ratio of the coarse grid of
    starts to the fine one (1: every start is searched); `scout_every` is the spacing of the scout durations (0: every duration is a scout)."""
    if triton is None or torch.device(device).type != "cuda":
        raise RuntimeError(NO_TLS_DEVICE)
    t, y, dy = (np.asarray(x, dtype=np.float64) for x in (t, y, dy))
    periods = np.asarray(periods, dtype=np.float64)
    n = len(t)
    widths = [int(w) for w in np.unique(overview["width_in_samples"])]
    max_width = max(widths)
    max_width += max_width % 2
    first_row = {w: int(np.flatnonzero(overview["width_in_samples"] == w)[0]) for w in widths}
    stride = {w: (max(int(w / (1 / t0_search_margin)), 1) if t0_search_margin > 0 and w > t0_search_margin else 1) for w in widths}
    templates = PLTemplates(y, dy, lc_arr, overview, widths, device, t0_search_margin, eps, min_length)
    allowed_all = allowed_widths(periods, widths, n, t.max() - t.min(), r_star_min, r_star_max, m_star_min, m_star_max)  # (widths, periods)
    time_t, y_t = torch.as_tensor(t, device=device), torch.as_tensor(y, device=device, dtype=torch.float64)
    row_of_width = torch.as_tensor([first_row[w] for w in widths], dtype=torch.int64, device=device)
    overshoot = torch.as_tensor(templates.overshoot, dtype=torch.float64, device=device)
    ordinal = np.cumsum(allowed_all, axis=0) - 1  # the rank of an allowed width among the allowed widths of its period
    count = allowed_all.sum(axis=0)
    scouts_only = scout_every > 1 and n >= SCOUT_MIN_POINTS
    is_scout = allowed_all & ((not scouts_only) | (count[None, :] <= scout_every) | (ordinal % max(scout_every, 1) == 0) | (ordinal == count[None, :] - 1))
    scout_all = torch.as_tensor(np.ascontiguousarray(is_scout), device=device).to(torch.int8).contiguous()
    follower_all = torch.as_tensor(np.ascontiguousarray(allowed_all & ~is_scout), device=device).to(torch.int8).contiguous()
    has_followers = bool((allowed_all & ~is_scout).any())
    slots = max(int(is_scout.sum(axis=0).max()), 1)  # the most scout durations of a period: the candidate centres of a period
    any_allowed_all = torch.as_tensor(allowed_all.any(axis=0), device=device)
    width_tensor = torch.as_tensor(widths, dtype=torch.int64, device=device)
    batch = batch or default_batch(n)
    tables = None
    kept = []
    for start in range(0, len(periods), batch):
        chunk = periods[start:start + batch]
        b = len(chunk)
        order = counting_order(time_t, torch.as_tensor(chunk, device=device))
        patched, cumulative, squares, double, constant = prefix_arrays(y_t, order, templates.weight, templates.mean_rw, max_width)
        size = patched.shape[1]
        if tables is None:
            tables = _tables(templates, widths, stride, size, transit_depth_min, device, coarsen, t0_search_margin)
        scouts = scout_all[:, start:start + b].contiguous()
        followers = follower_all[:, start:start + b].contiguous() if has_followers else None
        width_gain, width_start = _gains(templates, tables, patched, cumulative, double, scouts, is_scout[:, start:start + b], followers, slots, n, device)
        top, owner = width_gain.max(dim=1)
        improved = top > 0.0
        winning_start = width_start.gather(1, owner[:, None])[:, 0].clamp(min=0).to(torch.int64)
        width_of = width_tensor[owner]
        index = torch.arange(b, device=device)
        mean = 1.0 - (cumulative[index, winning_start + width_of] - cumulative[index, winning_start]) / width_of.to(torch.float64)
        depth = torch.where(improved, 1.0 - mean * overshoot[owner], torch.zeros_like(top))
        rows = torch.where(improved, row_of_width[owner], torch.zeros_like(owner))
        any_allowed = any_allowed_all[start:start + b]
        chi = torch.where(any_allowed, constant - torch.where(improved, top, torch.zeros_like(top)), torch.full_like(constant, float("inf")))
        kept.append((chi, rows, depth))
    return (torch.cat([k[0] for k in kept]).cpu().numpy(), torch.cat([k[1] for k in kept]).cpu().numpy(), torch.cat([k[2] for k in kept]).cpu().numpy())


class PLBackend:
    """TLS's fused-pl search of the trial periods on the device: `power(backend="gpu-pl")` gives TLS's own periodogram, statistics and uncertainties from the chi square of `pl_power`. Built on demand so that importing this module does not import TLS's backends."""

    def __new__(cls, *arguments, **options):
        from transitleastsquares.backends import SearchBackend, SearchResult

        class _Backend(SearchBackend):
            name = "torch-pl"
            exact = False

            def __init__(self, device=None, batch=None, eps=FIT_TOLERANCE, min_length=MINIMUM_LENGTH):
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.batch, self.eps, self.min_length = batch, eps, min_length

            def search(self, problem, periods, use_threads=1, progress=None):
                started = time.perf_counter()
                if torch.device(self.device).type == "cpu":
                    raise RuntimeError(NO_TLS_DEVICE)
                chi2, rows, depths = pl_power(problem.t, problem.y, problem.dy, periods, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min,
                                              problem.M_star_max, problem.T0_search_margin, device=self.device, batch=self.batch, eps=self.eps, min_length=self.min_length)
                self.seconds = time.perf_counter() - started
                if progress is not None:
                    progress(len(periods))
                return SearchResult.from_unsorted(periods, chi2, rows, depths)

        return _Backend(*arguments, **options)
