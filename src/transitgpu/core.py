"""Box least squares period search on a GPU, with PyTorch.

For each trial period the points are folded into phase bins, the flux and the number of points of each bin are
summed, and every window of a few widths over the bins is scored as a box dip with the statistic of a box search
with equal errors (Kovacs, Zucker & Mazeh 2002, A&A 391, 369):

    (sum of the flux in the window)^2 / (sigma^2 n (1 - n / N))

for a window that holds n of the N points and has a lower mean than the whole. This is the square of the signal to
noise of the depth with the level outside the box free, i.e. the square of the "snr" objective of
astropy.timeseries.BoxLeastSquares; astropy's default "likelihood" objective, 0.5 * ivar_in * depth^2, leaves out
the factor (1 - n / N). Nothing is sorted and no template is slid over the points, so the work per period is one
pass over the points plus a few hundred bins.

The trial periods are TLS's grid (transitgpu.grids.period_grid with its default star). The durations of a period run
from the shortest that TLS allows (T14 with its default stellar limits) in steps of DURATION_STEP up to the first
step that reaches the longest, so the longest box can be one step (10%) wider than TLS's longest duration.

The SDE is the z-score of the best statistic of each period over all periods, (max - mean) / std. It is not TLS's
SDE (TLS detrends its spectrum) and has no threshold of its own; one can be estimated from the SDEs of block
shuffles of the light curve, which `search_many` searches together on one time axis.

This module is the engine and not the public interface: `transitgpu.box_search`, `transitgpu.search` and
`transitgpu.search_many` call it after checking for a GPU. The functions here also run on a CPU, slowly. Callers
that choose the device themselves import from `transitgpu.engine`, whose names and signatures are stable.
"""

import warnings

import numpy as np
from transitgpu import grids
from transitgpu.grids import T14, duration_limit_masses, period_grid, period_uncertainty

# Bins across the shortest duration of a period. A box that is not aligned with the bins loses statistic; more bins
# lose less and cost more time.
MINIMUM_BINS = 12
MAXIMUM_BINS = 4096
MINIMUM_BINS_PER_PERIOD = 128
# Ratio of consecutive durations, as in TLS's grid (grids.DURATION_GRID_STEP).
DURATION_STEP = 1.1
# Fewest points inside a box, and outside it, for the box to be scored.
MINIMUM_POINTS = 3
# Periods per batch: as many as keep the (periods x points) fold table under this many elements (480 MB), between 8
# and 512. The one-pass accumulation stores no fold table and takes ONE_PASS_BATCH periods at a time.
BATCH_ELEMENTS = 60_000_000
ONE_PASS_BATCH = 512
# The flux is summed as integers of QUANTUM (1e-9 of the level). Floating-point atomic additions on a GPU depend on
# their order and integer additions do not, so a search gives the same numbers every time. A CPU and a GPU still
# differ in the last digits of the float arithmetic that follows.
QUANTUM = 1e-9
WEIGHT_UNIT = 64  # integer units of a weight equal to the median weight; integer weights keep the sums integer
MAXIMUM_WEIGHT = 4096  # largest weight, in those units: 64 times the median weight
PACK_SHIFT = 32  # a weighted count holds the number of points above this bit and the sum of their weights below it
# Duration steps scored together: as many as keep each (periods x steps x bins) tensor under this many elements.
STEP_ELEMENTS = 20_000_000
# Precision of the box statistic: "float32" (the default) gives the same best period as "float64" and a spectrum
# within a relative 1e-5 of it.
PRECISION = {"statistic": "float32"}
PRECISION_CHOICES = {"statistic": ("float64", "float32")}


# torch.compile keeps this many compiled variants of a function and runs it eagerly after that, silently; the eager scorer
# computes the float32 statistic differently from the compiled one (a relative 1e-7), so two searches of a long process
# would differ. A scorer is compiled once for each combination of the statistic's precision, weights or none, and dips
# only or dips and bumps: 8 combinations, which hold 17 compiled entries (measured over every combination of bumps,
# precision, weights, one flux or many, a spectrum, cuts, at two sizes). The limit is raised to this number with a margin
# of almost 4 and never lowered. Name in torch 2.12, with `cache_size_limit` as its alias:
# https://github.com/pytorch/pytorch/blob/v2.12.0/torch/_dynamo/config.py (`recompile_limit`)
RECOMPILE_LIMIT = 64


def raise_recompile_limit():
    """Raise torch's limit of compiled variants per function to RECOMPILE_LIMIT; a higher limit that the caller set stays. Older torch names it `cache_size_limit`."""
    import torch._dynamo.config as config

    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    if getattr(config, name) < RECOMPILE_LIMIT:
        setattr(config, name, RECOMPILE_LIMIT)


# The compiled fold_index: None until its first use on a GPU, False when compilation is not available.
_COMPILED = None


def device_name(device=None):
    """Return the name of the GPU that searches run on: `device`, else the current CUDA device; None without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    index = torch.device(device).index if device is not None else None
    return torch.cuda.get_device_name(torch.cuda.current_device() if index is None else index)


def on_device(device):
    """Return a context in which `device` is the current CUDA device; for a CPU it does nothing.

    Triton launches its kernels on the current device, which is cuda:0 until changed, not on the device of the tensors.
    """
    import contextlib

    import torch

    device = torch.device(device)
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def fold_index(t, p, bins, shifts=None):
    """Return the phase bin of every point of `t` (days) for every period of `p`, as a (periods x points) table.

    The phase is t times the frequency 1 / p, in float64, scaled to `bins`, truncated and clamped to the bins. A
    product is used rather than t / p because float64 division is slow on GPUs; the kernels compute the same product,
    so every path gives the same bins. `shifts`, a (periods x points) table in days, moves each point before the fold
    (a null signal template).
    """
    import torch

    moved = t.unsqueeze(0) if shifts is None else t.unsqueeze(0) - shifts
    return (torch.remainder(moved * (1.0 / p).unsqueeze(1), 1.0) * bins).long().clamp_(0, bins - 1)


_ACCUMULATE = None
_ACCUMULATE_MANY = None
# Fluxes added by one launch of accumulate_many. Its sums are (fluxes x periods x bins) int64 and stop fitting in the
# caches beyond a few fluxes.
FLUX_CHUNK = 8


def _accumulate_kernel():
    """Return (kernel, triton) for `accumulate`, built on first use, or False when Triton is not available."""
    global _ACCUMULATE
    if _ACCUMULATE is None:
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def kernel(t_ptr, f_ptr, q_ptr, w_ptr, sums_ptr, counts_ptr, n, bins, BLOCK: tl.constexpr):
                row = tl.program_id(1)
                offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                inside = offsets < n
                x = tl.load(t_ptr + offsets, mask=inside, other=0.0) * tl.load(f_ptr + row)
                phase = x - tl.floor(x)
                bin_of = tl.minimum(tl.maximum((phase * bins).to(tl.int32), 0), bins - 1)
                target = row * bins + bin_of
                tl.atomic_add(sums_ptr + target, tl.load(q_ptr + offsets, mask=inside, other=0), mask=inside)
                tl.atomic_add(counts_ptr + target, tl.load(w_ptr + offsets, mask=inside, other=0), mask=inside)

            _ACCUMULATE = (kernel, triton)
        except Exception as error:
            warnings.warn(f"the fused fold and accumulation is not available ({type(error).__name__}: {error}); the fold table and the scatters are used", stacklevel=2)
            _ACCUMULATE = False
    return _ACCUMULATE


def _accumulate_many_kernel():
    """Return (kernel, triton) for `accumulate_many`, built on first use, or False when Triton is not available."""
    global _ACCUMULATE_MANY
    if _ACCUMULATE_MANY is None:
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def kernel(t_ptr, f_ptr, q_ptr, w_ptr, sums_ptr, counts_ptr, n, bins, rows, fluxes, BLOCK: tl.constexpr):
                row = tl.program_id(1)
                offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                inside = offsets < n
                x = tl.load(t_ptr + offsets, mask=inside, other=0.0) * tl.load(f_ptr + row)
                phase = x - tl.floor(x)
                bin_of = tl.minimum(tl.maximum((phase * bins).to(tl.int32), 0), bins - 1)
                for f in range(0, fluxes):
                    target = ((f * rows + row).to(tl.int64) * bins) + bin_of
                    tl.atomic_add(sums_ptr + target, tl.load(q_ptr + f * n + offsets, mask=inside, other=0), mask=inside)
                    tl.atomic_add(counts_ptr + target, tl.load(w_ptr + f * n + offsets, mask=inside, other=0), mask=inside)

            _ACCUMULATE_MANY = (kernel, triton)
        except Exception as error:
            warnings.warn(f"the one-pass accumulation is not available ({type(error).__name__}: {error}); the fold table and the scatters are used", stacklevel=2)
            _ACCUMULATE_MANY = False
    return _ACCUMULATE_MANY


_ACCUMULATE_ONCE = None


def _accumulate_once_kernel():
    """Return (kernel, triton) for `accumulate_many` with one set of weights, or False when Triton is not available.

    It is the kernel of _accumulate_many_kernel with the counts added for the first flux only.
    """
    global _ACCUMULATE_ONCE
    if _ACCUMULATE_ONCE is None:
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def kernel(t_ptr, f_ptr, q_ptr, w_ptr, sums_ptr, counts_ptr, n, bins, rows, fluxes, BLOCK: tl.constexpr):
                row = tl.program_id(1)
                offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                inside = offsets < n
                x = tl.load(t_ptr + offsets, mask=inside, other=0.0) * tl.load(f_ptr + row)
                phase = x - tl.floor(x)
                bin_of = tl.minimum(tl.maximum((phase * bins).to(tl.int32), 0), bins - 1)
                for f in range(0, fluxes):
                    target = ((f * rows + row).to(tl.int64) * bins) + bin_of
                    tl.atomic_add(sums_ptr + target, tl.load(q_ptr + f * n + offsets, mask=inside, other=0), mask=inside)
                    if f == 0:
                        tl.atomic_add(counts_ptr + target, tl.load(w_ptr + offsets, mask=inside, other=0), mask=inside)

            _ACCUMULATE_ONCE = (kernel, triton)
        except Exception as error:
            warnings.warn(f"the accumulation with the counts added once is not available ({type(error).__name__}: {error}); the one-pass accumulation is used", stacklevel=2)
            _ACCUMULATE_ONCE = False
    return _ACCUMULATE_ONCE


def accumulate_many(t, p, bins, quantized, weight, shared_weight=False):
    """Return the integer sums and counts of the phase bins of several fluxes, as two (fluxes x periods x bins) tensors.

    `quantized` and `weight` are (fluxes x points) int64 tensors. Each thread computes the bin of one point for one
    period once and adds that point of every flux to it, so the fold table is never stored. The integers equal those
    of `fold_index` and scatter_add_, to the bit. `shared_weight` says that every flux has the same weights; the
    counts are then added for the first flux only. Return None when Triton is not available.
    """
    import torch

    from transitgpu import hip

    found_hip = hip.accumulate_many(t, p, bins, quantized, weight, shared_weight)
    if found_hip is not None:
        return found_hip
    once = _accumulate_once_kernel() if shared_weight and quantized.shape[0] > 1 else False
    found = once or _accumulate_many_kernel()
    if not found:
        return None
    kernel, triton = found
    rows, n, fluxes = p.numel(), t.numel(), quantized.shape[0]
    sums = torch.zeros(fluxes, rows, bins, dtype=torch.int64, device=t.device)
    counts = torch.zeros(fluxes, rows, bins, dtype=torch.int64, device=t.device)
    block = 1024
    with on_device(t.device):
        kernel[(triton.cdiv(n, block), rows)](t, 1.0 / p, quantized.contiguous(), weight.contiguous(), sums, counts, n, bins, rows, fluxes, BLOCK=block)
    if once:  # every flux has the counts of the first, since the weights are equal: a view, not a copy
        counts = counts[:1].expand(fluxes, -1, -1)
    return sums, counts


def accumulate(t, p, bins, quantized, weight):
    """Return the integer sums and counts of the phase bins of one flux, as two (periods x bins) int64 tensors.

    Each thread computes the bin of its point with the float64 product of `fold_index` and adds the point to it,
    so the fold table is never written. Integer additions do not depend on their order, so the result equals
    `fold_index` and scatter_add_ to the bit. `t` must be non-negative. Return None when Triton is not available.
    """
    import torch

    found = _accumulate_kernel()
    if not found:
        return None
    kernel, triton = found
    rows, n = p.numel(), t.numel()
    sums = torch.zeros(rows, bins, dtype=torch.int64, device=t.device)
    counts = torch.zeros(rows, bins, dtype=torch.int64, device=t.device)
    block = 1024
    with on_device(t.device):
        kernel[(triton.cdiv(n, block), rows)](t, 1.0 / p, quantized, weight, sums, counts, n, bins, BLOCK=block)
    return sums, counts


def fold(t, p, bins, shifts=None):
    """Return `fold_index(t, p, bins, shifts)`, compiled by torch.compile into one kernel on a GPU.

    On a CPU, or when compilation fails (with a warning, once), the eager `fold_index` is used; the table is the same.
    """
    global _COMPILED
    import torch

    if t.device.type != "cpu" and _COMPILED is not False:
        raise_recompile_limit()
        try:
            if _COMPILED is None:
                _COMPILED = torch.compile(fold_index, dynamic=True)
            return _COMPILED(t, p, bins, shifts)
        except torch.OutOfMemoryError:  # not a compiler failure: the eager passes need the same memory
            raise
        except Exception as error:  # no compiler, or a backend that cannot build the kernel
            warnings.warn(f"torch.compile of the fold failed ({type(error).__name__}: {error}); using the slower eager passes", stacklevel=2)
            _COMPILED = False
    return fold_index(t, p, bins, shifts)


# The compiled score_steps: None until its first use on a GPU, False when compilation is not available.
_SCORE_COMPILED = None


def score_steps(s2, c2, position, width, ok, n_points, bins, single=False, packed_total=None, bumps=False):
    """Return the best box of each period at each duration step, as (statistic, start bin) tensors of rows x steps.

    `s2` and `c2` are the cumulative sums of the flux and the counts of the bins, with the first bins repeated at the
    end so that a window wraps around the phase. A window is scored as inside^2 / (n (1 - n / N)) when it is a dip
    with at least MINIMUM_POINTS points inside and outside it, and 0 otherwise. With `packed_total` (the sum of all the
    weights) the counts are packed (PACK_SHIFT): the raw counts are held to MINIMUM_POINTS and the statistic is the
    weighted one, inside^2 / (W_in (1 - W_in / W)). `single` computes the statistic in float32. With `bumps` the best
    window that is a bump (a mean above the whole's) is scored from the same sums, with the same statistic, and the
    result is (statistic, start bin, bump statistic, bump start bin); the first two are those of `bumps=False`.
    """
    import torch

    end = position.unsqueeze(1) + width.unsqueeze(2)
    inside_exact = torch.gather(s2, 1, end.reshape(end.shape[0], -1)).reshape(end.shape) - s2[:, :bins].unsqueeze(1)
    number_exact = torch.gather(c2, 1, end.reshape(end.shape[0], -1)).reshape(end.shape) - c2[:, :bins].unsqueeze(1)
    inside = (inside_exact.float() if single else inside_exact.double()) * QUANTUM
    if packed_total is None:
        number = number_exact.float() if single else number_exact.double()
        enough = (number_exact >= MINIMUM_POINTS) & (number_exact <= n_points - MINIMUM_POINTS)
        total = n_points
    else:
        weight_exact = number_exact & ((1 << PACK_SHIFT) - 1)
        raw = number_exact >> PACK_SHIFT
        number = weight_exact.float() if single else weight_exact.double()
        enough = (raw >= MINIMUM_POINTS) & (raw <= n_points - MINIMUM_POINTS)
        total = packed_total
    statistic = torch.where((inside_exact < 0) & enough & ok.unsqueeze(2), inside * inside / (number * (1 - number / total)), torch.zeros_like(inside))
    if not bumps:
        return statistic.max(dim=2)
    raised = torch.where((inside_exact > 0) & enough & ok.unsqueeze(2), inside * inside / (number * (1 - number / total)), torch.zeros_like(inside))
    dip, bump = statistic.max(dim=2), raised.max(dim=2)
    return dip.values, dip.indices, bump.values, bump.indices


def score(s2, c2, position, width, ok, n_points, bins, single=False, packed_total=None, bumps=False):
    """Return `score_steps(...)`, compiled by torch.compile into one kernel on a GPU.

    The compiled kernel orders the floating-point operations differently, so the z-scores differ from the eager
    passes in the last digits (about 2e-15). On a CPU, or when compilation fails (with a warning, once), the eager
    passes are used.
    """
    global _SCORE_COMPILED
    import torch

    if s2.device.type != "cpu" and _SCORE_COMPILED is not False:
        raise_recompile_limit()
        try:
            if _SCORE_COMPILED is None:
                _SCORE_COMPILED = torch.compile(score_steps, dynamic=True)
            return _SCORE_COMPILED(s2, c2, position, width, ok, n_points, bins, single, packed_total, bumps)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            warnings.warn(f"torch.compile of the scoring failed ({type(error).__name__}: {error}); using the slower eager passes", stacklevel=2)
            _SCORE_COMPILED = False
    return score_steps(s2, c2, position, width, ok, n_points, bins, single, packed_total, bumps)


# The compiled advance_winner: None until its first use on a GPU, False when compilation is not available.
_ADVANCE_COMPILED = None


def advance_winner(winner, start, width_of, number_in, sum_in, value_of_step, where_of_step, width, s2, c2, rows_index, packed):
    """Return the best box of each period so far after one chunk of duration steps, as (statistic, start bin, width, sum, count) tensors of one value per period.

    `value_of_step` and `where_of_step` are the best box of each period at each step of the chunk (`score_steps`), `width` the widths of those steps, `s2` and `c2` the cumulative sums, and `packed` says that the counts
    hold weights (PACK_SHIFT). The best step of a chunk is the first of its maxima and a later chunk wins only if strictly better, so the result does not depend on the size of the chunk. The sum and the count of
    the chosen box are read from the cumulative sums, as the scorer computed them.
    """
    import torch

    value, step = value_of_step.double().max(dim=1)
    where = where_of_step.gather(1, step.unsqueeze(1)).squeeze(1)
    chosen_width = width.gather(1, step.unsqueeze(1)).squeeze(1)
    chosen_end = where + chosen_width
    better = value > winner
    inside_count = c2[rows_index, chosen_end] - c2[rows_index, where]
    return (torch.where(better, value, winner), torch.where(better, where, start), torch.where(better, chosen_width, width_of),
            torch.where(better, (s2[rows_index, chosen_end] - s2[rows_index, where]).double() * QUANTUM, sum_in),
            torch.where(better, (inside_count & ((1 << PACK_SHIFT) - 1) if packed else inside_count).double(), number_in))


def advance(winner, start, width_of, number_in, sum_in, value_of_step, where_of_step, width, s2, c2, rows_index, packed):
    """Return `advance_winner(...)`, compiled by torch.compile into a few kernels on a GPU (the eager passes launch about twenty); the integers are the same and so are the doubles, which are products and differences of exact values."""
    global _ADVANCE_COMPILED
    import torch

    if s2.device.type != "cpu" and _ADVANCE_COMPILED is not False:
        raise_recompile_limit()
        try:
            if _ADVANCE_COMPILED is None:
                _ADVANCE_COMPILED = torch.compile(advance_winner, dynamic=True)
            return _ADVANCE_COMPILED(winner, start, width_of, number_in, sum_in, value_of_step, where_of_step, width, s2, c2, rows_index, packed)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            warnings.warn(f"torch.compile of the update of the best boxes failed ({type(error).__name__}: {error}); using the slower eager passes", stacklevel=2)
            _ADVANCE_COMPILED = False
    return advance_winner(winner, start, width_of, number_in, sum_in, value_of_step, where_of_step, width, s2, c2, rows_index, packed)


def duration_fractions(periods):
    """Return the shortest and the longest transit duration TLS allows at each period, as fractions of the period.

    The limits are those of TLS's default stellar radius and mass range.
    """
    m_short, m_long = duration_limit_masses(grids.R_STAR_MIN, grids.R_STAR_MAX, grids.M_STAR_MIN, grids.M_STAR_MAX)
    shortest = np.array([T14(R_s=grids.R_STAR_MIN, M_s=m_short, P=p, small=True) for p in periods])
    longest = np.array([T14(R_s=grids.R_STAR_MAX, M_s=m_long, P=p, small=False) for p in periods])
    return shortest, longest


def grid_of(time, flux, period_min, period_max, oversampling_factor=None, n_transits_min=None):
    """Return the sorted trial periods (days) of a search: TLS's grid for the time span of the finite points.

    `oversampling_factor` and `n_transits_min` default to TLS's constants.
    """
    keep = np.isfinite(time) & np.isfinite(flux)
    span = float(np.max(time[keep]) - np.min(time[keep]))
    return np.sort(np.asarray(period_grid(R_star=grids.R_STAR, M_star=grids.M_STAR, time_span=span, period_min=period_min, period_max=period_max,
                                          oversampling_factor=grids.OVERSAMPLING_FACTOR if oversampling_factor is None else oversampling_factor,
                                          n_transits_min=grids.N_TRANSITS_MIN if n_transits_min is None else n_transits_min)))


def check_integer_range(maximum, count):
    """Raise ValueError when `count` values of absolute value up to `maximum` could overflow the integer sums.

    Such a value is a corrupt point: the 64-bit sums of QUANTUM would wrap and give a wrong spectrum without a word.
    """
    if float(maximum) / QUANTUM * count >= 2**62:
        raise ValueError(f"a flux value of {float(maximum):.3g} is too large for the integer sums of {QUANTUM:g}: the flux must be flattened and normalized")


def check_weighted_sums(centered, integer):
    """Raise ValueError when the integer sums of a weighted search could overflow.

    The sum of a box is at most the sum over the points of |quantized flux| times the integer weight, which must fit
    in 64 bits (a wrapped cumulative sum gives the right difference only then), and the sum of all the weights must
    fit in the PACK_SHIFT bits that hold the weights of a bin. `centered` is the flux minus its level and `integer`
    the integer weights.
    """
    box_bound = float(np.abs(np.round(np.asarray(centered, dtype=np.float64) / QUANTUM)) @ np.asarray(integer, dtype=np.float64))
    if box_bound >= 2**63:
        raise ValueError(f"the weights make the integer sums of the flux too large to hold ({box_bound:.3g} units of {QUANTUM:g} against {2**63:.3g}): the flux must be flattened and normalized")
    if float(np.sum(integer)) >= 2**PACK_SHIFT:
        raise ValueError(f"the sum of the weights of {len(integer):,} points, {float(np.sum(integer)):.3g} units, does not fit in the {PACK_SHIFT} bits that hold the weights of a bin: search fewer points or weights nearer to each other")


def _check(time, flux):
    """Raise ValueError for a flux that cannot be searched.

    On the finite points: none, too few, no time span, no variance, or a value that overflows the integer sums.
    """
    keep = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[keep], flux[keep]
    if time.size == 0:
        raise ValueError("no finite points to search")
    if time.size < 2 * MINIMUM_POINTS:
        raise ValueError(f"a search needs at least {2 * MINIMUM_POINTS} points (a box of {MINIMUM_POINTS} and as many outside it); got {time.size}")
    if not float(time.max() - time.min()) > 0:
        raise ValueError("the times do not span any time")
    if not np.var(flux) > 0:
        raise ValueError("the flux has no variance: there is nothing to search")
    check_integer_range(np.abs(flux - flux.mean()).max(), time.size)


def search_many(time, fluxes, period_min, period_max, spectrum=False, device=None, batch=None, minimum_bins=MINIMUM_BINS, duration_step=DURATION_STEP, periods=None, duration_hours=None, labeled=True, boxes=False, cuts=None, template=None, precision=None, weights=None, oversampling_factor=None, n_transits_min=None, bumps=False):
    """Search each flux of `fluxes`, all on the time axis `time`, and return a list with one result dict per flux.

    The fluxes share one grid of trial periods (`periods`, else TLS's grid for the first flux), so their SDEs are
    z-scores over the same periods. Points that are not finite are dropped.

    Arguments:
      time, fluxes: times in days and a list of flux arrays of the same length.
      period_min, period_max: the range of the period grid, in days.
      spectrum: add `periods` and `power` to each result.
      device: a torch device; by default the first GPU, else the CPU.
      batch: periods searched together (by default ONE_PASS_BATCH with the one-pass accumulation, else from
        BATCH_ELEMENTS); it changes the time and not the result.
      minimum_bins: bins across the shortest duration of a period (MINIMUM_BINS).
      duration_step: ratio of consecutive durations (DURATION_STEP).
      periods: the trial periods, in place of TLS's grid.
      duration_hours: a pair (low, high); only the duration steps longer than low and up to high hours are scored.
      labeled: put the index of the flux in the message of a ValueError about it.
      boxes: add `centers` and `widths_days`, the best box of every trial period.
      cuts: a list of periods; the result of each flux is then a list with one result per cut, over the periods of
        the grid up to that cut (its SDE is a z-score over those periods only).
      template: an object whose method shifts(t, p) returns the days by which each point is moved before the fold.
        It replaces the periodic template and gives one draw of a null of the data, for example the null signal
        template of Robnik & Seljak 2025 (arXiv:2509.07409).
      precision: a dict that overrides PRECISION; "statistic" is "float32" (the default) or "float64".
      weights: one weight per point of the time axis (or one row per flux), relative to the median weight, for
        example 1 / variance for sectors of unequal scatter. Equal weights give the unweighted search. Each weight is
        rounded to a multiple of 1/WEIGHT_UNIT (1/64) of the median weight and clipped to between 1/64 and 64 times it.
        A weight of zero masks its point, which is dropped as a point with no flux is. With one flux and a matrix of
        weights of one row for each result, the flux is searched once for each row, from one fold of the times;
        each result is that of a search of the flux with that row.
      oversampling_factor, n_transits_min: arguments of TLS's period grid (its constants when None); used only
        without `periods`.
      bumps: also search for bumps (a window whose mean is above the whole's) from the same fold and the same sums, with
        the same grid and statistic. Each result then has a `bump` entry, a result dict of the same fields that equals the
        result of a search of the negated flux: its SDE is a z-score over the best bump statistic of each period, and its
        depth is negative for a brightening.

    Each result holds plain floats (periods in days, durations in hours, depths in ppm):
      period_days, t0_days: the best period and the center of one transit of its best box.
      duration_hr: the width of the best box; duration_corrected_hr equals it and fill_factor is 1.
      depth_ppm, depth_mean_ppm: the mean depth of the box against the flux outside it.
      snr: the depth over its white-noise error, sigma / sqrt(n (1 - n / N)) for n of the N points in the box.
      sde, transit_count, period_uncertainty_days (TLS's half width at half maximum of the highest peak),
      period_min_searched_days, period_max_searched_days.
      periods, power (with spectrum=True): the trial periods and their z-scores, whose maximum is the SDE.
      centers, widths_days (with boxes=True): the center and the width, in days, of the best box of every period.

    With `bumps`, `bump` holds those fields for the bumps; its centers, widths and spectrum are those of the bumps.

    Raise ValueError for a flux with nothing to search, a cut that leaves fewer than two periods, an unknown
    precision, and weights that are not finite or are negative or whose integer sums could overflow.
    """
    import torch

    choice = {**PRECISION, **(precision or {})}
    for name, value in choice.items():
        if name not in PRECISION_CHOICES:
            raise ValueError(f"precision has no option {name!r}; the options are {sorted(PRECISION_CHOICES)}")
        if value not in PRECISION_CHOICES[name]:
            raise ValueError(f"precision {name}={value!r} is not one of {PRECISION_CHOICES[name]}")
    single = choice["statistic"] == "float32"
    time = np.asarray(time, dtype=float)
    fluxes = [np.asarray(flux, dtype=float) for flux in fluxes]
    if not fluxes:
        return []
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 2 and len(fluxes) == 1 and weights.shape[0] > 1:  # one flux and a row of weights for each result
            fluxes = fluxes * weights.shape[0]
    for k, flux in enumerate(fluxes):
        if flux.shape != time.shape:
            raise ValueError(f"flux {k} does not have the length of the time axis ({flux.size} against {time.size}): the fluxes of a search of many are on one time axis")
        try:
            _check(time, flux)
        except ValueError as error:
            raise ValueError(f"flux {k}: {error}" if labeled else str(error)) from None
    if periods is None:
        periods = grid_of(time, fluxes[0], period_min, period_max, oversampling_factor, n_transits_min)
    periods = np.sort(np.asarray(periods, dtype=float))
    ends = None if cuts is None else [int(np.searchsorted(periods, cut, side="right")) for cut in cuts]
    if ends is not None and min(ends) < 2:
        raise ValueError(f"a cut of {min(cuts)} d leaves fewer than two periods of the grid, which starts at {periods.min()} d")
    on_axis = np.isfinite(time)  # a point without a time is dropped; a point with a time and no flux has zero weight
    time_axis = time[on_axis]
    shortest, longest = duration_fractions(periods)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t = torch.as_tensor(time_axis, dtype=torch.float64, device=device)
    # The one-pass accumulation needs a CUDA device, no template, non-negative times and Triton; otherwise the fold
    # table and scatter_add_ give the same integers.
    together = template is None and device.type == "cuda" and float(time_axis.min()) >= 0.0 and bool(_accumulate_many_kernel())
    if not batch:  # a template adds a table of shifts as large as the fold table
        batch = ONE_PASS_BATCH if together else int(np.clip(BATCH_ELEMENTS // (time_axis.size * (1 if template is None else 2)), 8, 512))
    rows_of = []
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, (len(fluxes), weights.size))
        if weights.shape != (len(fluxes), time.size):
            raise ValueError(f"weights have the shape {weights.shape}, need one for each point of the time axis ({time.size}) or one row for each flux ({len(fluxes)})")
    for index, flux in enumerate(fluxes):
        flux = flux[on_axis]
        finite = np.isfinite(flux)
        if weights is None:
            level = float(flux[finite].mean())
            quantized = torch.round(torch.as_tensor(np.where(finite, flux - level, 0.0), dtype=torch.float64, device=device) / QUANTUM).to(torch.int64)
            row = {"finite": finite, "level": level, "sigma2": float(np.var(flux[finite])), "n": int(finite.sum()), "quantized": quantized, "weight": torch.as_tensor(finite, dtype=torch.int64, device=device), "packed_total": None}
        else:
            given = weights[index][on_axis]
            if not np.all(np.isfinite(given[finite])) or np.any(given[finite] < 0):
                raise ValueError(f"flux {index}: weights must be finite and not negative at every point with a flux" if labeled else "weights must be finite and not negative at every point with a flux")
            finite = finite & (given > 0)  # a weight of zero masks the point: it is dropped as a point with no flux is
            if finite.sum() < 2 * MINIMUM_POINTS:
                raise ValueError(f"flux {index}: fewer than {2 * MINIMUM_POINTS} points have a weight above zero" if labeled else f"fewer than {2 * MINIMUM_POINTS} points have a weight above zero")
            integer = np.where(finite, np.clip(np.round(WEIGHT_UNIT * given / np.median(given[finite])), 1, MAXIMUM_WEIGHT), 0).astype(np.int64)
            total = int(integer.sum())
            level = float((integer * np.where(finite, flux, 0.0)).sum() / total)
            centered = np.where(finite, flux - level, 0.0)
            check_weighted_sums(centered, integer)
            quantized = torch.round(torch.as_tensor(centered, dtype=torch.float64, device=device) / QUANTUM).to(torch.int64) * torch.as_tensor(integer, dtype=torch.int64, device=device)
            packed = torch.as_tensor(np.where(finite, integer + (1 << PACK_SHIFT), 0), dtype=torch.int64, device=device)
            row = {"finite": finite, "level": level, "sigma2": float(WEIGHT_UNIT * (integer * centered**2).sum() / total), "n": int(finite.sum()), "quantized": quantized, "weight": packed, "packed_total": total}
        rows_of.append(row)
    count = len(fluxes)
    found_by_side = [{"best": np.zeros((count, periods.size)), "sum": np.zeros((count, periods.size)), "start": np.zeros((count, periods.size), dtype=np.int64),
                      "width": np.zeros((count, periods.size), dtype=np.int64), "number": np.ones((count, periods.size))} for _ in range(2 if bumps else 1)]  # the dips, then the bumps
    best_bins = np.ones(periods.size, dtype=np.int64)

    # The number of phase bins depends on the period alone (a power of two from its shortest duration). The periods
    # with the same number of bins are searched together, `batch` at a time.
    bins_of = np.clip(2 ** np.ceil(np.log2(minimum_bins / shortest)), MINIMUM_BINS_PER_PERIOD, MAXIMUM_BINS).astype(np.int64)
    groups = [(int(bins), np.flatnonzero(bins_of == bins)) for bins in np.unique(bins_of)]
    quantized_all = weight_all = None  # every flux, stacked for accumulate_many
    shared_weight = all(torch.equal(row["weight"], rows_of[0]["weight"]) for row in rows_of[1:])  # block shuffles of a star share its weights
    for bins, members in groups:
        for first in range(0, members.size, batch):
            index = members[first:first + batch]
            rows = index.size
            p = torch.as_tensor(periods[index], dtype=torch.float64, device=device)
            steps = np.maximum(1, np.ceil(np.log(longest[index] / shortest[index]) / np.log(duration_step)).astype(int) + 1)
            fractions = shortest[index, None] * duration_step ** np.arange(steps.max())[None, :]
            valid_step = np.arange(steps.max())[None, :] < steps[:, None]
            if duration_hours is not None:
                hours = fractions * periods[index, None] * 24
                valid_step = valid_step & (hours > duration_hours[0]) & (hours <= duration_hours[1])
            widths = np.where(valid_step, np.clip(np.round(fractions * bins).astype(np.int64), 1, bins), 1)  # a masked step must still index inside the cumulative sums
            wmax = int(widths.max())
            widths_device, valid_device = torch.as_tensor(widths, device=device), torch.as_tensor(valid_step, device=device)
            steps_total = widths.shape[1]
            chunk = int(np.clip(STEP_ELEMENTS // (rows * bins), 1, steps_total))
            position = torch.arange(bins, device=device).unsqueeze(0)
            rows_index = torch.arange(rows, device=device)

            if together and quantized_all is None:
                quantized_all, weight_all = torch.stack([row["quantized"] for row in rows_of]), torch.stack([row["weight"] for row in rows_of])
            table = None if together else fold(t, p, bins, None if template is None else template.shifts(t, p))  # shared by every flux
            results = []
            c2_shared = None  # the cumulative counts of the batch, when every flux has the same weights
            for k, row in enumerate(rows_of):
                n_points = row["n"]
                if together:
                    if k % FLUX_CHUNK == 0:
                        sums_all, counts_all = accumulate_many(t, p, bins, quantized_all[k:k + FLUX_CHUNK], weight_all[k:k + FLUX_CHUNK], shared_weight=shared_weight)
                    sums, counts = sums_all[k % FLUX_CHUNK], counts_all[k % FLUX_CHUNK]
                else:
                    sums = torch.zeros(rows, bins, dtype=torch.int64, device=device).scatter_add_(1, table, row["quantized"].unsqueeze(0).expand(rows, time_axis.size))
                    if c2_shared is None:
                        counts = torch.zeros(rows, bins, dtype=torch.int64, device=device).scatter_add_(1, table, row["weight"].unsqueeze(0).expand(rows, time_axis.size))
                s2 = torch.nn.functional.pad(torch.cat([sums, sums[:, :wmax]], dim=1).cumsum(dim=1, dtype=torch.int64), (1, 0))
                if c2_shared is None:
                    c2 = torch.nn.functional.pad(torch.cat([counts, counts[:, :wmax]], dim=1).cumsum(dim=1, dtype=torch.int64), (1, 0))
                    if shared_weight:
                        c2_shared = c2
                else:
                    c2 = c2_shared

                states = [{"winner": torch.zeros(rows, dtype=torch.float64, device=device), "start": torch.zeros(rows, dtype=torch.int64, device=device), "width": torch.ones(rows, dtype=torch.int64, device=device),
                           "sum": torch.zeros(rows, dtype=torch.float64, device=device), "number": torch.ones(rows, dtype=torch.float64, device=device)} for _ in range(2 if bumps else 1)]  # the dips, then the bumps
                # The duration steps are scored a chunk at a time, as (rows x steps x bins) tensors, so that the dozen
                # small kernels of a step are launched once per chunk and not once per step. The best step of a chunk
                # is the first of its maxima and a later chunk wins only if strictly better, so the result does not
                # depend on the size of the chunk.
                for first_step in range(0, steps_total, chunk):
                    width = widths_device[:, first_step:first_step + chunk]
                    ok = valid_device[:, first_step:first_step + chunk]
                    scored = score(s2, c2, position, width, ok, n_points, bins, single, row["packed_total"], bumps)
                    for state, side in zip(states, range(len(states))):
                        state["winner"], state["start"], state["width"], state["sum"], state["number"] = advance(
                            state["winner"], state["start"], state["width"], state["number"], state["sum"], scored[2 * side], scored[2 * side + 1], width, s2, c2, rows_index, row["packed_total"] is not None)
                results.append(torch.stack([torch.stack([state["winner"], state["start"].double(), state["width"].double(), state["sum"], state["number"]]) for state in states]))  # kept on the device: one copy per batch
            read = torch.stack(results).cpu().numpy()  # fluxes x sides x 5 x rows; the starts and widths are integers below 2**53, exact in float64
            for k, row in enumerate(rows_of):
                for side, arrays in enumerate(found_by_side):
                    arrays["best"][k, index] = read[k, side, 0] / row["sigma2"]
                    arrays["start"][k, index] = read[k, side, 1].astype(np.int64)
                    arrays["width"][k, index] = read[k, side, 2].astype(np.int64)
                    arrays["sum"][k, index] = read[k, side, 3]
                    arrays["number"][k, index] = read[k, side, 4]
            best_bins[index] = bins

    def result(k, row, end, side=0):
        """Return the result of flux k over the first `end` periods of the grid: of its dips (side 0), or of its bumps (side 1), which is the result of a search of the negated flux."""
        kept = time_axis[row["finite"]]
        best, best_sum, best_start, best_width, best_number = (found_by_side[side][name] for name in ("best", "sum", "start", "width", "number"))
        sign = 1.0 if side == 0 else -1.0  # the negated flux has the level and the sums of opposite sign
        level, box_sum = sign * row["level"], sign * best_sum
        statistic, grid = best[k, :end], periods[:end]
        deviation = statistic.std()
        power = (statistic - statistic.mean()) / deviation if deviation > 0 else np.zeros_like(statistic)
        top = int(np.argmax(statistic))
        period = float(grid[top])
        center = (best_start[k, top] + best_width[k, top] / 2) / best_bins[top] * period
        first = center + np.ceil((kept.min() - center) / period) * period
        # best_sum is the flux in the box minus the mean of all the points, the box included; the mean in the box minus
        # the mean outside it is best_sum / (n (1 - n / N)).
        share = 1 - best_number[k, top] / (row["n"] if row["packed_total"] is None else row["packed_total"])
        depth = float(-box_sum[k, top] / best_number[k, top] / share / level * 1e6)
        found = {
            "period_min_searched_days": float(grid.min()),
            "period_max_searched_days": float(grid.max()),
            "period_days": period,
            "period_uncertainty_days": float(period_uncertainty(grid, power)),  # TLS's half width at half maximum of the highest peak
            "t0_days": float(first),
            "duration_hr": float(best_width[k, top] / best_bins[top] * period * 24),
            "duration_corrected_hr": float(best_width[k, top] / best_bins[top] * period * 24),
            "fill_factor": 1.0,
            "depth_ppm": depth,
            "depth_mean_ppm": depth,
            "sde": float(power[top]),
            "snr": float(depth * 1e-6 * level / (np.sqrt(row["sigma2"]) / np.sqrt(best_number[k, top] * share))),  # its square is the statistic searched
            "transit_count": int(np.floor((kept.max() - first) / period)) + 1,
        }
        if spectrum:
            found["periods"] = grid
            found["power"] = power
        if boxes:  # the center of one transit (the others are a period apart) and the width of the best box, in days
            found["centers"] = (best_start[k, :end] + best_width[k, :end] / 2) / best_bins[:end] * grid
            found["widths_days"] = best_width[k, :end] / best_bins[:end] * grid
        if bumps and side == 0:
            found["bump"] = result(k, row, end, side=1)
        return found

    if ends is None:
        return [result(k, row, periods.size) for k, row in enumerate(rows_of)]
    return [[result(k, row, end) for end in ends] for k, row in enumerate(rows_of)]


def search(time, flux, period_min, period_max, spectrum=False, threads=1, device=None, batch=None, minimum_bins=MINIMUM_BINS, duration_step=DURATION_STEP, periods=None, boxes=False, template=None, precision=None, weights=None, oversampling_factor=None, n_transits_min=None, bumps=False):
    """Search one flux and return its result dict: `search_many(time, [flux], ...)[0]` with unlabeled errors.

    See `search_many` for the arguments and the fields of the result. `threads` is accepted and ignored.
    """
    return search_many(time, [flux], period_min, period_max, spectrum=spectrum, device=device, batch=batch, minimum_bins=minimum_bins, duration_step=duration_step, periods=periods, labeled=False, boxes=boxes, template=template, precision=precision, weights=weights, oversampling_factor=oversampling_factor, n_transits_min=n_transits_min, bumps=bumps)[0]
