"""The stable phase order of the points for a batch of trial periods.

Transit Least Squares (transitleastsquares 2.0.1, `core_fused.fold_order_into`, MIT, https://github.com/hippke/tls)
orders the points of a period by their float64 phase, equal phases in index order: the permutation of
`torch.argsort(phase, stable=True)`. Both methods here give that permutation bit for bit.

On a CUDA device `counting_order` uses a counting sort: the points are counted into 2^k phase bins of a few points
each, scattered to the segments of their bins, and ranked inside a bin by (phase, index). The bins follow the phase
order, so no comparison sort of the whole row is needed.

Elsewhere `phase_order` sorts the phases rounded to float32, stable. Rounding never reverses two phases, so the
order can differ only inside runs of equal float32 keys; the points of those runs are sorted again by their float64
phase and put back in the slots that the runs occupy.
"""
import torch

from transitgpu.core import on_device

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch operations are used instead
    triton = None

if triton is not None:

    @triton.jit
    def _phase_kernel(times, periods, phase, key, points, row_points, block: tl.constexpr):
        """Write phase = t / P - floor(t / P) in float64 and its float32 rounding, in one pass over the times."""
        row = tl.program_id(1)
        index = tl.program_id(0) * block + tl.arange(0, block)
        inside = index < points
        scaled = tl.load(times + index, mask=inside, other=0.0) / tl.load(periods + row)
        value = scaled - tl.floor(scaled)
        tl.store(phase + row * row_points + index, value, mask=inside)
        tl.store(key + row * row_points + index, value.to(tl.float32), mask=inside)


    @triton.jit
    def _bin_kernel(times, periods, phase, counts, points, row_points, bins: tl.constexpr, block: tl.constexpr):
        """Write the float64 phase of every point (as _phase_kernel) and count the points of each phase bin.

        The bin is floor(phase * bins), exact for a power of two and so non-decreasing in the phase.
        """
        row = tl.program_id(1)
        index = tl.program_id(0) * block + tl.arange(0, block)
        inside = index < points
        scaled = tl.load(times + index, mask=inside, other=0.0) / tl.load(periods + row)
        value = scaled - tl.floor(scaled)
        tl.store(phase + row * row_points + index, value, mask=inside)
        tl.atomic_add(counts + row * bins + tl.minimum((value * bins).to(tl.int32), bins - 1), 1, mask=inside)

    @triton.jit
    def _scatter_kernel(phase, starts, cursor, grouped, points, row_points, bins: tl.constexpr, block: tl.constexpr):
        """Write the index of every point at a free slot of the segment of its bin, in the order the atomics give."""
        row = tl.program_id(1)
        index = tl.program_id(0) * block + tl.arange(0, block)
        inside = index < points
        value = tl.load(phase + row * row_points + index, mask=inside, other=0.0)
        slot_bin = tl.minimum((value * bins).to(tl.int32), bins - 1)
        slot = tl.atomic_add(cursor + row * bins + slot_bin, 1, mask=inside)
        tl.store(grouped + row * row_points + tl.load(starts + row * bins + slot_bin, mask=inside, other=0) + slot, index, mask=inside)

    @triton.jit
    def _rank_kernel(phase, starts, counts, grouped, order, points, row_points, limit, bins: tl.constexpr, block: tl.constexpr):
        """Write order[start of the bin + rank] = point, the rank being the number of points of the bin before it.

        Points are compared by (phase, index). A bin of m points costs each of its points m steps, and the other
        points of the block wait for them; the steps stop at `limit`, and counting_order sorts again any row with a
        bin over it.
        """
        row = tl.program_id(1)
        position = tl.program_id(0) * block + tl.arange(0, block)
        inside = position < points
        me = tl.load(grouped + row * row_points + position, mask=inside, other=0)
        key = tl.load(phase + row * row_points + me, mask=inside, other=0.0)
        mine = tl.minimum((key * bins).to(tl.int32), bins - 1)
        first = tl.load(starts + row * bins + mine, mask=inside, other=0)
        size = tl.load(counts + row * bins + mine, mask=inside, other=0)
        rank = tl.zeros([block], dtype=tl.int32)
        for step in range(0, tl.minimum(tl.max(size, axis=0), limit)):
            alive = inside & (step < size)
            other = tl.load(grouped + row * row_points + first + step, mask=alive, other=0)
            other_key = tl.load(phase + row * row_points + other, mask=alive, other=0.0)
            rank += (alive & ((other_key < key) | ((other_key == key) & (other < me)))).to(tl.int32)
        tl.store(order + row * row_points + first + rank, me.to(tl.int64), mask=inside)


def phases(times, periods):
    """Return (phase, float32 key) of `times` for each period: t / P - floor(t / P), as TLS's `foldfast` computes it.

    One kernel on a CUDA device, torch operations otherwise.
    """
    if triton is not None and times.is_cuda:
        phase = torch.empty((len(periods), len(times)), dtype=torch.float64, device=times.device)
        key = torch.empty((len(periods), len(times)), dtype=torch.float32, device=times.device)
        block = 1024
        with on_device(times.device):
            _phase_kernel[(triton.cdiv(len(times), block), len(periods))](times, periods, phase, key, len(times), phase.stride(0), block=block)
        return phase, key
    scaled = times[None, :] / periods[:, None]
    phase = scaled - torch.floor(scaled)
    return phase, phase.to(torch.float32)


# Most points of a phase bin that the rank kernel steps through. A regular cadence folded at a multiple m of the
# cadence puts N / m points in a bin, which would cost N^2 / m steps.
RANK_LIMIT = 256


def counting_order(times, periods):
    """Return the stable order of the float64 phases of `times` for each period, as (periods x points) int64.

    The result is the permutation of torch.argsort(phase, stable=True). On a CUDA device it is a counting sort on 2^k
    phase bins of two to four points each: a histogram, a scan of the counts, a scatter of the indices and a rank
    inside each bin, with torch's stable sort for a row with a bin of more than RANK_LIMIT points. Otherwise it is
    `phase_order`.
    """
    if triton is None or not times.is_cuda:
        phase, key = phases(times, periods)
        return phase_order(phase, key)
    rows, points = len(periods), len(times)
    bins = 1 << max(0, (max(points // 4, 1) - 1).bit_length())
    device = times.device
    phase = torch.empty((rows, points), dtype=torch.float64, device=device)
    counts = torch.zeros((rows, bins), dtype=torch.int32, device=device)
    cursor = torch.zeros_like(counts)
    grouped = torch.empty((rows, points), dtype=torch.int32, device=device)
    order = torch.empty((rows, points), dtype=torch.int64, device=device)
    block = 1024
    grid = (triton.cdiv(points, block), rows)
    with on_device(device):
        _bin_kernel[grid](times, periods, phase, counts, points, points, bins=bins, block=block)
        starts = torch.cumsum(counts, dim=1, dtype=torch.int32) - counts
        _scatter_kernel[grid](phase, starts, cursor, grouped, points, points, bins=bins, block=block)
        _rank_kernel[grid](phase, starts, counts, grouped, order, points, points, RANK_LIMIT, bins=bins, block=block)
        crowded = (counts.amax(dim=1) > RANK_LIMIT).nonzero().flatten()
        if len(crowded):
            order[crowded] = torch.argsort(phase[crowded], dim=1, stable=True)
    return order


def phase_order(phase, keys=None):
    """Return the permutation that sorts each row of `phase` (float64) ascending, equal values in index order.

    `keys` are the phases rounded to float32; they are computed when not given.
    """
    if keys is None:
        keys = phase.to(torch.float32)
    sorted_keys, order = torch.sort(keys, dim=1, stable=True)
    tied = sorted_keys[:, 1:] == sorted_keys[:, :-1]
    shared = torch.zeros_like(sorted_keys, dtype=torch.bool)
    shared[:, 1:] |= tied
    shared[:, :-1] |= tied
    row, slot = shared.nonzero(as_tuple=True)  # row-major: by period, then by slot
    if len(row) == 0:
        return order
    members = order[row, slot]
    by_phase = torch.argsort(phase[row, members], stable=True)
    by_row = torch.argsort(row[by_phase], stable=True)  # by period, then by phase; equal phases keep the index order
    order = order.clone()
    order[row, slot] = members[by_phase][by_row]
    return order
