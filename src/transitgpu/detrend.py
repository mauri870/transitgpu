"""The biweight filter of a light curve on a device, with wotan's windows, mask and stopping rule.

`biweight(time, flux, window)` returns the trend that `wotan.flatten(time, flux, method="biweight", window_length=window, return_trend=True)` returns
(https://github.com/hippke/wotan, `flatten.py`, `slider.py`, `location_estimates.py`; Hippke et al. 2019, AJ 158, 143), and `flux / trend` is the flattened flux. The rules that the result depends on:

  - A point with a NaN in `time` or `flux` has a NaN trend and is in no window.
  - The light curve is cut into segments where two consecutive times differ by more than `window / 2`, and a window never leaves its segment.
  - The window of a point at time t holds the points with t - window / 2 <= time < t + window / 2. The last point of a segment is in no window, so its trend is that of the points before it.
  - A point that `mask` marks is in no window; its own trend is still computed. A window with no point left has a NaN trend.
  - The trend is the biweight location of the flux of the window: the median, then steps of `sum((x - c) w) / sum(w)` with `w = (1 - u^2)^2` for `|u| < 1` and 0 otherwise, `u = (x - c) / (5 * MAD)`
    with the median absolute deviation unscaled, until a step is at most `tolerance`. A window whose MAD is 0 has its median as its trend. The iteration has no limit on the number of steps.
  - A trend that is exactly 0 is NaN.
  - A masked point, which wotan gives the sign of its flux changed and then drops with every point whose value is at or below -1e-12, keeps that rule here: a point with a flux of -1e-12 or less is in no window.

`biweight_many` does the same for a list of light curves in one sequence of launches. `time` must be in increasing order. The torch version runs on any torch device, the CPU included, and is the reference of the Triton kernel. The kernel is used on a CUDA or ROCm device where Triton can be imported, for windows padded to at most KERNEL_WIDTH_LIMIT points; it builds
no matrix of windows (each program reads its window from the flux array) and gives the same trend as the torch version up to rounding. Triton: https://triton-lang.org/main/python-api/triton.language.html
"""
import numpy as np
import torch

from transitgpu.core import on_device

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch version is used instead
    triton = None

CVAL = 5.0  # wotan's constant of the biweight
TOLERANCE = 1e-6  # wotan's stopping step (its FTOL)
BARRIER = 1e-12  # a point whose value is at or below -BARRIER is dropped, as wotan does
TORCH_ROWS = 8192  # the rows of one chunk of the torch version
KERNEL_ROWS = 16384  # the programs of one launch of the kernel at a window of 1,024 points or fewer
KERNEL_WIDTH_LIMIT = 2048  # the widest padded window the kernel takes; its compile grows about four times for each doubling (benchmarks/bench_detrend.py), and a wider window runs the torch version on the device
METHODS = ("kernel", "torch")


def windows(time, flux, keep, window):
    """Return (values, start, end) for the points of `time` (finite, increasing): `values` is the flux with the mask applied as wotan applies it, and the window of point i is values[start[i]:end[i]]."""
    size = len(time)
    start = np.zeros(size, dtype=np.int64)
    end = np.zeros(size, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(time) > window / 2) + 1
    half = window / 2
    for low, high in zip([0, *breaks], [*breaks, size]):
        segment = time[low:high]
        start[low:high] = np.searchsorted(segment, segment - half, side="left") + low
        end[low:high] = np.minimum(np.searchsorted(segment, segment + half, side="left"), len(segment) - 1) + low
    sign = np.where(keep, 1.0, -1.0)
    return flux * sign, start, end


def _median(sorted_values, count):
    low = ((count - 1) // 2).clamp(min=0)
    high = (count // 2).clamp(min=0)
    rows = torch.arange(sorted_values.shape[0], device=sorted_values.device)
    return (sorted_values[rows, low] + sorted_values[rows, high]) * 0.5


def trend_torch(values, start, end, width, tolerance=TOLERANCE, rows=TORCH_ROWS):
    """The trend of every window of `windows`, by torch operations on a matrix of at most `width` points a window, `rows` windows at a time."""
    pieces = []
    for first in range(0, len(start), rows):
        s, e = start[first : first + rows], end[first : first + rows]
        columns = torch.arange(width, device=values.device)
        index = s[:, None] + columns[None, :]
        gathered = values[index.clamp(max=values.shape[0] - 1)]
        ok = (index < e[:, None]) & (gathered > -BARRIER)
        count = ok.sum(1)
        infinity = torch.full_like(gathered, float("inf"))
        center = _median(torch.where(ok, gathered, infinity).sort(1).values, count)
        mad = _median(torch.where(ok, (gathered - center[:, None]).abs(), infinity).sort(1).values, count)
        active = (mad > 0) & (count > 0)
        center = torch.where(count > 0, center, torch.full_like(center, float("nan")))
        reciprocal = 1 / (CVAL * mad)
        data = torch.where(ok, gathered, torch.zeros_like(gathered))
        while bool(active.any()):
            distance = data - center[:, None]
            scaled = distance * reciprocal[:, None]
            weight = torch.where(ok & (scaled.abs() < 1), (1 - scaled**2) ** 2, torch.zeros_like(scaled))
            step = (distance * weight).sum(1) / weight.sum(1)
            center = center + torch.where(active, step, torch.zeros_like(step))
            active = active & (step.abs() > tolerance)
        pieces.append(center)
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=values.dtype, device=values.device)


if triton is not None:

    @triton.jit(do_not_specialize=["first"])  # a launch of a slice starts at any row: a specialization on the value of `first` would compile the kernel again for some of them
    def _kernel(values, start, end, out, first, width: tl.constexpr, tolerance, cval):
        i = first + tl.program_id(0)
        low = tl.load(start + i)
        high = tl.load(end + i)
        offsets = tl.arange(0, width)
        index = low + offsets
        inside = index < high
        value = tl.load(values + index, mask=inside, other=0.0)
        ok = inside & (value > -0.000000000001)
        count = tl.sum(ok.to(tl.int32), axis=0)
        infinity = float("inf")
        lower = (count - 1) // 2
        upper = count // 2
        ordered = tl.sort(tl.where(ok, value, infinity))
        median = (tl.sum(tl.where(offsets == lower, ordered, 0.0), axis=0) + tl.sum(tl.where(offsets == upper, ordered, 0.0), axis=0)) * 0.5
        deviations = tl.sort(tl.where(ok, tl.abs(value - median), infinity))
        mad = (tl.sum(tl.where(offsets == lower, deviations, 0.0), axis=0) + tl.sum(tl.where(offsets == upper, deviations, 0.0), axis=0)) * 0.5
        center = median
        reciprocal = 1.0 / (cval * mad)
        active = (mad > 0) & (count > 0)
        while active:
            distance = value - center
            scaled = distance * reciprocal
            weight = tl.where(ok & (tl.abs(scaled) < 1.0), (1.0 - scaled * scaled) * (1.0 - scaled * scaled), 0.0)
            step = tl.sum(distance * weight, axis=0) / tl.sum(weight, axis=0)
            center = center + step
            active = tl.abs(step) > tolerance
        tl.store(out + i, tl.where(count > 0, center, float("nan")))


def kernel_rows(width):
    """The programs of one launch for windows padded to `width` points: the work of a program grows faster than the width, so a wider window gets fewer, which keeps a launch to a few milliseconds."""
    return int(min(65536, max(64, KERNEL_ROWS * (1024 / width) ** 1.5)))


def trend_kernel(values, start, end, width, tolerance=TOLERANCE, rows=None):
    """The trend of every window of `windows` by the Triton kernel, `rows` programs at a time (`kernel_rows(width)` when None). `values`, `start` and `end` are on a CUDA or ROCm device."""
    out = torch.empty(len(start), dtype=torch.float64, device=values.device)
    rows = rows or kernel_rows(width)
    with on_device(values.device):
        for first in range(0, len(start), rows):
            _kernel[(min(rows, len(start) - first),)](values, start, end, out, first, width=width, tolerance=tolerance, cval=CVAL, num_warps=1)
    return out


def uses_kernel(device, width, method=None):
    """Whether the trend of windows padded to `width` points on `device` is computed by the Triton kernel: `method` "kernel" or "torch" chooses, and None takes the kernel on a CUDA or ROCm device where Triton
    can be imported and `width` is at most KERNEL_WIDTH_LIMIT."""
    if method not in (None, *METHODS):
        raise ValueError(f"method must be None, 'kernel' or 'torch', not {method!r}")
    if method == "kernel" and (device.type != "cuda" or triton is None):
        raise ValueError("method='kernel' needs a CUDA or ROCm device and Triton")
    if method is not None:
        return method == "kernel"
    return device.type == "cuda" and triton is not None and width <= KERNEL_WIDTH_LIMIT


def biweight_many(sectors, window, masks=None, device=None, tolerance=TOLERANCE, method=None):
    """Return the biweight trend of each light curve of `sectors`, a list of (time, flux) pairs, as a list of numpy arrays, with the windows of all of them in one sequence of launches.

    `masks` is None or a list with a mask (True for the points left out of the windows) or None for each light curve. The result of each equals `biweight` of that light curve alone; the batch pays the transfer and the
    launches once, which is what a pipeline that detrends many sectors of a star at a time needs. The arguments and the rules are those of `biweight`.
    """
    masks = [None] * len(sectors) if masks is None else list(masks)
    if len(masks) != len(sectors):
        raise ValueError("masks must have one entry for each light curve")
    prepared, offset, parts = [], 0, ([], [], [])
    for (time, flux), mask in zip(sectors, masks):
        time = np.array(time, dtype=np.float64)
        flux = np.array(flux, dtype=np.float64)
        keep = np.ones(len(time), dtype=bool) if mask is None else ~np.asarray(mask, dtype=bool)
        good = ~np.isnan(time * flux)
        if good.any() and np.any(np.diff(time[good]) < 0):
            raise ValueError("time must be in increasing order")
        if good.any():
            values, start, end = windows(time[good], flux[good], keep[good], window)
            parts[0].append(values)
            parts[1].append(start + offset)
            parts[2].append(end + offset)
            offset += len(values)
        prepared.append((len(time), good))
    if offset == 0:
        return [np.full(size, np.nan) for size, _ in prepared]
    values, start, end = (np.concatenate(part) for part in parts)
    width = int((end - start).max())
    result = np.full(len(start), np.nan)
    if width > 0:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        values_d, start_d, end_d = (torch.as_tensor(array, device=device) for array in (values, start, end))
        if uses_kernel(device, 1 << (width - 1).bit_length(), method):
            computed = trend_kernel(values_d, start_d, end_d, 1 << (width - 1).bit_length(), tolerance)
        else:
            computed = trend_torch(values_d, start_d, end_d, width, tolerance)
        result = computed.cpu().numpy()
        result[result == 0] = np.nan
    trends, at = [], 0
    for size, good in prepared:
        trend = np.full(size, np.nan)
        count = int(good.sum())
        trend[good] = result[at : at + count]
        at += count
        trends.append(trend)
    return trends


def biweight(time, flux, window, mask=None, device=None, tolerance=TOLERANCE, method=None):
    """Return the biweight trend of `flux` at `time` (days), a window of `window` days wide, as a numpy array of the length of `time`.

    `mask` is True for the points that the trend must leave out of its windows (the points of known transits), as wotan's `mask`. `device` is a torch device name: the default is "cuda" where torch finds a CUDA or ROCm
    device and "cpu" otherwise. On a CUDA or ROCm device the Triton kernel is used when Triton can be imported and the widest window, padded to a power of two, holds at most KERNEL_WIDTH_LIMIT points;
    otherwise the torch version runs on the same device. `method` "kernel" or "torch" chooses instead. `tolerance` is the stopping step of the iteration.
    """
    return biweight_many([(time, flux)], window, masks=[mask], device=device, tolerance=tolerance, method=method)[0]
