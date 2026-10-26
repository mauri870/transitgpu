"""The folded flux and the prefix sums of the screened search.

For a batch of periods, `order` is the stable phase order of the points (transitgpu.tls.fold). With f the flux in
that order, wrapped around by `extra` points so that a window across the end of the phase is a window of the array,
the screened search needs for each period:
  - the cumulative sum of f;
  - the cumulative sum of w (f - 1)^2;
  - the double prefix sum (the cumulative sum of the cumulative sum) of the centred weighted residual
    x = w (1 - f) - mean, on which TLS's piecewise-linear screening is a sum of a few terms;
  - the constant w * sum (1 - f)^2 over the points without the wrap (the residual of the constant model).

`prefix_arrays_torch` computes each with torch.cumsum on a gathered array. The kernel reads the points twice and
writes each result once: a first pass sums every chunk of CHUNK positions, a cumulative sum over the chunks gives the
carries, and a second pass scans each chunk again and adds its carries (the double prefix sum of a chunk is the carry
of the double sum, plus the position times the carry of the single sum, plus the double scan of the chunk).
"""
import torch

from transitgpu.core import on_device

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch version is used instead
    triton = None

CHUNK = 1024


def prefix_arrays_torch(y, order, weight, mean, extra):
    """Return (patched, cumulative, squares, double, constant) by torch operations; the reference of the kernel.

    `patched` is the folded flux with its wrap, `cumulative` and `squares` the cumulative sums of f and of
    w (f - 1)^2, `double` the double prefix sum of x and `constant` the residual of the constant model.
    """
    b = order.shape[0]
    flux = y[order]
    patched = torch.cat([flux, flux[:, :extra]], dim=1)
    zeros = torch.zeros((b, 1), dtype=torch.float64, device=y.device)
    single = torch.cat([zeros, torch.cumsum(weight * (1.0 - patched) - mean, dim=1)], dim=1)
    double = torch.cat([zeros, torch.cumsum(single, dim=1)], dim=1)
    cumulative = torch.cat([zeros, torch.cumsum(patched, dim=1)], dim=1)
    squares = torch.cat([zeros, torch.cumsum(weight * (patched - 1.0) ** 2, dim=1)], dim=1)
    constant = weight * ((1.0 - flux) ** 2).sum(dim=1)
    return patched, cumulative, squares, double, constant


if triton is not None:

    @triton.jit
    def _terms(y, order, row, n, size, chunk, constants, block: tl.constexpr):
        weight = tl.load(constants)
        mean = tl.load(constants + 1)
        position = chunk * block + tl.arange(0, block)
        real = position < size
        source = position % n
        flux = tl.load(y + tl.load(order + row * n + source, mask=real, other=0), mask=real, other=1.0)
        return position, real, flux, tl.where(real, flux, 0.0), tl.where(real, weight * (flux - 1.0) * (flux - 1.0), 0.0), tl.where(real, weight * (1.0 - flux) - mean, 0.0)

    @triton.jit
    def _chunk_sums(y, order, totals, n, size, chunks, constants, block: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        position, real, flux, x1, x2, x3 = _terms(y, order, row, n, size, chunk, constants, block)
        inside = tl.where(position < n, (1.0 - flux) * (1.0 - flux), 0.0)
        inside = tl.where(real, inside, 0.0)
        weighted = tl.sum(x3 * (block - 1 - tl.arange(0, block)), axis=0)
        base = totals + (row * chunks + chunk) * 5
        tl.store(base, tl.sum(x1, axis=0))
        tl.store(base + 1, tl.sum(x2, axis=0))
        tl.store(base + 2, tl.sum(x3, axis=0))
        tl.store(base + 3, weighted)
        tl.store(base + 4, tl.sum(inside, axis=0))

    @triton.jit
    def _chunk_scans(y, order, carries, patched, cumulative, squares, double, n, size, chunks, constants, block: tl.constexpr, residual: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        # the double scan runs over size + 1 positions: x is zero at the position `size`
        position, real, flux, x1, x2, x3 = _terms(y, order, row, n, size + 1, chunk, constants, block)
        real_flux = position < size
        flux = tl.where(real_flux, flux, 1.0)
        x1 = tl.where(real_flux, x1, 0.0)
        x2 = tl.where(real_flux, x2, 0.0)
        x3 = tl.where(real_flux, x3, 0.0)
        base = carries + (row * chunks + chunk) * 4
        carry_flux = tl.load(base)
        carry_squares = tl.load(base + 1)
        carry_single = tl.load(base + 2)
        carry_double = tl.load(base + 3)
        offset = tl.arange(0, block)
        if residual:
            tl.store(patched + row * size + position, tl.load(constants) * (1.0 - flux), mask=real_flux)
        else:
            tl.store(patched + row * size + position, flux, mask=real_flux)
        tl.store(cumulative + row * (size + 1) + position + 1, carry_flux + tl.cumsum(x1, axis=0), mask=real_flux)
        tl.store(squares + row * (size + 1) + position + 1, carry_squares + tl.cumsum(x2, axis=0), mask=real_flux)
        exclusive = tl.cumsum(x3, axis=0) - x3
        tl.store(double + row * (size + 2) + position + 1, carry_double + (offset + 1) * carry_single + tl.cumsum(exclusive, axis=0), mask=real)
        first = (chunk == 0) & (offset == 0)
        tl.store(cumulative + row * (size + 1) + offset * 0, tl.zeros([block], dtype=tl.float64), mask=first)
        tl.store(squares + row * (size + 1) + offset * 0, tl.zeros([block], dtype=tl.float64), mask=first)
        tl.store(double + row * (size + 2) + offset * 0, tl.zeros([block], dtype=tl.float64), mask=first)


def carry_scan(x):
    """Return the cumulative sum of each row of the (batch, chunks) array `x`, in an order independent of the batch.

    torch.cumsum along the last dimension chooses its algorithm by the shape, so its last digits can depend on the
    number of rows; along the first dimension each column is summed in sequence.
    """
    return torch.cumsum(x.t().contiguous(), dim=0).t()


def prefix_arrays(y, order, weight, mean, extra, residual=False):
    """Return what `prefix_arrays_torch` returns, by the kernels on a CUDA device and by torch operations otherwise.

    With `residual` the first array is weight * (1 - flux), the term that the dot products of the screened search sum, in place of the flux.
    """
    if triton is None or not order.is_cuda:
        patched, *sums = prefix_arrays_torch(y, order, weight, mean, extra)
        return (weight * (1.0 - patched) if residual else patched), *sums
    b, n = order.shape
    size = n + extra
    constants = torch.tensor([weight, mean], dtype=torch.float64, device=y.device)  # in a tensor: float64 scalar arguments of a kernel are rounded to float32
    chunks = triton.cdiv(size + 1, CHUNK)
    totals = torch.empty((b, chunks, 5), dtype=torch.float64, device=y.device)
    with on_device(y.device):
        _chunk_sums[(b, chunks)](y, order, totals, n, size, chunks, constants, block=CHUNK)
    single = carry_scan(totals[:, :, 2]) - totals[:, :, 2]  # the sum of x before each chunk
    step = totals[:, :, 3] + CHUNK * single
    carries = torch.stack([carry_scan(totals[:, :, 0]) - totals[:, :, 0], carry_scan(totals[:, :, 1]) - totals[:, :, 1], single, carry_scan(step) - step], dim=2).contiguous()
    patched = torch.empty((b, size), dtype=torch.float64, device=y.device)
    cumulative = torch.empty((b, size + 1), dtype=torch.float64, device=y.device)
    squares = torch.empty_like(cumulative)
    double = torch.empty((b, size + 2), dtype=torch.float64, device=y.device)
    with on_device(y.device):
        _chunk_scans[(b, chunks)](y, order, carries, patched, cumulative, squares, double, n, size, chunks, constants, block=CHUNK, residual=residual)
    return patched, cumulative, squares, double, weight * totals[:, :, 4].sum(dim=1)
