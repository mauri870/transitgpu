"""Exact screening of the starts of the template search of Transit Least Squares.

For a start i of a template of width w the gain over the constant residual is 2 k AR - k^2 A2, with
AR = sum_j p_j r_(i+j) w_(i+j) (p the template, r = 1 - y the folded residual), A2 = sum_j w p_j^2 and
k = overshoot (1 - mean_i) / SIGNAL_DEPTH. A start can lower the minimum only if its gain exceeds the best gain found
so far (the incumbent), and two bounds on the gain need no sum over the template (transitleastsquares 2.0.1,
`core_fused`, MIT, Michael Hippke and Rene Heller, https://github.com/hippke/tls):

- Cauchy-Schwarz: AR <= sqrt(A2 R2), with R2 = sum w r^2 over the window of the start, a difference of prefix sums
  to which an allowance for its rounding is added as TLS does in its proxies; so the gain is at most
  2 k sqrt(A2 (R2 + round)) - k^2 A2.
- A piecewise-linear proxy of the template, whose correlation is a sum of about 15 terms of the double prefix sum of
  r w; the distance of the template to the proxy bounds the difference by sqrt(w_max |p - proxy|^2 R2). The
  allowances for the rounding of the prefix sums are TLS's (`screen_round_budget`, `screen_templates`).

The starts whose bounds exceed the incumbent get their exact dot product in float64, so the chi square, the template
row and the depth are those of TLS's exact backend. The incumbent of a period is the best gain among the deepest box
starts of its widths. Of the survivors the highest gain wins and, on equal gains, the earlier width and start, as
TLS's strict comparisons in ascending order give.
"""
import numpy as np
import torch

from transitgpu.core import on_device

try:
    import triton
    import triton.language as tl
except ImportError:  # the bounds run as torch operations instead
    triton = None
from transitleastsquares.core_fused import flatten_templates, screen_round_budget, screen_templates

SIGNAL_DEPTH = 0.5  # transitleastsquares.tls_constants.SIGNAL_DEPTH: the depth of the cached templates
# Knot tolerance of the cheap proxy, relative to the largest sample of the template (TLS's own proxy uses 0.1 below
# 512 samples and 0.01 above).
LOOSE_TOLERANCE = 0.3
SEED_COARSEN = 8  # the incumbent is taken from the deepest of every 8th tested start of a width
DOT_LAUNCH = 1 << 23  # survivors dotted by one launch: 33,936,431 in one launch gave wrong dot products to millions of them (the work items of a grid must stay under 2^32)
ELEMENTS = 1 << 24  # largest block of gathered data for the dot products of the survivors on a CPU


if triton is not None:

    @triton.jit
    def _keep_all_kernel(cumulative, squares, double, cutoff, allowed, positions, coefficients, keep, ints, floats, tile_first, widths, rows, row_cumulative, row_double, row_keep, block: tl.constexpr):
        """Write the mask of the starts that can lower the minimum, for every width of a batch in one launch.

        The programs of one period run next to each other, so the three arrays of its row (a few megabytes) stay in
        the caches while its widths are screened. `ints` (width, step, count, first column, first term, terms,
        template length, first cheap term, cheap terms) and `floats` (overshoot factor, template square sum, proxy
        constant, proxy error, rounding of the squares, shallowest depth, cheap proxy constant, cheap proxy error)
        have a row per width; `tile_first` holds the first tile of each width.
        """
        tile_all = tl.program_id(0)  # the first axis varies fastest: the programs of a period run together
        row = tl.program_id(1)
        index = 0
        for other in range(1, widths):
            index += (tile_all >= tl.load(tile_first + other)).to(tl.int32)
        tile = tile_all - tl.load(tile_first + index)
        width = tl.load(ints + index * 9)
        step = tl.load(ints + index * 9 + 1)
        count = tl.load(ints + index * 9 + 2)
        first_column = tl.load(ints + index * 9 + 3)
        first_term = tl.load(ints + index * 9 + 4)
        terms = tl.load(ints + index * 9 + 5)
        length = tl.load(ints + index * 9 + 6)
        first_cheap = tl.load(ints + index * 9 + 7)
        cheap_terms = tl.load(ints + index * 9 + 8)
        factor = tl.load(floats + index * 8)
        a2 = tl.load(floats + index * 8 + 1)
        inverse_width = 1.0 / width.to(tl.float64)
        constant = tl.load(floats + index * 8 + 2)
        error = tl.load(floats + index * 8 + 3)
        round_r2 = tl.load(floats + index * 8 + 4)
        depth_min = tl.load(floats + index * 8 + 5)
        cheap_constant = tl.load(floats + index * 8 + 6)
        cheap_error = tl.load(floats + index * 8 + 7)
        column = tile * block + tl.arange(0, block)
        inside = column < count
        start = column * step
        before = tl.load(cumulative + row * row_cumulative + start, mask=inside, other=0.0)
        after = tl.load(cumulative + row * row_cumulative + start + width, mask=inside, other=0.0)
        mean = 1.0 - (after - before) * inverse_width  # no division: a float64 division costs dozens of multiplications on a GPU
        k = mean * factor
        live = inside & (mean > depth_min - 1e-12) & (tl.load(allowed + index * rows + row) != 0)  # the host tests the exact mean again for the survivors
        r2 = tl.load(squares + row * row_cumulative + start + length, mask=live, other=0.0) - tl.load(squares + row * row_cumulative + start, mask=live, other=0.0)
        limit = tl.load(cutoff + row)
        # Cauchy-Schwarz: a start passes where 2 k sqrt((r2 + round_r2) a2) - k^2 a2 > limit, less 1e-8 of the size of
        # its terms; round_r2 is the allowance for the rounding of r2. Squared, so that no square root is taken.
        threshold = limit + k * k * a2 - 1e-8 * (tl.abs(limit) + k * k * a2)
        passes = live & ((threshold <= 0.0) | (4.0 * k * k * tl.maximum(r2 + round_r2, 0.0) * a2 > threshold * threshold))
        margin_need = (limit + k * k * a2 - 1e-8 * (tl.abs(limit) + k * k * a2)) / (2.0 * tl.maximum(k, 1e-300))
        if tl.max(passes.to(tl.int32), axis=0) > 0:  # the cheap proxy, a few terms of the double prefix sum
            cheap = tl.zeros([block], dtype=tl.float64)
            for t in range(cheap_terms):
                position = tl.load(positions + first_cheap + t)
                cheap += tl.load(coefficients + first_cheap + t) * tl.load(double + row * row_double + start + position, mask=passes, other=0.0)
            need = margin_need - cheap - cheap_constant
            passes = passes & ((cheap_terms == 0) | (need <= 0.0) | (cheap_error * (r2 + round_r2) > need * need))
            # Then the proxy of TLS's tolerance:
            # 2 k (correlation + constant + sqrt(error (r2 + round))) - k^2 a2 > limit, with the same margin.
            if tl.max(passes.to(tl.int32), axis=0) > 0:
                correlation = tl.zeros([block], dtype=tl.float64)
                for t in range(terms):
                    position = tl.load(positions + first_term + t)
                    correlation += tl.load(coefficients + first_term + t) * tl.load(double + row * row_double + start + position, mask=passes, other=0.0)
                need = margin_need - correlation - constant
                passes = passes & ((terms == 0) | (need <= 0.0) | (error * (r2 + round_r2) > need * need))
        tl.store(keep + row * row_keep + first_column + column, passes.to(tl.int8), mask=inside)


    @triton.jit
    def _dots_kernel(patched, profiles, constants, out, rows, starts, widths, offsets, row_patched, block: tl.constexpr, residual: tl.constexpr):
        """Write the AR of each survivor s, out[s] = sum_j profile[j] * weight * (1 - patched[row, start + j]).

        The sum is in float64, along the template. With `residual`, `patched` holds weight * (1 - flux) already and the term is read as it is.
        """
        survivor = tl.program_id(0)
        row = tl.load(rows + survivor)
        start = tl.load(starts + survivor)
        width = tl.load(widths + survivor)
        offset = tl.load(offsets + survivor)
        weight = tl.load(constants)
        total = tl.zeros([block], dtype=tl.float64)
        for first in range(0, width, block):
            index = first + tl.arange(0, block)
            inside = index < width
            if residual:
                term = tl.load(patched + row * row_patched + start + index, mask=inside, other=0.0)
            else:
                term = weight * (1.0 - tl.load(patched + row * row_patched + start + index, mask=inside, other=1.0))
            total += tl.load(profiles + offset + index, mask=inside, other=0.0) * term
        tl.store(out + survivor, tl.sum(total, axis=0))

    @triton.jit
    def _deepest_kernel(cumulative, deepest, depth, ints, rows, row_cumulative, block: tl.constexpr):
        """Write, for each row and width of `ints` (width, step, count), the first start of the deepest box and its sum.

        The deepest box is the one with the smallest sum of the flux over the width.
        """
        index = tl.program_id(0)
        row = tl.program_id(1)
        width = tl.load(ints + index * 3)
        step = tl.load(ints + index * 3 + 1)
        count = tl.load(ints + index * 3 + 2)
        best = tl.full([block], float("inf"), dtype=tl.float64)
        best_at = tl.zeros([block], dtype=tl.int64)
        for first in range(0, count, block):
            column = first + tl.arange(0, block)
            inside = column < count
            difference = tl.load(cumulative + row * row_cumulative + column * step + width, mask=inside, other=0.0) - tl.load(cumulative + row * row_cumulative + column * step, mask=inside, other=0.0)
            better = inside & (difference < best)
            best = tl.where(better, difference, best)
            best_at = tl.where(better, column.to(tl.int64), best_at)
        smallest = tl.min(best, axis=0)
        first_at = tl.min(tl.where(best == smallest, best_at, 1 << 60), axis=0)
        tl.store(deepest + index * rows + row, first_at)
        tl.store(depth + index * rows + row, smallest)



class Screen:
    """The per-width constants of the screening of one problem, computed once and kept on the device."""

    def __init__(self, y, dy, lc_arr, overview, widths, device):
        if np.ptp(dy) != 0.0:
            raise ValueError("the screened search needs equal errors: with weights that differ the square term of the gain is not a constant of the template")
        self.widths = widths
        weight = float(1.0 / dy[0] ** 2)
        self.weight = weight
        self.mean_rw = float(np.mean((1.0 - y) * weight))  # the centering of the double prefix sums
        flat = flatten_templates(lc_arr, overview)
        profile, offsets, lengths = flat[4], flat[2], flat[3]
        budget = screen_round_budget(y, np.full(len(y), weight), (self.mean_rw, weight), flat[0][-1])
        self.round_r2 = float(budget[2])  # allowance for the rounding of R2, which TLS adds in its proxies and Cauchy-Schwarz takes too
        packed = screen_templates(flat, (np.zeros(len(widths)),), 1e-2, False, round_budget=budget)
        loose = screen_templates(flat, (np.zeros(len(widths)),), 1e-2, False, proxy_eps=LOOSE_TOLERANCE, round_budget=budget)  # the cheap proxy: a looser fit with fewer terms, certified the same way
        self.profile = {}
        self.square = {}
        self.index_of = {w: index for index, w in enumerate(widths)}
        self.length = {w: int(lengths[index]) for index, w in enumerate(widths)}  # samples of the template, which can be fewer than the `w` of its box
        self.lengths_tensor = torch.as_tensor([int(v) for v in lengths], dtype=torch.int64, device=device)
        self.weight_tensor = torch.tensor([weight], dtype=torch.float64, device=device)
        self.profiles = torch.as_tensor(np.concatenate([profile[offsets[i]:offsets[i] + lengths[i]] for i in range(len(widths))]), dtype=torch.float64, device=device)
        self.profile_offsets = torch.as_tensor(np.concatenate([[0], np.cumsum(lengths)[:-1]]), dtype=torch.int64, device=device)
        self.widths_tensor = torch.as_tensor(widths, dtype=torch.int64, device=device)
        self.proxy = {}
        self.proxy_index = {}
        self.loose = {}
        for index, w in enumerate(widths):
            shape = profile[offsets[index]:offsets[index] + lengths[index]]
            self.profile[w] = torch.as_tensor(shape, device=device)
            self.square[w] = weight * float(np.sum(shape**2))
            if packed is not None and packed[0][index] > 0:
                count, start = int(packed[0][index]), int(packed[1][index])
                positions = [int(p) for p in packed[4][start:start + count]]
                coefficients = [float(c) for c in packed[5][start:start + count]]
                constant = float(packed[6][index]) * self.mean_rw + float(packed[10][index])
                self.proxy_index[w] = (torch.as_tensor(positions, device=device)[:, None], torch.as_tensor(coefficients, device=device, dtype=torch.float64))
                self.proxy[w] = (positions, coefficients, constant, float(packed[8][index]) * weight, float(packed[12]))
            if loose is not None and loose[0][index] > 0:
                count, start = int(loose[0][index]), int(loose[1][index])
                self.loose[w] = ([int(p) for p in loose[4][start:start + count]], [float(c) for c in loose[5][start:start + count]], float(loose[6][index]) * self.mean_rw + float(loose[10][index]), float(loose[8][index]) * weight, float(loose[12]))

    def starts_kept(self, w, step, size, overshoot, depth_min, cumulative, squares, double, cutoff, allowed_row, keep):
        """Write into `keep` (a batch-by-starts int8 view) the mask of the starts of width w that can lower the minimum.

        A start is kept when its box is deeper than `depth_min`, its period allows the width and its bound on the gain
        is above `cutoff`. This is the CPU path of `starts_kept_all`, by torch operations in float64. Its margins
        differ from the kernel's (here 1e-9 of the bound and an exact depth test; the kernel keeps 1e-8 of its terms
        and a depth tolerance of 1e-12), so the kernel may keep a few more starts.
        """
        count = len(range(0, size - w + 1, step))
        starts = slice(0, size - w + 1, step)
        factor = overshoot / SIGNAL_DEPTH
        mean = 1.0 - (cumulative[:, slice(w, size + 1, step)] - cumulative[:, starts]) / float(w)
        k = mean * factor
        r2 = squares[:, self.length[w]:self.length[w] + (count - 1) * step + 1:step] - squares[:, starts]
        bound = self.upper_bound(w, k, r2, double, step, size)
        keep.copy_((mean > depth_min) & allowed_row[:, None] & (bound + 1e-9 * (bound.abs() + cutoff[:, None]) > cutoff[:, None]))

    def starts_kept_all(self, used, stride, overshoot, size, depth_min, cumulative, squares, double, cutoff, allowed, keep, edges):
        """Write the masks of `starts_kept` for the widths `used`: one kernel launch on a GPU, width by width on a CPU.

        `used` is in ascending order, `allowed` is a (widths x periods) bool mask and `edges` the first column of each
        width in `keep`.
        """
        rows = cumulative.shape[0]
        for name, tensor in (("cutoff", cutoff), ("allowed", allowed), ("keep", keep)):
            if not tensor.is_contiguous():
                raise ValueError(f"starts_kept_all: {name} must be contiguous (the kernel indexes it as dense data); it has strides {tensor.stride()} for shape {tuple(tensor.shape)}")
        for name, tensor, expected in (("cutoff", cutoff, (rows,)), ("allowed", allowed, (len(used), rows)), ("keep", keep, (rows, int(edges[len(used)]))), ("double", double, (rows, cumulative.shape[1] + 1))):
            if tuple(tensor.shape) != tuple(expected) or tensor.numel() == 0:
                raise ValueError(f"starts_kept_all: {name} has shape {tuple(tensor.shape)} where {tuple(expected)} is needed (the batch has {rows} periods and {len(used)} widths); an empty or short tensor would be read as a null pointer on a device")
        if triton is None or not cumulative.is_cuda:
            for position, w in enumerate(used):
                self.starts_kept(w, stride[w], size, overshoot[w], depth_min, cumulative, squares, double, cutoff, allowed[position], keep[:, edges[position]:edges[position + 1]])
            return
        block = 256
        ints, floats, positions, coefficients, tiles, first_term = [], [], [], [], [0], 0
        for position, w in enumerate(used):
            count = len(range(0, size - w + 1, stride[w]))
            proxy = self.proxy.get(w)
            cheap = self.loose.get(w) if proxy else None  # the cascade needs both
            terms = len(proxy[0]) if proxy else 0
            cheap_terms = len(cheap[0]) if cheap else 0
            ints.append([w, stride[w], count, int(edges[position]), first_term, terms, self.length[w], first_term + terms, cheap_terms])
            floats.append([overshoot[w] / SIGNAL_DEPTH, self.square[w], proxy[2] if proxy else 0.0, proxy[3] if proxy else 0.0, proxy[4] if proxy else self.round_r2, depth_min, cheap[2] if cheap else 0.0, cheap[3] if cheap else 0.0])
            if proxy:
                positions += proxy[0]
                coefficients += proxy[1]
            if cheap:
                positions += cheap[0]
                coefficients += cheap[1]
            first_term += terms + cheap_terms
            tiles.append(tiles[-1] + triton.cdiv(count, block))
        device = cumulative.device
        with on_device(device):
            _keep_all_kernel[(tiles[-1], cumulative.shape[0])](
                cumulative, squares, double, cutoff, allowed.to(torch.int8).contiguous(), torch.as_tensor(positions or [0], dtype=torch.int64, device=device), torch.as_tensor(coefficients or [0.0], dtype=torch.float64, device=device), keep,
                torch.as_tensor(ints, dtype=torch.int64, device=device), torch.as_tensor(floats, dtype=torch.float64, device=device), torch.as_tensor(tiles, dtype=torch.int64, device=device), len(used), cumulative.shape[0], cumulative.stride(0), double.stride(0), keep.stride(0), block=block)

    def correlations(self, patched, rows, starts, width_index, residual=False):
        """Return AR = sum_j p_j w (1 - patched[row, start + j]) for each start, in float64.

        `rows`, `starts` and `width_index` give the row in the batch, the first sample and the position of the width
        in `widths` of each start. On a GPU one kernel program computes each start; on a CPU blocks of gathered
        windows are multiplied by the template. With `residual`, `patched` is weight * (1 - flux) already.
        """
        if patched.dim() != 2 or patched.stride(1) != 1:
            raise ValueError(f"correlations: patched must have unit stride along the points (a kernel reads it as rows of dense data); it has shape {tuple(patched.shape)} and strides {patched.stride()}")
        for name, array in (("rows", rows), ("starts", starts), ("width_index", width_index)):
            if array.dim() != 1 or not array.is_contiguous():
                raise ValueError(f"correlations: {name} must be a contiguous one-dimensional array (a kernel reads it with unit stride and a view with stride 0 or 2 is read out of range); it has shape {tuple(array.shape)} and strides {array.stride()}")
        if not (len(rows) == len(starts) == len(width_index)):
            raise ValueError(f"correlations: rows, starts and width_index have {len(rows)}, {len(starts)} and {len(width_index)} entries; they must be one for each start")
        widths = self.lengths_tensor[width_index]  # the template's samples, not the box's
        out = torch.empty(len(rows), dtype=torch.float64, device=patched.device)
        if len(rows) == 0:
            return out
        if triton is not None and patched.is_cuda:
            offsets = self.profile_offsets[width_index]
            with on_device(patched.device):
                for first in range(0, len(rows), DOT_LAUNCH):  # a launch of 2^25 programs or more is not dispatched whole
                    part = slice(first, first + DOT_LAUNCH)
                    _dots_kernel[(len(rows[part]),)](patched, self.profiles, self.weight_tensor, out[part], rows[part], starts[part], widths[part], offsets[part], patched.stride(0), block=256, residual=residual)
            return out
        for position in torch.unique(width_index).tolist():
            chosen = (width_index == position).nonzero(as_tuple=True)[0]
            w = self.length[self.widths[position]]
            profile = self.profile[self.widths[position]]
            block = max(1, ELEMENTS // w)
            offsets = torch.arange(w, device=patched.device)
            for first in range(0, len(chosen), block):
                part = chosen[first:first + block]
                window = patched[rows[part][:, None], starts[part][:, None] + offsets[None, :]]
                window = window if residual else self.weight * (1.0 - window)
                out[part] = window @ profile
        return out

    def deepest(self, cumulative, used_widths, coarse):
        """Return (index, sum) of the deepest box of each width (rows) and period (columns).

        The starts are spaced `coarse[w]` samples apart; the index is that of the first start with the smallest sum of
        the flux over the width.
        """
        b, size = cumulative.shape[0], cumulative.shape[1] - 1
        counts = [len(range(0, size - w + 1, coarse[w])) for w in used_widths]
        if triton is not None and cumulative.is_cuda:
            ints = torch.as_tensor([[w, coarse[w], count] for w, count in zip(used_widths, counts)], dtype=torch.int64, device=cumulative.device)
            index = torch.empty((len(used_widths), b), dtype=torch.int64, device=cumulative.device)
            depth = torch.empty((len(used_widths), b), dtype=torch.float64, device=cumulative.device)
            with on_device(cumulative.device):
                _deepest_kernel[(len(used_widths), b)](cumulative, index, depth, ints, b, cumulative.stride(0), block=256)
            return index, depth
        indices, depths = [], []
        for w in used_widths:
            difference = cumulative[:, slice(w, size + 1, coarse[w])] - cumulative[:, slice(0, size - w + 1, coarse[w])]
            depths_w, index_w = torch.min(difference, dim=1)
            indices.append(index_w)
            depths.append(depths_w)
        return torch.stack(indices), torch.stack(depths)

    def upper_bound(self, w, k, r2, double_sum, step, size):
        """Return the bound on the gain of every start of width w (`k`, `r2`: batch by starts).

        It is the Cauchy-Schwarz bound and, where the template has proxies, the smallest of the proxy bounds.
        """
        a2 = self.square[w]
        bound = 2.0 * k * torch.sqrt(torch.clamp((r2 + self.round_r2) * a2, min=0.0)) - k * k * a2
        if w in self.proxy:
            _, _, constant, error, round_r2 = self.proxy[w]
            reach = size - w + 1
            columns = step * torch.arange(len(range(0, reach, step)), device=double_sum.device)[None, :]
            gathered = double_sum[:, self.proxy_index[w][0] + columns]
            correlation = torch.einsum("t,btc->bc", self.proxy_index[w][1], gathered)
            upper = correlation + constant + torch.sqrt(torch.clamp(error * (r2 + round_r2), min=0.0))
            bound = torch.minimum(bound, 2.0 * k * upper - k * k * a2)
            if w in self.loose:  # the cheap proxy is a second certified bound: a start must pass both
                positions, coefficients, constant, error, round_r2 = self.loose[w]
                gathered = double_sum[:, torch.as_tensor(positions, device=double_sum.device)[:, None] + columns]
                correlation = torch.einsum("t,btc->bc", torch.as_tensor(coefficients, device=double_sum.device, dtype=torch.float64), gathered)
                upper = correlation + constant + torch.sqrt(torch.clamp(error * (r2 + round_r2), min=0.0))
                bound = torch.minimum(bound, 2.0 * k * upper - k * k * a2)
        return bound
