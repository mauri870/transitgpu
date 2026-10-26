"""The funnel of the screening of the exact search, by template length and period range, at the benchmark sizes.

    taskset -c 0-15 python paper/scripts/screening_stats.py --json paper/scripts/screening_stats.json

One batch of 64 periods in each of four ranges of period, at 20,000, 100,000 and 200,000 points. The incumbent is
the screened search's own (the exact gain of the deepest box of each duration among every eighth start, as
`transitgpu.tls.window` takes it). Each stage is counted against the starts of the stage before it, and the starts
that survive all three bounds are checked against those that the screening kernel keeps for the same batch.
"""
import argparse, importlib.util, json, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("transitgpu_bench", Path(__file__).resolve().parents[2] / "benchmarks" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
from transitleastsquares import transitleastsquares  # noqa: E402
from transitleastsquares.validate import validate_args  # noqa: E402

from transitgpu.tls import window as W  # noqa: E402
from transitgpu.tls.fold import phase_order, phases  # noqa: E402
from transitgpu.tls.prefix import prefix_arrays  # noqa: E402
from transitgpu.tls.screen import SEED_COARSEN, SIGNAL_DEPTH, Screen  # noqa: E402

BUCKETS = ((0, 64, "under 64"), (64, 128, "64 to 127"), (128, 256, "128 to 255"), (256, 512, "256 to 511"), (512, 1024, "512 to 1,023"), (1024, 2048, "1,024 to 2,047"), (2048, 4096, "2,048 to 4,095"), (4096, 1 << 30, "4,096 or more"))
RANGES = ((0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 14.0))
BATCH = 64


def bucket_of(length):
    return next(label for low, high, label in BUCKETS if low <= length < high)


def bound_with(table, a2, k, r2, double, step, size, w):
    positions, coefficients, constant, error, round_r2 = table
    reach = size - w + 1
    index = torch.as_tensor(positions, device=double.device)[:, None] + step * torch.arange(len(range(0, reach, step)), device=double.device)[None, :]
    correlation = torch.einsum("t,btc->bc", torch.as_tensor(coefficients, device=double.device, dtype=torch.float64), double[:, index])
    return 2.0 * k * (correlation + constant + torch.sqrt(torch.clamp(error * (r2 + round_r2), min=0.0))) - k * k * a2


def run(points, places=None):
    t, y = bench.light_curve(points, 1000.0)
    model = transitleastsquares(t, y)
    model, _ = validate_args(model, dict(period_min=0.5, period_max=14.0, use_threads=1, show_progress_bar=False, verbose=False, backend="fused"))
    periods, durations = model._grids()
    overview, templates = model._templates(durations)
    problem = model._problem(model.t, model.y, model.dy, overview, templates)
    periods = np.asarray(periods)
    widths = sorted(int(w) for w in np.unique(overview["width_in_samples"]))
    max_width = max(widths) + max(widths) % 2
    margin = problem.T0_search_margin
    stride = {w: (max(int(w / (1 / margin)), 1) if margin > 0 and w > margin else 1) for w in widths}
    overshoot = {w: float(overview["overshoot"][np.flatnonzero(overview["width_in_samples"] == w)[0]]) for w in widths}
    first_row = {w: int(np.flatnonzero(overview["width_in_samples"] == w)[0]) for w in widths}
    screen = Screen(problem.y, problem.dy, problem.lc_arr, overview, widths, "cuda")
    time_t, y_t = torch.as_tensor(problem.t, device="cuda"), torch.as_tensor(problem.y, device="cuda")
    allowed_all = W.allowed_widths(periods, widths, len(t), float(t.max() - t.min()), problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max)
    rows = []
    selections = [(low, high, int(np.flatnonzero((periods >= low) & (periods < high))[len(np.flatnonzero((periods >= low) & (periods < high))) // 2 - BATCH // 2])) for low, high in RANGES] if places is None else [(place, place, place + 4 * BATCH) for place in places]
    for low, high, first in selections:
        chunk = periods[first:first + BATCH]
        phase, key = phases(time_t, torch.as_tensor(chunk, device="cuda"))
        order = phase_order(phase, key)
        patched, cumulative, squares, double, constant = prefix_arrays(y_t, order, screen.weight, screen.mean_rw, max_width)
        size = patched.shape[1]
        allowed = allowed_all[:, first:first + BATCH]
        used = [w for index, w in enumerate(widths) if allowed[index].any()]
        count_used = len(used)
        # the seed, as the search takes it
        coarse = {w: stride[w] * SEED_COARSEN for w in used}
        deepest, _ = screen.deepest(cumulative, used, coarse)
        seed_gain = torch.zeros((count_used, BATCH), dtype=torch.float64, device="cuda")
        for position, w in enumerate(used):
            starts = deepest[position] * coarse[w]
            mean = 1.0 - (cumulative[torch.arange(BATCH), starts + w] - cumulative[torch.arange(BATCH), starts]) / w
            k = mean * overshoot[w] / SIGNAL_DEPTH
            ar = screen.correlations(patched, torch.arange(BATCH, device="cuda"), starts, torch.full((BATCH,), screen.index_of[w], device="cuda"))
            gain = 2.0 * k * ar - k * k * screen.square[w]
            seed_gain[position] = torch.where((mean > problem.transit_depth_min) & torch.as_tensor(allowed[widths.index(w)], device="cuda"), gain, torch.zeros_like(gain))
        cutoff = seed_gain.clamp(min=0.0).amax(dim=0)
        stages = {label: dict(tested=0, columns=0, live=0, cs_without_allowance=0, cs=0, cheap=0, tight=0) for _, _, label in BUCKETS}
        for position, w in enumerate(used):
            step = stride[w]
            count = len(range(0, size - w + 1, step))
            mean = 1.0 - (cumulative[:, w:w + (count - 1) * step + 1:step] - cumulative[:, 0:(count - 1) * step + 1:step]) / float(w)
            k = mean * overshoot[w] / SIGNAL_DEPTH
            r2 = squares[:, screen.length[w]:screen.length[w] + (count - 1) * step + 1:step] - squares[:, 0:(count - 1) * step + 1:step]
            a2 = screen.square[w]
            row_ok = torch.as_tensor(allowed[widths.index(w)], device="cuda")[:, None]
            live = (mean > problem.transit_depth_min) & row_ok
            cs_old = live & (2.0 * k * torch.sqrt(torch.clamp(r2 * a2, min=0.0)) - k * k * a2 > cutoff[:, None])  # the bound without the allowance for the rounding of R2
            cs = live & (2.0 * k * torch.sqrt(torch.clamp((r2 + screen.round_r2) * a2, min=0.0)) - k * k * a2 > cutoff[:, None])
            cheap, tight = cs, cs
            if w in screen.proxy and w in screen.loose:
                cheap = cs & (bound_with(screen.loose[w], a2, k, r2, double, step, size, w) > cutoff[:, None])
                tight = cheap & (bound_with(screen.proxy[w], a2, k, r2, double, step, size, w) > cutoff[:, None])
            entry = stages[bucket_of(screen.length[w])]
            entry["tested"] += int(row_ok.sum()) * count
            entry["columns"] += BATCH * count
            for name, mask in (("live", live), ("cs_without_allowance", cs_old), ("cs", cs), ("cheap", cheap), ("tight", tight)):
                entry[name] += int(mask.sum())
        # the kernel's count for the same batch
        edges = np.concatenate([[0], np.cumsum([len(range(0, size - w + 1, stride[w])) for w in used])])
        keep = torch.zeros((BATCH, int(edges[-1])), dtype=torch.int8, device="cuda")
        screen.starts_kept_all(used, stride, overshoot, size, problem.transit_depth_min, cumulative, squares, double, cutoff, torch.as_tensor(allowed[[widths.index(w) for w in used]], device="cuda"), keep, edges)
        kernel_survivors = int(keep.sum())
        torch_survivors = sum(entry["tight"] for entry in stages.values())
        rows.append(dict(points=points, range=[low, high], first_period=float(chunk[0]), stages=stages, kernel_survivors=kernel_survivors, torch_survivors=torch_survivors))
        print(f"{points:7d} points, periods {low}-{high} d: survivors torch {torch_survivors}, kernel {kernel_survivors}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--points", type=int, nargs="+", default=[20000, 100000, 200000])
    parser.add_argument("--places", type=int, nargs="+", help="the first period (index in the grid) of each batch, in place of the four ranges (the batch starts 256 periods further)")
    options = parser.parse_args()
    results = []
    for points in options.points:
        results += run(points, options.places)
    if options.json:
        Path(options.json).write_text(json.dumps(results, indent=1))
    print()
    print("| points | periods (d) | template samples | tested | live | pass Cauchy-Schwarz | pass cheap proxy | pass TLS proxy (survivors) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in results:
        for label, entry in row["stages"].items():
            if entry["tested"] == 0:
                continue
            share = lambda n: f"{100 * n / entry['tested']:.3g}%"
            print(f"| {row['points']:,} | {row['range'][0]} to {row['range'][1]} | {label} | {entry['tested']:,} | {share(entry['live'])} | {share(entry['cs'])} | {share(entry['cheap'])} | {share(entry['tight'])} |")


if __name__ == "__main__":
    main()
