"""The share of an exhaustive box search that the binned box search keeps (Sect. "The box search" of the paper).

For trial periods spread over TLS's period grid, on Gaussian noise, the box that the search finds at each period is
scored exactly (the statistic of Eq. 1 without the factor 1/sigma^2, over the points in the box, no bins) and divided
by the best statistic of an exhaustive continuous search at that period (every start at a point, every duration of a
geometric grid of ratio 1.02 between the shortest and the longest duration the search tries). The settings are the
minimum number of bins across the shortest duration (3 and 12) and the ratio of consecutive durations (1.5 and 1.1).
The share is 1 where the binned search finds the best box.

    python paper/scripts/bins_share.py [--points 5000] [--baseline 400] [--periods 200] [--seed 1] [--out paper/bins_table.tex]

The searches run on the GPU; the exhaustive search runs on the host (about a minute for the defaults).
"""
import argparse
import json
from pathlib import Path

import numpy as np

MINIMUM_POINTS = 3  # the least number of points inside and outside a box, as in the search
FINE_RATIO = 1.02


def statistic(total, count, size):
    return total * total / (count * (1.0 - count / size))


def exhaustive(phase_days, y, period, durations):
    """Return the best statistic of a dip at one period, over every start at a point and every duration (days).

    The windows are over the points in phase order, wrapped around the period.
    """
    order = np.argsort(phase_days, kind="stable")
    phase, flux = phase_days[order], y[order]
    size = len(flux)
    doubled = np.concatenate([phase, phase + period])
    sums = np.concatenate([[0.0], np.cumsum(np.concatenate([flux, flux]))])
    best = 0.0
    for duration in durations:
        ends = np.searchsorted(doubled, phase + duration, side="left")
        count = ends - np.arange(size)
        total = sums[ends] - sums[np.arange(size)]
        usable = (total < 0) & (count >= MINIMUM_POINTS) & (count <= size - MINIMUM_POINTS)
        if usable.any():
            best = max(best, float(statistic(total[usable], count[usable], size).max()))
    return best


def box_statistic(phase_days, y, period, center, width):
    """Return the statistic of the box [center - width / 2, center + width / 2] (days, modulo the period), without bins.

    It is 0 for a box that is not a dip or holds too few points.
    """
    offset = (phase_days - (center - width / 2)) % period
    inside = offset < width
    count = int(inside.sum())
    total = float(y[inside].sum())
    if total >= 0 or count < MINIMUM_POINTS or count > len(y) - MINIMUM_POINTS:
        return 0.0
    return float(statistic(total, count, len(y)))


def shares(points, baseline, count, seed, settings):
    import transitgpu
    from transitgpu.core import duration_fractions

    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, baseline, points))
    flux = 1.0 + rng.normal(0.0, 1e-3, points)
    y = flux - flux.mean()
    grid = transitgpu.grid_of(time, flux, 0.5, 14.0)
    chosen = np.sort(grid[np.unique(np.linspace(0, len(grid) - 1, count).astype(int))])
    shortest, longest = duration_fractions(chosen)
    reference = []
    for period, low, high in zip(chosen, shortest, longest):
        durations = low * period * FINE_RATIO ** np.arange(int(np.ceil(np.log(high / low) / np.log(FINE_RATIO))) + 1)
        reference.append(exhaustive(time % period, y, period, durations))
    reference = np.array(reference)
    out = {}
    for bins, step in settings:
        found = transitgpu.search(time, flux, 0.5, 14.0, periods=chosen, boxes=True, minimum_bins=bins, duration_step=step)
        values = np.array([box_statistic(time % period, y, period, center, width) for period, center, width in zip(chosen, found["centers"], found["widths_days"])])
        usable = reference > 0
        out[(bins, step)] = values[usable] / reference[usable]
    return chosen, reference, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--points", type=int, default=5000)
    parser.add_argument("--baseline", type=float, default=400.0)
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "bins_table.tex")
    arguments = parser.parse_args()
    settings = [(3, 1.5), (3, 1.1), (12, 1.5), (12, 1.1)]
    chosen, reference, out = shares(arguments.points, arguments.baseline, arguments.periods, arguments.seed, settings)
    lines = [f"% written by scripts/bins_share.py (points {arguments.points}, baseline {arguments.baseline:g} d, {len(chosen)} periods, seed {arguments.seed})"]
    lines += ["\\begin{tabular}{rrrrr}", "\\toprule", "bins & step & median & 5th percentile & smallest \\\\", "\\midrule"]
    for (bins, step), share in out.items():
        lines.append(f"{bins} & {step:g} & {np.median(share):.3f} & {np.percentile(share, 5):.3f} & {share.min():.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    arguments.out.write_text("\n".join(lines) + "\n")
    arguments.out.with_suffix(".json").write_text(json.dumps({f"{b}_{s}": v.tolist() for (b, s), v in out.items()}))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
