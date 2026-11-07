"""Time the biweight detrend of transitgpu.detrend against wotan's, on synthetic light curves, and print a table.

    uv run --group rocm --group test python benchmarks/bench_detrend.py [--repeats 7] [--json benchmarks/results-detrend.json]

The light curves are made here with a fixed seed: 27.4 days of a cadence (30 min, 10 min, 2 min, 20 s), flux of 1 with Gaussian noise of 1,000 ppm, a sine of 2,000 ppm with a period of 3.7 days, a transit of 3,000 ppm and 3 hours every
5.3 days, a gap of one day in the middle and 0.2 days on each side of it removed, and 1 point in 500 a NaN. The transits are the mask of the filter. The window is 0.75 days. A row is timed after one warm-up call as the median of --repeats
calls, from the numpy arrays of the call to the numpy trend, so the device rows include building the windows, both transfers and the launches. Each row also gives its largest difference from wotan's trend, relative to the trend, and whether the NaN
positions are equal. A row that would take over --max-seconds a call, from the call before it at a shorter cadence scaled by the points, is skipped and said.
"""
import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time

import numpy as np

CADENCES = (("30 min", 1800), ("10 min", 600), ("2 min", 120), ("20 s", 20))
WINDOW = 0.75
DAYS = 27.4
ROWS = (("wotan", "wotan 1.10, CPU, 1 thread"), ("torch-cpu", "transitgpu.detrend, torch on the CPU"), ("device", "transitgpu.detrend, Triton kernel on the GPU"))


def light_curve(step_seconds, seed=0):
    """(time, flux, mask) of one synthetic light curve."""
    rng = np.random.default_rng(seed)
    step = step_seconds / 86400
    time_values = np.arange(0.0, DAYS, step)
    flux = 1 + 0.002 * np.sin(2 * np.pi * time_values / 3.7) + rng.normal(0, 1e-3, len(time_values))
    mask = (time_values % 5.3) < 3 / 24
    flux[mask] -= 0.003
    keep = ~((time_values > DAYS / 2 - 0.5 - 0.2) & (time_values < DAYS / 2 + 0.5 + 0.2))
    flux[rng.random(len(flux)) < 1 / 500] = np.nan
    return time_values[keep], flux[keep], mask[keep]


def versions():
    import torch

    out = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
    for name in ("wotan", "numba", "triton", "triton-rocm"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["hip"] = torch.version.hip
    return out


def timed(function, repeats):
    function()
    seconds = []
    for _ in range(repeats):
        began = time.perf_counter()
        result = function()
        seconds.append(time.perf_counter() - began)
    return result, seconds


def agreement(trend, reference):
    both = np.isfinite(trend) & np.isfinite(reference)
    return {"nan_equal": bool(np.array_equal(np.isnan(trend), np.isnan(reference))), "largest_relative_difference": float((np.abs(trend[both] - reference[both]) / np.abs(reference[both])).max()) if both.any() else 0.0}


def run(cadences=CADENCES, repeats=7, max_seconds=60.0, device="auto"):
    import torch
    from wotan import flatten

    from transitgpu import detrend

    use_device = torch.cuda.is_available() if device == "auto" else device == "cuda"
    results = {"versions": versions(), "window_days": WINDOW, "repeats": repeats, "cadences": []}
    for label, step in cadences:
        time_values, flux, mask = light_curve(step)
        entry = {"cadence": label, "step_seconds": step, "points": int(len(time_values)), "rows": {}}
        reference, seconds = timed(lambda: flatten(time_values, flux, method="biweight", window_length=WINDOW, return_trend=True, mask=mask)[1], repeats)
        entry["rows"]["wotan"] = {"seconds": seconds, "median": statistics.median(seconds)}
        previous = results["cadences"][-1]["rows"].get("torch-cpu") if results["cadences"] else None
        slow = previous is not None and "median" in previous and previous["median"] * len(time_values) / results["cadences"][-1]["points"] > max_seconds
        if slow:
            entry["rows"]["torch-cpu"] = {"skipped": f"over {max_seconds:g} s a call by scaling the shorter cadence"}
        else:
            trend, seconds = timed(lambda: detrend.biweight(time_values, flux, WINDOW, mask=mask, device="cpu"), repeats if len(time_values) < 50000 else 3)
            entry["rows"]["torch-cpu"] = {"seconds": seconds, "median": statistics.median(seconds), **agreement(trend, reference)}
        if use_device:
            trend, seconds = timed(lambda: detrend.biweight(time_values, flux, WINDOW, mask=mask, device="cuda"), repeats)
            entry["rows"]["device"] = {"seconds": seconds, "median": statistics.median(seconds), **agreement(trend, reference)}
        results["cadences"].append(entry)
    return results


def cell(row):
    if row is None:
        return "not run"
    if "skipped" in row:
        return "skipped"
    seconds = row["seconds"]
    return f"{1000 * row['median']:.3g} ({1000 * min(seconds):.3g}–{1000 * max(seconds):.3g})"


def table(results):
    """The table of the milliseconds a call takes, median with the minimum and maximum of the repeats, and the largest relative difference from wotan's trend of each row."""
    cadences = results["cadences"]
    lines = ["| detrend of one sector | " + " | ".join(f"{entry['cadence']} ({entry['points']:,} points)" for entry in cadences) + " |", "| --- |" + " --- |" * len(cadences)]
    for key, name in ROWS:
        lines.append(f"| {name} | " + " | ".join(cell(entry["rows"].get(key)) for entry in cadences) + " |")
    worst = max((row["largest_relative_difference"] for entry in cadences for row in entry["rows"].values() if "largest_relative_difference" in row), default=0.0)
    equal = all(row["nan_equal"] for entry in cadences for row in entry["rows"].values() if "nan_equal" in row)
    return "\n".join(lines) + f"\n\nLargest difference from wotan's trend, relative to the trend: {worst:.2g}; the NaN positions are {'equal' if equal else 'NOT equal'} in every row."


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--json", help="write the results here")
    options = parser.parse_args(arguments)
    results = run(repeats=options.repeats, max_seconds=options.max_seconds, device=options.device)
    if options.json:
        with open(options.json, "w") as handle:
            json.dump(results, handle, indent=1)
            handle.write("\n")
    print(table(results))
    print(json.dumps(results["versions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
