"""Time the biweight detrend of transitgpu.detrend against wotan's, on synthetic light curves, and print the tables.

    uv run --group rocm --group test python benchmarks/bench_detrend.py [--repeats 7] [--configuration N ...] [--rows ROW ...] [--json FILE]
    uv run --group rocm --group test python benchmarks/bench_detrend.py --merge OUT.json IN.json [IN.json ...]

The light curves are made here with a fixed seed: 27.4 days of a cadence, flux of 1 with Gaussian noise of 1,000 ppm, a sine of 2,000 ppm with a period of 3.7 days, a transit of 3,000 ppm and 3 hours every 5.3 days, a gap of one
day in the middle and 0.2 days on each side of it removed, and 1 point in 500 a NaN. The transits are the mask of the filter. Each configuration is a cadence and a window (CONFIGURATIONS), and its padded width is the widest window
in points rounded up to a power of two. A row is timed after one first call as the median of --repeats calls, from the numpy arrays of the call to the numpy trend, so the device rows include building the windows, both transfers
and the launches; the first call is given apart, since for the kernel it includes the compile when the Triton cache does not hold that width. Each row also gives its largest difference from wotan's trend, relative to the trend,
and whether the NaN positions are equal. The kernel row is not run where the padded width is over --kernel-max-width, whose compile does not end in bounded time; a torch row on the CPU that would take over --max-seconds a call,
scaled by the points times the padded width from the configuration before it, is skipped, and so is a row that --rows leaves out. A skipped cell says "over the limit" for the kernel above KERNEL_WIDTH_LIMIT and "not run" otherwise.
--configuration runs only the configurations of those indices of CONFIGURATIONS, so that each can run as a job of its own; --merge joins such files, in the order given, into one.
"""
import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time

import numpy as np

CONFIGURATIONS = (("30 min", 1800, 0.75), ("10 min", 600, 0.75), ("200 s", 200, 0.75), ("2 min", 120, 0.75), ("2 min", 120, 1.5), ("20 s", 20, 0.75))
DAYS = 27.4
ROWS = (("wotan", "wotan 1.10, CPU, 1 thread"), ("torch-cpu", "transitgpu.detrend, torch on the CPU, torch's default threads"), ("kernel", "transitgpu.detrend, Triton kernel on the GPU"), ("torch-device", "transitgpu.detrend, torch on the GPU"))


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


def padded_width(time_values, flux, mask, window):
    """The widest window of the light curve in points, rounded up to a power of two, as transitgpu.detrend pads it."""
    from transitgpu import detrend

    good = ~np.isnan(time_values * flux)
    _, start, end = detrend.windows(time_values[good], flux[good], ~mask[good], window)
    width = int((end - start).max())
    return 1 << (width - 1).bit_length()


def versions():
    import torch

    out = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    for name in ("wotan", "numba", "triton", "triton-rocm"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    try:  # the CPU scheduler, which the times depend on: sched_ext disabled is the kernel's default
        with open("/sys/kernel/sched_ext/state") as handle:
            out["sched_ext"] = handle.read().strip()
    except OSError:
        out["sched_ext"] = None
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["hip"] = torch.version.hip
    return out


def timed(function, repeats):
    """(result, seconds of the first call, seconds of each of `repeats` calls after it)."""
    began = time.perf_counter()
    function()
    first = time.perf_counter() - began
    seconds = []
    for _ in range(repeats):
        began = time.perf_counter()
        result = function()
        seconds.append(time.perf_counter() - began)
    return result, first, seconds


def agreement(trend, reference):
    both = np.isfinite(trend) & np.isfinite(reference)
    return {"nan_equal": bool(np.array_equal(np.isnan(trend), np.isnan(reference))), "largest_relative_difference": float((np.abs(trend[both] - reference[both]) / np.abs(reference[both])).max()) if both.any() else 0.0}


def run(configurations=CONFIGURATIONS, repeats=7, max_seconds=60.0, kernel_max_width=2048, device="auto", rows=None):
    import torch
    from wotan import flatten

    from transitgpu import detrend

    use_device = torch.cuda.is_available() if device == "auto" else device == "cuda"
    results = {"versions": versions(), "repeats": repeats, "kernel_width_limit": detrend.KERNEL_WIDTH_LIMIT, "configurations": []}
    previous = None
    for label, step, window in configurations:
        time_values, flux, mask = light_curve(step)
        width = padded_width(time_values, flux, mask, window)
        entry = {"cadence": label, "step_seconds": step, "window_days": window, "points": int(len(time_values)), "padded_width": width, "rows": {}}
        reference, first, seconds = timed(lambda: flatten(time_values, flux, method="biweight", window_length=window, return_trend=True, mask=mask)[1], repeats)
        entry["rows"]["wotan"] = {"first": first, "seconds": seconds, "median": statistics.median(seconds)}
        before = previous["rows"].get("torch-cpu") if previous else None
        if rows is not None and "torch-cpu" not in rows:
            entry["rows"]["torch-cpu"] = {"skipped": "not run: left out by --rows"}
        elif before is not None and "median" in before and before["median"] * len(time_values) * width / (previous["points"] * previous["padded_width"]) > max_seconds:  # the work is the points times the window
            entry["rows"]["torch-cpu"] = {"skipped": f"over {max_seconds:g} s a call by scaling the configuration before it"}
        else:
            trend, first, seconds = timed(lambda: detrend.biweight(time_values, flux, window, mask=mask, device="cpu", method="torch"), repeats if len(time_values) < 50000 else 3)
            entry["rows"]["torch-cpu"] = {"first": first, "seconds": seconds, "median": statistics.median(seconds), **agreement(trend, reference)}
        if use_device:
            for key, method in (("kernel", "kernel"), ("torch-device", "torch")):
                if rows is not None and key not in rows:
                    entry["rows"][key] = {"skipped": "not run: left out by --rows"}
                    continue
                if method == "kernel" and width > kernel_max_width:
                    entry["rows"][key] = {"skipped": f"padded width {width} over {kernel_max_width}: the compile does not end in bounded time"}
                    continue
                trend, first, seconds = timed(lambda method=method: detrend.biweight(time_values, flux, window, mask=mask, device="cuda", method=method), repeats)
                entry["rows"][key] = {"first": first, "seconds": seconds, "median": statistics.median(seconds), **agreement(trend, reference)}
        results["configurations"].append(entry)
        print(f"{heading(entry)}: " + ", ".join(f"{key} {cell(row)}" for key, row in entry["rows"].items()), file=sys.stderr, flush=True)
        previous = entry
    return results


def cell(row, over_limit=False):
    if row is None or "skipped" in row:
        return "over the limit" if over_limit else "not run"
    seconds = row["seconds"]

    def milliseconds(value):
        return f"{1000 * value:,.0f}" if value >= 1 else f"{1000 * value:.3g}"

    return f"{milliseconds(row['median'])} ({milliseconds(min(seconds))}–{milliseconds(max(seconds))})"


def heading(entry):
    return f"{entry['cadence']}, {entry['window_days']:g} d ({entry['points']:,} points, width {entry['padded_width']:,})"


def table(results):
    """The table of the milliseconds a call takes after the first, median with the minimum and maximum of the repeats, and the largest relative difference from wotan's trend. The first calls are in the results, not in the table."""
    entries = results["configurations"]
    lines = ["| detrend of one sector | " + " | ".join(heading(entry) for entry in entries) + " |", "| --- |" + " --- |" * len(entries)]
    for key, name in ROWS:
        lines.append(f"| {name} | " + " | ".join(cell(entry["rows"].get(key), key == "kernel" and entry["padded_width"] > results["kernel_width_limit"]) for entry in entries) + " |")
    rows = [row for entry in entries for row in entry["rows"].values()]
    worst = max((row["largest_relative_difference"] for row in rows if "largest_relative_difference" in row), default=0.0)
    equal = all(row["nan_equal"] for row in rows if "nan_equal" in row)
    return "\n".join(lines) + f"\n\nLargest difference from wotan's trend, relative to the trend: {worst:.2g}; the NaN positions are {'equal' if equal else 'NOT equal'} in every row."


def merge(paths):
    """The results of several files of this script joined into one: the configurations in the order of the files, and the versions, repeats and limit of the first, which all must share."""
    parts = [json.loads(open(path).read()) for path in paths]
    first = parts[0]
    for part in parts[1:]:
        for key in ("versions", "repeats", "kernel_width_limit"):
            if part[key] != first[key]:
                raise ValueError(f"the files differ in {key}: {first[key]!r} and {part[key]!r}")
    merged = {**first, "configurations": [entry for part in parts for entry in part["configurations"]]}
    if any("checks" in part for part in parts):
        merged["checks"] = [part.get("checks") for part in parts]  # the checks of the job of each part, in its order
    return merged


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--kernel-max-width", type=int, default=2048, help="the widest padded window whose kernel row is run")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--configuration", type=int, nargs="*", help="the indices of CONFIGURATIONS to run (all when not given)")
    parser.add_argument("--rows", nargs="*", choices=[key for key, _ in ROWS[1:]], help="the rows to run beside wotan's (all when not given)")
    parser.add_argument("--merge", nargs="+", metavar="FILE", help="write the first file from the configurations of the others, and print its table")
    parser.add_argument("--json", help="write the results here")
    options = parser.parse_args(arguments)
    if options.merge:
        results = merge(options.merge[1:])
        with open(options.merge[0], "w") as handle:
            json.dump(results, handle, indent=1)
            handle.write("\n")
    else:
        chosen = CONFIGURATIONS if options.configuration is None else tuple(CONFIGURATIONS[index] for index in options.configuration)
        results = run(chosen, repeats=options.repeats, max_seconds=options.max_seconds, kernel_max_width=options.kernel_max_width, device=options.device, rows=options.rows)
        if options.json:
            with open(options.json, "w") as handle:
                json.dump(results, handle, indent=1)
                handle.write("\n")
    print(table(results))
    print(json.dumps(results["versions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
