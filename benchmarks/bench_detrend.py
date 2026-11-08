"""Time the biweight detrend of transitgpu.detrend against wotan's, on synthetic light curves, and print the tables.

    uv run --group rocm --group test python benchmarks/bench_detrend.py [--repeats 7] [--processes 1 2 4 8 16] [--cold 16] [--json benchmarks/results-detrend.json]

The light curves are made here with a fixed seed: 27.4 days of a cadence, flux of 1 with Gaussian noise of 1,000 ppm, a sine of 2,000 ppm with a period of 3.7 days, a transit of 3,000 ppm and 3 hours every 5.3 days, a gap of one
day in the middle and 0.2 days on each side of it removed, and 1 point in 500 a NaN. The transits are the mask of the filter. Each configuration is a cadence and a window (CONFIGURATIONS), and its padded width is the widest window
in points rounded up to a power of two. A row is timed after one first call as the median of --repeats calls, from the numpy arrays of the call to the numpy trend, so the device rows include building the windows, both transfers
and the launches; the first call is given apart, since for the kernel it includes the compile when the Triton cache does not hold that width. Each row also gives its largest difference from wotan's trend, relative to the trend,
and whether the NaN positions are equal. The kernel row is not run where the padded width is over --kernel-max-width, whose compile does not end in bounded time; a torch row on the CPU that would take over --max-seconds a call,
scaled by the points times the padded width from the configuration before it, is skipped. Every skip is said in the table.
With --processes, that many processes each detrend the 2-minute light curve with the 0.75-day window on the device for --seconds after a common start, once for each count given. With --cold N, N processes start together with
an empty Triton cache and detrend the 2-minute light curve with the 1.5-day window once, and the table gives the time of that first call, which includes the compile of each process.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import tempfile
import time

import numpy as np

CONFIGURATIONS = (("30 min", 1800, 0.75), ("10 min", 600, 0.75), ("200 s", 200, 0.75), ("2 min", 120, 0.75), ("2 min", 120, 1.5), ("20 s", 20, 0.75))
DAYS = 27.4
ROWS = (("wotan", "wotan 1.10, CPU, 1 thread"), ("torch-cpu", "transitgpu.detrend, torch on the CPU"), ("kernel", "transitgpu.detrend, Triton kernel on the GPU"), ("torch-device", "transitgpu.detrend, torch on the GPU"))
SHARED = (120, 0.75)  # the light curve of the processes sharing the device
COLD = (120, 1.5)  # the light curve of the cold start


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


def run(configurations=CONFIGURATIONS, repeats=7, max_seconds=60.0, kernel_max_width=2048, device="auto"):
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
        if before is not None and "median" in before and before["median"] * len(time_values) * width / (previous["points"] * previous["padded_width"]) > max_seconds:  # the work is the points times the window
            entry["rows"]["torch-cpu"] = {"skipped": f"over {max_seconds:g} s a call by scaling the configuration before it"}
        else:
            trend, first, seconds = timed(lambda: detrend.biweight(time_values, flux, window, mask=mask, device="cpu"), repeats if len(time_values) < 50000 else 3)
            entry["rows"]["torch-cpu"] = {"first": first, "seconds": seconds, "median": statistics.median(seconds), **agreement(trend, reference)}
        if use_device:
            for key, method in (("kernel", "kernel"), ("torch-device", "torch")):
                if method == "kernel" and width > kernel_max_width:
                    entry["rows"][key] = {"skipped": f"padded width {width} over {kernel_max_width}: the compile does not end in bounded time"}
                    continue
                trend, first, seconds = timed(lambda method=method: detrend.biweight(time_values, flux, window, mask=mask, device="cuda", method=method), repeats)
                entry["rows"][key] = {"first": first, "seconds": seconds, "median": statistics.median(seconds), **agreement(trend, reference)}
        results["configurations"].append(entry)
        print(f"{heading(entry)}: " + ", ".join(f"{key} {cell(row)}" for key, row in entry["rows"].items()), file=sys.stderr, flush=True)
        previous = entry
    return results


def _worker(step_seconds, window, seconds, device, barrier, queue):
    """One process of `shared`: detrend the light curve of `step_seconds` once (the first call), wait for the others, then detrend it for `seconds`, and put (first, calls, elapsed) on `queue`."""
    from transitgpu import detrend

    time_values, flux, mask = light_curve(step_seconds)
    began = time.perf_counter()
    detrend.biweight(time_values, flux, window, mask=mask, device=device)
    first = time.perf_counter() - began
    barrier.wait()
    began, calls = time.perf_counter(), 0
    while time.perf_counter() - began < seconds:
        detrend.biweight(time_values, flux, window, mask=mask, device=device)
        calls += 1
    queue.put((first, calls, time.perf_counter() - began))


def shared(processes, step_seconds=SHARED[0], window=SHARED[1], seconds=10.0, device="cuda", cache_dir=None):
    """Detrends a second of `processes` processes that detrend the light curve of `step_seconds` with `window` at once on one device, each for `seconds` after a common start, with the first call of each.
    `cache_dir` is the Triton cache of the processes (TRITON_CACHE_DIR), for a start with an empty cache."""
    import multiprocessing

    context = multiprocessing.get_context("spawn")  # a fork of a process that has used the device is not safe
    barrier, queue = context.Barrier(processes), context.Queue()
    saved = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir is not None:
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)  # the spawned processes take the environment at their start
    try:
        workers = [context.Process(target=_worker, args=(step_seconds, window, seconds, device, barrier, queue)) for _ in range(processes)]
        for worker in workers:
            worker.start()
    finally:
        if cache_dir is not None:
            os.environ.pop("TRITON_CACHE_DIR") if saved is None else os.environ.__setitem__("TRITON_CACHE_DIR", saved)
    finished = [queue.get() for _ in workers]
    for worker in workers:
        worker.join()
    return {"processes": processes, "step_seconds": step_seconds, "window_days": window, "seconds": seconds, "first": [first for first, _, _ in finished], "calls": [calls for _, calls, _ in finished],
            "per_second": sum(calls / max(elapsed, 1e-9) for _, calls, elapsed in finished)}


def cell(row):
    if row is None:
        return "not run"
    if "skipped" in row:
        return "skipped"
    seconds = row["seconds"]
    return f"{1000 * row['median']:.3g} ({1000 * min(seconds):.3g}–{1000 * max(seconds):.3g})"


def heading(entry):
    return f"{entry['cadence']}, {entry['window_days']:g} d ({entry['points']:,} points, width {entry['padded_width']:,})"


def table(results):
    """The table of the milliseconds a call takes after the first, median with the minimum and maximum of the repeats; the first call of the device rows; and the largest relative difference from wotan's trend."""
    entries = results["configurations"]
    lines = ["| detrend of one sector | " + " | ".join(heading(entry) for entry in entries) + " |", "| --- |" + " --- |" * len(entries)]
    for key, name in ROWS:
        lines.append(f"| {name} | " + " | ".join(cell(entry["rows"].get(key)) for entry in entries) + " |")
    for key, name in ROWS[2:]:
        lines.append(f"| first call, {name.split(', ', 1)[1]} (s) | " + " | ".join(f"{entry['rows'][key]['first']:.3g}" if "first" in entry["rows"].get(key, {}) else "-" for entry in entries) + " |")
    rows = [row for entry in entries for row in entry["rows"].values()]
    worst = max((row["largest_relative_difference"] for row in rows if "largest_relative_difference" in row), default=0.0)
    equal = all(row["nan_equal"] for row in rows if "nan_equal" in row)
    skipped = sorted({row["skipped"] for row in rows if "skipped" in row})
    text = "\n".join(lines) + f"\n\nLargest difference from wotan's trend, relative to the trend: {worst:.2g}; the NaN positions are {'equal' if equal else 'NOT equal'} in every row."
    return text + "".join(f"\nSkipped: {reason}." for reason in skipped)


def shared_table(rows):
    """The table of the detrends a second that processes sharing one device reach together and each."""
    lines = ["| processes sharing the device | detrends a second, all processes | detrends a second, each process |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['processes']} | {row['per_second']:.0f} | {row['per_second'] / row['processes']:.0f} |")
    return "\n".join(lines)


def cold_table(row):
    """The first call of processes that start together with an empty Triton cache."""
    first = row["first"]
    return f"{row['processes']} processes starting together with an empty Triton cache, {row['window_days']:g} d window at {row['step_seconds']} s: first call {min(first):.1f} to {max(first):.1f} s (median {statistics.median(first):.1f} s)."


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--kernel-max-width", type=int, default=2048, help="the widest padded window whose kernel row is run")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--processes", type=int, nargs="*", default=[], help="also measure this many processes detrending at once on the device, for each count given")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long each process of --processes detrends")
    parser.add_argument("--cold", type=int, default=0, help="also measure the first call of this many processes starting together with an empty Triton cache")
    parser.add_argument("--json", help="write the results here")
    options = parser.parse_args(arguments)
    results = run(repeats=options.repeats, max_seconds=options.max_seconds, kernel_max_width=options.kernel_max_width, device=options.device)
    device = "cpu" if options.device == "cpu" else "cuda"
    if options.processes:
        results["shared"] = []
        for count in options.processes:
            results["shared"].append(shared(count, seconds=options.seconds, device=device))
            print(f"{count} processes sharing the device: {results['shared'][-1]['per_second']:.0f} detrends a second", file=sys.stderr, flush=True)
    if options.cold:
        with tempfile.TemporaryDirectory() as empty:
            results["cold"] = shared(options.cold, step_seconds=COLD[0], window=COLD[1], seconds=0.0, device=device, cache_dir=empty)
    if options.json:
        with open(options.json, "w") as handle:
            json.dump(results, handle, indent=1)
            handle.write("\n")
    print(table(results))
    if options.processes:
        print()
        print(shared_table(results["shared"]))
    if options.cold:
        print()
        print(cold_table(results["cold"]))
    print(json.dumps(results["versions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
