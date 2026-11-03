"""The duration of the launches of a search on stacks of growing size: for each size, the launches of every kernel (count, mean and largest device time) under torch's profiler.

usage: python benchmarks/launch_size.py [--sizes 50000 250000 1000000 2000000 2525000] [--period-min 0.5] [--period-max 14] [--seconds 120] [--elements N]

--elements sets transitgpu.core.ONE_PASS_ELEMENTS (the budget of a launch, in period x point pairs); a very large value gives 512 periods a launch at every size, which is how the search launched before the budget existed.

The stack is white noise of 1000 ppm on a 25 s cadence from the first point of a two-year span, flat. Each size runs only after the smaller ones, and a size that takes longer than --seconds stops the run, so that a size whose launches are too long is not tried. A search is run once to compile and once under the profiler. Run it on a machine whose kernel log (journalctl -k) is read before and after: a GPU that also drives a display wants launches of about 10 ms.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src")]

STEP_SECONDS = 25.0
NOISE = 1.0e-3


def stack(points, seed=781):
    time_axis = np.arange(points) * STEP_SECONDS / 86400.0
    return time_axis, 1.0 + np.random.default_rng(seed).normal(0.0, NOISE, points)


def launches(points, period_min=0.5, period_max=14.0):
    """Return {kernel name: [device milliseconds of each launch]} of one search of a stack of `points` points, and the seconds the search took."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from transitgpu import search

    time_axis, flux = stack(points)
    search(time_axis, flux, period_min, period_max)
    torch.cuda.synchronize()
    began = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as recording:
        search(time_axis, flux, period_min, period_max)
        torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    kernels = {}
    for event in recording.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and event.device_time > 0:
            kernels.setdefault(event.name[:70], []).append(event.device_time / 1000.0)
    return kernels, seconds


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[50_000, 250_000, 1_000_000, 2_000_000, 2_525_000])
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=14.0)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--elements", type=int)
    options = parser.parse_args(arguments)
    if options.elements:
        from transitgpu import core

        core.ONE_PASS_ELEMENTS = options.elements
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("a size took longer than --seconds")))
    for points in options.sizes:
        signal.alarm(options.seconds)
        kernels, seconds = launches(points, options.period_min, options.period_max)
        signal.alarm(0)
        total = sum(len(v) for v in kernels.values())
        largest = max(max(v) for v in kernels.values())
        print(f"{points} points: {total} launches of {len(kernels)} kernels in {seconds:.2f} s; largest launch of any kernel {largest:.2f} ms", flush=True)
        for name, times in sorted(kernels.items(), key=lambda item: -max(item[1]))[:5]:
            print(f"    {name:<72} launches {len(times):6d}, mean {np.mean(times):8.3f} ms, largest {max(times):8.3f} ms", flush=True)


if __name__ == "__main__":
    main()
