"""The search of a stack of each size with the periods of a launch bounded by the element budget (the default) against 512 periods a launch (the search before the budget), in one process: the periods a launch, the largest launch of any kernel, and the median search time over --repeats calls.

usage: python benchmarks/launch_compare.py [--sizes ...] [--repeats 5] [--old-limit 1000000] [--seconds 120]

The old behaviour is a budget so large that 512 periods are taken. It is run only up to --old-limit points, since its launches grow with the points (about 20 ms at 1,000,000). Sizes run in the order given, small to large, and a size that takes longer than --seconds stops the run. A row whose search is more than 5 percent slower with the budget is marked SLOWER; the stacks up to 312,500 points take the same batch either way, so their ratio is the noise of the measurement.
"""
import argparse
import signal
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src"), str(Path(__file__).resolve().parent)]

from launch_size import stack  # noqa: E402

OLD = 10**15


def measured(points, elements, repeats):
    import torch
    from torch.profiler import ProfilerActivity, profile

    from transitgpu import core, search

    core.ONE_PASS_ELEMENTS = elements
    time_axis, flux = stack(points)
    search(time_axis, flux, 0.5, 14.0)
    torch.cuda.synchronize()
    seconds = []
    for _ in range(repeats):
        began = time.perf_counter()
        search(time_axis, flux, 0.5, 14.0)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - began)
    with profile(activities=[ProfilerActivity.CUDA]) as recording:
        search(time_axis, flux, 0.5, 14.0)
        torch.cuda.synchronize()
    largest = max(event.device_time / 1000.0 for event in recording.events() if event.device_type == torch.autograd.DeviceType.CUDA)
    return core.one_pass_batch(points), largest, statistics.median(seconds)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 250_000, 312_500, 350_000, 450_000, 1_000_000, 1_300_000, 2_000_000, 2_525_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--old-limit", type=int, default=1_000_000)
    parser.add_argument("--seconds", type=int, default=120)
    options = parser.parse_args(arguments)
    from transitgpu import core

    budget = core.ONE_PASS_ELEMENTS
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("a size took longer than --seconds")))
    print(f"{'points':>9} {'batch old':>9} {'batch new':>9} {'largest old':>12} {'largest new':>12} {'search old':>11} {'search new':>11} {'new/old':>8}")
    for points in options.sizes:
        signal.alarm(options.seconds)
        old = measured(points, OLD, options.repeats) if points <= options.old_limit else None
        new = measured(points, budget, options.repeats)
        signal.alarm(0)
        if old:
            ratio = new[2] / old[2]
            print(f"{points:>9} {old[0]:>9} {new[0]:>9} {old[1]:>9.2f} ms {new[1]:>9.2f} ms {old[2]:>9.3f} s {new[2]:>9.3f} s {ratio:>8.3f}" + ("  SLOWER" if ratio > 1.05 else ""), flush=True)
        else:
            print(f"{points:>9} {'-':>9} {new[0]:>9} {'-':>12} {new[1]:>9.2f} ms {'-':>11} {new[2]:>9.3f} s {'-':>8}", flush=True)


if __name__ == "__main__":
    main()
