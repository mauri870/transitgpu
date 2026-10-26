"""Where the device time of the exact search goes, by stage, at the sizes of the benchmark.

    taskset -c 0-15 python paper/scripts/profile_split.py --json paper/scripts/profile_split.json

Runs `transitleastsquares(t, y).power(backend="gpu")` of transitgpu.tls on the light curves of the benchmark under
the torch profiler. The device time of each kernel (summed over its launches) is assigned to a stage by its name:
fold (the phase kernels and the sorts), scans (the prefix sums and cumulative sums), screening (the incumbent and the
screening kernel), correlations (the dot products of the survivors), and the rest, listed. Each kernel is counted
once, from the device events and not the torch operators that launched it. Device time is not wall time: the host
launches while the device works, and the profiler slows the run.
"""
import argparse, importlib.util, json, time, warnings
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("transitgpu_bench", Path(__file__).resolve().parents[2] / "benchmarks" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
from transitgpu.tls import transitleastsquares  # noqa: E402

STAGES = (
    ("fold", ("_phase_kernel", "_bin_kernel", "_rank_kernel", "sort", "rocprim", "radix", "scatter", "index_", "argsort", "bucketize")),
    ("scans", ("_chunk_sums", "_chunk_scans", "cumsum", "scan")),
    ("screening", ("_keep_all_kernel", "_deepest_kernel")),
    ("correlations", ("_dots_kernel",)),
)


def stage_of(name):
    lowered = name.lower()
    for stage, words in STAGES:
        if any(word.lower() in lowered for word in words):
            return stage
    return "rest"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--points", type=int, nargs="+", default=[20000, 100000, 200000])
    options = parser.parse_args()
    report = []
    for points in options.points:
        t, y = bench.light_curve(points, 1000.0)
        run = lambda: transitleastsquares(t, y).power(period_min=0.5, period_max=14, show_progress_bar=False, verbose=False, backend="gpu")
        run()
        torch.cuda.synchronize()
        started = time.perf_counter()
        run()
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
            torch.cuda.synchronize()
        totals, names = {}, {}
        for event in prof.events():
            if event.device_type != torch.autograd.DeviceType.CUDA:  # the kernels and copies on the device; the operators of torch carry the time of their kernels again
                continue
            stage = stage_of(event.name)
            seconds = event.device_time_total / 1e6
            totals[stage] = totals.get(stage, 0.0) + seconds
            names[(stage, event.name[:60])] = names.get((stage, event.name[:60]), 0.0) + seconds
        names = {stage: sorted(((s, n) for (st, n), s in names.items() if st == stage), reverse=True) for stage in totals}
        busy = sum(totals.values())
        entry = dict(points=points, wall_seconds=wall, device_seconds=busy, stages=totals, rest=sorted(names.get("rest", []), reverse=True)[:5])
        report.append(entry)
        print(f"{points:7d} points: wall {wall:.2f} s, device busy {busy:.2f} s; " + ", ".join(f"{stage} {seconds:.2f} s ({100 * seconds / busy:.0f}%)" for stage, seconds in sorted(totals.items(), key=lambda kv: -kv[1])), flush=True)
        print("   rest:", "; ".join(f"{name} {seconds:.2f}" for seconds, name in entry["rest"]), flush=True)
    if options.json:
        Path(options.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
