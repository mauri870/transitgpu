"""The throughput that the GPU attains: fused multiply-adds in FP32 and FP64 (TFLOPS), a copy in memory (GB/s) and the shader clock sampled while the kernels run.

    uv run --group rocm python benchmarks/peak.py [--json PEAK.json] [--spec-bandwidth GB_PER_S]

Each kernel runs eight independent chains of fused multiply-adds per thread, counted as 2 FLOPs each. The figures are what these kernels reach, a lower bound on the peak of the device; run it with nothing else on the device.
"""
import argparse
import json
import os
import subprocess
import threading
import time

import torch
import triton
import triton.language as tl


@triton.jit
def fma_chains(x_ptr, out_ptr, iters, DT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets).to(DT)
    a0, a1, a2, a3, a4, a5, a6, a7 = x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7
    m = (x * 0 + 0.999).to(DT)
    c = (x * 0 + 0.001).to(DT)
    for _ in range(iters):  # eight independent chains of fused multiply-adds per thread
        a0 = a0 * m + c
        a1 = a1 * m + c
        a2 = a2 * m + c
        a3 = a3 * m + c
        a4 = a4 * m + c
        a5 = a5 * m + c
        a6 = a6 * m + c
        a7 = a7 * m + c
    tl.store(out_ptr + offsets, a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)


def tflops(flops, seconds):
    """TFLOPS of a number of floating-point operations done in a number of seconds."""
    return flops / seconds / 1e12


def gigabytes_per_second(bytes_moved, seconds):
    """GB/s (10^9 bytes) of a number of bytes moved in a number of seconds."""
    return bytes_moved / seconds / 1e9


TARGET_SECONDS = 0.25


def iterations_for(iterations, seconds, target=TARGET_SECONDS):
    """The number of iterations that makes a kernel that took `seconds` for `iterations` last about `target` seconds, never fewer than `iterations`."""
    return max(iterations, int(iterations * target / seconds))


def device_holders(proc="/proc", own=None):
    """The ids of the other processes that hold the compute device (/dev/kfd) open, read from `proc`; the process `own` (this one by default) is left out."""
    own = os.getpid() if own is None else own
    found = []
    for entry in os.listdir(proc):
        if not entry.isdigit() or int(entry) == own:
            continue
        try:
            if any(os.readlink(os.path.join(proc, entry, "fd", fd)).endswith("/dev/kfd") for fd in os.listdir(os.path.join(proc, entry, "fd"))):
                found.append(int(entry))
        except OSError:
            continue
    return sorted(found)


def summary(device, runtime, fp32, fp64, copy, holders, spec_bandwidth=None):
    """The results as a dict: the attained TFLOPS of FP32 and FP64 (dicts of run()), their ratio, the copy of bandwidth() with its share of the specified bandwidth when one is given, the other processes that held the device, and the sampled clock range over both kernels."""
    out = {
        "device": device, "runtime": runtime, "fp32_tflops": fp32["tflops"], "fp64_tflops": fp64["tflops"], "fp32_over_fp64": fp32["tflops"] / fp64["tflops"],
        "copy_gb_per_s": copy["gb_per_s"], "clock_mhz": {"min": min(fp32["clock_min"], fp64["clock_min"]), "max": max(fp32["clock_max"], fp64["clock_max"])},
        "kernel_seconds": {"fp32": fp32["best_seconds"], "fp64": fp64["best_seconds"], "copy": copy["best_seconds"]}, "other_processes_on_the_device": holders,
    }
    if spec_bandwidth:
        out["spec_bandwidth_gb_per_s"] = spec_bandwidth
        out["copy_share_of_spec"] = copy["gb_per_s"] / spec_bandwidth
    return out


def clock_mhz():
    out = subprocess.run(["rocm-smi", "--showclocks"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "sclk" in line and "(" in line:
            return float(line.split("(")[1].split("Mhz")[0])
    return float("nan")


def run(dtype, name, threads=1 << 22, iters=2048, repeats=5):
    x = torch.rand(threads, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    block = 256
    grid = (threads // block,)
    DT = tl.float32 if dtype == torch.float32 else tl.float64
    fma_chains[grid](x, out, iters, DT=DT, BLOCK=block)  # compile
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fma_chains[grid](x, out, iters, DT=DT, BLOCK=block)
    end.record()
    end.synchronize()
    iters = iterations_for(iters, start.elapsed_time(end) / 1000)  # each timed kernel lasts about TARGET_SECONDS, so that the clock of the device has settled and can be sampled
    clocks = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            clocks.append(clock_mhz())
            time.sleep(0.05)

    thread = threading.Thread(target=sample)
    thread.start()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fma_chains[grid](x, out, iters, DT=DT, BLOCK=block)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    stop.set()
    thread.join()
    flops = 2 * 8 * iters * threads
    best = min(times)
    print(f"{name}: {tflops(flops, best):.2f} TFLOPS attained (best of {repeats}, {best * 1000:.1f} ms; median {sorted(times)[len(times) // 2] * 1000:.1f} ms), shader clock sampled {min(clocks):.0f}-{max(clocks):.0f} MHz, checksum {float(out.double().mean()):.4f}")
    return {"tflops": tflops(flops, best), "best_seconds": best, "median_seconds": sorted(times)[len(times) // 2], "iterations": iters, "clock_min": min(clocks), "clock_max": max(clocks)}


def bandwidth(gib=2, repeats=10):
    n = gib * 2**30 // 4
    a = torch.rand(n, device="cuda")
    b = torch.empty_like(a)
    b.copy_(a)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        b.copy_(a)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    rate = gigabytes_per_second(2 * n * 4, min(times))
    print(f"copy of {gib} GiB: {rate:.0f} GB/s attained (read plus write, best of {repeats})")
    return {"gb_per_s": rate, "best_seconds": min(times)}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", type=argparse.FileType("w"), help="write the results to this file")
    parser.add_argument("--spec-bandwidth", type=float, help="the memory bandwidth in GB/s that the manufacturer lists, to give the copy as a share of it")
    options = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: the throughput of a device is measured on a device")
    holders = device_holders()
    if holders:
        print(f"warning: other processes hold the device: {holders}")
    name = torch.cuda.get_device_name(0)
    print(name, torch.version.hip)
    fp32 = run(torch.float32, "FP32 FMA")
    fp64 = run(torch.float64, "FP64 FMA")
    print(f"FP64/FP32 attained ratio 1:{fp32['tflops'] / fp64['tflops']:.1f}")
    copy = bandwidth()
    holders = sorted(set(holders) | set(device_holders()))
    if options.json:
        json.dump(summary(name, torch.version.hip, fp32, fp64, copy, holders, options.spec_bandwidth), options.json, indent=1)
        options.json.close()


if __name__ == "__main__":
    main()
