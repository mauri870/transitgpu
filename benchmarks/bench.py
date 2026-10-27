"""Time the box search of transitgpu against the same code on a CPU, astropy's BoxLeastSquares and Transit Least Squares on synthetic light curves, and print a table.

    uv run python benchmarks/bench.py [--sizes 20000 100000 200000] [--rows gpu cpu1 cpu16 astropy tls1 tls16 tls1-exact tls16-exact tls-gpu tls-gpu-binned tls-gpu-pl] [--repeats 3] [--max-run-seconds 300]

The light curves are made here with a fixed seed: N points evenly spaced over a baseline of 1000 days (default), flat flux of 1 with Gaussian noise of 1000 ppm and one injected box transit (period 3.18 d, depth 3000 ppm,
duration 2.5 h), so that anyone can reproduce them. Every search uses the period grid of TLS (Ofir 2014) from 0.5 to 14 days for that baseline; transitgpu and TLS build it the same way and astropy's BoxLeastSquares is given
the same periods. The durations differ by design: transitgpu and TLS take theirs from the stellar limits of TLS's defaults, astropy's from one list of durations (ratio 1.1) between the shortest and the longest of those.
A row is timed after one run on 4,000 points that compiles and warms it, as the median of --repeats full runs; a row that a run of would take over --max-run-seconds, by linear scaling in the points from its time at the size before, is not run and is reported as that estimate. Rows whose package is not installed, or
whose device is missing, are skipped and said. `--spread-table JSON` prints, for every row of a --json file, the number of timed calls, the median, the quartiles, the minimum, the maximum, the first timed call and the build of the row (its warm-up call: for the first row of a process it
includes the compilation, so run one GPU row alone, with --rows and --repeats 30, to measure the cold call). TLS reads TLS_BACKEND for its backend; the rows set it. GTLS and cuvarbase need an NVIDIA device and are not run.
`--gtls-input DAYS` times the row set of the GTLS paper's timing instead: one light curve of DAYS days at 30-minute cadence with TLS's grid from 0 to half the span, the rows tls16-exact and tls-gpu by default, so that
TLS 2.0.1 and exact mode are timed on GTLS's input beside its published numbers; `--t0-fit-margin` gives both the T0 fit margin (TLS's default 0.01, GTLS's 0.125) and `--noise-ppm` the noise.
"""

import argparse
import glob
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
PERIOD_MIN, PERIOD_MAX = 0.5, 14.0
PLANET = {"period": 3.18, "t0": 1.2, "duration_hr": 2.5, "depth_ppm": 3000.0}
T0_FIT_MARGIN = 0.01  # TLS's default for the fraction of the duration between trial epochs; GTLS's source sets 0.125 (its constants.py, T0_FIT_MARGIN), and --t0-fit-margin gives the rows either
GTLS_CADENCE_MINUTES = 30.0  # the cadence of the synthetic light curves of the GTLS paper (Hu et al. 2026, Sect. 2.2.1): Kepler-like long cadence; 1,500 d give 72,000 points and 3,000 d 144,000
GTLS_ROWS = ("tls16-exact", "tls-gpu")  # the row set of --gtls-input: TLS 2.0.1's exact backend on 16 threads, and exact mode on the device
NOISE_PPM = 1000.0
SEED = 20260101
ROWS = ("gpu", "cpu1", "cpu16", "astropy", "tls1", "tls16", "tls1-exact", "tls16-exact", "tls-gpu", "tls-gpu-binned", "tls-gpu-pl", "hybrid")
EXTRA_ROWS = ("gpu-shuffles", "many", "tls16-many", "gpu-bumps", "gpu-two")  # run only when named
BUILT = []  # the rows built so far in this process
THROUGHPUT_CURVES = 64  # the light curves of the rows many and tls16-many


def inject(time_axis, flux):
    """The box transit of the synthetic input put into a given light curve, with the same model: period 3.18 d, depth 3,000 ppm, duration 2.5 h, the first transit 1.2 d after the first time. Returns a new flux."""
    flux = np.array(flux, dtype=float)
    phase = (time_axis - (time_axis[0] + PLANET["t0"]) + PLANET["period"] / 2) % PLANET["period"] - PLANET["period"] / 2
    flux[np.abs(phase) < PLANET["duration_hr"] / 48] -= PLANET["depth_ppm"] * 1e-6
    return flux


def quartiles(seconds):
    """The first and third quartiles of the timed runs, by linear interpolation between the sorted values (numpy's default)."""
    first, third = np.percentile(seconds, [25, 75])
    return float(first), float(third)


def light_curves(points, baseline, count):
    """One time axis and `count` fluxes of it: the light curve of the benchmark with the noise of seed SEED + k in the k-th (the first is light_curve itself) and the same injected box in all."""
    time_axis, first = light_curve(points, baseline)
    noise = [np.random.default_rng(SEED + k).normal(0.0, NOISE_PPM * 1e-6, points) for k in range(count)]
    return time_axis, [first + (extra - noise[0]) for extra in noise]


def light_curve(points, baseline):
    """N evenly spaced times over the baseline (days), flux 1 with Gaussian noise of NOISE_PPM (--noise-ppm) and one box transit."""
    time_axis = np.linspace(0.0, baseline, points)
    flux = 1.0 + np.random.default_rng(SEED).normal(0.0, NOISE_PPM * 1e-6, points)
    phase = (time_axis - PLANET["t0"] + PLANET["period"] / 2) % PLANET["period"] - PLANET["period"] / 2
    flux[np.abs(phase) < PLANET["duration_hr"] / 48] -= PLANET["depth_ppm"] * 1e-6
    return time_axis, flux


def versions():
    found = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "astropy", "transitleastsquares", "numba"):
        try:
            found[name] = __import__(name).__version__
        except ImportError:
            found[name] = None
    try:
        found["triton"] = __import__("triton").__version__  # the distribution is triton-rocm on ROCm and triton on CUDA
    except ImportError:
        pass
    for name, distribution in (("batman", "batman-package"), ("wotan", "wotan")):  # batman's module reports 2.5.1 for the distribution 2.5.3
        try:
            found[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def hardware():
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
    info = {"cpu": cpu, "logical_cpus": os.cpu_count(), "gpu": None, "gpu_runtime": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_runtime"] = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
    except ImportError:
        pass
    return info


def other_gpu_jobs(proc="/proc", own=None):
    """The process ids, other than this one and its descendants (the workers of a row), that hold the AMD compute device (/dev/kfd) open; a render node is held by every desktop program and is not a compute job."""
    own = os.getpid() if own is None else own

    def descends_from_own(pid):
        for _ in range(64):
            try:
                parent = int(open(os.path.join(proc, str(pid), "stat")).read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                return False
            if parent == own:
                return True
            if parent <= 1:
                return False
            pid = parent
        return False

    found = []
    for path in glob.glob(os.path.join(proc, "[0-9]*", "fd", "*")):
        try:
            target = os.readlink(path)
        except OSError:
            continue
        pid = int(path.split(os.sep)[-3])
        if pid != own and target == "/dev/kfd" and pid not in found and not descends_from_own(pid):
            found.append(pid)
    return sorted(found)


def gpu_busy_file():
    """The gpu_busy_percent file of the amdgpu card with the most memory, or None."""
    cards = glob.glob("/sys/class/drm/card[0-9]*/device/gpu_busy_percent")

    def memory(path):
        try:
            return int(open(os.path.join(os.path.dirname(path), "mem_info_vram_total")).read())
        except (OSError, ValueError):
            return 0

    return max(cards, key=memory, default=None)


class GpuBusy:
    """Reads the busy percentage of the device every `interval` seconds while a block runs: samples, mean and largest."""

    def __init__(self, path, interval=0.2):
        self.path, self.interval, self.values = path, interval, []

    def _read(self):
        try:
            self.values.append(float(open(self.path).read()))
        except (OSError, ValueError, TypeError):
            pass

    def __enter__(self):
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                self._read()
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exception):
        self._stop.set()
        self._thread.join()

    @property
    def samples(self):
        return len(self.values)

    @property
    def mean(self):
        return statistics.fmean(self.values) if self.values else None

    @property
    def largest(self):
        return max(self.values) if self.values else None


def timed(function, repeats):
    """Seconds of each of `repeats` runs, and the result of the first (the builders have already compiled and warmed the row on 4,000 points)."""
    seconds, first = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        found = function()
        seconds.append(time.perf_counter() - start)
        first = found if first is None else first
    return seconds, first


def duration_steps(periods):
    """The duration steps transitgpu scores for each period: from the shortest to the longest duration that TLS allows for it, in ratios of 1.1, as the search counts them."""
    from transitgpu.core import DURATION_STEP, duration_fractions

    shortest, longest = duration_fractions(periods)
    return np.maximum(1, np.ceil(np.log(longest / shortest) / np.log(DURATION_STEP)).astype(int) + 1)


def row_gpu(time_axis, flux, periods, device, threads):
    import torch

    from transitgpu import core

    if device == "cuda" and not torch.cuda.is_available():
        return None
    torch.set_num_threads(threads or torch.get_num_threads())
    warm = slice(0, 4000)
    core.search(time_axis[warm], flux[warm], PERIOD_MIN, PERIOD_MAX, device=device)
    steps = duration_steps(periods)
    return Row(lambda: core.search(time_axis, flux, PERIOD_MIN, PERIOD_MAX, periods=periods, device=device), lambda found: found["period_days"], len(periods),
               f"{steps.mean():.1f} steps per period ({steps.sum()} in all), the limits of TLS's defaults, ratio 1.1")


def row_bumps(time_axis, flux, periods, both):
    """The dips and the bumps of the light curve: one `search` with `bumps=True` (`both`), or two separate searches, the flux and the negated flux, which find the same dips and the same bumps from two folds."""
    import torch

    from transitgpu import core

    if not torch.cuda.is_available():
        return None
    warm = slice(0, 4000)
    core.search(time_axis[warm], flux[warm], PERIOD_MIN, PERIOD_MAX, device="cuda", bumps=both)
    if both:
        return Row(lambda: core.search(time_axis, flux, PERIOD_MIN, PERIOD_MAX, periods=periods, device="cuda", bumps=True), lambda found: found["period_days"], len(periods),
                   "one search of the dips and the bumps, from one fold and one set of bin sums")
    core.search(time_axis[warm], -flux[warm], PERIOD_MIN, PERIOD_MAX, device="cuda")

    def two():
        found = core.search(time_axis, flux, PERIOD_MIN, PERIOD_MAX, periods=periods, device="cuda")
        core.search(time_axis, -flux, PERIOD_MIN, PERIOD_MAX, periods=periods, device="cuda")
        return found

    return Row(two, lambda found: found["period_days"], len(periods), "two separate searches, of the flux and of the negated flux")


def row_astropy(time_axis, flux, periods, _device, _threads):
    """astropy takes one list of durations for all periods: the geometric grid of ratio 1.1 from the shortest to the longest duration that TLS allows over the periods, cut where a duration is not under the shortest period."""
    try:
        from astropy.timeseries import BoxLeastSquares
    except ImportError:
        return None
    from transitgpu.core import DURATION_STEP, duration_fractions

    shortest, longest = duration_fractions(periods)
    low, high = float((shortest * periods).min()), float((longest * periods).max())
    durations = low * DURATION_STEP ** np.arange(int(np.ceil(np.log(high / low) / np.log(DURATION_STEP))) + 1)
    durations = durations[durations < periods.min()]
    model = BoxLeastSquares(time_axis, flux, dy=np.full(flux.size, NOISE_PPM * 1e-6))
    return Row(lambda: model.power(periods, durations, objective="snr"), lambda found: float(periods[int(np.argmax(found.power))]), len(periods),
               f"{durations.size} durations for every period ({durations.size * len(periods)} in all), {durations.min() * 24:.2f} to {durations.max() * 24:.2f} h, ratio 1.1; one thread, as astropy runs")


def row_tls(time_axis, flux, periods, backend, threads):
    """TLS builds its own period grid from the same function (same stellar defaults, oversampling and transit minimum), so its trial periods are the shared grid, and its own duration grid, which it limits for each period."""
    try:
        from transitleastsquares import transitleastsquares
        from transitleastsquares.grid import duration_grid
    except ImportError:
        return None

    def run():
        os.environ["TLS_BACKEND"] = backend
        return transitleastsquares(time_axis, flux).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, use_threads=threads, show_progress_bar=False, verbose=False, T0_fit_margin=T0_FIT_MARGIN)

    os.environ["TLS_BACKEND"] = backend
    transitleastsquares(time_axis[:4000], flux[:4000]).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, use_threads=threads, show_progress_bar=False, verbose=False, T0_fit_margin=T0_FIT_MARGIN)
    grid = duration_grid(periods, 0.0)
    return Row(run, lambda found: float(found.period), len(periods),
               f"TLS's own grid of {len(grid)} durations (ratio 1.1, from {min(grid):.4f} to {max(grid):.4f} of the period), limited for each period to the stellar limits; its limb-darkened template, not a box; T0 fit margin {T0_FIT_MARGIN:g}")


def row_transitgpu_tls(time_axis, flux, periods, backend):
    """TLS's own search of the trial periods with a backend of transitgpu.tls on the device ("gpu", exact; "gpu-binned" and "gpu-pl", approximate): the same grids, the same final fit and statistics as TLS, only the search of the trial periods differs. Without a device the row is skipped: transitgpu.tls has no search on a CPU."""
    try:
        import torch
        from transitleastsquares.grid import duration_grid

        from transitgpu.tls import transitleastsquares
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    def run(t=time_axis, y=flux):
        return transitleastsquares(t, y).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, show_progress_bar=False, verbose=False, backend=backend, T0_fit_margin=T0_FIT_MARGIN)

    run(time_axis[:4000], flux[:4000])
    grid = duration_grid(periods, 0.0)
    how = {"gpu": "the exact statistic of TLS", "gpu-binned": "TLS's statistic with windows of fixed width in phase, on phase bins", "gpu-pl": "the piecewise-linear statistic of TLS's default backend (fused-pl)"}[backend]
    return Row(run, lambda found: float(found.period), len(periods), f"TLS's own grid of {len(grid)} durations (ratio 1.1), limited for each period to the stellar limits; its limb-darkened template; {how}, on the device; T0 fit margin {T0_FIT_MARGIN:g}")


def row_hybrid(time_axis, flux, periods):
    """transitgpu.bls_tls_hybrid: the box search over the whole grid, then TLS's exact search on a window of 1% around each of the 5 highest peaks. The trial periods counted are those of the box search, and the period read is that of the best peak."""
    try:
        import torch
        from transitleastsquares.grid import duration_grid

        import transitgpu
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    def run(t=time_axis, y=flux):
        return transitgpu.bls_tls_hybrid(t, y, None, period_min=PERIOD_MIN, period_max=PERIOD_MAX, peaks=5, window=0.01, backend="gpu")

    run(time_axis[:4000], flux[:4000])
    grid = duration_grid(periods, 0.0)
    return Row(run, lambda found: float(found.peaks[0].tls.period if found.peaks[0].tls is not None else found.peaks[0].box_period), len(periods),
               f"the box search over the grid, then TLS's own grid of {len(grid)} durations and its template on a window of 1% around each of 5 peaks, exact backend, on the device")


def found_everywhere(periods):
    """Whether each period found is the injected one within 1%."""
    return all(abs(float(period) / PLANET["period"] - 1) < 0.01 for period in periods)


def row_many(time_axis, periods):
    """transitgpu.search_many over THROUGHPUT_CURVES light curves of the same time axis and period grid: the call that searches a set on the device. Timed as the whole set."""
    try:
        import torch

        from transitgpu import search_many
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    _, fluxes = light_curves(len(time_axis), float(time_axis[-1] - time_axis[0]), THROUGHPUT_CURVES)
    search_many(time_axis[:4000], [flux[:4000] for flux in fluxes[:2]], PERIOD_MIN, PERIOD_MAX)
    return Row(lambda: search_many(time_axis, fluxes, PERIOD_MIN, PERIOD_MAX, periods=periods), lambda found: found[0]["period_days"], THROUGHPUT_CURVES * len(periods),
               f"the box search of {THROUGHPUT_CURVES} light curves of {len(time_axis):,} points in one call, the durations of the box row", check=lambda found: found_everywhere(entry["period_days"] for entry in found))


def row_tls_many(time_axis, periods, threads=16, backend="fused-pl"):
    """TLS 2.0.1 on the same set, one light curve after another, `threads` threads and the default backend: the loop that a user of TLS writes."""
    try:
        from transitleastsquares import transitleastsquares
    except ImportError:
        return None
    _, fluxes = light_curves(len(time_axis), float(time_axis[-1] - time_axis[0]), THROUGHPUT_CURVES)

    def run():
        os.environ["TLS_BACKEND"] = backend
        return [transitleastsquares(time_axis, flux).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, use_threads=threads, show_progress_bar=False, verbose=False).period for flux in fluxes]

    os.environ["TLS_BACKEND"] = backend
    transitleastsquares(time_axis[:4000], fluxes[0][:4000]).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, use_threads=threads, show_progress_bar=False, verbose=False)
    return Row(run, lambda found: float(found[0]), THROUGHPUT_CURVES * len(periods), f"TLS's own grid of durations, its template; {THROUGHPUT_CURVES} power() calls one after another", check=found_everywhere)


class Row:
    """A timed search: the function to time, how to read the period it found, the trial periods it evaluates and a statement of the durations it evaluates."""

    def __init__(self, function, answer, trial_periods, durations, check=None):
        self.function, self.answer, self.trial_periods, self.durations, self.check = function, answer, trial_periods, durations, check  # check(found): whether every light curve of a set gave the injected period


def block_shuffles(time_axis, flux, count, baseline):
    """`count` copies of the flux with its blocks of one day (as many points as a day holds on the even cadence) in another order each, from fixed seeds: the null of a star, as the thresholds of a search are made. The times stay."""
    block = max(1, int(round(len(flux) / baseline)))
    blocks = len(flux) // block
    copies = []
    for seed in range(count):
        order = np.random.default_rng(SEED + seed).permutation(blocks)
        body = np.concatenate([flux[i * block:(i + 1) * block] for i in order])
        copies.append(np.concatenate([body, flux[blocks * block:]]))
    return copies


def row_shuffles(time_axis, flux, periods, count, baseline):
    """The search of `count` block-shuffled copies of the light curve in one call of search_many, which folds the times once: the time of a threshold of a star (the largest SDE of its shuffles). The answer of the row is that largest SDE."""
    from transitgpu import core

    copies = block_shuffles(time_axis, flux, count, baseline)
    core.search_many(time_axis[:4000], [c[:4000] for c in copies[:2]], PERIOD_MIN, PERIOD_MAX, device=None)  # compiles
    return Row(lambda: core.search_many(time_axis, copies, PERIOD_MIN, PERIOD_MAX, periods=periods), lambda found: max(each["sde"] for each in found), len(periods) * count,
               f"{count} shuffled copies of one light curve, 1-day blocks, searched together on the grid; the answer column is the largest SDE of the shuffles, not a period")


BUILDERS = {
    "gpu": lambda t, f, p: row_gpu(t, f, p, "cuda", None),
    "cpu1": lambda t, f, p: row_gpu(t, f, p, "cpu", 1),
    "cpu16": lambda t, f, p: row_gpu(t, f, p, "cpu", 16),
    "astropy": lambda t, f, p: row_astropy(t, f, p, None, None),
    "tls1": lambda t, f, p: row_tls(t, f, p, "fused-pl", 1),
    "tls16": lambda t, f, p: row_tls(t, f, p, "fused-pl", 16),
    "tls1-exact": lambda t, f, p: row_tls(t, f, p, "fused", 1),
    "tls16-exact": lambda t, f, p: row_tls(t, f, p, "fused", 16),
    "tls-gpu": lambda t, f, p: row_transitgpu_tls(t, f, p, "gpu"),
    "tls-gpu-binned": lambda t, f, p: row_transitgpu_tls(t, f, p, "gpu-binned"),
    "tls-gpu-pl": lambda t, f, p: row_transitgpu_tls(t, f, p, "gpu-pl"),
    "hybrid": lambda t, f, p: row_hybrid(t, f, p),
    "gpu-shuffles": lambda t, f, p: row_shuffles(t, f, p, SHUFFLES, BASELINE),
    "many": lambda t, f, p: row_many(t, p),
    "tls16-many": lambda t, f, p: row_tls_many(t, p),
    "gpu-bumps": lambda t, f, p: row_bumps(t, f, p, True),
    "gpu-two": lambda t, f, p: row_bumps(t, f, p, False),
}
LABELS = {
    "gpu-shuffles": "transitgpu on the GPU, the block shuffles of one light curve",
    "gpu": "transitgpu on the GPU", "cpu1": "transitgpu on the CPU, 1 thread", "cpu16": "transitgpu on the CPU, 16 threads", "astropy": "astropy BoxLeastSquares, 1 thread, snr objective",
    "tls1": "TLS, default backend (fused-pl), 1 thread", "tls16": "TLS, default backend (fused-pl), 16 threads", "tls1-exact": "TLS, exact backend (fused), 1 thread", "tls16-exact": "TLS, exact backend (fused), 16 threads",
    "tls-gpu": "transitgpu.tls exact (gpu), on the GPU", "tls-gpu-binned": "transitgpu.tls binned (gpu-binned), on the GPU", "tls-gpu-pl": "transitgpu.tls piecewise-linear (gpu-pl), on the GPU",
    "hybrid": "transitgpu.bls_tls_hybrid (GPU), 5 peaks, window 1%, exact backend",
    "gpu-bumps": "transitgpu on the GPU, dips and bumps in one search", "gpu-two": "transitgpu on the GPU, two separate searches (flux and negated flux)",
    "many": "transitgpu.search_many on the GPU, 64 light curves in one call", "tls16-many": "TLS, default backend (fused-pl), 16 threads, 64 light curves one after another",
}


def command_line():
    """The command that ran, as `uv run python` with the script as it was named and any path of the --json file reduced to its name, so that the file carries no directory of the machine."""
    words = ["uv", "run", "python", *sys.argv]
    return " ".join(os.path.basename(word) if previous == "--json" else word for previous, word in zip(["", *words], words))


def merged(reports):
    """The reports of several runs of this script as one: the rows of each size joined, in the order of the reports; the hardware, the versions and the inputs are those of the first and must be the same."""
    first = json.loads(json.dumps(reports[0]))
    for other in reports[1:]:
        for key in ("hardware", "versions", "seed", "baseline_days", "period_range_days", "planet", "noise_ppm"):
            if other[key] != first[key]:
                raise ValueError(f"the reports differ in {key}: {first[key]} and {other[key]}")
        for points, entry in other["sizes"].items():
            first["sizes"].setdefault(points, {"periods": entry["periods"], "rows": {}})["rows"].update(entry["rows"])
    return first


def table(report):
    """The measurements of a --json file as markdown tables, one for each size, with the median and quartiles of the timed runs, the trial periods, the seconds per trial period, the period each row found and the durations it evaluated."""
    lines = []
    for points, entry in report["sizes"].items():
        lines += [f"{int(points):,} points over {report['baseline_days']:g} days, {entry['periods']:,} trial periods", "",
                  "| search | seconds (median, quartiles) | ms per trial period | period found (d) | durations evaluated |", "| --- | --- | --- | --- | --- |"]
        for name, row in entry["rows"].items():
            if "estimated_seconds" in row:
                lines.append(f"| {LABELS[name]} | about {row['estimated_seconds']:.0f}, not run: linear scaling from {row['measured_at_points']:,} points | | | |")
            else:
                seconds = row["seconds"]
                lines.append(f"| {LABELS[name]} | {row['median']:.3g} ({quartiles(seconds)[0]:.3g} to {quartiles(seconds)[1]:.3g}, quartiles) | {1e3 * row['seconds_per_trial_period']:.3g} | {row['period_found_days']:.5f} | {row['durations']} |")
        lines.append("")
    return "\n".join(lines)


def spread_table(report):
    """The spread of the timed calls of every row as markdown, one table for each size: the number of calls, the median, the quartiles (linear interpolation), the minimum and the maximum, the first timed call and the build of the row, which is its warm-up call and, for the
    first row of a process, includes the compilation of its kernels and the making of the device context (run a row alone in a process to measure that cold call)."""
    lines = []
    for points, entry in report["sizes"].items():
        lines += [f"{int(points):,} points over {report['baseline_days']:g} days, {entry['periods']:,} trial periods", "",
                  "| search | calls | median | quartiles | min | max | first timed call | build of the row (cold when first of its process) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, row in entry["rows"].items():
            if "estimated_seconds" in row:
                lines.append(f"| {LABELS[name]} | not run: linear scaling from {row['measured_at_points']:,} points, about {row['estimated_seconds']:.0f} s | | | | | | |")
                continue
            seconds, (first, third) = row["seconds"], quartiles(row["seconds"])
            cold = "" if "build_seconds" not in row else f"{row['build_seconds']:.3g}{' (first of its process)' if row.get('first_row_of_process') else ''}"
            lines.append(f"| {LABELS[name]} | {len(seconds)} | {row['median']:.3g} | {first:.3g} to {third:.3g} | {min(seconds):.3g} | {max(seconds):.3g} | {seconds[0]:.3g} | {cold} |")
        lines.append("")
    return "\n".join(lines)


def readme_table(report, rows=None):
    """The table of the README: one row for each search (of `rows`, when given), the median of the timed runs of each size in seconds with its range, to three figures; an empty cell where the row was not run at that size."""
    sizes = list(report["sizes"])
    lines = ["| search | " + " | ".join(f"{int(points):,} points" for points in sizes) + " |", "| --- |" + " --- |" * len(sizes)]
    for name in (rows or ROWS):
        cells = []
        for points in sizes:
            row = report["sizes"][points]["rows"].get(name)
            cells.append("" if row is None or "median" not in row else f"{row['median']:.3g} ({quartiles(row['seconds'])[0]:.3g}–{quartiles(row['seconds'])[1]:.3g})")
        if any(cells):
            lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def three_figures(value):
    """A time in seconds with three significant figures, with its trailing zeros (15.0, 0.277, 292)."""
    digits = max(0, 2 - math.floor(math.log10(abs(float(f"{value:.3g}")))))
    return f"{float(f'{value:.3g}'):.{digits}f}"


# the rows of the table of the README: label, device, the block of results.json and the row in it
PERFORMANCE_ROWS = (
    ("transitgpu box search", "GPU", "sizes", "gpu"),
    ("transitgpu.bls_tls_hybrid, 5 peaks, window 1%, exact", "GPU", "sizes", "hybrid"),
    ("transitgpu.tls exact", "GPU", "sizes", "tls-gpu"),
    ("transitgpu.tls binned", "GPU", "sizes", "tls-gpu-binned"),
    ("transitgpu.tls piecewise-linear", "GPU", "sizes", "tls-gpu-pl"),
    ("astropy BoxLeastSquares", "CPU, 1 thread", "sizes", "astropy"),
    ("transitleastsquares 2.0.1 exact (fused)", "CPU, 16 threads", "sizes", "tls16-exact"),
    ("transitleastsquares 2.0.1 default (fused-pl)", "CPU, 16 threads", "sizes", "tls16"),
    ("transitleastsquares 2.0.1 exact (fused)", "CPU, 1 thread", "sizes", "tls1-exact"),
    ("transitleastsquares 2.0.1 default (fused-pl)", "CPU, 1 thread", "sizes", "tls1"),
)


def performance_table(results):
    """The one table of the README from the block sizes of results.json: the median of each size with its quartiles, three significant figures in every cell."""
    sizes = list(results["sizes"])
    lines = ["| search | device | " + " | ".join(f"{int(points):,} points" for points in sizes) + " |", "| --- | --- |" + " --- |" * len(sizes)]
    for label, device, block, name in PERFORMANCE_ROWS:
        cells = []
        for points in sizes:
            row = (results if block == "sizes" else results[block])["sizes"][points]["rows"][name]
            cells.append(f"{three_figures(row['median'])} ({three_figures(quartiles(row['seconds'])[0])}–{three_figures(quartiles(row['seconds'])[1])})")
        lines.append(f"| {label} | {device} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


THROUGHPUT_ROWS = (("transitgpu.search_many", "GPU", "many"), ("transitleastsquares 2.0.1 default (fused-pl), one after another", "CPU, 16 threads", "tls16-many"))


def throughput_table(results):
    """The table of the README for the set of light curves: the median seconds of the whole set with its quartiles, and the light curves a second at the median."""
    entry = results["sizes"]["20000"]["rows"]
    lines = ["| search | device | seconds for the set | light curves per second |", "| --- | --- | --- | --- |"]
    for label, device, name in THROUGHPUT_ROWS:
        row = entry[name]
        first, third = quartiles(row["seconds"])
        lines.append(f"| {label} | {device} | {three_figures(row['median'])} ({three_figures(first)}–{three_figures(third)}) | {three_figures(THROUGHPUT_CURVES / row['median'])} |")
    return "\n".join(lines)


BUMPS_ROWS = (("`transitgpu.search`, dips only", "gpu"), ("`transitgpu.search` with `bumps=True`, dips and bumps in one search", "gpu-bumps"), ("two `transitgpu.search` calls, of the flux and of the negated flux", "gpu-two"))


def bumps_table(results):
    """The table of the README for the dips and the bumps, from its own results file (one session): the median of each size with its quartiles, three significant figures in every cell."""
    sizes = list(results["sizes"])
    lines = ["| search of one light curve | " + " | ".join(f"{int(points):,} points" for points in sizes) + " |", "| --- |" + " --- |" * len(sizes)]
    for label, name in BUMPS_ROWS:
        cells = []
        for points in sizes:
            row = results["sizes"][points]["rows"][name]
            cells.append(f"{three_figures(row['median'])} ({three_figures(quartiles(row['seconds'])[0])}–{three_figures(quartiles(row['seconds'])[1])})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[20000, 100000, 200000], help="numbers of points")
    parser.add_argument("--inject", action="store_true", help="with --npz, put the box transit of the synthetic input (3.18 d, 3,000 ppm, 2.5 h) into the light curve first, so that every row has a known answer")
    parser.add_argument("--npz", help="time the rows on the light curve of this .npz file (arrays time and flux, flux normalised) instead of the synthetic ones; the period range and the rows are as given")
    parser.add_argument("--baseline", type=float, default=1000.0, help="days spanned by the points")
    parser.add_argument("--period-max", type=float, default=14.0, help="the longest trial period searched, in days (TLS's grid from 0.5 d to this); half the baseline gives the grid of the benchmark of the GTLS paper")
    parser.add_argument("--gtls-input", type=float, metavar="DAYS", help="time the rows on the input of the GTLS paper's timing (Hu et al. 2026, Sects. 2.2.1 and 3.4; its source at 74e449c): one light curve of DAYS days at 30-minute cadence (1500 gives 72,000 points, 3000 gives 144,000), TLS's grid from 0 to half the span (solar star, oversampling 3, 2 transits), the box and noise of this benchmark unless --noise-ppm says otherwise; the rows default to the GTLS set (tls16-exact and tls-gpu) and --sizes and --baseline are ignored")
    parser.add_argument("--t0-fit-margin", type=float, help="TLS's T0_fit_margin given to the TLS rows and to exact mode: the fraction of the duration between trial epochs of the search and of the final fit (TLS's default 0.01; GTLS's source sets 0.125, which costs it less)")
    parser.add_argument("--noise-ppm", type=float, help="the Gaussian noise of the synthetic light curve in ppm (default 1000; the GTLS paper's recovery tests use 110 and its timing section does not say)")
    parser.add_argument("--shuffles", type=int, default=48, help="the number of block-shuffled copies of the light curve that the row gpu-shuffles searches together")
    parser.add_argument("--rows", nargs="+", choices=ROWS + EXTRA_ROWS, default=list(ROWS))
    parser.add_argument("--repeats", type=int, default=3, help="timed runs of each row")
    parser.add_argument("--max-run-seconds", type=float, default=300.0, help="a row whose run at this size is estimated to take longer, by linear scaling in points from its median at the size before, is not run and is reported as the estimate")
    parser.add_argument("--refuse-if-busy", action="store_true", help="stop before a row if another process holds the compute device (/dev/kfd) open")
    parser.add_argument("--json", help="write the measurements to this file")
    parser.add_argument("--table", nargs="+", metavar="JSON", help="print the markdown tables of one or more --json files (the rows of the same size are joined) and stop")
    parser.add_argument("--spread-table", nargs="+", metavar="JSON", help="print the table of the spread of the timed calls (median, quartiles, minimum, maximum, first call, build of the row) of one or more --json files and stop")
    parser.add_argument("--readme-table", nargs="+", metavar="JSON", help="print the table of the README (the median and range in seconds of every row at every size) from one or more --json files and stop")
    options = parser.parse_args(arguments)
    global PERIOD_MIN, PERIOD_MAX, SHUFFLES, BASELINE, T0_FIT_MARGIN, NOISE_PPM
    BUILT.clear()
    PERIOD_MIN, PERIOD_MAX, SHUFFLES = 0.5, options.period_max, options.shuffles
    T0_FIT_MARGIN = options.t0_fit_margin if options.t0_fit_margin is not None else 0.01  # TLS's default; the same margin steps the trial epochs of the search unless TLS is given T0_search_margin (its tls_constants.py)
    NOISE_PPM = options.noise_ppm if options.noise_ppm is not None else 1000.0
    if options.gtls_input:
        if options.npz:
            raise SystemExit("--gtls-input makes its own light curve; it cannot be given with --npz")
        PERIOD_MIN, PERIOD_MAX = 0.0, options.gtls_input / 2
        options.baseline, options.sizes = options.gtls_input, [int(round(options.gtls_input * 1440 / GTLS_CADENCE_MINUTES))]
        if "--rows" not in (arguments if arguments is not None else sys.argv):
            options.rows = list(GTLS_ROWS)
    if options.table:
        print(table(merged([json.load(open(path)) for path in options.table])))
        return None
    if options.spread_table:
        print(spread_table(merged([json.load(open(path)) for path in options.spread_table])))
        return None
    if options.readme_table:
        print(readme_table(merged([json.load(open(path)) for path in options.readme_table])))
        return None
    from transitgpu.core import grid_of

    report = {"command": command_line(), "hardware": hardware(), "versions": versions(), "seed": SEED, "baseline_days": options.baseline, "period_range_days": [PERIOD_MIN, PERIOD_MAX],
              "planet": PLANET, "noise_ppm": NOISE_PPM, "t0_fit_margin": T0_FIT_MARGIN, "sizes": {}}
    if options.gtls_input:
        report["input"] = {"gtls": True, "baseline_days": options.gtls_input, "cadence_minutes": GTLS_CADENCE_MINUTES, "points": options.sizes[0], "period_range_days": [PERIOD_MIN, PERIOD_MAX],
                           "grid": "TLS 2.0.1 (Ofir 2014): R_star 1, M_star 1, oversampling 3, at least 2 transits; durations in ratios of 1.1", "injected_box": PLANET, "noise_ppm": NOISE_PPM, "t0_fit_margin": T0_FIT_MARGIN,
                           "source": "Hu et al. 2026, Sects. 2.2.1 and 3.4; GTLS at 74e449c: constants.py (OVERSAMPLING_FACTOR, N_TRANSITS_MIN, R_STAR, M_STAR, DURATION_GRID_STEP, T0_FIT_MARGIN) and validate.py (the period maximum, half the span)"}
    print(json.dumps({key: report[key] for key in ("hardware", "versions")}))
    previous = None
    if options.npz:
        data = np.load(options.npz)
        time_in, flux_in = np.asarray(data["time"], dtype=float), np.asarray(data["flux"], dtype=float)
        if options.inject:
            flux_in = inject(time_in, flux_in)
        inputs = [(len(time_in), (time_in, flux_in))]
        report["input"] = {"file": os.path.basename(options.npz), "points": len(time_in), "baseline_days": float(np.ptp(time_in)), "injected_box": PLANET if options.inject else None}
        report["baseline_days"] = report["input"]["baseline_days"]
    else:
        inputs = [(points, None) for points in options.sizes]
    BASELINE = report["baseline_days"]
    for points, given in inputs:
        time_axis, flux = given if given is not None else light_curve(points, options.baseline)  # with --gtls-input, points at the cadence of the GTLS paper over its baseline
        periods = grid_of(time_axis, flux, PERIOD_MIN, PERIOD_MAX)
        entry = report["sizes"][str(points)] = {"periods": int(periods.size), "rows": {}}
        before = report["sizes"][str(previous)] if previous else None
        print(f"\n{points} points over {report['baseline_days']:g} days, {periods.size} trial periods")
        for name in options.rows:
            earlier = before["rows"].get(name) if before else None
            if earlier and earlier["seconds_per_point"] * points > options.max_run_seconds:
                estimate = earlier["seconds_per_point"] * points
                entry["rows"][name] = {"estimated_seconds": estimate, "seconds_per_point": earlier["seconds_per_point"], "measured_at_points": earlier.get("measured_at_points", previous)}
                print(f"  {LABELS[name]}: not run, estimated {estimate:.0f} s from {earlier['seconds_per_point'] * entry['rows'][name]['measured_at_points']:.1f} s at {entry['rows'][name]['measured_at_points']} points (linear in the points), over the {options.max_run_seconds:g} s a run may take")
                continue
            built_at = time.perf_counter()
            row = BUILDERS[name](time_axis, flux, periods)
            build_seconds = time.perf_counter() - built_at  # the warm-up call of the builder: it compiles when this row is the first of the process to use its kernels
            if row is None:
                print(f"  {LABELS[name]}: skipped (package or device missing)")
                continue
            if options.refuse_if_busy and other_gpu_jobs():
                raise SystemExit(f"{LABELS[name]}: not started, process {other_gpu_jobs()} holds the device")
            load_at_start = os.getloadavg()[0]  # one-minute load average before the row, the guard that the machine was quiet
            with GpuBusy(gpu_busy_file()) as busy:
                seconds, found = timed(row.function, options.repeats)
            median = statistics.median(seconds)
            entry["rows"][name] = {"seconds": seconds, "median": median, "build_seconds": build_seconds, "first_row_of_process": not BUILT, "period_found_days": float(row.answer(found)), "trial_periods": row.trial_periods, "seconds_per_trial_period": median / row.trial_periods, "durations": row.durations, "seconds_per_point": median / points, "measured_at_points": points, "load_at_start": load_at_start, "gpu_busy_mean_percent": busy.mean, "gpu_busy_largest_percent": busy.largest}
            BUILT.append(name)  # a later row of this process finds the kernels compiled and the device context made
            if row.check is not None:
                entry["rows"][name]["found_in_every_light_curve"] = bool(row.check(found))
            if options.json:  # after every row, so that a run that is stopped keeps the rows it finished
                with open(options.json, "w") as handle:
                    json.dump(report, handle, indent=1)
            print(f"  {LABELS[name]}: {median:.3f} s (median of {len(seconds)}; {min(seconds):.3f} to {max(seconds):.3f}); {1e3 * median / row.trial_periods:.4f} ms per trial period; found P = {row.answer(found):.5f} d; {row.durations}", flush=True)
        previous = points
    if options.json:
        with open(options.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return report


if __name__ == "__main__":
    main()
