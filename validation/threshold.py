"""The calibration of the block-shuffle threshold of paper/scripts/threshold_calibration.md.

    uv run python validation/threshold.py null CLASS COUNT [--out DIR] [--budget SECONDS]   noise-only light curves (CLASS white, red or variable)
    uv run python validation/threshold.py real [--out DIR] [--data DIR]     the real stars of stars.json, 199 shuffles each
    uv run python validation/threshold.py inj CLASS COUNT [--out DIR] [--budget SECONDS]    the same noise with the planet of each curve's cell

Rows are appended to DIR/threshold.jsonl (default validation/out), one case a line with its id; a case with a row is skipped before anything is generated.
"""
import argparse
import importlib.util
import json
import math
import time as clock
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ("white", "red", "variable")
SEED_BASE, SEED_STRIDE = 9_000_000, 100_000
SHUFFLES = 48
PERIOD_RANGE = (0.5, 14.0)
INJECTIONS = 1500  # the curves of a class that carry a planet (50 a cell); the null set is larger


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / ("paper/scripts" if name == "injection" else "validation") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_of(class_index, index):
    return SEED_BASE + SEED_STRIDE * class_index + index


def compare(best, shuffles):
    """'above' when the best SDE is strictly over the largest SDE of the shuffles, 'tie' when it equals it, 'below' otherwise."""
    largest = max(shuffles)
    return "above" if best > largest else "tie" if best == largest else "below"


def above_rank(best, shuffles, rank):
    """Whether the best SDE is strictly over the rank-th largest SDE of the shuffles (rank 1 is the largest)."""
    return bool(best > sorted(shuffles, reverse=True)[rank - 1])


def wilson(count, total, z=1.959964):
    """The Wilson score interval of count out of total (Wilson 1927), 95 percent by default."""
    p = count / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def false_alarm_quantile(quantile, shuffles=SHUFFLES):
    """The quantile of the false-alarm probability of a threshold that is the maximum of `shuffles` exchangeable draws: that probability is Beta(1, shuffles), whose mean is 1 / (shuffles + 1)."""
    return 1.0 - (1.0 - quantile) ** (1.0 / shuffles)


def invert(flux):
    """The flux reflected about its median."""
    flux = np.asarray(flux, dtype=float)
    return 2.0 * np.median(flux) - flux


def finished(path):
    """The ids of the cases with a row in the file."""
    path = Path(path)
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line} if path.exists() else set()


def run_cases(ids, path, measure):
    """Measure each case of ids that has no row in the file yet and append its row; a finished case is not generated."""
    done = finished(path)
    for case_id in ids:
        if case_id in done:
            continue
        row = measure(case_id)
        with open(path, "a") as handle:
            handle.write(json.dumps(row) + "\n")


def cache_path(directory, kind, index, injected):
    """The file of the flattened flux of a curve (with its planet or without), or None without a cache directory."""
    return None if directory is None else Path(directory) / f"{'inj2' if injected else 'null'}-{kind}-{seed_of(CLASSES.index(kind), index)}.npy"


def flattened(time, flux, path, flatten):
    """The flattened flux: read from the file when it exists, else computed (and written to the file when there is one). The 3-day biweight is the cost of the red and variable classes (about 0.8 s a curve), so a pool of CPU workers fills the cache before the device runs."""
    if path is not None and Path(path).exists():
        return np.load(path)
    flux = flatten(time, flux)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, flux)
    return flux


def shuffled_then_flattened(time, flux, edges, cadence, count, seed, flatten):
    """R13: `count` copies of the unflattened flux shuffled in blocks of one day within each sector, each then flattened, as the filter of the pipeline would see them."""
    injection = load("injection")
    return np.array([flatten(time, copy) for copy in injection.block_shuffles(flux, edges, cadence, count, seed=seed)])


def curve(kind, seed, sectors, baseline, path=None):
    """(time, flux, edges, cadence in days) of a noise-only light curve of the shared generator, flattened as the class asks (from the cache file `path` when it exists)."""
    generator = load("generator")
    time, edges = generator.time_axis(sectors, baseline=baseline)
    time, flux, _ = generator.noise(kind, seed, time)
    if kind != "white":
        flux = flattened(time, flux, path, generator.flatten)
    return time, flux, edges, generator.CADENCE_DAYS


def searched(time, fluxes, period_range, device):
    from transitgpu import engine

    return engine.search_many(time, fluxes, *period_range, device=device)


def null_case(kind, index, sectors=1, shuffles=SHUFFLES, baseline=27.4, period_range=PERIOD_RANGE, device=None, cache=None):
    """The row of a noise-only light curve: the best SDE and period of the curve, the SDEs of its shuffles and of the shuffles of its inversion (all searched in one call)."""
    started = clock.perf_counter()
    injection = load("injection")
    seed = seed_of(CLASSES.index(kind), index)
    time, flux, edges, cadence = curve(kind, seed, sectors, baseline, cache_path(cache, kind, index, False))
    copies = injection.block_shuffles(flux, edges, cadence, shuffles, seed=seed)
    inverted = injection.block_shuffles(invert(flux), edges, cadence, shuffles, seed=seed)
    found = searched(time, [flux, *copies, *inverted], period_range, device)
    return {"id": f"null:{kind}:{index}", "class": kind, "index": index, "seed": seed, "sde": found[0]["sde"], "period": found[0]["period_days"],
            "shuffle_sde": [f["sde"] for f in found[1:1 + shuffles]], "inverted_sde": [f["sde"] for f in found[1 + shuffles:]], "seconds": clock.perf_counter() - started}


PLANET_PERIODS = (0.8, 1.6, 3.2, 6.4, 12.8)
PLANET_SNRS = (4, 6, 8, 10, 14, 20)
WHITE_PPM = {"white": 1000.0, "red": 700.0, "variable": 700.0}  # the white noise per point of each class, the sigma of its signal to noise


def cell_of(index):
    """(period, target S/N) of the planet of curve `index`: the 30 cells in order, periods slowest."""
    cell = index % (len(PLANET_PERIODS) * len(PLANET_SNRS))
    return PLANET_PERIODS[cell // len(PLANET_SNRS)], PLANET_SNRS[cell % len(PLANET_SNRS)]


def planet_of(kind, index, seed, time, edges, baseline, sectors, cache):
    """(the planet of curve `index`, the sigma its S/N is against). White: the white noise of the class. Red and variable: the noise of the quiet (flattened, noise-only) curve at the duration of the transit (R11),
    with the radius ratio solved twice because the duration depends on it."""
    generator = load("generator")
    period, target = cell_of(index)
    draw = np.random.default_rng(seed)
    body = generator.planet(period, draw.uniform(0.0, generator.IMPACT_MAXIMUM), draw.uniform())
    if kind == "white":
        return generator.with_signal_to_noise(body, time, WHITE_PPM[kind] * 1e-6, target), WHITE_PPM[kind] * 1e-6
    quiet = curve(kind, seed, sectors, baseline, cache_path(cache, kind, index, False))[1]
    for _ in range(2):
        body = generator.with_signal_to_noise(body, time, generator.effective_sigma(quiet, generator.duration_days(body)), target)
    return body, generator.effective_sigma(quiet, generator.duration_days(body))


def injection_case(kind, index, sectors=1, shuffles=SHUFFLES, baseline=27.4, period_range=PERIOD_RANGE, device=None, cache=None):
    """The row of the noise of null curve `index` with the planet of its cell injected before flattening: the best SDE, period and epoch, the label of the peak, and the SDEs of the shuffles of the curve and of its inversion."""
    started = clock.perf_counter()
    generator, injection = load("generator"), load("injection")
    seed = seed_of(CLASSES.index(kind), index)
    period, target = cell_of(index)
    time, edges = generator.time_axis(sectors, baseline=baseline)
    time, flux, _ = generator.noise(kind, seed, time)
    body, sigma = planet_of(kind, index, seed, time, edges, baseline, sectors, cache)
    flux, first = generator.inject(body, time, flux)
    if kind != "white":
        flux = flattened(time, flux, cache_path(cache, kind, index, True), generator.flatten)
    copies = injection.block_shuffles(flux, edges, generator.CADENCE_DAYS, shuffles, seed=seed)
    inverted = injection.block_shuffles(invert(flux), edges, generator.CADENCE_DAYS, shuffles, seed=seed)
    found = searched(time, [flux, *copies, *inverted], period_range, device)
    return {"id": f"inj:{kind}:{index}", "class": kind, "index": index, "seed": seed, "period_injected": period, "target_snr": target, "radius_ratio": body["radius_ratio"], "sigma": sigma, "impact": body["impact"],
            "sde": found[0]["sde"], "period": found[0]["period_days"], "label": generator.classify(found[0]["period_days"], found[0]["t0_days"], body, first),
            "shuffle_sde": [f["sde"] for f in found[1:1 + shuffles]], "inverted_sde": [f["sde"] for f in found[1 + shuffles:]], "seconds": clock.perf_counter() - started}


def prepare(args):
    """Fill the cache with the flattened flux of one curve: (mode, kind, index, directory). Run by a pool of CPU workers."""
    mode, kind, index, directory = args
    if mode == "null":
        null_flux = curve(kind, seed_of(CLASSES.index(kind), index), 1, 27.4, cache_path(directory, kind, index, False))
        return null_flux[1].size
    generator = load("generator")
    seed = seed_of(CLASSES.index(kind), index)
    time, edges = generator.time_axis(1, baseline=27.4)
    time, flux, _ = generator.noise(kind, seed, time)
    body, _ = planet_of(kind, index, seed, time, edges, 27.4, 1, directory)
    return flattened(time, generator.inject(body, time, flux)[0], cache_path(directory, kind, index, True), generator.flatten).size


def reflat_path(directory, kind, index):
    return Path(directory) / f"reflat-{kind}-{seed_of(CLASSES.index(kind), index)}.npy"


def prepare_reflat(args):
    """R13, run by a pool of CPU workers: the 48 copies of the unflattened flux, shuffled in blocks and then flattened, saved as one file a curve. (kind, index, directory, sectors, baseline, shuffles)"""
    kind, index, directory, sectors, baseline, shuffles = args
    path = reflat_path(directory, kind, index)
    if path.exists():
        return 0
    generator = load("generator")
    seed = seed_of(CLASSES.index(kind), index)
    time, edges = generator.time_axis(sectors, baseline=baseline)
    time, raw, _ = generator.noise(kind, seed, time)
    copies = shuffled_then_flattened(time, raw, edges, generator.CADENCE_DAYS, shuffles, seed, generator.flatten)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, copies)
    return copies.shape[0]


def reflat_case(kind, index, directory, sectors=1, baseline=27.4, period_range=PERIOD_RANGE, device=None):
    """R13 on the device: the flattened curve (the cached flux of the null set) and its copies shuffled before flattening, searched together; the row has the best SDE and the SDEs of the copies."""
    started = clock.perf_counter()
    seed = seed_of(CLASSES.index(kind), index)
    time, flux, _, _ = curve(kind, seed, sectors, baseline, cache_path(directory, kind, index, False))
    found = searched(time, [flux, *np.load(reflat_path(directory, kind, index))], period_range, device)
    return {"id": f"reflat:{kind}:{index}", "class": kind, "index": index, "seed": seed, "sde": found[0]["sde"], "period": found[0]["period_days"], "shuffle_sde": [f["sde"] for f in found[1:]], "seconds": clock.perf_counter() - started}


REAL_SHUFFLES = 199


def real_case(entry, directory, shuffles=REAL_SHUFFLES, period_range=PERIOD_RANGE, device=None):
    """The row of a real star of stars.json (fetch.py): its best SDE and the SDEs of its shuffles, and the share of the shuffles below the best SDE (uniform on 0 to 1 for an exchangeable star)."""
    started = clock.perf_counter()
    generator, injection, fetch = load("generator"), load("injection"), load("fetch")
    time, flux, _ = fetch.light_curve(fetch.download(entry, directory))
    flux = generator.flatten(time, flux)
    copies = injection.block_shuffles(flux, [0, time.size], generator.CADENCE_DAYS, shuffles, seed=entry["tic"])
    found = searched(time, [flux, *copies], period_range, device)
    sdes = [f["sde"] for f in found[1:]]
    return {"id": f"real:{entry['tic']}:{entry['sector']}", "tic": entry["tic"], "sector": entry["sector"], "sde": found[0]["sde"], "period": found[0]["period_days"], "shuffle_sde": sdes,
            "rank_share": float(np.mean(np.array(sdes) < found[0]["sde"])), "seconds": clock.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("null", "inj", "real", "prepare", "prepare-reflat", "reflat"))
    parser.add_argument("kind", nargs="?", choices=CLASSES)
    parser.add_argument("count", nargs="?", type=int)
    parser.add_argument("--cache", type=Path, default=None, help="the directory of flattened fluxes (red and variable); `prepare` fills it, null and inj read it")
    parser.add_argument("--workers", type=int, default=1, help="processes of `prepare`")
    parser.add_argument("--data", type=Path, default=HERE / "out" / "data", help="where the light curves of the real stars are kept (downloaded when missing)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--budget", type=float, default=float("inf"), help="seconds after which no further case is started")
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    started = clock.perf_counter()
    if options.mode == "real":
        entries = {f"real:{entry['tic']}:{entry['sector']}": entry for entry in json.loads((HERE / "stars.json").read_text())}
        run_cases(list(entries), options.out / "threshold.jsonl", lambda case_id: print(case_id, flush=True) or real_case(entries[case_id], options.data))
        return

    if options.mode == "prepare-reflat":
        import multiprocessing

        jobs = [(options.kind, index, options.cache, 1, 27.4, SHUFFLES) for index in range(options.count) if not reflat_path(options.cache, options.kind, index).exists()]
        with multiprocessing.get_context("spawn").Pool(options.workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(prepare_reflat, jobs), 1):
                if done % 20 == 0 or done == len(jobs):
                    print(f"{done} of {len(jobs)} copies flattened, {clock.perf_counter() - started:.0f} s", flush=True)
        return
    if options.mode == "prepare":
        import multiprocessing

        if options.kind == "white" or options.cache is None:
            raise SystemExit("prepare fills the cache of the red or variable class: give a class and --cache DIR")
        jobs = [(mode, options.kind, index, options.cache) for mode in ("null", "inj") for index in range(options.count if mode == "null" else min(options.count, INJECTIONS))
                if not cache_path(options.cache, options.kind, index, mode == "inj").exists()]
        with multiprocessing.get_context("spawn").Pool(options.workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(prepare, jobs, chunksize=8), 1):
                if done % 200 == 0 or done == len(jobs):
                    print(f"{done} of {len(jobs)} flattened, {clock.perf_counter() - started:.0f} s", flush=True)
        return

    def measure(case_id):
        if options.mode == "reflat":
            return reflat_case(options.kind, int(case_id.rsplit(":", 1)[1]), options.cache)
        if clock.perf_counter() - started > options.budget:
            raise SystemExit(f"budget of {options.budget:.0f} s reached before {case_id}")
        row = (null_case if options.mode == "null" else injection_case)(options.kind, int(case_id.rsplit(":", 1)[1]), cache=options.cache)
        print(case_id, f"{row['seconds']:.2f} s", flush=True)
        return row

    run_cases([f"{options.mode}:{options.kind}:{index}" for index in range(options.count)], options.out / "threshold.jsonl", measure)


if __name__ == "__main__":
    main()
