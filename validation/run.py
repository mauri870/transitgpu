"""The injection-recovery evaluation of README.md: the cases of a part, each run through the searches of methods.py, one JSON line a case. A case whose line is in the output file is skipped before anything is generated. usage:
python run.py PART OUTPUT_DIRECTORY [WORKERS] [BUDGET_SECONDS] [LIMIT_OF_CASES]
PART is main-white, main-red, main-variable, real, four-white, four-red or four-variable. `touch OUTPUT_DIRECTORY/STOP` ends a run between cases.
"""

import json
import sys
import time as clock
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import generator
from generator import NOISE_CLASSES

HERE = Path(__file__).parent
PERIODS = (0.8, 1.6, 3.2, 6.4, 12.8)
SIGNAL_TO_NOISE = (4.0, 6.0, 8.0, 10.0, 14.0, 20.0)
CELLS = len(PERIODS) * len(SIGNAL_TO_NOISE)
INJECTED = {"main": 40, "four": 8}  # replicates a cell for the synthetic classes
NULLS = {"main": 1000, "four": 300}
REAL = {"injected": 3, "shuffles": 100}  # replicates a cell and shuffled nulls for each of the 12 real stars
CHECKED = 20  # injected light curves of a synthetic class (index below this) that the CPU backend of TLS also runs on
TIMED_EVERY = 8  # every eighth injected light curve also times the hybrid at each peak count
CLASS_INDEX = {"white": 0, "red": 1, "variable": 2, "real": 3}
BASELINE_SECTORS = {"main": 1, "four": 4}
BASELINE_SEED = {"main": 0, "four": 200_000}
NULL_SEED = 100_000


def case_ids(part):
    """The ids `class:baseline:kind:index` of a part, in order."""
    if part == "real":
        stars = len(json.loads((HERE / "stars.json").read_text()))
        return [f"real:main:inj:{i}" for i in range(stars * CELLS * REAL["injected"])] + [f"real:main:null:{i}" for i in range(stars * (REAL["shuffles"] + 1))]
    baseline, kind = part.split("-")
    return [f"{kind}:{baseline}:inj:{i}" for i in range(CELLS * INJECTED[baseline])] + [f"{kind}:{baseline}:null:{i}" for i in range(NULLS[baseline])]


def seed_of(kind, baseline, injected, index):
    return 1_000_000 * CLASS_INDEX[kind] + BASELINE_SEED[baseline] + (0 if injected else NULL_SEED) + index


def planet_of(index, seed, time, sigma_of):
    """The planet of injected case `index`: its cell gives the period and the S/N, the seed the impact parameter and the epoch. `sigma_of(duration)` is the noise per point that a transit of that duration (days) sees; the radius ratio is solved twice, since the duration depends on it."""
    cell = index % CELLS
    generator_ = np.random.default_rng(seed)
    body = generator.planet(PERIODS[cell // len(SIGNAL_TO_NOISE)], generator_.uniform(0.0, generator.IMPACT_MAXIMUM), generator_.uniform())
    target = SIGNAL_TO_NOISE[cell % len(SIGNAL_TO_NOISE)]
    for _ in range(2):
        body = generator.with_signal_to_noise(body, time, sigma_of(generator.duration_days(body)), target)
    return body, target, sigma_of(generator.duration_days(body))


def light_curve(case_id, directory):
    """Return (time, flux, errors, sector edges, the planet and its first transit or None, details) of a case."""
    kind, baseline, mode, number = case_id.split(":")
    index, injected = int(number), mode == "inj"
    if kind == "real":
        import fetch

        stars = fetch.stars()
        star = index % len(stars)
        time, flux, errors = fetch.light_curve(fetch.download(stars[star], directory / "stars"))
        errors = np.full(errors.size, float(np.median(errors)))  # binned TLS needs equal errors, and every search is given the same input
        edges = np.array([0, time.size])
        seed = seed_of("real", baseline, injected, index)
        body, snr, sigma = None, None, None
        if injected:
            quiet = generator.flatten(time, flux)
            body, snr, sigma = planet_of((index // len(stars)) % CELLS, seed, time, lambda duration: generator.effective_sigma(quiet, duration))
            flux, first = generator.inject(body, time, flux)
        flux = generator.flatten(time, flux)
        if not injected and index // len(stars) > 0:
            flux = generator.block_shuffle(flux, edges, index // len(stars))
        return time, flux, errors, edges, (body, first if injected else None), {"star": stars[star]["tic"], "snr": snr, "sigma_eff": sigma}
    time, edges = generator.time_axis(BASELINE_SECTORS[baseline])
    seed = seed_of(kind, baseline, injected, index)
    time, flux, errors = generator.noise(kind, seed, time)
    body, snr, sigma = None, None, None
    if injected:
        if kind == "white":
            body, snr, sigma = planet_of(index, seed, time, lambda duration: float(errors[0]))
        else:
            quiet = generator.flatten(time, flux)
            body, snr, sigma = planet_of(index, seed, time, lambda duration: generator.effective_sigma(quiet, duration))
        flux, first = generator.inject(body, time, flux)
    if kind != "white":
        flux = generator.flatten(time, flux)
    return time, flux, errors, edges, (body, first if injected else None), {"snr": snr, "sigma_eff": sigma}


def one(arguments):
    case_id, directory = arguments
    import methods

    started = clock.perf_counter()
    time, flux, errors, edges, (body, first), details = light_curve(case_id, Path(directory))
    index = int(case_id.split(":")[3])
    found = methods.run_all(time, flux, errors, time_hybrids=body is not None and index % TIMED_EVERY == 0)
    record = {"id": case_id, "points": int(time.size), "filtered_ppm": float(np.std(flux) * 1e6), **details, "planet": None if body is None else {k: body[k] for k in ("period", "impact", "radius_ratio", "epoch_fraction")} | {"first_transit": first, "duration_days": generator.duration_days(body)}}
    for name in ("exact", "binned", "box"):
        search = {k: v for k, v in found[name].items() if not k.startswith("spectrum")}
        if body is not None:
            search["label"] = generator.classify(search["period"], search["t0"], body, first)
        record[name] = search
    record["hybrid"] = {"sde": found["hybrid"]["sde"], "seconds": found["hybrid"]["seconds"], "peaks": found["hybrid"]["peaks"]}
    if body is not None:
        for peak in record["hybrid"]["peaks"]:
            period, t0 = (peak["period"], peak["t0"]) if peak["period"] is not None else (peak["box_period"], 0.0)
            peak["label"] = generator.classify(period, t0 if peak["period"] is not None else first, body, first)
    if "hybrid_seconds" in found:
        record["hybrid_seconds"] = found["hybrid_seconds"]
    if index < CHECKED and case_id.split(":")[1] == "main" and case_id.split(":")[0] in NOISE_CLASSES and case_id.split(":")[2] == "inj":
        record["cpu"] = methods.cpu_exact(time, flux, errors)
    record["case_seconds"] = clock.perf_counter() - started
    return record


def finished(output):
    """The ids of the complete lines of an output file; a last line cut by a killed run is not one."""
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def repair(output):
    """Cut a last line that a killed run left without its newline, so that the next line starts on a line of its own."""
    if output.exists():
        content = output.read_bytes()
        if content and not content.endswith(b"\n"):
            output.write_bytes(content[:content.rfind(b"\n") + 1])


def pending(part, output):
    done = finished(output)
    return [case_id for case_id in case_ids(part) if case_id not in done]


def main(part, directory, workers=4, budget=float("inf"), limit=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{part}.jsonl"
    repair(output)
    todo = pending(part, output)[:limit]
    print(f"{part}: {len(todo)} cases to run", flush=True)
    started, count = clock.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool, output.open("a") as handle:
        futures = []
        for case_id in todo:
            futures.append(pool.submit(one, (case_id, str(directory))))
        for future in futures:
            if clock.perf_counter() - started >= budget or (directory / "STOP").exists():
                for other in futures:
                    other.cancel()
                break
            record = future.result()
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            count += 1
            if count % 100 == 0:
                print(count, f"cases, {clock.perf_counter() - started:.0f} s", flush=True)
    print(f"{count} cases in {clock.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4, float(sys.argv[4]) if len(sys.argv) > 4 else float("inf"), int(sys.argv[5]) if len(sys.argv) > 5 else None)
