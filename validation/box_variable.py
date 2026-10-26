"""Three box searches on the variable class of the evaluation, to tell whether the loss of the box search there is the box statistic or this implementation of it: the box search of transitgpu, its SDE after the
spectrum detrending of TLS 2.0.1 (the running median of `transitleastsquares.stats.sde_trend`, then the z-score, applied to the box statistic's z-scores since the box search keeps no chi square), and astropy's
BoxLeastSquares with the periods of the box search and, at each period, the durations the box search scores there (the `snr` objective squared, which is the box statistic). The light curves are the 200 injected
variable-class light curves of signal to noise 8 and the first 200 with nothing injected, regenerated from their seeds by `run.light_curve`. Each search is counted at the threshold of its own nulls, and each is
paired with exact mode read from the evaluation's rows (`EVALUATION_DIRECTORY/main-variable.jsonl`) on the same cases, exact mode's threshold taken on the same 200 nulls. One JSON line a case into
OUTPUT_DIRECTORY/box-variable.jsonl (a case whose line is there is skipped), then the counts, the Wilson and bootstrap intervals and the paired Tango differences into OUT. usage:
python box_variable.py OUTPUT_DIRECTORY EVALUATION_DIRECTORY [--out ../paper/box_variable.json] [--workers 4] [--limit CASES]
"""

import argparse
import json
import sys
import time as clock
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import analyse
import generator
import run

HERE = Path(__file__).resolve().parent
CASES = 200
OVERSAMPLING = 3  # the oversampling of TLS's period grid, which the box search uses; the kernel of the running median depends on it
SEARCHES = ("box", "box_detrended", "astropy")
OUTPUT = "box-variable.jsonl"


def case_ids():
    """The injected variable-class cases of signal to noise 8 (the cells of that level: every sixth index from 2) and the first CASES nulls."""
    return [f"variable:main:inj:{6 * k + 2}" for k in range(CASES)] + [f"variable:main:null:{k}" for k in range(CASES)]


def detrended_sde(power, periods):
    """The SDE of a spectrum after the running-median detrending of TLS 2.0.1 (`stats.sde_trend` with the median kernel, then the renormalization of `stats.spectra`); the plain z-score when the grid is too short for the kernel."""
    from transitleastsquares.stats import sde_trend

    power = np.asarray(power, dtype=float)
    trend = sde_trend(power, np.asarray(periods, dtype=float), OVERSAMPLING, "median")
    if trend is None:
        return float(np.max((power - power.mean()) / power.std()))
    detrended = power - trend
    detrended = detrended - np.mean(detrended)
    return float(np.max(detrended / np.std(detrended)))


def durations_of(period, shortest, longest):
    """The durations, in days, that the box search scores at a period: from the shortest TLS allows in steps of its ratio to the longest."""
    from transitgpu.core import DURATION_STEP

    low, high = shortest * period, longest * period
    steps = max(1, int(np.ceil(np.log(high / low) / np.log(DURATION_STEP))) + 1)
    return low * DURATION_STEP ** np.arange(steps)


def astropy_box(time, flux, errors, periods):
    """astropy's BoxLeastSquares at each of `periods` with that period's durations: the box statistic (the `snr` objective squared, zero for a brightening) and its z-score over the periods."""
    from astropy.timeseries import BoxLeastSquares

    from transitgpu.core import duration_fractions

    shortest, longest = duration_fractions(periods)
    model = BoxLeastSquares(time, flux, dy=errors)
    statistic, centre = np.empty(periods.size), np.empty(periods.size)
    for index, period in enumerate(periods):
        found = model.power(np.array([period]), durations_of(period, shortest[index], longest[index]), objective="snr")
        signal = float(found.power[0])
        statistic[index] = signal * signal if signal > 0 else 0.0
        centre[index] = float(found.transit_time[0])
    best = int(np.argmax(statistic))
    return {"period": float(periods[best]), "t0": centre[best], "sde": float((statistic[best] - statistic.mean()) / statistic.std())}


def statistics(time, flux, errors, period_min, period_max, device=None):
    """The three searches on one light curve: {"box", "box_detrended", "astropy"}, each with the period and transit time of its best peak, its SDE and its seconds. `device` is for a CPU run in the tests; the box
    search runs on the GPU by default. The detrended spectrum's peak is its own argmax, with the centre of the best box of that period."""
    import transitgpu
    from transitleastsquares.stats import sde_trend

    started = clock.perf_counter()
    search = transitgpu.box_search(time, flux, errors)
    result = search._power(period_min=period_min, period_max=period_max, device=device, boxes=True) if device is not None else search.power(period_min=period_min, period_max=period_max, boxes=True)
    box_seconds = clock.perf_counter() - started
    box = {"period": float(result.period), "t0": float(result.T0), "sde": float(result.SDE), "seconds": box_seconds}
    started = clock.perf_counter()
    power, periods = np.asarray(result.power, dtype=float), np.asarray(result.periods, dtype=float)
    trend = sde_trend(power, periods, OVERSAMPLING, "median")
    best = int(np.argmax(power - trend)) if trend is not None else int(np.argmax(power))
    detrended = {"period": float(periods[best]), "t0": float(result.box["centers"][best]), "sde": detrended_sde(power, periods), "seconds": box_seconds + clock.perf_counter() - started}
    started = clock.perf_counter()
    astropy = astropy_box(time, flux, errors, periods)
    astropy["seconds"] = clock.perf_counter() - started
    return {"box": box, "box_detrended": detrended, "astropy": astropy}


def one(arguments):
    case_id, directory = arguments
    from methods import PERIOD_MAX, PERIOD_MIN

    started = clock.perf_counter()
    time, flux, errors, _, (body, first), details = run.light_curve(case_id, Path(directory))
    found = statistics(time, flux, errors, PERIOD_MIN, PERIOD_MAX)
    if body is not None:
        for search in found.values():
            search["label"] = generator.classify(search["period"], search["t0"], body, first)
    return {"id": case_id, "points": int(time.size), **details, "planet": None if body is None else {k: body[k] for k in ("period", "impact", "radius_ratio", "epoch_fraction")}, **found,
            "case_seconds": clock.perf_counter() - started}


def found(record, search, cut):
    return record[search]["sde"] >= cut and record[search].get("label") == "right"


def analysis(records, evaluation_records):
    """The counts of the three searches and of exact mode on the same cases, each at the threshold of its own nulls among these cases, with the Wilson interval, the bootstrap interval of the threshold, the
    right peaks and the harmonics whatever the SDE, and the paired Tango differences: each search minus exact, astropy minus the box search and the detrended box minus the box search."""
    exact = {r["id"]: r for r in evaluation_records}
    for record in records:
        if record["id"] not in exact:
            raise ValueError(f"{record['id']} is not in the evaluation")
    joined = [{**r, "exact": exact[r["id"]]["exact"]} for r in records]
    injected = [r for r in joined if r["id"].split(":")[2] == "inj"]
    null = [r for r in joined if r["id"].split(":")[2] == "null"]
    out = {"injected": len(injected), "null": len(null), "signal_to_noise": sorted({r["snr"] for r in injected}), "searches": {}, "paired": {}}
    cuts = {rate: {} for rate in analyse.RATES}
    for search in SEARCHES + ("exact",):
        entry = {"right_peak": sum(r[search].get("label") == "right" for r in injected), "harmonic": sum(r[search].get("label") == "harmonic" for r in injected)}
        for rate in analyse.RATES:
            cut, low, high = analyse.threshold([r[search]["sde"] for r in null], rate)
            cuts[rate][search] = cut
            k = sum(found(r, search, cut) for r in injected)
            entry[str(rate)] = {"threshold": cut, "threshold_bootstrap": [low, high], "found": k, "n": len(injected), "wilson": analyse.wilson(k, len(injected))}
        if search != "exact":
            entry["seconds_median"] = float(np.median([r[search]["seconds"] for r in joined]))
        out["searches"][search] = entry
    for rate in analyse.RATES:
        pairs = {}
        for first, second in [(s, "exact") for s in SEARCHES] + [("astropy", "box"), ("box_detrended", "box")]:
            only_first = sum(found(r, first, cuts[rate][first]) and not found(r, second, cuts[rate][second]) for r in injected)
            only_second = sum(found(r, second, cuts[rate][second]) and not found(r, first, cuts[rate][first]) for r in injected)
            estimate, low, high = analyse.tango(only_first, only_second, len(injected))
            pairs[f"{first}_minus_{second}"] = {"estimate": estimate, "low": low, "high": high, "only_first": only_first, "only_second": only_second, "n": len(injected)}
        out["paired"][str(rate)] = pairs
    return out


def main(directory, evaluation_directory, out, workers=4, limit=None):
    directory, evaluation_directory = Path(directory), Path(evaluation_directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / OUTPUT
    run.repair(output)
    done = run.finished(output)
    todo = [case_id for case_id in case_ids() if case_id not in done][:limit]
    print(f"{len(todo)} cases to run", flush=True)
    started = clock.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool, output.open("a") as handle:
        for count, record in enumerate(pool.map(one, [(case_id, str(directory)) for case_id in todo], chunksize=1), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if count % 100 == 0:
                print(count, f"cases, {clock.perf_counter() - started:.0f} s", flush=True)
    print(f"{len(todo)} cases in {clock.perf_counter() - started:.0f} s", flush=True)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    evaluation = [json.loads(line) for line in (evaluation_directory / "main-variable.jsonl").read_text().splitlines()]
    result = analysis(records, evaluation)
    Path(out).write_text(json.dumps(result, indent=1))
    for search, entry in result["searches"].items():
        at = entry["0.05"]
        print(f"{search}: {at['found']} of {at['n']} found at a false-alarm rate of 5 percent (threshold {at['threshold']:.2f}; Wilson {100 * at['wilson'][0]:.1f} to {100 * at['wilson'][1]:.1f} percent), right peak {entry['right_peak']}, harmonic {entry['harmonic']}")
    for name, pair in result["paired"]["0.05"].items():
        print(f"{name}: {100 * pair['estimate']:+.1f} points ({100 * pair['low']:+.1f} to {100 * pair['high']:+.1f}), only first {pair['only_first']}, only second {pair['only_second']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("evaluation_directory")
    parser.add_argument("--out", default=str(HERE.parent / "paper" / "box_variable.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="run this many cases and stop (the analysis still runs on every case in the output file)")
    arguments = parser.parse_args()
    main(arguments.directory, arguments.evaluation_directory, arguments.out, arguments.workers, arguments.limit)
    sys.exit(0)
