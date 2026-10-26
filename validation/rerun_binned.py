"""Binned TLS alone, run again on the cases of a part of the evaluation (the same light curves: the ids, seeds and nulls of run.py), one JSON line a case into OUTPUT_DIRECTORY/PART.binned-bands.jsonl, so that a
change of the binned mode (the bin count of each band) is measured against the exact rows that run.py already wrote, with analyse_binned.py. A case whose line is in the output file is skipped. usage:
BINNED_KWARGS='{"...": ...}' BINNED_BACKEND=gpu-binned python rerun_binned.py PART OUTPUT_DIRECTORY [WORKERS] [BUDGET_SECONDS] [LIMIT_OF_CASES]
BINNED_KWARGS is a JSON object of keyword arguments given to power() (the option of the bin count), recorded in every line; BINNED_BACKEND the backend (gpu-binned by default). `touch OUTPUT_DIRECTORY/STOP` ends a run between cases.
"""

import json
import os
import sys
import time as clock
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import generator
import run

SUFFIX = ".binned-bands.jsonl"


def settings():
    """The backend and the keyword arguments of power() from the environment."""
    return os.environ.get("BINNED_BACKEND", "gpu-binned"), json.loads(os.environ.get("BINNED_KWARGS", "{}"))


def binned(time, flux, errors, backend, kwargs):
    from methods import PERIOD_MAX, PERIOD_MIN
    from transitgpu.tls import transitleastsquares

    started = clock.perf_counter()
    found = transitleastsquares(time, flux, errors).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, backend=backend, show_progress_bar=False, verbose=False, **kwargs)
    return {"period": float(found.period), "t0": float(found.T0), "sde": float(found.SDE), "seconds": clock.perf_counter() - started}


def one(arguments):
    case_id, directory = arguments
    backend, kwargs = settings()
    started = clock.perf_counter()
    time, flux, errors, _, (body, first), details = run.light_curve(case_id, Path(directory))
    search = binned(time, flux, errors, backend, kwargs)
    if body is not None:
        search["label"] = generator.classify(search["period"], search["t0"], body, first)
    return {"id": case_id, "points": int(time.size), **details, "planet": None if body is None else {k: body[k] for k in ("period", "impact", "radius_ratio", "epoch_fraction")},
            "backend": backend, "kwargs": kwargs, "binned": search, "case_seconds": clock.perf_counter() - started}


def output_of(part, directory):
    return Path(directory) / f"{part}{SUFFIX}"


def main(part, directory, workers=4, budget=float("inf"), limit=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = output_of(part, directory)
    run.repair(output)
    done = run.finished(output)
    todo = [case_id for case_id in run.case_ids(part) if case_id not in done][:limit]
    print(f"{part}: {len(todo)} cases to run, backend and kwargs {settings()}", flush=True)
    started, count = clock.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool, output.open("a") as handle:
        futures = [pool.submit(one, (case_id, str(directory))) for case_id in todo]
        for future in futures:
            if clock.perf_counter() - started >= budget or (directory / "STOP").exists():
                for other in futures:
                    other.cancel()
                break
            handle.write(json.dumps(future.result()) + "\n")
            handle.flush()
            count += 1
            if count % 200 == 0:
                print(count, f"cases, {clock.perf_counter() - started:.0f} s", flush=True)
    print(f"{count} cases in {clock.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4, float(sys.argv[4]) if len(sys.argv) > 4 else float("inf"), int(sys.argv[5]) if len(sys.argv) > 5 else None)
