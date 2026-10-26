"""The searches of the evaluation on one light curve, each timed: exact TLS on the GPU (the reference), binned TLS, the box search and the hybrid at several peak counts. A result is a dict of plain numbers, so a case is one JSON line."""

import time as clock

import numpy as np

import transitgpu
from transitgpu.tls import transitleastsquares

PERIOD_MIN, PERIOD_MAX = 0.5, 14.0
PEAKS = (1, 5, 10, 20)
WINDOW = 0.01


def timed(function):
    started = clock.perf_counter()
    value = function()
    return value, clock.perf_counter() - started


def tls(time, flux, errors, backend):
    found, seconds = timed(lambda: transitleastsquares(time, flux, errors).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, backend=backend, show_progress_bar=False, verbose=False))
    return {"period": float(found.period), "t0": float(found.T0), "sde": float(found.SDE), "seconds": seconds, "spectrum_periods": found.periods, "spectrum_sde": found.power}


def box(time, flux, errors):
    found, seconds = timed(lambda: transitgpu.box_search(time, flux, errors).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX))
    return {"period": float(found.period), "t0": float(found.T0), "sde": float(found.SDE), "seconds": seconds}


def hybrid(time, flux, errors, peaks):
    found, seconds = timed(lambda: transitgpu.bls_tls_hybrid(time, flux, errors, period_min=PERIOD_MIN, period_max=PERIOD_MAX, peaks=peaks, window=WINDOW))
    rows = [{"rank": p.rank, "box_period": float(p.box_period), "box_sde": float(p.box_sde), "period": None if p.tls is None else float(p.tls.period), "t0": None if p.tls is None else float(p.tls.T0)} for p in found.peaks]
    return {"sde": float(found.sde), "peaks": rows, "seconds": seconds}


def run_all(time, flux, errors, time_hybrids=False):
    """The reference (exact TLS), binned TLS, the box search and the hybrid of 20 peaks (its first 1, 5 and 10 are the hybrids of those counts: the peaks are nested). With time_hybrids, the hybrid is also run at each count for its run time."""
    out = {"exact": tls(time, flux, errors, "gpu"), "binned": tls(time, flux, errors, "gpu-binned"), "box": box(time, flux, errors), "hybrid": hybrid(time, flux, errors, max(PEAKS))}
    if time_hybrids:
        out["hybrid_seconds"] = {str(count): hybrid(time, flux, errors, count)["seconds"] for count in PEAKS}
    return out


def cpu_exact(time, flux, errors, threads=4):
    """TLS 2.0.1's own exact backend on the CPU, for the reference check: the period, the transit time and the SDE."""
    import os

    from transitleastsquares import transitleastsquares as upstream

    os.environ["TLS_BACKEND"] = "fused"  # TLS 2.0.1's exact backend; its default is the piecewise-linear one
    found, seconds = timed(lambda: upstream(time, flux, errors).power(period_min=PERIOD_MIN, period_max=PERIOD_MAX, use_threads=threads, show_progress_bar=False, verbose=False))
    return {"period": float(found.period), "t0": float(found.T0), "sde": float(found.SDE), "seconds": seconds}
