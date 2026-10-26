"""Box search on the GPU over the whole period grid, then Transit Least Squares around the highest peaks.

    found = bls_tls_hybrid(t, y, dy, period_min=0.5, period_max=14, peaks=5, window=0.01)
    found.sde, found.periods, found.power        # the box spectrum, whose maximum is the SDE
    for peak in found.peaks:                      # in rank order, the highest first
        peak.box_period, peak.box_sde, peak.tls   # TLS's results for the window of this peak, or None and peak.reason

The SDE of a peak is the box search's: TLS's SDE of a window is a z-score over the periods of that window only. A
peak that TLS cannot refine (no period grid in the window, too few points) keeps the box period and SDE, has no TLS
results and records the reason. The refinement needs the extra `tls` (`uv pip install "transitgpu[tls] @ git+https://github.com/mauri870/transitgpu"`). Without a
GPU the call raises.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from transitgpu.model import box_search


@dataclass
class Peak:
    """One peak of the box spectrum.

    `rank` is 0 for the highest; `box_period` and `box_sde` are the box search's; `tls` is TLS's results for the
    window around the peak, or None when TLS could not refine it, and then `reason` says why.
    """

    rank: int
    box_period: float
    box_sde: float
    tls: Optional[Any] = None
    reason: Optional[str] = None


@dataclass
class HybridResults:
    """The box spectrum (`periods`, `power`, `sde`; `box` is its Results) and the peaks in rank order."""

    periods: np.ndarray
    power: np.ndarray
    sde: float
    box: Any
    peaks: list


def top_peaks(periods, power, count=5, window=0.01):
    """Return the periods of the highest peaks of a spectrum, highest first.

    A peak is a local maximum at least `window` (a fraction of the period) away from every higher peak chosen before
    it; at most `count` are returned. A spectrum without a local maximum gives its highest point.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if not 0 < window < 0.5:
        raise ValueError(f"window must be a fraction of the period between 0 and 0.5, got {window}")
    periods, power = np.asarray(periods, dtype=float), np.asarray(power, dtype=float)
    padded = np.concatenate([[-np.inf], power, [-np.inf]])
    maxima = np.flatnonzero((padded[1:-1] > padded[:-2]) & (padded[1:-1] >= padded[2:]))
    if maxima.size == 0:
        maxima = np.array([int(np.argmax(power))])
    chosen = []
    for index in maxima[np.argsort(-power[maxima], kind="stable")]:
        if all(abs(periods[index] / period - 1) >= window for period in chosen):
            chosen.append(float(periods[index]))
            if len(chosen) == count:
                break
    return chosen


def sde_at(periods, power, period):
    """Return the box spectrum at the trial period nearest to `period`."""
    return float(power[int(np.argmin(np.abs(np.asarray(periods) - period)))])


def refine_peaks(t, y, dy, periods, power, peaks=5, window=0.01, backend="gpu", tls_options=None):
    """Yield a Peak for each of the `peaks` highest peaks of a spectrum, in rank order.

    TLS's `power()` runs on the window of a peak when the generator reaches it, so closing the generator stops the
    refinement.
    """
    import transitgpu.tls.hybrid as refinement  # the extra tls, imported when the first peak is taken

    for rank, period in enumerate(top_peaks(periods, power, count=peaks, window=window)):
        results, reason = refinement.refine_peak(t, y, dy, period, window, backend, tls_options)
        yield Peak(rank=rank, box_period=period, box_sde=sde_at(periods, power, period), tls=results, reason=reason)


def bls_tls_hybrid(t, y, dy, period_min, period_max, peaks=5, window=0.01, backend="gpu", tls_options=None, **box_options):
    """Run the box search from period_min to period_max days, then TLS around each of its `peaks` highest peaks.

    TLS's `power()` searches period * (1 - window) to period * (1 + window) around each peak. `backend` is the TLS
    backend of the windows: "gpu" (exact, on the device), "gpu-binned" or any backend of TLS by name; None leaves
    TLS's default. `tls_options` are further arguments of TLS's `power()` (for example use_threads) and
    `box_options` those of `box_search(...).power()` (oversampling_factor, n_transits_min, duration_grid_step,
    minimum_bins, device). Return HybridResults.
    """
    box = box_search(t, y, dy).power(period_min=period_min, period_max=period_max, **box_options)  # raises without a GPU, before TLS is imported
    found = list(refine_peaks(t, y, dy, box.periods, box.power, peaks=peaks, window=window, backend=backend, tls_options=tls_options))
    return HybridResults(periods=box.periods, power=box.power, sde=float(box.SDE), box=box, peaks=found)
