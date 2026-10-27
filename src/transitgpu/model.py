"""Box search on a GPU with the interface of Transit Least Squares.

    model = box_search(t, y, dy)
    results = model.power(period_min=0.5, period_max=14, oversampling_factor=3)

The results have the attributes of TLS's (https://transitleastsquares.readthedocs.io/en/latest/Python%20interface.html):
periods, power, SDE, SDE_raw, period, period_uncertainty, T0, duration, depth, depth_mean, snr and transit_count.
Periods, T0, duration and period_uncertainty are in days; a depth is the mean flux in the transit relative to the flux
outside it (0.99 for a dip of 1%). SDE is the z-score of the best statistic of each period over the periods searched;
the spectrum is not detrended, so SDE and SDE_raw are equal.
"""

import functools
import inspect

import numpy as np
import torch

from transitgpu import core

NO_CUDA = "device 'cuda' requested but torch finds no CUDA or ROCm device; install the torch build for your driver: https://docs.astral.sh/uv/guides/integration/pytorch/"
NO_DEVICE = "transitgpu needs a GPU; on a CPU use astropy.timeseries.BoxLeastSquares"


def require_device(device=None):
    """Raise RuntimeError unless the search would run on a GPU.

    The message is NO_DEVICE for the CPU (asked for, or the only device) and NO_CUDA for a CUDA device that torch
    does not find.
    """
    chosen = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if chosen.type == "cpu":
        raise RuntimeError(NO_DEVICE)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(NO_CUDA)


def _on_a_gpu(function):
    """Wrap an engine function of transitgpu.core so that it calls `require_device` first."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def public(*arguments, **options):
        require_device(signature.bind_partial(*arguments, **options).arguments.get("device"))
        return function(*arguments, **options)

    return public


search = _on_a_gpu(core.search)
search_many = _on_a_gpu(core.search_many)


class Results:
    """The results of a search, with the attributes of TLS's results.

    `box` holds the center and the width in days of the best box of each period when the search was asked for them,
    `fields` is the dict that transitgpu.search returns, and `bump` is the Results of the best brightening (a depth
    above 1) when the search was asked for bumps, None otherwise.
    """

    def __init__(self, found, boxes=False, t=None, y=None):
        self.fields = found
        self.bump = Results(found["bump"], boxes, t, y) if "bump" in found else None
        self.periods = found["periods"]
        self.power = found["power"]
        self.SDE = found["sde"]
        self.SDE_raw = found["sde"]
        self.period = found["period_days"]
        self.period_uncertainty = found["period_uncertainty_days"]
        self.T0 = found["t0_days"]
        self.duration = found["duration_hr"] / 24
        self.depth = 1 - found["depth_ppm"] * 1e-6
        self.depth_mean = 1 - found["depth_mean_ppm"] * 1e-6
        self.snr = found["snr"]
        self.transit_count = found["transit_count"]
        self.period_min_searched = found["period_min_searched_days"]
        self.period_max_searched = found["period_max_searched_days"]
        self.box = {"centers": found["centers"], "widths_days": found["widths_days"]} if boxes else None
        if t is not None:
            self._fold(np.asarray(t, dtype=float), np.asarray(y, dtype=float))

    def _fold(self, t, y):
        """Fold the light curve at the best period and set the folded arrays of TLS's results.

        folded_phase runs from 0 to 1 with the transit at 0.5 and folded_y is the flux in phase order;
        model_folded_phase and model_folded_model are the box found, 1 outside its duration and `depth` inside.
        """
        keep = np.isfinite(t) & np.isfinite(y)
        phase = ((t[keep] - self.T0) / self.period + 0.5) % 1.0
        order = np.argsort(phase)
        self.folded_phase, self.folded_y = phase[order], y[keep][order]
        half = self.duration / self.period / 2
        self.model_folded_phase = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), [0.5 - half, 0.5 + half]]))
        inside = np.abs(self.model_folded_phase - 0.5) <= half
        self.model_folded_model = np.where(inside, self.depth, 1.0)


class box_search:
    """A light curve to search: times t in days, flux y and optional flux errors dy.

    With dy the points are weighted by 1 / dy**2 relative to their median; without it all points weigh the same.

    The bin sums, the periods and the depths of two searches of one light curve are equal. In a long process that has searched with many different settings, torch can stop compiling the scoring function and run it eagerly, and two searches
    can then differ by about 1e-9 relative in the spectrum.
    """

    def __init__(self, t, y, dy=None):
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.t.shape != self.y.shape or self.t.ndim != 1:
            raise ValueError(f"t and y must be one-dimensional and of one length; got shapes {self.t.shape} and {self.y.shape}")
        self.dy = None if dy is None else np.asarray(dy, dtype=float)
        if self.dy is not None and self.dy.shape != self.t.shape:
            raise ValueError(f"dy must have the shape of t; got {self.dy.shape} and {self.t.shape}")

    def power(self, period_min=None, period_max=None, oversampling_factor=None, n_transits_min=None, duration_grid_step=core.DURATION_STEP, minimum_bins=core.MINIMUM_BINS,
              use_threads=None, show_progress_bar=False, verbose=False, device=None, boxes=False, bumps=False):
        """Search the periods of TLS's grid from period_min to period_max days and return Results.

        period_min defaults to 0.5 d and period_max to half the time span. oversampling_factor and n_transits_min are
        the arguments of TLS's period grid, duration_grid_step is the ratio of consecutive durations, minimum_bins the
        number of bins across the shortest duration of a period, device a torch device name (by default the first
        GPU), boxes=True adds the best box of each period to the results, and bumps=True also searches for brightenings from
        the same fold: the results then have `bump`, the Results of that search. use_threads, show_progress_bar and
        verbose are accepted for TLS's interface and ignored. Raise RuntimeError without a GPU or for the device "cpu".
        """
        require_device(device)
        return self._power(period_min, period_max, oversampling_factor, n_transits_min, duration_grid_step, minimum_bins, use_threads, show_progress_bar, verbose, device, boxes, bumps)

    def _power(self, period_min=None, period_max=None, oversampling_factor=None, n_transits_min=None, duration_grid_step=core.DURATION_STEP, minimum_bins=core.MINIMUM_BINS,
              use_threads=None, show_progress_bar=False, verbose=False, device=None, boxes=False, bumps=False):
        """Return `power(...)` without the GPU check: on the CPU by default when there is no GPU."""
        if period_min is None:
            period_min = 0.5
        if period_max is None:
            keep = np.isfinite(self.t) & np.isfinite(self.y)
            period_max = float((self.t[keep].max() - self.t[keep].min()) / 2)
        weights = None if self.dy is None else 1.0 / self.dy**2
        found = core.search(self.t, self.y, period_min, period_max, spectrum=True, device=device, minimum_bins=minimum_bins, duration_step=duration_grid_step, boxes=boxes, weights=weights,
                            oversampling_factor=oversampling_factor, n_transits_min=n_transits_min, bumps=bumps)
        return Results(found, boxes=boxes, t=self.t, y=self.y)
