"""The search of many fluxes folds the times once for all of them: a block shuffle of a star only moves flux between points, so the fold depends on the times alone. A function test: it counts the calls of the engine's fold."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve  # noqa: E402

from transitgpu import core, engine  # noqa: E402


def four_fluxes():
    """The star with a gap and an infinite point, the same rolled by a day, another gap, and a deeper transit."""
    time, flux = light_curve(seed=3, cadence_min=30.0)
    first = flux.copy()
    first[100:140] = np.nan
    first[500] = np.inf
    second = flux.copy()
    second[300:330] = np.nan
    return time, [first, np.roll(first, 48), second, light_curve(seed=5, cadence_min=30.0, depth_ppm=6000)[1]]


def test_the_fold_is_made_once_for_all_the_fluxes_of_a_batch_of_periods(monkeypatch):
    time, rows = four_fluxes()
    calls = []
    monkeypatch.setattr(core, "fold", lambda *args: calls.append(1) or core.fold_index(*args))
    engine.search_many(time, rows[:1], 0.5, 14, device="cpu")
    alone = len(calls)
    calls.clear()
    engine.search_many(time, rows, 0.5, 14, device="cpu")
    assert len(calls) == alone > 0, "four fluxes cost the folds of one"
