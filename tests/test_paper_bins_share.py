"""The exhaustive search and the exact score of a box in paper/scripts/bins_share.py: against a brute-force loop on a small array, and the share of the box that the exhaustive search itself finds. Function tests of the code that makes the numbers."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).resolve().parent.parent / "paper" / "scripts" / "bins_share.py").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("bins_share", Path(__file__).resolve().parent.parent / "paper" / "scripts" / "bins_share.py")
bins_share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bins_share)


def brute_force(phase, y, period, durations):
    size, best = len(y), 0.0
    for duration in durations:
        for start in phase:
            inside = ((phase - start) % period) < duration
            count, total = int(inside.sum()), float(y[inside].sum())
            if total < 0 and 3 <= count <= size - 3:
                best = max(best, total * total / (count * (1 - count / size)))
    return best


def test_the_exhaustive_search_equals_a_loop_over_every_start_and_duration_on_a_small_array():
    rng = np.random.default_rng(3)
    phase = rng.uniform(0, 2.0, 40)
    y = rng.normal(0, 1, 40)
    y -= y.mean()
    durations = [0.1, 0.25, 0.6]
    assert abs(bins_share.exhaustive(phase, y, 2.0, durations) - brute_force(phase, y, 2.0, durations)) < 1e-9


def test_a_box_that_is_not_a_dip_or_holds_too_few_points_scores_zero_and_a_planted_dip_scores_its_own_statistic():
    phase = np.linspace(0, 1.99, 100)
    y = np.zeros(100)
    y[40:50] = -1.0
    y -= y.mean()
    assert bins_share.box_statistic(phase, y, 2.0, 0.5, 0.01) == 0.0  # no point inside
    assert bins_share.box_statistic(phase, y, 2.0, 1.5, 0.3) == 0.0  # not a dip
    center = (phase[40] + phase[49]) / 2
    width = phase[49] - phase[40] + 0.001
    expected = bins_share.exhaustive(phase, y, 2.0, [width])
    assert abs(bins_share.box_statistic(phase, y, 2.0, center, width) - expected) < 1e-9


def test_the_differences_of_two_result_dicts_are_relative_and_absolute_and_equal_nans_are_equal():
    spec = importlib.util.spec_from_file_location("validation", Path(__file__).resolve().parent.parent / "paper" / "scripts" / "validation.py")
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    reference = {"a": np.array([1.0, 2.0, np.nan]), "b": np.array([0.0, 0.0])}
    mine = {"a": np.array([1.0, 2.0 + 2e-6, np.nan]), "b": np.array([0.0, 1e-9])}
    out = validation.differences(reference, mine)
    assert abs(out["a"][0] - 1e-6) < 1e-12 and abs(out["a"][1] - 2e-6) < 1e-12
    assert out["b"][1] == 1e-9 and out["b"][0] == 1e-9 / 1e-12
