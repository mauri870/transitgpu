"""The injected planets, the criterion of recovery and the shuffles of paper/scripts/injection.py: seeds, priors, the
boundary of the tolerances, and the blocks that stay inside their sector."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "paper" / "scripts"
if not SCRIPTS.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("injection", SCRIPTS / "injection.py")
injection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(injection)


def test_a_planet_is_drawn_from_its_index_alone_and_within_the_priors():
    assert injection.draw(7) == injection.draw(7) and injection.draw(7) != injection.draw(8)
    for index in range(injection.PLANETS):
        planet = injection.draw(index)
        assert 0.5 <= planet["period"] <= 14 and 0.005 <= planet["radius_ratio"] <= 0.1 and 0 <= planet["impact"] <= 0.9 and planet["semi_major"] >= 2


def test_the_duration_of_a_central_transit_of_a_planet_with_a_one_day_period_is_the_known_value():
    planet = {"period": 1.0, "semi_major": 4.2, "impact": 0.0, "radius_ratio": 0.01}
    assert abs(injection.duration_days(planet) - 1 / np.pi * np.arcsin(1.01 / 4.2)) < 1e-12


def test_the_period_and_time_tolerances_are_inclusive_and_a_peak_below_the_threshold_is_right_and_not_found():
    planet = {"period": 2.0, "semi_major": 6.7, "impact": 0.0, "radius_ratio": 0.05}
    half = injection.duration_days(planet) / 2
    assert injection.found(planet, 10.0, 2.0 * 1.0099, 10.0 + 2.0 * 3 + half * 0.99, 9.0, 8.0) == {"right": True, "found": True}
    assert injection.found(planet, 10.0, 2.0, 10.0 + half * 1.01, 9.0, 8.0)["right"] is False
    assert injection.found(planet, 10.0, 2.0 * 1.0101, 10.0, 9.0, 8.0)["right"] is False
    assert injection.found(planet, 10.0, 2.0, 10.0, 7.9, 8.0) == {"right": True, "found": False}


def test_a_shuffle_keeps_the_points_of_each_sector_in_that_sector_and_the_leftover_points_in_place():
    flux = np.arange(23, dtype=float)
    copies = injection.block_shuffles(flux, [0, 11, 23], 1 / 5, 3, seed=4)  # blocks of 5 points
    assert len(copies) == 3
    for copy in copies:
        assert sorted(copy[:11]) == sorted(flux[:11]) and sorted(copy[11:]) == sorted(flux[11:])
        assert copy[10] == 10 and copy[22] == 22  # the leftover point of each sector stays at its end
    assert not np.array_equal(copies[0], copies[1])


def test_the_counts_by_rank_use_the_kth_largest_shuffle_as_the_threshold_and_ties_count_as_found():
    assert injection.counts_by_rank([5, 7, 9], [4.0, 6.0, 8.0], ranks=[1, 2, 3]) == {1: 1, 2: 2, 3: 3}
    assert injection.counts_by_rank([6.0], [6.0, 5.0], ranks=[1]) == {1: 1}
    assert injection.counts_by_rank([], [6.0], ranks=[1]) == {1: 0}
