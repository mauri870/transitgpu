"""The stellar limits of the duration grid: `radius_limits` and `mass_limits` of `duration_fractions`, `search_many` and `search` set the shortest and the longest transit duration of each period.

These are function tests and not archives: the properties are that the default path is the search as it was, and that a lowered floor keeps every default duration on the lattice of the grid and adds
durations below it, which a recipe of one synthetic star does not say.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve

from transitgpu import core, grids

PERIODS = np.array([0.5, 1.0, 3.0, 7.0, 14.0, 20.0, 100.0, 243.0])
DEFAULT_RADIUS = (grids.R_STAR_MIN, grids.R_STAR_MAX)
DEFAULT_MASS = (grids.M_STAR_MIN, grids.M_STAR_MAX)


def test_the_limits_of_the_default_stars_give_the_default_durations_bit_for_bit():
    shortest, longest = core.duration_fractions(PERIODS)
    given_shortest, given_longest = core.duration_fractions(PERIODS, radius_limits=DEFAULT_RADIUS, mass_limits=DEFAULT_MASS)
    assert shortest.tobytes() == given_shortest.tobytes() and longest.tobytes() == given_longest.tobytes()


def test_a_smaller_minimum_radius_lowers_the_shortest_duration_in_proportion_and_leaves_the_longest():
    shortest, longest = core.duration_fractions(PERIODS)
    lowered_shortest, lowered_longest = core.duration_fractions(PERIODS, radius_limits=(grids.R_STAR_MIN / 2, grids.R_STAR_MAX))
    np.testing.assert_allclose(lowered_shortest, shortest / 2, rtol=1e-12)
    assert lowered_longest.tobytes() == longest.tobytes()


def test_a_lower_minimum_mass_with_the_default_radius_does_not_lower_the_shortest_duration_below_the_default_corner():
    """T14 grows as the mass falls, and the default corner (0.13 solar radii, 0.1 solar masses) is already the least massive: a mass limit below it is not a shorter transit."""
    shortest, _ = core.duration_fractions(PERIODS)
    lowered, _ = core.duration_fractions(PERIODS, mass_limits=(grids.M_STAR_MIN / 2, grids.M_STAR_MAX))
    assert np.all(lowered >= shortest)


@pytest.mark.parametrize("bad", [{"radius_limits": (0.0, 3.5)}, {"radius_limits": (2.0, 1.0)}, {"mass_limits": (-0.1, 1.0)}, {"mass_limits": (1.0, 0.5)}, {"radius_limits": (0.1,)}])
def test_limits_that_are_not_an_increasing_pair_of_positive_numbers_raise(bad):
    with pytest.raises(ValueError, match="limits"):
        core.duration_fractions(PERIODS, **bad)


def test_a_search_with_the_default_limits_given_equals_the_search_without_them_bit_for_bit():
    time, flux = light_curve(seed=3, cadence_min=30.0, duration_hr=6.0, depth_ppm=6000)
    plain = core.search_many(time, [flux], 0.5, 14, spectrum=True, boxes=True, device="cpu")[0]
    given = core.search_many(time, [flux], 0.5, 14, spectrum=True, boxes=True, device="cpu", radius_limits=DEFAULT_RADIUS, mass_limits=DEFAULT_MASS)[0]
    for name in plain:
        assert np.asarray(plain[name]).tobytes() == np.asarray(given[name]).tobytes(), name


def test_search_passes_the_limits_to_search_many():
    time, flux = light_curve(seed=3, cadence_min=30.0, duration_hr=1.0, depth_ppm=6000)
    one = core.search(time, flux, 0.5, 14, spectrum=True, device="cpu", radius_limits=(0.05, 3.5))
    many = core.search_many(time, [flux], 0.5, 14, spectrum=True, device="cpu", radius_limits=(0.05, 3.5))[0]
    assert np.asarray(one["power"]).tobytes() == np.asarray(many["power"]).tobytes()


def test_a_lowered_floor_keeps_every_default_duration_on_its_lattice_and_adds_durations_below_them():
    """The durations of a period are a geometric sequence of ratio `duration_step`. With the floor lowered, the default durations stay in the sequence (the same numbers) and the steps below are new."""
    default = core.duration_lattice(PERIODS, 1.1)
    lowered = core.duration_lattice(PERIODS, 1.1, radius_limits=(grids.R_STAR_MIN / 3, grids.R_STAR_MAX))
    for period_index in range(PERIODS.size):
        kept = lowered[period_index]
        assert set(default[period_index].tolist()) <= set(kept.tolist()), f"period {PERIODS[period_index]}: a default duration is missing from the lowered grid"
        added = [value for value in kept.tolist() if value not in set(default[period_index].tolist())]
        assert added and max(added) < default[period_index].min(), f"period {PERIODS[period_index]}: the added durations are not all under the default shortest"


@pytest.mark.parametrize("steps_below", [-1, -2, -3, -5])
def test_a_floor_set_exactly_on_a_lattice_point_starts_the_grid_at_that_point(steps_below):
    """Floating point must not move the first duration one step off when the floor lands on the lattice: the radius limit is the default one times step^k, so the shortest duration is the anchor times step^k.
    (A radius limit above the default does not raise the floor: the mass paired with it keeps the density of the default corner, see grids.duration_limit_masses.)"""
    step = 1.1
    default = core.duration_lattice(PERIODS, step)
    limit = grids.R_STAR_MIN * step**steps_below
    lattice = core.duration_lattice(PERIODS, step, radius_limits=(limit, grids.R_STAR_MAX))
    for index in range(PERIODS.size):
        expected_first = default[index][0] * step**steps_below
        assert lattice[index][0] == pytest.approx(expected_first, rel=1e-9), f"period {PERIODS[index]}: the grid starts at {lattice[index][0]}, not at the lattice point {expected_first}"
        assert len(lattice[index]) == len(default[index]) - steps_below, f"period {PERIODS[index]}: {len(lattice[index])} steps against {len(default[index]) - steps_below}"


def one_period_search(period, **limits):
    time = np.linspace(0.0, 4.0 * period, 3000)
    flux = 1.0 + np.random.default_rng(3).normal(0.0, 1e-3, time.size)
    return core.search_many(time, [flux], period * 0.99, period * 1.01, periods=np.array([period, period * 1.01]), device="cpu", **limits)


def test_a_lowered_floor_under_half_a_bin_at_a_long_period_warns_once_that_the_durations_collapse_onto_one_bin():
    """Review of 0.2.0 (F2): the phase bins of a period stop at MAXIMUM_BINS, so at 100 d a bin is 35 minutes, and the shortest durations of a lowered floor all round to one bin: the floor is not honoured and nothing said so. A requested duration under half a bin is scored as one bin, more than twice what was asked."""
    import warnings

    with pytest.warns(UserWarning, match="shorter than half a phase bin") as caught:
        one_period_search(100.0, radius_limits=(0.005, grids.R_STAR_MAX))
    assert len([w for w in caught if "shorter than half a phase bin" in str(w.message)]) == 1, "one warning for the call, not one for each batch"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for period in (1.0, 14.0, 100.0, 1000.0):
            one_period_search(period)  # the default limits give no duration under half a bin at any period of the search
        one_period_search(14.0, radius_limits=(grids.R_STAR_MIN / 2, grids.R_STAR_MAX))


def test_the_count_of_durations_under_half_a_bin_is_the_same_without_and_with_a_duration_window():
    """The warning counts the short durations from the first steps of each period when there is no window (a table of periods by steps took 18 ms of a search of 0.2 s) and from the table with one. A window that keeps every duration must give the same count."""
    import warnings

    periods = np.array([50.0, 100.0, 300.0, 1000.0])
    exponents = core.duration_exponents(periods, core.DURATION_STEP, radius_limits=(0.005, grids.R_STAR_MAX))
    _, anchor, offset, steps_of = exponents
    bins_of = np.full(periods.size, core.MAXIMUM_BINS)
    messages = []
    for window in (None, (0.0, 1e9)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            core._warn_durations_under_half_a_bin(periods, bins_of, anchor, offset, steps_of, core.DURATION_STEP, window)
        messages.append(str(caught[0].message))
    assert messages[0] == messages[1], messages


def test_the_two_paths_of_the_count_agree_on_a_grid_to_long_periods_and_where_half_a_bin_is_an_exact_power_of_the_step():
    """Rule 20: without a duration window the count comes from the first steps of each period, with one from the table of periods by steps (starlord-20's review of 2d2dd3e). Where 0.5 over the shortest duration in bins is an exact power of the step, a count by floor(log) + 1 takes one step that the strict `< 0.5` of the table does not."""
    import warnings

    def message(periods, bins_of, anchor, offset, steps_of, step, window):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            core._warn_durations_under_half_a_bin(periods, bins_of, anchor, offset, steps_of, step, window)
        return [str(w.message) for w in caught]

    periods = np.geomspace(10.0, 2000.0, 60)
    _, anchor, offset, steps_of = core.duration_exponents(periods, core.DURATION_STEP, radius_limits=(0.002, grids.R_STAR_MAX))
    bins_of = np.full(periods.size, core.MAXIMUM_BINS)
    plain = message(periods, bins_of, anchor, offset, steps_of, core.DURATION_STEP, None)
    windowed = message(periods, bins_of, anchor, offset, steps_of, core.DURATION_STEP, (0.0, float("inf")))
    assert plain and plain == windowed, (plain, windowed)
    step = 2.0  # exact in binary: the shortest duration of the first period is 0.0625 bins, and 0.5 / 0.0625 = 2 ** 3
    exact = dict(periods=np.array([100.0, 200.0]), bins_of=np.full(2, 4096), anchor=np.full(2, 0.0625 / 4096), offset=np.zeros(2), steps_of=np.full(2, 10))
    plain = message(step=step, window=None, **exact)
    windowed = message(step=step, window=(0.0, float("inf")), **exact)
    assert plain and plain == windowed, (plain, windowed)
    assert plain[0].startswith("6 of the durations"), "3 steps below half a bin (0.0625, 0.125, 0.25) at each of the two periods, not 4: 0.5 is not shorter than 0.5"
