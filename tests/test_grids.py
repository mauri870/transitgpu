"""The grids copied from transitleastsquares 2.0.1 (transitgpu/grids.py) equal the installed TLS to the bit. TLS is a test dependency of the package and not a dependency at run time. Function tests: the checks are on the
equality of two implementations over a range of inputs, which an archive of one light curve does not hold."""

import warnings

import numpy as np
import pytest

tls_grid = pytest.importorskip("transitleastsquares.grid")
tls_stats = pytest.importorskip("transitleastsquares.stats")
tls_constants = pytest.importorskip("transitleastsquares.tls_constants")

from transitgpu import grids  # noqa: E402

CONSTANTS = ("G", "R_sun", "R_jup", "M_sun", "SECONDS_PER_DAY", "R_STAR", "M_STAR", "OVERSAMPLING_FACTOR", "N_TRANSITS_MIN", "M_STAR_MIN", "M_STAR_MAX", "R_STAR_MIN", "R_STAR_MAX", "DURATION_GRID_STEP",
             "FRACTIONAL_TRANSIT_DURATION_MAX", "MINIMUM_PERIOD_GRID_SIZE")


def test_the_constants_are_those_of_tls():
    assert {name: getattr(grids, name) for name in CONSTANTS} == {name: getattr(tls_constants, name) for name in CONSTANTS}


@pytest.mark.parametrize("span", [3.0, 27.0, 400.0, 1979.0, 2720.58])
@pytest.mark.parametrize("period_min, period_max", [(0.5, 14.0), (0.5, 906.86), (2.0, 7.0), (0.0, float("inf"))])
@pytest.mark.parametrize("oversampling, n_transits_min", [(3, 2), (1, 2), (5, 3)])
def test_the_period_grid_is_the_grid_of_tls(span, period_min, period_max, oversampling, n_transits_min):
    arguments = dict(R_star=1.0, M_star=1.0, time_span=span, period_min=period_min, period_max=period_max, oversampling_factor=oversampling, n_transits_min=n_transits_min)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            expected = tls_grid.period_grid(**arguments)
        except ValueError as error:
            with pytest.raises(ValueError, match=str(error)[:20]):
                grids.period_grid(**arguments)
            return
        found = grids.period_grid(**arguments)
    assert found.dtype == expected.dtype and found.tobytes() == expected.tobytes()


@pytest.mark.parametrize("radius, mass", [(0.5, 0.5), (1.0, 1.0), (3.0, 2.0), (0.005, 0.005), (20000, 5000)])
def test_the_period_grid_of_other_stars_and_of_unphysical_ones_is_the_grid_of_tls_with_its_clamps(radius, mass):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = tls_grid.period_grid(R_star=radius, M_star=mass, time_span=300.0, period_min=0.5, period_max=50.0)
        found = grids.period_grid(R_star=radius, M_star=mass, time_span=300.0, period_min=0.5, period_max=50.0)
    assert found.tobytes() == expected.tobytes()


def test_the_fractional_duration_is_the_one_of_tls_to_the_bit_for_both_planet_sizes():
    """TLS compiles T14 with fastmath; the plain Python arithmetic differs from it in the last digits for most inputs (32224 of 40000 in a trial), so the copy keeps the compilation."""
    rng = np.random.default_rng(0)
    for _ in range(3000):
        period = float(np.exp(rng.uniform(np.log(0.3), np.log(2000))))
        radius, mass = float(rng.uniform(0.1, 4.0)), float(rng.uniform(0.08, 1.2))
        for small in (True, False):
            assert np.float64(grids.T14(radius, mass, period, small=small)).tobytes() == np.float64(tls_grid.T14(radius, mass, period, small=small)).tobytes()


@pytest.mark.parametrize("limits", [(0.13, 3.5, 0.1, 1.0), (0.5, 1.5, 0.5, 1.2), (0.9, 1.1, 0.95, 1.05), (0.13, 3.5, 0.5, 0.6), (2.0, 3.0, 0.1, 0.2), (1.0, 1.0, 1.0, 1.0)])
def test_the_masses_that_set_the_duration_limits_are_those_of_tls_for_default_narrow_and_equal_limits(limits):
    assert grids.duration_limit_masses(*limits) == tls_grid.duration_limit_masses(*limits)


def test_the_duration_fractions_of_the_search_are_those_of_a_search_built_on_tls_functions():
    from transitgpu import core

    periods = tls_grid.period_grid(R_star=1.0, M_star=1.0, time_span=2720.58, period_min=0.5, period_max=906.86)
    shortest, longest = core.duration_fractions(periods)
    m_short, m_long = tls_grid.duration_limit_masses(tls_constants.R_STAR_MIN, tls_constants.R_STAR_MAX, tls_constants.M_STAR_MIN, tls_constants.M_STAR_MAX)
    assert shortest.tobytes() == np.array([tls_grid.T14(R_s=tls_constants.R_STAR_MIN, M_s=m_short, P=p, small=True) for p in periods]).tobytes()
    assert longest.tobytes() == np.array([tls_grid.T14(R_s=tls_constants.R_STAR_MAX, M_s=m_long, P=p, small=False) for p in periods]).tobytes()


def test_the_uncertainty_of_a_peak_is_that_of_tls_and_infinite_when_the_peak_is_not_bracketed():
    rng = np.random.default_rng(4)
    periods = np.linspace(1, 20, 4000)
    for center in (3.0, 10.0, 19.9, 1.01):
        power = np.exp(-0.5 * ((periods - center) / 0.05) ** 2) + 0.01 * rng.random(periods.size)
        assert grids.period_uncertainty(periods, power) == tls_stats.period_uncertainty(periods, power)
    flat_top = np.ones(periods.size)
    assert grids.period_uncertainty(periods, flat_top) == tls_stats.period_uncertainty(periods, flat_top) == float("inf")
    assert grids.period_uncertainty(periods, np.arange(periods.size, dtype=float)) == float("inf"), "a peak at the end is not bracketed"
