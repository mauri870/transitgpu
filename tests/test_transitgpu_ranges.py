"""One box search for two ranges of period: the grid to 14 days is the start of the grid to a third of the baseline, so one search over the longer grid gives the statistic of every period
of both, and the SDE of each range is its z-score over that range's periods.

A function test of two paths that must agree to the bit (a differential test): the short range read from the one search against the search of the short range alone, and the long range against the search of the long range.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from transitgpu.core import grid_of, search, search_many

pytestmark = pytest.mark.gpu  # every test of this module may search on the device


def stack():
    """Three sectors of 8 days, 40 days apart, at 30 minutes (1,152 points, about 8,000 trial periods to a third of the baseline: a stack of 3 sectors of 27 days 400 days apart has 100,000 and took 5 minutes a test on a CPU), a planet of 3.18 days and one of 40 days, in 400 ppm of noise."""
    time = np.concatenate([np.arange(0.0, 8.0, 1 / 48), np.arange(40.0, 48.0, 1 / 48), np.arange(80.0, 88.0, 1 / 48)])
    flux = 1 + np.random.default_rng(4).normal(0, 400e-6, time.size)
    for period, t0, depth in ((3.18, 1.2, 1500e-6), (40.0, 5.0, 3000e-6)):
        flux[np.abs(time - (t0 + np.round((time - t0) / period) * period)) < 2.5 / 48] -= depth
    return time, flux


def test_the_short_range_of_one_search_is_the_search_of_the_short_range_to_the_bit():
    time, flux = stack()
    span = float(time.max() - time.min())
    short_grid, long_grid = grid_of(time, flux, 0.5, 14.0), grid_of(time, flux, 0.5, span / 3)
    np.testing.assert_array_equal(long_grid[long_grid <= short_grid.max()], short_grid)
    ((short, long),) = search_many(time, [flux], 0.5, span / 3, spectrum=True, periods=long_grid, cuts=(float(short_grid.max()), float(long_grid.max())))
    alone_short = search(time, flux, 0.5, 14.0, spectrum=True, periods=short_grid)
    alone_long = search(time, flux, 0.5, span / 3, spectrum=True, periods=long_grid)
    for one, alone in ((short, alone_short), (long, alone_long)):
        np.testing.assert_array_equal(one["periods"], alone["periods"])
        np.testing.assert_array_equal(one["power"], alone["power"])
        assert {key: one[key] for key in alone if key not in ("periods", "power")} == {key: alone[key] for key in alone if key not in ("periods", "power")}
    assert short["period_max_searched_days"] == float(short_grid.max()) and long["period_max_searched_days"] == float(long_grid.max())


def test_many_fluxes_and_two_cuts_give_a_result_for_each_flux_and_each_cut_and_a_cut_must_end_on_the_grid():
    time, flux = stack()
    span = float(time.max() - time.min())
    grid = grid_of(time, flux, 0.5, span / 3)
    found = search_many(time, [flux, flux[::-1].copy()], 0.5, span / 3, periods=grid, cuts=(14.0, float(grid.max())))
    assert len(found) == 2 and all(len(pair) == 2 for pair in found)
    assert found[0][0]["period_max_searched_days"] <= 14.0
    with pytest.raises(ValueError, match="cut"):
        search_many(time, [flux], 0.5, span / 3, periods=grid, cuts=(0.3,))


def strip(found):
    return {key: value for key, value in found.items() if key not in ("periods", "power")}
