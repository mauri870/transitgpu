"""The search of many fluxes of one time axis at once (transitgpu.search_many): the fold of the points into phase bins depends on the times alone, and a block shuffle of a star only moves flux between
the times, so the fold is computed once for all the shuffles of a null. The single search is the batch of one.

These are function tests and not archives: the property is that two paths agree on random inputs (and their error paths), which a recipe of one synthetic star does not say.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu import core as gpu  # noqa: E402
from transitgpu.core import grid_of, search, search_many  # noqa: E402

from test_transitgpu import light_curve  # noqa: E402

pytestmark = pytest.mark.gpu  # every test of this module may search on the device

FIELDS = (
    "period_min_searched_days", "period_max_searched_days", "period_days", "period_uncertainty_days", "t0_days", "duration_hr", "duration_corrected_hr", "fill_factor", "depth_ppm", "depth_mean_ppm", "sde",
    "snr", "transit_count",
)


def same(found, expected, relative=1e-12):
    assert set(found) == set(expected)
    for name in found:
        if name in ("periods", "power"):
            np.testing.assert_allclose(found[name], expected[name], rtol=relative, atol=1e-12, err_msg=name)
        else:
            assert found[name] == pytest.approx(expected[name], rel=relative, abs=1e-12), name


def test_a_search_with_no_limit_on_the_durations_equals_one_whose_limits_cover_every_duration():
    """A search restricted to a band of hours scores only the duration steps in it; a band that covers every step is the search as it is."""
    time, flux = light_curve(seed=3, cadence_min=30.0, duration_hr=6.0, depth_ppm=6000)
    plain = search_many(time, [flux], 3.0, 5.0, spectrum=True, boxes=True)[0]
    covered = search_many(time, [flux], 3.0, 5.0, spectrum=True, boxes=True, duration_hours=(0.0, 1e9))[0]
    for key, value in plain.items():
        assert np.array_equal(np.asarray(value), np.asarray(covered[key]), equal_nan=True), key


def test_a_search_limited_to_a_band_of_durations_places_every_best_box_inside_the_band():
    time, flux = light_curve(seed=3, cadence_min=30.0, duration_hr=6.0, depth_ppm=6000)
    found = search_many(time, [flux], 3.0, 5.0, spectrum=True, boxes=True, duration_hours=(3.0, 6.0))[0]
    hours = found["widths_days"] * 24
    assert hours.min() > 3.0 / 1.3 and hours.max() <= 6.0 * 1.3, (hours.min(), hours.max())
    wide = search_many(time, [flux], 3.0, 5.0, spectrum=True, boxes=True)[0]
    assert (wide["widths_days"] * 24).min() < 3.0, "the unlimited search scores shorter durations, which the band leaves out"


def test_a_band_of_durations_that_no_period_reaches_is_searched_without_an_error_and_finds_nothing():
    time, flux = light_curve(seed=3, cadence_min=30.0)
    found = search_many(time, [flux], 3.0, 5.0, spectrum=True, duration_hours=(1000.0, 2000.0))[0]
    assert found["sde"] == 0.0 and not found["power"].any() and found["depth_ppm"] == 0.0
