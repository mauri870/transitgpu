"""Dips and bumps from one fold: `search_many(..., bumps=True)` scores the windows whose mean is above the whole's from the same sums, grid and statistic as the dips, and puts the result in the `bump` entry of each result.

A function test of two paths that must agree (a differential test): the bump entry against a search of the negated flux, and the dip fields of a search with bumps against the search without. The tolerance of the first was
written before the first run: periods, epochs, durations, transit counts and the boxes of every period are equal; the SDE, the power, the depth and the SNR are equal to a relative 1e-12.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from transitgpu.core import grid_of, search, search_many

pytestmark = pytest.mark.gpu  # every test of this module may search on the device


@pytest.fixture(autouse=True)
def fresh_compiler(monkeypatch):
    """Each test starts with an empty cache of compiled scorers and room for every variant it uses. After more variants than torch's cache limit (8) a process runs the scorer eagerly, whose float32 statistic differs from the compiled one in the 7th digit, and the equality between two searches that the tests state is of two searches made the same way (the package's README says so of long processes)."""
    import torch

    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 64)

EXACT = ("period_days", "t0_days", "duration_hr", "duration_corrected_hr", "fill_factor", "transit_count", "period_min_searched_days", "period_max_searched_days", "period_uncertainty_days")
CLOSE = ("sde", "depth_ppm", "depth_mean_ppm", "snr")
RELATIVE = 1e-12


def stack(sign):
    """Three sectors of 8 days, 40 days apart, at 30 minutes (1,152 points), a 3.18 day signal of 2.5 hours and 1,500 ppm, a dip for sign -1 and a bump for sign +1, in 400 ppm of noise."""
    time = np.concatenate([np.arange(0.0, 8.0, 1 / 48), np.arange(40.0, 48.0, 1 / 48), np.arange(80.0, 88.0, 1 / 48)])
    flux = 1 + np.random.default_rng(4).normal(0, 400e-6, time.size)
    flux[np.abs(time - (1.2 + np.round((time - 1.2) / 3.18) * 3.18)) < 2.5 / 48] += sign * 1500e-6
    return time, flux


def same(found, other):
    for key in EXACT:
        assert found[key] == other[key], key
    for key in CLOSE:
        np.testing.assert_allclose(found[key], other[key], rtol=RELATIVE, atol=0, err_msg=key)
    for key in ("periods", "centers", "widths_days"):
        if key in other:
            np.testing.assert_array_equal(found[key], other[key], err_msg=key)
    if "power" in other:
        np.testing.assert_allclose(found["power"], other["power"], rtol=RELATIVE, atol=0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_the_bump_entry_is_the_search_of_the_negated_flux_and_the_dip_fields_are_the_search_without_bumps(sign):
    time, flux = stack(sign)
    both = search(time, flux, 0.5, 14.0, spectrum=True, boxes=True, bumps=True)
    plain = search(time, flux, 0.5, 14.0, spectrum=True, boxes=True)
    negated = search(time, -flux, 0.5, 14.0, spectrum=True, boxes=True)
    assert "bump" not in plain and set(both) - {"bump"} == set(plain)
    for key, value in plain.items():
        np.testing.assert_array_equal(both[key], value, err_msg=f"the dip field {key} changed")  # bit-identical
    same(both["bump"], negated)
    found = both["bump"] if sign == 1 else both
    assert abs(found["period_days"] - 3.18) < 0.02, "the injected signal is found by its own side"


def test_a_bump_star_has_a_larger_bump_sde_than_dip_sde_and_a_dip_star_the_reverse():
    for sign in (-1, 1):
        time, flux = stack(sign)
        found = search(time, flux, 0.5, 14.0, bumps=True)
        own, other = (found, found["bump"]) if sign == -1 else (found["bump"], found)
        assert own["sde"] > other["sde"] + 3, (sign, own["sde"], other["sde"])


def test_many_fluxes_with_cuts_weights_and_a_missing_point_give_each_bump_entry_as_the_search_of_the_negated_flux():
    time, flux = stack(1)
    second = flux[::-1].copy()
    second[::7] = np.nan  # points with a time and no flux have no weight
    weights = np.random.default_rng(5).uniform(0.5, 2.0, time.size)
    span = float(time.max() - time.min())
    grid = grid_of(time, flux, 0.5, span / 3)
    cuts = (14.0, float(grid.max()))
    both = search_many(time, [flux, second], 0.5, span / 3, spectrum=True, periods=grid, cuts=cuts, weights=weights, bumps=True)
    plain = search_many(time, [flux, second], 0.5, span / 3, spectrum=True, periods=grid, cuts=cuts, weights=weights)
    negated = search_many(time, [-flux, -second], 0.5, span / 3, spectrum=True, periods=grid, cuts=cuts, weights=weights)
    for k in range(2):
        for cut in range(2):
            for key, value in plain[k][cut].items():
                np.testing.assert_array_equal(both[k][cut][key], value, err_msg=f"flux {k} cut {cut}: the dip field {key} changed")
            same(both[k][cut]["bump"], negated[k][cut])


def test_a_flux_with_no_window_above_its_mean_still_has_a_bump_entry_equal_to_the_negated_search():
    """The edge: every window of the box widths is a dip or empty (a single deep dip on a flat line, the bump statistic zero at most periods), so the bump z-scores come from a mostly constant row."""
    time = np.arange(0.0, 30.0, 1 / 24)
    flux = np.ones(time.size)
    flux[100:106] -= 0.01
    flux = flux + np.random.default_rng(6).normal(0, 1e-5, time.size)
    both = search(time, flux, 0.5, 8.0, spectrum=True, bumps=True)
    same(both["bump"], search(time, -flux, 0.5, 8.0, spectrum=True))


def test_box_search_power_with_bumps_has_the_results_of_the_bumps_as_bump_and_without_it_has_none():
    """The interface of TLS's results: `bump` is a Results whose period, SDE and depth are those of the search of the negated flux, with a depth above 1 for a brightening."""
    from transitgpu import box_search

    time, flux = stack(1)
    model = box_search(time, flux)
    plain = model._power(period_min=0.5, period_max=14.0)  # power without the check for a GPU, so that the test also runs on a CPU
    both = model._power(period_min=0.5, period_max=14.0, bumps=True)
    assert plain.bump is None and both.bump is not None and both.bump.bump is None
    np.testing.assert_array_equal(both.power, plain.power)
    negated = box_search(time, -flux)._power(period_min=0.5, period_max=14.0)
    assert both.bump.period == negated.period and both.bump.SDE == negated.SDE
    assert both.bump.depth > 1 and both.bump.model_folded_model.max() == both.bump.depth, "a brightening has a depth above 1 and the model is that box"
