"""The interface modelled on TLS's: box_search(t, y, dy)._power(...) returns the attributes of TLS's results and is transitgpu.core.search_many under another name, to the bit. Function tests: the checks are on the names,
the units and the equality of two call paths."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve  # noqa: E402

import transitgpu  # noqa: E402
from transitgpu import core  # noqa: E402

pytestmark = pytest.mark.gpu


def test_the_results_have_the_attributes_and_units_of_tls_for_an_injected_box():
    time, flux = light_curve(period=3.18, t0=1.2, duration_hr=2.5, depth_ppm=3000, noise_ppm=500)
    results = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14)
    assert abs(results.period / 3.18 - 1) < 0.01
    assert 0.99 < results.depth < 0.9985 and results.depth_mean == results.depth, "TLS's depth is the mean flux in the transit: 0.997 for a dip of 3000 ppm"
    assert 1.5 / 24 < results.duration < 4 / 24, "the duration is in days"
    assert len(results.periods) == len(results.power) > 1000 and results.SDE == results.SDE_raw == pytest.approx(float(results.power.max()), rel=1e-5)
    assert results.transit_count >= 7 and results.snr > 5 and results.period_uncertainty > 0
    assert results.box is None, "the boxes of every period are given only when asked"


def test_the_interface_is_the_search_of_the_package_to_the_bit_with_and_without_errors():
    time, flux = light_curve()
    plain = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14)
    direct = core.search(time, flux, 0.5, 14, spectrum=True)
    assert np.array_equal(plain.power, direct["power"]) and plain.period == direct["period_days"] and plain.SDE == direct["sde"]
    dy = 1e-3 * (1 + np.random.default_rng(2).random(time.size))
    weighted = transitgpu.box_search(time, flux, dy)._power(period_min=0.5, period_max=14)
    direct_weighted = core.search(time, flux, 0.5, 14, spectrum=True, weights=1 / dy**2)
    assert np.array_equal(weighted.power, direct_weighted["power"]) and weighted.period == direct_weighted["period_days"]


def test_oversampling_factor_and_the_duration_step_reach_the_search_and_the_defaults_are_tls_s():
    time, flux = light_curve()
    default = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14)
    coarse = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14, oversampling_factor=1)
    assert len(coarse.periods) < len(default.periods) < 4 * len(coarse.periods) + 1000
    wide = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14, duration_grid_step=1.5)
    assert np.array_equal(wide.power, core.search(time, flux, 0.5, 14, spectrum=True, duration_step=1.5)["power"])
    assert np.array_equal(default.periods, core.grid_of(time, flux, 0.5, 14)), "the default grid is the grid of the package, which is TLS's"


def test_the_boxes_of_every_period_are_given_when_asked_and_a_period_max_left_out_is_half_the_span():
    time, flux = light_curve()
    results = transitgpu.box_search(time, flux)._power(period_min=0.5, boxes=True)
    assert results.box["centers"].shape == results.periods.shape == results.box["widths_days"].shape
    assert results.period_max_searched <= (time.max() - time.min()) / 2


def test_input_that_does_not_fit_together_is_refused_by_name():
    time, flux = light_curve()
    with pytest.raises(ValueError, match="one length"):
        transitgpu.box_search(time, flux[:-1])
    with pytest.raises(ValueError, match="dy must have the shape of t"):
        transitgpu.box_search(time, flux, dy=np.ones(time.size - 2))


def test_the_results_fold_the_light_curve_at_the_best_period_and_give_the_box_that_was_found_with_the_names_of_tls():
    """folded_phase, folded_y, model_folded_phase and model_folded_model are TLS's names: the phase runs from 0 to 1 with the transit at 0.5, the flux is in the order of the phase and the box is 1 outside its duration and `depth` inside."""
    time, flux = light_curve(period=3.18, t0=1.2, duration_hr=2.5, depth_ppm=3000, noise_ppm=500)
    flux[5] = float("nan")
    results = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14)
    assert results.folded_phase.min() >= 0 and results.folded_phase.max() <= 1 and (np.diff(results.folded_phase) >= 0).all() and len(results.folded_y) == len(results.folded_phase) == len(time) - 1, "the point with no flux is left out"
    half = results.duration / results.period / 2
    inside = np.abs(results.folded_phase - 0.5) < half * 0.9
    assert results.folded_y[inside].mean() < results.folded_y[~inside & (np.abs(results.folded_phase - 0.5) > 2 * half)].mean() - 0.002
    assert results.model_folded_model.min() == results.depth and results.model_folded_model.max() == 1.0
    assert (results.model_folded_model[np.abs(results.model_folded_phase - 0.5) <= half] == results.depth).all() and (results.model_folded_model[np.abs(results.model_folded_phase - 0.5) > half + 1e-9] == 1.0).all()


def test_the_public_call_raises_without_a_device_and_names_astropys_box_least_squares(monkeypatch):
    """transitgpu searches on a GPU: box_search(...).power() raises without one, or for the device "cpu", and names the library to use on a CPU. The search itself on a CPU stays callable under a private name (`_power`), which these tests use as the oracle."""
    import torch

    time, flux = light_curve(2000, 60.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for options in ({}, {"device": "cpu"}):
        with pytest.raises(RuntimeError, match="BoxLeastSquares"):
            transitgpu.box_search(time, flux).power(period_min=0.5, period_max=14, **options)
    assert transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14).period > 0, "the private oracle searches on a CPU"


@pytest.mark.gpu
def test_the_public_call_searches_on_the_device_and_gives_the_results_of_the_private_oracle():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no device")
    time, flux = light_curve(3000, 60.0)
    public = transitgpu.box_search(time, flux).power(period_min=0.5, period_max=14)
    oracle = transitgpu.box_search(time, flux)._power(period_min=0.5, period_max=14, device="cpu")
    assert abs(public.period / oracle.period - 1) < 1e-3 and abs(public.SDE - oracle.SDE) < 0.05 * abs(oracle.SDE)
