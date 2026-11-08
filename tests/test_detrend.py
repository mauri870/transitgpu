"""transitgpu.detrend against wotan's biweight (wotan 1.10), on the CPU, and the Triton kernel against the torch version where a CUDA or ROCm device and Triton are present.

The edge cases are the rules of wotan's `flatten` that the trend depends on (the segments, the half-open window, the last point of a segment, the mask and its drop of values at or below -1e-12, the median absolute deviation of 0).
"""
import importlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
wotan = pytest.importorskip("wotan")

from transitgpu import detrend

WINDOW = 0.75
RELATIVE = 1e-9  # the trend of each agrees with wotan's to rounding (1e-15 on these inputs); a step of the iteration at its stopping step of 1e-6 would show as 1e-6


def curve(points=3000, step=2 / 1440, seed=0, gaps=((1.0, 1.4), (1.8, 1.9)), nans=7):
    """A regular light curve of a sine, a box transit and white noise, with two gaps (one longer than WINDOW / 2 and one shorter) and a few NaN."""
    rng = np.random.default_rng(seed)
    time = np.arange(points) * step
    flux = 1 + 0.002 * np.sin(2 * np.pi * time / 1.7) + rng.normal(0, 0.0005, points)
    flux[(time % 2.3 > 1.0) & (time % 2.3 < 1.06)] -= 0.004
    keep = np.ones(points, dtype=bool)
    for low, high in gaps:
        keep &= ~((time >= low) & (time < high))
    flux[rng.choice(points, nans, replace=False)] = np.nan
    return time[keep], flux[keep]


def reference(time, flux, mask=None):
    return wotan.flatten(time, flux, method="biweight", window_length=WINDOW, return_trend=True, mask=mask)[1]


def assert_same(ours, theirs):
    assert np.array_equal(np.isnan(ours), np.isnan(theirs)), "the positions of NaN differ"
    both = ~np.isnan(theirs)
    np.testing.assert_allclose(ours[both], theirs[both], rtol=RELATIVE, atol=0)


def test_the_trend_of_a_light_curve_with_gaps_and_nan_is_wotans():
    time, flux = curve()
    assert_same(detrend.biweight(time, flux, WINDOW, device="cpu"), reference(time, flux))


def test_the_trend_with_a_mask_of_known_transits_is_wotans():
    time, flux = curve(seed=1)
    mask = (time % 2.3 > 1.0) & (time % 2.3 < 1.06)
    assert_same(detrend.biweight(time, flux, WINDOW, mask=mask, device="cpu"), reference(time, flux, mask))


def test_a_mask_that_empties_a_window_gives_nan_where_wotan_does():
    time = np.arange(40) * 0.01
    flux = 1 + 0.001 * np.arange(40) % 3
    mask = np.zeros(40, dtype=bool)
    mask[:] = True
    ours = detrend.biweight(time, flux, 0.1, mask=mask, device="cpu")
    theirs = wotan.flatten(time, flux, method="biweight", window_length=0.1, return_trend=True, mask=mask)[1]
    assert np.isnan(theirs).all() and np.isnan(ours).all()


def test_a_flux_with_no_spread_has_its_median_for_trend():
    time = np.arange(300) * 0.01
    flux = np.full(300, 1.25)
    assert_same(detrend.biweight(time, flux, 0.5, device="cpu"), wotan.flatten(time, flux, method="biweight", window_length=0.5, return_trend=True)[1])


@pytest.mark.parametrize("points", [1, 2, 3])
def test_a_segment_of_one_two_or_three_points_has_wotans_trend(points):
    time = np.concatenate([np.arange(200) * 0.01, 10 + np.arange(points) * 0.01])
    flux = 1 + 0.01 * np.sin(np.arange(200 + points))
    assert_same(detrend.biweight(time, flux, 0.5, device="cpu"), wotan.flatten(time, flux, method="biweight", window_length=0.5, return_trend=True)[1])


def test_the_last_point_of_a_segment_is_in_no_window():
    time = np.arange(20) * 0.01
    flux = np.ones(20)
    flux[-1] = 5.0
    trend = detrend.biweight(time, flux, 0.1, device="cpu")
    assert trend[-1] == 1.0 and trend[-2] == 1.0, trend[-3:]
    assert_same(trend, wotan.flatten(time, flux, method="biweight", window_length=0.1, return_trend=True)[1])


@pytest.mark.parametrize("gap", [0.0499, 0.05, 0.0501])
def test_a_gap_of_half_the_window_splits_the_light_curve_where_wotan_splits_it(gap):
    first = np.arange(60) * 0.01
    time = np.concatenate([first, first[-1] + gap + np.arange(60) * 0.01])
    flux = 1 + 0.01 * np.sin(np.arange(120) / 3)
    assert_same(detrend.biweight(time, flux, 0.1, device="cpu"), wotan.flatten(time, flux, method="biweight", window_length=0.1, return_trend=True)[1])


def test_values_at_or_below_minus_1e_12_are_in_no_window_as_in_wotan():
    time = np.arange(100) * 0.01
    flux = 0.001 * np.sin(np.arange(100))
    flux[10:20] = -1.0
    assert_same(detrend.biweight(time, flux, 0.3, device="cpu"), wotan.flatten(time, flux, method="biweight", window_length=0.3, return_trend=True)[1])


def test_a_window_longer_than_the_light_curve_is_one_window():
    time, flux = curve(points=200)
    assert_same(detrend.biweight(time, flux, 30.0, device="cpu"), wotan.flatten(time, flux, method="biweight", window_length=30.0, return_trend=True)[1])


def test_empty_and_all_nan_inputs_give_nan_and_an_unsorted_time_is_refused():
    assert detrend.biweight([], [], WINDOW, device="cpu").size == 0
    assert np.isnan(detrend.biweight([0.0, 1.0], [np.nan, np.nan], WINDOW, device="cpu")).all()
    with pytest.raises(ValueError, match="increasing"):
        detrend.biweight([1.0, 0.0], [1.0, 1.0], WINDOW, device="cpu")


def test_the_launch_slice_shrinks_as_the_window_widens_and_stays_in_bounds():
    rows = [detrend.kernel_rows(width) for width in (64, 128, 1024, 2048, 4096, 16384)]
    assert rows == sorted(rows, reverse=True) and rows[2] == detrend.KERNEL_ROWS and min(rows) >= 64 and max(rows) <= 65536, rows


def test_the_module_works_where_triton_cannot_be_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)
    module = importlib.reload(detrend)
    try:
        assert module.triton is None
        time, flux = curve(points=400)
        assert_same(module.biweight(time, flux, WINDOW, device="cpu"), reference(time, flux))
    finally:
        monkeypatch.undo()
        importlib.reload(detrend)


needs_kernel = pytest.mark.skipif(detrend.triton is None or not torch.cuda.is_available(), reason="the Triton kernel needs Triton and a CUDA or ROCm device")


@needs_kernel
def test_the_kernel_gives_the_trend_of_the_torch_version_and_of_wotan():
    time, flux = curve(seed=2)
    mask = (time % 2.3 > 1.0) & (time % 2.3 < 1.06)
    device = detrend.biweight(time, flux, WINDOW, mask=mask, device="cuda")
    assert_same(device, detrend.biweight(time, flux, WINDOW, mask=mask, device="cpu"))
    assert_same(device, reference(time, flux, mask))


@needs_kernel
def test_the_kernel_handles_a_window_wider_than_a_thousand_points_and_a_slice_of_one_program():
    time, flux = curve(points=6000, step=20 / 86400)  # a window of 0.75 d holds 3,240 points
    assert detrend.triton.next_power_of_2(3241) == 4096
    assert_same(detrend.biweight(time, flux, WINDOW, device="cuda"), reference(time, flux))
    values, start, end = detrend.windows(time[~np.isnan(flux)], flux[~np.isnan(flux)], np.ones((~np.isnan(flux)).sum(), dtype=bool), WINDOW)
    tensors = [torch.as_tensor(array, device="cuda") for array in (values, start, end)]
    sliced = detrend.trend_kernel(*tensors, 4096, rows=1).cpu().numpy()
    np.testing.assert_allclose(sliced, detrend.trend_kernel(*tensors, 4096).cpu().numpy(), rtol=0, atol=0)


def sectors_of_a_star():
    """Three light curves of different lengths and gaps, one all NaN and one of a single point, with the transit mask of the first."""
    first = curve(seed=3)
    second = curve(points=1500, seed=4, gaps=((0.5, 0.6),), nans=3)
    third = (np.array([0.0]), np.array([1.0]))
    empty = (np.array([0.0, 0.01]), np.array([np.nan, np.nan]))
    mask = (first[0] % 2.3 > 1.0) & (first[0] % 2.3 < 1.06)
    return [first, second, third, empty], [mask, None, None, None]


def test_the_trends_of_many_light_curves_in_one_call_are_those_of_each_alone_and_wotans():
    sectors, masks = sectors_of_a_star()
    together = detrend.biweight_many(sectors, WINDOW, masks=masks, device="cpu")
    assert len(together) == len(sectors)
    for (time, flux), mask, trend in zip(sectors, masks, together):
        assert_same(trend, detrend.biweight(time, flux, WINDOW, mask=mask, device="cpu"))
        if len(time) > 2 and not np.isnan(flux).all():
            assert_same(trend, reference(time, flux, mask))


def test_many_light_curves_with_no_list_or_with_no_masks_and_a_mask_list_of_the_wrong_length():
    assert detrend.biweight_many([], WINDOW, device="cpu") == []
    time, flux = curve(points=300)
    assert_same(detrend.biweight_many([(time, flux)], WINDOW, device="cpu")[0], reference(time, flux))
    with pytest.raises(ValueError, match="one entry"):
        detrend.biweight_many([(time, flux)], WINDOW, masks=[None, None], device="cpu")


def test_a_batch_of_only_nan_light_curves_gives_nan_for_each():
    nan = (np.array([0.0, 1.0]), np.array([np.nan, np.nan]))
    out = detrend.biweight_many([nan, nan], WINDOW, device="cpu")
    assert len(out) == 2 and all(np.isnan(trend).all() and len(trend) == 2 for trend in out)


@needs_kernel
def test_the_kernel_on_many_light_curves_in_one_call_equals_the_torch_version_on_each():
    sectors, masks = sectors_of_a_star()
    together = detrend.biweight_many(sectors, WINDOW, masks=masks, device="cuda")
    for (time, flux), mask, trend in zip(sectors, masks, together):
        assert_same(trend, detrend.biweight(time, flux, WINDOW, mask=mask, device="cpu"))
