"""The weights of a weighted search are held as integers: each is rounded to a multiple of 1/WEIGHT_UNIT (1/64) of the median weight and clipped to between 1/64 and 64 times it. On a generated light curve with sectors 5 and 10 times
noisier than the others, the rounding gives a weight of 3 units for the sector at 5 times (2.56 exactly, 17% too heavy) and 1 unit for the one at 10 times (0.64 exactly), and this test holds that the SDE and the period of the box search
are those of weights held at 16384 units (rounded to 0.006% or less) within 0.5% and exactly."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu import core  # noqa: E402

pytestmark = pytest.mark.gpu

RATIOS = [1, 1, 1, 1, 1, 5, 5, 10]  # the scatter of each sector, in units of the scatter of the first


def noisy_sectors():
    """Eight sectors of 1,300 points at 30 minutes, 33 days apart, with the scatter 500 ppm times RATIOS and a box of 1,000 ppm and 2.5 h every 3.18 d; the weights are 1 / scatter squared."""
    rng = np.random.default_rng(7)
    time = np.concatenate([k * 33.0 + np.arange(1300) * 0.02083 for k in range(len(RATIOS))])
    sigma = np.concatenate([np.full(1300, 500e-6 * ratio) for ratio in RATIOS])
    flux = 1 + sigma * rng.normal(size=time.size)
    flux[np.abs((time - 1.2 + 1.59) % 3.18 - 1.59) < 0.05] -= 1000e-6
    return time, flux, 1 / sigma**2


def search_at(unit, time, flux, weights):
    original = core.WEIGHT_UNIT, core.MAXIMUM_WEIGHT
    core.WEIGHT_UNIT, core.MAXIMUM_WEIGHT = unit, 64 * unit
    try:
        return core.search(time, flux, 2.5, 4.0, spectrum=True, weights=weights)
    finally:
        core.WEIGHT_UNIT, core.MAXIMUM_WEIGHT = original


def test_a_sector_5_times_and_one_10_times_noisier_are_held_at_3_units_and_1_unit_and_the_search_is_that_of_the_fine_weights_within_half_a_percent():
    time, flux, weights = noisy_sectors()
    held = np.unique(np.clip(np.round(core.WEIGHT_UNIT * weights / np.median(weights)), 1, core.MAXIMUM_WEIGHT))
    assert held.tolist() == [1.0, 3.0, 64.0], held
    rounded, fine = search_at(64, time, flux, weights), search_at(16384, time, flux, weights)
    assert rounded["period_days"] == fine["period_days"] and abs(rounded["period_days"] / 3.18 - 1) < 0.01
    assert abs(rounded["sde"] - fine["sde"]) / fine["sde"] < 0.005, (rounded["sde"], fine["sde"])
