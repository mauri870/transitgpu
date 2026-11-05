"""The depth of a result is relative to the mean level of the flux (review of 0.2.0, F4).

Function tests: the property is a field of a result for a flux with no relative depth.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_input_checks import search
from test_transitgpu import light_curve


def test_the_depth_is_nan_when_the_mean_level_of_the_flux_is_not_above_zero_and_the_snr_and_the_period_are_not():
    """Review of 0.2.0 (F4): the depth is relative to the mean level of the flux, as a flux flattened and normalized to 1 has it. A flux with the mean subtracted has a mean of 0 and a depth of infinity or of the wrong sign, which a result must not hold as a number."""
    time, flux = light_curve(days=60.0, cadence_min=10.0)
    normal = search(time=time, flux=flux, period_min=2.0, period_max=6.0, periods=None, bumps=True)[0]
    centered = search(time=time, flux=flux - flux.mean(), period_min=2.0, period_max=6.0, periods=None, bumps=True)[0]
    assert np.isfinite(normal["depth_ppm"]) and normal["depth_ppm"] > 0 and np.isfinite(normal["bump"]["depth_ppm"])
    assert np.isnan(centered["depth_ppm"]) and np.isnan(centered["depth_mean_ppm"]) and np.isnan(centered["bump"]["depth_ppm"])
    assert centered["period_days"] == normal["period_days"] and np.isfinite(centered["snr"]) and abs(centered["snr"] - normal["snr"]) < 1e-6 * abs(normal["snr"]), "the period and the SNR do not depend on the level"
