"""The inputs of `search_many` that have no result are refused with a ValueError that names the argument (review of 0.2.0, F3), and not searched into a NaN, a loop or an empty answer.

Function tests: the property is an error path, and a txtar archive describes a light curve and a result (testdata/README.md).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve

from transitgpu import core

PERIODS = np.linspace(2.0, 6.0, 40)


def search(**changes):
    time, flux = light_curve()
    arguments = {"period_min": 2.0, "period_max": 6.0, "periods": PERIODS, "device": "cpu"}
    arguments.update(changes)
    return core.search_many(arguments.pop("time", time), [arguments.pop("flux", flux)], arguments.pop("period_min"), arguments.pop("period_max"), **arguments)


def test_a_search_with_valid_arguments_still_runs():
    assert np.isfinite(search()[0]["sde"])


@pytest.mark.parametrize("step", [1.0, 0.5, 0.0, -2.0, float("nan"), float("inf")])
def test_a_duration_step_of_one_or_less_or_not_finite_is_refused(step):
    with pytest.raises(ValueError, match="duration_step"):
        search(duration_step=step)


def test_a_duration_window_that_leaves_no_duration_to_score_is_refused():
    with pytest.raises(ValueError, match="duration_hours"):
        search(duration_hours=(1000.0, 2000.0))


@pytest.mark.parametrize("batch", [0, -1, -64])
def test_a_batch_under_one_is_refused(batch):
    with pytest.raises(ValueError, match="batch"):
        search(batch=batch)


@pytest.mark.parametrize("periods", [[0.0, 2.0, 3.0], [-1.0, 2.0, 3.0], [2.0, float("nan"), 3.0], [2.0, float("inf"), 3.0]])
def test_periods_that_are_not_positive_and_finite_are_refused(periods):
    with pytest.raises(ValueError, match="periods"):
        search(periods=np.array(periods))


@pytest.mark.parametrize("minimum", [0, -1, 0.0])
def test_a_minimum_of_bins_under_one_is_refused(minimum):
    with pytest.raises(ValueError, match="minimum_bins"):
        search(minimum_bins=minimum)


@pytest.mark.parametrize("periods", [[3.0], [3.0, 3.0], [3.0, 3.0, 3.0]])
def test_fewer_than_two_distinct_periods_are_refused_because_the_sde_is_a_z_score_over_the_periods(periods):
    with pytest.raises(ValueError, match="two distinct periods"):
        search(periods=np.array(periods))


def test_a_weighted_search_of_a_flux_with_no_variance_under_its_weights_is_refused():
    time, _ = light_curve()
    constant = np.ones(time.size)
    with pytest.raises(ValueError, match="variance"):
        search(flux=constant, weights=np.linspace(1.0, 2.0, time.size))
    with pytest.raises(ValueError, match="variance"):
        search(flux=constant)
