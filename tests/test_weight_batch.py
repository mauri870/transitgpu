"""Many weight vectors of one flux in one search: with one flux and a matrix of weights, one row for each result, `search_many` gives for each row the result of a search of the flux with that row, to the bit; a weight of zero masks its point.

Function tests of two paths that must agree (differential tests): the batch against the separate searches, and a zero weight against a point with no flux.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from transitgpu.core import search, search_many

pytestmark = pytest.mark.gpu  # every test of this module may search on the device


@pytest.fixture(autouse=True)
def fresh_compiler(monkeypatch):
    import torch

    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit", 64)


def stack():
    """Three sectors of 3 days, 15 days apart, at 30 minutes (432 points), a 3.18 day dip of 2.5 hours and 1,500 ppm, in 400 ppm of noise."""
    time = np.concatenate([np.arange(0.0, 3.0, 1 / 48), np.arange(15.0, 18.0, 1 / 48), np.arange(30.0, 33.0, 1 / 48)])
    flux = 1 + np.random.default_rng(4).normal(0, 400e-6, time.size)
    flux[np.abs(time - (1.2 + np.round((time - 1.2) / 3.18) * 3.18)) < 2.5 / 48] -= 1500e-6
    return time, flux


def vectors(count, size):
    """Weight vectors: the first has unequal positive weights, the others mask a different tenth of the points with a weight of zero."""
    rng = np.random.default_rng(5)
    rows = [rng.uniform(0.5, 2.0, size)]
    for _ in range(count - 1):
        row = rng.uniform(0.5, 2.0, size)
        row[rng.random(size) < 0.1] = 0.0
        rows.append(row)
    return np.array(rows)


def same(found, other):
    assert found.keys() == other.keys()
    for key in found:
        np.testing.assert_array_equal(found[key], other[key], err_msg=key)


@pytest.mark.parametrize("bumps", [False, True])
def test_each_row_of_a_batch_is_the_search_of_the_flux_with_that_row_to_the_bit(bumps):
    time, flux = stack()
    rows = vectors(3, time.size)
    batch = search_many(time, [flux], 0.6, 6.0, spectrum=True, boxes=True, weights=rows, bumps=bumps)
    assert len(batch) == 3
    for found, row in zip(batch, rows):
        alone = search(time, flux, 0.6, 6.0, spectrum=True, boxes=True, weights=row, bumps=bumps)
        same({key: value for key, value in found.items() if key != "bump"}, {key: value for key, value in alone.items() if key != "bump"})
        if bumps:
            same(found["bump"], alone["bump"])


def test_a_weight_of_zero_is_a_point_with_no_flux():
    """The edge: the masked points are not counted in N, in the median weight or in the level, so masking by weight and by NaN give one search."""
    time, flux = stack()
    row = vectors(2, time.size)[1]
    masked = flux.copy()
    masked[row == 0] = np.nan
    by_weight = search(time, flux, 0.6, 6.0, spectrum=True, weights=row)
    by_nan = search(time, masked, 0.6, 6.0, spectrum=True, weights=np.where(row > 0, row, 1.0))
    same(by_weight, by_nan)


def test_one_row_of_weights_for_one_flux_is_one_result_and_a_matrix_for_many_fluxes_keeps_its_rows():
    time, flux = stack()
    rows = vectors(2, time.size)
    one = search_many(time, [flux], 0.6, 6.0, weights=rows[:1])
    assert len(one) == 1
    same(one[0], search(time, flux, 0.6, 6.0, weights=rows[0]))
    two = search_many(time, [flux, flux[::-1].copy()], 0.6, 6.0, weights=rows)
    assert len(two) == 2
    same(two[1], search(time, flux[::-1].copy(), 0.6, 6.0, weights=rows[1]))


@pytest.mark.parametrize("bad", ["negative", "nan", "all zero", "too few"])
def test_weights_that_are_negative_not_finite_or_leave_too_few_points_raise(bad):
    time, flux = stack()
    row = np.ones(time.size)
    if bad == "negative":
        row[10] = -1.0
    elif bad == "nan":
        row[10] = np.nan
    elif bad == "all zero":
        row[:] = 0.0
    else:
        row[:] = 0.0
        row[:5] = 1.0
    with pytest.raises(ValueError, match="weight"):
        search(time, flux, 0.6, 6.0, weights=row)
