"""`search_many` keeps the fields of the best box of each flux at the best period of each cut and not at every trial period, unless `boxes=True` asks for every period.

A differential test of two paths that must agree: the same fluxes searched with `boxes=True` (the fields of every period are kept, the best box is read from them) and with `boxes=False` (only the best box of each cut is kept), whose results are equal in every field they share, bit for bit. The memory test compares the peak of the numpy allocations of the two."""

import tracemalloc

import numpy as np
import pytest

pytest.importorskip("torch")

from transitgpu.core import grid_of, search_many

pytestmark = pytest.mark.gpu  # every test of this module may search on the device


def stack():
    """Three sectors of 4 days, 20 days apart, at 30 minutes (576 points), a 3.18 day dip of 2.5 hours and 1,500 ppm in 400 ppm of noise, and five fluxes: the stack and four block shuffles of it."""
    time = np.concatenate([np.arange(0.0, 4.0, 1 / 48), np.arange(20.0, 24.0, 1 / 48), np.arange(40.0, 44.0, 1 / 48)])
    rng = np.random.default_rng(4)
    flux = 1 + rng.normal(0, 400e-6, time.size)
    flux[np.abs(time - (1.2 + np.round((time - 1.2) / 3.18) * 3.18)) < 2.5 / 48] -= 1500e-6
    blocks = np.array_split(np.arange(time.size), 24)
    shuffled = [flux[np.concatenate([blocks[i] for i in rng.permutation(24)])] for _ in range(4)]
    return time, [flux, *shuffled]


def assert_same(reduced, full):
    """Every field of the reduced result equals that of the full one, as a result or as a list of results (one for each cut)."""
    if isinstance(full, list):
        assert len(reduced) == len(full)
        for one, other in zip(reduced, full):
            assert_same(one, other)
        return
    assert set(reduced) <= set(full)
    for key, value in reduced.items():
        if key == "bump":
            assert_same(value, full[key])
        else:
            np.testing.assert_array_equal(value, full[key], err_msg=key)  # bit-identical


@pytest.mark.parametrize("options", [{}, {"bumps": True}, {"spectrum": True}])
def test_the_best_box_kept_for_the_best_period_equals_the_one_read_from_every_period(options):
    time, fluxes = stack()
    reduced = search_many(time, fluxes, 0.5, 6.0, **options)
    assert_same(reduced, search_many(time, fluxes, 0.5, 6.0, boxes=True, **options))


def test_every_cut_of_a_list_of_cuts_has_its_own_best_box_and_equals_the_full_result():
    time, fluxes = stack()
    grid = grid_of(time, fluxes[0], 0.5, 14.0)
    cuts = (2.0, 6.0, float(grid.max()))
    reduced = search_many(time, fluxes, 0.5, 14.0, periods=grid, cuts=cuts, bumps=True)
    assert_same(reduced, search_many(time, fluxes, 0.5, 14.0, periods=grid, cuts=cuts, bumps=True, boxes=True))
    assert len({round(each["period_days"], 6) for each in reduced[0]}) > 1, "the cuts must end on different best periods for the test to separate them"


def test_two_trial_periods_with_the_same_statistic_in_different_batches_give_the_earlier_one():
    time, fluxes = stack()
    grid = np.repeat(grid_of(time, fluxes[0], 2.0, 6.0)[::7], 2)  # each period twice, so each statistic occurs at two indices
    reduced = search_many(time, fluxes[:2], 2.0, 6.0, periods=grid, batch=1, spectrum=True)
    full = search_many(time, fluxes[:2], 2.0, 6.0, periods=grid, batch=1, spectrum=True, boxes=True)
    assert_same(reduced, full)
    assert all(np.count_nonzero(each["power"] == each["power"].max()) >= 2 for each in reduced), "the maximum must be a tie for the test to separate the two indices"


def test_a_search_without_boxes_holds_less_than_the_arrays_of_every_period_that_a_search_with_boxes_holds():
    time, fluxes = stack()
    fluxes = fluxes * 2
    search_many(time, fluxes, 0.5, 6.0)  # compiled and cached before either peak is read
    peaks = {}
    for boxes in (True, False):
        tracemalloc.start()
        search_many(time, fluxes, 0.5, 6.0, boxes=boxes)
        peaks[boxes] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    array_bytes = 8 * len(fluxes) * grid_of(time, fluxes[0], 0.5, 6.0).size
    assert peaks[True] - peaks[False] >= 3 * array_bytes, f"the arrays of four fields of {array_bytes} bytes each were not freed: peaks {peaks}"
