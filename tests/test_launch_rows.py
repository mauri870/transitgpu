"""The periods of a launch of the one-pass accumulation: a launch visits every point for every period of its batch, so the batch shrinks as the stack grows and a launch stays near 10 ms (docs/launch-size.md)."""
import numpy as np
import pytest

from transitgpu import core


@pytest.mark.parametrize("points", [1_000, 20_000, 105_000, 250_000, 312_500])
def test_a_stack_of_up_to_312_thousand_points_takes_512_periods_a_launch_as_before(points):
    assert core.one_pass_batch(points) == core.ONE_PASS_BATCH == 512


@pytest.mark.parametrize("points,rows", [(312_501, 511), (350_000, 457), (450_000, 355), (500_000, 320), (1_000_000, 160), (1_300_000, 123), (2_000_000, 80), (2_525_000, 63), (100_000_000, 1), (160_000_000, 1), (10**10, 1)])
def test_a_longer_stack_takes_the_periods_that_keep_a_launch_under_the_element_budget(points, rows):
    assert core.one_pass_batch(points) == rows
    assert rows * points <= core.ONE_PASS_ELEMENTS or rows == 1


def test_the_batch_never_grows_with_the_points_and_is_never_below_one():
    sizes = np.unique(np.geomspace(1, 10**10, 200).astype(np.int64))
    batches = [core.one_pass_batch(int(n)) for n in sizes]
    assert all(b >= 1 for b in batches) and all(a >= b for a, b in zip(batches, batches[1:]))


def test_the_search_of_a_stack_with_a_small_budget_equals_the_search_with_the_default_bit_for_bit(monkeypatch):
    """Integer sums added in any order, and periods independent of each other: a batch of 3 periods and a batch of 512 give the same result (device, one-pass accumulation)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the one-pass accumulation needs a GPU")
    from test_transitgpu import light_curve

    time, flux = light_curve(seed=3, cadence_min=30.0)
    default = core.search_many(time, [flux], 0.5, 14, spectrum=True)
    monkeypatch.setattr(core, "ONE_PASS_ELEMENTS", 3 * time.size)
    assert core.one_pass_batch(time.size) == 3
    small = core.search_many(time, [flux], 0.5, 14, spectrum=True)
    assert repr(default) == repr(small), "a batch of 3 periods changed the result"


def test_no_kernel_of_a_search_launches_longer_at_2_5_million_points_than_at_the_largest_stack_that_keeps_the_old_batch():
    """The bound is for every kernel of a call and not for the accumulation alone: the largest launch of any kernel at 2,525,000 points is at most 1.5 times the largest at 312,500 points, where 512 periods are taken as before (device; the absolute figure, about 10 ms on the card of docs/launch-size.md, is printed by benchmarks/launch_size.py)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the launches are of a GPU")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("launch_size", Path(__file__).resolve().parent.parent / "benchmarks" / "launch_size.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    largest = {}
    for points in (312_500, 2_525_000):
        kernels, _ = module.launches(points)
        largest[points] = max(max(times) for times in kernels.values())
    assert largest[2_525_000] <= 1.5 * largest[312_500], f"{largest}"
