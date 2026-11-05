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


def test_the_launches_of_a_search_at_2_5_million_points_are_as_long_as_at_the_largest_stack_that_keeps_the_old_batch():
    """The mean launch of the longest kernel at 2,525,000 points is at most 1.5 times its mean at 312,500 points, where 512 periods are taken as before, and no launch of any kernel at 2,525,000 points exceeds 10 ms (device). The largest launch is not compared between sizes: it is the maximum of 18 launches at one size and of about 1,000 at the other, and rises with the count (8.6 ms and 5.3 to 7.4 ms with equal means of 4.2 and 4.3 ms)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the launches are of a GPU")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("launch_size", Path(__file__).resolve().parent.parent / "benchmarks" / "launch_size.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old, _ = module.launches(312_500)
    new, _ = module.launches(2_525_000)
    name = max(old, key=lambda kernel: sum(old[kernel]) / len(old[kernel]))
    mean_old = sum(old[name]) / len(old[name])
    mean_new = sum(new[name]) / len(new[name])
    def describe(times):
        ordered = sorted(times)
        return f"{len(ordered)} launches, median {ordered[len(ordered) // 2]:.3f} ms, largest {ordered[-1]:.3f} ms"

    assert mean_new <= 1.5 * mean_old, f"{name}: mean {mean_new:.3f} ms against {mean_old:.3f} ms; at 2,525,000 points {describe(new[name])}; at 312,500 points {describe(old[name])}"
    largest = max(max(times) for times in new.values())
    assert largest <= 10.0, f"largest launch {largest:.2f} ms"


@pytest.mark.parametrize("points", [1_000, 105_000, 312_500, 400_000, 2_525_000, 100_000_000])
def test_a_call_of_one_flux_takes_the_periods_that_it_took_before_the_fluxes_were_counted(points):
    assert core.one_pass_batch(points, 1) == core.one_pass_batch(points)


@pytest.mark.parametrize("points,fluxes,rows", [(153_046, 8, 343), (236_736, 8, 222), (236_736, 4, 384), (600_000, 8, 87), (2_524_608, 8, 20), (2_524_608, 12, 20), (2_524_608, 1, 63), (20_000, 8, 512), (50_000_000, 8, 1)])
def test_the_periods_of_a_launch_fall_with_the_fluxes_of_its_call(points, fluxes, rows):
    """The accumulation kernel visits every point of its block for every flux of the call, so the work of a launch grows with the fluxes; a flux counts for 0.4 of the first flux in the work of a launch."""
    assert core.one_pass_batch(points, fluxes) == rows


def test_the_periods_of_a_launch_never_grow_with_the_points_or_the_fluxes_and_the_work_stays_under_the_budget():
    sizes = np.unique(np.geomspace(1_000, 10**8, 80).astype(np.int64))
    for fluxes in (1, 2, 4, 8, 12):
        batches = [core.one_pass_batch(int(n), fluxes) for n in sizes]
        assert all(b >= 1 for b in batches) and all(a >= b for a, b in zip(batches, batches[1:]))
        for n, rows in zip(sizes, batches):
            weight = 1 + core.ONE_PASS_FLUX_WEIGHT * (min(fluxes, core.FLUX_CHUNK) - 1)
            assert rows == 1 or rows * n * weight <= core.ONE_PASS_FLUX_ELEMENTS or rows * n <= core.ONE_PASS_ELEMENTS and fluxes == 1
            assert rows <= core.one_pass_batch(int(n), 1)
    for n in sizes:
        assert core.one_pass_batch(int(n), 8) <= core.one_pass_batch(int(n), 4) <= core.one_pass_batch(int(n), 1)


def test_a_search_of_many_fluxes_with_a_small_flux_budget_equals_the_search_with_the_default_bit_for_bit(monkeypatch):
    """The integer sums and the independent periods give the same result for any division of the periods into launches, also with many fluxes in a call (device, one-pass accumulation)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the one-pass accumulation needs a GPU")
    from test_transitgpu import light_curve

    time, flux = light_curve(seed=5, cadence_min=30.0)
    rng = np.random.default_rng(5)
    fluxes = [flux] + [flux[rng.permutation(len(flux))] for _ in range(11)]
    default = core.search_many(time, fluxes, 0.5, 14, spectrum=True)
    monkeypatch.setattr(core, "ONE_PASS_FLUX_ELEMENTS", 3 * time.size * (1 + core.ONE_PASS_FLUX_WEIGHT * 7))
    assert core.one_pass_batch(time.size, 8) == 3
    small = core.search_many(time, fluxes, 0.5, 14, spectrum=True)
    assert repr(default) == repr(small), "a batch of 3 periods changed the result of 12 fluxes"


def largest_launch(points, fluxes, period_max):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the launches are of a GPU")
    from torch.profiler import ProfilerActivity, profile

    rng = np.random.default_rng(833)
    time = np.sort(rng.uniform(0.0, 2877.0, points))
    flux = 1.0 + 1e-3 * rng.standard_normal(points)
    many = [flux[rng.permutation(points)] for _ in range(fluxes)]
    core.search_many(time, many[:2], 0.5, period_max)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as recording:
        core.search_many(time, many, 0.5, period_max)
        torch.cuda.synchronize()
    times = [event.device_time / 1000.0 for event in recording.events() if event.device_type == torch.autograd.DeviceType.CUDA and event.device_time > 0]
    return max(times), len(times)


@pytest.mark.gpu
def test_a_call_of_8_fluxes_at_236_thousand_points_has_no_launch_over_10_ms():
    largest, launches = largest_launch(236_736, 8, 14.0)
    assert largest <= 10.0, f"largest launch {largest:.2f} ms of {launches}"


@pytest.mark.slow
@pytest.mark.gpu
def test_a_call_of_12_fluxes_at_2_5_million_points_has_no_launch_over_10_ms():
    largest, launches = largest_launch(2_524_608, 12, 14.0)
    assert largest <= 10.0, f"largest launch {largest:.2f} ms of {launches}"
