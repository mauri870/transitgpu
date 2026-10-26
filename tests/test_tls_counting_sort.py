"""The stable phase order of a batch of periods by a counting sort on phase bins (transitgpu.tls.fold.counting_order) against the stable sort of the float64 phases: the same permutation bit for bit, whatever the times. Function tests:
the inputs are times made here, the reference is torch's stable argsort of the float64 phases that the same kernel's arithmetic gives."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu.tls.fold import counting_order, phases  # noqa: E402

pytestmark = pytest.mark.gpu


def assert_same_order(times, periods):
    if not torch.cuda.is_available():
        pytest.skip("no device")
    times = torch.as_tensor(np.asarray(times, dtype=np.float64), device="cuda")
    periods = torch.as_tensor(np.asarray(periods, dtype=np.float64), device="cuda")
    phase, _ = phases(times, periods)
    found = counting_order(times, periods)
    assert found.dtype == torch.int64 and found.shape == phase.shape
    assert torch.equal(found, torch.argsort(phase, dim=1, stable=True))


def test_random_times_give_the_stable_order_of_the_float64_phases():
    rng = np.random.default_rng(1)
    assert_same_order(np.sort(rng.uniform(0, 700, 20000)), rng.uniform(0.5, 14, 8))


def test_a_regular_cadence_at_a_multiple_of_the_cadence_puts_thousands_of_points_in_one_bin_and_the_order_is_the_same():
    cadence = 2.0 / 1440.0
    times = np.arange(5000) * cadence + 1e-12 * np.random.default_rng(2).normal(size=5000)  # 30 cadences make 5000 points fall on 30 phases up to rounding: bins of about 170 points
    assert_same_order(times, [30 * cadence, 30.0000001 * cadence, 1.0, 7 * cadence])


def test_equal_phases_keep_the_index_order_and_all_equal_phases_are_the_identity():
    assert_same_order(np.arange(300, dtype=np.float64), [1.0, 3.0, 0.5])  # integer times and integer periods: every phase is 0


def test_a_group_of_equal_phases_next_to_spread_phases_in_one_row():
    rng = np.random.default_rng(5)
    times = np.concatenate([np.full(2000, 3.0), rng.uniform(0, 50, 3000), np.full(100, 7.0)])
    assert_same_order(times, [1.0, 2.5, 0.37])


@pytest.mark.parametrize("points", [1, 2, 3, 1023, 1024, 1025, 2049])
def test_the_sizes_around_one_block_of_the_kernel_and_the_smallest_inputs(points):
    rng = np.random.default_rng(points)
    assert_same_order(np.sort(rng.uniform(0, 100, points)), [0.7, 3.3])


def test_one_period_and_a_phase_just_below_one():
    times = np.array([0.0, 1.0 - 2.0 ** -53, 2.0 - 2.0 ** -52, 3.0, 0.5, 4.999999999999999])
    assert_same_order(times, [1.0])


def test_phases_that_differ_below_float32_resolution_are_ordered_by_the_float64_value():
    times = np.array([0.5 + 3e-9, 0.5, 0.5 + 1e-9, 0.25, 0.5 + 2e-9, 1.5 + 4e-9])
    assert_same_order(times, [1.0, 2.0])


def test_a_row_with_a_bin_over_the_limit_is_sorted_by_torch_and_the_order_is_the_same(monkeypatch):
    import transitgpu.tls.fold as fold

    monkeypatch.setattr(fold, "RANK_LIMIT", 8)  # the regular cadence below puts 100 points in each of 50 bins; the random periods keep bins of a few points
    cadence = 2.0 / 1440.0
    times = np.arange(5000) * cadence
    assert_same_order(times, [50 * cadence, 0.8731, 50.0000001 * cadence, 1 * cadence])


def test_times_far_from_zero_and_a_period_that_rounds_the_scaled_time_to_an_integer_give_the_same_order():
    rng = np.random.default_rng(7)
    times = np.sort(rng.uniform(1.0e5, 1.0e5 + 800, 20000))  # t / P near 1e5 / 0.5: the phases come from a subtraction of numbers that agree to 11 digits
    assert_same_order(times, [0.5, 0.5 + 1e-13, 3.17996, 1.0e5 / 7919])
