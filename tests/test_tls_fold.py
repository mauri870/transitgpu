"""The stable phase order of a batch from a sort of 32-bit keys (transitgpu.tls.fold) against the stable sort of the float64 phases. Function tests: the inputs are phases made here and the reference is torch's own stable argsort."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitgpu.tls.fold import phase_order  # noqa: E402
from transitgpu.tls.fold import phases as fold_phases  # noqa: E402


def phases(times, periods):
    scaled = torch.as_tensor(times)[None, :] / torch.as_tensor(periods)[:, None]
    return scaled - torch.floor(scaled)


def assert_same_order(phase):
    assert torch.equal(phase_order(phase), torch.argsort(phase, dim=1, stable=True))


def test_random_times_give_the_stable_order_of_the_float64_phases():
    rng = np.random.default_rng(1)
    assert_same_order(phases(np.sort(rng.uniform(0, 700, 20000)), rng.uniform(0.5, 14, 8)))


def test_a_regular_cadence_at_a_multiple_of_the_cadence_has_long_runs_of_nearly_equal_phases_and_the_same_order():
    cadence = 2.0 / 1440.0
    times = np.arange(5000) * cadence + 1e-12 * np.random.default_rng(2).normal(size=5000)  # a period of 30 cadences puts 5000 points on 30 phases up to rounding
    assert_same_order(phases(times, np.array([30 * cadence, 30.0000001 * cadence, 1.0])))


def test_equal_phases_keep_the_index_order_and_all_equal_phases_are_the_identity():
    phase = torch.zeros((2, 50), dtype=torch.float64)
    assert torch.equal(phase_order(phase), torch.arange(50)[None, :].expand(2, 50))
    phase = torch.tensor([[0.5, 0.25, 0.5, 0.25, 0.5, 0.0]], dtype=torch.float64)
    assert phase_order(phase).tolist() == [[5, 1, 3, 0, 2, 4]]


def test_phases_that_differ_below_float32_resolution_are_ordered_by_the_float64_value():
    phase = torch.tensor([[0.5 + 3e-9, 0.5, 0.5 + 1e-9, 0.25, 0.5 + 2e-9]], dtype=torch.float64)
    assert torch.equal(phase_order(phase), torch.argsort(phase, dim=1, stable=True))
    assert phase_order(phase).tolist() == [[3, 1, 2, 4, 0]]


def test_one_point_and_the_largest_phase_below_one():
    assert phase_order(torch.zeros((1, 1), dtype=torch.float64)).tolist() == [[0]]
    phase = torch.tensor([[np.nextafter(1.0, 0.0), 0.0, 1.0 - 1e-12]], dtype=torch.float64)
    assert_same_order(phase)


@pytest.mark.gpu
def test_the_phase_kernel_gives_the_float64_phases_of_torch_bit_for_bit_and_their_float32_keys():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    rng = np.random.default_rng(3)
    times = torch.as_tensor(np.sort(rng.uniform(0, 700, 5000)), device="cuda")
    periods = torch.as_tensor(np.concatenate([rng.uniform(0.5, 14, 6), [1.0, 2e-3, 20000.0]]), device="cuda")
    phase, key = fold_phases(times, periods)
    scaled = times[None, :] / periods[:, None]
    expected = scaled - torch.floor(scaled)
    assert torch.equal(phase, expected), "the same division, floor and subtraction"
    assert torch.equal(key, expected.to(torch.float32))
