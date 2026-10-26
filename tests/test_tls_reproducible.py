"""The exact backend of transitgpu.tls with unequal errors (the direct correlation of transitgpu.tls.window) gives the same bits for a period whatever the other periods of its batch are: TLS passes the periods to a backend in an order that is shuffled differently in each call. Function tests: the inputs are light curves
made here and the reference is the same call."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_tls_binned import exact_arguments, light_curve, problem_of  # noqa: E402

from transitgpu.tls.window import window_power  # noqa: E402

pytestmark = pytest.mark.gpu


def unequal_problem(points=3000):
    t, y, dy = light_curve(points=points)
    dy = dy * (1.0 + 0.5 * np.random.default_rng(3).uniform(size=points))
    problem, periods = problem_of(t, y, dy, 0.8, 8.0)
    return exact_arguments(problem, periods[::20])


def test_the_direct_correlation_with_unequal_errors_gives_the_same_bits_each_time_it_is_called():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    given = unequal_problem()
    first = window_power(*given, device="cuda", dtype=torch.float64, batch=8, correlate="direct")
    for call in range(3):
        again = window_power(*given, device="cuda", dtype=torch.float64, batch=8, correlate="direct")
        for name, a, b in zip(("chi2", "rows", "depths"), first, again):
            assert np.array_equal(a, b), f"call {call + 1}, {name}: the largest difference {np.max(np.abs(a - b)):.3e} in {int(np.sum(a != b))} of {len(a)} periods"


@pytest.mark.parametrize("points", (3000, 3001, 3002))
def test_the_direct_correlation_gives_a_period_the_same_bits_in_whatever_order_the_periods_come(points):
    if not torch.cuda.is_available():
        pytest.skip("no device")
    given = unequal_problem(points)
    periods = given[3]
    reference = window_power(*given, device="cuda", dtype=torch.float64, batch=8, correlate="direct")
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(len(periods))
        shuffled = window_power(*given[:3], periods[order], *given[4:], device="cuda", dtype=torch.float64, batch=8, correlate="direct")
        back = np.empty(len(periods), dtype=int)
        back[order] = np.arange(len(periods))
        for name, a, b in zip(("chi2", "rows", "depths"), reference, shuffled):
            assert np.array_equal(a, b[back]), f"seed {seed}, {name}: the largest difference {np.max(np.abs(a - b[back])):.3e} in {int(np.sum(a != b[back]))} of {len(a)} periods"
