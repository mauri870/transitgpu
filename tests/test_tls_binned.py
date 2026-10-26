"""The template search of TLS on phase bins (transitgpu.tls.binned): the pieces on a CPU, and the search against the exact search of transitgpu.tls.window on light curves made here. Function tests: the inputs are made here and the references are numpy and the exact search."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitleastsquares import transitleastsquares  # noqa: E402
from transitleastsquares.backends import get_backend  # noqa: E402
from transitleastsquares.validate import validate_args  # noqa: E402

from transitgpu.tls import binned as tls_binned  # noqa: E402
from transitgpu.tls.window import window_power  # noqa: E402


def light_curve(points=6000, seed=1, period=3.0, depth=0.004, duration=0.12, span=60.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, points))
    y = 1 + rng.normal(0, 0.001, points)
    phase = ((t - 1.1) / period + 0.5) % 1 - 0.5
    y[np.abs(phase) * period < duration / 2] -= depth
    return t, y, np.full(points, 0.001)


def problem_of(t, y, dy, period_min, period_max):
    model = transitleastsquares(t, y, dy)
    model, _ = validate_args(model, dict(period_min=period_min, period_max=period_max, use_threads=1, show_progress_bar=False, verbose=False, backend="fused"))
    periods, durations = model._grids()
    overview, templates = model._templates(durations)
    return model._problem(model.t, model.y, model.dy, overview, templates), periods


def arguments(problem, periods):
    return (problem.t, problem.y, problem.dy, periods, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max)


def exact_arguments(problem, periods):
    """The arguments of window_power, which also takes TLS's T0_search_margin."""
    return (*arguments(problem, periods), problem.T0_search_margin)


def test_the_sums_and_counts_of_the_bins_are_those_of_a_direct_fold():
    t, y, dy = light_curve(points=2000, seed=2)
    periods = np.array([1.3, 2.9, 7.0])
    sums, counts = tls_binned.fold_sums(torch.as_tensor(t), torch.as_tensor(1.0 - y), torch.as_tensor(periods), 64)
    for row, period in enumerate(periods):
        phase = (t * (1.0 / period)) % 1.0
        index = np.minimum((phase * 64).astype(int), 63)
        assert np.array_equal(counts[row].numpy(), np.bincount(index, minlength=64))
        assert np.allclose(sums[row].numpy(), np.bincount(index, weights=1.0 - y, minlength=64), rtol=0, atol=1e-8)


def test_the_binned_template_keeps_its_area_and_a_delta_in_one_bin_correlates_to_the_template():
    problem, periods = problem_of(*light_curve(), 2.9, 3.1)
    bins = 1024
    rows = tls_binned.binned_templates(problem.lc_arr, problem.lc_cache_overview, bins)
    assert len(rows["widths"]) == len(np.unique(problem.lc_cache_overview["width_in_samples"]))
    for width, profile, square in zip(rows["widths"], rows["profile"], rows["square"]):
        row = int(np.flatnonzero(problem.lc_cache_overview["width_in_samples"] == width)[0])
        shape = 1.0 - np.asarray(problem.lc_arr[row], dtype=float)
        wb = len(profile)
        assert wb == max(1, round(problem.lc_cache_overview["duration"][row] * bins))
        assert np.sum(profile) * len(shape) / wb == pytest.approx(np.sum(shape), rel=1e-9), "the area of the template"
        assert np.sum(square) * len(shape) / wb == pytest.approx(np.sum(shape**2), rel=1e-9), "the area of its square"
    delta = np.zeros(bins)
    delta[100] = 1.0
    spectrum = np.fft.rfft(delta)
    correlation = np.fft.irfft(spectrum * np.conj(np.fft.rfft(np.concatenate([rows["profile"][3], np.zeros(bins - len(rows["profile"][3]))]))), n=bins)
    wb = len(rows["profile"][3])
    assert np.allclose(correlation[100 - wb + 1:101][::-1], rows["profile"][3][:wb], atol=1e-12), "start s sees the delta at template bin 100 - s"


def test_the_search_finds_the_injected_period_where_the_exact_search_does_and_gives_the_baseline_where_no_duration_is_allowed():
    t, y, dy = light_curve(points=3000, seed=4, depth=0.006)
    problem, periods = problem_of(t, y, dy, 2.5, 3.5)
    chi, rows, depths = tls_binned.binned_power(*arguments(problem, periods), bins_per_shortest=12, device="cpu")
    exact = window_power(*exact_arguments(problem, periods), correlate="direct", batch=32)
    assert periods[np.argmin(chi)] == periods[np.argmin(exact[0])], (periods[np.argmin(chi)], periods[np.argmin(exact[0])])
    assert np.isfinite(chi).all() and (chi <= chi.max()).all()
    extremes = np.array([periods[0], 0.02, 20000.0])  # a period for which no duration is allowed is infinite, as the exact search has it
    chi_e, _, _ = tls_binned.binned_power(*arguments(problem, extremes), bins_per_shortest=12, device="cpu")
    exact_e = window_power(*exact_arguments(problem, extremes), correlate="direct")
    assert (np.isinf(chi_e) == np.isinf(exact_e[0])).all()


def test_errors_that_differ_are_refused():
    t, y, dy = light_curve(points=3000)
    problem, periods = problem_of(t, y, dy, 2.9, 3.1)
    uneven = 0.001 * np.random.default_rng(1).uniform(0.7, 1.4, len(t))
    with pytest.raises(ValueError, match="equal errors"):
        tls_binned.binned_power(problem.t, problem.y, uneven, periods, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max, device="cpu")


@pytest.mark.gpu
def test_on_the_device_the_kernel_gives_the_chi_square_rows_and_depths_of_the_torch_search_to_float32_rounding():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    t, y, dy = light_curve(points=8000, seed=6, depth=0.005)
    problem, periods = problem_of(t, y, dy, 1.5, 4.5)
    host = tls_binned.binned_power(*arguments(problem, periods), device="cpu", bins_per_shortest=12)
    device = tls_binned.binned_power(*arguments(problem, periods), device="cuda", bins_per_shortest=12)
    finite = np.isfinite(host[0])
    assert (finite == np.isfinite(device[0])).all()
    baseline = host[0][finite].max()
    assert np.max(np.abs(host[0][finite] - device[0][finite])) < 1e-4 * (baseline - host[0][finite].min()), "the gain agrees to float32 rounding"
    assert periods[np.argmin(host[0])] == periods[np.argmin(device[0])]
    assert np.mean(host[1] == device[1]) > 0.98 and np.allclose(host[2], device[2], atol=2e-4)


@pytest.mark.parametrize("batch", [1, 3, 7, 64])
def test_the_search_does_not_depend_on_the_batch_including_a_last_batch_of_one_period(batch):
    """The masks of the allowed durations are made once for the grid and sliced by batch: 7 periods in batches of 3 end with one period, and a batch of 64 holds them all."""
    problem, periods = problem_of(*light_curve(points=3000, seed=4), 2.5, 3.5)
    periods = periods[:7]
    reference = tls_binned.binned_power(*arguments(problem, periods), device="cpu", batch=7)
    got = tls_binned.binned_power(*arguments(problem, periods), device="cpu", batch=batch)
    for a, b in zip(reference, got):
        assert np.array_equal(a, b)


def _dense_light_curve(cadence_seconds, days):
    time = np.arange(0, days, cadence_seconds / 86400)
    flux = 1 + 5e-4 * np.random.default_rng(5).normal(size=time.size)
    flux[np.abs((time - 1.2 + 1.59) % 3.18 - 1.59) < 0.05] -= 3e-3
    return time, flux


def _power_of(time, flux, exact):
    from transitleastsquares import transitleastsquares as Tls

    from transitgpu.tls.binned import BinnedBackend

    backend = BinnedBackend()
    backend.exact = exact
    return Tls(time, flux).power(period_min=1.0, period_max=5.0, backend=backend, show_progress_bar=False, verbose=False)


@pytest.mark.gpu
def test_the_binned_backend_says_exact_so_that_tls_does_not_bin_the_light_curve_a_second_time():
    """With exact = False, TLS 2.0.1 (main.py, _search_plan) also bins the light curve in time where the shortest trial
    duration allows it: at a cadence of 20 s that changes the periodogram, at 2 minutes it does not."""
    transitleastsquares = pytest.importorskip("transitleastsquares")  # noqa: F841
    if not torch.cuda.is_available():
        pytest.skip("no device: the binned backend searches on a GPU only and raises without one")
    time, flux = _dense_light_curve(120, 30)
    assert np.array_equal(_power_of(time, flux, True).power, _power_of(time, flux, False).power), "at a cadence of 2 minutes no period is binned"
    time, flux = _dense_light_curve(20, 12)
    kept, binned = _power_of(time, flux, True), _power_of(time, flux, False)
    assert float(np.abs(kept.power - binned.power).max()) > 1e-3, "at a cadence of 20 s the light curve is binned a second time when the flag says approximate"
    assert kept.period == binned.period


@pytest.mark.gpu
def test_the_integer_fold_of_the_binned_backend_refuses_a_residual_the_sums_cannot_hold():
    """fold_sums quantizes the residual to 64-bit integers, so it refuses a residual whose sums could overflow, as core._check does."""
    from transitgpu.tls import binned

    if not torch.cuda.is_available():
        pytest.skip("no device: the integer fold runs on a device")
    count = 1_000_000
    t = torch.linspace(0.0, 1000.0, count, dtype=torch.float64, device="cuda")
    periods = torch.tensor([3.18, 3.19], dtype=torch.float64, device="cuda")
    normal = torch.full((count,), 1e-3, dtype=torch.float64, device="cuda")
    sums, counts = binned.fold_sums(t, normal, periods, 64)
    assert counts.sum().item() == 2 * count
    with pytest.raises(ValueError, match="too large for the integer sums"):
        binned.fold_sums(t, normal * 1e7, periods, 64)  # 1e4 per point: 1e4 / 1e-9 * 1e6 points = 1e19 > 2**62


@pytest.mark.gpu
def test_the_binned_search_refuses_a_residual_the_sums_cannot_hold_before_it_folds_any_period(monkeypatch):
    """The residual 1 - y is the same for every batch of periods, so its range is checked once, with numpy, before the loop and not by a device-to-host copy for each batch."""
    from transitgpu.tls import binned

    if not torch.cuda.is_available():
        pytest.skip("no device: the integer fold runs on a device")
    problem, periods = problem_of(*light_curve(), 2.9, 3.1)
    overflow = problem.y.copy()
    overflow[0] = -1e8  # 1e8 / 1e-9 * a few thousand points is above 2**62
    given = list(arguments(problem, periods))
    given[1] = overflow
    monkeypatch.setattr(binned, "fold_sums", lambda *a, **k: pytest.fail("a period was folded before the range of the residual was checked"))
    with pytest.raises(ValueError, match="too large for the integer sums"):
        binned.binned_power(*given, device="cuda", bins_per_shortest=12)


@pytest.mark.gpu
def test_the_range_of_the_residual_is_checked_once_for_all_the_batches_of_a_binned_search(monkeypatch):
    from transitgpu import core
    from transitgpu.tls import binned

    if not torch.cuda.is_available():
        pytest.skip("no device")
    problem, periods = problem_of(*light_curve(), 2.9, 3.1)
    calls = []
    original = core.check_integer_range
    monkeypatch.setattr(core, "check_integer_range", lambda *a: calls.append(a) or original(*a))
    binned.binned_power(*arguments(problem, periods), device="cuda", bins_per_shortest=12, batch=3)
    assert len(periods) > 6, "the periods fill several batches"
    assert len(calls) == 1, f"{len(calls)} checks for {-(-len(periods) // 3)} batches"


def test_the_number_of_bins_does_not_depend_on_the_order_of_the_periods():
    """TLS passes the periods in an unseeded shuffled order, so B is taken at the longest period, not from a sample."""
    from transitleastsquares.grid import T14

    def needed(period):
        return np.log2(12 / T14(R_s=0.13, M_s=0.1, P=period, small=True))

    low, high = 1.0, 14.0  # needed() rises with the period: find where it crosses the next integer above needed(1)
    target = np.ceil(needed(low))
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if needed(middle) < target else (low, middle)
    periods = np.linspace(0.5, high * 1.0005, 5000)  # only the last few periods need the larger B
    expected = tls_binned.bins_of(periods, 12, 0.13, 0.1)
    assert expected == 2 ** (int(target) + 1)  # the longest periods need one doubling more than the rest
    for seed in range(20):
        assert tls_binned.bins_of(np.random.default_rng(seed).permutation(periods), 12, 0.13, 0.1) == expected


def test_the_bin_count_of_each_period_is_that_of_the_rule_for_the_period_alone_and_never_falls_with_the_period():
    periods = np.array([0.6, 1.0, 3.0, 9.0, 14.0, 40.0, 150.0, 750.0])
    counts = tls_binned.bin_counts(periods, 12, 0.13, 0.1)
    assert counts.tolist() == [tls_binned.bins_of(np.array([p]), 12, 0.13, 0.1) for p in periods], counts
    assert np.all(np.diff(counts) >= 0), counts
    assert counts[0] < counts[-1] == tls_binned.MAXIMUM_PHASE_BINS, "a grid from 0.6 to 750 d has several bands"
    assert tls_binned.bin_counts(np.array([5.0, 5.0]), 12, 0.13, 0.1).tolist() == [tls_binned.bins_of(np.array([5.0]), 12, 0.13, 0.1)] * 2, "a repeated period"


def test_a_grid_in_one_band_gives_the_chi_squares_of_the_single_bin_count_exactly():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    t, y, dy = light_curve()
    problem, periods = problem_of(t, y, dy, 3.0, 4.0)
    counts = tls_binned.bin_counts(periods, 12, problem.R_star_min, problem.M_star_min)
    assert len(set(counts.tolist())) == 1, counts
    single = tls_binned.binned_power(*arguments(problem, periods), device="cuda")
    banded = tls_binned.banded_power(*arguments(problem, periods), device="cuda")
    for one, other in zip(single, banded):
        assert np.array_equal(one, other)


def test_a_grid_over_two_bands_gives_for_each_band_the_chi_squares_of_that_band_alone_whatever_the_order_of_the_periods():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    t, y, dy = light_curve(points=8000, span=120.0)
    problem, periods = problem_of(t, y, dy, 0.6, 60.0)
    counts = tls_binned.bin_counts(periods, 12, problem.R_star_min, problem.M_star_min)
    assert len(set(counts.tolist())) >= 2, "the grid must cross a band boundary"
    shuffled = np.random.default_rng(3).permutation(len(periods))
    banded = tls_binned.banded_power(*arguments(problem, periods[shuffled]), device="cuda")
    for bins in sorted(set(counts.tolist())):
        inside = np.flatnonzero(counts[shuffled] == bins)
        alone = tls_binned.binned_power(*arguments(problem, periods[shuffled][inside]), device="cuda", bins=bins)
        for one, other in zip(alone, banded):
            assert np.array_equal(one, other[inside]), bins
