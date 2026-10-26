"""The search of a window of trial periods of TLS in PyTorch (transitgpu.tls.window) against TLS 2.0.1's exact backend.

On a CPU these tests run `window_power` with torch's CPU kernels; the last one holds the GPU lock shared and runs on the device when there is one. Function tests: the inputs are light curves made here and the reference is TLS itself."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitleastsquares import transitleastsquares  # noqa: E402
from transitleastsquares.backends import get_backend  # noqa: E402
from transitleastsquares.validate import validate_args  # noqa: E402

from transitgpu.tls.window import WindowBackend, window_power  # noqa: E402


def light_curve(points=3000, seed=1, period=3.0, depth=0.004, duration=0.12, span=27.0, error=None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, points))
    y = 1 + rng.normal(0, 0.001, points)
    phase = ((t - 1.1) / period + 0.5) % 1 - 0.5
    y[np.abs(phase) * period < duration / 2] -= depth
    dy = np.full(points, 0.001) if error is None else error
    return t, y, dy


def reference(t, y, dy, period_min, period_max, backend="fused"):
    """TLS's problem for the window and the exact backend's SearchResult."""
    model = transitleastsquares(t, y, dy)
    model, _ = validate_args(model, dict(period_min=period_min, period_max=period_max, use_threads=1, show_progress_bar=False, verbose=False, backend=backend))
    engine = get_backend(backend)
    periods, durations = model._grids()
    overview, templates = model._templates(durations)
    problem = model._problem(model.t, model.y, model.dy, overview, templates)
    found = model._search(engine, periods, overview, templates)
    finish = getattr(engine, "finish", None)
    if finish is not None:
        finish()
    return problem, found


def ported(problem, periods, **options):
    return window_power(problem.t, problem.y, problem.dy, periods, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max, problem.T0_search_margin, **options)


def test_the_chi_square_of_every_period_is_the_one_of_tls_fused_in_float64_with_the_same_template_row_and_depth_and_the_same_best_period():
    problem, found = reference(*light_curve(), 2.97, 3.03)
    chi, rows, depths = ported(problem, found.periods, correlate="direct")
    finite = np.isfinite(found.chi2)
    assert (np.isfinite(chi) == finite).all()
    assert np.max(np.abs(chi[finite] - found.chi2[finite]) / np.abs(found.chi2[finite])) < 1e-9, "T1: a relative 1e-9 on every period"
    assert (rows == found.rows).all() and np.allclose(depths, found.depths, rtol=1e-9, atol=1e-12), "T2: the template row and the depth at the minimum"
    assert found.periods[np.argmin(chi)] == found.periods[np.argmin(found.chi2)], "T3: the best period is the same grid point"


def test_the_fourier_correlation_in_float32_keeps_the_best_period_and_the_chi_square_to_1e_6():
    problem, found = reference(*light_curve(points=4000, seed=2), 2.9, 3.1)
    chi, rows, depths = ported(problem, found.periods, correlate="fft")
    assert np.max(np.abs(chi - found.chi2) / np.abs(found.chi2)) < 1e-6
    assert found.periods[np.argmin(chi)] == found.periods[np.argmin(found.chi2)]
    assert np.mean(rows == found.rows) >= 0.99, "T2: the rows agree for at least 99% of the periods"


def test_errors_that_differ_between_points_take_the_general_path_and_agree_with_tls():
    rng = np.random.default_rng(5)
    t, y, _ = light_curve(points=3000, seed=3)
    dy = 0.001 * rng.uniform(0.7, 1.4, len(t))
    problem, found = reference(t, y, dy, 2.97, 3.03)
    for correlate in ("direct", "fft"):
        chi, rows, depths = ported(problem, found.periods, correlate=correlate)
        assert np.max(np.abs(chi - found.chi2) / np.abs(found.chi2)) < 1e-6, correlate
        assert found.periods[np.argmin(chi)] == found.periods[np.argmin(found.chi2)], correlate


def test_a_batch_larger_than_the_window_and_a_window_of_one_period_give_the_same_numbers():
    problem, found = reference(*light_curve(points=2500, seed=4), 2.98, 3.02)
    whole = ported(problem, found.periods, correlate="direct", batch=1000)
    one_by_one = ported(problem, found.periods, correlate="direct", batch=1)
    for a, b in zip(whole, one_by_one):
        assert np.array_equal(a, b)
    single = ported(problem, found.periods[:1], correlate="direct")
    assert single[0][0] == whole[0][0] and single[1][0] == whole[1][0]


def test_periods_far_outside_the_window_give_the_numbers_of_the_reference_search_of_one_period():
    from transitleastsquares.core import search_period

    problem, found = reference(*light_curve(points=2000, seed=6), 2.97, 3.03)
    extremes = np.array([found.periods[0], 0.02, 0.3, 20000.0])  # a period for which the physical limits leave few or no durations, one shorter than a transit may be and one longer than the data
    chi, rows, depths = ported(problem, extremes, correlate="direct")
    for i, period in enumerate(extremes):
        expected = search_period(period, problem.t, problem.y, problem.dy, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max, problem.lc_arr, problem.lc_cache_overview, problem.T0_search_margin)
        assert (np.isinf(expected[1]) and np.isinf(chi[i])) or chi[i] == pytest.approx(expected[1], rel=1e-9), period
        assert rows[i] == expected[2] and depths[i] == pytest.approx(expected[3], rel=1e-9, abs=1e-12), period


def test_tls_run_with_the_backend_gives_the_periodogram_and_the_refined_numbers_of_the_exact_backend():
    t, y, dy = light_curve(points=3500, seed=7)
    exact = transitleastsquares(t, y, dy).power(period_min=2.9, period_max=3.1, use_threads=1, show_progress_bar=False, verbose=False, backend="fused")
    port = transitleastsquares(t, y, dy).power(period_min=2.9, period_max=3.1, use_threads=1, show_progress_bar=False, verbose=False, backend=WindowBackend(device="cpu", dtype=torch.float64, correlate="direct"))
    assert port.period == exact.period
    for name in ("duration", "depth", "T0", "snr", "transit_count", "SDE"):
        assert getattr(port, name) == pytest.approx(getattr(exact, name), rel=1e-6), name
    assert np.allclose(port.periods, exact.periods) and np.allclose(port.power, exact.power, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
def test_on_the_device_the_search_agrees_with_tls_when_there_is_one():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    problem, found = reference(*light_curve(points=4000, seed=8), 2.95, 3.05)
    chi, rows, depths = ported(problem, found.periods, device="cuda", correlate="fft")
    assert np.max(np.abs(chi - found.chi2) / np.abs(found.chi2)) < 1e-6
    assert found.periods[np.argmin(chi)] == found.periods[np.argmin(found.chi2)]


def test_the_screened_search_in_float64_is_the_chi_square_of_tls_fused_and_correlates_few_starts_of_long_templates():
    problem, found = reference(*light_curve(points=12000, seed=9, span=60.0), 2.99, 3.01)
    assert problem.lc_cache_overview["width_in_samples"].max() >= 64, "the screening applies only to templates of at least 64 samples"
    report = {}
    chi, rows, depths = ported(problem, found.periods, correlate="screen", report=report)
    finite = np.isfinite(found.chi2)
    assert (np.isfinite(chi) == finite).all()
    assert np.max(np.abs(chi[finite] - found.chi2[finite]) / np.abs(found.chi2[finite])) < 1e-9, "T1"
    assert (rows == found.rows).all() and np.allclose(depths, found.depths, rtol=1e-9, atol=1e-12), "T2"
    assert found.periods[np.argmin(chi)] == found.periods[np.argmin(found.chi2)], "T3"
    assert report["survivors"] < 0.05 * report["starts"], report


def test_the_screened_search_of_a_light_curve_without_a_dip_and_of_errors_that_differ_is_refused_or_equal_to_the_oracle():
    t, y, dy = light_curve(points=3000, seed=10, depth=0.0)
    problem, found = reference(t, y, dy, 2.97, 3.03)
    chi, rows, depths = ported(problem, found.periods, correlate="screen")
    assert np.max(np.abs(chi - found.chi2) / np.abs(found.chi2)) < 1e-9 and (rows == found.rows).all(), "no dip: the minima are noise, and equal to TLS's"
    uneven = 0.001 * np.random.default_rng(11).uniform(0.7, 1.4, len(t))
    problem, found = reference(t, y, uneven, 2.97, 3.03)
    with pytest.raises(ValueError, match="equal errors"):
        ported(problem, found.periods, correlate="screen")


def test_the_screened_search_gives_the_numbers_of_the_reference_search_for_periods_with_few_or_no_durations():
    from transitleastsquares.core import search_period

    problem, found = reference(*light_curve(points=2000, seed=6), 2.97, 3.03)
    extremes = np.array([found.periods[0], 0.02, 0.3, 20000.0])
    chi, rows, depths = ported(problem, extremes, correlate="screen")
    for i, period in enumerate(extremes):
        expected = search_period(period, problem.t, problem.y, problem.dy, problem.transit_depth_min, problem.R_star_min, problem.R_star_max, problem.M_star_min, problem.M_star_max, problem.lc_arr, problem.lc_cache_overview, problem.T0_search_margin)
        assert (np.isinf(expected[1]) and np.isinf(chi[i])) or chi[i] == pytest.approx(expected[1], rel=1e-9), period
        assert rows[i] == expected[2] and depths[i] == pytest.approx(expected[3], rel=1e-9, abs=1e-12), period


@pytest.mark.gpu
def test_on_the_device_the_screening_kernel_gives_the_numbers_of_tls_fused_and_the_survivors_of_the_torch_bounds():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    problem, found = reference(*light_curve(points=12000, seed=9, span=60.0), 2.99, 3.01)
    on_device, on_host = {}, {}
    chi, rows, depths = ported(problem, found.periods, device="cuda", correlate="screen", report=on_device)
    assert np.max(np.abs(chi - found.chi2) / np.abs(found.chi2)) < 1e-9 and (rows == found.rows).all() and np.allclose(depths, found.depths, rtol=1e-9, atol=1e-12)
    ported(problem, found.periods, device="cpu", correlate="screen", report=on_host)
    assert on_device == on_host, "the kernel keeps the starts that the torch bounds keep"


@pytest.mark.gpu
def test_the_launch_of_all_widths_keeps_every_start_that_the_float64_bounds_keep_and_few_others_for_a_sweep_of_incumbents():
    """The kernel avoids divisions and square roots and keeps a margin of 1e-8 of the size of the terms: it may keep a start the float64 bounds drop, never the reverse. The incumbents are the bounds of the starts themselves scaled by 1 plus or minus 1e-7 and 1e-5, so many starts sit at the threshold."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls.prefix import prefix_arrays_torch
    from transitgpu.tls.screen import Screen

    problem, found = reference(*light_curve(points=12000, seed=9, span=60.0), 2.99, 3.01)
    overview = problem.lc_cache_overview
    widths = sorted(int(w) for w in np.unique(overview["width_in_samples"]))
    cpu, gpu = (Screen(problem.y, problem.dy, problem.lc_arr, overview, widths, device) for device in ("cpu", "cuda"))
    y = torch.as_tensor(problem.y, dtype=torch.float64)
    phase = torch.as_tensor(problem.t) / found.periods[0] % 1.0
    order = torch.argsort(phase, stable=True)[None, :]
    extra = widths[-1] + widths[-1] % 2
    _, cumulative, squares, double, _ = prefix_arrays_torch(y, order, cpu.weight, cpu.mean_rw, extra)
    size = cumulative.shape[1] - 1
    margin = problem.T0_search_margin
    stride = {w: (max(int(w / (1 / margin)), 1) if margin > 0 and w > margin else 1) for w in widths}
    overshoot = {w: float(overview["overshoot"][np.flatnonzero(overview["width_in_samples"] == w)[0]]) for w in widths}
    counts = [len(range(0, size - w + 1, stride[w])) for w in widths]
    edges = np.concatenate([[0], np.cumsum(counts)])
    bounds = []
    for w in widths:
        step = stride[w]
        reach = len(range(0, size - w + 1, step))
        mean = 1.0 - (cumulative[:, w::step][:, :reach] - cumulative[:, :size - w + 1:step]) / float(w)
        r2 = squares[:, w::step][:, :reach] - squares[:, :size - w + 1:step]
        bounds.append(cpu.upper_bound(w, mean * (overshoot[w] / 0.5), r2, double, step, size)[0])
    everything = torch.cat(bounds)
    sample = everything[torch.randperm(len(everything), generator=torch.Generator().manual_seed(1))[:60]]
    cutoff = torch.cat([sample * (1.0 + factor) for factor in (-1e-5, -1e-7, 1e-7, 1e-5)])
    rows = len(cutoff)
    masks = []
    for screen, device in ((cpu, "cpu"), (gpu, "cuda")):
        keep = torch.zeros((rows, int(edges[-1])), dtype=torch.int8, device=device)
        arrays = [a.to(device).expand(rows, -1).contiguous() for a in (cumulative, squares, double)]
        screen.starts_kept_all(widths, stride, overshoot, size, 0.0, *arrays, cutoff.to(device), torch.ones((len(widths), rows), dtype=torch.bool, device=device), keep, edges)
        masks.append(keep.cpu().bool())
    missing = int((masks[0] & ~masks[1]).sum())
    extra_kept = int((masks[1] & ~masks[0]).sum())
    assert missing == 0, f"{missing} starts kept by the float64 bounds were dropped by the kernel"
    assert extra_kept <= 0.001 * int(masks[0].sum()) + 20, (extra_kept, int(masks[0].sum()))


def test_a_template_with_fewer_samples_than_its_box_gives_the_chi_square_of_tls_fused_in_the_screened_and_the_direct_search():
    """With an eccentric orbit and a square-root limb darkening the template of a duration has one sample fewer than the box of the duration (20 and 21): the correlations take the template's samples and the mean the box's."""
    rng = np.random.default_rng(2)
    t = np.arange(-15, 15, 0.01)
    phase = np.abs((t + 1.0 + 0.5 * 3.3) % 3.3 - 0.5 * 3.3)
    y = 1 + rng.normal(0, 4e-4, len(t))
    y[phase < 0.06] -= 1.5e-3
    options = dict(period_min=2.8, period_max=3.8, ecc=0.3, w=40, b=0.4, limb_dark="squareroot", u=[0.3, 0.3], show_progress_bar=False, verbose=False, use_threads=1)
    exact = transitleastsquares(t, y).power(backend="fused", **options)
    for correlate in ("screen", "direct"):
        mine = transitleastsquares(t, y).power(backend=WindowBackend(device="cpu", correlate=correlate), **options)
        assert np.max(np.abs(mine.chi2 - exact.chi2) / np.abs(exact.chi2)) < 1e-9, correlate


@pytest.mark.parametrize("backend", ["gpu", "gpu-binned"])
def test_without_a_device_the_backends_refuse_and_name_transitleastsquares_as_the_library_for_a_cpu(backend, monkeypatch):
    """transitgpu.tls is for a GPU: without one its backends raise an error that names transitleastsquares, and do not search on a CPU (torch's operations on a CPU are the oracle of the kernels, not a method)."""
    import transitgpu.tls

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t, y, dy = light_curve(points=2000, seed=12)
    with pytest.raises(RuntimeError, match="transitleastsquares"):
        transitgpu.tls.transitleastsquares(t, y).power(period_min=2.97, period_max=3.03, show_progress_bar=False, verbose=False, use_threads=1, backend=backend)


def test_a_cpu_device_with_a_named_correlation_keeps_the_torch_search():
    """The torch search on a CPU stays as the reference of the kernels of the device: a named correlation (direct, fft, screen) is that search, not TLS's kernel."""
    t, y, dy = light_curve(points=3000, seed=13)
    options = dict(period_min=2.97, period_max=3.03, show_progress_bar=False, verbose=False, use_threads=1)
    exact = transitleastsquares(t, y).power(backend="fused", **options)
    port = transitleastsquares(t, y).power(backend=WindowBackend(device="cpu", correlate="direct"), **options)
    assert not np.array_equal(port.chi2, exact.chi2) and np.allclose(port.chi2, exact.chi2, rtol=1e-9)


@pytest.mark.gpu
def test_the_kernel_keeps_every_start_whose_exact_gain_is_above_the_incumbent_through_the_cascade_of_proxies():
    """The soundness of the screening, with the true gain as the reference and not another bound: for incumbents taken at the quantiles of the exact gains of a width, every start with a gain above the incumbent is in the mask (a start may be kept without one, never dropped with one)."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls.prefix import prefix_arrays
    from transitgpu.tls.screen import Screen

    problem, found = reference(*light_curve(points=12000, seed=9, span=60.0), 2.99, 3.01)
    overview = problem.lc_cache_overview
    widths = sorted(int(w) for w in np.unique(overview["width_in_samples"]))
    screen = Screen(problem.y, problem.dy, problem.lc_arr, overview, widths, "cuda")
    width = widths[-1]
    assert width in screen.proxy and width in screen.loose, "a width with both proxies"
    extra = widths[-1] + widths[-1] % 2
    y = torch.as_tensor(problem.y, dtype=torch.float64, device="cuda")
    phase = torch.as_tensor(problem.t, device="cuda") / found.periods[0] % 1.0
    order = torch.argsort(phase, stable=True)[None, :]
    patched, cumulative, squares, double, _ = prefix_arrays(y, order, screen.weight, screen.mean_rw, extra)
    size = patched.shape[1]
    margin = problem.T0_search_margin
    step = max(int(width * margin), 1) if margin > 0 and width > margin else 1
    count = len(range(0, size - width + 1, step))
    starts = torch.arange(count, device="cuda") * step
    mean = 1.0 - (cumulative[0, starts + width] - cumulative[0, starts]) / float(width)
    overshoot = float(overview["overshoot"][np.flatnonzero(overview["width_in_samples"] == width)[0]])
    k = mean * (overshoot / 0.5)
    ar = screen.correlations(patched, torch.zeros(count, dtype=torch.int64, device="cuda"), starts, torch.full((count,), screen.index_of[width], device="cuda"))
    gain = 2.0 * k * ar - k * k * screen.square[width]
    live = mean > 0.0
    quantiles = torch.quantile(gain[live], torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64, device="cuda"))
    rows = len(quantiles)
    keep = torch.zeros((rows, count), dtype=torch.int8, device="cuda")
    expand = lambda a: a.expand(rows, -1).contiguous()
    screen.starts_kept_all([width], {width: step}, {width: overshoot}, size, 0.0, expand(cumulative), expand(squares), expand(double), quantiles, torch.ones((1, rows), dtype=torch.bool, device="cuda"), keep, np.array([0, count]))
    kept = keep.bool()
    for row in range(rows):
        above = live & (gain > quantiles[row])
        assert bool((kept[row] | ~above).all()), f"a start above the incumbent {float(quantiles[row]):.3f} was dropped"
    assert int(kept[3].sum()) < 0.5 * int(live.sum()), "the screening drops the starts far below the incumbent"


@pytest.mark.parametrize("name", ["cutoff", "allowed", "keep"])
def test_a_launch_of_the_screening_with_a_tensor_of_the_wrong_size_is_refused_and_names_it(name):
    """An empty tensor has the pointer 0 and a kernel that reads it faults the device: the launch is refused on the
    host, for a CPU as for a device, naming the argument."""
    from transitgpu.tls.screen import Screen

    problem, found = reference(*light_curve(points=3000, seed=9, span=60.0), 2.99, 3.01)
    widths = sorted(int(w) for w in np.unique(problem.lc_cache_overview["width_in_samples"]))
    screen = Screen(problem.y, problem.dy, problem.lc_arr, problem.lc_cache_overview, widths, "cpu")
    size, rows = 3100, 4
    cumulative, squares, double = (torch.zeros((rows, size + extra), dtype=torch.float64) for extra in (1, 1, 2))
    arguments = dict(cutoff=torch.zeros(rows, dtype=torch.float64), allowed=torch.ones((1, rows), dtype=torch.bool), keep=torch.zeros((rows, 100), dtype=torch.int8))
    arguments[name] = arguments[name][:0]  # empty, as a slice past the end is
    with pytest.raises(ValueError, match=name):
        screen.starts_kept_all(widths[:1], {widths[0]: 4}, {widths[0]: 1.0}, size, 0.0, cumulative, squares, double, arguments["cutoff"], arguments["allowed"], arguments["keep"], np.array([0, 100]))


def test_the_default_batch_is_the_largest_power_of_two_under_the_element_budget_between_the_limits():
    """Edge cases: one point, the sizes of the benchmark, and a light curve so long that the lower limit holds."""
    from transitgpu.tls.window import default_batch

    assert [default_batch(points) for points in (1, 1000, 20_000, 32_768, 50_000, 100_000, 200_000, 5_000_000)] == [256, 256, 256, 256, 128, 64, 64, 64]


def test_a_batch_of_one_period_gives_the_arrays_of_the_correlations_contiguous_and_the_numbers_of_a_batch_of_two(monkeypatch):
    """torch.arange(b).expand(count, b).reshape(-1) is a view with stride 0 for b = 1, which a kernel reading `rows` with
    unit stride would read out of range. The arrays of the starts must be contiguous for any number of periods; the
    last batch of a search can hold one period."""
    from transitgpu.tls.screen import Screen

    seen = []
    original = Screen.correlations

    def spy(self, patched, rows, starts, width_index, residual=False):
        seen.append((rows.is_contiguous(), starts.is_contiguous(), width_index.is_contiguous()))
        return original(self, patched, rows, starts, width_index, residual=residual)

    monkeypatch.setattr(Screen, "correlations", spy)
    problem, found = reference(*light_curve(points=2500, seed=6), 2.98, 3.02)
    periods = found.periods[:2]
    alone = ported(problem, periods[:1], correlate="screen", batch=1)
    together = ported(problem, periods, correlate="screen", batch=2)
    assert seen and all(all(flags) for flags in seen), f"the arrays given to the correlations must be contiguous: {seen}"
    for single, pair in zip(alone, together):
        assert np.array_equal(single, pair[:1]), "a period alone equals the same period in a batch of two"


def test_the_correlations_refuse_arrays_that_are_not_contiguous():
    """A kernel indexes `rows`, `starts` and `width_index` with unit stride; a view with stride 0 or 2 would be read out of range on a device."""
    from transitgpu.tls.screen import Screen

    problem, found = reference(*light_curve(points=1500, seed=7), 2.98, 3.02)
    widths = [int(w) for w in np.unique(problem.lc_cache_overview["width_in_samples"])]
    screen = Screen(problem.y, problem.dy, problem.lc_arr, problem.lc_cache_overview, widths, "cpu")
    patched = torch.ones((1, 300), dtype=torch.float64)
    starts = torch.zeros(4, dtype=torch.int64)
    wide = torch.arange(8, dtype=torch.int64)[::2]  # stride 2
    stuck = torch.zeros(1, dtype=torch.int64).expand(4)  # stride 0
    for rows in (wide, stuck):
        with pytest.raises(ValueError, match="contiguous"):
            screen.correlations(patched, rows, starts, starts)


@pytest.mark.gpu
def test_on_the_device_a_batch_of_one_period_gives_the_numbers_of_a_batch_of_two():
    """The kernels' version of the test above: the last batch of a search can hold one period."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    problem, found = reference(*light_curve(points=2500, seed=6), 2.98, 3.02)
    periods = found.periods[:2]
    alone = ported(problem, periods[:1], device="cuda", correlate="screen", batch=1)
    together = ported(problem, periods, device="cuda", correlate="screen", batch=2)
    for single, pair in zip(alone, together):
        assert np.array_equal(single, pair[:1]), "a period alone equals the same period in a batch of two"


def test_the_screening_refuses_a_mask_or_a_cutoff_that_is_not_contiguous():
    """starts_kept_all hands `cutoff`, `allowed` and `keep` to a kernel that indexes them as dense data."""
    from transitgpu.tls.screen import Screen

    problem, found = reference(*light_curve(points=1500, seed=8), 2.98, 3.02)
    widths = [int(w) for w in np.unique(problem.lc_cache_overview["width_in_samples"])]
    screen = Screen(problem.y, problem.dy, problem.lc_arr, problem.lc_cache_overview, widths, "cpu")
    rows, size = 2, 100
    cumulative = torch.zeros((rows, size + 1), dtype=torch.float64)
    cutoff = torch.zeros(2 * rows, dtype=torch.float64)[::2]  # stride 2
    with pytest.raises(ValueError, match="contiguous"):
        screen.starts_kept_all([widths[0]], {widths[0]: 1}, {widths[0]: 1.0}, size, 0.0, cumulative, cumulative, cumulative, cutoff, torch.ones((1, rows), dtype=torch.bool), torch.zeros((rows, 10), dtype=torch.int8), np.array([0, 10]))


def near_edge_case(outlier, relative):
    """A folded array of 400 points with a flux outlier at its start, which makes the prefix sum of the squares large, and two dips of the shape of one template: the second `relative` deeper than the first, so that its gain is higher by about 2 `relative`.
    Returns the screen, the width, the arrays of the screened search, the gain of each start computed directly and the starts."""
    from transitgpu.tls.prefix import prefix_arrays
    from transitgpu.tls.screen import Screen

    problem, _ = reference(*light_curve(points=1500, seed=7), 2.98, 3.02)
    overview = problem.lc_cache_overview
    widths = [int(w) for w in np.unique(overview["width_in_samples"])]
    width = next(w for w in widths if w >= 20)
    row = int(np.flatnonzero(overview["width_in_samples"] == width)[0])
    shape = 1.0 - np.asarray(problem.lc_arr[row], dtype=np.float64)
    points, first, second = 400, 60, 220
    folded = np.ones(points)
    folded[0] += outlier
    folded[first:first + len(shape)] -= 0.01 * shape
    folded[second:second + len(shape)] -= 0.01 * (1 + relative) * shape
    screen = Screen(folded, np.full(points, 0.001), problem.lc_arr, overview, widths, "cpu")
    max_width = max(widths) + max(widths) % 2
    patched, cumulative, squares, double, _ = prefix_arrays(torch.as_tensor(folded), torch.arange(points)[None, :], screen.weight, screen.mean_rw, max_width)
    overshoot = float(overview["overshoot"][row])

    def gain(start):
        mean = -(folded[start:start + width] - 1.0).sum() / width
        k = mean * overshoot / 0.5
        return 2 * k * screen.weight * float(np.dot(1.0 - folded[start:start + len(shape)], shape)) - k * k * screen.square[width]

    return screen, width, overshoot, problem.transit_depth_min, patched, cumulative, squares, double, gain(first), gain(second), first, second


@pytest.mark.parametrize("relative", [1e-8, 1e-6])
def test_a_start_better_than_the_incumbent_by_a_small_fraction_is_kept_after_a_flux_outlier_makes_the_prefix_sum_of_the_squares_large(relative):
    """Cauchy-Schwarz bounds the correlation by sqrt(A2 R2), with R2 a difference of two prefix sums of the squares that
    carries a rounding error of the size of the larger prefix. With an outlier of 1e4 in the flux (the prefix of the
    squares reaches 2e14 at the weight 1e6) the bound needs the allowance for that rounding to keep the better start."""
    screen, width, overshoot, depth_min, patched, cumulative, squares, double, incumbent, better, first, second = near_edge_case(1e4, relative)
    assert better > incumbent, "the second start is better than the incumbent by construction"
    size = patched.shape[1]
    keep = torch.zeros((1, len(range(0, size - width + 1))), dtype=torch.int8)
    screen.starts_kept(width, 1, size, overshoot, depth_min, cumulative, squares, double, torch.tensor([incumbent], dtype=torch.float64), torch.ones(1, dtype=torch.bool), keep)
    assert keep[0, second] == 1, f"the better start {second} (gain {better!r}, incumbent {incumbent!r}, relative {relative}) was pruned"


@pytest.mark.parametrize("outlier", [0.0, 1e2, 1e3, 1e4, 1e5])
def test_no_start_that_beats_the_incumbent_by_a_relative_1e_10_or_more_is_pruned_over_the_sweep_of_outliers_and_margins(outlier):
    """Five outliers by ten margins (the second dip deeper or shallower than the first by 1e-6, 1e-8, 1e-10, 1e-12 and 1e-14 relative) is 50 cases over the five parametrized runs. A margin under 1e-10 is inside the rounding of the gain computed directly,
    so only the starts that beat the incumbent by more than that are required to be kept."""
    missed = []
    for relative in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14, -1e-6, -1e-8, -1e-10, -1e-12, -1e-14):
        screen, width, overshoot, depth_min, patched, cumulative, squares, double, incumbent, better, first, second = near_edge_case(outlier, relative)
        size = patched.shape[1]
        keep = torch.zeros((1, len(range(0, size - width + 1))), dtype=torch.int8)
        screen.starts_kept(width, 1, size, overshoot, depth_min, cumulative, squares, double, torch.tensor([incumbent], dtype=torch.float64), torch.ones(1, dtype=torch.bool), keep)
        if better > incumbent * (1 + 1e-10) and keep[0, second] == 0:
            missed.append((outlier, relative, better, incumbent))
    assert not missed, f"pruned although better than the incumbent: {missed}"


@pytest.mark.gpu
@pytest.mark.parametrize("relative", [1e-6, 1e-8])
def test_on_the_device_the_screening_kernel_keeps_a_start_better_than_the_incumbent_after_a_flux_outlier(relative):
    """The kernel's version of the near-edge test: the same folded array with an outlier of 1e4, the better second dip, through starts_kept_all on the device (the Cauchy-Schwarz line of the kernel takes the allowance for the rounding of R2)."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    screen, width, overshoot, depth_min, patched, cumulative, squares, double, incumbent, better, first, second = near_edge_case(1e4, relative)
    assert better > incumbent
    size = patched.shape[1]
    count = len(range(0, size - width + 1))
    keep = torch.zeros((1, count), dtype=torch.int8, device="cuda")
    screen.starts_kept_all([width], {width: 1}, {width: overshoot}, size, depth_min, cumulative.cuda(), squares.cuda(), double.cuda(), torch.tensor([incumbent], dtype=torch.float64, device="cuda"),
                           torch.ones((1, 1), dtype=torch.bool, device="cuda"), keep, np.array([0, count]))
    assert keep[0, second] == 1, f"the better start {second} was pruned by the kernel (relative {relative})"


# The order of the periods handed to the backend: TLS 2.0.1 hands them in a random order by default, a batch of the backend then holds periods of the whole range and the union of their template widths,
# and the screening kernel runs over every width of the union for every period of the batch.


def columns_of_the_screening(monkeypatch, run):
    """Run `run()` and return the columns that the screening kernel was asked to test, summed over the batches."""
    from transitgpu.tls import screen

    columns = []
    original = screen.Screen.starts_kept_all

    def logging(self, used, stride, overshoot, size, depth_min, cumulative, squares, double, cutoff, allowed, keep, edges):
        columns.append(int(edges[len(used)]) * int(cumulative.shape[0]))
        return original(self, used, stride, overshoot, size, depth_min, cumulative, squares, double, cutoff, allowed, keep, edges)

    monkeypatch.setattr(screen.Screen, "starts_kept_all", logging)
    result = run()
    return sum(columns), result


@pytest.mark.gpu
def test_the_periods_are_batched_by_period_whatever_their_order_so_the_chi_square_is_equal_and_the_screening_tests_no_more_starts(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("the screened search runs on a device")
    problem, found = reference(*light_curve(points=6000, seed=4, span=300.0), 0.6, 60.0)
    periods = np.asarray(found.periods)
    shuffled = np.random.default_rng(3).permutation(periods)
    backend = WindowBackend(device="cuda", batch=64, correlate="screen")
    sorted_columns, ordered = columns_of_the_screening(monkeypatch, lambda: backend.search(problem, periods))
    shuffled_columns, mixed = columns_of_the_screening(monkeypatch, lambda: backend.search(problem, shuffled))
    assert np.array_equal(ordered.chi2, mixed.chi2) and np.array_equal(ordered.periods, mixed.periods) and np.array_equal(ordered.rows, mixed.rows)
    assert shuffled_columns == sorted_columns, f"the screening tested {shuffled_columns} columns for the shuffled periods and {sorted_columns} for the sorted"
    from transitgpu.tls.window import window_power

    batch_union, _ = columns_of_the_screening(monkeypatch, lambda: window_power(problem.t, problem.y, problem.dy, shuffled, problem.lc_arr, problem.lc_cache_overview, problem.transit_depth_min, problem.R_star_min, problem.R_star_max,
                                                                                    problem.M_star_min, problem.M_star_max, problem.T0_search_margin, device="cuda", batch=64, correlate="screen"))
    assert batch_union > 1.5 * sorted_columns, "window_power itself batches the periods in the order given, which is why the backend sorts them"


@pytest.mark.gpu
def test_periods_given_as_a_reversed_view_are_searched_and_give_the_chi_square_of_the_periods_in_order():
    if not torch.cuda.is_available():
        pytest.skip("the screened search runs on a device")
    problem, found = reference(*light_curve(points=3000, seed=5, span=100.0), 0.8, 20.0)
    backend = WindowBackend(device="cuda", batch=64, correlate="screen")
    reversed_view = np.asarray(found.periods)[::-1]
    assert reversed_view.strides[0] < 0
    result = backend.search(problem, reversed_view)
    assert np.array_equal(result.chi2, backend.search(problem, np.ascontiguousarray(found.periods)).chi2)


def test_the_batch_of_the_exact_search_is_256_when_the_memory_allows_and_falls_by_multiples_of_64_to_64():
    """Edge cases: one point, the sizes of the benchmark and of the GTLS paper, a device with little free memory."""
    from transitgpu.tls.window import exact_batch

    free = 24_000_000_000
    assert [exact_batch(points, free) for points in (1, 20_000, 72_000, 144_000, 200_000)] == [256, 256, 256, 256, 192]
    assert [exact_batch(200_000, free_bytes) for free_bytes in (4_000_000_000, 12_000_000_000, 24_000_000_000)] == [64, 64, 192]
    assert exact_batch(5_000_000, 1_000_000) == 64, "never under the lower limit"


def test_a_batch_whose_starts_would_pass_the_32_bit_offsets_of_the_screening_is_refused_with_its_numbers():
    from transitgpu.tls.window import check_offsets

    check_offsets(256, 2_268_753)
    check_offsets(1, 2**30 - 1)
    with pytest.raises(ValueError, match=r"1024 periods.*2268753 starts"):
        check_offsets(1024, 2_268_753)


def test_on_the_device_a_batch_of_256_periods_gives_the_numbers_of_a_batch_of_64_over_long_and_short_periods():
    """The periods of the batches reach from 0.6 d to the baseline, so a batch holds many template widths."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no device")
    problem, found = reference(*light_curve(points=8000, seed=11, span=400.0), 0.6, 200.0, backend="fused")
    assert len(found.periods) > 300, len(found.periods)
    small = ported(problem, found.periods, correlate="screen", batch=64)
    large = ported(problem, found.periods, correlate="screen", batch=256)
    for a, b in zip(small, large):
        assert np.array_equal(a, b)


@pytest.mark.gpu
def test_on_the_device_a_launch_of_more_than_2_to_the_25_survivors_gives_every_survivor_the_dot_product_of_a_small_launch():
    """A launch of 33,936,431 survivors (a batch of 512 periods of 72,000 points) gave other dot products to millions of them: a grid whose work items pass 2^32 is not dispatched correctly. The survivors are dotted in launches of at most 2^23."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls.screen import Screen

    problem, _ = reference(*light_curve(points=1500, seed=7), 2.98, 3.02)
    widths = [int(w) for w in np.unique(problem.lc_cache_overview["width_in_samples"])]
    screen = Screen(problem.y, problem.dy, problem.lc_arr, problem.lc_cache_overview, widths, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(5)
    patched = 1.0 - 0.01 * torch.rand((2, 1500 + max(widths)), dtype=torch.float64, device="cuda", generator=generator)
    count = (1 << 25) + 1000
    rows = torch.zeros(count, dtype=torch.int64, device="cuda")
    starts = torch.full((count,), 17, dtype=torch.int64, device="cuda")
    width_index = torch.full((count,), len(widths) - 1, dtype=torch.int64, device="cuda")
    small = screen.correlations(patched, rows[:1000], starts[:1000], width_index[:1000])
    large = screen.correlations(patched, rows, starts, width_index)
    assert large.shape[0] == count
    assert torch.equal(large, small[0].expand(count)), f"{int((large != small[0]).sum())} of {count} survivors have another dot product than the one of a launch of 1000"


def test_a_batch_whose_mask_of_starts_would_pass_the_limit_is_halved_until_it_fits_and_a_single_period_is_never_dropped():
    from transitgpu.tls import window

    widths = [100, 200]
    stride = {100: 1, 200: 2}
    allowed = np.array([[True] * 8, [False] * 4 + [True] * 4])  # widths x periods: the second width only for the last four periods
    size = 1000
    first, second = len(range(0, size - 100 + 1, 1)), len(range(0, size - 200 + 1, 2))
    assert (first, second) == (901, 401)
    assert window.fitting_batch(8, allowed, widths, stride, size) == 8, "it fits under the real limit"
    assert window.fitting_batch(8, allowed, widths, stride, size) == 8
    original = window.OFFSET_LIMIT
    try:
        window.OFFSET_LIMIT = 8 * (first + second) - 1  # the batch of 8 has both widths, 8 x 1302 starts: too many
        assert window.fitting_batch(8, allowed, widths, stride, size) == 4, "four periods that need only the first width: 4 x 901 starts"
        window.OFFSET_LIMIT = 901
        assert window.fitting_batch(8, allowed, widths, stride, size) == 1, "one period is never dropped"
        many = np.ones((2, 256), dtype=bool)
        window.OFFSET_LIMIT = 150 * (first + second)
        assert [window.fitting_batch(count, many, widths, stride, size) for count in (256, 200, 128, 64)] == [128, 64, 128, 64], "multiples of 64 while there are more than 64 periods"
    finally:
        window.OFFSET_LIMIT = original


@pytest.mark.gpu
def test_on_the_device_a_batch_split_by_the_limit_of_the_mask_gives_the_numbers_of_the_whole_batch(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls import window

    problem, found = reference(*light_curve(points=8000, seed=11, span=400.0), 0.6, 200.0, backend="fused")
    whole = ported(problem, found.periods, correlate="screen", batch=256)
    seen = []
    original = window.check_offsets

    def spy(periods, starts):
        seen.append(periods * starts)
        return original(periods, starts)

    monkeypatch.setattr(window, "check_offsets", spy)
    monkeypatch.setattr(window, "OFFSET_LIMIT", 6_000_000)
    split = ported(problem, found.periods, correlate="screen", batch=256)
    assert len(seen) > len(found.periods) // 256 + 1, "the limit made more batches"
    assert max(seen) < 6_000_000, max(seen)
    assert np.allclose(whole[0], split[0], rtol=1e-12, atol=0.0), "the chi square, to the rounding of the last digit (a batch of another size can change it in a few periods)"
    assert np.array_equal(whole[1], split[1]) and np.array_equal(whole[2], split[2])


@pytest.mark.gpu
def test_the_correlations_of_a_weighted_residual_are_those_of_the_flux_bit_for_bit_on_a_device():
    """screen.correlations(residual, ..., residual=True) reads weight * (1 - flux) already formed and gives the same sums as the call on the flux."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls.screen import Screen

    problem, _ = reference(*light_curve(points=6000, seed=4, span=60.0), 2.99, 3.01)
    widths = sorted(int(w) for w in np.unique(problem.lc_cache_overview["width_in_samples"]))
    screen = Screen(problem.y, problem.dy, problem.lc_arr, problem.lc_cache_overview, widths, "cuda")
    rng = np.random.default_rng(3)
    patched = torch.as_tensor(1 + 0.001 * rng.normal(size=(5, 6400)), device="cuda")
    count = 300
    rows = torch.as_tensor(rng.integers(0, 5, count), device="cuda")
    index = torch.as_tensor(rng.integers(0, len(widths), count), device="cuda")
    starts = torch.as_tensor([int(rng.integers(0, 6400 - screen.length[widths[i]])) for i in index.tolist()], device="cuda")
    plain = screen.correlations(patched, rows, starts, index)
    residual = screen.correlations(screen.weight * (1.0 - patched), rows, starts, index, residual=True)
    assert torch.equal(plain, residual)


def test_the_correlations_of_a_weighted_residual_on_a_cpu_are_those_of_the_flux_to_rounding():
    from transitgpu.tls.screen import Screen

    problem, _ = reference(*light_curve(points=3000, seed=4, span=60.0), 2.99, 3.01)
    widths = sorted(int(w) for w in np.unique(problem.lc_cache_overview["width_in_samples"]))
    screen = Screen(problem.y, problem.dy, problem.lc_arr, problem.lc_cache_overview, widths, "cpu")
    rng = np.random.default_rng(5)
    patched = torch.as_tensor(1 + 0.001 * rng.normal(size=(4, 3200)))
    count = 40
    rows = torch.as_tensor(rng.integers(0, 4, count))
    index = torch.as_tensor(rng.integers(0, len(widths), count))
    starts = torch.as_tensor([int(rng.integers(0, 3200 - screen.length[widths[i]])) for i in index.tolist()])
    plain = screen.correlations(patched, rows, starts, index)
    residual = screen.correlations(screen.weight * (1.0 - patched), rows, starts, index, residual=True)
    assert torch.allclose(plain, residual, rtol=1e-12, atol=0.0)
