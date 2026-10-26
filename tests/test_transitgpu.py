import numpy as np
import pytest

torch = pytest.importorskip("torch")

from astropy.timeseries import BoxLeastSquares  # noqa: E402

from transitgpu import core  # noqa: E402
from transitgpu.core import search  # noqa: E402

pytestmark = pytest.mark.gpu  # every test of this module may search on the device

RESULT_FIELDS = (
    "period_min_searched_days", "period_max_searched_days", "period_days", "t0_days", "duration_hr", "duration_corrected_hr", "fill_factor",
    "depth_ppm", "depth_mean_ppm", "sde", "snr", "transit_count",
)


@pytest.fixture
def fresh_compile_cache():
    """Start a test that compares two calls from an empty compile cache: torch's Dynamo compiles a function for 8 variants and runs the later ones eagerly, whose spectrum differs from the compiled one in the last digits (about 1e-9 relative), so two calls
    of one test can take different paths when the process crosses that limit between them; it did in a full run of the suite, depending on what ran before."""
    torch._dynamo.reset()
    yield


def light_curve(period=3.18, t0=1.2, duration_hr=2.5, depth_ppm=3000, noise_ppm=1000, days=27.0, seed=1, cadence_min=30.0):
    """Flat noise with a box transit; the same recipe as a txtar archive's synthetic star, on a slower cadence so that the search on a CPU is fast."""
    time = np.arange(0.0, days, cadence_min / 1440)
    flux = 1 + np.random.default_rng(seed).normal(0, noise_ppm * 1e-6, time.size)
    cycle = np.round((time - t0) / period)
    flux[np.abs(time - (t0 + cycle * period)) < duration_hr / 48] -= depth_ppm * 1e-6
    return time, flux


def recovered(found, period=3.18, t0=1.2, duration_hr=2.5):
    """The rule of the completeness map: the period within 1% and the transit time within half a duration, modulo the period."""
    error = (found["t0_days"] - t0 + period / 2) % period - period / 2
    return abs(found["period_days"] / period - 1) < 0.01 and abs(error) < duration_hr / 48


def test_the_result_has_the_fields_the_batch_reads_and_the_spectrum_when_asked():
    time, flux = light_curve()
    found = search(time, flux, 0.5, 14, spectrum=True)
    assert set(RESULT_FIELDS) <= set(found)
    assert all(isinstance(found[name], float) or name == "transit_count" for name in RESULT_FIELDS)
    assert len(found["periods"]) == len(found["power"]) > 1000
    assert found["sde"] == pytest.approx(float(np.max(found["power"])), rel=1e-5), "the SDE is the best of the spectrum, which is a z-score"
    assert "periods" not in search(time, flux, 0.5, 14)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_an_injected_box_is_recovered_at_its_period_and_time(seed):
    time, flux = light_curve(seed=seed)
    found = search(time, flux, 0.5, 14)
    assert recovered(found), found
    assert found["depth_ppm"] == pytest.approx(3000, rel=0.25)
    assert found["duration_hr"] == pytest.approx(2.5, rel=0.5)


def test_noise_alone_gives_a_lower_sde_than_a_signal():
    time, flux = light_curve(depth_ppm=0)
    noise = search(time, flux, 0.5, 14)["sde"]
    time, flux = light_curve(depth_ppm=3000)
    assert search(time, flux, 0.5, 14)["sde"] > 2 * noise


def test_the_same_search_twice_gives_the_same_result(fresh_compile_cache):
    time, flux = light_curve()
    assert search(time, flux, 0.5, 14) == search(time, flux, 0.5, 14)


def test_it_agrees_with_astropys_box_least_squares_on_the_same_periods():
    """An independent implementation of the same statistic: the best period is the same and the two spectra rank the periods alike."""
    time, flux = light_curve(depth_ppm=2500, seed=5)
    found = search(time, flux, 0.5, 14, spectrum=True)
    stride = 7
    periods = np.asarray(found["periods"])[::stride]
    reference = BoxLeastSquares(time, flux, dy=np.full(time.size, np.std(flux))).power(periods, [0.04, 0.08, 0.12, 0.2], objective="likelihood", oversample=20)
    assert periods[np.argmax(reference.power)] == pytest.approx(found["period_days"], rel=0.01)
    ranks = lambda a: np.argsort(np.argsort(a))
    own = np.asarray(found["power"])[::stride]
    correlation = np.corrcoef(ranks(own), ranks(reference.power))[0, 1]
    assert correlation > 0.8, f"rank correlation of the spectra {correlation:.2f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_gpu_and_the_cpu_agree_to_rounding(fresh_compile_cache):
    """The sums are exact integers, so the two devices differ only in the last digits of the float arithmetic that follows (a division and a product in float64): a differential test of one function on two devices."""
    time, flux = light_curve(seed=7)
    cpu, gpu = search(time, flux, 0.5, 14, device="cpu", spectrum=True), search(time, flux, 0.5, 14, device="cuda", spectrum=True)
    assert np.allclose(cpu["power"], gpu["power"], rtol=1e-12, atol=1e-12)
    for name in ("period_days", "t0_days", "duration_hr", "depth_ppm", "sde", "snr"):
        assert cpu[name] == pytest.approx(gpu[name], rel=1e-12), name


def test_the_batch_size_changes_the_time_and_not_the_result(fresh_compile_cache):
    """The number of phase bins depends on the period alone, not on the other periods of its batch."""
    time, flux = light_curve(seed=9, days=54.0)
    small, large = search(time, flux, 0.5, 14, spectrum=True, batch=16), search(time, flux, 0.5, 14, spectrum=True, batch=500)
    assert small["power"].tolist() == large["power"].tolist()
    assert {k: v for k, v in small.items() if k not in ("periods", "power")} == {k: v for k, v in large.items() if k not in ("periods", "power")}


@pytest.mark.parametrize("period, duration_hr", [(3.18, 2.5), (0.9, 2.5), (1.4, 4.0)])
def test_the_depth_is_that_of_a_box_against_the_flux_outside_it(period, duration_hr):
    """Against the mean of all the flux, box included, the depth would be low by the share of the points in the box:
    3% at 3.18 d and 12% at 0.9 d for a 2.5 hour box."""
    time, flux = light_curve(period=period, duration_hr=duration_hr, depth_ppm=4000, noise_ppm=100, days=40.0, cadence_min=10.0)
    found = search(time, flux, 0.5, 14)
    assert found["period_days"] == pytest.approx(period, rel=0.005)
    # the box that the search finds is a rounded version of the injected one (the widths are on a grid and the edges on bins), so the depth is allowed 8%; the bias of 3% to 12% is not
    assert found["depth_ppm"] == pytest.approx(4000, rel=0.08), found["depth_ppm"]


def test_input_with_nothing_to_search_is_a_clear_error_and_not_a_crash():
    time, flux = light_curve()
    with pytest.raises(ValueError, match="no finite points"):
        search(np.full(10, np.nan), np.full(10, np.nan), 0.5, 14)
    with pytest.raises(ValueError, match="no finite points"):
        search(np.array([]), np.array([]), 0.5, 14)
    with pytest.raises(ValueError, match="at least"):
        search(time[:5], flux[:5], 0.5, 14)
    with pytest.raises(ValueError, match="no variance"):
        search(time, np.ones(time.size), 0.5, 14)
    with pytest.raises(ValueError, match="Empty period grid"):  # the message of the period grid of TLS
        search(time, flux, 20.0, 30.0)  # the longest period that two transits fit in 27 days is 13.5


def test_a_flux_value_that_would_overflow_the_integer_sums_is_refused_and_not_summed_wrongly():
    """The flux is accumulated as integers of 1e-9: a corrupted point of 1e12 would wrap around and give a wrong spectrum without a warning."""
    time, flux = light_curve()
    corrupt = flux.copy()
    corrupt[100] = 1e12
    with pytest.raises(ValueError, match="too large"):
        search(time, corrupt, 0.5, 14)


def test_a_box_that_holds_all_but_a_few_points_is_not_scored():
    """Three clumps of 0.01 to 0.1 days, 14 days apart, fall into one box at some period, where n = N makes
    n (1 - n / N) zero; a box needs MINIMUM_POINTS points outside it too, so the depth, SNR and SDE stay finite."""
    for width in (0.01, 0.03, 0.1):
        time = np.concatenate([k * 14.0 + np.linspace(0, width, 100) for k in range(3)])
        flux = 1 + np.random.default_rng(1).normal(0, 1e-3, time.size)
        found = search(time, flux, 0.5, 14, spectrum=True, device="cpu")
        assert np.isfinite(found["power"]).all(), width
        assert all(np.isfinite(found[name]) for name in ("depth_ppm", "snr", "sde")), (width, found)
        assert found["sde"] > 0, (width, found)


def test_unsorted_times_give_the_result_of_the_sorted_ones(fresh_compile_cache):
    time, flux = light_curve(seed=4)
    order = np.random.default_rng(0).permutation(time.size)
    assert search(time[order], flux[order], 0.5, 14, spectrum=True)["power"].tolist() == search(time, flux, 0.5, 14, spectrum=True)["power"].tolist()


def test_duplicate_times_and_narrower_dtypes_are_searched():
    time, flux = light_curve(seed=4)
    assert recovered(search(np.concatenate([time, time]), np.concatenate([flux, flux]), 0.5, 14))
    assert recovered(search(time.astype(np.float32), flux.astype(np.float32), 0.5, 14))
    assert recovered(search(time, flux.astype(np.float32), 0.5, 14))


def test_a_barycentric_time_offset_does_not_hide_the_signal():
    """The stack's clock starts at 0, but a caller can pass BTJD or BJD: the fold is in float64 and the epoch is reported on the clock given."""
    time, flux = light_curve(seed=4)
    for offset in (2500.0, 2459000.0):
        found = search(time + offset, flux, 0.5, 14)
        assert found["period_days"] == pytest.approx(3.18, rel=0.01)
        error = (found["t0_days"] - offset - 1.2 + 3.18 / 2) % 3.18 - 3.18 / 2
        assert abs(error) < 2.5 / 48, f"offset {offset}: the time is {error * 24:.2f} h off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_fused_fold_gives_the_indices_of_the_passes_it_replaces():
    """The fold is one compiled kernel on a GPU; it must give the bin of every point for every period exactly as the eager passes do."""
    from transitgpu import core as gpu

    generator = np.random.default_rng(3)
    t = torch.as_tensor(np.sort(generator.uniform(0, 1900, 20000)), device="cuda")
    p = torch.as_tensor(np.sort(generator.uniform(0.6, 14, 37)), device="cuda")
    for bins in (128, 1024, 4096):
        assert torch.equal(gpu.fold(t, p, bins), gpu.fold_index(t, p, bins))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_compilation_that_fails_falls_back_to_the_eager_fold_with_a_warning(monkeypatch):
    from transitgpu import core as gpu

    def refuse(*arguments, **keywords):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(gpu, "_COMPILED", None)
    monkeypatch.setattr(torch, "compile", refuse)
    t = torch.linspace(0, 100, 5000, dtype=torch.float64, device="cuda")
    p = torch.tensor([1.5, 2.5], dtype=torch.float64, device="cuda")
    with pytest.warns(UserWarning, match="compil"):
        index = gpu.fold(t, p, 256)
    assert torch.equal(index, gpu.fold_index(t, p, 256))
    time, flux = light_curve(seed=2)
    assert recovered(search(time, flux, 0.5, 14)), "the search still works on the eager path"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_running_out_of_memory_is_not_taken_for_a_compiler_that_is_missing(monkeypatch):
    """The eager passes would need the same memory, and turning compilation off for the rest of the process would slow
    every later search for a failure that was not the compiler's."""
    from transitgpu import core as gpu

    def exhaust(*arguments, **keywords):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(gpu, "_COMPILED", exhaust)
    t = torch.linspace(0, 100, 5000, dtype=torch.float64, device="cuda")
    p = torch.tensor([1.5, 2.5], dtype=torch.float64, device="cuda")
    with pytest.raises(torch.OutOfMemoryError):
        gpu.fold(t, p, 256)
    assert gpu._COMPILED is exhaust, "the compiled fold is kept"


def test_the_period_uncertainty_is_that_of_tls_applied_to_the_spectrum():
    """The field is the half width at half maximum of the highest peak of the spectrum, as TLS defines it
    (transitleastsquares.stats.period_uncertainty), infinite where the peak is not bracketed."""
    from transitleastsquares.stats import period_uncertainty

    time, flux = light_curve(seed=2, days=54.0)
    found = search(time, flux, 0.5, 14, spectrum=True)
    assert found["period_uncertainty_days"] == period_uncertainty(found["periods"], found["power"])
    assert 0 < found["period_uncertainty_days"] < 0.1 * found["period_days"]


@pytest.mark.parametrize("step_elements", [1, 70_000, 10_000_000])
def test_the_number_of_duration_steps_scored_together_changes_the_time_and_not_the_result(monkeypatch, step_elements, fresh_compile_cache):
    time, flux = light_curve(seed=2)
    expected = search(time, flux, 0.5, 14, spectrum=True)
    monkeypatch.setattr("transitgpu.core.STEP_ELEMENTS", step_elements)
    chunked = search(time, flux, 0.5, 14, spectrum=True)
    assert np.array_equal(chunked["power"], expected["power"]) and all(chunked[name] == expected[name] for name in RESULT_FIELDS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_fused_scoring_of_the_duration_steps_gives_the_spectrum_of_the_eager_passes_to_rounding(monkeypatch):
    """Fusing the passes changes the order of the floating-point operations, so the z-scores differ in the last digits
    and the best period, time, duration, depth and SNR are those of the eager passes."""
    from transitgpu import core as gpu

    time, flux = light_curve(seed=3)
    precision = {"statistic": "float64"}  # to rounding at 64 bits; the default statistic is in 32 bits
    fused = search(time, flux, 0.5, 14, spectrum=True, precision=precision)
    monkeypatch.setattr(gpu, "_SCORE_COMPILED", False)
    eager = search(time, flux, 0.5, 14, spectrum=True, precision=precision)
    assert np.allclose(fused["power"], eager["power"], rtol=1e-12, atol=1e-12)
    assert all(fused[name] == pytest.approx(eager[name], rel=1e-12) for name in RESULT_FIELDS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_scoring_compilation_that_fails_falls_back_to_the_eager_passes_with_a_warning(monkeypatch):
    from transitgpu import core as gpu

    def refuse(*arguments, **keywords):
        raise RuntimeError("no compiler")

    time, flux = light_curve(seed=2)
    expected = search(time, flux, 0.5, 14)
    monkeypatch.setattr(gpu, "_SCORE_COMPILED", None)
    monkeypatch.setattr(torch, "compile", refuse)
    with pytest.warns(UserWarning, match="compil"):
        found = search(time, flux, 0.5, 14)
    assert recovered(found) and found["period_days"] == pytest.approx(expected["period_days"], rel=1e-12)


def test_the_boxes_of_every_period_are_given_when_asked_and_the_best_one_is_the_detection():
    """boxes=True adds the center and the width of the best box of each trial period; at the best period they are the transit time (modulo the period) and the duration of the result."""
    pytest.importorskip("torch")
    from transitgpu.core import search

    time = np.arange(0.0, 27.0, 30.0 / 1440)
    flux = 1 + np.random.default_rng(3).normal(0, 300e-6, time.size)
    flux[np.abs(time - (1.2 + np.round((time - 1.2) / 3.18) * 3.18)) < 2.5 / 48] -= 3000e-6
    found = search(time, flux, 0.7, 12.0, spectrum=True, boxes=True)
    assert "centers" not in search(time, flux, 0.7, 12.0, spectrum=True)
    top = int(np.argmax(found["power"]))
    period = found["periods"][top]
    assert found["centers"].shape == found["periods"].shape == found["widths_days"].shape
    offset = (found["centers"][top] - found["t0_days"] + period / 2) % period - period / 2
    assert abs(offset) < 1e-9 and found["widths_days"][top] * 24 == pytest.approx(found["duration_hr"], rel=1e-12)


def test_the_integer_weights_are_part_of_the_bound_of_the_integer_sums():
    """The integer weights (64 units for a point of the median weight, up to 4,096) multiply the sums, so a flux that
    passes the unweighted bound can still overflow a weighted search."""
    centered = np.full(1_000_000, 2.9e3)  # the unweighted bound is 2.9e18 < 2**62
    assert float(np.abs(centered).max()) / core.QUANTUM * centered.size < 2**62
    with pytest.raises(ValueError, match="weights"):
        core.check_weighted_sums(centered * 0.5, np.full(centered.size, core.WEIGHT_UNIT))
    core.check_weighted_sums(np.full(1000, 1e-3), np.full(1000, core.WEIGHT_UNIT))  # a normalised flux passes
    with pytest.raises(ValueError, match="weights"):
        core.check_weighted_sums(np.full(1_100_000, 1e-3), np.full(1_100_000, core.MAXIMUM_WEIGHT))  # the sum of the weights of a bin would not fit in the 32 bits that hold it


@pytest.mark.gpu
@pytest.mark.parametrize("duration_hours", [2.5, 12.0])
def test_weights_of_ones_give_the_unweighted_search_or_an_error_for_a_flux_of_counts(duration_hours, fresh_compile_cache):
    """2e4 points of a flux of about 2.6e5 counts with a 34% box pass the unweighted bound (1.8e18); with weights of
    ones the search must give the unweighted result or refuse the weights, never wrong sums."""
    n = 20_000
    time = np.linspace(0.0, 1000.0, n)
    flux = 2.6e5 * (1 + 5e-4 * np.random.default_rng(3).normal(size=n))
    flux[np.abs((time - 1.2 + 3.18 / 2) % 3.18 - 3.18 / 2) < duration_hours / 48] -= 0.34 * 2.6e5
    plain = core.search(time, flux, 3.0, 3.4, spectrum=True)
    try:
        weighted = core.search(time, flux, 3.0, 3.4, spectrum=True, weights=np.ones(n))
    except ValueError as error:
        assert "weights" in str(error)
        return
    differing = np.flatnonzero(weighted["power"] != plain["power"])
    batch_plain = core.ONE_PASS_BATCH  # the batch of a search on a device with the one-pass accumulation; the other searches take int(clip(BATCH_ELEMENTS // points, 8, 512)), which is 512 at these 20,000 points
    batch_other = int(np.clip(core.BATCH_ELEMENTS // n, 8, 512))
    detail = (f"depth_ppm plain {plain['depth_ppm']!r} weighted {weighted['depth_ppm']!r}; power differs at {differing.size} of {plain['power'].size} periods"
              f" (first {differing[:5].tolist()}, largest difference {np.abs(weighted['power'] - plain['power']).max():.3e}); periods per batch {batch_plain} (one pass) or {batch_other} (fold table)")
    assert weighted["depth_ppm"] == pytest.approx(plain["depth_ppm"], rel=1e-9) and differing.size == 0, detail


def test_the_tests_that_compare_two_calls_start_from_an_empty_compile_cache(request):
    import inspect

    names = ("test_the_same_search_twice_gives_the_same_result", "test_the_gpu_and_the_cpu_agree_to_rounding", "test_the_batch_size_changes_the_time_and_not_the_result", "test_unsorted_times_give_the_result_of_the_sorted_ones",
             "test_the_number_of_duration_steps_scored_together_changes_the_time_and_not_the_result", "test_weights_of_ones_give_the_unweighted_search_or_an_error_for_a_flux_of_counts")
    assert all("fresh_compile_cache" in inspect.signature(getattr(request.module, name)).parameters for name in names)
