"""transitgpu.bls_tls_hybrid: the box search over the whole grid, then TLS's power() on a window around each of the highest peaks. Function tests: the inputs are light curves made here and the checks are on structure,
on a period against the exact template search over the whole grid, and on the reason a peak records when TLS cannot refine it. Tests marked gpu search on the device; without one the public call raises."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import transitgpu  # noqa: E402
from transitgpu.hybrid import Peak, top_peaks  # noqa: E402


def light_curve(period=3.18, t0=1.2, duration_hr=2.5, depth_ppm=3000, noise_ppm=1000, days=27.0, seed=3, cadence_min=30.0):
    time = np.arange(0.0, days, cadence_min / 1440)
    flux = 1 + np.random.default_rng(seed).normal(0, noise_ppm * 1e-6, time.size)
    flux[np.abs(time - (t0 + np.round((time - t0) / period) * period)) < duration_hr / 48] -= depth_ppm * 1e-6
    return time, flux, np.full(time.size, noise_ppm * 1e-6)


def spectrum(heights):
    """A spectrum on a grid of 3000 periods from 0.6 to 14 days: noise near 1 and a peak of each given height (period -> SDE) a few tenths of a percent wide."""
    periods = np.geomspace(0.6, 14.0, 3000)
    power = 1 + 0.1 * np.random.default_rng(0).normal(size=periods.size)
    for period, height in heights.items():
        power += height * np.exp(-0.5 * (np.log(periods / period) / 0.002) ** 2)
    return periods, power


def test_the_peaks_are_the_highest_local_maxima_highest_first_and_a_peak_in_the_window_of_a_higher_one_is_the_same_peak():
    periods, power = spectrum({3.18: 20, 1.59: 15, 6.36: 12, 9.5: 8, 3.20: 18})
    found = top_peaks(periods, power, count=4, window=0.01)
    assert [round(p, 1) for p in found] == [3.2, 1.6, 6.4, 9.5]
    assert sum(abs(p / 3.19 - 1) < 0.01 for p in found) == 1


def test_a_spectrum_without_a_local_maximum_gives_its_highest_point_and_a_bad_count_or_window_is_refused():
    periods = np.linspace(1.0, 2.0, 50)
    power = np.linspace(1.0, 5.0, 50)  # rising to its last point: the last point is a maximum of the padded spectrum
    assert top_peaks(periods, power, count=3, window=0.01) == [2.0]
    with pytest.raises(ValueError, match="count"):
        top_peaks(periods, power, count=0)
    with pytest.raises(ValueError, match="window"):
        top_peaks(periods, power, window=0.6)


def test_without_a_device_the_public_call_raises_and_names_astropys_box_least_squares_before_it_imports_tls(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    t, y, dy = light_curve()
    with pytest.raises(RuntimeError, match="BoxLeastSquares"):
        transitgpu.bls_tls_hybrid(t, y, dy, 0.5, 14)


def test_a_peak_that_tls_cannot_refine_keeps_the_box_values_and_records_the_reason():
    """A light curve of three days has no period grid of TLS around a period of 2.9 days (fewer than two transits fit): TLS raises, the peak keeps the box period and SDE, no TLS results, and the reason names the error."""
    from transitgpu.tls.hybrid import refine_peak

    t = np.arange(0.0, 3.0, 0.02)
    y = np.ones_like(t)
    result, reason = refine_peak(t, y, None, 2.9, 0.01, "fused")
    assert result is None and "ValueError" in reason and reason.strip() != ""
    assert isinstance(Peak(rank=0, box_period=2.9, box_sde=7.0, tls=None, reason=reason).reason, str)


@pytest.mark.gpu
def test_the_refined_period_is_the_one_of_the_exact_template_search_over_the_whole_grid_when_the_window_holds_it():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    from transitgpu.tls import transitleastsquares

    t, y, dy = light_curve(days=60.0, seed=5)
    found = transitgpu.bls_tls_hybrid(t, y, dy, 0.5, 14, peaks=3, window=0.01)
    whole = transitleastsquares(t, y, dy).power(period_min=0.5, period_max=14, show_progress_bar=False, verbose=False, backend="gpu")
    best = found.peaks[0]
    assert best.rank == 0 and best.tls is not None and best.reason is None
    assert abs(best.box_period / whole.period - 1) < 0.01, "the window holds the period of the whole grid"
    assert abs(best.tls.period / whole.period - 1) < 1e-3, f"refined {best.tls.period} against the whole grid {whole.period}"
    assert np.array_equal(found.periods, found.box.periods) and found.sde == pytest.approx(found.box.SDE)


@pytest.mark.gpu
def test_one_peak_gives_one_record_and_the_peaks_come_in_rank_order_of_the_box_spectrum():
    if not torch.cuda.is_available():
        pytest.skip("no device")
    t, y, dy = light_curve(days=60.0, seed=5)
    one = transitgpu.bls_tls_hybrid(t, y, dy, 0.5, 14, peaks=1)
    assert [p.rank for p in one.peaks] == [0]
    several = transitgpu.bls_tls_hybrid(t, y, dy, 0.5, 14, peaks=4)
    sdes = [p.box_sde for p in several.peaks]
    assert [p.rank for p in several.peaks] == list(range(len(several.peaks))) and sdes == sorted(sdes, reverse=True)
    assert several.peaks[0].box_period == one.peaks[0].box_period


@pytest.mark.gpu
def test_a_peak_at_the_edge_of_the_grid_has_a_window_that_leaves_the_range_and_is_refined_or_records_why():
    """TLS's grid stops at 0.6016 d for its default star, so a period of 0.63 d is a peak at the first points of the grid and the window around it starts below the grid."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    t, y, dy = light_curve(period=0.63, t0=0.1, duration_hr=0.8, depth_ppm=6000, days=20.0, seed=2)
    found = transitgpu.bls_tls_hybrid(t, y, dy, 0.5, 14, peaks=3, window=0.05)
    assert found.periods.min() > 0.6, "the grid starts at the limit of TLS's star"
    assert all(p.tls is not None or (p.reason and ":" in p.reason) for p in found.peaks)
    edge = [p for p in found.peaks if p.box_period < 0.7]
    assert edge and edge[0].box_period * (1 - 0.05) < found.periods.min(), "the window of the peak at the edge leaves the range of the box search"


def test_refine_peaks_runs_tls_for_a_peak_only_when_the_peak_is_asked_for(monkeypatch):
    """The generator is lazy: no call of TLS before the first peak is taken, one call for each peak taken and none for the peaks left, so that a caller that stops at the peak it wants pays for no more."""
    import transitgpu.tls.hybrid as tls_hybrid
    from transitgpu.hybrid import refine_peaks

    calls = []

    def spy(t, y, dy, period, window, backend, tls_options):
        calls.append((period, window, backend, tls_options))
        return f"results of {period:.2f}", None

    monkeypatch.setattr(tls_hybrid, "refine_peak", spy)
    periods, power = spectrum({3.18: 20, 1.59: 15, 6.36: 12, 9.5: 8})
    stream = refine_peaks(np.arange(10.0), np.ones(10), None, periods, power, peaks=4, window=0.02, backend="fused", tls_options={"use_threads": 2})
    assert calls == [], "nothing runs before the first peak is taken"
    first = next(stream)
    assert len(calls) == 1 and first.rank == 0 and abs(first.box_period / 3.18 - 1) < 0.01 and first.tls == f"results of {first.box_period:.2f}" and first.reason is None
    assert calls[0][1:] == (0.02, "fused", {"use_threads": 2}), "the window, backend and TLS options reach the call"
    second = next(stream)
    assert len(calls) == 2 and second.rank == 1 and second.box_sde == pytest.approx(power[np.argmin(np.abs(periods - second.box_period))])
    stream.close()
    assert len(calls) == 2, "closing the generator calls TLS for no other peak"


def test_refine_peaks_yields_the_reason_of_a_window_tls_cannot_search_and_goes_on_to_the_next_peak():
    from transitgpu.hybrid import refine_peaks

    t = np.arange(0.0, 3.0, 0.02)
    periods, power = spectrum({2.9: 20, 2.99: 10})
    found = list(refine_peaks(t, np.ones_like(t), None, periods, power, peaks=2, window=0.01, backend="fused"))
    assert [p.rank for p in found] == [0, 1] and all(p.tls is None and "Error" in p.reason for p in found)
