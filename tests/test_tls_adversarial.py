"""transitgpu.tls "gpu" against TLS 2.0.1's exact backend ("fused", on the CPU) on adversarial light curves: peaks that nearly tie, the shortest and the longest templates of the grid, phase ties (duplicated and lattice time stamps), large time offsets
(BJD against BTJD), errors that differ by five orders of magnitude and extreme errors, and tiny and gapped light curves.

Each case is one test. What is compared is what test_tls_golden.py compares: the grids and the integer fields equal; the chi square of every period a relative 1e-6; the best period the same grid point and the fields of the fit at it a relative 1e-6 (the spectra that
divide by the spread of chi square an absolute 1e-5 on top). Where two periods tie to the rounding of chi square (the tiny light curve: 1.4e-13 apart), the best period of the device may be the other one: the case then asserts that the grids and chi square agree at every period and that chi square at the chosen period is the minimum, and compares no field that follows from the best period.
Function tests: the reference is TLS's result, run here."""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transitleastsquares")

import transitgpu.tls  # noqa: E402,F401  (registers the backends "gpu" and "gpu-binned")

pytestmark = pytest.mark.gpu
SPECTRA = ("power", "power_raw", "SR")
OPTIONS = dict(show_progress_bar=False, verbose=False, use_threads=1, oversampling_factor=2, duration_grid_step=1.1)


def box(time, period, t0, duration, depth):
    """1 - depth inside a box transit of the given period, epoch and duration (days)."""
    phase = np.abs((time - t0 + period / 2) % period - period / 2)
    return 1.0 - depth * (phase < duration / 2)


def noisy(time, signal, noise, seed):
    return signal + np.random.default_rng(seed).normal(0.0, noise, time.size)


def tied_peaks():
    """Two planets of one depth and one duration at 2.0 and 3.0 d: the two best periods of the spectrum differ by rounding."""
    time = np.arange(0.0, 30.0, 0.02)
    flux = noisy(time, box(time, 2.0, 0.3, 0.12, 1.5e-3) * box(time, 3.0, 0.9, 0.12, 1.5e-3), 3e-4, 1)
    return time, flux, None, dict(period_min=1.5, period_max=4.0)


def shortest_template():
    """The shortest periods of the grid, where the template holds a few cadences."""
    time = np.arange(0.0, 8.0, 10 / 1440)
    flux = noisy(time, box(time, 0.55, 0.1, 0.03, 2e-3), 5e-4, 2)
    return time, flux, None, dict(period_min=0.5, period_max=0.7)


def longest_template():
    """Periods of a third of the baseline: one to three transits, the longest durations of the grid."""
    time = np.arange(0.0, 90.0, 30 / 1440)
    flux = noisy(time, box(time, 30.0, 12.0, 0.4, 2e-3), 4e-4, 3)
    return time, flux, None, dict(period_min=20.0, period_max=45.0)


def duplicated_times():
    """Every time stamp twice, with independent noise: the points tie in phase at every trial period."""
    time = np.repeat(np.arange(0.0, 20.0, 0.02), 2)
    flux = noisy(time, box(time, 2.5, 0.4, 0.1, 2e-3), 4e-4, 4)
    return time, flux, None, dict(period_min=1.5, period_max=3.5)


def lattice_times():
    """A cadence of 0.02 d and a planet of 1.00 d: every transit falls on the same 50 phases of the lattice, and a trial period that is a multiple of the cadence puts points on the edge of a bin."""
    time = np.arange(0.0, 20.0, 0.02)
    flux = noisy(time, box(time, 1.0, 0.31, 0.1, 2e-3), 4e-4, 5)
    return time, flux, None, dict(period_min=0.9, period_max=1.1)


def offset_case(offset):
    time = np.arange(0.0, 25.0, 0.02) + offset
    flux = noisy(time - offset, box(time - offset, 2.7, 0.5, 0.11, 2e-3), 4e-4, 6)
    return time, flux, None, dict(period_min=1.5, period_max=4.0)


def five_orders_of_errors():
    """Errors from 1e-6 to 1e-1 (log-uniform), the flux scattered by its own error: a few points carry nearly all the weight."""
    time = np.arange(0.0, 25.0, 0.03)
    dy = 10 ** np.random.default_rng(7).uniform(-6, -1, time.size)
    flux = box(time, 2.1, 0.6, 0.12, 5e-2) + np.random.default_rng(8).normal(0, 1, time.size) * dy
    return time, flux, dy, dict(period_min=1.2, period_max=3.5)


def extreme_errors():
    """One error of 1e-9 and one of 1e3 among errors of 5e-4: a point that fixes the model and a point that does not count."""
    time = np.arange(0.0, 25.0, 0.03)
    dy = np.full(time.size, 5e-4)
    dy[100], dy[400] = 1e-9, 1e3
    flux = box(time, 2.1, 0.6, 0.12, 3e-3) + np.random.default_rng(9).normal(0, 1, time.size) * dy
    flux[400] += 5.0
    return time, flux, dy, dict(period_min=1.2, period_max=3.5)


def tiny_light_curve():
    time = np.linspace(0.0, 10.0, 40)
    flux = noisy(time, box(time, 3.0, 0.5, 0.3, 8e-3), 1e-3, 10)
    return time, flux, None, dict(period_min=1.0, period_max=4.0)


def gapped_light_curve():
    """Three gaps that remove 60 percent of the baseline, one with NaN flux and time."""
    time = np.arange(0.0, 60.0, 0.03)
    flux = noisy(time, box(time, 3.3, 1.0, 0.13, 3e-3), 4e-4, 11)
    for low, high in ((8.0, 22.0), (30.0, 41.0), (47.0, 55.0)):
        flux[(time > low) & (time < high)] = np.nan
    flux[5] = np.nan
    return time, flux, None, dict(period_min=1.5, period_max=5.0)


CASES = {
    "nearly_tied_peaks": tied_peaks,
    "shortest_template": shortest_template,
    "longest_template": longest_template,
    "duplicated_time_stamps": duplicated_times,
    "lattice_time_stamps": lattice_times,
    "btjd": lambda: offset_case(0.0),
    "bjd": lambda: offset_case(2459000.0),
    "five_orders_of_errors": five_orders_of_errors,
    "extreme_errors": extreme_errors,
    "tiny_light_curve": tiny_light_curve,
    "gapped_light_curve": gapped_light_curve,
}


def run(name, backend):
    from transitleastsquares import transitleastsquares

    time, flux, dy, window = CASES[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = transitleastsquares(time, flux, dy, verbose=False).power(backend=backend, **OPTIONS, **window)
    return {key: np.asarray(np.nan if value is None else value, dtype=float) for key, value in results.items()}


def disagreements(mine, reference):
    bad = []
    for key, expected in reference.items():
        got = mine[key]
        if got.shape != expected.shape:
            bad.append((key, f"shape {got.shape} != {expected.shape}"))
        elif not np.allclose(got, expected, rtol=1e-6, atol=1e-5 if key in SPECTRA else 1e-12, equal_nan=True):
            bad.append((key, f"max abs difference {np.nanmax(np.abs(got - expected)):.3e}"))
    return bad


@pytest.mark.parametrize("name", list(CASES))
def test_transitgpu_tls_gives_the_results_of_tls_exact_backend_on_an_adversarial_case(name):
    if not torch.cuda.is_available():
        pytest.skip("no device: transitgpu.tls searches on a GPU only")
    reference, mine = run(name, "fused"), run(name, "gpu")
    assert set(mine) == set(reference)
    bad = disagreements(mine, reference)
    if bad and mine["period"] != reference["period"]:  # a tie: the best period is the other of two periods that the exact statistic cannot tell apart
        periods = reference["periods"]
        chi2_there = reference["chi2"][np.argmin(np.abs(periods - mine["period"]))]
        np.testing.assert_allclose(chi2_there, np.nanmin(reference["chi2"]), rtol=1e-6, err_msg=f"{name}: the device chose {float(mine['period'])} and TLS {float(reference['period'])}, and chi square differs there")
        bad = [item for item in bad if item[0] == "periods" or item[0].startswith("chi2")]  # the fit at the other period differs in every field that follows from the best period; the search itself is what must agree
    assert not bad, f"{name}: {bad}"


def test_a_time_offset_of_2459000_days_gives_the_period_and_the_shape_of_the_same_curve_without_it():
    if not torch.cuda.is_available():
        pytest.skip("no device: transitgpu.tls searches on a GPU only")
    plain, shifted = run("btjd", "gpu"), run("bjd", "gpu")
    np.testing.assert_allclose(shifted["period"], plain["period"], rtol=1e-9, err_msg="the period grid is built from the span of the times, which BJD rounds at 1e-10 d")
    np.testing.assert_allclose(shifted["chi2"], plain["chi2"], rtol=1e-6)
    assert abs(float(shifted["T0"]) - 2459000.0 - float(plain["T0"])) < float(plain["duration"]), "the epoch is the same transit"
