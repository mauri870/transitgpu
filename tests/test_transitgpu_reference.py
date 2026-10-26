import numpy as np
import pytest

torch = pytest.importorskip("torch")

from transitleastsquares import tls_constants  # noqa: E402
from transitleastsquares.grid import period_grid  # noqa: E402

from transitgpu import core as gpu  # noqa: E402

pytestmark = pytest.mark.gpu  # every test of this module may search on the device


def reference_best_statistic(time, flux, periods, minimum_bins=gpu.MINIMUM_BINS, duration_step=gpu.DURATION_STEP):
    """The specification of transitgpu.core.search in plain numpy, a loop over the periods and the windows: the best box statistic of each period, divided by the variance of the flux.

    Phase fold at the stack's clock, a number of bins that is a power of two from the shortest duration of the period, integer sums of the flux in units of QUANTUM, windows of a geometric grid
    of durations that wrap around the fold, scored as inside^2 / (n (1 - n / N)) for a dip with at least MINIMUM_POINTS points inside the window and as many outside it.
    """
    level = flux.mean()
    quantized = np.round((flux - level) / gpu.QUANTUM).astype(np.int64)
    sigma2 = np.var(flux)
    n_points = time.size
    shortest, longest = gpu.duration_fractions(periods)
    best = np.zeros(periods.size)
    for i, period in enumerate(periods):
        bins = int(np.clip(2 ** np.ceil(np.log2(minimum_bins / shortest[i])), gpu.MINIMUM_BINS_PER_PERIOD, gpu.MAXIMUM_BINS))
        phase = np.mod(time * (1.0 / period), 1.0)
        index = np.clip((phase * bins).astype(np.int64), 0, bins - 1)
        sums = np.zeros(bins, dtype=np.int64)
        np.add.at(sums, index, quantized)
        counts = np.bincount(index, minlength=bins).astype(np.int64)
        steps = max(1, int(np.ceil(np.log(longest[i] / shortest[i]) / np.log(duration_step))) + 1)
        for k in range(steps):
            width = int(np.clip(np.round(shortest[i] * duration_step**k * bins), 1, bins))
            doubled_sums, doubled_counts = np.concatenate([sums, sums[:width]]), np.concatenate([counts, counts[:width]])
            s, c = np.concatenate([[0], np.cumsum(doubled_sums)]), np.concatenate([[0], np.cumsum(doubled_counts)])
            inside_exact = s[width:width + bins] - s[:bins]
            number_exact = c[width:width + bins] - c[:bins]
            inside, number = inside_exact.astype(float) * gpu.QUANTUM, number_exact.astype(float)
            ok = (inside_exact < 0) & (number_exact >= gpu.MINIMUM_POINTS) & (number_exact <= n_points - gpu.MINIMUM_POINTS)
            statistic = np.where(ok, inside * inside / (number * (1 - number / n_points)), 0.0)
            best[i] = max(best[i], statistic.max())
    return best / sigma2


def light_curve(seed, days=20.0, cadence_min=30.0, depth_ppm=2500):
    time = np.arange(0.0, days, cadence_min / 1440)
    flux = 1 + np.random.default_rng(seed).normal(0, 1000e-6, time.size)
    flux[np.abs(np.mod(time - 1.2 + 1.5, 3.18) - 1.5) < 2.5 / 48] -= depth_ppm * 1e-6
    return time, flux


@pytest.mark.parametrize("seed, depth_ppm, cadence_min", [(1, 2500, 30.0), (2, 0, 30.0), (3, 6000, 240.0), (4, 0, 240.0)])
def test_the_code_gives_the_spectrum_of_its_specification(seed, depth_ppm, cadence_min):
    check(seed, depth_ppm, cadence_min, precision={"statistic": "float64"})


@pytest.mark.parametrize("seed, depth_ppm, cadence_min", [(1, 2500, 30.0), (2, 0, 30.0), (3, 6000, 240.0), (4, 0, 240.0)])
def test_the_default_statistic_in_32_bits_gives_the_spectrum_of_its_specification_within_5e_6_on_a_z_score(seed, depth_ppm, cadence_min):
    check(seed, depth_ppm, cadence_min)


@pytest.mark.parametrize("minimum_bins, duration_step", [(6, 1.5), (3, 1.25), (12, 1.25)])
def test_the_bins_and_the_duration_step_are_parameters_of_the_search_and_of_its_specification(minimum_bins, duration_step):
    check(5, 3000, 30.0, precision={"statistic": "float64"}, minimum_bins=minimum_bins, duration_step=duration_step)


def check(seed, depth_ppm, cadence_min, precision=None, **parameters):
    """The torch code and a plain loop in numpy must agree on the statistic of every period to rounding: a differential test of the implementation against what the module's docstring says it is."""
    # A cadence of 4 hours leaves about 120 points: boxes of 3 or 4 points win, which is where MINIMUM_POINTS decides (the dense light curves never do).
    time, flux = light_curve(seed, depth_ppm=depth_ppm, cadence_min=cadence_min)
    periods = np.sort(np.asarray(period_grid(R_star=tls_constants.R_STAR, M_star=tls_constants.M_STAR, time_span=float(time.max() - time.min()), period_min=0.5, period_max=6.0,
                                     oversampling_factor=tls_constants.OVERSAMPLING_FACTOR, n_transits_min=tls_constants.N_TRANSITS_MIN)))
    expected = reference_best_statistic(time, flux, periods, **parameters)
    found = gpu.search(time, flux, 0.5, 6.0, spectrum=True, precision=precision, **parameters)
    assert np.array_equal(found["periods"], periods)
    z = (expected - expected.mean()) / expected.std()
    tolerance = 1e-9 if (precision or {}).get("statistic") == "float64" else 5e-6  # the specification is in 64 bits; the default statistic is in 32 bits, which is within 2e-6 of it on a z-score
    assert np.allclose(found["power"], z, rtol=0, atol=tolerance), f"largest difference {np.abs(found['power'] - z).max():.3g} at period {periods[np.abs(found['power'] - z).argmax()]:.4f}"
    assert found["period_days"] == periods[np.argmax(expected)]


def test_the_snr_squared_is_the_statistic_of_the_best_period():
    """The depth over its error, sigma / sqrt(n (1 - n / N)), is the square root of the box statistic S^2 / (sigma^2 n (1 - n / N)) that is searched: two reported numbers, one identity."""
    time, flux = light_curve(1, cadence_min=30.0)
    found = gpu.search(time, flux, 0.5, 6.0, spectrum=True)
    i = int(np.flatnonzero(found["periods"] == found["period_days"])[0])
    statistic = reference_best_statistic(time, flux, found["periods"][i:i + 1])[0]
    # the depth is rounded to the bin grid through the sum of the box, so the identity holds to the quantum of the flux
    assert found["snr"] ** 2 == pytest.approx(statistic, rel=1e-6)
