"""Light curves and injected planets for the injection-recovery evaluation (README.md): the noise classes, the planets with a target signal to noise, the flattening and the classification of a recovered period.

Everything is a function of its arguments and a seed, so a light curve and a planet are the same wherever they are drawn. Self-contained: numpy, batman (Kreidberg 2015,
https://lweb.cfa.harvard.edu/~lkreidberg/batman/) and wotan (Hippke et al. 2019, https://github.com/hippke/wotan).
"""

import numpy as np

CADENCE_DAYS = 2.0 / 1440.0  # the 2-minute cadence of TESS
BASELINE_DAYS = 27.4  # one sector
WHITE_PPM = 1000.0
RED_WHITE_PPM, RED_CORRELATED_PPM, RED_TIMESCALE_DAYS = 700.0, 500.0, 0.3
VARIABLE_AMPLITUDE_PPM, VARIABLE_PERIOD_RANGE = 3000.0, (0.8, 10.0)
FLATTEN_WINDOW_DAYS = 3.0
LIMB_DARKENING = (0.4, 0.26)
IMPACT_MAXIMUM = 0.9
SOLAR_A_OVER_R_AT_ONE_DAY = 4.2  # a / R of a planet of one day around a star of the solar density (Kepler's third law)
PERIOD_TOLERANCE = 0.01
HARMONIC_RATIOS = (1 / 3, 1 / 2, 2 / 3, 3 / 2, 2, 3)
NOISE_CLASSES = ("white", "red", "variable")


GAP_DAYS = 1.0  # between sectors


def time_axis(sectors=1, baseline=BASELINE_DAYS, cadence=CADENCE_DAYS, gap=GAP_DAYS):
    """The times of `sectors` consecutive sectors of `baseline` days at `cadence`, a gap of `gap` days between them, and the index at which each sector starts, plus the length (the edges of a block shuffle)."""
    pieces = [np.arange(0.0, baseline, cadence) + number * (baseline + gap) for number in range(sectors)]
    edges = np.concatenate([[0], np.cumsum([piece.size for piece in pieces])])
    return np.concatenate(pieces), edges


def noise(kind, seed, time=None):
    """Return (time, flux, errors) of a star with no planet. The flux is 1 plus the noise; the errors are the white part of it, as a pipeline reports.

    white: Gaussian noise of WHITE_PPM. red: Gaussian noise of RED_WHITE_PPM plus an Ornstein-Uhlenbeck process of RED_CORRELATED_PPM and a correlation time of RED_TIMESCALE_DAYS (an autoregressive
    step exp(-cadence / timescale)). variable: Gaussian noise of RED_WHITE_PPM plus a rotation, a sinusoid of VARIABLE_AMPLITUDE_PPM at a period drawn log-uniformly from VARIABLE_PERIOD_RANGE and its
    first harmonic at 40 percent of the amplitude, at random phases.
    """
    time = time_axis()[0] if time is None else np.asarray(time, dtype=float)
    generator = np.random.default_rng(seed)
    if kind == "white":
        return time, 1.0 + WHITE_PPM * 1e-6 * generator.standard_normal(time.size), np.full(time.size, WHITE_PPM * 1e-6)
    flux = 1.0 + RED_WHITE_PPM * 1e-6 * generator.standard_normal(time.size)
    if kind == "red":
        step = np.exp(-CADENCE_DAYS / RED_TIMESCALE_DAYS)
        kicks = generator.standard_normal(time.size) * np.sqrt(1 - step**2)
        process = np.empty(time.size)
        process[0] = generator.standard_normal()
        for i in range(1, time.size):
            process[i] = step * process[i - 1] + kicks[i]
        flux += RED_CORRELATED_PPM * 1e-6 * process
    elif kind == "variable":
        period = float(np.exp(generator.uniform(*np.log(VARIABLE_PERIOD_RANGE))))
        first, second = generator.uniform(0, 2 * np.pi, 2)
        flux += VARIABLE_AMPLITUDE_PPM * 1e-6 * (np.sin(2 * np.pi * time / period + first) + 0.4 * np.sin(4 * np.pi * time / period + second))
    else:
        raise ValueError(f"unknown noise class {kind!r}, one of {', '.join(NOISE_CLASSES)}")
    return time, flux, np.full(time.size, RED_WHITE_PPM * 1e-6)


def flatten(time, flux):
    """The flux divided by a biweight filter of FLATTEN_WINDOW_DAYS (wotan), the filter a search of a variable star is run on."""
    from wotan import flatten as wotan_flatten

    return wotan_flatten(time, flux, method="biweight", window_length=FLATTEN_WINDOW_DAYS, return_trend=False, break_tolerance=0.5)


def planet(period, impact, epoch_fraction):
    """A planet of the star of the solar density: its a / R from Kepler's third law, the radius ratio set later by `with_signal_to_noise`."""
    return {"period": float(period), "impact": float(impact), "epoch_fraction": float(epoch_fraction), "semi_major": SOLAR_A_OVER_R_AT_ONE_DAY * float(period) ** (2 / 3), "radius_ratio": 0.05}


def transit_flux(body, time, start=0.0, exposure=CADENCE_DAYS):
    """Return (the flux of the planet at `time`, its first transit time), integrated over the exposure (batman, supersample factor 5)."""
    import batman

    parameters = batman.TransitParams()
    parameters.t0 = start + body["epoch_fraction"] * body["period"]
    parameters.per, parameters.rp, parameters.a = body["period"], body["radius_ratio"], body["semi_major"]
    parameters.inc = float(np.degrees(np.arccos(body["impact"] / body["semi_major"])))
    parameters.ecc, parameters.w = 0.0, 90.0
    parameters.u, parameters.limb_dark = list(LIMB_DARKENING), "quadratic"
    model = batman.TransitModel(parameters, np.asarray(time, dtype=float), supersample_factor=5, exp_time=exposure)
    return model.light_curve(parameters), parameters.t0


def duration_days(body):
    """The total duration T14 of a circular transit (Winn 2010, arXiv:1001.2010, equation 14), in days."""
    a, b, k, period = body["semi_major"], body["impact"], body["radius_ratio"], body["period"]
    return float(period / np.pi * np.arcsin(np.sqrt((1 + k) ** 2 - b**2) / (a * np.sqrt(1 - (b / a) ** 2))))


def signal_to_noise(body, time, sigma):
    """The signal to noise of the transits of a planet on a time axis: sqrt(sum of (1 - model)^2) / sigma, with sigma the noise per point (`effective_sigma` for a light curve with correlated noise)."""
    model, _ = transit_flux(body, time)
    return float(np.sqrt(np.sum((1.0 - model) ** 2)) / sigma)


def effective_sigma(flux, duration_days, cadence=CADENCE_DAYS):
    """The noise per point that a transit of `duration_days` sees: 1.4826 times the median absolute deviation of the means of the flux over non-overlapping windows of one duration, times the square root of the number of cadences in a window. The flux is a filtered light curve with no planet. For white noise it is the standard deviation per point; for correlated noise it is larger, since the mean over a window does not fall as the root of the points."""
    length = max(2, int(round(duration_days / cadence)))
    flux = np.asarray(flux, dtype=float)
    flux = flux[np.isfinite(flux)]
    windows = flux[: flux.size // length * length].reshape(-1, length).mean(axis=1)
    return float(1.4826 * np.median(np.abs(windows - np.median(windows))) * np.sqrt(length))


def with_signal_to_noise(body, time, sigma, target, iterations=6):
    """The planet with the radius ratio that gives the target signal to noise (signal_to_noise) on the time axis; the radius ratio is found by scaling it by the square root of the ratio of the targets (the depth goes as its square)."""
    body = dict(body)
    for _ in range(iterations):
        current = signal_to_noise(body, time, sigma)
        body["radius_ratio"] = float(np.clip(body["radius_ratio"] * (target / current) ** 0.5, 1e-4, 0.5))
    return body


def inject(body, time, flux):
    """Return (the flux with the planet, the first transit time)."""
    model, first = transit_flux(body, time)
    return flux * model, first


def classify(period, t0, body, first_transit):
    """"right" when the period is within PERIOD_TOLERANCE of the planet's and the transit time within half a duration of a transit; "harmonic" when the period is within PERIOD_TOLERANCE of 1/3, 1/2, 2/3, 3/2, 2 or 3 times the planet's; else "wrong". The epoch of a harmonic is not asked."""
    if not np.isfinite(period) or period <= 0:
        return "wrong"
    ratio = period / body["period"]
    if abs(ratio - 1.0) <= PERIOD_TOLERANCE:
        offset = abs(((t0 - first_transit) + body["period"] / 2) % body["period"] - body["period"] / 2)
        return "right" if offset <= duration_days(body) / 2 else "wrong"
    if any(abs(ratio - r) <= PERIOD_TOLERANCE * r for r in HARMONIC_RATIOS):
        return "harmonic"
    return "wrong"


def block_shuffle(flux, edges, copy, cadence=CADENCE_DAYS, block_days=1.0):
    """The flux with its blocks of `block_days` shuffled within each sector (edges as time_axis returns them); copy 0 is the flux itself, copy i a permutation drawn by a generator seeded with i."""
    flux = np.asarray(flux, dtype=float)
    if copy == 0:
        return flux.copy()
    generator = np.random.default_rng(copy)
    block = max(1, int(round(block_days / cadence)))
    out = flux.copy()
    for start, stop in zip(edges[:-1], edges[1:]):
        count = (stop - start) // block
        if count < 2:
            continue
        order = generator.permutation(count)
        body = np.concatenate([flux[start + i * block:start + (i + 1) * block] for i in order])
        out[start:start + body.size] = body
    return out
