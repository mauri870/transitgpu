"""Injected planets, thresholds and the criterion of recovery for an injection-recovery evaluation (the paper does not report one).

Planet k is drawn from a generator seeded with k, so a planet is the same whichever star it is injected into; a star
takes a block of consecutive k. The priors are log-uniform in the period
(0.5 to 14 d) and in the radius ratio (0.005 to 0.1), and uniform in the impact parameter (0 to 0.9) and in the epoch
(over one period). The star has the solar density, a/R = 4.2 P^(2/3) for P in days (Kepler's third law), and an orbit
with a/R < 2 is drawn again. The limb darkening is quadratic with (0.4, 0.26). The transit is computed with batman
(Kreidberg 2015, https://lweb.cfa.harvard.edu/~lkreidberg/batman/), integrated over the exposure time of the cadence.
"""

import numpy as np

PERIOD_RANGE = (0.5, 14.0)
RADIUS_RATIO_RANGE = (0.005, 0.1)
IMPACT_MAXIMUM = 0.9
LIMB_DARKENING = (0.4, 0.26)
SEMI_MAJOR_AXIS_MINIMUM = 2.0
SOLAR_A_OVER_R_AT_ONE_DAY = 4.2  # a/R of a planet of one day around a star of the solar density
PLANETS = 50  # planets drawn by the tests
PERIOD_TOLERANCE = 0.01  # a planet is found when the period is within this fraction of the injected one
BLOCK_DAYS = 1.0
SHUFFLES = 48


def draw(index):
    """Return planet `index`: period, radius ratio, impact parameter, epoch (a fraction of the period) and a/R."""
    generator = np.random.default_rng(index)
    while True:
        period = float(np.exp(generator.uniform(*np.log(PERIOD_RANGE))))
        ratio = float(np.exp(generator.uniform(*np.log(RADIUS_RATIO_RANGE))))
        impact = float(generator.uniform(0.0, IMPACT_MAXIMUM))
        epoch = float(generator.uniform())
        semi_major = SOLAR_A_OVER_R_AT_ONE_DAY * period ** (2 / 3)
        if semi_major >= SEMI_MAJOR_AXIS_MINIMUM:
            return {"index": index, "period": period, "radius_ratio": ratio, "impact": impact, "epoch_fraction": epoch, "semi_major": semi_major}


def transit_flux(planet, time, start, exposure_days):
    """Return the flux of the planet at `time` (days) and its first transit time, integrated over the exposure.

    The first transit is at `start` plus the epoch fraction of a period.
    """
    import batman

    parameters = batman.TransitParams()
    parameters.t0 = start + planet["epoch_fraction"] * planet["period"]
    parameters.per = planet["period"]
    parameters.rp = planet["radius_ratio"]
    parameters.a = planet["semi_major"]
    parameters.inc = float(np.degrees(np.arccos(planet["impact"] / planet["semi_major"])))
    parameters.ecc, parameters.w = 0.0, 90.0
    parameters.u, parameters.limb_dark = list(LIMB_DARKENING), "quadratic"
    model = batman.TransitModel(parameters, np.asarray(time, dtype=float), supersample_factor=5, exp_time=exposure_days)
    return model.light_curve(parameters), parameters.t0


def duration_days(planet):
    """Return the total duration T14 of a circular transit, in days.

    T14 = P / pi * asin(sqrt((1 + k)^2 - b^2) / (a sin i)), Winn 2010 (arXiv:1001.2010) eq. 14.
    """
    a, b, k, period = planet["semi_major"], planet["impact"], planet["radius_ratio"], planet["period"]
    sin_inclination = np.sqrt(1 - (b / a) ** 2)
    return float(period / np.pi * np.arcsin(np.sqrt((1 + k) ** 2 - b**2) / (a * sin_inclination)))


def found(planet, injected_t0, period, t0, sde, threshold):
    """Return {"right": ..., "found": ...} for a search result of the planet.

    The peak is right when the period is within PERIOD_TOLERANCE and a transit time (t0, modulo the period) within
    half a duration of the injected one; the planet is found when the peak is right and the SDE is at or above the
    threshold.
    """
    duration = duration_days(planet)
    offset = abs(((t0 - injected_t0) + planet["period"] / 2) % planet["period"] - planet["period"] / 2)
    right = bool(abs(period - planet["period"]) <= PERIOD_TOLERANCE * planet["period"] and offset <= duration / 2)
    return {"right": right, "found": bool(right and sde >= threshold)}


def block_shuffles(flux, sector_edges, cadence_days, count, seed=0):
    """Return `count` copies of the flux with its blocks of BLOCK_DAYS shuffled within each sector.

    Copy i uses a generator seeded with seed + i. `sector_edges` are the indices at which the sectors start, plus the
    length.
    """
    block = max(1, int(round(BLOCK_DAYS / cadence_days)))
    copies = []
    for number in range(count):
        generator = np.random.default_rng(seed + number)
        copy = np.array(flux, dtype=float)
        for start, stop in zip(sector_edges[:-1], sector_edges[1:]):
            blocks = (stop - start) // block
            order = generator.permutation(blocks)
            body = np.concatenate([flux[start + i * block:start + (i + 1) * block] for i in order]) if blocks else flux[start:start]
            copy[start:start + body.size] = body
        copies.append(copy)
    return copies


def counts_by_rank(sdes, threshold_sdes, ranks=range(1, 25)):
    """Return the number of planets found with the k-th largest SDE of the shuffles as the threshold, for each k.

    `sdes` are the SDEs of the planets whose peak is right.
    """
    ordered = sorted(threshold_sdes, reverse=True)
    return {k: int(sum(1 for sde in sdes if sde >= ordered[k - 1])) for k in ranks}
