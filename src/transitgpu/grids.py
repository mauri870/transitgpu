"""The period grid, transit durations and peak uncertainty of Transit Least Squares, copied so that TLS is optional.

Source: https://github.com/hippke/tls, version 2.0.1 (11 October 2026): transitleastsquares/grid.py (T14, _density,
duration_limit_masses, _clamp, period_grid), transitleastsquares/stats.py (period_uncertainty) and
transitleastsquares/tls_constants.py (the constants below). TLS is MIT licensed, Copyright (c) 2018 Michael Hippke;
the licence is in THIRD_PARTY/TLS-LICENSE. The period grid is that of Ofir (2014, A&A, 561, A138).

The code is TLS's with its names, without the numba cache decorators, and is tested to give the installed TLS's
numbers to the bit. T14 keeps TLS's numba compilation with fastmath: plain Python differs from it in the last digits
for most inputs, and those digits decide the number of bins and duration steps of a period.
"""

import warnings

import numba
import numpy
from numpy import pi, sqrt

# tls_constants.py of TLS 2.0.1
G = 6.673e-11  # gravitational constant [m^3 / kg / s^2]
R_sun = 695508000  # radius of the Sun [m]
R_jup = 69911000  # radius of Jupiter [m]
M_sun = 1.989 * 10**30  # mass of the Sun [kg]
SECONDS_PER_DAY = 86400
R_STAR = 1.0
M_STAR = 1.0
OVERSAMPLING_FACTOR = 3
N_TRANSITS_MIN = 2
M_STAR_MIN = 0.1
M_STAR_MAX = 1.0
R_STAR_MIN = 0.13
R_STAR_MAX = 3.5
DURATION_GRID_STEP = 1.1
FRACTIONAL_TRANSIT_DURATION_MAX = 0.12
MINIMUM_PERIOD_GRID_SIZE = 100


@numba.jit(fastmath=True, parallel=False, nopython=True)
def T14(R_s, M_s, P, upper_limit=FRACTIONAL_TRANSIT_DURATION_MAX, small=False):
    """Input:  Stellar radius and mass; planetary period
            Units: Solar radius and mass; days
    Output: Maximum planetary transit duration T_14max
            Unit: Fraction of period P"""

    P = P * SECONDS_PER_DAY
    R_s = R_sun * R_s
    M_s = M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * R_jup) * ((4 * P) / (pi * G * M_s)) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result


def _density(M, R):
    return M / (R * R * R)


# Stellar densities (solar units) of the default corners: the smallest star (R_STAR_MIN, M_STAR_MIN) and the largest one (R_STAR_MAX, M_STAR_MAX).
RHO_DEFAULT_MAX = _density(M_STAR_MIN, R_STAR_MIN)
RHO_DEFAULT_MIN = _density(M_STAR_MAX, R_STAR_MAX)


def duration_limit_masses(R_star_min, R_star_max, M_star_min, M_star_max):
    """Masses to pair with R_star_min (shortest) and R_star_max (longest duration).

    T14 ~ R * M^(-1/3) ~ (P / rho)^(1/3). The interval extremes are (R_min, M_max)
    for the shortest and (R_max, M_min) for the longest duration. TLS <= 1.33
    paired (R_min, M_min) and (R_max, M_max) instead (BUGS.md F1), which is too
    narrow for narrow user priors. But for wide intervals the extreme corners are
    unphysical (R 0.13 with M 1.0 is 455 rho_sun). So the extreme density is
    clipped to the more extreme of the old corner and the default corner
    (RHO_DEFAULT_MAX / _MIN). Hence: default limits give the TLS 1.33 grid exactly,
    narrow priors get the full interval range, and no range is ever narrower than
    in TLS 1.33.

    Returns (m_short, m_long).
    """
    # shortest duration: highest density
    rho_cap = max(_density(M_star_min, R_star_min), RHO_DEFAULT_MAX)
    if _density(M_star_max, R_star_min) <= rho_cap:
        m_short = M_star_max
    elif _density(M_star_min, R_star_min) >= RHO_DEFAULT_MAX:
        m_short = M_star_min  # old corner (exact)
    else:
        m_short = RHO_DEFAULT_MAX * R_star_min**3
    # longest duration: lowest density
    rho_floor = min(_density(M_star_max, R_star_max), RHO_DEFAULT_MIN)
    if _density(M_star_min, R_star_max) >= rho_floor:
        m_long = M_star_min
    elif _density(M_star_max, R_star_max) <= RHO_DEFAULT_MIN:
        m_long = M_star_max  # old corner (exact)
    else:
        m_long = RHO_DEFAULT_MIN * R_star_max**3
    return m_short, m_long


def _clamp(name, value, lower, upper):
    if value < lower:
        warnings.warn(f"Warning: {name} was set to {lower} for period_grid (was unphysical: {value})")
        return lower
    if value > upper:
        warnings.warn(f"Warning: {name} was set to {upper} for period_grid (was unphysical: {value})")
        return upper
    return value


def period_grid(
    R_star,
    M_star,
    time_span,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=OVERSAMPLING_FACTOR,
    n_transits_min=N_TRANSITS_MIN,
    _is_fallback=False,
):
    """Returns array of optimal sampling periods for transit search in light curves
    Following Ofir (2014, A&A, 561, A138)"""

    R_star = _clamp("R_star", R_star, 0.01, 10000)
    M_star = _clamp("M_star", M_star, 0.01, 1000)

    R_star = R_star * R_sun
    M_star = M_star * M_sun
    time_span = time_span * SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = 1.0 / (2 * pi) * sqrt(G * M_star / (3 * R_star) ** 3)

    # optimal frequency sampling, Equations (5), (6), (7)
    A = (2 * pi) ** (2.0 / 3) / pi * R_star / (G * M_star) ** (1.0 / 3) / (time_span * oversampling_factor)
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

    X = numpy.arange(N_opt) + 1
    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x

    # Cut to given (optional) selection of periods
    periods = P_x / SECONDS_PER_DAY
    selected_index = numpy.where(numpy.logical_and(periods > period_min, periods <= period_max))
    number_of_periods = numpy.size(periods[selected_index])

    if number_of_periods > 10**6:
        warnings.warn(
            f"period_grid generates a very large grid ({number_of_periods}). "
            "Recommend to check physical plausibility for stellar mass, radius, "
            "and time series duration."
        )

    if number_of_periods < MINIMUM_PERIOD_GRID_SIZE and _is_fallback:
        # The fallback grid (R_star=M_star=1) is still small, e.g. because the
        # user requested a narrow [period_min, period_max]. Return it as is
        # rather than recursing forever or ignoring the user's period range.
        if number_of_periods == 0:
            raise ValueError("Empty period grid. Check period_min, period_max and time span.")
        return periods[selected_index]

    if number_of_periods < MINIMUM_PERIOD_GRID_SIZE:
        if time_span < 5 * SECONDS_PER_DAY:
            time_span = 5 * SECONDS_PER_DAY
        warnings.warn("period_grid defaults to R_star=1 and M_star=1 as given density yielded grid with too few values")
        return period_grid(
            R_star=1,
            M_star=1,
            time_span=time_span / SECONDS_PER_DAY,
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=oversampling_factor,
            n_transits_min=n_transits_min,
            _is_fallback=True,
        )
    return periods[selected_index]  # periods in [days]


def period_uncertainty(periods, power):
    """Half width at half maximum of the highest peak (inf if not bracketed)"""
    try:
        index_highest_power = numpy.argmax(power)
        half = 0.5 * power[index_highest_power]
        # Upper limit
        idx = index_highest_power
        while True:
            idx += 1
            if power[idx] <= half:
                idx_upper = idx
                break
        # Lower limit (negative indices must not wrap around)
        idx = index_highest_power
        while True:
            idx -= 1
            if idx < 0:
                raise IndexError
            if power[idx] <= half:
                idx_lower = idx
                break
        return 0.5 * (periods[idx_upper] - periods[idx_lower])
    except (IndexError, ValueError):
        return float("inf")
