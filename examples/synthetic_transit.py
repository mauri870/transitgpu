"""Recover a synthetic transit: a 3,000 ppm, 2.5 h box every 3.18 days in 27 days of 30-minute data with 300 ppm of white noise.

Needs a GPU and takes a few seconds:

    uv run --group rocm python examples/synthetic_transit.py
"""
import numpy as np

import transitgpu

PERIOD, T0, DURATION_HOURS, DEPTH_PPM, SCATTER_PPM = 3.18, 1.2, 2.5, 3000.0, 300.0


def make_light_curve(seed=7):
    """Return times (days), flux and flux errors with the box transit injected into white noise."""
    t = np.arange(0.0, 27.0, 30.0 / 1440)
    y = 1 + np.random.default_rng(seed).normal(0, SCATTER_PPM * 1e-6, t.size)
    in_transit = np.abs((t - T0 + PERIOD / 2) % PERIOD - PERIOD / 2) < DURATION_HOURS / 48
    y[in_transit] -= DEPTH_PPM * 1e-6
    return t, y, np.full(t.size, SCATTER_PPM * 1e-6)


def main():
    t, y, dy = make_light_curve()
    results = transitgpu.box_search(t, y, dy).power(period_min=1.0, period_max=10.0)
    print(f"period    {results.period:.4f} d   (injected {PERIOD})")
    print(f"T0        {results.T0:.3f} d   (injected {T0}, modulo the period)")
    print(f"duration  {results.duration * 24:.2f} h   (injected {DURATION_HOURS})")
    print(f"depth     {(1 - results.depth) * 1e6:.0f} ppm   (injected {DEPTH_PPM:.0f})")
    print(f"SDE       {results.SDE:.1f}")
    return results


if __name__ == "__main__":
    main()
