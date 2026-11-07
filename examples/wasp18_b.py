"""Find WASP-18 b in 13 days of TESS 2-minute photometry from sector 2, flattened (see examples/data/README.md), with the box search, the box search with bumps, TLS on the GPU and the hybrid search.

Needs a GPU:

    uv run --group rocm python examples/wasp18_b.py
"""
from pathlib import Path

import numpy as np

import transitgpu
from transitgpu.tls import transitleastsquares

data = np.load(Path(__file__).resolve().parent / "data" / "wasp18_s2.npz")  # time in days (BTJD), flattened PDCSAP flux, error of the flux
t, y, dy = data["time"], data["flux"], data["flux_err"]

results = transitgpu.box_search(t, y, dy).power(period_min=0.5, period_max=5.0)
print(f"period    {results.period:.5f} d")
print(f"duration  {results.duration * 24:.2f} h")
print(f"depth     {(1 - results.depth) * 1e6:.0f} ppm")
print(f"SDE       {results.SDE:.1f}")
print()

results = transitgpu.box_search(t, y, dy).power(period_min=0.5, period_max=5.0, bumps=True)
print(f"bump      {results.bump.period:.5f} d")
print(f"SDE       {results.bump.SDE:.1f}")
print()

results = transitleastsquares(t, y, dy).power(period_min=0.5, period_max=5.0, verbose=False)
print(f"period    {results.period:.5f} d")
print(f"duration  {results.duration * 24:.2f} h")
print(f"depth     {(1 - results.depth) * 1e6:.0f} ppm")
print(f"SDE       {results.SDE:.1f}")
print()

found = transitgpu.bls_tls_hybrid(t, y, dy, period_min=0.5, period_max=5.0, peaks=3)
for peak in found.peaks:  # strongest first; peak.tls is TLS's results object, or None with peak.reason
    if peak.tls is not None:
        print(f"peak {peak.rank}: box {peak.box_period:.5f} d (SDE {peak.box_sde:.1f}), TLS {peak.tls.period:.5f} d, {peak.tls.duration * 24:.2f} h")
