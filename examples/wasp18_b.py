"""Find WASP-18 b in 13 days of TESS 2-minute photometry from sector 2, flattened (see examples/data/README.md).

Needs a GPU:

    uv run --group rocm python examples/wasp18_b.py
"""
from pathlib import Path

import numpy as np

import transitgpu

data = np.load(Path(__file__).resolve().parent / "data" / "wasp18_s2.npz")  # time in days (BTJD), flattened PDCSAP flux, error of the flux
t, y, dy = data["time"], data["flux"], data["flux_err"]
results = transitgpu.box_search(t, y, dy).power(period_min=0.5, period_max=5.0)
print(f"period    {results.period:.5f} d")
print(f"T0        {results.T0:.4f} BTJD")
print(f"duration  {results.duration * 24:.2f} h")
print(f"depth     {(1 - results.depth) * 1e6:.0f} ppm")
print(f"SDE       {results.SDE:.1f}")
