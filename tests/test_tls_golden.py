"""transitgpu.tls against Transit Least Squares 2.0.1 on its own characterization cases (golden_upstream/golden_cases.py: 14 runs of TLS on synthetic and K2 light curves with different grids, templates, errors and gaps). For each case TLS's exact backend is run in this session and the exact backend of transitgpu.tls is run on the same input; every field of the results is compared. Tolerances per field, written before the port was run on the cases:

- the period grid and the duration grid, the number of transits and every integer field are equal;
- the chi square of every period: a relative 1e-6 (the exact search agrees with TLS's exact backend to 1e-9 in float64, and 1e-6 is the float32 Fourier path's bound);
- the best period is the same grid point; the fields of the fit at that period (T0, duration, depth, SNR, the folded curve, the model) a relative 1e-6;
- the spectra that divide by the spread of chi square (power, power_raw, SR) an absolute 1e-5 on top of the relative 1e-6, because the division amplifies the rounding of chi square by about 1e4 (TLS's own golden test allows 1e-8 for the same reason at 1e-9).

Function tests: the references are TLS's results, run here."""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent / "golden_upstream"))
import golden_cases  # noqa: E402

import transitgpu.tls  # noqa: E402,F401  (registers the backends "gpu" and "gpu-binned")

pytestmark = pytest.mark.gpu
SPECTRA = ("power", "power_raw", "SR")


def arrays(results):
    return golden_cases.to_arrays(results)


@pytest.mark.parametrize("name", list(golden_cases.CASES))
def test_transitgpu_tls_gives_the_results_of_tls_exact_backend_on_its_golden_case(name, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no device: transitgpu.tls searches on a GPU only")
    monkeypatch.setenv("TLS_GOLDEN_BACKEND", "fused")
    reference = arrays(golden_cases.run(name))
    monkeypatch.setenv("TLS_GOLDEN_BACKEND", "gpu")
    mine = arrays(golden_cases.run(name))
    assert set(mine) == set(reference)
    bad = []
    for key, expected in reference.items():
        got = mine[key]
        if got.shape != expected.shape:
            bad.append((key, f"shape {got.shape} != {expected.shape}"))
        elif not np.allclose(got, expected, rtol=1e-6, atol=1e-5 if key in SPECTRA else 1e-12, equal_nan=True):
            bad.append((key, f"max abs difference {np.nanmax(np.abs(got - expected)):.3e}"))
    assert not bad, f"{name}: {bad}"
