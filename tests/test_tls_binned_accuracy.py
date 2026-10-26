"""How far the approximate searches are from the exact backend of TLS on its 14 characterization cases (golden_upstream/golden_cases.py): the best period, the SDE and, for the binned search, the cases it refuses.

The binned search of transitgpu.tls ("gpu-binned") folds each period into phase bins narrower than the shortest transit duration and needs equal errors; TLS's approximate backend "fused-pl" fits a piecewise-linear template to the folded points. Function tests: the references are TLS's results, run here."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent / "golden_upstream"))
import golden_cases  # noqa: E402

import transitgpu.tls  # noqa: E402,F401  (registers the backends "gpu" and "gpu-binned")

pytestmark = pytest.mark.gpu


def best_period_and_sde(name, backend, monkeypatch):
    monkeypatch.setenv("TLS_GOLDEN_BACKEND", backend)
    arrays = golden_cases.to_arrays(golden_cases.run(name))
    return float(arrays["period"]), float(arrays["SDE"])


def test_gpu_binned_finds_the_period_of_the_exact_backend_on_10_of_12_cases_it_accepts_and_fused_pl_on_all(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no device: transitgpu.tls searches on a GPU only")
    refused, binned_differs, binned_sde, fused_pl_differs, fused_pl_sde = [], [], [], [], []
    for name in golden_cases.CASES:
        exact_period, exact_sde = best_period_and_sde(name, "fused", monkeypatch)
        pl_period, pl_sde = best_period_and_sde(name, "fused-pl", monkeypatch)
        fused_pl_differs += [name] if not (pl_period == exact_period or (np.isnan(pl_period) and np.isnan(exact_period))) else []
        fused_pl_sde.append(abs(pl_sde - exact_sde) if np.isfinite(exact_sde) else 0.0)
        try:
            period, sde = best_period_and_sde(name, "gpu-binned", monkeypatch)
        except ValueError as error:
            assert "equal errors" in str(error)
            refused.append(name)
            continue
        binned_differs += [name] if not (period == exact_period or (np.isnan(period) and np.isnan(exact_period))) else []
        binned_sde.append(abs(sde - exact_sde) if np.isfinite(exact_sde) else 0.0)
    searched = len(golden_cases.CASES) - len(refused)
    assert (len(golden_cases.CASES), len(refused), searched) == (14, 2, 12), refused
    assert len(binned_differs) == 2, binned_differs  # syn_earth_gap and short_lc
    assert fused_pl_differs == [], fused_pl_differs
    assert max(binned_sde) == pytest.approx(0.870, abs=0.005) and max(fused_pl_sde) == pytest.approx(0.0648, abs=0.002), (max(binned_sde), max(fused_pl_sde))  # the SDE of gpu-binned is off by 0.88 at most, TLS's fused-pl's by 0.065
