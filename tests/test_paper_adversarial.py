"""The summary of paper/scripts/adversarial.py: the largest relative difference of chi square and of the other fields between two results, and whether the best period is the same. Function tests on small arrays; the cases themselves need a GPU
(tests/test_tls_adversarial.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "adversarial.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("adversarial_summary", SCRIPT)
adversarial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adversarial)


def result(chi2, depth, period):
    return {"chi2": np.array(chi2), "depth": np.array(depth), "period": np.array(period)}


def test_the_largest_relative_difference_of_chi_square_is_read_apart_from_the_other_fields():
    reference = result([10.0, 20.0, 30.0], [0.5, np.nan], 2.5)
    mine = result([10.0, 20.0 * (1 + 3e-9), 30.0], [0.5 * (1 + 2e-4), np.nan], 2.5)
    found = adversarial.summary(reference, mine)
    assert found["chi2_max_relative"] == pytest.approx(3e-9, rel=1e-3) and found["other_fields_max_relative"] == pytest.approx(2e-4, rel=1e-3), found
    assert found["best_period_same"] is True


def test_a_tie_that_changes_the_best_period_is_reported_and_a_shape_that_changes_is_infinite():
    found = adversarial.summary(result([1.0, 2.0], [0.1], 1.3898), result([1.0, 2.0], [0.1], 1.4135))
    assert found["best_period_same"] is False and (found["best_period_tls"], found["best_period_gpu"]) == (1.3898, 1.4135) and found["chi2_max_relative"] == 0.0
    assert adversarial.summary(result([1.0, 2.0], [0.1], 1.0), result([1.0], [0.1], 1.0))["chi2_max_relative"] == float("inf")
