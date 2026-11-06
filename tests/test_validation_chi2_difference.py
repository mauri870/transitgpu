"""The comparison of the chi-square of exact mode with TLS fused (validation/chi2_difference.py): the relative difference at the trial periods and the summary of a case. Function tests of the arithmetic, with arrays; the runs
of the 14 characterization cases on a device are the paper's table (paper/chi2_difference.json) and are checked by tests/test_tls_golden.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parent.parent
if not (PACKAGE / "validation").exists():
    pytest.skip("the validation scripts are not in the source distribution", allow_module_level=True)
sys.path.insert(0, str(PACKAGE / "validation"))
spec = importlib.util.spec_from_file_location("chi2_difference", PACKAGE / "validation" / "chi2_difference.py")
chi2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chi2)


def test_the_relative_difference_is_the_other_over_the_reference_less_one_at_every_period():
    reference = np.array([2.0, 4.0, 8.0])
    assert chi2.relative_difference(reference, np.array([2.0, 4.0 * (1 + 1e-9), 8.0 * (1 - 2e-9)])) == pytest.approx([0.0, 1e-9, -2e-9], abs=1e-15)


def test_a_period_where_the_reference_is_not_positive_or_not_finite_has_no_difference():
    found = chi2.relative_difference(np.array([0.0, -1.0, np.nan, 5.0]), np.array([1.0, 1.0, 1.0, 5.0]))
    assert np.isnan(found[:3]).all() and found[3] == 0.0, found


def test_a_case_with_the_same_periods_is_summarised_by_its_largest_difference_in_magnitude():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    summary = chi2.summary(np.array([0.5, 0.6, 0.7, 0.8]), reference, np.array([1.0, 2.0 * (1 - 3e-11), 3.0 * (1 + 1e-11), 4.0]), np.array([0.5, 0.6, 0.7, 0.8]))
    assert summary == {"periods": 4, "periods_equal": True, "maximum_relative_difference": pytest.approx(3e-11, rel=1e-6), "compared": 4}, summary


def test_a_case_whose_period_grids_differ_is_flagged_and_not_compared():
    """The comparison is at the periods of the same grid: a grid that differs by one period is a failure of the case, and the difference is not computed on a mismatch."""
    summary = chi2.summary(np.array([0.5, 0.6]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.5, 0.61]))
    assert summary["periods_equal"] is False and summary["maximum_relative_difference"] is None and summary["compared"] == 0, summary


def test_a_period_with_no_finite_reference_is_not_counted_as_compared():
    summary = chi2.summary(np.array([0.5, 0.6, 0.7]), np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.6, 0.7]))
    assert summary["compared"] == 2 and summary["periods"] == 3 and summary["maximum_relative_difference"] == 0.0, summary
