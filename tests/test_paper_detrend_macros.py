"""The macros of the detrend benchmark (paper/scripts/fill_numbers.py, detrend_macros) on a made-up results file. Function tests on made-up numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "fill_numbers.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("fill_numbers_detrend", SCRIPT)
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)


def results(device=True, shared=True):
    rows = {"wotan": {"median": 0.1913, "largest_relative_difference": 0.0}, "torch-cpu": {"median": 0.5, "largest_relative_difference": 3.3e-16, "nan_equal": True}}
    if device:
        rows["device"] = {"median": 0.0121, "largest_relative_difference": 2.2e-16, "nan_equal": True}
    found = {"versions": {"torch": "2.12.0+rocm7.2", "wotan": "1.10", "triton": None}, "cadences": [{"step_seconds": 120, "points": 18934, "rows": rows},
                                                                                                     {"step_seconds": 20, "points": 113600, "rows": {"wotan": {"median": 1.2}, "torch-cpu": {"skipped": "over 60 s"}}}]}
    if shared:
        found["shared"] = [{"processes": 1, "per_second": 347.2}, {"processes": 4, "per_second": 420.4}]
    return found


def test_the_times_the_points_the_speedup_the_difference_the_shared_rates_and_the_versions():
    text = fill.detrend_macros(results())
    assert "\\providecommand{\\Dt}" in text
    for expected in ("dt@wotan@120\\endcsname{191}", "dt@device@120\\endcsname{12.1}", "dt@torch-cpu@120\\endcsname{500}", "dt@points@120\\endcsname{18\\,934}",
                     "dt@speedup@120\\endcsname{16}", "dt@difference@all\\endcsname{$3.3\\times10^{-16}$}", "dt@shared@4\\endcsname{420}", "dt@each@4\\endcsname{105}",
                     "dt@torch@version\\endcsname{2.12.0+rocm7.2}", "dt@wotan@version\\endcsname{1.10}"):
        assert "\\csname " + expected in text, expected


def test_a_skipped_row_a_run_without_the_device_and_one_without_shared_processes_define_nothing_for_them():
    text = fill.detrend_macros(results(device=False, shared=False))
    assert "dt@torch-cpu@20" not in text and "dt@wotan@20\\endcsname{1200}" in text
    assert "dt@device@" not in text and "dt@speedup@" not in text and "dt@shared@" not in text
    assert "dt@triton@version" not in text
