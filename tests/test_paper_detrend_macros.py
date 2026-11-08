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


def results(device=True):
    rows = {"wotan": {"first": 0.2, "median": 0.1913, "largest_relative_difference": 0.0}, "torch-cpu": {"first": 0.6, "median": 0.5, "largest_relative_difference": 3.3e-16, "nan_equal": True}}
    if device:
        rows["kernel"] = {"first": 33.4, "median": 0.0121, "largest_relative_difference": 2.2e-16, "nan_equal": True}
        rows["torch-device"] = {"first": 0.9, "median": 0.0382, "largest_relative_difference": 2.2e-16, "nan_equal": True}
    found = {"versions": {"torch": "2.12.0+rocm7.2", "wotan": "1.10", "triton": None}, "kernel_width_limit": 2048,
             "configurations": [{"step_seconds": 120, "window_days": 1.5, "points": 18934, "padded_width": 2048, "rows": rows},
                                {"step_seconds": 20, "window_days": 0.75, "points": 113600, "padded_width": 4096, "rows": {"wotan": {"first": 1.3, "median": 1.2}, "torch-cpu": {"skipped": "over 60 s"}}}]}
    return found


def test_the_times_the_first_calls_the_widths_the_speedups_the_difference_the_limit_and_the_versions():
    text = fill.detrend_macros(results())
    assert "\\providecommand{\\Dt}" in text
    for expected in ("dt@wotan@120-1.5\\endcsname{191}", "dt@kernel@120-1.5\\endcsname{12.1}", "dt@torch-device@120-1.5\\endcsname{38.2}", "dt@first-kernel@120-1.5\\endcsname{33.4}",
                     "dt@points@120-1.5\\endcsname{18\\,934}", "dt@width@120-1.5\\endcsname{2\\,048}", "dt@speedup-kernel@120-1.5\\endcsname{16}", "dt@speedup-torch-device@120-1.5\\endcsname{5}",
                     "dt@difference@all\\endcsname{$3.3\\times10^{-16}$}", "dt@limit@all\\endcsname{2\\,048}", "dt@torch@version\\endcsname{2.12.0+rocm7.2}", "dt@wotan@version\\endcsname{1.10}"):
        assert "\\csname " + expected in text, expected


def test_a_skipped_row_and_a_run_without_the_device_define_nothing_for_them():
    text = fill.detrend_macros(results(device=False))
    assert "dt@torch-cpu@20-0.75" not in text and "dt@wotan@20-0.75\\endcsname{1\\,200}" in text
    assert "dt@kernel@" not in text and "dt@speedup-" not in text
    assert "dt@triton@version" not in text
