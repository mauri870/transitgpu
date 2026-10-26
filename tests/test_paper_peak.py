"""The macros of the attained throughput of the GPU (paper/scripts/fill_numbers.py, peak_macros), from a results file of benchmarks/peak.py that these tests make up. Function tests on made-up numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "fill_numbers.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("fill_numbers_peak", SCRIPT)
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)

PEAK = {"fp32_tflops": 35.02, "fp64_tflops": 1.1213, "fp32_over_fp64": 31.23, "copy_gb_per_s": 772.6, "copy_share_of_spec": 0.8048, "clock_mhz": {"min": 122.0, "max": 2943.0}, "other_processes_on_the_device": []}


def test_the_macros_give_the_attained_figures_with_their_roundings():
    lines = fill.peak_macros(PEAK)
    assert "\\providecommand{\\Pk}[1]" in lines
    for expected in ("pk@fp32\\endcsname{35}", "pk@fp64\\endcsname{1.12}", "pk@ratio\\endcsname{31}", "pk@copy\\endcsname{773}", "pk@copyshare\\endcsname{80}", "pk@clockmax\\endcsname{2943}"):
        assert "\\csname " + expected in lines, expected


def test_a_file_without_the_specified_bandwidth_has_no_share_and_no_file_has_no_macros():
    without = {key: value for key, value in PEAK.items() if key != "copy_share_of_spec"}
    assert "pk@copyshare" not in fill.peak_macros(without) and "pk@copy\\endcsname" in fill.peak_macros(without)
    assert fill.peak_macros({}) == ""


def test_a_run_beside_another_process_is_named_in_a_comment_of_the_macros():
    assert "% other processes held the device: 4242" in fill.peak_macros({**PEAK, "other_processes_on_the_device": [4242]})
    assert "other processes held" not in fill.peak_macros(PEAK)
