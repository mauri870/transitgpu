"""The macros of the box searches and exact mode on the variable class (paper/scripts/fill_numbers.py, box_variable_macros), from a file of the schema of validation/box_variable.py that these tests make up. Function tests on made-up numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "fill_numbers.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("fill_numbers_box", SCRIPT)
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)


def cell(found, threshold):
    return {"threshold": threshold, "found": found, "n": 200, "wilson": [max(0.0, found / 200 - 0.03), found / 200 + 0.04]}


BOX = {"searches": {"box": {"0.05": cell(7, 7.8486)}, "box_detrended": {"0.05": cell(77, 8.9657)}, "astropy": {"0.05": cell(5, 7.918)}, "exact": {"0.05": cell(99, 9.2662)}},
       "paired": {"0.05": {"box_minus_exact": {"estimate": -0.46, "low": -0.529, "high": -0.392}, "box_detrended_minus_exact": {"estimate": -0.11, "low": -0.1667, "high": -0.0609},
                           "box_detrended_minus_box": {"estimate": 0.35, "low": 0.287, "high": 0.418}, "astropy_minus_box": {"estimate": 0.0, "low": -0.0188, "high": 0.0188}}}}


def test_each_search_has_the_count_the_share_the_wilson_interval_and_the_threshold():
    text = fill.box_variable_macros(BOX)
    assert "\\providecommand{\\Bv}" in text and "\\providecommand{\\Bvd}" in text
    for expected in ("bv@exact@found\\endcsname{99}", "bv@exact@share\\endcsname{49.5}", "bv@exact@low\\endcsname{46.5}", "bv@exact@high\\endcsname{53.5}", "bv@exact@threshold\\endcsname{9.27}",
                     "bv@box@share\\endcsname{3.5}", "bv@astropy@found\\endcsname{5}", "bv@box_detrended@found\\endcsname{77}", "bv@detrended@found\\endcsname{77}", "bv@detrended@threshold\\endcsname{8.97}"):
        assert "\\csname " + expected in text, expected


def test_a_difference_is_signed_points_under_the_name_of_the_file_and_the_short_name():
    text = fill.box_variable_macros(BOX)
    for expected in ("bvd@box_minus_exact@est\\endcsname{$-46.0$}", "bvd@box-exact@est\\endcsname{$-46.0$}", "bvd@box-exact@low\\endcsname{$-52.9$}", "bvd@box-exact@high\\endcsname{$-39.2$}", "bvd@detrended-exact@est\\endcsname{$-11.0$}",
                     "bvd@detrended-box@est\\endcsname{$+35.0$}", "bvd@box_detrended_minus_box@high\\endcsname{$+41.8$}", "bvd@astropy-box@est\\endcsname{$0.0$}"):
        assert "\\csname " + expected in text, expected


def test_no_file_gives_no_macros():
    assert fill.box_variable_macros({}) == ""
