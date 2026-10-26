"""The table and the macros of the split of the device time of exact mode by stage (paper/scripts/split_table.py and fill_numbers.py), from a profile file of the schema of paper/scripts/profile_split.py that these tests make up: for each size the device seconds and the seconds of each stage.
Function tests on synthetic numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "paper" / "scripts"
if not (SCRIPTS / "split_table.py").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_split", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split, fill = load("split_table"), load("fill_numbers")
PROFILE = [{"points": 20000, "device_seconds": 2.04, "stages": {"rest": 0.15, "fold": 0.36, "scans": 0.23, "screening": 1.19, "correlations": 0.11}},
           {"points": 200000, "device_seconds": 10.06, "stages": {"rest": 0.63, "fold": 2.61, "scans": 2.38, "screening": 2.53, "correlations": 1.91}}]


def test_the_table_has_a_row_for_each_size_with_the_seconds_and_the_share_of_each_stage():
    text = split.split_table(PROFILE)
    assert "20\\,000 & 2.04 & 0.36 (18\\%) & 0.23 (11\\%) & 1.19 (58\\%) & 0.11 (5\\%) & 0.15 (7\\%) \\\\" in text, text
    assert "200\\,000 & 10.06 & 2.61 (26\\%) & 2.38 (24\\%) & 2.53 (25\\%) & 1.91 (19\\%) & 0.63 (6\\%) \\\\" in text
    assert text.count("\\\\") == 3 and "\\begin{tabular}" in text


def test_the_macros_give_the_seconds_of_each_stage_at_each_size():
    lines = fill.split_macros(PROFILE)
    for expected in ("sp@20000@fold\\endcsname{0.36}", "sp@200000@fold\\endcsname{2.61}", "sp@20000@screening\\endcsname{1.19}", "sp@200000@screening\\endcsname{2.53}", "sp@200000@total\\endcsname{10.06}"):
        assert "\\csname " + expected in lines, expected
    assert fill.split_macros([]) == ""
