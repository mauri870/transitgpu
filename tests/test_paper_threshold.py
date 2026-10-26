"""The table and the macros of the shuffle calibration (paper/scripts/threshold_table.py and fill_numbers.py), on an analysis file of the schema of validation/threshold_analysis.py that these tests make up: an aggregate for each noise class of the share of curves
whose best peak exceeds the threshold of their 48 shuffles, against the nominal 1/49, the same for the inverted-light-curve null, and the recovery of injected planets above each threshold. Function tests on synthetic numbers."""
import importlib.util
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parent.parent / "paper" / "scripts"
if not (PAPER / "threshold_table.py").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name, PAPER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table, fill = load("threshold_table"), load("fill_numbers")


def share(count, total, low, high, **extra):
    return {"count": count, "of": total, "share": count / total, "low": low, "high": high, **extra}


ANALYSIS = {
    "null": {"white": {"curves": 2000, "above": 40, "ties": 0, **share(40, 2000, 0.0151, 0.0278, nominal=1 / 49)}, "red": {"curves": 1000, "above": 70, "ties": 1, **share(70, 1000, 0.0558, 0.0874, nominal=1 / 49)}},
    "inverted": {"white": {"curves": 2000, "above": 38, **share(38, 2000, 0.0139, 0.0260), "thresholds_differ_by_more_than_10_percent": share(300, 2000, 0.135, 0.166)}, "red": {"curves": 1000, "above": 90, **share(90, 1000, 0.0738, 0.1093), "thresholds_differ_by_more_than_10_percent": share(500, 1000, 0.469, 0.531)}},
    "injection": {"white": {"6": {"cases": 100, "right": 20, "harmonic": 5, "found_shuffle": 18, "found_inverted": 17, "found_shuffle_share": share(18, 100, 0.117, 0.268), "found_inverted_share": share(17, 100, 0.109, 0.255), "threshold_raised_by_planet": 3, "threshold_ratio_median": 1.01, "paired": 100}}},
}


def test_the_null_table_has_a_row_for_each_class_with_the_observed_share_its_interval_and_the_nominal_rate():
    text = table.null_table(ANALYSIS)
    assert "\\begin{tabular}" in text and text.count("\\\\") >= 3
    assert re.search(r"white & 2\\,000 & 40 & 2\.0\\% \(1\.5 to 2\.8\) & 2\.0\\% &", text), text
    assert re.search(r"red & 1\\,000 & 70 & 7\.0\\% \(5\.6 to 8\.7\) & 2\.0\\% &", text), text
    assert "9.0\\% (7.4 to 10.9)" in text and "36" not in text.split("white")[0]


def test_the_injection_table_gives_the_recovery_above_the_shuffle_and_the_inverted_threshold_for_each_class_and_snr():
    text = table.injection_table(ANALYSIS)
    assert re.search(r"white & 6 & 100 & 20 & 5 & 18\.0\\% \(11\.7 to 26\.8\) & 17\.0\\% \(10\.9 to 25\.5\)", text), text
    assert table.injection_table({"injection": {}}).count("\\\\") == 1, "no class: the header row only"


def test_the_macros_of_the_calibration_give_the_share_and_the_interval_of_each_class_and_are_absent_without_an_analysis():
    lines = fill.threshold_macros(ANALYSIS)
    assert "\\expandafter\\def\\csname th@white@share\\endcsname{2.0}" in lines and "\\expandafter\\def\\csname th@red@low\\endcsname{5.6}" in lines and "\\expandafter\\def\\csname th@red@high\\endcsname{8.7}" in lines
    assert "\\expandafter\\def\\csname th@red@above\\endcsname{70}" in lines and "\\expandafter\\def\\csname thi@red@share\\endcsname{9.0}" in lines
    assert fill.threshold_macros({}) == ""


def test_the_classes_are_listed_white_red_variable_whatever_the_order_of_the_file_and_counts_have_thousands_separators():
    analysis = {"null": {"variable": {**ANALYSIS["null"]["red"], "count": 1325, "of": 2000}, "red": ANALYSIS["null"]["red"], "white": ANALYSIS["null"]["white"]}}
    text = table.null_table(analysis)
    assert text.index("white &") < text.index("red &") < text.index("variable &") and "1\\,325" in text


def test_the_macros_of_the_rotation_crossings_and_of_the_false_alarm_quantiles_follow_the_file():
    analysis = {**ANALYSIS, "variable_rotation": {"curves": 2000, "crossings": 1325, "crossings_explained": share(1264, 1325, 0.9413, 0.9640), "crossings_not_explained": 61, "not_crossing": 675, "not_crossing_explained": share(434, 675, 0.6061, 0.6782)},
                "false_alarm_quantiles_of_a_threshold_of_48": {"0.5": 0.0143, "0.9": 0.0468, "0.99": 0.0914}}
    lines = fill.threshold_macros(analysis)
    for expected in ("vr@explained@count\\endcsname{1\\,264}", "vr@explained@share\\endcsname{95.4}", "vr@unexplained@count\\endcsname{61}", "vr@unexplained@share\\endcsname{3.05}", "vr@unexplained@low\\endcsname{2.4}", "vr@unexplained@high\\endcsname{3.9}",
                     "beta@median\\endcsname{1.4}", "beta@p90\\endcsname{4.7}", "beta@p99\\endcsname{9.1}", "vr@rest@share\\endcsname{64.3}"):
        assert "\\csname " + expected in lines, expected
    assert "vr@" not in fill.threshold_macros(ANALYSIS) and "beta@" not in fill.threshold_macros(ANALYSIS), "no rotation block and no quantiles: no such macros"
