"""The paper's use of the injections into real stars and of the shuffle calibration on real stars (paper/scripts/completeness.py and fill_numbers.py): the fourth panel and table of the figure, the paired differences as the class real, the shares at three signal to noise levels with
their Wilson bounds, and the rank statistics of the calibration. Function tests on made-up files."""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "paper" / "scripts"
if not (SCRIPTS / "completeness.py").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import sys

sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("fill_numbers_real", SCRIPTS / "fill_numbers.py")
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)
import completeness


def cell(found):
    return {"found": found, "n": 180, "wilson": [found / 180 - 0.05, min(1.0, found / 180 + 0.05)]}


def search(scale):
    return {"0.05": {"threshold": 8.7, "threshold_bootstrap": [8.4, 9.1], "by_snr": {"6.0": cell(int(60 * scale)), "8.0": cell(int(140 * scale)), "10.0": cell(int(176 * scale)), "14.0": cell(180)}}}


REAL = {"real": {"injected": 1080, "null": 1212, "searches": {"exact": search(1.0), "box": search(0.9)},
                 "paired": {"0.05": {"box_minus_exact": {"estimate": -0.0278, "low": -0.04, "high": -0.015, "n": 720}}}, "hybrid_seconds": {}}}
MAIN = {"main-white": {"searches": {"exact": search(1.0), "box": search(0.9)}, "paired": {"0.05": {"box_minus_exact": {"estimate": -0.001, "low": -0.005, "high": 0.003, "n": 800}}}}}


def test_the_real_part_is_added_to_the_results_and_gets_a_panel_titled_for_the_stars_and_its_own_table(tmp_path):
    results = completeness.merge_real(MAIN, REAL)
    assert list(results) == ["main-white", "real"] and "real" not in MAIN
    figure = completeness.completeness_figure(results, ["main-white", "real"], "0.05", tmp_path / "c.pdf")
    assert [axis.get_title() for axis in figure.axes if axis.axison] == ["white", "12 TESS stars"]
    assert "exact & 8.7 (8.4 to 9.1) & 33\\% (28 to 38)" in completeness.recall_table(results, "real", "0.05", snr=(6, 8, 10))


def test_the_paired_differences_of_the_real_stars_are_the_class_real_and_enter_the_largest_size():
    lines = fill.recovery_macros(completeness.merge_real(MAIN, REAL))
    assert "pd@real@box@est\\endcsname{$-2.8$}" in lines and "pd@real@box@round\\endcsname{3}" in lines
    assert "\\csname pd@box@maxsize\\endcsname{2.8}" in lines, "the largest size is over the four classes"


def test_the_shares_of_the_real_stars_are_percent_with_wilson_bounds_and_the_planets_a_cell_hold():
    lines = fill.recovery_macros(completeness.merge_real(MAIN, REAL))
    for expected in ("rr@exact@8\\endcsname{78}", "rr@exact@8low\\endcsname{73}", "rr@exact@8high\\endcsname{83}", "rr@exact@10\\endcsname{98}", "rr@box@6\\endcsname{30}", "rr@n@planets\\endcsname{180}", "rr@injected@planets\\endcsname{1080}"):
        assert "\\csname " + expected in lines, expected
    assert "rr@exact@14" not in lines and "rr@exact@4" not in lines, "only the levels of the table"
    assert "rr@" not in fill.recovery_macros(MAIN), "no real part, no shares"


def test_the_rank_statistics_of_the_calibration_on_real_stars_are_the_current_list_and_the_pooled_stars():
    block = lambda stars, mean, ks, p: {"stars": stars, "rank_shares": {}, "mean_rank_share": mean, "kolmogorov": ks, "p": p}
    lines = fill.threshold_macros({"real": {"current_list": block(12, 0.6109715, 0.3090452, 0.164497), "first_list": block(8, 0.67, 0.4, 0.12), "pooled": block(20, 0.63718, 0.26432, 0.1017477)}})
    for expected in ("thr@n\\endcsname{12}", "thr@mean\\endcsname{0.61}", "thr@ks\\endcsname{0.31}", "thr@p\\endcsname{0.16}", "thr@pooledn\\endcsname{20}", "thr@pooledmean\\endcsname{0.64}", "thr@pooledp\\endcsname{0.1}"):
        assert "\\csname " + expected in lines, expected
    assert "thr@" not in fill.threshold_macros({"null": {}}), "no real part, no macros"
