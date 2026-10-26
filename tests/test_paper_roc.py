"""The macros \\Roc{part}{search}{rate} (the share found at a false-alarm rate, in percent to one decimal) and \\Rocl, \\Roch (the Wilson bounds) of paper/scripts/fill_numbers.py from paper/roc.json, and the figure of
paper/scripts/roc.py: function tests on a small file of the schema of validation/roc.py, and the file that ships."""
import importlib.util
import json
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
if not (PACKAGE / "paper").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fill = load("fill_numbers_roc", PACKAGE / "paper" / "scripts" / "fill_numbers.py")
drawing = load("roc_figure", PACKAGE / "paper" / "scripts" / "roc.py")

ROC = {"fpr": [0.01, 0.05, 0.5], "parts": {"main-white": {"positives": 800, "nulls": 1000, "searches": {"exact": {"threshold": [9.0, 8.0, 6.0], "tpr": [0.4, 0.5, 0.9], "low": [0.37, 0.47, 0.88], "high": [0.43, 0.53, 0.92]}}},
                                          "real": {"positives": 720, "nulls": 1212, "searches": {"exact": {"threshold": [11.0, 8.7, 6.0], "tpr": [0.25, 0.5, 0.875], "low": [0.22, 0.46, 0.85], "high": [0.28, 0.54, 0.9]}}}}}


def need_matplotlib():
    """The figure tests skip where matplotlib is not installed (the cpu group of the package does not install it; the examples extra does)."""
    matplotlib = pytest.importorskip("matplotlib", reason="the figure needs matplotlib, which the examples extra installs and the cpu group does not")
    matplotlib.use("Agg")


def test_the_share_found_at_the_rates_of_one_and_five_percent_has_a_macro_with_its_wilson_bounds_and_the_part_loses_its_prefix():
    lines = fill.roc_macros(ROC)
    for expected in ("roc@white@exact@0.01\\endcsname{40.0}", "rocl@white@exact@0.01\\endcsname{37.0}", "roch@white@exact@0.05\\endcsname{53.0}", "roc@real@exact@0.05\\endcsname{50.0}", "roc@real@exact@0.01\\endcsname{25.0}"):
        assert "\\csname " + expected in lines or "\\csname " + expected.replace("@0.0", "@.0") in lines or expected in lines, expected
    assert "0.5\\endcsname" not in lines, "only the rates of one and five percent have macros"


def test_a_file_without_a_roc_gives_no_macros():
    assert fill.roc_macros({}) == "" and fill.roc_macros(None) == ""


def test_the_figure_has_a_panel_for_each_part_and_a_line_for_each_search_and_marks_the_rates_of_the_macros(tmp_path):
    need_matplotlib()
    figure = drawing.roc_figure(ROC, tmp_path / "roc.pdf")
    assert (tmp_path / "roc.pdf").exists() and len([axis for axis in figure.axes if axis.axison]) == 2
    assert len(figure.axes[0].get_lines()) >= 1


def test_the_roc_file_that_ships_has_the_four_searches_in_each_of_the_four_parts_and_agrees_with_the_recovery_files_at_five_percent():
    roc = json.loads((PACKAGE / "paper" / "roc.json").read_text())
    recovery = {**json.loads((PACKAGE / "paper" / "recovery.json").read_text()), **json.loads((PACKAGE / "paper" / "recovery_real.json").read_text())}
    assert list(roc["parts"]) == ["main-white", "main-red", "main-variable", "real"]
    for part, content in roc["parts"].items():
        assert list(content["searches"]) == ["exact", "binned", "box", "hybrid5"]
        assert content["ordering"]["0.05"] and content["ordering"]["0.01"]
        if part in recovery:
            for search, entry in content["searches"].items():
                cells = [cell for level, cell in recovery[part]["searches"][search]["0.05"]["by_snr"].items() if float(level) >= roc["snr_low"]]
                found, total = sum(cell["found"] for cell in cells), sum(cell["n"] for cell in cells)
                assert entry["tpr"][roc["fpr"].index(0.05)] == pytest.approx(found / total, abs=1e-12), (part, search)


def test_the_figure_is_a_grid_of_two_columns_at_the_width_of_the_text_with_the_legend_in_the_first_panel_only_and_a_line_style_for_each_search(tmp_path):
    need_matplotlib()
    four = {"fpr": ROC["fpr"], "snr_low": 8.0, "parts": {name: ROC["parts"]["main-white"] for name in ("main-white", "main-red", "main-variable", "real")}}
    two_searches = {**ROC["parts"]["main-white"]["searches"], "box": ROC["parts"]["main-white"]["searches"]["exact"]}
    four["parts"] = {name: {**content, "searches": two_searches} for name, content in four["parts"].items()}
    figure = drawing.roc_figure(four, tmp_path / "roc.pdf")
    assert figure.get_size_inches()[0] <= 7.0 and len(figure.axes) == 4
    assert [axis.get_legend() is not None for axis in figure.axes] == [True, False, False, False]
    assert len({line.get_linestyle() for line in figure.axes[0].get_lines() if line.get_label() in ("exact TLS", "box search")}) == 2
    assert all(label.get_fontsize() >= 8 for axis in figure.axes for label in axis.get_xticklabels() + axis.get_yticklabels())
