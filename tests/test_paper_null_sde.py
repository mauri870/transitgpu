"""The figure of the noise-only SDE (paper/scripts/null_sde.py) and its shipped file (paper/null_sde.json): the figure of a small file, and the file that ships against the recovery files and the calibration."""
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


drawing = load("null_sde_figure", PACKAGE / "paper" / "scripts" / "null_sde.py")


def cell(threshold=6.0):
    return {"edges": [2.0, 4.0, 6.0, 8.0], "counts": [10, 80, 10], "threshold_5": threshold}


DATA = {"searches": ["exact", "binned", "box"], "shuffles": 48, "parts": {"main-white": {"nulls": 100, "searches": {"exact": cell(), "binned": cell(7.0), "box": cell(5.0)}, "shuffle_threshold": {"curves": 100, "values": None, "edges": [5.0, 6.0, 7.0], "counts": [60, 40], "median": 5.8}},
                                                                   "real": {"nulls": 100, "searches": {"exact": cell(), "binned": cell(7.0), "box": cell(5.0)}, "shuffle_threshold": {"curves": 3, "values": [5.5, 6.0, 7.0], "median": 6.0}}}}


def need_matplotlib():
    """The figure tests skip where matplotlib is not installed (the cpu group of the package does not install it; the examples extra does)."""
    matplotlib = pytest.importorskip("matplotlib", reason="the figure needs matplotlib, which the examples extra installs and the cpu group does not")
    matplotlib.use("Agg")


def test_the_figure_is_a_grid_of_two_columns_at_the_width_of_the_text_with_a_panel_for_each_part_and_the_legend_in_the_first_panel_only(tmp_path):
    need_matplotlib()
    figure = drawing.null_sde_figure(DATA, tmp_path / "null_sde.pdf")
    assert (tmp_path / "null_sde.pdf").exists() and figure.get_size_inches()[0] <= 7.0
    shown = [axis for axis in figure.axes if axis.axison]
    assert len(shown) == 2 and [axis.get_title() for axis in shown] == ["white", "12 TESS stars"]
    assert [axis.get_legend() is not None for axis in figure.axes][:2] == [True, False]
    white = shown[0]
    assert len(white.patches) >= 4, "a histogram for each search shown and the shuffle thresholds"
    assert all(label.get_fontsize() >= 8 for label in white.get_xticklabels())


def test_the_file_that_ships_has_the_four_searches_in_each_part_and_its_thresholds_are_those_of_the_recovery_files():
    data = json.loads((PACKAGE / "paper" / "null_sde.json").read_text())
    recovery = {**json.loads((PACKAGE / "paper" / "recovery.json").read_text()), **json.loads((PACKAGE / "paper" / "recovery_real.json").read_text())}
    assert list(data["parts"]) == ["main-white", "main-red", "main-variable", "real"] and data["searches"] == ["exact", "binned", "box", "hybrid5"]
    for part, content in data["parts"].items():
        for search in data["searches"]:
            entry = content["searches"][search]
            assert sum(entry["counts"]) == content["nulls"], (part, search)
            assert entry["threshold_5"] == pytest.approx(recovery[part]["searches"][search]["0.05"]["threshold"], abs=1e-9), (part, search)
        assert content["shuffle_threshold"]["curves"] >= 12, part


def test_the_shuffle_thresholds_of_the_shipped_file_are_in_the_range_of_the_nulls_of_the_part_and_the_real_stars_have_a_mark_each():
    data = json.loads((PACKAGE / "paper" / "null_sde.json").read_text())
    real = data["parts"]["real"]["shuffle_threshold"]
    assert real["values"] is not None and len(real["values"]) == real["curves"] == 20
    for part in ("main-white", "main-red", "main-variable"):
        shuffle = data["parts"][part]["shuffle_threshold"]
        assert sum(shuffle["counts"]) == shuffle["curves"] == 2000 and shuffle["values"] is None


def test_the_hybrid_has_the_box_searchs_sde_so_one_histogram_stands_for_both():
    data = json.loads((PACKAGE / "paper" / "null_sde.json").read_text())
    for part, content in data["parts"].items():
        assert content["searches"]["box"]["counts"] == content["searches"]["hybrid5"]["counts"] and content["searches"]["box"]["threshold_5"] == content["searches"]["hybrid5"]["threshold_5"], part
