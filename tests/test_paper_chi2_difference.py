"""The figure and the table of the chi-square of exact mode against TLS fused (paper/scripts/chi2_difference.py) from a small file, and the shipped file (paper/chi2_difference.json) against its table."""
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


drawing = load("chi2_difference_figure", PACKAGE / "paper" / "scripts" / "chi2_difference.py")
DATA = {"cases": {"alpha": {"periods": 278, "periods_equal": True, "maximum_relative_difference": 1.8e-11, "compared": 278},
                  "beta_k2": {"periods": 5222, "periods_equal": True, "maximum_relative_difference": 4.0e-12, "compared": 5222}},
        "figure": {"case": "beta_k2", "periods_days": [0.5, 0.6, 0.7, 0.8], "relative_difference": [0.0, 1e-12, -2e-12, 4e-12]}}


def test_the_table_has_a_row_for_each_case_with_its_periods_and_its_largest_difference():
    rows = drawing.table(DATA).splitlines()
    assert any("alpha" in row and "278" in row and "1.8" in row for row in rows) and any("beta\\_k2" in row and "5222" in row for row in rows), rows


def test_a_case_whose_grids_differ_is_marked_in_the_table_and_not_given_a_difference():
    bad = {"cases": {"gamma": {"periods": 10, "periods_equal": False, "maximum_relative_difference": None, "compared": 0}}, "figure": DATA["figure"]}
    row = next(row for row in drawing.table(bad).splitlines() if "gamma" in row)
    assert "grids differ" in row and "e-" not in row, row


def test_the_figure_is_drawn_from_the_file_and_has_the_line_at_zero(tmp_path):
    pytest.importorskip("matplotlib")
    figure = drawing.figure(DATA, tmp_path / "f.pdf")
    axis = figure.axes[0]
    assert (tmp_path / "f.pdf").stat().st_size > 1000 and any(list(line.get_ydata()) == [0.0, 0.0] for line in axis.get_lines()), "a line at zero"


def test_the_shipped_file_names_the_figure_case_among_its_cases_and_every_grid_agrees():
    data = json.loads((PACKAGE / "paper" / "chi2_difference.json").read_text())
    assert data["figure"]["case"] in data["cases"] and len(data["cases"]) == 14, list(data["cases"])
    assert all(case["periods_equal"] for case in data["cases"].values()) and all(case["maximum_relative_difference"] < 1e-9 for case in data["cases"].values())
