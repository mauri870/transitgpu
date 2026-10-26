"""The figure and the table of the completeness of the searches (paper/scripts/completeness.py), on a results file of the schema of validation/analyse.py that these tests make up: for each part (noise class and baseline) and search, at the false-alarm rates 0.05
and 0.01, the threshold, the found count with its Wilson interval at each signal to noise, and the median run time of the hybrid at each number of peaks. Function tests on synthetic numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "completeness.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
spec = importlib.util.spec_from_file_location("completeness", SCRIPT)
completeness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completeness)

SEARCHES = ("exact", "binned", "box", "hybrid1", "hybrid5", "hybrid10", "hybrid20")
LEVELS = {"6": (12, 40), "8": (24, 40), "10": (36, 40)}


def entry(scale):
    by_snr = {level: {"found": int(found * scale), "n": n, "wilson": [max(0.0, found * scale / n - 0.1), min(1.0, found * scale / n + 0.1)]} for level, (found, n) in LEVELS.items()}
    return {"threshold": 9.5, "threshold_bootstrap": [9.1, 9.9], "by_snr": by_snr, "by_period": {}, "seconds_median": 0.1}


def results():
    searches = {name: {"0.05": entry(1.0 - 0.1 * index / 7), "0.01": entry(0.8 - 0.1 * index / 7)} for index, name in enumerate(SEARCHES)}
    part = {"injected": 120, "null": 1000, "searches": searches, "hybrid_seconds": {"1": {"median": 0.03, "q1": 0.03, "q3": 0.04, "n": 5}, "5": {"median": 0.08, "q1": 0.07, "q3": 0.09, "n": 5}, "10": {"median": 0.13, "q1": 0.12, "q3": 0.14, "n": 5}, "20": {"median": 0.25, "q1": 0.24, "q3": 0.3, "n": 5}}}
    return {"main-white": part, "main-red": part, "main-variable": part}


def test_the_table_gives_the_threshold_and_the_completeness_with_its_interval_at_each_snr_for_each_search():
    text = completeness.recall_table(results(), "main-white", "0.05")
    assert "\\begin{tabular}" in text and text.count("\\\\") == 1 + len(SEARCHES)
    assert "exact & 9.5 (9.1 to 9.9) & 30\\% (20 to 40) & 60\\% (50 to 70) & 90\\% (80 to 100)" in text, text
    assert completeness.recall_table(results(), "main-white", "0.01").count("hybrid20") == 1
    assert completeness.recall_table({"main-white": {"searches": {}}}, "main-white", "0.05").count("\\\\") == 1


def test_the_figure_has_a_panel_for_each_class_with_a_line_and_a_band_for_each_search(tmp_path):
    path = tmp_path / "completeness.pdf"
    figure = completeness.completeness_figure(results(), ["main-white", "main-red", "main-variable"], "0.05", path)
    assert path.stat().st_size > 1000 and len([axis for axis in figure.axes if axis.axison]) == 3
    assert all(len(axis.get_lines()) == len(SEARCHES) and len(axis.collections) == len(SEARCHES) for axis in figure.axes if axis.axison), "a line and a Wilson band for each search"
    assert figure.axes[0].get_ylim()[1] >= 1.0 and "S/N" in figure.axes[0].get_xlabel()


def test_the_hybrid_figure_gives_the_recall_and_the_run_time_against_the_number_of_peaks(tmp_path):
    path = tmp_path / "hybrid.pdf"
    figure = completeness.hybrid_figure(results(), "main-white", "0.05", path, minimum_snr=8)
    recall, seconds = figure.axes
    assert path.stat().st_size > 1000 and [len(line.get_xdata()) for line in recall.get_lines()] == [4] and [len(line.get_xdata()) for line in seconds.get_lines()] == [4]
    assert list(recall.get_lines()[0].get_xdata()) == [1, 5, 10, 20]


def test_the_table_can_be_limited_to_some_signal_to_noise_levels_and_names_them_without_a_decimal():
    text = completeness.recall_table(results(), "main-white", "0.05", snr=(6, 10))
    assert "search & threshold & S/N 6 & S/N 10 \\\\" in text and "S/N 8" not in text and text.count(" & ") >= 3 * (len(SEARCHES) + 1)


def test_the_completeness_figure_is_a_grid_of_two_columns_at_the_width_of_the_text_with_the_legend_in_the_first_panel_only(tmp_path):
    figure = completeness.completeness_figure(results(), ["main-white", "main-red", "main-variable"], "0.05", tmp_path / "c.pdf")
    assert figure.get_size_inches()[0] <= 7.0 and len(figure.axes) == 4 and not figure.axes[3].axison
    assert [axis.get_legend() is not None for axis in figure.axes] == [True, False, False, False]
    assert all(label.get_fontsize() >= 8 for axis in figure.axes[:3] for label in axis.get_xticklabels() + axis.get_yticklabels())
