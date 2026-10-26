"""The architecture diagram of the searches (paper/scripts/pipeline_figure.py): the host above a boundary and the device below it, the stages of the three searches as rows with the names of Sect. 2 and what runs each, the TLS code marked apart, every text at 7 points or more at 4.3 in, the width it is placed at, and the shipped figure the one that the script draws. Function tests on the specification of the script and on the drawing."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
if not (PACKAGE / "paper").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
spec = importlib.util.spec_from_file_location("pipeline_figure", PACKAGE / "paper" / "scripts" / "pipeline_figure.py")
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)


def flat(text):
    return " ".join(text.split())


def test_the_rows_are_the_three_searches_with_the_stages_of_the_text_in_order():
    stages = {title: [flat(stage) for stage, _ in row] for title, row in drawing.PATHS}
    assert stages == {
        "box search": ["fold to bins", "integer accumulation", "window scores", "best window"],
        "exact mode": ["counting sort", "prefix sums", "cascade U1 U2 U3", "float64 correlation"],
        "binned mode": ["bins per band", "Fourier correlation", "best start"],
    }


def test_each_stage_says_what_runs_it():
    runs = {flat(stage): run for _, row in drawing.PATHS for stage, run in row}
    assert runs["counting sort"] == runs["prefix sums"] == runs["cascade U1 U2 U3"] == runs["float64 correlation"] == "Triton"
    assert runs["fold to bins"] == runs["window scores"] == "torch.compile" and runs["Fourier correlation"] == "torch FFT" and "HIP" in runs["integer accumulation"]


def test_the_figure_has_the_boundary_with_the_host_above_it_and_the_device_below_and_a_box_for_each_stage_and_host_step(tmp_path):
    from matplotlib.patches import FancyBboxPatch

    figure = drawing.pipeline_figure(tmp_path / "pipeline.pdf")
    axis = figure.axes[0]
    boxes = [patch for patch in axis.patches if isinstance(patch, FancyBboxPatch)]
    assert len(boxes) == sum(len(row) for _, row in drawing.PATHS) + len(drawing.HOST_BOXES)
    texts = {text.get_text(): text for text in axis.texts}
    boundary = [line for line in axis.get_lines() if line.get_linestyle() != "-" and tuple(line.get_xdata()) == (0, drawing.WIDTH)]
    assert len(boundary) == 1
    y = boundary[0].get_ydata()[0]
    assert texts["host"].get_position()[1] < y < texts["device"].get_position()[1], "the y axis runs downward"
    host = [patch for patch in boxes if patch.get_y() + patch.get_height() < y]
    device = [patch for patch in boxes if patch.get_y() > y]
    assert len(host) == len(drawing.HOST_BOXES) and len(device) == sum(len(row) for _, row in drawing.PATHS)
    grey = {patch.get_facecolor()[:3] for patch in host if patch.get_facecolor()[:3] == matplotlib.colors.to_rgb(drawing.FILLS[drawing.TLS])}
    assert len(grey) == 1 and sum(patch.get_facecolor()[:3] in grey for patch in host) == 3, "the grids, the batches and the results are TLS code"


def test_every_text_is_7_points_or_more_and_the_figure_is_4_3_inches_wide_the_width_it_is_placed_at(tmp_path):
    figure = drawing.pipeline_figure(tmp_path / "pipeline.pdf")
    assert all(text.get_fontsize() >= 7 for text in figure.axes[0].texts)
    assert figure.get_size_inches()[0] == pytest.approx(4.3, abs=0.01)


def test_the_shipped_figure_is_the_one_the_script_draws(tmp_path):
    done = subprocess.run([sys.executable, str(PACKAGE / "paper" / "scripts" / "pipeline_figure.py"), "--out", str(tmp_path / "pipeline.pdf")], capture_output=True, text=True, cwd=PACKAGE)
    assert done.returncode == 0, done.stderr[-1500:]
    assert (tmp_path / "pipeline.pdf").read_bytes() == (PACKAGE / "paper" / "pipeline.pdf").read_bytes(), "paper/pipeline.pdf is not what pipeline_figure.py draws: run the script again and commit the figure"
