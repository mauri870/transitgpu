"""The figure of the screening (paper/scripts/funnel_figure.py): the share of the trial epochs that survives each bound, summed over the template lengths of a range, and the stacked device times of the stages; and the shipped figure the one that the script draws from the shipped files. Function tests on made-up numbers."""
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
spec = importlib.util.spec_from_file_location("funnel_figure", PACKAGE / "paper" / "scripts" / "funnel_figure.py")
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)


def entry(points, span, scale=1):
    stage = lambda tested, live, cs, cheap, tight: {"tested": tested * scale, "columns": tested * scale, "live": live * scale, "cs": cs * scale, "cheap": cheap * scale, "tight": tight * scale}
    return {"points": points, "range": list(span), "stages": {"512 to 1,023": stage(1000, 900, 800, 100, 10), "1,024 to 2,047": stage(3000, 2700, 2000, 500, 20), "4,096 or more": stage(0, 0, 0, 0, 0)}}


STATS = [entry(points, span) for points in (20000, 200000) for span in ((0.5, 1.0), (1.0, 2.0))]
PROFILE = [{"points": points, "device_seconds": 3.0, "stages": {"fold": 1.0, "scans": 0.5, "screening": 1.0, "correlations": 0.3, "rest": 0.2}} for points in (20000, 200000)]


def test_the_survival_is_the_share_of_all_the_trial_epochs_tested_that_pass_each_bound_over_all_template_lengths():
    assert drawing.survival(entry(20000, (0.5, 1.0))) == [90.0, 70.0, 15.0, 0.75]


def test_the_left_panel_has_a_line_for_each_size_and_range_over_the_bounds_and_the_right_one_a_stack_for_each_size(tmp_path):
    figure = drawing.funnel_figure(STATS, PROFILE, tmp_path / "funnel.pdf")
    left, right = figure.axes
    assert (tmp_path / "funnel.pdf").exists() and left.get_yscale() == "log"
    lines = [line for line in left.get_lines() if len(line.get_xdata()) == 4]
    assert len(lines) == 4 and all(len(line.get_ydata()) == 4 for line in lines)
    assert len(right.patches) == 2 * 5, "five stages at each of two sizes"
    top = max(patch.get_y() + patch.get_height() for patch in right.patches)
    assert top == pytest.approx(3.0), "the stack of the stages is the device time"


def test_the_shipped_figure_is_the_one_the_script_draws_from_the_shipped_files(tmp_path):
    done = subprocess.run([sys.executable, str(PACKAGE / "paper" / "scripts" / "funnel_figure.py"), "--out", str(tmp_path / "funnel.pdf")], capture_output=True, text=True, cwd=PACKAGE)
    assert done.returncode == 0, done.stderr[-1500:]
    assert (tmp_path / "funnel.pdf").read_bytes() == (PACKAGE / "paper" / "funnel.pdf").read_bytes(), "paper/funnel.pdf is not what funnel_figure.py draws from screening_stats.json and profile_split.json: run the script again and commit the figure"
