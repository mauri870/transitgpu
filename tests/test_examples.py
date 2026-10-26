"""The example that draws the figure of the README runs headless on a GPU and writes a PNG that is small enough for a README, with both searches finding the period of WASP-18 b. Function tests: the checks are on a file written and on two periods, not on a measurement."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "plot_wasp18.py"


@pytest.mark.gpu
def test_the_plot_example_writes_a_small_png_of_the_shipped_light_curve_and_both_searches_find_the_period_of_wasp_18_b(tmp_path):
    """The examples search on a device: the box search and the template search of transitgpu.tls raise without one."""
    if not torch.cuda.is_available():
        pytest.skip("no device")
    spec = importlib.util.spec_from_file_location("plot_wasp18", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "figure.png"
    box, template = module.draw(output)
    for results in (box, template):
        assert abs(results.period / 0.94145 - 1) < 0.001, "WASP-18 b has a period of 0.94145 days"
    step = 0.94145 * 0.001  # the trial periods of the grid are closer than this near the best period
    assert abs(box.period - template.period) < 2 * step, "both rows find the same period within the grid's step"
    data = output.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and 10_000 < len(data) < 300_000
