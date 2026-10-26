"""benchmarks/peak.py attains the throughput of a device with kernels of fused multiply-adds and a copy; its arithmetic and its refusal to run without a device are checked here, the throughput itself needs a GPU. Function tests."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "peak.py"
spec = importlib.util.spec_from_file_location("transitgpu_peak", SCRIPT)
peak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peak)


def test_the_figures_are_flops_over_seconds_in_tera_and_bytes_over_seconds_in_giga():
    assert peak.tflops(2 * 8 * 2048 * (1 << 22), 1.0) == pytest.approx(2 * 8 * 2048 * (1 << 22) / 1e12)
    assert peak.tflops(61.4e12, 1.0) == pytest.approx(61.4) and peak.gigabytes_per_second(960e9, 1.0) == pytest.approx(960.0)
    assert peak.tflops(1e12, 0.5) == pytest.approx(2.0) and peak.gigabytes_per_second(2e9, 4.0) == pytest.approx(0.5)


def test_the_script_refuses_to_run_without_a_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no GPU"):
        peak.main([])


def test_a_kernel_that_was_too_short_is_lengthened_to_the_target_and_a_long_one_is_left_alone():
    assert peak.iterations_for(2048, 0.0044) == int(2048 * 0.25 / 0.0044)
    assert peak.iterations_for(2048, 1.0) == 2048, "never fewer iterations than asked"
    assert peak.iterations_for(100, 0.25) == 100 and peak.iterations_for(100, 0.025, target=0.25) == 1000


def test_the_summary_gives_the_ratio_the_clock_range_the_share_of_the_specified_bandwidth_and_the_other_holders():
    fp32 = {"tflops": 35.0, "best_seconds": 0.2, "clock_min": 122.0, "clock_max": 2871.0}
    fp64 = {"tflops": 1.12, "best_seconds": 0.24, "clock_min": 2394.0, "clock_max": 2943.0}
    out = peak.summary("gpu", "7.2", fp32, fp64, {"gb_per_s": 773.0, "best_seconds": 0.01}, [123], spec_bandwidth=960.0)
    assert out["fp32_over_fp64"] == pytest.approx(31.25) and out["clock_mhz"] == {"min": 122.0, "max": 2943.0} and out["other_processes_on_the_device"] == [123]
    assert out["copy_share_of_spec"] == pytest.approx(773 / 960) and out["kernel_seconds"] == {"fp32": 0.2, "fp64": 0.24, "copy": 0.01}
    assert "copy_share_of_spec" not in peak.summary("gpu", "7.2", fp32, fp64, {"gb_per_s": 773.0, "best_seconds": 0.01}, [])


def test_the_holders_of_the_device_are_the_processes_with_dev_kfd_open_without_this_one(tmp_path):
    for pid, target in ((101, "/dev/kfd"), (102, "/dev/null"), (103, "/dev/kfd"), (104, "/dev/dri/renderD128")):
        directory = tmp_path / str(pid) / "fd"
        directory.mkdir(parents=True)
        (directory / "3").symlink_to(target)
    (tmp_path / "self").mkdir()
    assert peak.device_holders(proc=str(tmp_path), own=103) == [101]
    assert peak.device_holders(proc=str(tmp_path), own=999) == [101, 103]
