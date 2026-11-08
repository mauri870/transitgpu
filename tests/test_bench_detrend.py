"""benchmarks/bench_detrend.py: its light curves, its small runs on the CPU, and the README's table of detrend times equal to the table of the stored results. Function tests: they check the structure and the rule of the table, not the seconds."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("wotan")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transitgpu_bench_detrend", ROOT / "benchmarks" / "bench_detrend.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_the_light_curve_is_the_same_for_the_same_cadence_and_has_the_gap_the_nan_and_the_transits_of_the_mask():
    time, flux, mask = bench.light_curve(600)
    again = bench.light_curve(600)
    assert np.array_equal(time, again[0]) and np.array_equal(flux, again[1], equal_nan=True) and np.array_equal(mask, again[2])
    gap = np.diff(time).max()
    assert 0.8 < gap < 1.5, gap
    assert np.isnan(flux).any() and mask.any() and not mask.all() and bool(np.all(np.diff(time) > 0))


def test_the_padded_width_of_the_configurations_is_that_of_the_windows_rounded_up_to_a_power_of_two():
    widths = {(step, window): bench.padded_width(*bench.light_curve(step), window) for _, step, window in bench.CONFIGURATIONS}
    assert widths[(120, 0.75)] == 1024 and widths[(120, 1.5)] == 2048 and widths[(20, 0.75)] == 4096 and widths[(1800, 0.75)] == 64, widths


def test_a_small_run_on_the_cpu_has_a_row_for_wotan_and_for_the_torch_version_that_agree_and_a_table_with_both():
    results = bench.run(configurations=(("30 min", 1800, 0.75),), repeats=1, device="cpu")
    entry = results["configurations"][0]
    assert set(entry["rows"]) == {"wotan", "torch-cpu"} and entry["padded_width"] == 64
    assert entry["rows"]["torch-cpu"]["nan_equal"] and entry["rows"]["torch-cpu"]["largest_relative_difference"] < 1e-9 and entry["rows"]["torch-cpu"]["first"] > 0
    text = bench.table(results)
    assert "wotan 1.10" in text and "torch on the CPU" in text and "NaN positions are equal" in text and "30 min, 0.75 d" in text and "width 64" in text


def test_a_row_that_would_take_too_long_is_skipped_and_said():
    results = bench.run(configurations=(("30 min", 1800, 0.75), ("10 min", 600, 0.75)), repeats=1, max_seconds=1e-9, device="cpu")
    assert "skipped" in results["configurations"][1]["rows"]["torch-cpu"]
    assert "Skipped: over" in bench.table(results)


def test_two_processes_sharing_a_device_both_detrend_and_their_rate_is_the_sum_of_theirs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))  # the spawned processes import the worker by its module name
    import bench_detrend

    row = bench_detrend.shared(2, step_seconds=1800, seconds=0.5, device="cpu")
    assert row["processes"] == 2 and len(row["calls"]) == 2 and all(calls > 0 for calls in row["calls"]) and len(row["first"]) == 2, row
    assert row["per_second"] > 0
    assert bench_detrend.shared_table([row]).splitlines()[2].startswith("| 2 | ")


def test_a_cold_start_reports_the_first_call_of_each_process_and_restores_the_cache_variable(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    monkeypatch.setenv("TRITON_CACHE_DIR", "kept")
    import bench_detrend

    row = bench_detrend.shared(2, step_seconds=1800, seconds=0.0, device="cpu", cache_dir=tmp_path)
    assert len(row["first"]) == 2 and all(first > 0 for first in row["first"])
    assert "2 processes starting together with an empty Triton cache" in bench_detrend.cold_table(row)
    import os

    assert os.environ["TRITON_CACHE_DIR"] == "kept"


def test_the_table_of_the_readme_is_the_table_of_the_stored_results_and_the_readme_gives_the_command():
    path = ROOT / "benchmarks" / "results-detrend.json"
    results = json.loads(path.read_text())
    readme = (ROOT / "README.md").read_text()
    assert bench.table(results) in readme, "the README's table differs from bench_detrend.table(benchmarks/results-detrend.json)"
    assert "benchmarks/bench_detrend.py" in readme
    versions = results["versions"]
    assert versions["torch"] in readme and versions["wotan"] in readme and (versions.get("device") or "") in readme, "the README names the versions and the device of the results"
