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


def test_a_small_run_on_the_cpu_has_a_row_for_wotan_and_for_the_torch_version_that_agree_and_a_table_with_both():
    results = bench.run(cadences=(("30 min", 1800),), repeats=1, device="cpu")
    rows = results["cadences"][0]["rows"]
    assert set(rows) == {"wotan", "torch-cpu"}
    assert rows["torch-cpu"]["nan_equal"] and rows["torch-cpu"]["largest_relative_difference"] < 1e-9
    text = bench.table(results)
    assert "wotan 1.10" in text and "torch on the CPU" in text and "NaN positions are equal" in text


def test_a_row_that_would_take_too_long_is_skipped_and_said():
    results = bench.run(cadences=(("30 min", 1800), ("10 min", 600)), repeats=1, max_seconds=1e-9, device="cpu")
    assert "skipped" in results["cadences"][1]["rows"]["torch-cpu"]
    assert "skipped" in bench.table(results)


def test_the_table_of_the_readme_is_the_table_of_the_stored_results_and_the_readme_gives_the_command():
    path = ROOT / "benchmarks" / "results-detrend.json"
    results = json.loads(path.read_text())
    readme = (ROOT / "README.md").read_text()
    assert bench.table(results) in readme, "the README's table differs from bench_detrend.table(benchmarks/results-detrend.json)"
    assert "benchmarks/bench_detrend.py" in readme
    versions = results["versions"]
    assert versions["torch"] in readme and versions["wotan"] in readme and (versions.get("device") or "") in readme, "the README names the versions and the device of the results"
