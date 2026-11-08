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


def test_the_skip_of_a_slow_row_scales_by_the_points_times_the_padded_width(monkeypatch):
    def one_second(function, repeats):
        return function(), 1.0, [1.0] * repeats

    monkeypatch.setattr(bench, "timed", one_second)
    results = bench.run(configurations=(("30 min", 1800, 0.75), ("10 min", 600, 0.75)), repeats=1, max_seconds=4.0, device="cpu")
    first, second = results["configurations"]
    assert second["points"] / first["points"] > 2.9 and second["padded_width"] == 2 * first["padded_width"]
    assert "skipped" in second["rows"]["torch-cpu"], "1 s times about 3 times the points and twice the width is 6 s, over the 4 s limit"


def test_a_row_left_out_by_rows_is_skipped_and_said():
    results = bench.run(configurations=(("30 min", 1800, 0.75),), repeats=1, device="cpu", rows=[])
    assert results["configurations"][0]["rows"]["torch-cpu"] == {"skipped": "not run: left out by --rows"}
    assert "Skipped: not run: left out by --rows." in bench.table(results)


def test_configurations_run_one_by_one_and_merged_give_the_table_of_one_run(tmp_path):
    for index in (0, 1):
        assert bench.main(["--device", "cpu", "--repeats", "1", "--configuration", str(index), "--json", str(tmp_path / f"part{index}.json")]) == 0
    assert bench.main(["--merge", str(tmp_path / "all.json"), str(tmp_path / "part0.json"), str(tmp_path / "part1.json")]) == 0
    merged = json.loads((tmp_path / "all.json").read_text())
    assert [entry["step_seconds"] for entry in merged["configurations"]] == [1800, 600]
    assert bench.table(merged).splitlines()[0].count("0.75 d") == 2


def test_files_of_different_versions_are_not_merged(tmp_path):
    part = {"versions": {"torch": "a"}, "repeats": 7, "kernel_width_limit": 2048, "configurations": []}
    (tmp_path / "a.json").write_text(json.dumps(part))
    (tmp_path / "b.json").write_text(json.dumps({**part, "versions": {"torch": "b"}}))
    with pytest.raises(ValueError, match="differ in versions"):
        bench.merge([tmp_path / "a.json", tmp_path / "b.json"])


def test_the_table_of_the_readme_is_the_table_of_the_stored_results_and_the_readme_gives_the_command():
    path = ROOT / "benchmarks" / "results-detrend.json"
    results = json.loads(path.read_text())
    readme = (ROOT / "README.md").read_text()
    assert bench.table(results) in readme, "the README's table differs from bench_detrend.table(benchmarks/results-detrend.json)"
    assert "benchmarks/bench_detrend.py" in readme
    versions = results["versions"]
    assert versions["torch"] in readme and versions["wotan"] in readme and (versions.get("device") or "") in readme, "the README names the versions and the device of the results"
