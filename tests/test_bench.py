"""The benchmark that ships with the package (benchmarks/bench.py) times a row only when a run of it is estimated to fit the limit, reports the estimate otherwise, and reads the period each row found. Function tests: the checks
are on the output structure and the rule of the estimate, not on the seconds."""

import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("transitgpu_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_row_is_timed_with_its_answer_and_a_row_over_the_limit_is_reported_as_the_estimate_of_the_size_before(bench, tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "bench.json"
    report = bench.main(["--sizes", "3000", "6000", "--baseline", "60", "--rows", "cpu1", "--repeats", "1", "--max-run-seconds", "0.0000001", "--json", str(path)])
    first, second = report["sizes"]["3000"]["rows"]["cpu1"], report["sizes"]["6000"]["rows"]["cpu1"]
    assert first["median"] > 0 and abs(first["period_found_days"] / 3.18 - 1) < 0.01, "the row found the injected period"
    assert first["trial_periods"] == report["sizes"]["3000"]["periods"] and first["seconds_per_trial_period"] == pytest.approx(first["median"] / first["trial_periods"])
    assert "estimated_seconds" in second and second["estimated_seconds"] == pytest.approx(2 * first["median"]) and "median" not in second
    printed = bench.table(json.loads(path.read_text()))
    assert "not run: linear scaling from 3,000 points" in printed and "3.18" in printed


def test_the_light_curve_is_the_same_for_the_same_size_and_has_the_injected_transit(bench):
    first, second = bench.light_curve(2000, 100.0), bench.light_curve(2000, 100.0)
    assert (first[0] == second[0]).all() and (first[1] == second[1]).all()
    time_axis, flux = first
    assert time_axis[0] == 0.0 and time_axis[-1] == 100.0 and len(flux) == 2000
    phase = (time_axis - 1.2 + 3.18 / 2) % 3.18 - 3.18 / 2
    assert flux[abs(phase) < 2.5 / 48].mean() < flux[abs(phase) > 0.2].mean() - 0.002, "the box is 3000 ppm deep"


def test_the_reports_of_two_runs_of_the_same_inputs_are_joined_row_by_row_and_reports_of_other_inputs_are_refused(bench):
    one = {"hardware": "h", "versions": "v", "seed": 1, "baseline_days": 10, "period_range_days": [0.5, 14], "planet": {}, "noise_ppm": 1, "sizes": {"100": {"periods": 5, "rows": {"gpu": {"median": 1}}}}}
    two = {**one, "sizes": {"100": {"periods": 5, "rows": {"tls1": {"median": 2}}}, "200": {"periods": 5, "rows": {"tls1": {"median": 4}}}}}
    joined = bench.merged([one, two])
    assert sorted(joined["sizes"]["100"]["rows"]) == ["gpu", "tls1"] and list(joined["sizes"]) == ["100", "200"]
    with pytest.raises(ValueError, match="baseline_days"):
        bench.merged([one, {**two, "baseline_days": 20}])


def test_the_recorded_command_is_the_uv_run_of_the_script_with_the_json_file_and_no_directory(bench, monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmarks/bench.py", "--rows", "gpu", "--json", "/work/a/b/out.json"])
    monkeypatch.setattr("sys.executable", "/opt/venv/bin/python3")
    assert bench.command_line() == "uv run python benchmarks/bench.py --rows gpu --json out.json"


def test_a_light_curve_given_as_a_file_is_timed_in_place_of_the_synthetic_ones_and_its_period_is_read(bench, tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    time_axis, flux = bench.light_curve(3000, 60.0)
    path = tmp_path / "curve.npz"
    np.savez(path, time=time_axis, flux=flux)
    report = bench.main(["--npz", str(path), "--rows", "cpu1", "--repeats", "1"])
    assert list(report["sizes"]) == ["3000"] and report["input"]["file"] == "curve.npz" and report["baseline_days"] == pytest.approx(60.0)
    assert abs(report["sizes"]["3000"]["rows"]["cpu1"]["period_found_days"] / 3.18 - 1) < 0.01


def test_a_light_curve_given_as_a_file_can_carry_the_box_of_the_synthetic_input_and_every_row_finds_it(bench, tmp_path):
    """The point of the injection: the rows are then held to one known answer. The edge case is a file whose first time is not zero, as a real stack's is (the transit is placed from the first time)."""
    pytest.importorskip("torch")
    import numpy as np

    time_axis, flux = bench.light_curve(3000, 60.0)
    time_axis = time_axis + 1325.0
    flux = flux * 0 + 1.0 + np.random.default_rng(3).normal(0, 1e-3, flux.size)
    path = tmp_path / "curve.npz"
    np.savez(path, time=time_axis, flux=flux)
    report = bench.main(["--npz", str(path), "--inject", "--rows", "cpu1", "--repeats", "1"])
    assert report["input"]["injected_box"]["period"] == 3.18
    assert abs(report["sizes"]["3000"]["rows"]["cpu1"]["period_found_days"] / 3.18 - 1) < 0.01
    injected = bench.inject(time_axis, flux)
    assert (injected < flux).sum() > 0 and (injected <= flux).all(), "the box only lowers the flux"
    assert abs((flux - injected).max() - 3000e-6) < 1e-12


def test_the_period_range_and_the_block_shuffles_of_a_light_curve_are_options_of_the_benchmark(bench):
    """--period-max sets the grid (half the baseline gives the grid of the GTLS paper's benchmark) and the row gpu-shuffles searches K block-shuffled copies in one call: the copies keep the flux values, change their order by blocks of one day from fixed seeds, and the row answers with the largest SDE."""
    pytest.importorskip("torch")
    import numpy as np

    time_axis, flux = bench.light_curve(2400, 40.0)
    copies = bench.block_shuffles(time_axis, flux, 3, 40.0)
    assert len(copies) == 3 and all(len(c) == len(flux) and np.allclose(np.sort(c), np.sort(flux)) for c in copies) and not np.array_equal(copies[0], copies[1])
    assert all(np.array_equal(a, b) for a, b in zip(copies, bench.block_shuffles(time_axis, flux, 3, 40.0))), "the seeds are fixed"
    report = bench.main(["--sizes", "2400", "--baseline", "40", "--period-max", "20", "--rows", "gpu", "gpu-shuffles", "--shuffles", "3", "--repeats", "1"])
    entry = report["sizes"]["2400"]
    assert report["period_range_days"] == [0.5, 20.0] and entry["rows"]["gpu-shuffles"]["trial_periods"] == 3 * entry["periods"]
    assert entry["rows"]["gpu-shuffles"]["period_found_days"] > 0, "the answer is the largest SDE of the shuffles"


def test_the_rows_of_transitgpu_tls_are_rows_of_the_benchmark_with_labels_and_builders(bench):
    for name in ("tls-gpu", "tls-gpu-binned", "tls-gpu-pl"):
        assert name in bench.ROWS and name in bench.LABELS and name in bench.BUILDERS


def test_the_readme_table_has_a_row_for_each_search_that_ran_and_an_empty_cell_where_it_did_not(bench):
    report = {"sizes": {"100": {"periods": 5, "rows": {"gpu": {"median": 0.12345, "seconds": [0.1, 0.12345, 0.2]}, "tls1": {"median": 12.0, "seconds": [11.0, 12.0, 13.5]}}},
                        "200": {"periods": 5, "rows": {"gpu": {"median": 2.0, "seconds": [2.0]}, "tls1": {"estimated_seconds": 24.0}}}}}
    lines = bench.readme_table(report).splitlines()
    assert lines[0] == "| search | 100 points | 200 points |"
    assert lines[2] == "| transitgpu on the GPU | 0.123 (0.112–0.162) | 2 (2–2) |"
    assert lines[3].endswith("| 12 (11.5–12.8) |  |"), "a row that was only estimated has an empty cell, not a number"
    assert len(lines) == 4, "rows that did not run at any size do not appear"


def test_the_performance_table_of_the_readme_is_the_table_built_from_the_results_file_and_so_is_the_table_of_the_set(bench):
    """The two tables of the README's Performance section are built from benchmarks/results.json and hold no other number: every cell is the median of its runs with the quartiles, three significant figures."""
    root = BENCH.parents[1]
    results = json.loads((root / "benchmarks" / "results.json").read_text())
    assert results["session_note"] and "tls" not in results and "tls_earlier_session" not in results, "one block of rows, one session"
    assert all(len(row["seconds"]) == 7 for size in results["sizes"].values() for row in size["rows"].values()), "seven runs in every row"
    readme = (root / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(readme) if line.startswith("| search | device |"))
    end = next(i for i in range(start, len(readme)) if not readme[i].startswith("|"))
    assert "\n".join(readme[start:end]) == bench.performance_table(results)
    assert sum(line.startswith("| search | device |") for line in readme) == 2, "the README has the table of times and the table of the set"
    start = next(i for i, line in enumerate(readme) if line.startswith("| search | device | seconds for the set"))
    end = next(i for i in range(start, len(readme)) if not readme[i].startswith("|"))
    assert "\n".join(readme[start:end]) == bench.throughput_table(results)


def test_the_bumps_table_of_the_readme_is_the_table_built_from_its_own_results_file_of_one_session(bench):
    """The third table of the README's Performance section is built from benchmarks/results-bumps.json, a file of its own: its rows are from one session and no other session's rows are in it."""
    root = BENCH.parents[1]
    results = json.loads((root / "benchmarks" / "results-bumps.json").read_text())
    assert set(row for size in results["sizes"].values() for row in size["rows"]) == {"gpu", "gpu-bumps", "gpu-two"}, "the three rows of the table and no other"
    assert all(len(row["seconds"]) == 7 for size in results["sizes"].values() for row in size["rows"].values()), "seven runs in every row"
    readme = (root / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(readme) if line.startswith("| search of one light curve |"))
    end = next(i for i in range(start, len(readme)) if not readme[i].startswith("|"))
    assert "\n".join(readme[start:end]) == bench.bumps_table(results)
    assert results["command"] in "\n".join(readme).replace("uv run --group rocm python", "uv run python"), "the README gives the command that made the file"


def test_the_weights_table_of_the_readme_is_the_table_built_from_its_own_results_file_of_one_session(bench):
    """The fourth table of the README's Performance section is built from benchmarks/results-weights.json, a file of its own, of one session."""
    root = BENCH.parents[1]
    results = json.loads((root / "benchmarks" / "results-weights.json").read_text())
    assert set(row for size in results["sizes"].values() for row in size["rows"]) == {"gpu", "gpu-weights", "gpu-weights-separate"}, "the three rows of the table and no other"
    assert all(len(row["seconds"]) == 7 for size in results["sizes"].values() for row in size["rows"].values()), "seven runs in every row"
    readme = (root / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(readme) if line.startswith("| search of one light curve with weight vectors |"))
    end = next(i for i in range(start, len(readme)) if not readme[i].startswith("|"))
    assert "\n".join(readme[start:end]) == bench.weights_table(results)
    assert results["command"] in "\n".join(readme).replace("uv run --group rocm python", "uv run python"), "the README gives the command that made the file"


def test_the_figures_of_the_table_have_three_significant_figures_and_the_same_form_in_every_cell(bench):
    """Edge cases of the format: a round number keeps its zeros (15.0, not 15), a number over 100 has no decimals, and one under 1 has three figures."""
    assert [bench.three_figures(value) for value in (15.0, 93.96, 291.78, 0.27711, 2.15, 5.74, 1234.0)] == ["15.0", "94.0", "292", "0.277", "2.15", "5.74", "1230"]


def test_the_readme_of_the_benchmarks_names_every_row_of_bench_py_and_gives_the_commands_as_uv_runs(bench):
    text = (BENCH.parent / "README.md").read_text()
    for row in [*bench.ROWS, *bench.EXTRA_ROWS]:
        assert f"`{row}`" in text, row
    assert "uv run --group rocm python benchmarks/bench.py --rows" in text and "python -m pip" not in text and "pip install" not in text
    assert not (BENCH.parent / "PLAN.md").exists(), "the plan of a run is not part of the package"
    assert "Written before the run" not in text and "hours" not in text, "no schedule of a run"


def test_the_readme_of_the_benchmarks_has_a_table_of_the_rows_and_a_table_of_the_timed_region_for_every_row(bench):
    text = (BENCH.parent / "README.md").read_text()
    assert "| row | what it runs |" in text and "| row | inside the timer | outside the timer |" in text
    region = text.split("## Timed region")[1].split("## Conditions")[0]
    for name in bench.ROWS:
        assert f"`{name}`" in region, f"{name} has a line in the table of the timed region"
    assert len(text.split()) < 520, "the README of the benchmarks stays short"
    assert "not measured" not in text and "was read from" not in text


def test_the_hybrid_is_a_row_of_the_benchmark_with_a_label_and_a_builder_and_finds_the_injected_period_on_a_device(bench, tmp_path):
    assert "hybrid" in bench.ROWS and "5 peaks, window 1%, exact backend" in bench.LABELS["hybrid"] and "hybrid" in bench.BUILDERS
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no device: the hybrid searches on a GPU only")
    report = bench.main(["--sizes", "20000", "--baseline", "200", "--rows", "hybrid", "--repeats", "1", "--json", str(tmp_path / "hybrid.json")])
    row = report["sizes"]["20000"]["rows"]["hybrid"]
    assert row["median"] > 0 and abs(row["period_found_days"] / 3.18 - 1) < 0.01, "the best peak is the injected period"


def test_the_quartiles_of_seven_runs_are_the_linear_ones_and_the_cell_of_the_table_shows_the_median_with_them(bench):
    assert bench.quartiles([7, 1, 2, 3, 4, 5, 6]) == (2.5, 5.5)
    assert bench.quartiles([2.0]) == (2.0, 2.0), "a single run has no spread"
    report = {"sizes": {"100": {"periods": 5, "rows": {"gpu": {"median": 4.0, "seconds": [1, 2, 3, 4, 5, 6, 7]}}}}}
    assert bench.readme_table(report).splitlines()[2] == "| transitgpu on the GPU | 4 (2.5–5.5) |"


def test_the_set_of_light_curves_has_one_time_axis_one_injected_box_and_other_noise_in_each_and_its_first_is_the_light_curve_of_the_benchmark(bench):
    import numpy as np

    time_axis, fluxes = bench.light_curves(2000, 100.0, 4)
    single_time, single = bench.light_curve(2000, 100.0)
    assert len(fluxes) == 4 and np.array_equal(fluxes[0], single) and np.array_equal(time_axis, single_time)
    assert not np.array_equal(fluxes[0], fluxes[1]), "other noise in each"
    inside = np.abs((time_axis - bench.PLANET["t0"] + bench.PLANET["period"] / 2) % bench.PLANET["period"] - bench.PLANET["period"] / 2) < bench.PLANET["duration_hr"] / 48
    for flux in fluxes:
        assert abs(flux[inside].mean() - flux[~inside].mean() + bench.PLANET["depth_ppm"] * 1e-6) < 5 * bench.NOISE_PPM * 1e-6 / np.sqrt(inside.sum()) * 2, "the same box in each"


def test_the_throughput_rows_are_extra_rows_with_labels_and_builders_and_run_only_when_named(bench):
    for name in ("many", "tls16-many"):
        assert name in bench.EXTRA_ROWS and name not in bench.ROWS and name in bench.LABELS and name in bench.BUILDERS
    assert "64 light curves" in bench.LABELS["many"] and "64 light curves" in bench.LABELS["tls16-many"]


def test_the_search_of_a_set_finds_the_injected_period_in_every_light_curve_of_it(bench, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no device: the search of a set runs on a GPU only")
    report = bench.main(["--sizes", "20000", "--baseline", "200", "--rows", "many", "--repeats", "1", "--json", str(tmp_path / "many.json")])
    row = report["sizes"]["20000"]["rows"]["many"]
    assert row["median"] > 0 and abs(row["period_found_days"] / 3.18 - 1) < 0.01 and row["found_in_every_light_curve"] is True


def test_the_json_file_holds_the_rows_that_finished_when_a_later_row_stops_the_run(bench, tmp_path, monkeypatch):
    """The final session was stopped after 47 minutes and the file, written only at the end, did not exist: the rows are written as each one ends."""
    pytest.importorskip("torch")
    path = tmp_path / "partial.json"

    def stops(_t, _f, _p):
        raise KeyboardInterrupt

    monkeypatch.setitem(bench.BUILDERS, "cpu16", stops)
    with pytest.raises(KeyboardInterrupt):
        bench.main(["--sizes", "3000", "--baseline", "60", "--rows", "cpu1", "cpu16", "--repeats", "1", "--json", str(path)])
    assert "cpu1" in json.loads(path.read_text())["sizes"]["3000"]["rows"]


def test_a_process_with_the_compute_device_open_is_another_gpu_job_and_the_own_process_is_not(bench, tmp_path):
    for pid, target in ((100, "/dev/kfd"), (101, "/dev/null"), (102, "/dev/dri/renderD128")):
        directory = tmp_path / str(pid) / "fd"
        directory.mkdir(parents=True)
        (directory / "3").symlink_to(target)
    (tmp_path / "self_pid_marker").write_text("")
    assert bench.other_gpu_jobs(proc=str(tmp_path), own=100) == [], "a render node is held open by every desktop program and is not a compute job"
    assert bench.other_gpu_jobs(proc=str(tmp_path), own=102) == [100]


def test_the_busy_sampler_reports_the_mean_and_the_largest_of_the_percentages_read_while_it_runs(bench, tmp_path):
    import time

    path = tmp_path / "gpu_busy_percent"
    path.write_text("40\n")
    with bench.GpuBusy(str(path), interval=0.01) as busy:
        time.sleep(0.1)
        path.write_text("80\n")
        time.sleep(0.1)
    assert busy.samples >= 5 and 40 <= busy.mean <= 80 and busy.largest == 80, (busy.samples, busy.mean, busy.largest)


def test_a_row_refuses_to_start_when_another_job_has_the_device_and_the_option_is_given(bench, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setattr(bench, "other_gpu_jobs", lambda **kwargs: [4242])
    with pytest.raises(SystemExit, match="4242"):
        bench.main(["--sizes", "3000", "--baseline", "60", "--rows", "cpu1", "--repeats", "1", "--refuse-if-busy", "--json", str(tmp_path / "bench.json")])


def test_a_child_of_this_process_that_inherited_the_device_is_not_another_job(bench, tmp_path):
    for pid, parent in ((200, 1), (201, 200), (202, 201), (300, 1)):
        directory = tmp_path / str(pid) / "fd"
        directory.mkdir(parents=True)
        (directory / "3").symlink_to("/dev/kfd")
        (tmp_path / str(pid) / "stat").write_text(f"{pid} (python) S {parent} 0 0 0")
    assert bench.other_gpu_jobs(proc=str(tmp_path), own=200) == [300], "the children and grandchildren of the process (the workers of a row) are left out"


def test_a_row_records_the_cold_call_and_the_spread_table_gives_the_median_the_quartiles_and_the_first_call_apart(bench, tmp_path):
    """TG4: the spread of the timed calls (at least 30 for a row of the paper's table), the first timed call and the build of the row (its warm-up call, which compiles when the row is the first of its process) are reported apart."""
    pytest.importorskip("torch")
    path = tmp_path / "spread.json"
    report = bench.main(["--sizes", "600", "--baseline", "12", "--rows", "cpu1", "cpu16", "--repeats", "4", "--json", str(path)])
    rows = report["sizes"]["600"]["rows"]
    assert rows["cpu1"]["first_row_of_process"] is True and rows["cpu16"]["first_row_of_process"] is False
    assert rows["cpu1"]["build_seconds"] > 0 and rows["cpu16"]["build_seconds"] > 0 and len(rows["cpu1"]["seconds"]) == 4
    printed = bench.spread_table(json.loads(path.read_text()))
    assert "| calls | median | quartiles | min | max | first timed call | build of the row (cold when first of its process) |" in printed
    assert f"| 4 | {rows['cpu1']['median']:.3g} |" in printed
    assert bench.spread_table({**json.loads(path.read_text()), "sizes": {"600": {"periods": 1, "rows": {"cpu1": {"estimated_seconds": 5.0, "seconds_per_point": 1.0, "measured_at_points": 3000}}}}}).count("not run") == 1


def test_the_gtls_input_is_one_light_curve_at_30_minute_cadence_with_the_grid_to_half_the_span_and_the_gtls_rows(bench, tmp_path):
    """The input of the GTLS paper's timing (Hu et al. 2026): 30-minute cadence over the baseline, TLS's grid from 0 to half the span, the rows tls16-exact and tls-gpu; here 20 days, 960 points, so the TLS row runs in seconds."""
    pytest.importorskip("transitleastsquares")
    path = tmp_path / "gtls.json"
    report = bench.main(["--gtls-input", "20", "--rows", "tls16-exact", "--repeats", "1", "--max-run-seconds", "600", "--t0-fit-margin", "0.125", "--noise-ppm", "500", "--json", str(path)])
    assert list(report["sizes"]) == ["960"] and report["baseline_days"] == 20 and report["period_range_days"] == [0.0, 10.0]
    assert report["input"]["gtls"] is True and report["input"]["cadence_minutes"] == 30 and report["input"]["points"] == 960 and report["input"]["t0_fit_margin"] == 0.125 and report["noise_ppm"] == 500
    row = report["sizes"]["960"]["rows"]["tls16-exact"]
    assert row["median"] > 0 and len(row["seconds"]) == 1 and abs(row["period_found_days"] / 3.18 - 1) < 0.01, "the row found the injected period"
    assert "T0 fit margin 0.125" in row["durations"]
    assert json.loads(path.read_text())["input"]["source"].startswith("Hu et al. 2026")


def test_the_gtls_rows_default_to_the_set_of_the_paper_and_the_points_follow_the_cadence(bench, monkeypatch):
    assert bench.GTLS_ROWS == ("tls16-exact", "tls-gpu") and bench.GTLS_CADENCE_MINUTES == 30
    assert int(round(1500 * 1440 / bench.GTLS_CADENCE_MINUTES)) == 72000 and int(round(3000 * 1440 / bench.GTLS_CADENCE_MINUTES)) == 144000
    with pytest.raises(SystemExit, match="cannot be given with --npz"):
        bench.main(["--gtls-input", "20", "--npz", "none.npz"])


def test_the_readme_of_the_benchmarks_describes_the_gtls_input_and_its_commands(bench):
    text = (BENCH.parent / "README.md").read_text()
    assert "--gtls-input" in text and "30-minute cadence" in text and "T0_fit_margin" in text and "half the span" in text
    assert "--gtls-input" in text


def test_the_versions_include_triton_batman_and_wotan_from_the_installed_distributions_and_leave_out_what_is_not_installed(bench, monkeypatch):
    """batman's module says 2.5.1 where its distribution batman-package is 2.5.3, so the distribution is read; triton's module version is its distribution's (triton-rocm on ROCm, triton on CUDA). A package that is not installed is absent, whatever the environment (the cpu build of torch brings no triton)."""
    import importlib.metadata as metadata
    import sys

    def installed(*names):
        for name in names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return None

    found = bench.versions()
    for key, names in (("triton", ("triton-rocm", "triton")), ("batman", ("batman-package",)), ("wotan", ("wotan",))):
        version = installed(*names)
        assert (key in found) == (version is not None), f"{key}: the versions have it only when a distribution of {names} is installed"
        if version is not None:
            assert found[key] == version, key

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    monkeypatch.setitem(sys.modules, "triton", None)
    absent = bench.versions()
    assert not {"triton", "batman", "wotan"} & set(absent), "absent when not installed, not None"
