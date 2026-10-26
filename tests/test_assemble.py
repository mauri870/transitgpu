"""benchmarks/assemble.py joins the --json files of the parts of one timing session into the results.json of the paper: the rows of the table of the paper in one block, the shuffles and the long grids in blocks of their own, the cold calls
with their definition, and the load and free memory at the start and end of each part. Function tests on reports that these tests make up."""
import importlib.util
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
spec = importlib.util.spec_from_file_location("transitgpu_assemble", BENCHMARKS / "assemble.py")
assemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assemble)

HARDWARE, VERSIONS = {"cpu": "x", "logical_cpus": 32, "gpu": "y", "gpu_runtime": "z"}, {"python": "3.12", "torch": "2"}


def report(sizes, rows, calls, baseline=1000.0, first_row=None):
    entry = {}
    for points in sizes:
        entry[str(points)] = {"periods": 10, "rows": {name: {"seconds": [1.0 + 0.01 * k for k in range(calls)], "median": 1.0, "build_seconds": 2.0 if name == first_row else 0.3, "first_row_of_process": name == first_row,
                                                              "trial_periods": 10, "seconds_per_trial_period": 0.1, "period_found_days": 3.18} for name in rows}}
    return {"command": "uv run python bench.py", "hardware": HARDWARE, "versions": VERSIONS, "seed": 1, "baseline_days": baseline, "period_range_days": [0.5, 14.0], "planet": {}, "noise_ppm": 1000.0, "sizes": entry}


PARTS = {
    "main": report([20000, 100000], ["gpu", "tls-gpu"], 30),
    "slow20": report([20000], ["tls1"], 30),
    "slow": report([100000], ["tls1"], 7),
    "shuffles": report([20000], ["gpu-shuffles"], 30),
    "gtls72": report([72000], ["gpu"], 30, baseline=1500.0),
    "gtls-input-1500": report([72000], ["tls16-exact", "tls-gpu"], 7, baseline=1500.0),
    "gtls-input-3000": report([144000], ["tls16-exact", "tls-gpu"], 7, baseline=3000.0),
    "gtls-input-1500-margin": report([72000], ["tls16-exact", "tls-gpu"], 7, baseline=1500.0),
    "gtls-input-3000-margin": report([144000], ["tls16-exact", "tls-gpu"], 7, baseline=3000.0),
    "cold-gpu": report([20000], ["gpu"], 1, first_row="gpu"),
    "cold-cleared-gpu": report([20000], ["gpu"], 1, first_row="gpu"),
}
PARTS["cold-cleared-gpu"]["sizes"]["20000"]["rows"]["gpu"]["build_seconds"] = 21.5
LOG = [{"part": "main", "start": {"time": "t0", "load": 0.5, "memory_available_kib": 100}, "end": {"time": "t1", "load": 3.0, "memory_available_kib": 90}}]


def test_the_rows_of_the_table_are_in_one_block_with_the_number_of_calls_of_each_and_the_other_blocks_have_their_own_files():
    results = assemble.assemble(PARTS, LOG)
    assert set(results["sizes"]) == {"20000", "100000"} and set(results["sizes"]["20000"]["rows"]) == {"gpu", "tls-gpu", "tls1"} and set(results["sizes"]["100000"]["rows"]) == {"gpu", "tls-gpu", "tls1"}
    assert results["sizes"]["20000"]["rows"]["gpu"]["calls"] == 30 and results["sizes"]["20000"]["rows"]["tls1"]["calls"] == 30 and results["sizes"]["100000"]["rows"]["tls1"]["calls"] == 7
    assert results["block_shuffles"]["synthetic"]["sizes"]["20000"]["rows"]["gpu-shuffles"]["calls"] == 30 and [run["baseline_days"] for run in results["gtls_benchmark_sizes"]["runs"]] == [1500.0]


def test_the_cold_calls_are_recorded_with_their_definition_and_the_session_log_with_its_loads():
    results = assemble.assemble(PARTS, LOG)
    cold = results["cold_first_call"]
    assert "compiled caches" in cold["definition"] and cold["caches_kept"]["gpu"] == pytest.approx(2.0) and cold["caches_cleared"]["gpu"] == pytest.approx(21.5)
    assert results["session"][0]["start"]["load"] == 0.5 and results["session"][0]["end"]["memory_available_kib"] == 90
    assert assemble.assemble({k: v for k, v in PARTS.items() if not k.startswith("cold")}, [])["cold_first_call"]["caches_kept"] == {}, "a session without the cold parts has none"


def test_parts_of_different_inputs_are_refused_in_one_block():
    other = dict(PARTS, slow=report([100000], ["tls1"], 7))
    other["slow"]["seed"] = 2
    with pytest.raises(ValueError, match="seed"):
        assemble.assemble(other, LOG)


def test_the_session_script_runs_every_part_the_assembler_reads_pinned_and_refusing_a_busy_device():
    text = (BENCHMARKS / "session.sh").read_text()
    assert "taskset -c 0-15" in text and "--refuse-if-busy" in text and "--repeats 7" in text and "--repeats 30" not in text, "the calls of the published table are 7"
    for name in assemble.TABLE_PARTS + ("shuffles", "gtls72", "gtls144", "gtls-input-1500", "gtls-input-3000", "gtls-input-1500-margin", "gtls-input-3000-margin", "cold-$row", "cold-cleared-$row"):
        assert f"part {name} " in text or f'part "{name}"' in text, name
    assert "TRITON_CACHE_DIR" in text and "TORCH_EXTENSIONS_DIR" in text and "profile_split.py" in text and "screening_stats.py" in text


def test_the_input_of_the_gtls_paper_is_a_block_of_its_own_with_one_run_a_part_and_the_rows_of_the_table_do_not_take_it():
    results = assemble.assemble(PARTS, LOG)
    runs = results["gtls_input"]["runs"]
    assert [run["baseline_days"] for run in runs] == [1500.0, 3000.0, 1500.0, 3000.0] and [list(run["sizes"]) for run in runs] == [["72000"], ["144000"], ["72000"], ["144000"]]
    assert runs[0]["sizes"]["72000"]["rows"]["tls-gpu"]["calls"] == 7 and "72000" not in results["sizes"], "the rows of the table are the sizes of the benchmark"
    assert "gtls_input" not in assemble.assemble({k: v for k, v in PARTS.items() if not k.startswith("gtls-input")}, [])


def test_rows_of_the_table_that_a_stopped_part_did_not_reach_are_joined_from_the_parts_named_slow_rest():
    parts = dict(PARTS, **{"slow-rest-200000": report([200000], ["tls1"], 7), "slow-rest-100000": report([100000], ["tls1-exact"], 7)})
    results = assemble.assemble(parts, LOG)
    assert set(results["sizes"]["200000"]["rows"]) == {"tls1"} and set(results["sizes"]["100000"]["rows"]) == {"gpu", "tls-gpu", "tls1", "tls1-exact"}
    assert results["sizes"]["100000"]["rows"]["tls1-exact"]["calls"] == 7


def test_the_note_of_the_session_file_is_kept_in_the_results_and_a_session_without_one_has_none():
    assert assemble.assemble(PARTS, LOG, "pinned to cores 8-15 from 17:44")["session_note"] == "pinned to cores 8-15 from 17:44"
    assert "session_note" not in assemble.assemble(PARTS, LOG)


def test_the_session_script_checks_the_stop_file_before_the_steps_that_are_not_parts():
    text = (BENCHMARKS / "session.sh").read_text()
    assert 'stopped before profile_split' in text and 'stopped before screening_stats' in text


def test_the_versions_of_the_first_part_reach_results_json_with_triton_batman_and_wotan_when_the_parts_have_them():
    versions = {**VERSIONS, "triton": "3.7.0", "batman": "2.5.3", "wotan": "1.10"}
    parts = {name: {**part, "versions": versions} for name, part in PARTS.items()}
    results = assemble.assemble(parts, LOG)
    assert results["versions"] == versions
