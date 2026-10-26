"""Every table and every file of macros of the paper that a script writes from a shipped file is the file that the script writes now: a table that was not written again after its source changed fails here.

Covered, on the shipped sources and without a device (validation_table.tex and binned_table.tex are rendered from the json that the device run wrote beside them, by the function table_of of their scripts): split_table.tex (scripts/split_table.py from scripts/profile_split.json), threshold_table.tex and injection_table.tex (scripts/threshold_table.py from threshold.json),
period_table.tex (scripts/period_table.py from recovery.json and recovery_real.json), recall_table_main-red.tex, recall_table_main-variable.tex, recall_table_main-white.tex and recall_table_real.tex (scripts/completeness.py from recovery.json and recovery_real.json), and numbers.tex, split_numbers.tex,
peak_numbers.tex, recovery_numbers.tex, roc_numbers.tex, threshold_numbers.tex and adversarial_numbers.tex (scripts/fill_numbers.py from adversarial_table.json, benchmarks/results.json, scripts/profile_split.json, peak.json, recovery.json, recovery_real.json, box_variable.json and
threshold.json). Left out because their scripts search on a device: bins_table.tex (scripts/bins_share.py) and
adversarial_table.json (scripts/adversarial.py); and refs_check.json (scripts/check_refs.py), which asks the web. Function tests: the structure under test is the file that each script writes."""
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
if not (PACKAGE / "paper").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
PAPER, SCRIPTS = PACKAGE / "paper", PACKAGE / "paper" / "scripts"


def run(script, *arguments):
    done = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)], capture_output=True, text=True, cwd=PACKAGE)
    assert done.returncode == 0, f"{script} {' '.join(map(str, arguments))}\n{done.stderr[-2000:]}"


def same(written, name):
    committed = (PAPER / name).read_text()
    assert written.read_text() == committed, f"paper/{name} is not what its script writes from the shipped source: run the script again and commit the file"


def test_the_split_table_is_what_split_table_py_writes_from_the_profile_of_the_split(tmp_path):
    run("split_table.py", SCRIPTS / "profile_split.json", "--out", tmp_path / "split_table.tex")
    same(tmp_path / "split_table.tex", "split_table.tex")


def test_the_threshold_and_injection_tables_are_what_threshold_table_py_writes_from_the_analysis(tmp_path):
    run("threshold_table.py", PAPER / "threshold.json", "--out", tmp_path / "threshold_table.tex")
    same(tmp_path / "threshold_table.tex", "threshold_table.tex")
    same(tmp_path / "injection_table.tex", "injection_table.tex")


def test_the_recall_tables_are_what_completeness_py_writes_from_the_recovery_files(tmp_path):
    """The script is run with --tables-only: the figure needs matplotlib, which the cpu group of the package (the command of the CI) does not install; the figure is the default of the script."""
    run("completeness.py", PAPER / "recovery.json", "--real", PAPER / "recovery_real.json", "--out-directory", tmp_path, "--tables-only")
    assert not (tmp_path / "completeness.pdf").exists(), "--tables-only draws no figure"
    for name in ("recall_table_main-red.tex", "recall_table_main-variable.tex", "recall_table_main-white.tex", "recall_table_real.tex"):
        same(tmp_path / name, name)


def test_the_period_table_is_what_period_table_py_writes_from_the_recovery_files(tmp_path):
    run("period_table.py", PAPER / "recovery.json", "--real", PAPER / "recovery_real.json", "--out", tmp_path / "period_table.tex")
    same(tmp_path / "period_table.tex", "period_table.tex")


def test_the_files_of_macros_are_what_fill_numbers_py_writes_from_the_shipped_results(tmp_path):
    names = {"--out": "numbers.tex", "--profile-out": "split_numbers.tex", "--peak-out": "peak_numbers.tex", "--recovery-out": "recovery_numbers.tex", "--threshold-out": "threshold_numbers.tex", "--adversarial-out": "adversarial_numbers.tex", "--roc-out": "roc_numbers.tex"}
    run("fill_numbers.py", *[part for option, name in names.items() for part in (option, tmp_path / name)])
    for name in names.values():
        same(tmp_path / name, name)


def load(script):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"paper_{script}", SCRIPTS / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_validation_table_is_what_validation_py_renders_from_the_json_of_its_device_run():
    """validation_table.json is what the run on a device wrote; the table is rendered from it by table_of, which the script also uses after its run."""
    import json

    sys.path.insert(0, str(PACKAGE / "tests" / "golden_upstream"))
    import golden_cases

    worst = json.loads((PAPER / "validation_table.json").read_text())
    rendered = load("validation").table_of(worst, len(golden_cases.CASES))
    assert rendered == (PAPER / "validation_table.tex").read_text(), "paper/validation_table.tex is not what validation.py renders from validation_table.json"


def test_the_binned_table_is_what_binned_accuracy_py_renders_from_the_json_of_its_device_run():
    import json

    rows = json.loads((PAPER / "binned_table.json").read_text())
    assert load("binned_accuracy").table_of(rows) == (PAPER / "binned_table.tex").read_text(), "paper/binned_table.tex is not what binned_accuracy.py renders from binned_table.json"
