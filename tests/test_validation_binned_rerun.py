"""The rerun of the binned mode on the cases of the evaluation (validation/rerun_binned.py, analyse_binned.py): the same case ids as run.py, one file a part, and a comparison joined by id to the rows of run.py with the
thresholds and the paired interval of analyse.py. Function tests on fabricated records: the device is not needed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

VALIDATION = Path(__file__).resolve().parents[1] / "validation"


@pytest.fixture(scope="module")
def modules():
    sys.path.insert(0, str(VALIDATION))
    found = {}
    for name in ("run", "analyse", "rerun_binned", "analyse_binned"):
        spec = importlib.util.spec_from_file_location(name, VALIDATION / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        found[name] = module
    return found


def record(case_id, snr, period, exact, binned, new):
    """A fabricated evaluation record and a rerun record: (sde, label) of each search."""
    kind = case_id.split(":")[2]
    planet = None if kind == "null" else {"period": period, "impact": 0.1, "radius_ratio": 0.05, "epoch_fraction": 0.3}
    evaluation = {"id": case_id, "snr": snr, "planet": planet, "exact": {"sde": exact[0], "label": exact[1], "seconds": 0.1}, "binned": {"sde": binned[0], "label": binned[1], "seconds": 0.05},
                  "box": {"sde": 1.0, "label": "wrong", "seconds": 0.01}, "hybrid": {"sde": 1.0, "peaks": [], "seconds": 0.1}}
    rerun = {"id": case_id, "snr": snr, "planet": planet, "backend": "gpu-binned", "kwargs": {"bins": 8}, "binned": {"sde": new[0], "label": new[1], "seconds": 0.05}}
    return evaluation, rerun


def test_the_rerun_runs_the_case_ids_of_run_py_and_names_its_file_by_the_part(modules):
    run, rerun = modules["run"], modules["rerun_binned"]
    assert rerun.output_of("main-white", "out").name == "main-white.binned-bands.jsonl"
    assert run.case_ids("main-white")[:2] == ["white:main:inj:0", "white:main:inj:1"] and len(run.case_ids("main-white")) == 1200 + 1000


def test_the_comparison_joins_by_id_and_gives_the_paired_difference_of_the_rerun_against_exact_beside_the_binned_of_the_evaluation(modules, tmp_path):
    analyse_binned = modules["analyse_binned"]
    evaluation, rerun = [], []
    # 40 injected at S/N 10: exact finds 30, the evaluation's binned 20, the rerun's binned 28; 200 nulls with SDEs that put the 5 percent threshold near 6
    for i in range(40):
        e, r = record(f"white:main:inj:{i}", 10.0, 3.2, (9.0 if i < 30 else 3.0, "right"), (9.0 if i < 20 else 3.0, "right"), (9.0 if i < 28 else 3.0, "right"))
        evaluation.append(e)
        rerun.append(r)
    for i in range(200):
        sde = 2.0 + 4.0 * i / 199
        e, r = record(f"white:main:null:{i}", None, None, (sde, None), (sde, None), (sde, None))
        evaluation.append(e)
        rerun.append(r)
    records = analyse_binned.joined(rerun, evaluation)
    assert len(records) == 240 and records[0][analyse_binned.NEW]["sde"] == 9.0
    part = analyse_binned.part_of(records)
    at = part["paired"]["0.05"]
    assert at["binned_bands_minus_exact"]["estimate"] == pytest.approx((28 - 30) / 40) and at["binned_minus_exact"]["estimate"] == pytest.approx((20 - 30) / 40)
    assert at["binned_bands_minus_binned"]["estimate"] == pytest.approx((28 - 20) / 40)
    assert at["binned_bands_minus_exact"]["low"] < at["binned_bands_minus_exact"]["estimate"] < at["binned_bands_minus_exact"]["high"]
    assert part["searches"]["binned_bands"]["0.05"]["by_snr"]["10.0"]["found"] == 28
    with pytest.raises(ValueError, match="not in the evaluation"):
        analyse_binned.joined([{"id": "white:main:inj:999", "binned": {}}], evaluation)
