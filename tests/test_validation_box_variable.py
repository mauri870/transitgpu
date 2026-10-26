"""The box search against two other box searches on the variable class (validation/box_variable.py): the same light curves as the evaluation, three statistics on each, and the counts, intervals and
paired differences against exact mode. The searches run on the CPU on a tiny light curve; the analysis is tested on fabricated records."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[1] / "validation"


@pytest.fixture(scope="module")
def modules():
    sys.path.insert(0, str(VALIDATION))
    found = {}
    for name in ("run", "generator", "analyse", "box_variable"):
        spec = importlib.util.spec_from_file_location(name, VALIDATION / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        found[name] = module
    return found


def tiny_light_curve():
    """Four days at a 10-minute cadence with a box transit of 1.3 d and 5,000 ppm in 300 ppm of white noise; the planet's first transit is at 0.4 d and lasts 2 h."""
    time = np.arange(0.0, 4.0, 10.0 / 1440.0)
    flux = 1.0 + 300e-6 * np.random.default_rng(3).standard_normal(time.size)
    phase = ((time - 0.4) / 1.3 + 0.5) % 1.0 - 0.5
    flux[np.abs(phase * 1.3) < 1.0 / 24.0] -= 5000e-6
    return time, flux, np.full(time.size, 300e-6)


def test_the_cases_are_the_injected_light_curves_of_signal_to_noise_8_and_the_first_nulls_of_the_variable_class(modules):
    box_variable, run = modules["box_variable"], modules["run"]
    ids = box_variable.case_ids()
    assert len(ids) == 400 and ids[0] == "variable:main:inj:2" and ids[1] == "variable:main:inj:8" and ids[200] == "variable:main:null:0" and ids[-1] == "variable:main:null:199"
    assert set(ids) <= set(run.case_ids("main-variable")), "every case is one of the evaluation's, so exact mode's row exists for it"
    assert all(run.planet_of(int(i.split(":")[3]), 0, np.arange(0.0, 27.4, 2 / 1440), lambda duration: 1e-3)[1] == 8.0 for i in ids[:3]), "the injected cases are the cells of S/N 8"


def test_the_three_statistics_find_a_box_transit_at_its_period_on_the_cpu_and_the_detrended_spectrum_falls_back_to_the_plain_z_score_when_the_grid_is_short(modules):
    box_variable = modules["box_variable"]
    time, flux, errors = tiny_light_curve()
    found = box_variable.statistics(time, flux, errors, 0.5, 2.0, device="cpu")
    assert set(found) == {"box", "box_detrended", "astropy"}
    for name, search in found.items():
        assert abs(search["period"] / 1.3 - 1) < 0.01, (name, search)
        assert abs(((search["t0"] - 0.4) + 0.65) % 1.3 - 0.65) < 1.0 / 24.0, (name, search)
        assert search["sde"] > 3, (name, search)
    power = np.array([1.0, 2.0, 1.0, 9.0, 1.0])
    assert box_variable.detrended_sde(power, np.linspace(1, 2, 5)) == pytest.approx(float((9.0 - power.mean()) / power.std())), "a grid shorter than twice the kernel of TLS has no trend to remove"


def record(case_id, snr, box, detrended, astropy):
    kind = case_id.split(":")[2]
    planet = None if kind == "null" else {"period": 3.2, "impact": 0.1, "radius_ratio": 0.05, "epoch_fraction": 0.3}
    out = {"id": case_id, "snr": snr, "planet": planet}
    for name, (sde, label) in (("box", box), ("box_detrended", detrended), ("astropy", astropy)):
        out[name] = {"sde": sde, "period": 3.2, "t0": 1.0, "seconds": 0.01}
        if label is not None:
            out[name]["label"] = label
    return out


def exact_record(case_id, sde, label):
    return {"id": case_id, "exact": {"sde": sde, "label": label, "period": 3.2, "t0": 1.0, "seconds": 0.1}}


def test_the_analysis_counts_each_search_at_the_threshold_of_its_own_nulls_and_pairs_it_with_exact_mode_on_the_same_cases(modules):
    box_variable = modules["box_variable"]
    records, exact = [], []
    # 40 injected: the box search finds 4 (right and above its cut of about 6), the detrended 10, astropy 3 (one of them at a harmonic, so not found), exact 20
    for i in range(40):
        case_id = f"variable:main:inj:{6 * i + 2}"
        records.append(record(case_id, 8.0, (9.0 if i < 4 else 3.0, "right"), (9.0 if i < 10 else 3.0, "right"), (9.0 if i < 3 else 3.0, "harmonic" if i == 0 else "right")))
        exact.append(exact_record(case_id, 12.0 if i < 20 else 3.0, "right"))
    for i in range(200):
        case_id = f"variable:main:null:{i}"
        sde = 2.0 + 4.0 * i / 199
        records.append(record(case_id, None, (sde, None), (sde, None), (sde, None)))
        exact.append(exact_record(case_id, sde + 1.0, None))
    out = box_variable.analysis(records, exact)
    assert out["injected"] == 40 and out["null"] == 200
    at = out["searches"]
    assert [at[name]["0.05"]["found"] for name in ("box", "box_detrended", "astropy", "exact")] == [4, 10, 2, 20]
    assert at["astropy"]["right_peak"] == 39 and at["astropy"]["harmonic"] == 1 and at["box"]["right_peak"] == 40
    assert at["box"]["0.05"]["wilson"][0] < 4 / 40 < at["box"]["0.05"]["wilson"][1]
    assert 5.7 < at["exact"]["0.05"]["threshold"] < 7.1 and at["exact"]["0.05"]["threshold"] == pytest.approx(at["box"]["0.05"]["threshold"] + 1.0)
    pairs = out["paired"]["0.05"]
    assert pairs["box_minus_exact"]["estimate"] == pytest.approx((4 - 20) / 40) and pairs["box_minus_exact"]["only_first"] == 0 and pairs["box_minus_exact"]["only_second"] == 16
    assert pairs["astropy_minus_box"]["estimate"] == pytest.approx((2 - 4) / 40) and pairs["box_detrended_minus_box"]["estimate"] == pytest.approx((10 - 4) / 40)
    assert pairs["box_minus_exact"]["low"] < pairs["box_minus_exact"]["estimate"] < pairs["box_minus_exact"]["high"]
    with pytest.raises(ValueError, match="not in the evaluation"):
        box_variable.analysis(records + [record("variable:main:inj:9999", 8.0, (1.0, "wrong"), (1.0, "wrong"), (1.0, "wrong"))], exact)
