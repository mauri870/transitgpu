"""The ROC curves of the searches (validation/roc.py): the share of the injected light curves of signal to noise 8 and above that a search finds, against the share of the null light curves whose SDE it exceeds, at every false-alarm
rate of a grid, from the per-case records of the recovery. Function tests on made-up records, and a differential test against the thresholds and counts of validation/analyse.py."""
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
    for name in ("analyse", "roc"):
        spec = importlib.util.spec_from_file_location(name, VALIDATION / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        found[name] = module
    return found


def record(case_id, snr, sde, label=None, hybrid=None):
    out = {"id": case_id, "snr": snr, "planet": None if snr is None else {"period": 3.2}}
    for search in ("exact", "binned", "box"):
        out[search] = {"sde": sde, "period": 3.2, "t0": 1.0, "seconds": 0.1}
        if label is not None:
            out[search]["label"] = label
    out["hybrid"] = {"sde": sde, "seconds": 0.2, "peaks": [{"rank": k, "label": label_k} for k, label_k in enumerate(hybrid or ([label] if label else []))]}
    return out


def records(n_inj=100, n_null=200, seed=0):
    """Injected light curves of S/N 4 to 20 whose SDE grows with the S/N, and nulls with SDE in 2 to 6."""
    generator = np.random.default_rng(seed)
    out = []
    for i in range(n_inj):
        snr = (4.0, 6.0, 8.0, 10.0, 14.0, 20.0)[i % 6]
        out.append(record(f"white:main:inj:{i}", snr, float(snr + generator.normal(0, 1.5)), "right"))
    for i in range(n_null):
        out.append(record(f"white:main:null:{i}", None, float(generator.uniform(2, 6))))
    return out


def test_the_positives_are_the_injected_light_curves_of_snr_8_and_above_and_the_nulls_are_the_null_light_curves(modules):
    roc = modules["roc"]
    positives, nulls = roc.split(records(), "exact", snr_low=8.0)
    assert len(positives) == 100 // 6 * 4 + len([i for i in range(100 % 6) if (4.0, 6.0, 8.0, 10.0, 14.0, 20.0)[i] >= 8.0]) and len(nulls) == 200
    assert all(right for _, right in positives)


def test_the_tpr_at_a_false_alarm_rate_uses_the_one_minus_rate_quantile_of_the_nulls_as_threshold_and_counts_ties_as_found(modules):
    roc = modules["roc"]
    nulls = [float(x) for x in range(1, 101)]  # the 95th percentile of 1..100 is 95.05
    positives = [(95.05, True), (95.0, True), (99.0, True), (99.0, False), (50.0, True)]  # at the threshold a tie is found; a wrong period is never found
    curve = roc.curve(nulls, positives, [0.05])
    assert curve["threshold"][0] == pytest.approx(95.05) and curve["tpr"][0] == pytest.approx(2 / 5)  # 95.05 (a tie) and 99.0 right
    assert curve["low"][0] < 2 / 5 < curve["high"][0]


def test_the_curve_is_non_decreasing_in_the_false_alarm_rate_and_reaches_the_share_of_right_peaks_at_a_rate_of_one(modules):
    roc = modules["roc"]
    grid = roc.grid(1000)
    positives, nulls = roc.split(records(n_inj=300, n_null=1000), "exact", snr_low=8.0)
    curve = roc.curve([sde for sde, _ in nulls], positives, grid)
    assert all(b >= a - 1e-12 for a, b in zip(curve["tpr"], curve["tpr"][1:]))
    assert curve["tpr"][-1] == pytest.approx(sum(right and sde >= min(s for s, _ in nulls) for sde, right in positives) / len(positives))


def test_the_tpr_equals_the_completeness_of_analyse_at_its_threshold_pooled_over_the_levels_of_8_and_above(modules):
    roc, analyse = modules["roc"], modules["analyse"]
    data = records(n_inj=600, n_null=1000, seed=4)
    for search in ("exact", "box", "hybrid5"):
        positives, nulls = roc.split(data, search, snr_low=8.0)
        curve = roc.curve([sde for sde, _ in nulls], positives, [0.05, 0.01])
        injected = [r for r in data if r["id"].split(":")[2] == "inj" and r["snr"] >= 8.0]
        for index, rate in enumerate((0.05, 0.01)):
            cut = analyse.threshold([analyse.statistic(r, search) for r in data if r["id"].split(":")[2] == "null"], rate)[0]
            expected = sum(analyse.found(r, search, cut) for r in injected)
            assert curve["tpr"][index] * len(injected) == pytest.approx(expected), (search, rate)
            assert curve["threshold"][index] == pytest.approx(cut)


def test_the_hybrid_with_five_peaks_finds_a_planet_when_any_of_its_first_five_peaks_is_right_and_not_a_later_one(modules):
    roc = modules["roc"]
    data = [record("white:main:inj:0", 8.0, 10.0, "wrong", ["wrong", "wrong", "wrong", "wrong", "right", "right"]), record("white:main:inj:1", 8.0, 10.0, "wrong", ["wrong"] * 5 + ["right"]),
            record("white:main:null:0", None, 1.0), record("white:main:null:1", None, 2.0)]
    positives, _ = roc.split(data, "hybrid5", snr_low=8.0)
    assert positives == [(10.0, True), (10.0, False)]


def test_the_ordering_of_the_searches_and_the_crossings_of_their_curves_are_read_from_the_grid(modules):
    roc = modules["roc"]
    grid = [0.01, 0.05, 0.1, 0.5]
    tprs = {"a": [0.2, 0.5, 0.7, 0.9], "b": [0.1, 0.6, 0.8, 0.95], "c": [0.0, 0.1, 0.2, 0.3]}
    assert roc.ordering(tprs, 1) == ["b", "a", "c"] and roc.ordering(tprs, 0) == ["a", "b", "c"]
    crossings = roc.crossings(tprs, grid)
    assert crossings["a|b"] == [[0.01, 0.05]] and "a|c" not in crossings and "b|c" not in crossings


def test_a_part_without_nulls_or_without_positives_is_an_error_that_says_which(modules):
    roc = modules["roc"]
    with pytest.raises(ValueError, match="no null"):
        roc.part_curves([record("white:main:inj:0", 8.0, 9.0, "right")], ("exact",), 8.0, [0.05])
    with pytest.raises(ValueError, match="no injected"):
        roc.part_curves([record("white:main:null:0", None, 1.0)], ("exact",), 8.0, [0.05])


def test_the_binned_search_of_a_part_can_be_taken_from_the_rerun_of_the_banded_search_joined_by_id_and_an_id_that_the_evaluation_lacks_is_an_error(modules):
    roc = modules["roc"]
    data = [record("white:main:inj:0", 8.0, 5.0, "right"), record("white:main:null:0", None, 1.0)]
    rerun = [{"id": "white:main:inj:0", "binned": {"period": 3.2, "sde": 9.5, "label": "right"}}, {"id": "white:main:null:0", "binned": {"period": 3.2, "sde": 2.5}}]
    joined = roc.with_bands(data, rerun)
    assert joined[0]["binned"]["sde"] == 9.5 and joined[1]["binned"]["sde"] == 2.5 and joined[0]["exact"]["sde"] == 5.0
    with pytest.raises(ValueError, match="not in the evaluation"):
        roc.with_bands(data, rerun + [{"id": "white:main:null:99", "binned": {"sde": 1.0}}])


def test_curves_that_never_differ_by_more_than_the_tolerance_do_not_cross_and_a_real_change_of_order_does(modules):
    roc = modules["roc"]
    grid = [0.01, 0.05, 0.1, 0.5]
    close = {"a": [0.50, 0.50, 0.51, 0.52], "b": [0.501, 0.499, 0.512, 0.519]}  # always within 0.002
    assert roc.crossings(close, grid, tolerance=0.0025) == {} and roc.crossings(close, grid) != {}
    apart = {"a": [0.2, 0.5, 0.7, 0.9], "b": [0.1, 0.6, 0.8, 0.95]}
    assert roc.crossings(apart, grid, tolerance=0.0025) == {"a|b": [[0.01, 0.05]]}
