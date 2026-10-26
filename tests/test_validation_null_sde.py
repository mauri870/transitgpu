"""The histograms of the noise-only SDE of the searches (validation/null_sde.py): the binned counts of each search's SDE over a part's null light curves with its 5% threshold, and, for the box search, the distribution of
the threshold of 48 block shuffles (the largest SDE of the first 48 shuffles of a light curve) from the calibration rows. Function tests on made-up records."""
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
    for name in ("analyse", "roc", "null_sde"):
        spec = importlib.util.spec_from_file_location(name, VALIDATION / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        found[name] = module
    return found


def null_record(index, sde, part="white"):
    out = {"id": f"{part}:main:null:{index}", "snr": None, "planet": None}
    for search in ("exact", "binned", "box"):
        out[search] = {"sde": sde, "period": 3.0, "t0": 1.0, "seconds": 0.1}
    out["hybrid"] = {"sde": sde, "seconds": 0.2, "peaks": []}
    return out


def test_the_counts_of_a_histogram_sum_to_the_nulls_and_the_edges_span_their_range(modules):
    module = modules["null_sde"]
    sdes = list(np.random.default_rng(1).normal(5.0, 1.0, 1000))
    entry = module.histogram(sdes, bins=40)
    assert sum(entry["counts"]) == 1000 and len(entry["counts"]) == 40 and len(entry["edges"]) == 41
    assert entry["edges"][0] == pytest.approx(min(sdes)) and entry["edges"][-1] == pytest.approx(max(sdes))
    assert all(b > a for a, b in zip(entry["edges"], entry["edges"][1:]))


def test_the_threshold_of_a_search_is_the_95th_percentile_of_its_nulls_as_analyse_takes_it_and_the_hybrid_with_five_peaks_has_the_hybrids_sde(modules):
    module, analyse = modules["null_sde"], modules["analyse"]
    records = [null_record(i, 2.0 + 4.0 * i / 999) for i in range(1000)] + [{"id": "white:main:inj:0", "snr": 8.0, "planet": {}, **{s: {"sde": 99.0, "label": "right"} for s in ("exact", "binned", "box")}, "hybrid": {"sde": 99.0, "peaks": []}}]
    part = module.part_histograms(records, ("exact", "box", "hybrid5"), bins=20)
    assert part["nulls"] == 1000, "injected records are not nulls"
    for search in ("exact", "box", "hybrid5"):
        assert part["searches"][search]["threshold_5"] == pytest.approx(analyse.threshold([r["exact"]["sde"] for r in records[:1000]], 0.05)[0])


def test_the_shuffle_threshold_of_a_curve_is_the_largest_of_its_first_48_shuffles_and_a_class_has_only_its_own_curves(modules):
    module = modules["null_sde"]
    shuffles = list(range(1, 100))
    rows = [{"id": "null:white:0", "class": "white", "shuffle_sde": [float(x) for x in shuffles]}, {"id": "null:red:0", "class": "red", "shuffle_sde": [5.0] * 48},
            {"id": "inj:white:0", "class": "white", "shuffle_sde": [50.0] * 48}, {"id": "real:111:1", "tic": 111, "sector": 1, "shuffle_sde": [float(x) for x in range(1, 200)]}]
    assert module.shuffle_thresholds(rows, "white") == [48.0]
    assert module.shuffle_thresholds(rows, "red") == [5.0]
    assert module.shuffle_thresholds(rows, "real") == [48.0]
    assert module.shuffle_thresholds(rows, "variable") == []


def test_a_row_with_fewer_than_48_shuffles_is_an_error_that_says_so(modules):
    with pytest.raises(ValueError, match="fewer than 48"):
        modules["null_sde"].shuffle_thresholds([{"id": "null:white:0", "class": "white", "shuffle_sde": [1.0] * 10}], "white")
