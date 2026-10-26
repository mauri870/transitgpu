"""The arithmetic and the bookkeeping of validation/threshold.py: the comparison of a best SDE with its shuffles, the Wilson interval, the false-alarm quantiles of a threshold of 48 shuffles, the inversion, the seeds and the resume.

Function tests and not data files: they check counting rules on made-up numbers; the search itself is tested on a short light curve on the CPU.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parent.parent / "validation"
if not (VALIDATION / "threshold.py").exists():
    pytest.skip("validation/threshold.py is not in this tree", allow_module_level=True)
spec = importlib.util.spec_from_file_location("threshold", VALIDATION / "threshold.py")
threshold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold)


def test_the_best_peak_is_above_only_when_strictly_over_the_largest_shuffle_and_a_tie_is_not_above():
    assert threshold.compare(5.0, [4.0, 4.9]) == "above"
    assert threshold.compare(4.9, [4.0, 4.9]) == "tie"
    assert threshold.compare(4.8, [4.0, 4.9]) == "below"


def test_the_kth_largest_shuffle_is_the_threshold_of_rank_k():
    shuffles = [1.0, 5.0, 3.0, 4.0, 2.0]
    assert threshold.above_rank(4.5, shuffles, 1) is False  # the largest is 5.0
    assert threshold.above_rank(4.5, shuffles, 2) is True  # the second largest is 4.0
    assert threshold.above_rank(4.0, shuffles, 2) is False  # at the second largest is not above it


def test_the_wilson_interval_of_a_count_at_its_edges_and_the_known_value():
    low, high = threshold.wilson(0, 12)
    assert low == 0.0 and abs(high - 0.2425) < 1e-3  # 0 of 12: the upper end 0.2425
    low, high = threshold.wilson(8, 400)
    assert abs(low - 0.01017) < 2e-4 and abs(high - 0.03885) < 2e-4
    assert threshold.wilson(5, 5)[1] == 1.0


def test_the_false_alarm_quantiles_of_a_threshold_of_48_shuffles_have_the_mean_one_in_49():
    assert abs(threshold.false_alarm_quantile(0.5, 48) - (1 - 0.5 ** (1 / 48))) < 1e-15
    assert abs(threshold.false_alarm_quantile(0.5, 48) - 0.01433) < 1e-4
    mean = np.mean([threshold.false_alarm_quantile(q, 48) for q in (np.arange(100000) + 0.5) / 100000])
    assert abs(mean - 1 / 49) < 1e-4


def test_the_inverted_flux_is_reflected_about_its_median_and_keeps_the_median():
    flux = np.array([1.0, 1.0, 1.0, 0.9, 1.2])
    inverted = threshold.invert(flux)
    assert np.median(inverted) == np.median(flux) and np.allclose(inverted, [1.0, 1.0, 1.0, 1.1, 0.8])


def test_the_seeds_of_the_classes_do_not_meet_and_stay_in_the_reserved_range():
    seeds = {threshold.seed_of(c, i) for c in range(3) for i in range(5000)}
    assert len(seeds) == 15000 and min(seeds) >= 9_000_000 and max(seeds) < 10_000_000


def test_a_finished_case_is_skipped_before_anything_is_generated(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "null:white:3"}) + "\n" + json.dumps({"id": "null:red:3"}) + "\n")
    assert threshold.finished(rows) == {"null:white:3", "null:red:3"}
    calls = []
    threshold.run_cases(["null:white:3", "null:white:4"], rows, lambda case_id: calls.append(case_id) or {"id": case_id})
    assert calls == ["null:white:4"]
    assert threshold.finished(rows) == {"null:white:3", "null:red:3", "null:white:4"}
    assert threshold.finished(tmp_path / "absent.jsonl") == set()


def test_a_short_light_curve_gives_a_row_with_the_best_sde_and_the_shuffle_sdes_on_the_cpu():
    row = threshold.null_case("white", 7, sectors=1, shuffles=4, baseline=3.0, period_range=(0.5, 1.5), device="cpu")
    assert row["id"] == "null:white:7" and len(row["shuffle_sde"]) == 4 and len(row["inverted_sde"]) == 4 and np.isfinite(row["sde"])
    again = threshold.null_case("white", 7, sectors=1, shuffles=4, baseline=3.0, period_range=(0.5, 1.5), device="cpu")
    assert {k: v for k, v in again.items() if k != "seconds"} == {k: v for k, v in row.items() if k != "seconds"}, "a curve is the same wherever it is built"


def test_the_cells_of_the_planets_run_through_the_thirty_cells_with_periods_slowest():
    cells = [threshold.cell_of(i) for i in range(30)]
    assert len(set(cells)) == 30 and cells[0] == (0.8, 4) and cells[5] == (0.8, 20) and cells[6] == (1.6, 4) and cells[29] == (12.8, 20)
    assert threshold.cell_of(30) == cells[0] and threshold.cell_of(61) == cells[1]


def test_an_injected_curve_shares_its_noise_with_the_null_curve_and_the_peak_of_a_deep_planet_is_labelled_right():
    kwargs = dict(sectors=1, shuffles=3, baseline=6.0, period_range=(0.5, 3.0), device="cpu")
    null = threshold.null_case("white", 10, **kwargs)
    row = threshold.injection_case("white", 10, **kwargs)  # cell 10 is (1.6 d, S/N 14)
    assert row["id"] == "inj:white:10" and row["seed"] == null["seed"] and row["period_injected"] == 1.6 and row["target_snr"] == 14
    assert row["label"] == "right" and abs(row["period"] / 1.6 - 1) < 0.01, f"a planet of S/N 14 was labelled {row['label']} at {row['period']} d"
    assert row["sde"] > null["sde"], "the planet did not raise the best SDE over the noise alone"


def test_a_flattened_flux_is_computed_once_and_read_back_from_the_cache_bit_for_bit(tmp_path):
    calls = []

    def flatten(time, flux):
        calls.append(1)
        return flux / np.median(flux)

    time, flux = np.arange(5.0), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    path = tmp_path / "red-9100000.npy"
    first = threshold.flattened(time, flux, path, flatten)
    second = threshold.flattened(time, flux * 100, path, flatten)  # the cache answers, whatever the arguments
    assert len(calls) == 1 and np.array_equal(first, second) and path.exists()
    assert threshold.flattened(time, flux, None, flatten).tolist() == first.tolist() and len(calls) == 2  # no cache, computed


def test_the_cache_names_of_a_class_and_a_planet_differ_so_that_a_null_flux_is_never_read_as_an_injected_one(tmp_path):
    assert threshold.cache_path(tmp_path, "red", 7, injected=False) != threshold.cache_path(tmp_path, "red", 7, injected=True)
    assert threshold.cache_path(tmp_path, "red", 7, injected=False) != threshold.cache_path(tmp_path, "variable", 7, injected=False)
    assert threshold.cache_path(None, "red", 7, injected=False) is None


def test_a_red_planet_is_scaled_to_the_noise_of_the_quiet_curve_at_its_duration_and_a_white_one_to_the_white_sigma():
    """R11: the correlated noise does not average over a transit, so the sigma of a red planet is above the white 700 ppm of its class; the white class keeps its white sigma."""
    kwargs = dict(sectors=1, shuffles=2, baseline=8.0, period_range=(0.5, 3.0), device="cpu")
    white = threshold.injection_case("white", 10, **kwargs)
    red = threshold.injection_case("red", 10, **kwargs)
    assert white["sigma"] == 1000e-6, white["sigma"]
    assert red["sigma"] > 700e-6, f"the effective sigma of the red noise at the transit duration is {red['sigma'] * 1e6:.0f} ppm, not above the 700 ppm of white noise alone"
    assert red["radius_ratio"] > threshold.injection_case("red", 10, **kwargs)["radius_ratio"] * 0.999  # deterministic


def test_copies_shuffled_before_flattening_are_the_flattening_of_the_shuffled_raw_flux_and_differ_from_shuffled_flat_flux():
    """R13: a filter that is not a permutation of its input (a running mean here) gives a different copy when the blocks are shuffled before it than after it; the copy keeps the points of each sector inside it."""
    time = np.arange(40.0)
    flux = np.sin(time / 3.0) + 0.01 * time

    def flatten(t, f):
        kernel = np.ones(5) / 5
        return f - np.convolve(np.pad(f, 2, mode="edge"), kernel, mode="valid")

    edges = [0, 20, 40]
    cadence = 1 / 5  # blocks of 5 points
    copies = threshold.shuffled_then_flattened(time, flux, edges, cadence, 3, 11, flatten)
    assert copies.shape == (3, 40)
    injection = threshold.load("injection")
    raw = injection.block_shuffles(flux, edges, cadence, 3, seed=11)
    assert np.array_equal(copies[1], flatten(time, raw[1]))
    after = np.array(injection.block_shuffles(flatten(time, flux), edges, cadence, 3, seed=11))
    assert not np.allclose(copies, after), "the order of the shuffle and the filter made no difference, so the check could not tell the two apart"


def test_the_red_check_prepares_one_file_of_copies_a_curve_and_the_device_case_reads_it_and_skips_nothing_it_lacks(tmp_path):
    threshold.prepare_reflat(("red", 4, tmp_path, 1, 6.0, 3))
    path = threshold.reflat_path(tmp_path, "red", 4)
    assert path.exists() and np.load(path).shape == (3, 4320)  # 3 copies of a 6-day sector at 2 minutes
    assert threshold.prepare_reflat(("red", 4, tmp_path, 1, 6.0, 3)) == 0, "a finished curve is not flattened again"
    row = threshold.reflat_case("red", 4, tmp_path, baseline=6.0, period_range=(0.5, 2.0), device="cpu")
    assert row["id"] == "reflat:red:4" and len(row["shuffle_sde"]) == 3 and np.isfinite(row["sde"])
