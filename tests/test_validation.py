"""The generator, the case numbering, the resume rule and the statistics of validation/ (README.md there): light curves and planets are functions of their seeds, the target signal to noise is met, a period is classified as right or as a harmonic as the README says, a finished case is skipped before anything is generated, and the interval of a difference of paired shares has its nominal coverage.

Test functions and not archives: the checks are on numbers and on the files of the evaluation, not on a search of a light curve.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(DIRECTORY))


def load(name):
    specification = importlib.util.spec_from_file_location(f"validation_{name}", DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


generator, analyse, run = load("generator"), load("analyse"), load("run")


def test_a_light_curve_is_a_function_of_its_class_and_its_seed():
    time, _ = generator.time_axis(1)
    for kind in generator.NOISE_CLASSES:
        first, second, other = (generator.noise(kind, seed, time)[1] for seed in (7, 7, 8))
        assert np.array_equal(first, second) and not np.array_equal(first, other)


def test_the_time_axis_of_sectors_has_a_gap_and_edges_at_the_starts_of_the_sectors():
    time, edges = generator.time_axis(2)
    assert edges[0] == 0 and edges[-1] == time.size and len(edges) == 3
    assert time[edges[1]] - time[edges[1] - 1] == pytest.approx(generator.GAP_DAYS + generator.CADENCE_DAYS)


def test_the_noise_classes_have_the_stated_scatter():
    time, _ = generator.time_axis(1)
    assert np.std(generator.noise("white", 1, time)[1]) == pytest.approx(1000e-6, rel=0.03)
    red = generator.noise("red", 1, time)[1]
    assert np.std(red) == pytest.approx(np.hypot(700e-6, 500e-6), rel=0.2)
    assert np.corrcoef(red[:-1] - 1, red[1:] - 1)[0, 1] > 0.2, "the red noise is correlated from cadence to cadence"


@pytest.mark.parametrize("target", [4.0, 10.0, 20.0])
def test_the_radius_ratio_is_solved_for_the_target_signal_to_noise(target):
    time, _ = generator.time_axis(1)
    body = generator.with_signal_to_noise(generator.planet(3.2, 0.4, 0.3), time, 1e-3, target)
    assert generator.signal_to_noise(body, time, 1e-3) == pytest.approx(target, rel=1e-3)


def test_a_period_is_right_when_within_a_percent_and_on_a_transit_a_harmonic_when_at_a_simple_ratio_else_wrong():
    body = generator.planet(3.0, 0.2, 0.5)
    time, _ = generator.time_axis(1)
    _, first = generator.transit_flux(body, time)
    assert generator.classify(3.0, first, body, first) == "right"
    assert generator.classify(3.0 * 1.009, first + 3.0, body, first) == "right"
    assert generator.classify(3.0 * 1.02, first, body, first) == "wrong"
    assert generator.classify(3.0, first + 1.0, body, first) == "wrong", "the right period at another epoch"
    assert generator.classify(6.0, first + 1.0, body, first) == "harmonic"
    assert generator.classify(1.5, first, body, first) == "harmonic"
    assert generator.classify(2.0, first, body, first) == "harmonic"
    assert generator.classify(float("nan"), first, body, first) == "wrong"


def test_a_block_shuffle_keeps_the_values_of_each_sector_and_copy_zero_is_the_flux():
    time, edges = generator.time_axis(2)
    flux = np.arange(time.size, dtype=float)
    assert np.array_equal(generator.block_shuffle(flux, edges, 0), flux)
    copy = generator.block_shuffle(flux, edges, 3)
    assert not np.array_equal(copy, flux)
    for start, stop in zip(edges[:-1], edges[1:]):
        assert np.array_equal(np.sort(copy[start:stop]), np.sort(flux[start:stop])), "no value leaves its sector"
    assert np.array_equal(copy, generator.block_shuffle(flux, edges, 3))


def test_the_seeds_of_this_evaluation_stay_out_of_the_range_of_the_other_one():
    seeds = [run.seed_of(kind, baseline, injected, index) for kind in generator.NOISE_CLASSES + ("real",) for baseline in ("main", "four") for injected, count in ((True, 1500), (False, 1500)) for index in range(0, count, 7)]
    assert max(seeds) < 9_000_000 and len(set(seeds)) == len(seeds)


def test_the_cases_of_the_parts_are_numbered_as_the_readme_says():
    assert len(run.case_ids("main-white")) == 30 * 40 + 1000
    assert len(run.case_ids("four-red")) == 30 * 8 + 300
    assert len(run.case_ids("real")) == 12 * 30 * 3 + 12 * 101
    assert len(set(run.case_ids("real"))) == len(run.case_ids("real"))


def test_a_case_in_the_output_file_is_skipped_before_it_is_generated(tmp_path):
    ids = run.case_ids("main-white")
    output = tmp_path / "out.jsonl"
    output.write_text("".join(json.dumps({"id": case_id}) + "\n" for case_id in ids[:5]))
    assert run.pending("main-white", output) == ids[5:]
    assert run.pending("main-white", tmp_path / "none.jsonl") == ids


def test_the_real_stars_are_three_of_each_type_with_a_hash_and_an_effective_temperature():
    stars = json.loads((DIRECTORY / "stars.json").read_text())
    assert len(stars) == 12 and {s["type"] for s in stars} == set("FGKM")
    assert all(sum(s["type"] == kind for s in stars) == 3 for kind in "FGKM")
    assert all(len(s["sha256"]) == 64 and s["url"].startswith("https://") and s["teff"] > 2500 and s["logg"] >= 4.0 for s in stars)


def test_the_interval_of_paired_shares_contains_the_estimate_is_symmetric_and_narrows_with_more_light_curves():
    estimate, low, high = analyse.tango(20, 5, 100)
    assert low < estimate < high and estimate == pytest.approx(0.15)
    swapped = analyse.tango(5, 20, 100)
    assert swapped[1] == pytest.approx(-high, abs=1e-6) and swapped[2] == pytest.approx(-low, abs=1e-6)
    wide, narrow = analyse.tango(20, 5, 100), analyse.tango(200, 50, 1000)
    assert narrow[2] - narrow[1] < wide[2] - wide[1]
    assert analyse.tango(0, 0, 100)[1] < 0 < analyse.tango(0, 0, 100)[2], "no discordant light curve still leaves an interval around zero"


def test_the_interval_of_paired_shares_covers_the_true_difference_about_95_percent_of_the_time():
    generator_ = np.random.default_rng(1)
    cells = np.array([0.25, 0.10, 0.04, 0.61])  # both, only the first, only the second, neither
    truth = cells[1] - cells[2]
    covered = 0
    trials = 1500
    for _ in range(trials):
        both, first, second, neither = generator_.multinomial(150, cells)
        _, low, high = analyse.tango(first, second, 150)
        covered += low <= truth <= high
    assert 0.93 <= covered / trials <= 0.97, covered / trials


def test_a_threshold_is_the_quantile_of_the_null_with_a_bootstrap_interval_around_it():
    null = np.random.default_rng(2).standard_normal(1000)
    cut, low, high = analyse.threshold(null, 0.05)
    assert cut == pytest.approx(np.quantile(null, 0.95)) and low < cut < high
    assert analyse.threshold(null, 0.01)[0] > cut


def test_a_last_line_cut_by_a_killed_run_is_not_a_finished_case(tmp_path):
    ids = run.case_ids("main-white")
    output = tmp_path / "out.jsonl"
    output.write_text(json.dumps({"id": ids[0]}) + "\n" + json.dumps({"id": ids[1]})[:10])
    assert run.finished(output) == {ids[0]}
    assert run.pending("main-white", output)[0] == ids[1]


def test_a_cut_last_line_is_removed_before_the_next_line_is_written(tmp_path):
    output = tmp_path / "out.jsonl"
    output.write_text('{"id": "a"}\n{"id": "b"')
    run.repair(output)
    assert output.read_text() == '{"id": "a"}\n'
    run.repair(output)
    assert output.read_text() == '{"id": "a"}\n'


def _record(kind, index, snr, sde, label, period=3.2, peaks=None):
    return {"id": f"white:main:{kind}:{index}", "snr": snr, "planet": {"period": period}, "exact": {"sde": sde, "label": label, "seconds": 0.1}, "binned": {"sde": sde, "label": label, "seconds": 0.1},
            "box": {"sde": sde, "label": label, "seconds": 0.01}, "hybrid": {"sde": sde, "seconds": 0.3, "peaks": peaks or [{"rank": 0, "label": label}]}}


def test_the_analysis_reads_the_records_into_completeness_with_a_threshold_from_the_nulls_and_a_paired_difference(tmp_path):
    nulls = [{"id": f"white:main:null:{i}", "exact": {"sde": 5.0 + i * 0.01, "seconds": 0.1}, "binned": {"sde": 5.0 + i * 0.01, "seconds": 0.1}, "box": {"sde": 5.0 + i * 0.01, "seconds": 0.1}, "hybrid": {"sde": 5.0 + i * 0.01, "seconds": 0.3}} for i in range(100)]
    injected = [{**_record("inj", i, 10.0, 12.0, "right" if i % 4 else "wrong"), **({"hybrid_seconds": {"1": 0.02, "5": 0.08, "10": 0.13, "20": 0.25}} if i % 8 == 0 else {})} for i in range(40)] + [_record("inj", 40 + i, 4.0, 5.2, "right") for i in range(10)]
    (tmp_path / "main-white.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nulls + injected))
    analyse.main(tmp_path)
    result = json.loads((tmp_path / "results.json").read_text())["main-white"]
    five = result["searches"]["exact"]["0.05"]
    assert result["injected"] == 50 and result["null"] == 100
    assert five["threshold"] == pytest.approx(np.quantile([5.0 + i * 0.01 for i in range(100)], 0.95))
    assert five["by_snr"]["10.0"]["found"] == 30 and five["by_snr"]["4.0"]["found"] == 0, "an SDE of 5.2 is under the threshold of 5.94"
    assert result["paired"]["0.05"]["hybrid5_minus_exact"]["estimate"] == 0.0 and result["paired"]["0.05"]["hybrid5_minus_exact"]["n"] == 40
    assert set(result["searches"]) == {"exact", "binned", "box", "hybrid1", "hybrid5", "hybrid10", "hybrid20"}
    assert result["hybrid_seconds"]["5"] == {"peaks": 5, "median": 0.08, "quartiles": [0.08, 0.08], "n": 5}


def test_the_effective_sigma_is_the_scatter_per_point_for_white_noise_and_larger_for_red_noise():
    time, _ = generator.time_axis(1)
    white = generator.noise("white", 4, time)[1]
    red = generator.noise("red", 4, time)[1]
    assert generator.effective_sigma(white, 2.6 / 24) == pytest.approx(1000e-6, rel=0.08)
    assert generator.effective_sigma(red, 2.6 / 24) > 2000e-6, "the correlated noise does not average over the transit"
    assert generator.effective_sigma(red, 6 / 24) > generator.effective_sigma(red, 2.6 / 24), "a longer transit sees more of it"


def test_a_planet_in_correlated_noise_is_solved_for_the_target_against_the_effective_sigma_and_recovered_in_a_case():
    case = run.light_curve("red:main:inj:29", Path("/tmp"))
    time, flux, errors, edges, (body, first), details = case
    assert details["snr"] == 20.0 and details["sigma_eff"] > 2 * 700e-6
    assert generator.signal_to_noise(body, time, details["sigma_eff"]) == pytest.approx(20.0, rel=0.02)
