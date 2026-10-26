"""The counting of validation/threshold_analysis.py on made-up rows: the share above the threshold, the ranks, the inverted null, the planets found above each threshold and the planet's effect on its own null."""
import importlib.util
from pathlib import Path

import pytest

VALIDATION = Path(__file__).resolve().parent.parent / "validation"
if not (VALIDATION / "threshold_analysis.py").exists():
    pytest.skip("validation/threshold_analysis.py is not in this tree", allow_module_level=True)
spec = importlib.util.spec_from_file_location("threshold_analysis", VALIDATION / "threshold_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def null(index, sde, shuffles, inverted):
    return {"id": f"null:white:{index}", "class": "white", "index": index, "sde": sde, "shuffle_sde": shuffles, "inverted_sde": inverted}


def test_the_share_above_counts_strict_exceedances_and_ties_apart_with_the_rank_curve():
    rows = [null(0, 9.0, [5.0, 6.0], [4.0, 4.5]), null(1, 6.0, [5.0, 6.0], [4.0, 7.0]), null(2, 5.5, [5.0, 6.0], [4.0, 4.5]), null(3, 7.0, [5.0, 6.0], [8.0, 4.5])]
    summary = analysis.null_summary(rows, ranks=(1, 2))
    assert summary["curves"] == 4 and summary["above"] == 2 and summary["ties"] == 1 and summary["share"] == 0.5
    assert summary["rank_above"] == {1: 2, 2: 4}  # over the second largest shuffle (5.0): 9.0, 6.0, 5.5 and 7.0
    inverted = analysis.inverted_summary(rows)
    assert inverted["above"] == 2 and inverted["curves"] == 4  # 9.0 over 4.5 and 5.5 over 4.5; 6.0 under 7.0 and 7.0 under 8.0


def test_a_planet_is_found_only_when_right_and_above_the_threshold_and_a_harmonic_is_not_found():
    nulls = [null(i, 5.0, [6.0, 6.5], [6.0, 7.0]) for i in range(3)]
    injections = [
        {"id": "inj:white:0", "class": "white", "index": 0, "target_snr": 10, "period_injected": 1.6, "sde": 9.0, "label": "right", "shuffle_sde": [6.0, 6.5], "inverted_sde": [6.0, 7.0]},
        {"id": "inj:white:1", "class": "white", "index": 1, "target_snr": 10, "period_injected": 1.6, "sde": 6.8, "label": "right", "shuffle_sde": [6.0, 6.5], "inverted_sde": [6.0, 7.0]},
        {"id": "inj:white:2", "class": "white", "index": 2, "target_snr": 10, "period_injected": 1.6, "sde": 9.0, "label": "harmonic", "shuffle_sde": [6.0, 8.0], "inverted_sde": [6.0, 7.0]},
    ]
    summary = analysis.injection_summary(injections, nulls)["white"]["10"]
    assert summary["cases"] == 3 and summary["right"] == 2 and summary["harmonic"] == 1
    assert summary["found_shuffle"] == 2 and summary["found_inverted"] == 1  # 9.0 and 6.8 over 6.5; only 9.0 over 7.0
    assert summary["threshold_raised_by_planet"] == 1  # the shuffles of case 2 reach 8.0 against the noise alone's 6.5


def test_the_kolmogorov_statistic_of_uniform_ranks_is_zero_for_the_midpoints_and_large_for_a_pile_up():
    assert analysis.kolmogorov([(i + 0.5) / 10 for i in range(10)])[0] == pytest.approx(0.05)
    statistic, p = analysis.kolmogorov([0.95] * 12)
    assert statistic == pytest.approx(0.95) and p < 1e-6


def test_exchangeable_made_up_curves_give_one_in_49_at_the_first_rank_and_k_over_49_at_rank_k():
    """A control of the counting itself: the best SDE and its 48 shuffles drawn from one distribution are exchangeable, so the share above the largest shuffle is 1/49 and above the k-th largest k/49, inside the Wilson interval."""
    import numpy as np

    generator = np.random.default_rng(1)
    rows = []
    for index in range(6000):
        draws = generator.gumbel(5.0, 1.0, 49)
        rows.append(null(index, float(draws[0]), draws[1:].tolist(), draws[1:].tolist()))
    summary = analysis.null_summary(rows)
    assert summary["low"] <= 1 / 49 <= summary["high"], f"{summary['count']} of {summary['of']}: [{summary['low']:.4f}, {summary['high']:.4f}] does not contain 1/49"
    for k, entry in summary["rank_shares"].items():
        assert entry["low"] <= k / 49 <= entry["high"], f"rank {k}: {entry['share']:.4f} against {k / 49:.4f}"


def test_the_rotation_period_of_a_variable_curve_is_the_period_the_generator_draws_and_the_explained_ratios_are_exact():
    """R12: the period is the draw that follows the white-noise draw; it is the period of the strongest peak of the spectrum of the generator's own flux, and a period within 1 percent of r times it, for the five ratios, is explained."""
    import numpy as np

    generator_module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("generator_t", VALIDATION / "generator.py"))
    generator_module.__spec__.loader.exec_module(generator_module)
    time, flux, _ = generator_module.noise("variable", 9_200_005, generator_module.time_axis()[0])
    period = analysis.rotation_period(9_200_005, time.size)
    frequencies = np.linspace(1 / 12.0, 1 / 0.5, 20000)
    power = [abs(np.sum((flux - flux.mean()) * np.exp(-2j * np.pi * f * time))) for f in frequencies]
    assert abs(1 / frequencies[int(np.argmax(power))] / period - 1) < 0.02, f"the peak of the spectrum is not at the drawn period {period:.4f} d"
    for ratio in (1 / 3, 1 / 2, 1, 2, 3):
        assert analysis.explained_by_rotation(period * ratio * 1.0099, period) and analysis.explained_by_rotation(period * ratio * 0.9901, period)
        assert not analysis.explained_by_rotation(period * ratio * 1.0101, period)
    assert not analysis.explained_by_rotation(period * 1.5, period)


def test_the_red_check_reports_the_rank_share_of_the_new_copies_beside_the_old_and_the_paired_difference():
    """R13 (a) and (b) on made-up rows: the mean share of the copies below the original for the old and the new copies of the same curves, the paired difference with its standard error, the crossings of the new copies and
    the median ratio of the largest new copy to the largest old one."""
    old = [null(i, 5.0, [4.0, 6.0, 7.0, 8.0], [0.0]) for i in range(4)]  # the original is above 1 of 4 copies: share 0.25
    new = [{"id": f"reflat:red:{i}", "class": "red", "index": i, "sde": 5.0, "shuffle_sde": [4.0, 4.5, 5.5, 9.0]} for i in range(4)]  # above 2 of 4: share 0.5; largest copy 9.0 against 8.0
    summary = analysis.reflat_summary(old, new)
    assert summary["curves"] == 4
    assert summary["old_rank_share"] == pytest.approx(0.25) and summary["new_rank_share"] == pytest.approx(0.5)
    assert summary["difference"] == pytest.approx(0.25) and summary["difference_standard_error"] == pytest.approx(0.0, abs=1e-12)
    assert summary["crossings"]["count"] == 0 and summary["median_ratio_of_largest_copy"] == pytest.approx(9.0 / 8.0)


def test_real_stars_of_two_lists_are_reported_apart_and_pooled_over_the_distinct_stars():
    def star(tic, share):
        return {"id": f"real:{tic}:1", "tic": tic, "sector": 1, "rank_share": share}

    rows = [star(1, 0.9), star(2, 0.8), star(3, 0.1), star(3, 0.1)]  # star 3 appears twice, as the two lists share some stars
    summary = analysis.real_summary(rows, {"real:1:1", "real:3:1"})
    assert summary["current_list"]["stars"] == 2 and summary["first_list"]["stars"] == 1 and summary["pooled"]["stars"] == 3
    assert summary["pooled"]["mean_rank_share"] == pytest.approx((0.9 + 0.8 + 0.1) / 3)
    assert analysis.real_summary(rows)["stars"] == 3
