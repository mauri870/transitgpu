"""The macros of the paper's numbers: a time of results.json is one macro per row and size, the first block that holds a row wins, and a time that is missing is not invented. A function test: the
structure under test is the file that the script writes."""
import importlib.util
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
if not (PACKAGE / "paper").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("numbers_script", PACKAGE / "paper" / "scripts" / "fill_numbers.py")
numbers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numbers)


def block(median, periods=91607):
    return {"sizes": {"200000": {"periods": periods, "rows": {"gpu": {"median": median, "trial_periods": periods}}}}}


def test_the_first_block_that_holds_a_row_wins_and_a_time_is_given_to_three_figures():
    text = numbers.macros({**block(0.84829), "tls": block(9.9), "hardware": {"gpu": "AMD"}, "versions": {"torch": "2.12"}})
    assert "\\csname t@gpu@200000\\endcsname{0.848}" in text and "9.9" not in text
    assert "\\csname p@gpu@200000\\endcsname{91\\,607}" in text and "\\csname h@gpu\\endcsname{AMD}" in text


def test_a_results_file_without_a_block_gives_no_time_and_a_time_of_a_hundred_seconds_or_more_has_no_decimals():
    assert numbers.macros({}) == "\n"
    assert "{292}" in numbers.macros(block(291.784))


def test_every_row_of_the_results_file_that_ships_has_a_macro():
    import json

    results = json.loads((PACKAGE / "benchmarks" / "results.json").read_text())
    text = numbers.macros(results)
    assert "t@gpu@200000" in text and "t@tls-gpu@200000" in text and "t@gpu-shuffles@200000" in text and "t@tls16@200000" in text


def test_the_numbers_file_of_the_paper_is_the_one_that_results_json_gives():
    import json

    written = (PACKAGE / "paper" / "numbers.tex").read_text()
    results = json.loads((PACKAGE / "benchmarks" / "results.json").read_text())
    expected = numbers.macros(results) + numbers.gtls_macros(results) + numbers.cold_macros(results)
    assert written.split("\n", 1)[1] == expected, "run python paper/scripts/fill_numbers.py"


def test_a_ratio_is_the_time_of_the_slower_row_over_that_of_the_faster_at_the_same_size_and_is_absent_where_a_row_is_missing():
    block_ = {"sizes": {"200000": {"periods": 5, "rows": {"gpu": {"median": 0.8, "trial_periods": 5}, "tls16": {"median": 23.2, "trial_periods": 5}}}}}
    text = numbers.macros(block_)
    assert "\\csname r@tls16@gpu@200000\\endcsname{29}" in text and "tls-gpu" not in text
    assert numbers.ratio(344.3) == "344" and numbers.ratio(1.049) == "1"


def test_the_runs_at_the_input_of_the_gtls_paper_have_macros_and_the_run_at_the_margin_of_gtls_keeps_its_rows_apart():
    def run(margin, tls, exact):
        return {"t0_fit_margin": margin, "sizes": {"72000": {"periods": 191837, "rows": {"tls16-exact": {"median": tls, "trial_periods": 191837}, "tls-gpu": {"median": exact, "trial_periods": 191837}}}}}

    text = numbers.macros({"t0_fit_margin": 0.01, "gtls_input": {"runs": [run(0.01, 37.06, 21.42), run(0.125, 9.34, 10.87)]}})
    assert "\\csname t@tls16-exact@72000\\endcsname{37.1}" in text and "\\csname t@tls-gpu@72000\\endcsname{21.4}" in text
    assert "\\csname t@tls16-exact-margin@72000\\endcsname{9.34}" in text and "\\csname t@tls-gpu-margin@72000\\endcsname{10.9}" in text
    assert "\\csname r@tls16-exact-margin@tls-gpu-margin@72000\\endcsname{0.86}" in text


def test_a_published_time_of_four_digits_is_set_with_a_thin_space_like_every_number_of_four_digits_of_the_paper():
    text = numbers.gtls_macros({})
    assert "\\csname gt@tls@144000\\endcsname{3\\,289}" in text, "3289 s is written 3\\,289"
    assert "\\csname gt@tls@72000\\endcsname{522}" in text and "\\csname gt@time@144000\\endcsname{138}" in text, "three digits stay as they are"


def test_the_cold_first_call_of_each_row_has_a_macro_for_the_caches_kept_and_for_the_caches_cleared_and_a_row_missing_from_a_dictionary_has_none():
    results = {"cold_first_call": {"caches_kept": {"gpu": 2.7123, "many": 2.18}, "caches_cleared": {"gpu": 2.865, "many": 18.0659}}}
    text = numbers.cold_macros(results)
    assert "\\providecommand{\\Cold}[2]" in text
    assert "\\csname cold@gpu@kept\\endcsname{2.71}" in text and "\\csname cold@many@cleared\\endcsname{18.1}" in text
    partial = numbers.cold_macros({"cold_first_call": {"caches_kept": {"hybrid": 2.62}, "caches_cleared": {}}})
    assert "cold@hybrid@kept" in partial and "cold@hybrid@cleared" not in partial
    assert numbers.cold_macros({}) == "", "a results file without the block gives nothing"


def test_the_cold_macros_of_the_results_file_that_ships_are_its_cold_first_calls_to_three_figures():
    import json

    results = json.loads((PACKAGE / "benchmarks" / "results.json").read_text())
    text = numbers.cold_macros(results)
    for kind, key in (("kept", "caches_kept"), ("cleared", "caches_cleared")):
        for row, value in results["cold_first_call"][key].items():
            assert f"\\csname cold@{row}@{kind}\\endcsname{{{numbers.seconds(value)}}}" in text, (row, kind)


def adversarial(chi2, other, same, tls, gpu):
    return {"chi2_max_relative": chi2, "other_fields_max_relative": other, "best_period_same": same, "best_period_tls": tls, "best_period_gpu": gpu}


def test_the_adversarial_maxima_and_periods_have_macros_with_the_maxima_over_the_cases_and_infinity_where_a_field_is_undefined():
    cases = {"tied": adversarial(1.17e-13, 4.52e-9, True, 2.001, 2.001), "tiny": adversarial(2.8e-13, float("inf"), False, 1.3897885, 1.4135101), "gapped": adversarial(1.1e-13, 1.06e-9, True, 3.3, 3.3)}
    text = numbers.adversarial_macros(cases)
    assert "\\providecommand{\\Adv}[2]" in text
    assert "\\csname adv@tied@chi2\\endcsname{\\ensuremath{1.2\\times10^{-13}}}" in text and "\\csname adv@tied@other\\endcsname{\\ensuremath{4.5\\times10^{-9}}}" in text
    assert "\\csname adv@tiny@other\\endcsname{\\ensuremath{\\infty}}" in text
    assert "\\csname adv@tiny@periodtls\\endcsname{1.3898}" in text and "\\csname adv@tiny@periodgpu\\endcsname{1.4135}" in text
    assert "\\csname adv@all@chi2\\endcsname{\\ensuremath{2.8\\times10^{-13}}}" in text, "the maximum over every case"
    assert "\\csname adv@all@other\\endcsname{\\ensuremath{4.5\\times10^{-9}}}" in text, "over the cases whose best period is the same: the infinite one is left out"
    assert numbers.adversarial_macros({}) == ""


def test_the_adversarial_macros_of_the_file_that_ships_are_its_maxima():
    import json

    cases = json.loads((PACKAGE / "paper" / "adversarial_table.json").read_text())
    text = numbers.adversarial_macros(cases)
    assert "\\csname adv@all@chi2\\endcsname{\\ensuremath{2.8\\times10^{-13}}}" in text and "\\csname adv@all@other\\endcsname{\\ensuremath{4.5\\times10^{-9}}}" in text
    assert "\\csname adv@tiny_light_curve@periodtls\\endcsname{1.3898}" in text and "\\csname adv@tiny_light_curve@periodgpu\\endcsname{1.4135}" in text


def test_the_versions_of_triton_batman_and_wotan_are_in_the_results_that_ship_and_have_macros():
    import json

    results = json.loads((PACKAGE / "benchmarks" / "results.json").read_text())
    text = numbers.macros(results)
    for name in ("triton", "batman", "wotan"):
        assert results["versions"][name], name
        assert f"\\csname v@{name}\\endcsname{{{results['versions'][name]}}}" in text, name


def period_results():
    cell = lambda found, n, low, high: {"found": found, "n": n, "wilson": [low, high]}
    searches = {"exact": {"0.05": {"by_period": {"0.8": cell(194, 240, 0.7538, 0.8531), "12.8": cell(120, 240, 0.4373, 0.5627)}}}, "box": {"0.05": {"by_period": {"0.8": cell(150, 240, 0.5664, 0.6878)}}}}
    return {"main-white": {"searches": searches}, "real": {"searches": {"exact": {"0.05": {"by_period": {"0.8": cell(154, 216, 0.6493, 0.7692)}}}}}}


def test_the_share_found_at_each_period_has_a_macro_with_its_wilson_bounds_and_the_part_loses_its_prefix():
    text = numbers.period_macros(period_results())
    assert "\\providecommand{\\Rp}[3]" in text
    assert "\\csname rp@white@exact@0.8\\endcsname{80.8}" in text and "\\csname rpl@white@exact@0.8\\endcsname{75.4}" in text and "\\csname rph@white@exact@0.8\\endcsname{85.3}" in text
    assert "\\csname rp@white@exact@12.8\\endcsname{50.0}" in text and "\\csname rp@white@box@0.8\\endcsname{62.5}" in text and "\\csname rp@real@exact@0.8\\endcsname{71.3}" in text
    assert "rp@white@box@12.8" not in text, "a period missing from the file has no macro"
    assert numbers.period_macros({}) == ""


def test_the_period_macros_of_the_files_that_ship_are_their_cells():
    import json

    recovery = json.loads((PACKAGE / "paper" / "recovery.json").read_text())
    real = json.loads((PACKAGE / "paper" / "recovery_real.json").read_text())
    text = numbers.period_macros({**recovery, "real": real["real"]})
    cell = recovery["main-white"]["searches"]["binned"]["0.05"]["by_period"]["12.8"]
    assert f"\\csname rp@white@binned@12.8\\endcsname{{{100 * cell['found'] / cell['n']:.1f}}}" in text
