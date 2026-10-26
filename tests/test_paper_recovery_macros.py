"""The macros of the paired recovery differences (paper/scripts/fill_numbers.py, recovery_macros), from a recovery file of the schema of validation/analyse.py that these tests make up: for each class and each search the difference of the share found from exact mode
at a matched false-alarm rate, as signed points with the interval, its size, and the largest size of a search over the classes. Function tests on synthetic numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "fill_numbers.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("fill_numbers_recovery", SCRIPT)
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)


def paired(**differences):
    return {"0.05": {f"{name}_minus_exact": {"estimate": value[0], "low": value[1], "high": value[2], "n": 800} for name, value in differences.items()}}


RESULTS = {
    "main-white": {"paired": paired(binned=(0.001, -0.005, 0.007), box=(0.001, -0.006, 0.009), hybrid5=(0.006, 0.001, 0.015))},
    "main-red": {"paired": paired(binned=(-0.001, -0.004, 0.007), box=(-0.061, -0.080, -0.047), hybrid5=(-0.051, -0.071, -0.035))},
    "main-variable": {"paired": paired(binned=(-0.0140, -0.025, -0.006), box=(-0.310, -0.343, -0.279), hybrid5=(-0.309, -0.342, -0.277))},
}


def test_each_class_and_search_has_the_signed_difference_and_its_interval_in_points():
    lines = fill.recovery_macros(RESULTS)
    for expected in ("pd@variable@binned@est\\endcsname{$-1.4$}", "pd@variable@binned@low\\endcsname{$-2.5$}", "pd@variable@binned@high\\endcsname{$-0.6$}", "pd@white@hybrid5@est\\endcsname{$+0.6$}", "pd@red@box@est\\endcsname{$-6.1$}",
                     "pd@red@box@size\\endcsname{6.1}", "pd@red@box@round\\endcsname{6}", "pd@red@hybrid5@round\\endcsname{5}"):
        assert "\\csname " + expected in lines, expected


def test_the_largest_size_of_a_search_over_the_classes_is_written_to_one_decimal_with_halves_rounded_up_and_without_trailing_zeros():
    lines = fill.recovery_macros(RESULTS)
    assert "\\csname pd@binned@maxsize\\endcsname{1.4}" in lines and "\\csname pd@box@maxsize\\endcsname{31}" in lines, lines
    exact = {name: {"paired": paired(binned=(0.0125, 0.0, 0.02))} for name in RESULTS}
    assert "\\csname pd@binned@maxsize\\endcsname{1.3}" in fill.recovery_macros(exact), "1.25 points is 1.3"
    banded = {name: {"paired": paired(binned=(-0.01375, -0.0249, -0.006))} for name in RESULTS}
    assert "\\csname pd@binned@maxsize\\endcsname{1.4}" in fill.recovery_macros(banded), "1.375 points is 1.4"


def test_a_file_without_a_paired_block_gives_no_macros():
    assert fill.recovery_macros({}) == "" and fill.recovery_macros({"main-white": {"searches": {}}}) == ""


def test_the_largest_change_from_5_to_20_refined_peaks_is_a_macro():
    results = {name: {"paired": paired(hybrid5=(-0.309, -0.342, -0.277), hybrid20=(-0.306, -0.340, -0.275))} for name in ("main-variable",)}
    assert "\\csname pd@hybrid20from5@maxsize\\endcsname{0.3}" in fill.recovery_macros(results)


def test_the_banded_search_takes_the_place_of_binned_mode_part_by_part_and_a_missing_part_is_refused():
    merge_spec = importlib.util.spec_from_file_location("merge_recovery", SCRIPT.parent / "merge_recovery.py")
    merge = importlib.util.module_from_spec(merge_spec)
    merge_spec.loader.exec_module(merge)
    old = {"main-white": {"searches": {"exact": {"a": 1}, "binned": {"a": 2}, "box": {"a": 3}}, "paired": {"0.05": {"binned_minus_exact": {"estimate": 0.0}, "box_minus_exact": {"estimate": 0.1}}}}}
    bands = {"main-white": {"searches": {"exact": {"a": 1}, "binned": {"a": 2}, "binned_bands": {"a": 9}}, "paired": {"0.05": {"binned_bands_minus_exact": {"estimate": -0.014}, "binned_minus_exact": {"estimate": 0.0}}}}}
    out = merge.merged(old, bands)
    assert out["main-white"]["searches"]["binned"] == {"a": 9} and out["main-white"]["searches"]["box"] == {"a": 3}
    assert out["main-white"]["paired"]["0.05"]["binned_minus_exact"] == {"estimate": -0.014} and out["main-white"]["paired"]["0.05"]["box_minus_exact"] == {"estimate": 0.1}
    assert old["main-white"]["searches"]["binned"] == {"a": 2}, "the input is not changed"
    with pytest.raises(ValueError, match="main-red"):
        merge.merged({**old, "main-red": old["main-white"]}, bands)
