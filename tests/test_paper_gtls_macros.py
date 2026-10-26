"""The macros of the times that GTLS publishes and their ratios to our rows at the margin of GTLS (paper/scripts/fill_numbers.py, gtls_macros), on a made-up results file. Function tests on made-up numbers."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "fill_numbers.py"
if not SCRIPT.exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("fill_numbers_gtls", SCRIPT)
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)


def results(rows):
    return {"t0_fit_margin": 0.01, "gtls_input": {"runs": [{"t0_fit_margin": 0.125, "sizes": {str(points): {"rows": {row: {"median": median, "trial_periods": 10} for row, median in found.items()}} for points, found in rows.items()}}]}}


def test_the_published_times_and_the_ratios_to_the_rows_of_the_margin_of_gtls():
    text = fill.gtls_macros(results({72000: {"tls-gpu": 6.0, "tls16-exact": 3.0}, 144000: {"tls-gpu": 30.0, "tls16-exact": 12.0}}))
    assert "\\providecommand{\\Gt}" in text
    for expected in ("gt@time@72000\\endcsname{33.3}", "gt@time@144000\\endcsname{138}", "gt@tls@72000\\endcsname{522}", "gt@tls@144000\\endcsname{3\\,289}",
                     "gt@ratio@72000\\endcsname{5.5}", "gt@ratio@144000\\endcsname{4.6}", "gt@tlsratio@72000\\endcsname{174}", "gt@tlsratio@144000\\endcsname{274}"):
        assert "\\csname " + expected in text, expected


def test_a_size_without_the_rows_of_the_margin_has_the_published_times_and_no_ratio():
    text = fill.gtls_macros(results({72000: {"tls-gpu": 6.0}}))
    assert "gt@ratio@72000" in text and "gt@tlsratio@72000" not in text
    assert "gt@time@144000" in text and "gt@ratio@144000" not in text and "gt@tlsratio@144000" not in text
