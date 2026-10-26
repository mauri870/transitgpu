"""The measured differences between exact mode and TLS fused on the eleven adversarial cases of tests/test_tls_adversarial.py, written to paper/adversarial_table.json: for each case the largest relative difference of chi square over the trial periods,
the largest relative difference over every other field, whether the best period is the same and the two best periods.

    python paper/scripts/adversarial.py [--out paper/adversarial_table.json]

Needs a GPU (a few seconds). The tests bound chi square at a relative 1e-6; this file holds the largest value that was measured.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(reference, mine):
    """{"chi2_max_relative", "other_fields_max_relative", "best_period_same", "best_period_tls", "best_period_gpu"} of two result dicts of arrays: the largest |a - b| / max(|b|, 1e-12) over chi square, and over the other fields of the results (elements that are NaN in both count as equal)."""
    def largest(key):
        got, expected = np.asarray(mine[key], dtype=float), np.asarray(reference[key], dtype=float)
        if got.shape != expected.shape:
            return float("inf")
        absolute = np.where(np.isnan(expected) & np.isnan(got), 0.0, np.abs(got - expected))
        relative = absolute / np.maximum(np.abs(expected), 1e-12)
        return float(np.nanmax(relative)) if relative.size and not np.all(np.isnan(relative)) else 0.0

    others = [largest(key) for key in reference if key != "chi2"]
    return {"chi2_max_relative": largest("chi2"), "other_fields_max_relative": max(others) if others else 0.0,
            "best_period_same": bool(float(mine["period"]) == float(reference["period"])), "best_period_tls": float(reference["period"]), "best_period_gpu": float(mine["period"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE.parent / "adversarial_table.json")
    arguments = parser.parse_args()
    cases = load(PACKAGE / "tests" / "test_tls_adversarial.py", "adversarial_cases")
    table = {name: summary(cases.run(name, "fused"), cases.run(name, "gpu")) for name in cases.CASES}
    arguments.out.write_text(json.dumps(table, indent=1))
    for name, row in table.items():
        print(f"{name:26s} chi2 {row['chi2_max_relative']:.1e}  other fields {row['other_fields_max_relative']:.1e}  best period {'same' if row['best_period_same'] else 'differs'}")


if __name__ == "__main__":
    main()
