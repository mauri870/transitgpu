"""The largest differences, by field, between the exact search of transitgpu.tls and TLS 2.0.1's exact backend.

Over the 14 characterization cases of TLS (Sect. "The exact template search against TLS" of the paper), each case is
run with TLS_GOLDEN_BACKEND=fused and with the backend "gpu" of transitgpu.tls, and every field of the results is
compared: the largest relative difference |a - b| / max(|b|, 1e-12) over the elements and the cases, and for the
spectra (power, power_raw, SR) the largest absolute difference.

    python paper/scripts/validation.py [--out paper/validation_table.tex]

Needs a GPU (about half a minute for the 14 cases).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
sys.path.insert(0, str(PACKAGE / "tests" / "golden_upstream"))
SPECTRA = ("power", "power_raw", "SR")


def differences(reference, mine):
    """Return {field: (largest relative difference, largest absolute difference)} between two result dicts.

    Elements that are NaN in both count as equal.
    """
    out = {}
    for key, expected in reference.items():
        got = mine[key]
        both_nan = np.isnan(expected) & np.isnan(got)
        absolute = np.where(both_nan, 0.0, np.abs(got - expected))
        relative = absolute / np.maximum(np.abs(expected), 1e-12)
        out[key] = (float(np.nanmax(relative)) if relative.size else 0.0, float(np.nanmax(absolute)) if absolute.size else 0.0)
    return out


def table_of(worst, case_count):
    """Return the LaTeX table of the ten largest differences from {field: (relative, absolute, case)}, as validation_table.json holds it."""
    rows = sorted(((key, *value) for key, value in worst.items() if value[0] > 0 or value[1] > 0), key=lambda row: -row[1])
    lines = [f"% written by scripts/validation.py: {len(worst)} fields over {case_count} cases; {len(worst) - len(rows)} fields are equal", "\\begin{tabular}{lrrl}", "\\toprule", "field & relative & absolute & case \\\\", "\\midrule"]
    for key, relative, absolute, name in rows[:10]:
        shown = key.replace("_", r"\_")
        lines.append(f"\\texttt{{{shown}}} & {relative:.1e} & {absolute:.1e} & \\texttt{{{name.replace('_', chr(92) + '_')}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE.parent / "validation_table.tex")
    arguments = parser.parse_args()
    import golden_cases
    import transitgpu.tls  # noqa: F401  (registers the backend "gpu")

    worst = {}
    for name in golden_cases.CASES:
        os.environ["TLS_GOLDEN_BACKEND"] = "fused"
        reference = golden_cases.to_arrays(golden_cases.run(name))
        os.environ["TLS_GOLDEN_BACKEND"] = "gpu"
        mine = golden_cases.to_arrays(golden_cases.run(name))
        for key, (relative, absolute) in differences(reference, mine).items():
            old = worst.get(key, (0.0, 0.0, ""))
            worst[key] = (max(old[0], relative), max(old[1], absolute), name if relative > old[0] else old[2])
    arguments.out.with_suffix(".json").write_text(json.dumps(worst, indent=1))
    text = table_of(worst, len(golden_cases.CASES))
    arguments.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
