"""How far the approximate searches are from the exact backend of TLS on its 14 characterization cases: for each case, the best period of TLS fused (the reference), whether the binned mode of transitgpu.tls (backend "gpu-binned")
and TLS fused-pl find that period, and the difference of their SDE from that of TLS fused. The binned mode needs equal errors and refuses the cases with unequal ones.

    python paper/scripts/binned_accuracy.py [--out paper/binned_table.tex]

Needs a GPU (about a minute). The table is the source of the figures of Sect. "Binned mode" of the paper.
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
import golden_cases  # noqa: E402

import transitgpu.tls  # noqa: E402,F401  (registers the backends)


def search(name, backend):
    os.environ["TLS_GOLDEN_BACKEND"] = backend
    arrays = golden_cases.to_arrays(golden_cases.run(name))
    return float(arrays["period"]), float(arrays["SDE"])


def same(a, b):
    return a == b or (np.isnan(a) and np.isnan(b))


def table_of(rows):
    """Return the LaTeX table of the cases from the list of rows that binned_table.json holds."""
    lines = [f"% written by scripts/binned_accuracy.py: {len(rows)} cases", r"\begin{tabular}{lcrcr}", r"\toprule", r"case & binned period & binned $|\Delta\mathrm{SDE}|$ & fused-pl period & fused-pl $|\Delta\mathrm{SDE}|$ \\", r"\midrule"]
    for r in rows:
        name = r["case"].replace("_", r"\_")
        if r["binned"] == "refused":
            binned = r"\multicolumn{2}{c}{unequal errors, refused}"
        else:
            binned = f"{'same' if r['binned_same_period'] else 'different'} & {r['binned_sde_difference']:.3f}"
        lines.append(rf"\texttt{{{name}}} & {binned} & {'same' if r['fused_pl_same_period'] else 'different'} & {r['fused_pl_sde_difference']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PACKAGE / "paper" / "binned_table.tex"))
    options = parser.parse_args()
    rows = []
    for name in golden_cases.CASES:
        exact_period, exact_sde = search(name, "fused")
        pl_period, pl_sde = search(name, "fused-pl")
        row = {"case": name, "period": exact_period, "fused_pl_same_period": bool(same(pl_period, exact_period)), "fused_pl_sde_difference": abs(pl_sde - exact_sde) if np.isfinite(exact_sde) else 0.0}
        try:
            period, sde = search(name, "gpu-binned")
            row.update({"binned": "searched", "binned_same_period": bool(same(period, exact_period)), "binned_sde_difference": abs(sde - exact_sde) if np.isfinite(exact_sde) else 0.0})
        except ValueError as error:
            assert "equal errors" in str(error)
            row["binned"] = "refused"
        rows.append(row)
    Path(options.out).write_text(table_of(rows))
    Path(options.out).with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
    searched = [r for r in rows if r["binned"] == "searched"]
    print(f"{len(rows)} cases, {len(rows) - len(searched)} refused by binned mode; binned: {sum(not r['binned_same_period'] for r in searched)} different periods, largest SDE difference {max(r['binned_sde_difference'] for r in searched):.3f}; "
          f"fused-pl: {sum(not r['fused_pl_same_period'] for r in rows)} different periods, largest SDE difference {max(r['fused_pl_sde_difference'] for r in rows):.3f}")


if __name__ == "__main__":
    main()
