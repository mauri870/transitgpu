"""The chi-square of exact mode against TLS fused at every trial period of the 14 characterization cases of TLS (tests/golden_upstream/golden_cases.py), for the figure and the table of the paper: for each case the
relative difference (exact mode over TLS fused, less one) at the trial periods, its largest magnitude, and for one case the difference at every period. Exact mode runs on a GPU, TLS fused on the CPU.

    python validation/chi2_difference.py [--out paper/chi2_difference.json] [--figure-case k2_box]

The two searches are those of tests/test_tls_golden.py: the same light curve, grid and options, the backend `fused` of TLS 2.0.1 and the backend `gpu` of transitgpu.tls."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests" / "golden_upstream"))


def relative_difference(reference, other):
    """other / reference - 1 at every period where the reference is finite and positive, NaN elsewhere."""
    reference, other = np.asarray(reference, dtype=float), np.asarray(other, dtype=float)
    found = np.full(reference.shape, np.nan)
    usable = np.isfinite(reference) & (reference > 0)
    found[usable] = other[usable] / reference[usable] - 1.0
    return found


def summary(periods_reference, reference, other, periods_other):
    """{"periods", "periods_equal", "maximum_relative_difference", "compared"} of a case: the number of trial periods, whether the two grids are the same, the largest magnitude of the relative difference of the chi-square
    over the periods that have a reference, and how many periods that is. A case whose grids differ is not compared."""
    equal = bool(np.array_equal(np.asarray(periods_reference), np.asarray(periods_other)))
    if not equal:
        return {"periods": int(np.size(periods_reference)), "periods_equal": False, "maximum_relative_difference": None, "compared": 0}
    found = relative_difference(reference, other)
    kept = found[np.isfinite(found)]
    return {"periods": int(found.size), "periods_equal": True, "maximum_relative_difference": float(np.max(np.abs(kept))) if kept.size else None, "compared": int(kept.size)}


def run(name, backend):
    import golden_cases

    previous = os.environ.get("TLS_GOLDEN_BACKEND")
    os.environ["TLS_GOLDEN_BACKEND"] = backend
    try:
        return golden_cases.to_arrays(golden_cases.run(name))
    finally:
        if previous is None:
            os.environ.pop("TLS_GOLDEN_BACKEND", None)
        else:
            os.environ["TLS_GOLDEN_BACKEND"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE.parent / "paper" / "chi2_difference.json")
    parser.add_argument("--figure-case", default="k2_box")
    arguments = parser.parse_args()
    import golden_cases
    import torch

    import transitgpu.tls  # noqa: F401  (registers the backends "gpu" and "gpu-binned")

    if not torch.cuda.is_available():
        sys.exit("exact mode searches on a GPU only: no device")
    if arguments.figure_case not in golden_cases.CASES:
        sys.exit(f"{arguments.figure_case} is not a characterization case: {list(golden_cases.CASES)}")
    cases, figure = {}, None
    for name in golden_cases.CASES:
        reference, mine = run(name, "fused"), run(name, "gpu")
        cases[name] = summary(reference["periods"], reference["chi2"], mine["chi2"], mine["periods"])
        print(name, cases[name], flush=True)
        if name == arguments.figure_case and cases[name]["periods_equal"]:
            figure = {"case": name, "periods_days": [float(p) for p in reference["periods"]], "relative_difference": [None if not np.isfinite(v) else float(v) for v in relative_difference(reference["chi2"], mine["chi2"])]}
    arguments.out.write_text(json.dumps({"cases": cases, "figure": figure}, indent=1) + "\n")


if __name__ == "__main__":
    main()
