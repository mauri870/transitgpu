"""The recovery file of the paper when binned mode is the banded search: the results.json of validation/analyse.py for the box search, the hybrid and exact mode, and the banded search in the place of binned mode.

    python paper/scripts/merge_recovery.py RESULTS.json BANDS.json [--out paper/recovery.json]

RESULTS.json is the analysis of the parts main-white, main-red and main-variable with the searches exact, binned, box and hybrid; BANDS.json is the analysis of the same parts with the searches exact, binned and binned_bands. For each part the search binned of the output is
the search binned_bands of BANDS.json, and the paired difference binned_minus_exact is its binned_bands_minus_exact; the other searches are those of RESULTS.json.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def merged(results, bands):
    """The results with binned mode replaced by the banded search of `bands`, part by part; a part that `bands` lacks is refused."""
    out = json.loads(json.dumps(results))
    for part, content in out.items():
        if part not in bands:
            raise ValueError(f"no banded results for the part {part}")
        banded = bands[part]
        content["searches"]["binned"] = banded["searches"]["binned_bands"]
        for rate, differences in banded["paired"].items():
            content["paired"][rate]["binned_minus_exact"] = differences["binned_bands_minus_exact"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("bands", type=Path)
    parser.add_argument("--out", type=Path, default=HERE.parent / "recovery.json")
    arguments = parser.parse_args()
    arguments.out.write_text(json.dumps(merged(json.loads(arguments.results.read_text()), json.loads(arguments.bands.read_text())), indent=1))


if __name__ == "__main__":
    main()
