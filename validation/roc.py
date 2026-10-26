"""ROC curves of the searches on the per-case records of the recovery: for each part and each search, the share of the injected light curves of signal to noise `snr_low` and above that the search finds (the SDE at or above
a threshold, and the best period right) against the share of the null light curves whose SDE is at or above that threshold, at every false-alarm rate of a grid. The threshold at a rate is the 1 - rate quantile of the
SDE of the part's nulls, as validation/analyse.py takes it, so the point at 0.05 is the completeness of analyse.py pooled over the levels of signal to noise. usage:
python roc.py DIRECTORY [--bands BANDS_DIRECTORY] [--real REAL_DIRECTORY] [--out ../paper/roc.json] [--snr-low 8]
DIRECTORY holds main-white.jsonl, main-red.jsonl and main-variable.jsonl (run.py); BANDS_DIRECTORY the binned search run again (rerun_binned.py: PART.binned-bands.jsonl), whose binned column replaces that of the part
when it is given; REAL_DIRECTORY holds real.jsonl. The file written has, for each part, each search's threshold and share found (with its Wilson interval) at each rate, the order of the searches at each rate and the
rates between which two curves cross."""
import argparse
import json
from pathlib import Path

import numpy as np

import analyse

PARTS = ("main-white", "main-red", "main-variable")
SEARCHES = ("exact", "binned", "box", "hybrid5")
RATES = (0.01, 0.05)  # the rates of the macros of the paper
MINIMUM_RATE = 1e-3  # the nulls of a part number about a thousand: a rate below is not resolved


def grid(nulls):
    """The false-alarm rates of the curve: logarithmic from 1e-3 to 1, with the rates of the macros in it."""
    values = set(np.round(np.logspace(np.log10(MINIMUM_RATE), 0, 61), 6)) | set(RATES)
    return sorted(float(v) for v in values)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def with_bands(records, rerun):
    """The records with the binned column of the rerun of the banded search, joined by id; a rerun id that the evaluation lacks is an error, since the comparison is paired."""
    by_id = {r["id"]: r for r in records}
    for new in rerun:
        if new["id"] not in by_id:
            raise ValueError(f"{new['id']} is in the rerun and not in the evaluation")
    replaced = {new["id"]: new["binned"] for new in rerun}
    return [{**r, "binned": replaced.get(r["id"], r["binned"])} for r in records]


def split(records, search, snr_low):
    """(positives, nulls) of a search: the positives are (SDE, found-if-above-threshold) of the injected records with a signal to noise of at least snr_low, where the second value is whether the best period is right
    (for the hybrid, whether one of its first peaks is); the nulls are the SDE of the null records."""
    positives, nulls = [], []
    for record in records:
        kind = record["id"].split(":")[2]
        if kind == "null":
            nulls.append((analyse.statistic(record, search), False))
        elif record["snr"] is not None and record["snr"] >= snr_low:
            right = analyse.found(record, search, -np.inf)
            positives.append((analyse.statistic(record, search), bool(right)))
    return positives, nulls


def curve(null_sde, positives, rates):
    """{"threshold", "tpr", "low", "high"} at each rate: the 1 - rate quantile of the null SDE, the share of the positives whose SDE is at or above it and whose peak is right, and its Wilson interval."""
    null_sde = np.asarray(null_sde, dtype=float)
    out = {"threshold": [], "tpr": [], "low": [], "high": []}
    for rate in rates:
        cut = float(np.quantile(null_sde, 1 - rate))
        k = sum(right and sde >= cut for sde, right in positives)
        low, high = analyse.wilson(k, len(positives))
        out["threshold"].append(cut)
        out["tpr"].append(k / len(positives))
        out["low"].append(float(low))
        out["high"].append(float(high))
    return out


def ordering(tprs, index):
    """The searches from the highest share found to the lowest at the rate at position `index` of the grid (ties in the order given)."""
    return sorted(tprs, key=lambda name: -tprs[name][index])


def crossings(tprs, rates, tolerance=0.0):
    """For each pair of searches whose curves change order on the grid: the intervals of the grid between which the sign of the difference changes, keyed 'first|second'. A difference of at most `tolerance` is a tie
    and not a sign (part_curves takes two injected light curves, so that two curves that never differ by more than that do not cross)."""
    out = {}
    names = list(tprs)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            difference = [a - b for a, b in zip(tprs[first], tprs[second])]
            signs = [(index, np.sign(value)) for index, value in enumerate(difference) if abs(value) > tolerance]
            changes = [[rates[a], rates[b]] for (a, sa), (b, sb) in zip(signs, signs[1:]) if sa != sb]
            if changes:
                out[f"{first}|{second}"] = changes
    return out


def part_curves(records, searches, snr_low, rates):
    """The ROC of the searches on the records of one part: the numbers of positives and nulls, the curves, the order of the searches at the rates of the paper and the crossings."""
    out = {"positives": None, "nulls": None, "searches": {}}
    for search in searches:
        positives, nulls = split(records, search, snr_low)
        if not nulls:
            raise ValueError("the part has no null light curve")
        if not positives:
            raise ValueError(f"the part has no injected light curve of signal to noise {snr_low} or more")
        out["positives"], out["nulls"] = len(positives), len(nulls)
        out["searches"][search] = curve([sde for sde, _ in nulls], positives, rates)
    tprs = {search: entry["tpr"] for search, entry in out["searches"].items()}
    out["ordering"] = {str(rate): ordering(tprs, rates.index(rate)) for rate in RATES if rate in rates}
    out["crossings"] = crossings(tprs, rates, tolerance=2.0 / out["positives"])
    return out


def main(directory, bands, real, out, snr_low):
    result = {"snr_low": snr_low, "searches": list(SEARCHES), "parts": {}}
    rates = None
    parts = {name: read(Path(directory) / f"{name}.jsonl") for name in PARTS}
    if bands:
        parts = {name: with_bands(records, read(Path(bands) / f"{name}.binned-bands.jsonl")) for name, records in parts.items()}
    if real:
        parts["real"] = read(Path(real) / "real.jsonl")
    for name, records in parts.items():
        rates = rates or grid(None)
        result["parts"][name] = part_curves(records, SEARCHES, snr_low, rates)
    result["fpr"] = rates
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for name, entry in result["parts"].items():
        print(name, entry["positives"], "positives,", entry["nulls"], "nulls; order at 0.05:", entry["ordering"]["0.05"], "at 0.01:", entry["ordering"]["0.01"], "crossings:", entry["crossings"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--bands")
    parser.add_argument("--real")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "paper" / "roc.json"))
    parser.add_argument("--snr-low", type=float, default=8.0)
    arguments = parser.parse_args()
    main(arguments.directory, arguments.bands, arguments.real, arguments.out, arguments.snr_low)
