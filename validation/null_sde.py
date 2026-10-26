"""The distribution of the noise-only SDE of the searches, for the figure of the paper: for each part and each search, the binned counts of its SDE over the part's null light curves with the 5 percent threshold (the 95th
percentile that analyse.py takes), and, from the rows of the calibration of the block-shuffle threshold (threshold.py), the distribution of the box search's 48-shuffle threshold of each noise-only light curve of the same
class. usage:
python null_sde.py DIRECTORY [--bands BANDS_DIRECTORY] [--real REAL_DIRECTORY] [--calibration ROWS.jsonl] [--out ../paper/null_sde.json] [--bins 40]
DIRECTORY, BANDS_DIRECTORY and REAL_DIRECTORY are those of roc.py; ROWS.jsonl is the file that threshold.py writes (null rows `null:CLASS:INDEX` and real-star rows `real:TIC:SECTOR`)."""
import argparse
import json
from pathlib import Path

import numpy as np

import analyse
import roc

SEARCHES = roc.SEARCHES
SHUFFLES = 48
CLASSES = {"main-white": "white", "main-red": "red", "main-variable": "variable", "real": "real"}


def histogram(sdes, bins):
    """The counts and the edges of `bins` equal bins from the smallest to the largest value."""
    counts, edges = np.histogram(np.asarray(sdes, dtype=float), bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def part_histograms(records, searches, bins):
    """The null light curves of a part: their number and, for each search, the histogram of its SDE and its 5 percent threshold."""
    nulls = [r for r in records if r["id"].split(":")[2] == "null"]
    out = {"nulls": len(nulls), "searches": {}}
    for search in searches:
        sdes = [analyse.statistic(r, search) for r in nulls]
        out["searches"][search] = {**histogram(sdes, bins), "threshold_5": float(analyse.threshold(sdes, 0.05)[0])}
    return out


def shuffle_thresholds(rows, kind):
    """The 48-shuffle threshold (the largest SDE of the first 48 shuffles) of each noise-only light curve of a class of the calibration rows ('real' for the real stars); a row with fewer shuffles is an error."""
    values = []
    for row in rows:
        parts = row["id"].split(":")
        if kind == "real":
            if parts[0] != "real":
                continue
        elif parts[0] != "null" or row.get("class") != kind:
            continue
        shuffles = row["shuffle_sde"]
        if len(shuffles) < SHUFFLES:
            raise ValueError(f"{row['id']}: fewer than {SHUFFLES} shuffles")
        values.append(float(max(shuffles[:SHUFFLES])))
    return values


def main(directory, bands, real, calibration, out, bins):
    parts = {name: roc.read(Path(directory) / f"{name}.jsonl") for name in roc.PARTS}
    if bands:
        parts = {name: roc.with_bands(records, roc.read(Path(bands) / f"{name}.binned-bands.jsonl")) for name, records in parts.items()}
    if real:
        parts["real"] = roc.read(Path(real) / "real.jsonl")
    rows = roc.read(calibration) if calibration else []
    result = {"searches": list(SEARCHES), "shuffles": SHUFFLES, "parts": {}}
    for name, records in parts.items():
        entry = part_histograms(records, SEARCHES, bins)
        values = shuffle_thresholds(rows, CLASSES[name]) if rows else []
        if values:
            entry["shuffle_threshold"] = {"curves": len(values), "values": sorted(values) if name == "real" else None, **(histogram(values, bins) if name != "real" else {}), "median": float(np.median(values))}
        result["parts"][name] = entry
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for name, entry in result["parts"].items():
        print(name, entry["nulls"], "nulls; thresholds at 5%:", {s: round(e["threshold_5"], 2) for s, e in entry["searches"].items()}, "; shuffle thresholds:", entry.get("shuffle_threshold", {}).get("curves"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--bands")
    parser.add_argument("--real")
    parser.add_argument("--calibration")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "paper" / "null_sde.json"))
    parser.add_argument("--bins", type=int, default=40)
    arguments = parser.parse_args()
    main(arguments.directory, arguments.bands, arguments.real, arguments.calibration, arguments.out, arguments.bins)
