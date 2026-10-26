"""The numbers of the calibration of paper/scripts/threshold_calibration.md from validation/out/threshold.jsonl.

    uv run python validation/threshold_analysis.py [ROWS.jsonl] [--json threshold.json]
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("threshold", HERE / "threshold.py")
threshold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold)
RANKS = tuple(range(1, 11))


def share(count, total):
    low, high = threshold.wilson(count, total) if total else (None, None)
    return {"count": count, "of": total, "share": count / total if total else None, "low": low, "high": high}


def null_summary(rows, ranks=RANKS):
    """Per class: the curves whose best SDE is strictly above the largest of their shuffles, the ties, and the curves above the rank-th largest, with the Wilson intervals of R1 and R3."""
    states = [threshold.compare(row["sde"], row["shuffle_sde"]) for row in rows]
    above = sum(state == "above" for state in states)
    rank_above = {k: sum(threshold.above_rank(row["sde"], row["shuffle_sde"], k) for row in rows) for k in ranks}
    count = len(rows[0]["shuffle_sde"]) + 1 if rows else 49
    return {"curves": len(rows), "above": above, "ties": sum(state == "tie" for state in states), **share(above, len(rows)), "nominal": 1 / count,
            "rank_above": rank_above, "rank_shares": {k: share(rank_above[k], len(rows)) | {"nominal": k / count} for k in ranks}}


def inverted_summary(rows):
    """R5: the curves above the largest SDE of the shuffles of their inversion, and the curves whose two thresholds differ by more than 10 percent of the shuffle threshold."""
    above = sum(threshold.compare(row["sde"], row["inverted_sde"]) == "above" for row in rows)
    far = sum(abs(max(row["inverted_sde"]) - max(row["shuffle_sde"])) > 0.1 * max(row["shuffle_sde"]) for row in rows)
    return {"curves": len(rows), "above": above, **share(above, len(rows)), "thresholds_differ_by_more_than_10_percent": share(far, len(rows))}


def injection_summary(injections, nulls):
    """R6: per class and target S/N (and per injected period), the cases, the peaks that are right or harmonics, the planets found above the shuffle threshold and above the inverted one, and the planets whose own shuffle
    threshold is above that of the same noise without the planet, with the median ratio of the two."""
    by_index = {(row["class"], row["index"]): row for row in nulls}
    out = {}
    for key, name in (("target_snr", None), ("period_injected", "by_period")):
        for kind in sorted({row["class"] for row in injections}):
            for level in sorted({row[key] for row in injections if row["class"] == kind}):
                mine = [row for row in injections if row["class"] == kind and row[key] == level]
                pairs = [(row, by_index[(kind, row["index"])]) for row in mine if (kind, row["index"]) in by_index]
                found_shuffle = sum(row["label"] == "right" and row["sde"] >= max(row["shuffle_sde"]) for row in mine)
                found_inverted = sum(row["label"] == "right" and row["sde"] >= max(row["inverted_sde"]) for row in mine)
                ratios = [max(row["shuffle_sde"]) / max(noise["shuffle_sde"]) for row, noise in pairs]
                cell = {"cases": len(mine), "right": sum(row["label"] == "right" for row in mine), "harmonic": sum(row["label"] == "harmonic" for row in mine), "found_shuffle": found_shuffle, "found_inverted": found_inverted,
                        "found_shuffle_share": share(found_shuffle, len(mine)), "found_inverted_share": share(found_inverted, len(mine)), "threshold_raised_by_planet": sum(r > 1 for r in ratios),
                        "threshold_ratio_median": float(np.median(ratios)) if ratios else None, "paired": len(pairs)}
                target = out.setdefault(kind, {}) if name is None else out.setdefault(kind, {}).setdefault(name, {})
                target[str(level)] = cell
    return out


ROTATION_RATIOS = (1 / 3, 1 / 2, 1, 2, 3)  # R12
ROTATION_TOLERANCE = 0.01


def rotation_period(seed, size):
    """The rotation period of a curve of the variable class, in days: the draw that follows the white-noise draw of `size` points in the generator of that seed."""
    generator = threshold.load("generator")
    draws = np.random.default_rng(seed)
    draws.standard_normal(size)
    return float(np.exp(draws.uniform(*np.log(generator.VARIABLE_PERIOD_RANGE))))


def explained_by_rotation(period, rotation):
    """R12: the period is within 1 percent of r times the rotation period, for r in ROTATION_RATIOS."""
    return any(abs(period / (ratio * rotation) - 1.0) <= ROTATION_TOLERANCE for ratio in ROTATION_RATIOS)


def rotation_summary(rows):
    """R12 on the variable curves: the crossings (best SDE strictly above the largest shuffle) and the curves that do not cross, each split into those whose best period is explained by rotation and the rest."""
    size = threshold.load("generator").time_axis()[0].size
    crossing = [row for row in rows if threshold.compare(row["sde"], row["shuffle_sde"]) == "above"]
    other = [row for row in rows if threshold.compare(row["sde"], row["shuffle_sde"]) != "above"]
    explained = lambda group: sum(explained_by_rotation(row["period"], rotation_period(row["seed"], size)) for row in group)
    crossing_explained, other_explained = explained(crossing), explained(other)
    return {"curves": len(rows), "crossings": len(crossing), "crossings_explained": share(crossing_explained, len(crossing)), "crossings_not_explained": len(crossing) - crossing_explained,
            "not_crossing": len(other), "not_crossing_explained": share(other_explained, len(other)), "ratios": list(ROTATION_RATIOS), "tolerance": ROTATION_TOLERANCE}


def real_summary(rows, current_ids=None):
    """R9 and R14: the rank shares of real-star rows with the Kolmogorov statistic; with the ids of the current list (stars.json), the same for the current list, for the other rows (the first list) and pooled over the distinct stars."""
    def one(group):
        if not group:
            return None
        statistic, p = kolmogorov([r["rank_share"] for r in group])
        return {"stars": len(group), "rank_shares": {f"{r['tic']}:{r['sector']}": r["rank_share"] for r in group}, "mean_rank_share": float(np.mean([r["rank_share"] for r in group])), "kolmogorov": statistic, "p": p}

    distinct = list({r["id"]: r for r in rows}.values())
    if current_ids is None:
        return one(distinct)
    return {"current_list": one([r for r in distinct if r["id"] in current_ids]), "first_list": one([r for r in distinct if r["id"] not in current_ids]), "pooled": one(distinct)}


def reflat_summary(null_rows, reflat_rows):
    """R13 on the curves that have both rows: the mean share of the copies below the original for the flatten-then-shuffle copies (old) and the shuffle-then-flatten copies (new), the paired difference with its standard error,
    the crossings of the new copies and the median ratio of the largest new copy to the largest old one."""
    old_by_index = {row["index"]: row for row in null_rows}
    pairs = [(old_by_index[row["index"]], row) for row in reflat_rows if row["index"] in old_by_index]
    old = np.array([np.mean(np.array(a["shuffle_sde"]) < a["sde"]) for a, _ in pairs])
    new = np.array([np.mean(np.array(b["shuffle_sde"]) < b["sde"]) for _, b in pairs])
    crossing = sum(threshold.compare(b["sde"], b["shuffle_sde"]) == "above" for _, b in pairs)
    difference = new - old
    return {"curves": len(pairs), "old_rank_share": float(old.mean()), "old_standard_error": float(old.std(ddof=1) / math.sqrt(len(pairs))), "new_rank_share": float(new.mean()),
            "new_standard_error": float(new.std(ddof=1) / math.sqrt(len(pairs))), "difference": float(difference.mean()), "difference_standard_error": float(difference.std(ddof=1) / math.sqrt(len(pairs))),
            "crossings": share(crossing, len(pairs)), "median_ratio_of_largest_copy": float(np.median([max(b["shuffle_sde"]) / max(a["shuffle_sde"]) for a, b in pairs]))}


def kolmogorov(values):
    """The Kolmogorov-Smirnov statistic of values against the uniform distribution on 0 to 1 and its asymptotic p-value (the Kolmogorov series); few values give little power."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    d = float(max(np.max(np.arange(1, n + 1) / n - ordered), np.max(ordered - np.arange(0, n) / n)))
    x = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * x * x) for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", type=Path, default=HERE / "out" / "threshold.jsonl")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args()
    rows = [json.loads(line) for line in options.rows.read_text().splitlines() if line]
    nulls = [row for row in rows if row["id"].startswith("null:")]
    injections = [row for row in rows if row["id"].startswith("inj:")]
    real = [row for row in rows if row["id"].startswith("real:")]
    kinds = sorted({row["class"] for row in nulls})
    result = {"null": {kind: null_summary([r for r in nulls if r["class"] == kind]) for kind in kinds},
              "inverted": {kind: inverted_summary([r for r in nulls if r["class"] == kind]) for kind in kinds},
              "false_alarm_quantiles_of_a_threshold_of_48": {str(q): threshold.false_alarm_quantile(q, 48) for q in (0.5, 0.9, 0.99)},
              "injection": injection_summary(injections, nulls)}
    if "variable" in kinds:
        result["variable_rotation"] = rotation_summary([r for r in nulls if r["class"] == "variable"])
    reflat = [row for row in rows if row["id"].startswith("reflat:")]
    if reflat:
        result["red_check"] = reflat_summary([r for r in nulls if r["class"] == "red"], reflat)
    if real:
        stars = HERE / "stars.json"
        current = {f"real:{s['tic']}:{s['sector']}" for s in json.loads(stars.read_text())} if stars.exists() else None
        result["real"] = real_summary(real, current)
    text = json.dumps(result, indent=1)
    if options.json:
        options.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
