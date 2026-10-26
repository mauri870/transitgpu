"""The completeness curves, the matched thresholds and the comparisons of README.md from the output of run.py. usage: python analyse.py OUTPUT_DIRECTORY"""

import json
import sys
from pathlib import Path

import numpy as np

SEARCHES = ("exact", "binned", "box", "hybrid1", "hybrid5", "hybrid10", "hybrid20")
RATES = (0.05, 0.01)
BOOTSTRAPS = 1000


def wilson(k, n, z=1.96):
    """The Wilson score interval of k out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def _score(only_first, only_second, n, theta, z):
    """The score statistic of Tango (1998) for a difference theta of paired proportions, minus z squared: the first search finds only_first light curves that the second does not, the second finds only_second that the first does not, n in all."""
    both_or_neither = n - only_first - only_second
    low, high = max(0.0, -theta) + 1e-12, (1.0 - theta) / 2 - 1e-12  # the cell probabilities q and q + theta are positive
    if high <= low:
        return float("inf")

    def slope(q):
        return only_first / (q + theta) + only_second / q - 2 * both_or_neither / (1 - 2 * q - theta)

    if slope(low) <= 0:
        q = low
    elif slope(high) >= 0:
        q = high
    else:
        for _ in range(100):
            middle = (low + high) / 2
            low, high = (middle, high) if slope(middle) > 0 else (low, middle)
        q = (low + high) / 2
    variance = n * (2 * q + theta - theta**2)
    return (only_first - only_second - n * theta) ** 2 / variance - z * z if variance > 0 else float("inf")


def tango(only_first, only_second, n, z=1.96):
    """The estimate and the score interval of Tango (1998, Statistics in Medicine 17, 891, doi 10.1002/(SICI)1097-0258(19980430)17:8<891::AID-SIM780>3.0.CO;2-B) for the difference of the shares of light curves that two searches find, on the same light curves."""
    estimate = (only_first - only_second) / n
    low_edge, high_edge = -0.999, 0.999

    def edge(side):
        inner, outer = estimate, side
        if _score(only_first, only_second, n, outer, z) < 0:
            return outer
        for _ in range(80):
            middle = (inner + outer) / 2
            inner, outer = (middle, outer) if _score(only_first, only_second, n, middle, z) < 0 else (inner, middle)
        return (inner + outer) / 2

    return estimate, edge(low_edge), edge(high_edge)


def threshold(null_sde, rate, seed=0):
    """The SDE that a share `rate` of the null light curves exceeds (the 1 - rate quantile), with a bootstrap interval of the quantile: 2.5 and 97.5 percent of BOOTSTRAPS resamples of the nulls."""
    null_sde = np.asarray(null_sde, dtype=float)
    value = float(np.quantile(null_sde, 1 - rate))
    generator = np.random.default_rng(seed)
    resampled = np.quantile(generator.choice(null_sde, (BOOTSTRAPS, null_sde.size)), 1 - rate, axis=1)
    return value, float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


def found(record, search, cut):
    """Whether a search finds the planet of an injected record at the SDE `cut`."""
    if search.startswith("hybrid"):
        peaks = record["hybrid"]["peaks"][:int(search[6:])]
        return record["hybrid"]["sde"] >= cut and any(peak["label"] == "right" for peak in peaks)
    return record[search]["sde"] >= cut and record[search]["label"] == "right"


def statistic(record, search):
    return record["hybrid"]["sde"] if search.startswith("hybrid") else record[search]["sde"]


def completeness(records, search, cut, key):
    """{group value: {"found", "n", "wilson"}} of the injected records at the SDE `cut`, grouped by key(record)."""
    out = {}
    for value in sorted({key(r) for r in records}):
        group = [r for r in records if key(r) == value]
        k = sum(found(r, search, cut) for r in group)
        out[str(value)] = {"found": k, "n": len(group), "wilson": wilson(k, len(group))}
    return out


def harmonic_share(records, search, key):
    """{group value: {"harmonic", "n", "wilson"}}: the share of injected records whose best peak (the first of the hybrid's) is a harmonic of the planet's period, whatever the SDE."""
    out = {}
    for value in sorted({key(r) for r in records}):
        group = [r for r in records if key(r) == value]
        label = (lambda r: r["hybrid"]["peaks"][0]["label"]) if search.startswith("hybrid") else (lambda r: r[search]["label"])
        k = sum(label(r) == "harmonic" for r in group)
        out[str(value)] = {"harmonic": k, "n": len(group), "wilson": wilson(k, len(group))}
    return out


def paired(records, first, second, cuts, snr_low=8.0):
    """The difference of completeness of `first` minus `second` over the injected records with a nominal S/N of at least snr_low, each at its own cut, with the Tango interval: {"estimate", "low", "high", "only_first", "only_second", "n"}."""
    group = [r for r in records if r["snr"] >= snr_low]
    only_first = sum(found(r, first, cuts[first]) and not found(r, second, cuts[second]) for r in group)
    only_second = sum(found(r, second, cuts[second]) and not found(r, first, cuts[first]) for r in group)
    estimate, low, high = tango(only_first, only_second, len(group))
    return {"estimate": estimate, "low": low, "high": high, "only_first": only_first, "only_second": only_second, "n": len(group)}


def main(directory):
    directory = Path(directory)
    out = {}
    for path in sorted(directory.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        injected = [r for r in records if r["id"].split(":")[2] == "inj"]
        null = [r for r in records if r["id"].split(":")[2] == "null"]
        part = {"injected": len(injected), "null": len(null), "searches": {}, "paired": {}, "hybrid_seconds": {}, "reference_check": {}}
        cuts = {rate: {} for rate in RATES}
        for search in SEARCHES:
            entry = {}
            for rate in RATES:
                if not null:
                    continue
                cut, low, high = threshold([statistic(r, search) for r in null], rate)
                cuts[rate][search] = cut
                entry[str(rate)] = {"threshold": cut, "threshold_bootstrap": [low, high], "by_snr": completeness(injected, search, cut, lambda r: r["snr"]),
                                    "by_period": completeness(injected, search, cut, lambda r: r["planet"]["period"])}
            entry["harmonic_share_by_period"] = harmonic_share(injected, search, lambda r: r["planet"]["period"])
            entry["seconds_median"] = float(np.median([r["hybrid"]["seconds"] if search.startswith("hybrid") else r[search]["seconds"] for r in records])) if records else None
            part["searches"][search] = entry
        if null:
            for rate in RATES:
                part["paired"][str(rate)] = {f"{other}_minus_exact": paired(injected, other, "exact", cuts[rate]) for other in ("binned", "box", "hybrid1", "hybrid5", "hybrid10", "hybrid20")}
        timed = [r["hybrid_seconds"] for r in injected if "hybrid_seconds" in r]
        part["hybrid_seconds"] = {count: {"peaks": int(count), "median": float(np.median([t[count] for t in timed])), "quartiles": [float(x) for x in np.percentile([t[count] for t in timed], [25, 75])], "n": len(timed)} for count in ("1", "5", "10", "20")} if timed else {}
        checked = [r for r in injected if "cpu" in r]
        if checked:
            part["reference_check"] = {"n": len(checked), "same_period": sum(r["cpu"]["period"] == r["exact"]["period"] for r in checked), "largest_relative_sde_difference": max(abs(r["cpu"]["sde"] - r["exact"]["sde"]) / r["exact"]["sde"] for r in checked)}
        out[path.stem] = part
    (directory / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"injected": v["injected"], "null": v["null"]} for k, v in out.items()}))


if __name__ == "__main__":
    main(sys.argv[1])
