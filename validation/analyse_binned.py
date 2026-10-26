"""The binned mode run again (rerun_binned.py) against the rows of the evaluation (run.py): for each part, the binned column of the rerun joined by id to the exact and binned columns that run.py wrote, the matched
thresholds at the false-alarm rates of analyse.py from the nulls of each column, the completeness by S/N and period, and the difference of completeness binned minus exact with Tango's interval, for the rerun beside the
binned of the evaluation. usage: python analyse_binned.py RERUN_DIRECTORY EVALUATION_DIRECTORY
Writes RERUN_DIRECTORY/results-bands.json."""

import json
import sys
from pathlib import Path

import numpy as np

import analyse
import rerun_binned

NEW = "binned_bands"


def joined(rerun_records, evaluation_records):
    """The evaluation's records that the rerun covers, each with the rerun's binned search under NEW; a rerun id that the evaluation lacks is an error, since the comparison is paired."""
    by_id = {r["id"]: r for r in evaluation_records}
    out = []
    for new in rerun_records:
        if new["id"] not in by_id:
            raise ValueError(f"{new['id']} is in the rerun and not in the evaluation")
        out.append({**by_id[new["id"]], NEW: new["binned"]})
    return out


def part_of(records):
    injected = [r for r in records if r["id"].split(":")[2] == "inj"]
    null = [r for r in records if r["id"].split(":")[2] == "null"]
    out = {"injected": len(injected), "null": len(null), "searches": {}, "paired": {}}
    cuts = {rate: {} for rate in analyse.RATES}
    for search in ("exact", "binned", NEW):
        entry = {}
        for rate in analyse.RATES:
            cut, low, high = analyse.threshold([analyse.statistic(r, search) for r in null], rate)
            cuts[rate][search] = cut
            entry[str(rate)] = {"threshold": cut, "threshold_bootstrap": [low, high], "by_snr": analyse.completeness(injected, search, cut, lambda r: r["snr"]),
                                "by_period": analyse.completeness(injected, search, cut, lambda r: r["planet"]["period"])}
        entry["harmonic_share_by_period"] = analyse.harmonic_share(injected, search, lambda r: r["planet"]["period"])
        entry["seconds_median"] = float(np.median([r[search]["seconds"] for r in records]))
        out["searches"][search] = entry
    for rate in analyse.RATES:
        out["paired"][str(rate)] = {f"{NEW}_minus_exact": analyse.paired(injected, NEW, "exact", cuts[rate]), "binned_minus_exact": analyse.paired(injected, "binned", "exact", cuts[rate]),
                                    f"{NEW}_minus_binned": analyse.paired(injected, NEW, "binned", cuts[rate])}
    return out


def main(rerun_directory, evaluation_directory):
    rerun_directory, evaluation_directory = Path(rerun_directory), Path(evaluation_directory)
    out = {}
    for path in sorted(rerun_directory.glob(f"*{rerun_binned.SUFFIX}")):
        part = path.name[: -len(rerun_binned.SUFFIX)]
        rerun_records = [json.loads(line) for line in path.read_text().splitlines()]
        evaluation = [json.loads(line) for line in (evaluation_directory / f"{part}.jsonl").read_text().splitlines()]
        out[part] = part_of(joined(rerun_records, evaluation))
        out[part]["settings"] = {"backend": rerun_records[0]["backend"], "kwargs": rerun_records[0]["kwargs"]} if rerun_records else {}
    (rerun_directory / "results-bands.json").write_text(json.dumps(out, indent=1))
    for part, found in out.items():
        print(f"\n{part}: {found['injected']} injected, {found['null']} null; {found.get('settings')}")
        for rate in analyse.RATES:
            pairs = found["paired"][str(rate)]
            print(f"  at a false-alarm rate of {rate:g}: thresholds exact {found['searches']['exact'][str(rate)]['threshold']:.2f}, binned {found['searches']['binned'][str(rate)]['threshold']:.2f}, rerun {found['searches'][NEW][str(rate)]['threshold']:.2f}")
            for name, pair in pairs.items():
                print(f"    {name}: {pair['estimate']:+.3f} ({pair['low']:+.3f} to {pair['high']:+.3f}), only first {pair['only_first']}, only second {pair['only_second']}, n {pair['n']} at S/N 8 and over")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
