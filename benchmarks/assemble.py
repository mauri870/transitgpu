"""Join the --json files of the parts of one timing session into the results.json of the paper.

    uv run python benchmarks/assemble.py SESSION_DIRECTORY [--out benchmarks/results.json]

SESSION_DIRECTORY holds the files that session.sh writes: main.json, slow20.json, slow.json, many.json, many-cpu.json (the rows of the table of the paper), shuffles.json, gtls72.json, gtls144.json, gtls-input-1500.json, gtls-input-3000.json and gtls-input-1500-margin.json and gtls-input-3000-margin.json (blocks of their own), cold-ROW.json and cold-cleared-ROW.json
(one call each in a fresh process) session_note.txt (the note of the session that results.json keeps) and session.jsonl (the load and the free memory at the start and end of each part). Every row gets `calls`, the number of timed calls.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_PARTS = ("main", "slow20", "slow", "many", "many-cpu")
COLD = (
    "A cold call is the first call of a fresh process, measured as the warm-up call of the row (build_seconds of a row that is the first of its process). With the compiled caches kept, the process finds the Triton kernels and the torch extension of the HIP kernel on disk; "
    "with the caches cleared, the call compiles them, including the hipcc build of the HIP kernel."
)


def bench_module():
    spec = importlib.util.spec_from_file_location("transitgpu_bench", HERE / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_calls(report):
    """The report with the number of timed calls in each row that has them."""
    report = json.loads(json.dumps(report))
    for entry in report["sizes"].values():
        for row in entry["rows"].values():
            if "seconds" in row:
                row["calls"] = len(row["seconds"])
    return report


def cold_calls(parts, prefix):
    """{row: seconds of the build of the row} of the parts named prefix-ROW, for the rows that were the first of their process."""
    found = {}
    for name, report in parts.items():
        if name.startswith(prefix + "-") and not (prefix == "cold" and name.startswith("cold-cleared-")):
            for entry in report["sizes"].values():
                for row, measured in entry["rows"].items():
                    if measured.get("first_row_of_process"):
                        found[row] = measured["build_seconds"]
    return found


def assemble(parts, session, note=""):
    """The results of a session from its parts (a dict of the reports read from the --json files, by file name without the extension), its log and its note."""
    bench = bench_module()
    rest = sorted(name for name in parts if name.startswith("slow-rest"))  # rows of the table that a stopped part did not reach, run again by hand
    table = [with_calls(parts[name]) for name in TABLE_PARTS + tuple(rest) if name in parts]
    results = bench.merged(table)
    if "shuffles" in parts:
        results["block_shuffles"] = {"note": "48 block-shuffled copies of a light curve searched together (the null of a star); the answer field of each row is the largest SDE of the shuffles", "synthetic": with_calls(parts["shuffles"])}
    runs = [with_calls(parts[name]) for name in ("gtls72", "gtls144") if name in parts]
    if runs:
        results["gtls_benchmark_sizes"] = {"note": "the box search on light curves of the sizes of the GTLS paper's benchmark (30-minute cadence, TLS's grid to half the baseline), in the session of the rows above", "runs": runs}
    inputs = [with_calls(parts[name]) for name in ("gtls-input-1500", "gtls-input-3000", "gtls-input-1500-margin", "gtls-input-3000-margin") if name in parts]
    if inputs:
        results["gtls_input"] = {"note": "the input of the GTLS paper's timing (30-minute cadence, TLS's grid from 0 to half the baseline): the runs of 1,500 and 3,000 days at TLS's T0 fit margin and of 1,500 days at the margin of GTLS (0.125), in the session of the rows above", "runs": inputs}
    results["cold_first_call"] = {"definition": COLD, "caches_kept": cold_calls(parts, "cold"), "caches_cleared": cold_calls(parts, "cold-cleared")}
    results["session"] = session
    if note:
        results["session_note"] = note
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    arguments = parser.parse_args()
    parts = {path.stem: json.loads(path.read_text()) for path in sorted(arguments.directory.glob("*.json"))}
    log = arguments.directory / "session.jsonl"
    session = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    note_file = arguments.directory / "session_note.txt"
    note = note_file.read_text().strip() if note_file.exists() else ""
    arguments.out.write_text(json.dumps(assemble(parts, session, note), indent=1))


if __name__ == "__main__":
    main()
