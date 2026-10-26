"""The table of the share of the injected planets found at each period, by noise class and search, with the Wilson interval, from the recovery files (paper/recovery.json, paper/recovery_real.json).

    python paper/scripts/period_table.py [RECOVERY.json] [--real RECOVERY_REAL.json] [--rate 0.05] [--out paper/period_table.tex]

A cell is the share of the injected light curves of that period at every signal to noise of the grid (4 to 20) that the search found at the false-alarm rate: the recovery file holds the shares by period over all signal to noise, not for signal to noise 8 and above.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = (("main-white", "white noise"), ("main-red", "red noise"), ("main-variable", "variable star"), ("real", "real stars"))
SEARCHES = (("exact", "exact"), ("binned", "binned"), ("box", "box"), ("hybrid20", "hybrid, 20 peaks"))


def period_table(results, rate="0.05"):
    """A tabular with a block for each part of `results` (white, red, variable, real) and a row for each of the four searches the part has; the columns are the periods of the file."""
    periods = sorted({period for content in results.values() for entry in content["searches"].values() for period in entry[rate]["by_period"]}, key=float)
    lines = ["\\begin{tabular}{l" + "r" * len(periods) + "}", "\\toprule", "search & " + " & ".join(f"{period}\\,d" for period in periods) + " \\\\"]
    for part, title in PARTS:
        if part not in results:
            continue
        lines += ["\\midrule", f"\\multicolumn{{{len(periods) + 1}}}{{l}}{{{title}}} \\\\"]
        for search, label in SEARCHES:
            if search not in results[part]["searches"]:
                continue
            cells = results[part]["searches"][search][rate]["by_period"]
            row = [f"{100 * cells[period]['found'] / cells[period]['n']:.1f} ({100 * cells[period]['wilson'][0]:.1f}--{100 * cells[period]['wilson'][1]:.1f})" if period in cells else "" for period in periods]
            lines.append(label + " & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, nargs="?", default=HERE.parent / "recovery.json")
    parser.add_argument("--real", type=Path, default=HERE.parent / "recovery_real.json")
    parser.add_argument("--rate", default="0.05")
    parser.add_argument("--out", type=Path, default=HERE.parent / "period_table.tex")
    arguments = parser.parse_args()
    results = json.loads(arguments.results.read_text())
    if arguments.real.exists():
        results = {**results, "real": json.loads(arguments.real.read_text())["real"]}
    arguments.out.write_text("% written by scripts/period_table.py\n" + period_table(results, arguments.rate))


if __name__ == "__main__":
    main()
