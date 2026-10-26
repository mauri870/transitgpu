"""LaTeX tables of the shuffle calibration, from the analysis file that validation/threshold_analysis.py writes: for each noise class the share of curves whose best peak exceeds the threshold of their 48 shuffles against the nominal 1/49 and the same for the
inverted-light-curve null, and for each class and signal to noise the recovery of injected planets above the shuffle and the inverted threshold.

    python paper/scripts/threshold_table.py ANALYSIS.json [--out paper/threshold_table.tex]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ("white", "red", "variable")  # the classes in the order of the paper, then any other


def in_order(block):
    """The items of a dict of classes, white, red and variable first."""
    return sorted(block.items(), key=lambda item: (ORDER.index(item[0]) if item[0] in ORDER else len(ORDER), item[0]))


def percent(entry):
    """A share as a percentage with the 95% interval, `2.0\\% (1.5 to 2.8)`."""
    return f"{100 * entry['share']:.1f}\\% ({100 * entry['low']:.1f} to {100 * entry['high']:.1f})"


def thousands(number):
    return f"{number:,}".replace(",", "\\,")


def tabular(columns, header, rows):
    return "\n".join(["\\begin{tabular}{" + columns + "}", "\\toprule", header + " \\\\", "\\midrule", *[row + " \\\\" for row in rows], "\\bottomrule", "\\end{tabular}"]) + "\n"


def null_table(analysis):
    """One row for each class of curves with nothing injected: the curves, those above their threshold, the share with its interval, the nominal rate, and the share above the threshold of the inverted curves."""
    rows = []
    for name, entry in in_order(analysis.get("null", {})):
        inverted = analysis.get("inverted", {}).get(name)
        rows.append(f"{name} & {thousands(entry['of'])} & {thousands(entry['count'])} & {percent(entry)} & {100 * entry['nominal']:.1f}\\% & " + (percent(inverted) if inverted else ""))
    return tabular("lrrrrl", "class & curves & above & shuffle threshold & nominal & inverted threshold", rows)


def injection_table(analysis):
    """One row for each class and signal to noise: the injected planets, those found at the injected period and at a harmonic, and the share found above the shuffle threshold and above the inverted threshold, with intervals."""
    rows = []
    for name, by_snr in in_order(analysis.get("injection", {})):
        for snr, entry in sorted(by_snr.items(), key=lambda item: float(item[0]) if item[0] != 'by_period' else 0):
            if snr == "by_period":
                continue
            rows.append(f"{name} & {snr} & {entry['cases']} & {entry['right']} & {entry['harmonic']} & {percent(entry['found_shuffle_share'])} & {percent(entry['found_inverted_share'])}")
    return tabular("lrrrrll", "class & S/N & planets & period & harmonic & above shuffle threshold & above inverted threshold", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, default=HERE.parent / "threshold_table.tex")
    arguments = parser.parse_args()
    analysis = json.loads(arguments.analysis.read_text())
    arguments.out.write_text("% written by scripts/threshold_table.py\n" + null_table(analysis))
    arguments.out.with_name("injection_table.tex").write_text("% written by scripts/threshold_table.py\n" + injection_table(analysis))


if __name__ == "__main__":
    main()
