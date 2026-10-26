"""The table of the split of the device time of exact mode by stage, from paper/scripts/profile_split.json (the output of profile_split.py).

    python paper/scripts/split_table.py [PROFILE.json] [--out paper/split_table.tex]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("fold", "scans", "screening", "correlations", "rest")


def split_table(profile):
    """A tabular with a row for each size: the device seconds, then each stage in seconds with its share of the total in percent."""
    lines = ["\\begin{tabular}{rrlllll}", "\\toprule", "points & GPU total & fold & scans & screening & correlations & other \\\\", "\\midrule"]
    for entry in profile:
        total = entry["device_seconds"]
        cells = " & ".join(f"{entry['stages'][stage]:.2f} ({100 * entry['stages'][stage] / total:.0f}\\%)" for stage in STAGES)
        lines.append(f"{entry['points']:,}".replace(",", "\\,") + f" & {total:.2f} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", type=Path, nargs="?", default=HERE / "profile_split.json")
    parser.add_argument("--out", type=Path, default=HERE.parent / "split_table.tex")
    arguments = parser.parse_args()
    arguments.out.write_text("% written by scripts/split_table.py\n" + split_table(json.loads(arguments.profile.read_text())))


if __name__ == "__main__":
    main()
