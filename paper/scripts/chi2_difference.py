"""The chi-square of exact mode against TLS fused over the trial periods of one characterization case, and the largest difference of the 14 cases, from paper/chi2_difference.json (validation/chi2_difference.py): the figure
(the relative difference at every trial period, with a line at zero) and the table of the cases as paper/chi2_table.tex.

    python paper/scripts/chi2_difference.py [CHI2_DIFFERENCE.json] [--out paper/chi2_difference.pdf] [--table paper/chi2_table.tex]

Needs matplotlib and no device."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT = 8


def cell(case):
    """The row of a case in the table: its trial periods and its largest relative difference, or a note that its grids differ."""
    if not case["periods_equal"]:
        return f"{case['periods']} & grids differ"
    value = case["maximum_relative_difference"]
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{case['periods']} & ${mantissa}\\times10^{{{int(exponent)}}}$" if value else f"{case['periods']} & $0$"


def table(data):
    """The tabular of the cases, a row for each in the order of the file: the name (underscores escaped), the number of trial periods and the largest magnitude of the relative difference of the chi-square."""
    rows = "".join(f"{name.replace('_', chr(92) + '_')} & {cell(case)} \\\\\n" for name, case in data["cases"].items())
    return "\\begin{tabular}{lrr}\n\\toprule\ncase & trial periods & largest difference \\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n"


def figure(data, path):
    """The relative difference at every trial period of the figure case, as a line with a line at zero."""
    import matplotlib.pyplot as plt

    shown = data["figure"]
    periods = [p for p, v in zip(shown["periods_days"], shown["relative_difference"]) if v is not None]
    values = [v for v in shown["relative_difference"] if v is not None]
    fig, axis = plt.subplots(figsize=(6.0, 2.6))
    axis.plot(periods, values, ".", markersize=2.0, color="C0", label="exact mode against TLS fused")
    axis.plot([min(periods), max(periods)], [0.0, 0.0], color="0.3", linewidth=0.8)
    axis.set_xlabel("trial period (d)", fontsize=FONT)
    axis.set_ylabel("relative difference of $\\chi^2$", fontsize=FONT)
    axis.set_title(shown["case"].replace("_", " "), fontsize=FONT + 1)
    axis.tick_params(labelsize=FONT)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(path)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, nargs="?", default=HERE.parent / "chi2_difference.json")
    parser.add_argument("--out", type=Path, default=HERE.parent / "chi2_difference.pdf")
    parser.add_argument("--table", type=Path, default=HERE.parent / "chi2_table.tex")
    arguments = parser.parse_args()
    data = json.loads(arguments.data.read_text())
    figure(data, arguments.out)
    arguments.table.write_text(table(data))


if __name__ == "__main__":
    main()
