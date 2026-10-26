"""The ROC curves of the four searches, from paper/roc.json (validation/roc.py): a panel for each part, the share of the injected light curves found against the false-alarm rate of the nulls, with the rates of the macros marked.

    python paper/scripts/roc.py [ROC.json] [--out paper/roc.pdf]

Needs matplotlib and no device."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {"exact": "exact TLS", "binned": "binned TLS", "box": "box search", "hybrid5": "hybrid, 5 peaks"}
COLORS = {"exact": "C0", "binned": "C1", "box": "C2", "hybrid5": "C4"}  # the colours of the same searches in the completeness figure
TITLES = {"real": "12 TESS stars"}
MARKED = (0.01, 0.05)


STYLES = {"exact": "-", "binned": "-.", "box": "--", "hybrid5": ":"}  # the box and hybrid curves lie on each other in places: told apart by the line as well as the colour
FONT = 8


def roc_figure(roc, path):
    """One panel for each part of the file, two to a row at the width of the text: the share found of every search at the false-alarm rates of the grid, on a logarithmic axis of rates, the legend in the first panel.
    The vertical range of a panel is that of its curves. Saves the figure to `path` and returns it."""
    import matplotlib.pyplot as plt

    parts = list(roc["parts"])
    columns = 2
    rows = -(-len(parts) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.8, 3.0 * rows), squeeze=False)
    flat = [axis for line in axes for axis in line]
    for index, (axis, name) in enumerate(zip(flat, parts)):
        part = roc["parts"][name]
        low = 1.0
        for search, entry in part["searches"].items():
            axis.plot(roc["fpr"], entry["tpr"], label=LABELS.get(search, search), color=COLORS.get(search), linestyle=STYLES.get(search, "-"), linewidth=1.5)
            low = min(low, min(entry["tpr"]))
        for rate in MARKED:
            axis.axvline(rate, color="0.6", linestyle=":", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(max(0.0, low - 0.04), 1.01)
        axis.set_title(TITLES.get(name, name.replace("main-", "")), fontsize=FONT + 1)
        axis.tick_params(labelsize=FONT)
        if index >= len(parts) - columns:
            axis.set_xlabel("false-alarm rate", fontsize=FONT)
        if index % columns == 0:
            axis.set_ylabel(f"share found, S/N {roc.get('snr_low', 8):g} and above", fontsize=FONT)
    for axis in flat[len(parts):]:
        axis.axis("off")
    flat[0].legend(fontsize=FONT, loc="lower right")
    figure.tight_layout()
    figure.savefig(path)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roc", type=Path, nargs="?", default=HERE.parent / "roc.json")
    parser.add_argument("--out", type=Path, default=HERE.parent / "roc.pdf")
    arguments = parser.parse_args()
    roc_figure(json.loads(arguments.roc.read_text()), arguments.out)


if __name__ == "__main__":
    main()
