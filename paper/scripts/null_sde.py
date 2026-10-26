"""The distribution of the noise-only SDE of the four searches, from paper/null_sde.json (validation/null_sde.py): a panel for each part, the histogram of each search's SDE over the part's null light curves
with its 5 percent threshold marked, and the distribution of the box search's 48-shuffle threshold of the noise-only light curves of the calibration (for the real stars, one mark for each star).

    python paper/scripts/null_sde.py [NULL_SDE.json] [--out paper/null_sde.pdf]

Needs matplotlib and no device."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"exact": "exact TLS", "binned": "binned TLS", "box": "box search and hybrid"}  # the hybrid's SDE is the box search's, so one histogram shows both
COLORS = {"exact": "C0", "binned": "C1", "box": "C2"}
STYLES = {"exact": "-", "binned": "-.", "box": "--"}
SHOWN = ("exact", "binned", "box")
TITLES = {"real": "12 TESS stars"}
FONT = 8


def density(entry):
    """The histogram as a density (the counts over their total and the width of the bin), with its edges."""
    edges, counts = np.asarray(entry["edges"]), np.asarray(entry["counts"], dtype=float)
    return edges, counts / (counts.sum() * np.diff(edges))


def null_sde_figure(data, path):
    """A panel for each part, two to a row at the width of the text: the histogram of each search's SDE over the part's null light curves with its 5 percent threshold as a vertical line of the same colour, and the
    distribution of the box search's 48-shuffle threshold of the noise-only light curves of the calibration (one mark for each star for the real stars), the legend in the first panel. Saves the figure to `path`."""
    import matplotlib.pyplot as plt

    parts = list(data["parts"])
    columns = 2
    rows = -(-len(parts) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.8, 3.0 * rows), squeeze=False)
    flat = [axis for line in axes for axis in line]
    for index, (axis, name) in enumerate(zip(flat, parts)):
        part = data["parts"][name]
        for search in [s for s in SHOWN if s in part["searches"]]:
            entry = part["searches"][search]
            edges, heights = density(entry)
            axis.stairs(heights, edges, color=COLORS[search], linestyle=STYLES[search], linewidth=1.3, label=LABELS[search])
            axis.axvline(entry["threshold_5"], color=COLORS[search], linestyle=STYLES[search], linewidth=0.8, alpha=0.8)
        shuffle = part.get("shuffle_threshold")
        if shuffle:
            if shuffle.get("values") is not None:
                axis.plot(shuffle["values"], [0.0] * len(shuffle["values"]), "|", color="C3", markersize=9, markeredgewidth=1.2, clip_on=False, label="48-shuffle threshold of each star")
            else:
                shuffle_edges, shuffle_heights = density(shuffle)
                axis.stairs(shuffle_heights, shuffle_edges, color="C3", fill=True, alpha=0.3, linewidth=0.0, label="48-shuffle threshold (box search)")
        axis.set_title(TITLES.get(name, name.replace("main-", "")), fontsize=FONT + 1)
        axis.set_xlabel("SDE", fontsize=FONT)
        axis.tick_params(labelsize=FONT)
        axis.set_yticks([])
        if index % columns == 0:
            axis.set_ylabel("share of light curves, per unit of SDE", fontsize=FONT)
    for axis in flat[len(parts):]:
        axis.axis("off")
    flat[0].legend(fontsize=FONT - 1, loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, nargs="?", default=HERE.parent / "null_sde.json")
    parser.add_argument("--out", type=Path, default=HERE.parent / "null_sde.pdf")
    arguments = parser.parse_args()
    null_sde_figure(json.loads(arguments.data.read_text()), arguments.out)


if __name__ == "__main__":
    main()
