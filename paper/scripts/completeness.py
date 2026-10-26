"""The figures and the table of the completeness of the searches, from the results.json that validation/analyse.py writes (keyed by part, then `searches`, search, false-alarm rate, `by_snr`; and `hybrid_seconds`).

    python paper/scripts/completeness.py RESULTS.json [--real RECOVERY_REAL.json] [--rate 0.05] [--out-directory paper] [--tables-only]

Writes (without --tables-only) completeness.pdf (a panel for each noise class of the main case and, with --real, a panel for the real stars, every search on it, with Wilson bands) and recall_table_PART.tex for each part (the threshold and the completeness at signal to noise 6, 8 and 10, for each search, at the matched false-alarm rate).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEARCHES = ("exact", "binned", "box", "hybrid1", "hybrid5", "hybrid10", "hybrid20")
LABELS = {"exact": "exact TLS", "binned": "binned TLS", "box": "box search", "hybrid1": "hybrid, 1 peak", "hybrid5": "hybrid, 5 peaks", "hybrid10": "hybrid, 10 peaks", "hybrid20": "hybrid, 20 peaks"}
PEAKS = (1, 5, 10, 20)
TITLES = {"real": "12 TESS stars"}  # the panel of the part with the injections into the light curves of real stars; the other parts are named by their noise class


def merge_real(results, real):
    """Return the results with the part `real` of the file of the real stars (a results file that has that one part) added."""
    return {**results, "real": real["real"]}


def wilson(found, total, z=1.96):
    """The Wilson score interval of `found` out of `total`."""
    if total == 0:
        return float("nan"), float("nan")
    p = found / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(centre - half), float(centre + half)


def levels(part, rate):
    """The signal to noise levels of the first search of a part that has them, in increasing order."""
    for search in part["searches"].values():
        if rate in search and search[rate]["by_snr"]:
            return sorted(search[rate]["by_snr"], key=float)
    return []


def recall_table(results, part_name, rate, snr=None):
    """The table of a part at a false-alarm rate: for each search its threshold (SDE, with the bootstrap interval) and its completeness at each signal to noise as a percentage with the Wilson interval."""
    part = results[part_name]
    shown = [level for level in levels(part, rate) if snr is None or float(level) in snr]
    lines = ["\\begin{tabular}{l" + "r" * (1 + len(shown)) + "}", "\\toprule", "search & threshold" + "".join(f" & S/N {float(level):g}" for level in shown) + " \\\\", "\\midrule"]
    for search in SEARCHES:
        if search not in part["searches"] or rate not in part["searches"][search]:
            continue
        row = part["searches"][search][rate]
        low, high = row["threshold_bootstrap"]
        cells = []
        for level in shown:
            cell = row["by_snr"][level]
            cells.append(f"{100 * cell['found'] / cell['n']:.0f}\\% ({100 * cell['wilson'][0]:.0f} to {100 * cell['wilson'][1]:.0f})")
        lines.append(f"{search} & {row['threshold']:.1f} ({low:.1f} to {high:.1f})" + "".join(f" & {cell}" for cell in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


FONT = 8
STYLES = {"exact": "-", "binned": "-.", "box": "--", "hybrid1": "-", "hybrid5": "--", "hybrid10": "-.", "hybrid20": ":"}  # the hybrids are told apart by the line as well as the colour


def completeness_figure(results, parts, rate, path):
    """One panel for each part, two to a row at the width of the text: the completeness of every search against the signal to noise, each line with its Wilson band, the legend in the first panel. Saves the figure to
    `path` and returns it."""
    import matplotlib.pyplot as plt

    columns = 2
    rows = -(-len(parts) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.8, 3.0 * rows), sharey=True, squeeze=False)
    flat = [axis for line in axes for axis in line]
    for index, (axis, name) in enumerate(zip(flat, parts)):
        part = results[name]
        shown = levels(part, rate)
        x = [float(level) for level in shown]
        for search in SEARCHES:
            if search not in part["searches"] or rate not in part["searches"][search]:
                continue
            cells = [part["searches"][search][rate]["by_snr"][level] for level in shown]
            axis.plot(x, [c["found"] / c["n"] for c in cells], marker="o", markersize=3, linestyle=STYLES.get(search, "-"), linewidth=1.2, label=LABELS[search])
            axis.fill_between(x, [c["wilson"][0] for c in cells], [c["wilson"][1] for c in cells], alpha=0.15, linewidth=0)
        axis.set_title(TITLES.get(name, name.replace("main-", "")), fontsize=FONT + 1)
        axis.tick_params(labelsize=FONT)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("target S/N", fontsize=FONT)
        if index % columns == 0:
            axis.set_ylabel(f"completeness at a false-alarm rate of {float(rate):g}", fontsize=FONT)
    for axis in flat[len(parts):]:
        axis.axis("off")
    flat[0].legend(fontsize=FONT - 1, loc="lower right")
    figure.tight_layout()
    figure.savefig(path)
    return figure


def hybrid_figure(results, part_name, rate, path, minimum_snr=8):
    """Two panels: the recall of the hybrid (the share of the injected light curves of signal to noise `minimum_snr` and above that it finds, with its Wilson band) and its median run time (with the quartiles) against the number of peaks."""
    import matplotlib.pyplot as plt

    part = results[part_name]
    found = total = None
    recall, low, high = [], [], []
    for count in PEAKS:
        cells = [cell for level, cell in part["searches"][f"hybrid{count}"][rate]["by_snr"].items() if float(level) >= minimum_snr]
        found, total = sum(cell["found"] for cell in cells), sum(cell["n"] for cell in cells)
        interval = wilson(found, total)
        recall.append(found / total)
        low.append(interval[0])
        high.append(interval[1])
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    left.plot(PEAKS, recall, marker="o")
    left.fill_between(PEAKS, low, high, alpha=0.2, linewidth=0)
    left.set_xlabel("peaks refined")
    left.set_ylabel(f"recall, S/N {minimum_snr} and above")
    left.set_ylim(0, 1.02)
    seconds = part["hybrid_seconds"]
    quartiles = [(entry["quartiles"] if "quartiles" in entry else (entry["q1"], entry["q3"])) for entry in (seconds[str(count)] for count in PEAKS)]
    medians = [seconds[str(count)]["median"] for count in PEAKS]
    right.plot(PEAKS, medians, marker="o")
    right.fill_between(PEAKS, [q[0] for q in quartiles], [q[1] for q in quartiles], alpha=0.2, linewidth=0)
    right.set_xlabel("peaks refined")
    right.set_ylabel("seconds per light curve")
    figure.tight_layout()
    figure.savefig(path)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--real", type=Path, help="the results file of the injections into real stars, with the part `real`")
    parser.add_argument("--rate", default="0.05")
    parser.add_argument("--out-directory", type=Path, default=HERE.parent)
    parser.add_argument("--tables-only", action="store_true", help="write the recall tables and not the figure (the figure needs matplotlib)")
    arguments = parser.parse_args()
    results = json.loads(arguments.results.read_text())
    if arguments.real:
        results = merge_real(results, json.loads(arguments.real.read_text()))
    parts = [name for name in ("main-white", "main-red", "main-variable", "real") if name in results]
    if not arguments.tables_only:
        completeness_figure(results, parts, arguments.rate, arguments.out_directory / "completeness.pdf")
    for name in parts:
        (arguments.out_directory / f"recall_table_{name}.tex").write_text("% written by scripts/completeness.py\n" + recall_table(results, name, arguments.rate, snr=(6, 8, 10)))


if __name__ == "__main__":
    main()
