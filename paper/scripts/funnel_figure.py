"""The screening of exact mode in two panels, from paper/scripts/screening_stats.json and paper/scripts/profile_split.json: the share of the trial epochs tested that survives each bound of the cascade, for each period range at each size, and the device time of the stages of exact mode at each size as stacked bars.

    python paper/scripts/funnel_figure.py [SCREENING_STATS.json PROFILE_SPLIT.json] [--out paper/funnel.pdf]

Needs matplotlib and no device."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOUNDS = (("live", "live"), ("cs", "U1"), ("cheap", "U2"), ("tight", "U3"))
STAGES = ("fold", "scans", "screening", "correlations", "rest")
FONT = 8


def survival(entry):
    """The shares of the trial epochs tested, over all the template lengths of the entry, that survive each of BOUNDS, in percent."""
    tested = sum(stage["tested"] for stage in entry["stages"].values())
    return [100.0 * sum(stage[name] for stage in entry["stages"].values()) / tested for name, _ in BOUNDS]


def funnel_figure(stats, profile, path):
    """Left: the survival of each (size, period range) over the bounds, a color for each range and a line style for each size. Right: the device time of the stages of exact mode at each size, stacked. Saves the figure to `path` and returns it."""
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ranges = sorted({tuple(entry["range"]) for entry in stats})
    sizes = sorted({entry["points"] for entry in stats})
    styles = {size: ("-", "--", ":")[index % 3] for index, size in enumerate(sizes)}
    colors = {span: f"C{index}" for index, span in enumerate(ranges)}
    positions = range(len(BOUNDS))
    for entry in stats:
        span = tuple(entry["range"])
        left.plot(positions, survival(entry), linestyle=styles[entry["points"]], color=colors[span], marker="o", markersize=3, linewidth=1.1)
    left.set_yscale("log")
    left.set_xticks(list(positions))
    left.set_xticklabels([label for _, label in BOUNDS], fontsize=FONT - 1)
    left.set_ylabel("trial epochs surviving the bound, % of those tested", fontsize=FONT)
    left.tick_params(labelsize=FONT)
    left.grid(True, which="major", linewidth=0.3, alpha=0.5)
    handles = [left.plot([], [], color=colors[span], linewidth=1.5, label=f"{span[0]:g} to {span[1]:g} d")[0] for span in ranges]
    handles += [left.plot([], [], color="0.3", linestyle=styles[size], label=f"{size:,} points".replace(",", " "))[0] for size in sizes]
    left.legend(handles=handles, fontsize=FONT - 1, loc="lower left", frameon=False, ncol=2)
    bottoms = [0.0] * len(profile)
    labels = [f"{entry['points']:,}".replace(",", " ") for entry in profile]
    for index, stage in enumerate(STAGES):
        heights = [entry["stages"].get(stage, 0.0) for entry in profile]
        right.bar(labels, heights, bottom=bottoms, color=f"C{index + 4}", label=stage, width=0.6)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    right.set_xlabel("points", fontsize=FONT)
    right.set_ylabel("device time of exact mode, s", fontsize=FONT)
    right.tick_params(labelsize=FONT)
    right.legend(fontsize=FONT - 1, loc="upper left", frameon=False)
    figure.tight_layout()
    figure.savefig(path, metadata={"CreationDate": None, "Producer": None})
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stats", type=Path, nargs="?", default=HERE / "screening_stats.json")
    parser.add_argument("profile", type=Path, nargs="?", default=HERE / "profile_split.json")
    parser.add_argument("--out", type=Path, default=HERE.parent / "funnel.pdf")
    arguments = parser.parse_args()
    funnel_figure(json.loads(arguments.stats.read_text()), json.loads(arguments.profile.read_text()), arguments.out)


if __name__ == "__main__":
    main()
