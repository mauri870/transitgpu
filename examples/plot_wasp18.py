"""Plot WASP-18 b as the box search and TLS on the GPU find it: one row per search, each with the light curve, the periodogram and the folded transit.

The data are 13 days of TESS 2-minute photometry from sector 2, flattened (see examples/data/README.md). Needs matplotlib and a GPU:

    uv run --group rocm --extra examples python examples/plot_wasp18.py [docs/wasp18.png]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import transitgpu
from transitgpu.tls import transitleastsquares

DATA = Path(__file__).resolve().parent / "data" / "wasp18_s2.npz"
PERIODS = (0.5, 5.0)
WINDOW_HOURS = 6.0


def fold_panel(axis, results, label):
    phase_hours = (results.folded_phase - 0.5) * results.period * 24
    bins = np.linspace(-WINDOW_HOURS, WINDOW_HOURS, 49)
    middle = 0.5 * (bins[1:] + bins[:-1])
    index = np.digitize(phase_hours, bins) - 1
    binned = np.array([results.folded_y[index == k].mean() if np.any(index == k) else np.nan for k in range(len(middle))])
    axis.plot(phase_hours, results.folded_y, ".", ms=1.5, color="0.7", label="light curve")
    axis.plot(middle, binned, "o", ms=3, color="C0", label="binned")
    axis.plot((results.model_folded_phase - 0.5) * results.period * 24, results.model_folded_model, color="C3", lw=1.5, label=label)
    axis.set(xlim=(-WINDOW_HOURS, WINDOW_HOURS), xlabel="hours from mid-transit", ylabel="relative flux", title="folded at the best period")
    axis.legend(loc="lower left", fontsize=9, frameon=False)


def draw(output):
    """Run both searches on the WASP-18 light curve, save the figure to `output` and return both results."""
    data = np.load(DATA)
    t, y, dy = data["time"], data["flux"], data["flux_err"]
    box = transitgpu.box_search(t, y, dy).power(period_min=PERIODS[0], period_max=PERIODS[1])
    template = transitleastsquares(t, y, dy).power(period_min=PERIODS[0], period_max=PERIODS[1], show_progress_bar=False, verbose=False)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column in (0, 1):
        axes[1, column].sharex(axes[0, column])
    axes[1, 2].sharex(axes[0, 2])
    for row, (results, name, model) in enumerate(((box, "box search", "box"), (template, "TLS", "template"))):
        curve, spectrum, fold = axes[row]
        curve.plot(t, y, ".", ms=2, color="0.35")
        curve.set(ylabel="relative flux", title="WASP-18, TESS sector 2")
        spectrum.plot(results.periods, results.power, lw=0.8, color="0.25")
        spectrum.axvline(results.period, color="C3", lw=1, alpha=0.7)
        spectrum.set(ylabel="SDE", title=f"best period {results.period:.4f} d, SDE {results.SDE:.0f}")
        fold_panel(fold, results, f"{model}: {(1 - results.depth) * 1e6:.0f} ppm, {results.duration * 24:.1f} h")
        spectrum.text(0.98, 0.95, f"period {results.period:.5f} d\ndepth {(1 - results.depth) * 1e6:.0f} ppm\nduration {results.duration * 24:.2f} h", transform=spectrum.transAxes, ha="right", va="top", fontsize=9)
        curve.annotate(name, xy=(0, 0.5), xytext=(-0.28, 0.5), xycoords="axes fraction", textcoords="axes fraction", rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)
    axes[0, 2].set(xlabel=None)
    axes[1, 0].set(xlabel="time (BTJD, days)")
    axes[1, 1].set(xlabel="trial period (days)")
    figure.savefig(output, dpi=100)
    plt.close(figure)
    return box, template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", nargs="?", default=str(Path(__file__).resolve().parent.parent / "docs" / "wasp18.png"))
    arguments = parser.parse_args()
    box, template = draw(arguments.output)
    for name, found in (("box search", box), ("TLS", template)):
        print(f"{name}: period {found.period:.5f} d, depth {(1 - found.depth) * 1e6:.0f} ppm, duration {found.duration * 24:.2f} h")
    print(f"-> {arguments.output}")
