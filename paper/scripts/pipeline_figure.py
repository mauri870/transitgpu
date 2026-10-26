"""The architecture of the searches: the host above a boundary and the device below it, the three searches as rows of stages on the device with what runs each stage, the results crossing back to the host, and the loop of the hybrid.

    python paper/scripts/pipeline_figure.py [--out paper/pipeline.pdf]

Needs matplotlib and no device. Nothing in the drawing is measured; it states the structure of the code (the backends are those of src/transitgpu/core.py, hip.py, tls/*.py and hybrid.py)."""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH = 4.3  # inches: the width at which the figure is placed, 0.62 of the width of the text of the paper (6.8 in), so that the text is not scaled
FONT = 7
DEVICE, TLS, HOST, DATA = "device", "tls", "host", "data"
FILLS = {DEVICE: "#cfe3f5", TLS: "#e3e3e3", HOST: "#f7ecc6", DATA: "white"}
# the rows of stages on the device: (title, [(stage, what runs it)]); the stage names are those of Sect. 2
PATHS = (
    ("box search", [("fold\nto bins", "torch.compile"), ("integer\naccumulation", "HIP / Triton"), ("window\nscores", "torch.compile"), ("best\nwindow", "torch")]),
    ("exact mode", [("counting\nsort", "Triton"), ("prefix\nsums", "Triton"), ("cascade\nU1 U2 U3", "Triton"), ("float64\ncorrelation", "Triton")]),
    ("binned mode", [("bins\nper band", "HIP / Triton"), ("Fourier\ncorrelation", "torch FFT"), ("best\nstart", "Triton")]),
)
HOST_BOXES = {  # name: (x, row, width, text, kind)
    "light curve": (0.3, 0, 0.8, "light curve", DATA),
    "grids": (1.25, 0, 1.45, "TLS: period and duration\ngrids, templates", TLS),
    "results": (2.9, 0, 1.3, "TLS: SDE, periodogram,\nfinal fit", TLS),
    "batches": (0.3, 1, 1.15, "TLS: batches of\ntrial periods", TLS),
    "peaks": (2.85, 1, 1.3, "peaks of the box\nspectrum (NumPy)", HOST),
}


def pipeline_figure(path):
    """The diagram, 4.3 in wide and every text at 7 points or more. Saves the figure to `path` and returns it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    box_width, box_height, gap = 0.76, 0.6, 0.14
    left = 0.62
    host_height, row_gap, title_height = 0.42, 0.26, 0.2
    top_margin = 0.05
    host_rows = (top_margin, top_margin + host_height + row_gap)
    boundary = host_rows[1] + host_height + 0.35
    device_top = boundary + 0.3
    row_pitch = title_height + box_height + 0.26
    height = device_top + len(PATHS) * row_pitch + 0.36
    figure = plt.figure(figsize=(WIDTH, height))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, WIDTH)
    axis.set_ylim(height, 0)  # y runs downward, from the top of the figure
    axis.axis("off")

    def box(x, y, w, h, text, kind, runs=None):
        """A box with `text` and, when `runs` is given, what runs it in italic under the text."""
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", facecolor=FILLS[kind], edgecolor="0.3", linewidth=0.8))
        axis.text(x + w / 2, y + h / 2 - (0.08 if runs else 0.0), text, ha="center", va="center", fontsize=FONT, linespacing=1.05)
        if runs:
            axis.text(x + w / 2, y + h - 0.05, runs, ha="center", va="bottom", fontsize=FONT, style="italic", color="0.2")

    def arrow(points, color="0.25", style="-", head=True):
        """A line through `points`, with an arrowhead on its last segment when `head`."""
        xs, ys = zip(*points)
        axis.plot(xs, ys, color=color, linestyle=style, linewidth=1.0, solid_capstyle="butt")
        if head:
            axis.annotate("", xy=points[-1], xytext=points[-2], arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))

    placed = {}
    for name, (x, row, w, text, kind) in HOST_BOXES.items():
        y = host_rows[row]
        box(x, y, w, host_height, text, kind)
        placed[name] = (x, y, w, host_height)
    axis.plot([0, WIDTH], [boundary, boundary], color="0.15", linestyle=(0, (5, 3)), linewidth=1.0)
    axis.text(0.03, boundary - 0.07, "host", fontsize=FONT, fontweight="bold", va="bottom")
    axis.text(0.03, boundary + 0.07, "device", fontsize=FONT, fontweight="bold", va="top")
    # the host: the light curve to the grids, the grids to the batches and the results back
    arrow([(1.1, host_rows[0] + host_height / 2), (1.25, host_rows[0] + host_height / 2)])
    arrow([(1.38, host_rows[0] + host_height), (1.38, host_rows[1])])
    centers = []
    lows = []
    for index, (title, stages) in enumerate(PATHS):
        top = device_top + index * row_pitch
        axis.text(left + 0.6, top + 0.02, title, fontsize=FONT, fontweight="bold", va="top")  # clear of the arrow that enters the exact path at its first stage
        y = top + title_height
        lows.append(y + box_height / 2)
        for number, (stage, backend) in enumerate(stages):
            x = left + number * (box_width + gap)
            box(x, y, box_width, box_height, stage, DEVICE, runs=backend)
            if number:
                arrow([(x - gap, y + box_height / 2), (x, y + box_height / 2)])
        centers.append((left + (len(stages) - 1) * (box_width + gap) + box_width, y))
    # the batches go down the left side to the first stage of each row, the results come up the right side
    bus_left, bus_right = 0.46, WIDTH - 0.12
    arrow([(bus_left, host_rows[1] + host_height), (bus_left, lows[-1])], head=False)
    for low in lows:
        arrow([(bus_left, low), (left, low)])
    arrow([(bus_right, lows[-1]), (bus_right, host_rows[0] + host_height)], head=False)
    for (x_end, y), low in zip(centers, lows):
        arrow([(x_end, low), (bus_right, low)], head=False)
    axis.annotate("", xy=(bus_right, host_rows[0] + host_height), xytext=(bus_right, host_rows[0] + host_height + 0.12), arrowprops=dict(arrowstyle="-|>", color="0.25", linewidth=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))
    # the box spectrum goes up to the peaks, and the peaks go back down into the exact path on narrow windows: the hybrid
    peak_x, peak_y, peak_w, peak_h = placed["peaks"]
    box_end = centers[0][0] - box_width / 2
    arrow([(box_end, centers[0][1]), (box_end, peak_y + peak_h)], color="#9a7b00")
    exact_first = left + box_width / 2
    gap_y = centers[0][1] + box_height + 0.14
    arrow([(peak_x + peak_w, peak_y + peak_h / 2), (WIDTH - 0.04, peak_y + peak_h / 2), (WIDTH - 0.04, gap_y), (exact_first, gap_y), (exact_first, centers[1][1])], color="#9a7b00", style=(0, (4, 2)))
    axis.text(2.0, gap_y - 0.04, "windows of 1% in period", fontsize=FONT, ha="center", va="bottom", color="#6b5600")
    axis.text(box_end - 0.07, boundary - 0.04, "box spectrum", fontsize=FONT, ha="right", va="bottom", color="#6b5600")
    axis.text(bus_left + 0.07, boundary + 0.1, "periods", fontsize=FONT, ha="left", va="top", style="italic")
    axis.text(bus_right - 0.07, boundary + 0.2, "chi square,\nrow, depth", fontsize=FONT, ha="right", va="top", style="italic", linespacing=1.05)
    axis.text(0.03, height - 0.06, "grey: TLS code   blue: this package, device   yellow: this package, host", fontsize=FONT, va="bottom")
    figure.savefig(path, metadata={"CreationDate": None, "Producer": None})
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE.parent / "pipeline.pdf")
    arguments = parser.parse_args()
    pipeline_figure(arguments.out)


if __name__ == "__main__":
    main()
