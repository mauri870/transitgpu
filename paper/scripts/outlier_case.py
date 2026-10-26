"""A dip next to a large flux outlier, searched by TLS 2.0.1 and by the exact search of transitgpu.tls.

See Sect. "The exact template search against TLS" of the paper. With one point 1e5 above the level before the dip,
TLS 2.0.1's fused backend reports no improvement although the dip lowers the chi square.

The light curve is one period (PERIOD = 3 d) of N = 6,000 points whose phase order is the order of the array: one dip
with the shape of TLS's template of 684 samples and depth 0.002, and one point 1e5 above the level at index 30, before
the dip. Errors are equal (1e-3). The script evaluates the chi square of that template at every start by direct sums
in long double (no prefix sums), and prints what the numba and fused backends (and transitgpu, with a GPU) return for
the same period.
"""
import warnings

import numpy as np
from transitleastsquares import transitleastsquares
from transitleastsquares.backends import get_backend
from transitleastsquares.validate import validate_args

N, PERIOD, WIDTH = 6000, 3.0, 684
OUTLIER, DEPTH, START = 1e5, 0.002, 2000

t = (np.arange(N) + 0.5) * PERIOD / N  # at the period PERIOD the phase order of the points is their order
dy = np.full(N, 1e-3)


def problem_for(y, backend):
    model = transitleastsquares(t, y, dy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = validate_args(model, dict(period_min=0.9 * PERIOD, period_max=1.1 * PERIOD, use_threads=1, show_progress_bar=False, verbose=False, backend=backend, n_transits_min=1))
        periods, durations = model._grids()
        overview, templates = model._templates(durations)
        model._problem(model.t, model.y, model.dy, overview, templates)
    return model, overview, templates


model, overview, templates = problem_for(np.ones(N), "fused")
row = int(np.flatnonzero(overview["width_in_samples"] == WIDTH)[0])
template = 1.0 - np.asarray(model._problem(model.t, model.y, model.dy, overview, templates).lc_arr[row], dtype=float)
dip = np.ones(N)
dip[START:START + len(template)] -= DEPTH * template


def array(outlier, at):
    y = dip.copy()
    y[at] += outlier
    return y


# The chi square of the template of row `row` at every start by direct dot products:
# chi2(start) = constant - (2 k AR - k^2 A2) / weight, TLS's chi square for unit errors.
weight = 1.0 / dy[0] ** 2
overshoot = float(overview["overshoot"][row])
signal_depth = 0.5  # transitleastsquares.tls_constants.SIGNAL_DEPTH
a2 = weight * float(np.dot(template, template))
y = array(OUTLIER, 30)
deviation = (y - 1.0).astype(np.longdouble)
means = -np.array([deviation[s:s + WIDTH].sum() for s in range(N - len(template) + 1)]) / WIDTH
windows = np.lib.stride_tricks.sliding_window_view(1.0 - y.astype(np.longdouble), len(template))
ar = weight * (windows @ template.astype(np.longdouble))
k = means * overshoot / signal_depth
gain = (2 * k * ar - k * k * a2)[means > 0] / weight
constant = float(np.sum((1.0 - y.astype(np.longdouble)) ** 2))
best = int(np.argmax(np.where(means > 0, (2 * k * ar - k * k * a2) / weight, -np.inf)))
print(f"direct evaluation: constant chi2 {constant:.12g}; template row {row} at start {best} lowers it by {float(((2 * k * ar - k * k * a2) / weight)[best]):.6g}")

backends = [("numba", "numba", get_backend("numba")), ("fused", "fused", get_backend("fused"))]
try:
    import torch

    if torch.cuda.is_available():
        from transitgpu.tls.window import WindowBackend

        backends.append(("transitgpu", "fused", WindowBackend()))
except ImportError:
    pass

for label, outlier, at in (("no outlier", 0.0, 30), ("outlier 1e5 at index 5970, after the dip", OUTLIER, 5970), ("outlier 1e5 at index 30, before the dip", OUTLIER, 30)):
    for backend, name, engine in backends:
        model, overview, templates = problem_for(array(outlier, at), name)
        found = model._search(engine, np.array([PERIOD]), overview, templates)
        finish = getattr(engine, "finish", None)
        if finish is not None:
            finish()
        print(f"{label:42s} {backend:6s} template row {int(found.rows[0]):3d}  depth {float(found.depths[0]):.6f}  chi2 {float(found.chi2[0]):.12g}")
