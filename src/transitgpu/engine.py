"""The search engine of transitgpu for callers that choose the device themselves.

`transitgpu.search` and `transitgpu.search_many` refuse to run without a GPU. The functions here are the same engine
without that check, for a caller that names its device (the CPU is slow but works), with the constants and helpers
that reproduce the grids and the boxes of a search and the Transit Least Squares parts of the hybrid search:

    from transitgpu import engine
    found = engine.search_many(time, [flux], 0.5, 14.0, device="cpu", template=template, cuts=[7.0, 14.0])

The names in `__all__` keep their meaning and signatures between releases; `transitgpu.core`, `transitgpu.hybrid`
and `transitgpu.tls` hold the implementation and may change. Besides the arguments of the public searches, `search`
and `search_many` take `template`, `cuts`, `precision`, `weights` and `bumps` (see `search_many`). `WindowBackend`,
`refine_peak` and `tls_hybrid` need Transit Least Squares (the extra `tls`) and are imported on first use;
`tls_hybrid` is the module whose `refine_peak` the hybrid search calls, for a caller that times or replaces it.
"""

import importlib

from transitgpu.core import (
    DURATION_STEP,
    MAXIMUM_BINS,
    MAXIMUM_WEIGHT,
    MINIMUM_BINS,
    MINIMUM_BINS_PER_PERIOD,
    MINIMUM_POINTS,
    PRECISION,
    WEIGHT_UNIT,
    device_name,
    duration_fractions,
    fold_index,
    grid_of,
    search,
    search_many,
)
from transitgpu.hybrid import refine_peaks, top_peaks

_NEEDS_TLS = {"WindowBackend": ("transitgpu.tls.window", "WindowBackend"), "refine_peak": ("transitgpu.tls.hybrid", "refine_peak"), "tls_hybrid": ("transitgpu.tls.hybrid", None)}

__all__ = ["DURATION_STEP", "MAXIMUM_BINS", "MAXIMUM_WEIGHT", "MINIMUM_BINS", "MINIMUM_BINS_PER_PERIOD", "MINIMUM_POINTS", "PRECISION", "WEIGHT_UNIT", "WindowBackend", "device_name", "duration_fractions",
           "fold_index", "grid_of", "refine_peak", "refine_peaks", "search", "search_many", "tls_hybrid", "top_peaks"]


def __getattr__(name):
    """Import the names that need Transit Least Squares on first access (PEP 562)."""
    if name not in _NEEDS_TLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _NEEDS_TLS[name]
    loaded = importlib.import_module(module)
    return loaded if attribute is None else getattr(loaded, attribute)
