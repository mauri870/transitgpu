"""Transit Least Squares (TLS 2.0.1, Hippke & Heller 2019) with its search of the trial periods on a GPU.

The grids, the transit model, the statistics and the final fit are TLS's. Only the import changes:

    from transitgpu.tls import transitleastsquares
    results = transitleastsquares(t, y, dy).power(period_min=0.5, period_max=14)

The backend of `power()` is "gpu" unless the call or the environment variable TLS_BACKEND names another. "gpu" is
exact: TLS's statistic to float64 rounding. "gpu-binned" is approximate: the points are folded into phase bins and
the windows have a fixed width in phase, which differs from TLS's statistic where the folded points cover the phase
unevenly. TLS's own backends ("fused", "fused-pl", ...) remain available by name.
"""
try:
    from transitleastsquares import (  # noqa: F401  TLS's interface, re-exported
        FAP,
        SearchBackend,
        __version__ as tls_version,
        available_backends,
        catalog_info,
        cleaned_array,
        duration_grid,
        fold,
        get_backend,
        period_grid,
        register_backend,
        resample,
        shutdown_workers,
        transit_mask,
        warmup,
    )
    from transitleastsquares import transitleastsquares as _tls
except ModuleNotFoundError as error:
    if error.name != "transitleastsquares":
        raise
    raise ImportError('transitgpu.tls needs Transit Least Squares: uv pip install "transitgpu[tls] @ git+https://github.com/mauri870/transitgpu"') from None

from transitgpu import __version__

TLS_VERSION = "2.0.1"
__all__ = [
    "FAP", "SearchBackend", "TLS_VERSION", "__version__", "available_backends", "catalog_info", "check_version", "cleaned_array", "duration_grid", "fold", "get_backend", "period_grid", "register_backend", "resample",
    "shutdown_workers", "tls_version", "transit_mask", "transitleastsquares", "warmup",
]


def check_version(installed):
    """Raise ImportError unless `installed` is TLS_VERSION, the TLS version whose statistic the backends reproduce."""
    if installed != TLS_VERSION:
        raise ImportError(f'transitgpu.tls needs transitleastsquares=={TLS_VERSION}, found {installed}: uv pip install "transitleastsquares=={TLS_VERSION}"')


check_version(tls_version)


def _exact():
    from .window import WindowBackend

    return WindowBackend()


def _binned():
    from .binned import BinnedBackend

    return BinnedBackend()


def _piecewise_linear():
    from .pl import PLBackend

    return PLBackend()


register_backend("gpu", _exact)
register_backend("gpu-binned", _binned)
register_backend("gpu-pl", _piecewise_linear)


class transitleastsquares(_tls):
    """TLS's class with "gpu" as the default backend of `power()`."""

    def power(self, **kwargs):
        if kwargs.get("backend") is None:
            import os

            kwargs["backend"] = os.environ.get("TLS_BACKEND") or "gpu"
        return super().power(**kwargs)
