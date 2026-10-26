"""GPU-accelerated BLS and TLS for exoplanet transit detection."""

from importlib.metadata import version as _version

from transitgpu.core import device_name, grid_of
from transitgpu.hybrid import bls_tls_hybrid
from transitgpu.model import Results, box_search, search, search_many

__version__ = _version("transitgpu")
__all__ = ["Results", "__version__", "bls_tls_hybrid", "box_search", "device_name", "grid_of", "search", "search_many"]
