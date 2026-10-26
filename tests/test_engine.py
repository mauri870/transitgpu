"""transitgpu.engine, the interface for callers that choose the device: its names, the signatures they are called with, that the engine is the one of transitgpu.core without the check of the public searches, and that
importing it does not import Transit Least Squares. Function tests: they check names, signatures and an import, which no data archive describes."""
import inspect
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from test_transitgpu import light_curve  # noqa: E402

import transitgpu  # noqa: E402
from transitgpu import core, engine  # noqa: E402

NAMES = ["DURATION_STEP", "MAXIMUM_BINS", "MAXIMUM_WEIGHT", "MINIMUM_BINS", "MINIMUM_BINS_PER_PERIOD", "MINIMUM_POINTS", "PRECISION", "WEIGHT_UNIT", "WindowBackend", "device_name", "duration_fractions",
         "fold_index", "grid_of", "refine_peak", "refine_peaks", "search", "search_many", "tls_hybrid", "top_peaks"]
SEARCH_KEYWORDS = {"spectrum", "device", "batch", "minimum_bins", "duration_step", "periods", "boxes", "template", "precision", "weights", "oversampling_factor", "n_transits_min"}


def test_the_engine_lists_the_names_it_keeps_and_every_one_resolves():
    assert sorted(engine.__all__) == sorted(NAMES)
    for name in NAMES:
        assert getattr(engine, name) is not None, name


def test_the_searches_of_the_engine_are_those_of_core_and_not_the_public_ones_that_refuse_a_cpu():
    assert engine.search is core.search and engine.search_many is core.search_many
    assert transitgpu.search is not engine.search and transitgpu.search_many is not engine.search_many


def test_the_searches_take_the_arguments_that_callers_pass():
    many = inspect.signature(engine.search_many).parameters
    one = inspect.signature(engine.search).parameters
    assert list(many)[:4] == ["time", "fluxes", "period_min", "period_max"] and list(one)[:4] == ["time", "flux", "period_min", "period_max"]
    assert SEARCH_KEYWORDS <= set(many) | {"threads"} and SEARCH_KEYWORDS <= set(one) | {"threads"}
    assert {"cuts", "duration_hours", "labeled"} <= set(many) and "threads" in one
    assert list(inspect.signature(engine.grid_of).parameters)[:4] == ["time", "flux", "period_min", "period_max"]
    assert list(inspect.signature(engine.refine_peak).parameters) == ["t", "y", "dy", "period", "window", "backend", "tls_options"]
    assert list(inspect.signature(engine.refine_peaks).parameters)[:6] == ["t", "y", "dy", "periods", "power", "peaks"]
    assert list(inspect.signature(engine.top_peaks).parameters) == ["periods", "power", "count", "window"]
    assert list(inspect.signature(engine.fold_index).parameters) == ["t", "p", "bins", "shifts"]
    assert list(inspect.signature(engine.WindowBackend).parameters) == ["device", "dtype", "batch", "correlate"]


def test_the_constants_have_the_types_and_the_keys_that_callers_read():
    assert isinstance(engine.DURATION_STEP, float) and engine.DURATION_STEP > 1
    assert all(isinstance(getattr(engine, name), int) for name in ("MINIMUM_BINS", "MAXIMUM_BINS", "MINIMUM_BINS_PER_PERIOD", "MINIMUM_POINTS", "WEIGHT_UNIT", "MAXIMUM_WEIGHT"))
    assert engine.MINIMUM_BINS < engine.MAXIMUM_BINS and engine.MAXIMUM_WEIGHT >= engine.WEIGHT_UNIT
    assert engine.PRECISION == {"statistic": "float32"}


def test_the_engine_searches_on_the_cpu_where_the_public_search_refuses():
    time, flux = light_curve()
    found = engine.search(time, flux, 0.5, 14.0, device="cpu")
    assert found["period_days"] > 0 and found["sde"] > 0
    with pytest.raises(RuntimeError):
        transitgpu.search(time, flux, 0.5, 14.0, device="cpu")


def test_importing_the_engine_does_not_import_transit_least_squares_and_its_tls_names_do():
    code = "import sys, transitgpu.engine as e; print('transitleastsquares' in sys.modules); print(e.WindowBackend.__name__); print('transitleastsquares' in sys.modules)"
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert found.returncode == 0, found.stderr
    assert found.stdout.split() == ["False", "WindowBackend", "True"], found.stdout


def test_a_name_that_the_engine_does_not_have_raises_attribute_error():
    with pytest.raises(AttributeError):
        engine.accumulate_many
