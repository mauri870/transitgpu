"""The names and the version of transitgpu and transitgpu.tls: one version, from the metadata of the package; what each module exports; and how `import transitgpu.tls` fails without Transit Least Squares or with another version of it."""
import importlib.metadata
import subprocess
import sys

import pytest

import transitgpu


def run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_the_version_of_the_package_is_the_one_of_its_metadata_and_transitgpu_tls_has_the_same():
    assert transitgpu.__version__ == importlib.metadata.version("transitgpu")
    pytest.importorskip("transitleastsquares")
    from transitgpu import tls

    assert tls.__version__ == transitgpu.__version__


def test_the_interface_of_transitgpu_lists_its_names():
    assert sorted(transitgpu.__all__) == ["Results", "__version__", "bls_tls_hybrid", "box_search", "device_name", "grid_of", "search", "search_many"]
    assert all(hasattr(transitgpu, name) for name in transitgpu.__all__)


def test_transitgpu_tls_lists_the_names_it_exports_and_a_star_import_brings_only_those():
    pytest.importorskip("transitleastsquares")
    found = run("import transitgpu.tls as t; ns = {}; exec('from transitgpu.tls import *', ns); ns.pop('__builtins__'); print(sorted(ns) == sorted(t.__all__)); print('transitleastsquares' in t.__all__, '__version__' in t.__all__, 'os' in ns)")
    assert found.returncode == 0, found.stderr
    assert found.stdout.split() == ["True", "True", "True", "False"], found.stdout


def test_importing_transitgpu_tls_without_transit_least_squares_says_how_to_install_it():
    found = run("import sys; sys.modules['transitleastsquares'] = None\ntry:\n    import transitgpu.tls\nexcept ImportError as error:\n    print(type(error).__name__); print(error)")
    assert found.returncode == 0, found.stderr
    kind, message = found.stdout.split("\n", 1)
    assert kind == "ImportError" and 'pip install "transitgpu[tls] @ git+https://github.com/mauri870/transitgpu"' in message, found.stdout


def test_another_version_of_transit_least_squares_than_the_supported_one_is_an_error_that_names_both():
    pytest.importorskip("transitleastsquares")
    from transitgpu import tls

    assert tls.check_version(tls.TLS_VERSION) is None
    with pytest.raises(ImportError, match=r"2\.0\.1.*9\.9\.9|9\.9\.9.*2\.0\.1") as error:
        tls.check_version("9.9.9")
    assert 'pip install "transitleastsquares==2.0.1"' in str(error.value)
