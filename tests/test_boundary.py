"""The dependencies of the package: no module of transitgpu imports anything but the standard library, numpy, numba, torch, triton and itself; the modules of transitgpu/tls, which are backends of Transit Least Squares (the extra `tls`), import that too.
A structure check on the source, which a data archive does not hold."""

import ast
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "transitgpu"
ALLOWED = {"transitgpu", "numpy", "numba", "torch", "triton"}
TLS = {"transitleastsquares"}  # imported by the modules of the subpackage tls only


def imported_modules(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def outside_the_allowed(path):
    allowed = ALLOWED | (TLS if path.parent.name == "tls" else set())
    return sorted({name.split(".")[0] for name in imported_modules(path)} - allowed - set(sys.stdlib_module_names))


def test_no_module_of_the_package_imports_anything_but_its_dependencies():
    files = sorted(SOURCE.rglob("*.py"))
    assert files and any(path.parent.name == "tls" for path in files), "the scan found no module, or not those of tls"
    found = {str(path.relative_to(SOURCE)): outside_the_allowed(path) for path in files}
    assert not any(found.values()), f"imports outside the standard library, numpy, numba, torch and triton (and TLS in transitgpu/tls): {found}"


def test_the_scan_sees_an_import_of_another_package_in_a_module_that_has_one(tmp_path):
    """The edge case that would hide a failure: a scan that finds nothing in any module."""
    (tmp_path / "bad.py").write_text("import os\n\ndef f():\n    from pandas.core import frame\n    import numpy\n")
    assert outside_the_allowed(tmp_path / "bad.py") == ["pandas"]
