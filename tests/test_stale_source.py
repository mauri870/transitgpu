"""A process that has imported transitgpu survives the replacement of a file of the package (what a sync of the environment does under a running process).

Triton reads the source of a kernel with inspect.getsource when the kernel is defined, with the line numbers of the code that was imported. A kernel defined inside a function on its first call reads the file as it is then, and a file that was replaced since the import has other lines. The kernels are therefore defined when their module is imported."""
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

triton = pytest.importorskip("triton")

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "transitgpu"
WITH_KERNELS = sorted(path for path in PACKAGE.rglob("*.py") if "@triton.jit" in path.read_text())


def module_name(path):
    return ".".join(("transitgpu", *path.relative_to(PACKAGE).with_suffix("").parts)).removesuffix(".__init__")


def test_the_package_has_files_that_define_triton_kernels():
    assert PACKAGE / "core.py" in WITH_KERNELS and len(WITH_KERNELS) >= 2


@pytest.mark.parametrize("path", WITH_KERNELS, ids=lambda path: path.name)
def test_every_triton_kernel_of_a_module_is_defined_when_the_module_is_imported(path):
    import importlib

    from triton.runtime.jit import JITFunction

    module = importlib.import_module(module_name(path))
    defined = [name for name, value in vars(module).items() if isinstance(value, JITFunction)]
    wanted = len(re.findall(r"^\s*@triton\.jit", path.read_text(), re.MULTILINE))
    assert len(defined) == wanted, f"{module.__name__} has {wanted} @triton.jit and {len(defined)} kernels at module level"


def run(tmp_path, program):
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PACKAGE, tmp_path / "transitgpu", ignore=shutil.ignore_patterns("__pycache__"))
    environment = {"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(program)], env=environment, capture_output=True, text=True, timeout=600)


REPLACE = """
    import linecache, pathlib
    import transitgpu
    from transitgpu import core
    path = pathlib.Path(core.__file__)
    path.write_text("# a block of lines\\n" * 40 + path.read_text())  # other line numbers, as after a sync
    linecache.clearcache()  # IPython and pytest do this
"""


SOURCES = """
    import hashlib
    kernels = [core._accumulate_kernel(), core._accumulate_many_kernel(), core._accumulate_once_kernel()]
    assert all(kernels), kernels
    print("sources", [hashlib.sha256(kernel[0].src.encode()).hexdigest()[:12] for kernel in kernels])
"""


def test_the_source_of_the_kernels_of_core_is_the_same_after_its_file_is_replaced(tmp_path):
    plain = run(tmp_path / "plain", "\n    import transitgpu\n    from transitgpu import core\n" + SOURCES)
    replaced = run(tmp_path / "replaced", REPLACE + SOURCES)
    assert plain.returncode == 0, plain.stderr[-2000:]
    assert replaced.returncode == 0, replaced.stderr[-2000:]
    assert plain.stdout.strip().splitlines()[-1] == replaced.stdout.strip().splitlines()[-1], (plain.stdout, replaced.stdout)


@pytest.mark.gpu
def test_a_search_after_the_file_was_replaced_still_works(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the kernels are built on a device")
    result = run(tmp_path, REPLACE + """
    import numpy as np
    rng = np.random.default_rng(1)
    time = np.arange(20000) * 120 / 86400.0
    flux = 1.0 + rng.normal(0, 1e-3, time.size)
    transitgpu.search(time, flux, 0.5, 14.0)
    print("searched")
    """)
    assert result.returncode == 0 and "searched" in result.stdout, result.stderr[-2000:]


def test_the_hip_source_that_is_built_is_the_text_read_when_the_module_was_imported(tmp_path):
    """accumulate.hip was read from its path at the first build, which may come long after the import: a process that survives the replacement of the package's files (a sync of the environment) built the new file with the Python of the old. The text is read at import and the build is made from a file written from it."""
    program = """
    import hashlib, pathlib, tempfile
    import transitgpu
    from transitgpu import hip
    original = hashlib.sha256(hip.SOURCE.read_bytes()).hexdigest()
    hip.SOURCE.write_text("// replaced after the import\\n")
    built = hip._source_path(pathlib.Path(tempfile.mkdtemp()))
    print("same", hashlib.sha256(built.read_bytes()).hexdigest() == original, built.name.startswith("accumulate-") and built.suffix == ".hip")
    """
    result = run(tmp_path, program)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "same True True", result.stdout
