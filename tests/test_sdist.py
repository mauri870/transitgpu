"""The source distribution carries the package, its tests and its examples and leaves out the paper and the workflow (pyproject.toml lists what it includes), so its tests must run without them. A test that builds it and runs its tests from the
unpacked tree. A function test: it is about the files of a build and not about a light curve."""
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / ".github").exists():
    pytest.skip("the source distribution is built from the repository, not from an unpacked copy of it", allow_module_level=True)


def test_the_source_distribution_collects_all_its_tests_and_passes_the_package_tests_from_the_unpacked_tree(tmp_path):
    build = subprocess.run(["uv", "build", "--sdist", "--offline", "--out-dir", str(tmp_path / "dist"), str(ROOT)], capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip("the source distribution cannot be built offline here: " + build.stderr.strip().splitlines()[-1])
    archive = next((tmp_path / "dist").glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "unpacked", filter="data")
    tree = next((tmp_path / "unpacked").iterdir())
    names = {path.name for path in tree.iterdir()}
    assert "paper" not in names and ".github" not in names and "tests" in names, names
    environment = {**os.environ, "PYTHONPATH": str(tree / "src")}  # the wheel test of the unpacked tree builds with uv
    where = subprocess.run([sys.executable, "-c", "import transitgpu; print(transitgpu.__file__)"], cwd=tree, capture_output=True, text=True, env=environment)
    assert where.returncode == 0 and Path(where.stdout.strip()).resolve().is_relative_to(tree.resolve()), "the unpacked tests import the package of the unpacked tree: " + where.stdout + where.stderr
    collect = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=tree, capture_output=True, text=True, env=environment)
    assert collect.returncode == 0, collect.stdout[-3000:] + collect.stderr[-1000:]
    run = subprocess.run([sys.executable, "-m", "pytest", "tests/test_package_files.py", "tests/test_public_api.py", "-q", "-p", "no:cacheprovider"], cwd=tree, capture_output=True, text=True, env=environment)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-1000:]
