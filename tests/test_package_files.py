"""The files that a published package carries: the licence of the package, the notice of the code copied from TLS, and the licence field of the metadata."""
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_the_package_is_licensed_under_mit_and_keeps_the_notice_of_the_code_copied_from_tls():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert metadata["license"] == "MIT" and set(metadata["license-files"]) == {"LICENSE", "THIRD_PARTY/TLS-LICENSE"}
    licence = (ROOT / "LICENSE").read_text()
    assert licence.startswith("MIT License") and "Copyright (c) 2026 Mauri de Souza Meneguzzo" in licence
    notice = (ROOT / "THIRD_PARTY" / "TLS-LICENSE").read_text()
    assert "Copyright (c) 2018 Michael Hippke" in notice
    assert "THIRD_PARTY/TLS-LICENSE" in (ROOT / "README.md").read_text(), "the README says where the notice of the copied code is"


def test_the_readme_says_which_devices_the_package_has_been_run_on_and_installs_with_uv_and_names_the_torch_options():
    text = (ROOT / "README.md").read_text()
    assert "ROCm 7.2" in text and "gfx1100" in text and "not yet tested on NVIDIA" in text
    for form in ('uv pip install "transitgpu[tls] @ git+https://github.com/mauri870/transitgpu"', "https://download.pytorch.org/whl/rocm7.2", "https://docs.astral.sh/uv/guides/integration/pytorch/"):
        assert form in text, form
    assert "python -m pip" not in text and "pytest transitgpu/tests" not in text, "the commands are uv's, run from the package root"
    assert "https://github.com/mauri870/transitgpu/issues" in text and "![WASP-18 b found by the box search and by TLS](docs/wasp18.png)" in text


def test_the_commands_of_the_readme_the_benchmark_and_the_examples_are_uv_runs_from_the_package_root():
    for path in ("README.md", "benchmarks/bench.py", "examples/wasp18_b.py", "examples/synthetic_transit.py", "examples/plot_wasp18.py"):
        text = (ROOT / path).read_text()
        assert "python -m pip" not in text and "pip install" not in text.replace("uv pip install", ""), path
    assert "uv run python benchmarks/bench.py" in (ROOT / "benchmarks" / "bench.py").read_text()
    assert "uv run --group rocm python examples/synthetic_transit.py" in (ROOT / "examples" / "synthetic_transit.py").read_text()
    assert "uv run --group rocm python examples/wasp18_b.py" in (ROOT / "examples" / "wasp18_b.py").read_text()
    assert "uv run --group rocm --extra examples python examples/plot_wasp18.py" in (ROOT / "examples" / "plot_wasp18.py").read_text()


def test_the_metadata_names_the_author_the_addresses_and_the_lower_bounds_the_code_needs():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = metadata["project"]
    assert project["authors"] == [{"name": "Mauri de Souza Meneguzzo"}] and "Mauri de Souza Meneguzzo" in (ROOT / "LICENSE").read_text()
    assert set(project["urls"]) == {"Homepage", "Source", "Issues"} and {value.split("/issues")[0] for value in project["urls"].values()} == {"https://github.com/mauri870/transitgpu"}
    assert "https://github.com/mauri870/transitgpu" in (ROOT / "CITATION.cff").read_text(), "the citation file points at the address of the project metadata"
    bounds = {requirement.split(">=")[0]: requirement.split(">=")[1] for requirement in project["dependencies"]}
    assert bounds == {"numpy": "1.26", "numba": "0.59", "torch": "2.5"}, "torch 2.5 for torch.OutOfMemoryError, numba 0.59 and numpy 1.26 for Python 3.12"
    assert metadata["build-system"]["requires"] == ["hatchling>=1.27", "hatch-fancy-pypi-readme>=24.1.0"] and project["dynamic"] == ["readme"]
    assert any(classifier == "Programming Language :: Python :: 3.12" for classifier in project["classifiers"]) and project["requires-python"] == ">=3.12"
    assert set(project["optional-dependencies"]) == {"tls", "examples", "bench"}, "the extras are what a user installs; the tests and the builds of torch are dependency groups, which a wheel does not publish"
    groups = metadata["dependency-groups"]
    assert {"dev", "test"} <= set(groups) <= {"dev", "test", "validation", "cpu", "cu128", "rocm"} and groups["dev"] == [{"include-group": "test"}]  # cpu, cu128 and rocm are added by the export


def test_the_sdist_carries_what_a_reader_needs_to_run_the_tests_and_the_examples():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    included = set(metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert {"src", "tests", "examples", "benchmarks", "docs", "THIRD_PARTY", "LICENSE", "README.md"} <= included
    assert (ROOT / ".gitignore").exists() and all((ROOT / name).exists() for name in included)


def test_the_data_of_the_examples_and_of_the_golden_cases_carry_the_acknowledgement_that_the_archive_asks_for():
    tess = (ROOT / "examples" / "data" / "README.md").read_text()
    assert "This paper includes data collected by the TESS mission" in tess and "10.17909/t9-nmc8-f686" in tess and "NAS 5-26555" in tess
    k2 = (ROOT / "tests" / "golden_upstream" / "UPSTREAM").read_text()
    assert "EPIC201367065" in k2 and "EPIC206154641" in k2 and "K2" in k2 and "data collected by the Kepler mission" in k2 and "https://archive.stsci.edu/missions-and-data/k2" in k2
    assert "not of the data" in k2, "the licence of the code is not claimed for the data"


def test_the_readme_says_which_torch_group_the_tests_of_a_clone_run_with():
    assert "uv run --group rocm pytest" in (ROOT / "README.md").read_text()


def test_the_long_description_of_the_built_wheel_has_the_absolute_address_of_the_image(tmp_path):
    """The README's relative path renders on GitHub; PyPI shows the description away from the repository, so hatch-fancy-pypi-readme rewrites it in the metadata. The build needs the plugin, which an offline machine may not have: then the test is skipped."""
    done = subprocess.run(["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path), str(ROOT)], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip("the wheel cannot be built offline here: " + done.stderr.strip().splitlines()[-1])
    wheel = zipfile.ZipFile(next(tmp_path.glob("*.whl")))
    metadata = wheel.read(next(name for name in wheel.namelist() if name.endswith("METADATA"))).decode()
    assert "![WASP-18 b found by the box search and by TLS](https://raw.githubusercontent.com/mauri870/transitgpu/main/docs/wasp18.png)" in metadata
    assert "](https://raw.githubusercontent.com/mauri870/transitgpu/main/paper/main.pdf)" in metadata
    assert "](docs/" not in metadata and "](paper/" not in metadata


def test_the_paper_is_in_the_repository_and_in_neither_the_wheel_nor_the_source_distribution():
    if not (ROOT / "paper").exists():
        pytest.skip("the paper is not in the source distribution")
    targets = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]
    assert (ROOT / "paper" / "main.tex").is_file()
    assert targets["wheel"]["packages"] == ["src/transitgpu"]
    assert "paper" not in targets["sdist"]["include"], "the sdist is the list of what it carries, and the paper is not in it"


def test_the_readme_has_a_citation_section_that_links_the_paper_and_the_citation_file_and_a_licence_section_after_it():
    text = (ROOT / "README.md").read_text()
    assert text.index("## Attribution") < text.index("## Citation") < text.index("## Licence")
    citation = text[text.index("## Citation"):text.index("## Licence")]
    assert "(paper/main.pdf)" in citation and "Parallel Search for Alien Worlds: GPU-Accelerated BLS and TLS for Exoplanet Transit Detection" in citation
    assert "Hippke" not in citation and "Kovács" not in citation, "the references of the methods are given once, in Attribution"
    assert text.rstrip().endswith("`THIRD_PARTY/TLS-LICENSE`.") and "MIT licensed" in text[text.index("## Licence"):]


def test_every_relative_image_and_link_of_the_readme_exists_in_the_tree():
    if not (ROOT / "paper").exists():
        pytest.skip("the paper is not in the source distribution")
    text = (ROOT / "README.md").read_text()
    relative = [target for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text) if not re.match(r"[a-z]+:|#", target)]
    assert {"docs/wasp18.png", "paper/main.pdf"} <= set(relative)
    assert [target for target in relative if not (ROOT / target.split("#")[0]).exists()] == []


def test_the_third_line_of_the_readme_is_the_sentence_that_opens_its_description():
    lines = (ROOT / "README.md").read_text().splitlines()
    assert lines[0] == "# transitgpu" and lines[2] == "transitgpu is a parallel search for alien worlds: Box Least Squares (BLS) and Transit Least Squares (TLS) for exoplanet transit detection, accelerated on the GPU with PyTorch for CUDA and ROCm."


def test_the_readme_says_what_gpu_pl_is_and_that_it_needs_equal_errors():
    text = (ROOT / "README.md").read_text()
    assert '`backend="gpu-pl"` is TLS\'s default backend (`fused-pl`) on the GPU and needs equal errors.' in text
