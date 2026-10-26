"""The PDF of the paper that the README links is not older than its sources: paper/main.pdf.sources holds the SHA-256 of main.tex, numbers.tex and refs.bib at the build (paper/build.py), and a source that changed
after it fails the test with the command that rebuilds the PDF. The PDF itself is not compared (it carries the date of the build)."""
import importlib.util
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parent.parent / "paper"
if not (PAPER / "build.py").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("transitgpu_paper_build", PAPER / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_the_pdf_exists_and_was_built_from_the_sources_that_are_in_the_tree():
    assert (PAPER / "main.pdf").stat().st_size > 10_000 and (PAPER / "main.pdf").read_bytes().startswith(b"%PDF")
    assert (PAPER / "main.pdf.sources").read_text() == build.digests(), "a source of the paper changed after main.pdf was built: run `uv run python paper/build.py` and commit main.pdf and main.pdf.sources"


def test_the_record_names_the_three_sources_and_a_change_of_one_changes_it(tmp_path, monkeypatch):
    lines = build.digests().splitlines()
    assert [line.split("  ")[1] for line in lines] == ["main.tex", "numbers.tex", "refs.bib"] and all(len(line.split("  ")[0]) == 64 for line in lines)
    for name in build.SOURCES:
        (tmp_path / name).write_text("a")
    monkeypatch.setattr(build, "HERE", tmp_path)
    before = build.digests()
    (tmp_path / "refs.bib").write_text("b")
    assert build.digests() != before
