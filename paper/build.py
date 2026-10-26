"""Build paper/main.pdf with tectonic (https://tectonic-typesetting.github.io, it fetches the LaTeX packages it needs on its first run) and record the sources it was built from.

usage: uv run python paper/build.py
The PDF carries the date of the build, so it is not compared byte for byte: main.pdf.sources holds the SHA-256 of main.tex, numbers.tex and refs.bib at the build, and tests/test_paper_pdf.py fails when a source changed after it.
"""
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ("main.tex", "numbers.tex", "refs.bib")


def digests():
    return "".join(f"{hashlib.sha256((HERE / name).read_bytes()).hexdigest()}  {name}\n" for name in SOURCES)


def main():
    subprocess.run(["tectonic", "main.tex"], cwd=HERE, check=True)
    (HERE / "main.pdf.sources").write_text(digests())
    print("wrote main.pdf and main.pdf.sources")


if __name__ == "__main__":
    main()
