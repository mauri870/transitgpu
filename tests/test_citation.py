"""CITATION.cff: a file that GitHub's Cite this repository reads. It parses, has the fields of the format that the software and the paper need, holds the version of the package, and gives the paper the title of the
paper's source and the references the titles of refs.bib, so that a change of one without the other is caught."""
import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "paper" / "main.tex").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
CFF = yaml.safe_load((ROOT / "CITATION.cff").read_text())
STATUSES = {"abstract", "advance-online", "in-preparation", "in-press", "preprint", "submitted"}  # the values that CFF 1.2.0 allows for the status of a reference


def title_of_the_paper():
    match = re.search(r"^\\title\{(.*)\}\s*$", (ROOT / "paper" / "main.tex").read_text(), re.M)
    assert match, "main.tex has a \\title on one line"
    return match.group(1).replace("\\\\ ", " ")  # a line break of the title is a space of the citation


def title_of(bibliography, key):
    entry = re.search(r"@\w+\{" + key + r",(.*?)\n@", bibliography + "\n@", re.S).group(1)
    return re.sub(r"[{}]", "", re.search(r"title = \{(.*)\},?\n", entry).group(1))


def test_the_file_has_the_fields_cff_requires_for_software_and_the_licence_and_addresses_of_the_package():
    assert CFF["cff-version"] == "1.2.0" and CFF["type"] == "software" and CFF["title"] == "transitgpu" and CFF["message"]
    assert CFF["authors"] == [{"family-names": "Meneguzzo", "given-names": "Mauri de Souza"}] and CFF["license"] == "MIT"
    urls = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["urls"]
    assert CFF["repository-code"] == CFF["url"] == urls["Homepage"] == urls["Source"] == "https://github.com/mauri870/transitgpu"


def test_the_version_is_the_version_of_the_package():
    assert CFF["version"] == tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_the_preferred_citation_is_the_paper_with_the_title_of_its_source_and_a_valid_status():
    paper = CFF["preferred-citation"]
    assert paper["type"] == "article" and paper["title"] == title_of_the_paper(), "the title of main.tex is the title of the citation"
    assert paper["authors"] == CFF["authors"] and paper["status"] in STATUSES and paper["year"] == 2026
    assert paper["url"] == "https://github.com/mauri870/transitgpu/blob/main/paper/main.pdf"


def test_the_references_are_tls_and_the_box_search_with_the_titles_and_the_pages_of_the_bibliography():
    bibliography = (ROOT / "paper" / "refs.bib").read_text()
    found = {reference["title"]: reference for reference in CFF["references"]}
    assert set(found) == {title_of(bibliography, "HippkeHeller2019"), title_of(bibliography, "Kovacs2002")}
    tls, bls = found[title_of(bibliography, "HippkeHeller2019")], found[title_of(bibliography, "Kovacs2002")]
    assert (tls["volume"], tls["start"], tls["year"]) == (623, "A39", 2019) and (bls["volume"], bls["start"], bls["year"]) == (391, "369", 2002)
    assert all(reference["type"] == "article" and reference["journal"] == "Astronomy & Astrophysics" for reference in found.values())
