"""The references of the paper: every key that main.tex cites is in refs.bib, and no entry of refs.bib is a work that is not yet released (a note or a status of in preparation, submitted or unpublished). A function test of the text of two files."""
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parent.parent / "paper"
if not (PAPER / "main.tex").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)

UNRELEASED = re.compile(r"in preparation|submitted|unpublished", re.I)


def entries():
    """{key: text of the entry} of refs.bib."""
    text = (PAPER / "refs.bib").read_text()
    starts = [(match.group(1), match.start()) for match in re.finditer(r"^@[A-Za-z]+\{([^,\s]+),", text, re.M)]
    ends = [position for _, position in starts[1:]] + [len(text)]
    return {key: text[start:end] for (key, start), end in zip(starts, ends)}


def cited_keys():
    """The keys of every \\cite command of main.tex (cite, citet, citep, citealt and the like, with their optional arguments)."""
    keys = set()
    for match in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", (PAPER / "main.tex").read_text()):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def test_every_key_that_the_paper_cites_is_in_the_bibliography():
    assert not (cited_keys() - set(entries())), sorted(cited_keys() - set(entries()))


def test_no_entry_of_the_bibliography_is_a_work_in_preparation_submitted_or_unpublished():
    unreleased = sorted(key for key, text in entries().items() if UNRELEASED.search(text))
    assert not unreleased, unreleased


def test_the_check_sees_a_missing_key_and_an_unreleased_note_in_a_probe():
    assert cited_keys() - {"nothing"} == cited_keys(), "the keys are read from main.tex"
    assert UNRELEASED.search("note = {Research Note in preparation}") and UNRELEASED.search("note = {Submitted to AJ}") and not UNRELEASED.search("journal = {MNRAS}")
