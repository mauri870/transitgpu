"""The table of the completeness by period (paper/scripts/period_table.py): a row for each part and search, a column for each period, each cell the share found at that period with its Wilson interval. Function tests on results made up here."""
import importlib.util
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
if not (PACKAGE / "paper").exists():
    pytest.skip("the paper is not in the source distribution", allow_module_level=True)
spec = importlib.util.spec_from_file_location("period_table", PACKAGE / "paper" / "scripts" / "period_table.py")
period_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(period_table)


def results():
    cell = lambda found, n, low, high: {"found": found, "n": n, "wilson": [low, high]}
    by_period = {"0.8": cell(194, 240, 0.7538, 0.8531), "12.8": cell(120, 240, 0.4373, 0.5627)}
    searches = lambda: {name: {"0.05": {"by_period": by_period}} for name in ("exact", "binned", "box", "hybrid20", "hybrid5")}
    return {"main-white": {"searches": searches()}, "real": {"searches": searches()}}


def test_the_table_has_a_block_for_each_part_a_row_for_each_of_the_four_searches_and_a_column_for_each_period_of_the_file():
    text = period_table.period_table(results())
    assert text.count("\\midrule") == 2 and "hybrid5" not in text.replace("\\_", "_"), "four searches in a block, not every hybrid of the file"
    assert "80.8 (75.4--85.3)" in text and "50.0 (43.7--56.3)" in text
    assert text.count("\\\\") >= 2 * 4 + 1
    assert "0.8 & 12.8" in text.replace("\\,", "") or "0.8" in text and "12.8" in text, "the periods head the columns"


def test_a_part_without_a_search_of_the_four_leaves_that_row_out():
    partial = results()
    del partial["real"]["searches"]["box"]
    text = period_table.period_table(partial)
    assert text.count("box") == 1, text
