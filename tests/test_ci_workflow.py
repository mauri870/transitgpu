"""The workflow of continuous integration (.github/workflows/ci.yml): a lint of the YAML, and of what it must hold. It is a file for GitHub and is not run here; this test reads it.

A structure check on a text file, which a data archive of the package's own format does not hold."""
from pathlib import Path

import pytest

import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
if not WORKFLOW.exists():
    pytest.skip("the workflow is not in the source distribution", allow_module_level=True)


def load():
    text = WORKFLOW.read_text()
    assert "\t" not in text, "a tab in a YAML file"
    assert text.endswith("\n") and not any(line != line.rstrip() for line in text.splitlines()), "a trailing space or no final newline"
    return yaml.safe_load(text)


def test_the_workflow_is_valid_yaml_with_a_trigger_a_job_and_steps_that_use_or_run_something():
    workflow = load()
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads the key on as True
    assert triggers and {"push", "pull_request"} <= set(triggers)
    steps = workflow["jobs"]["test"]["steps"]
    assert steps and all(("uses" in step) != ("run" in step) for step in steps), "each step uses an action or runs a command, not both and not neither"
    assert workflow["jobs"]["test"]["runs-on"] == "ubuntu-latest"


def test_the_workflow_installs_the_cpu_build_of_torch_with_uv_runs_the_tests_and_no_example_and_asks_for_no_device():
    text = WORKFLOW.read_text()
    runs = [step["run"] for step in load()["jobs"]["test"]["steps"] if "run" in step]
    assert any(run == "uv sync --group cpu" for run in runs)
    assert any(run.startswith("uv run --group cpu pytest tests") for run in runs)
    assert not any("examples/" in run for run in runs), "the examples search on a GPU and the runner has none"
    assert "pip" not in text and "whl/" not in text, "uv installs it, from the index the cpu group names"
    assert "--group cu" not in text and "--group rocm" not in text and "cuda" not in text.lower() and "rocm" not in text.lower(), "no wheel or runner of a device"
    assert "continue-on-error" not in text, "a step that fails fails the job"


def test_the_actions_are_pinned_to_a_major_version_uv_has_a_version_and_the_job_is_bounded_and_read_only():
    workflow = load()
    steps = [step for step in workflow["jobs"]["test"]["steps"] if "uses" in step]
    assert steps and all("@v" in step["uses"] for step in steps), steps
    setup = [step for step in steps if step["uses"].startswith("astral-sh/setup-uv@")]
    assert len(setup) == 1 and setup[0]["with"]["version"] == "0.11.2" and setup[0]["with"]["enable-cache"] is True
    assert workflow["permissions"] == {"contents": "read"} and workflow["jobs"]["test"]["timeout-minutes"] <= 60
