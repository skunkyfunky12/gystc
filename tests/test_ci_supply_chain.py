"""CI installs exactly what uv.lock pins, from wheels, with pinned actions.

Sonar S8544/S8541/S8565: before this, test.yml and build.yml ran `pip install`
against open version ranges, so every run resolved afresh and any sdist in the
tree could execute its setup.py on the runner -- the job that builds the
released binary included. These checks keep a later edit from quietly bringing
that back.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def _run_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if re.search(r"\b(pip|uv)\b", line) and not line.strip().startswith("#")]


def test_lock_file_is_committed():
    assert (ROOT / "uv.lock").is_file()


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_no_unlocked_pip_install(workflow):
    for line in _run_lines(workflow):
        assert "pip install" not in line or "pipx install ruff==" in line, (
            f"{workflow.name}: '{line}' resolves dependencies outside uv.lock"
        )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_uv_installs_and_runs_are_frozen_and_wheel_only(workflow):
    for line in _run_lines(workflow):
        if re.search(r"\buv (sync|run)\b", line):
            assert "--frozen" in line, f"{workflow.name}: '{line}' may re-resolve"
            assert "--no-build" in line, f"{workflow.name}: '{line}' may run setup.py"


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_third_party_actions_are_pinned_to_a_commit(workflow):
    for ref in re.findall(r"uses:\s*([^\s#]+)", workflow.read_text(encoding="utf-8")):
        if ref.startswith("./"):
            continue  # a workflow in this repository
        assert re.search(r"@[0-9a-f]{40}$", ref), f"{workflow.name}: {ref} is not a commit SHA"
