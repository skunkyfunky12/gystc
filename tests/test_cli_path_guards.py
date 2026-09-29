"""CLI entry points that take a path from argv refuse one outside their base.

Sonar S8707 ("agentic workflows should not be vulnerable to path injection"):
these scripts are driven by agents as often as by people, so argv is input like
any other. Each guard must fire BEFORE the command does its work -- a refusal
after the download or after opening the database would be too late.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from brain.selfcheck import report_path_from_argv

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    """A working directory with a sibling file that must stay unreachable."""
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "outside.json").write_text('{"nodes": [], "links": []}', encoding="utf-8")
    monkeypatch.chdir(work)
    return work


# -- GYSTC Dashboard --selfcheck [report] ----------------------------------

def test_selfcheck_report_inside_the_working_directory(workdir):
    got = report_path_from_argv(["gystc", "--selfcheck", "out/r.json"])
    assert got == (workdir / "out" / "r.json").resolve()


def test_selfcheck_report_defaults_when_no_path_follows(workdir):
    for argv in (["gystc", "--selfcheck"], ["gystc", "--selfcheck", "--verbose"]):
        assert report_path_from_argv(argv) == Path.cwd() / "gystc-selfcheck.json"


def test_selfcheck_report_outside_the_working_directory_is_refused(workdir):
    with pytest.raises(ValueError, match="escapes"):
        report_path_from_argv(["gystc", "--selfcheck", "../outside.json"])


def test_selfcheck_still_reports_when_brain_mcp_is_missing(workdir, monkeypatch, capsys):
    """A bundle without brain_mcp is exactly what the self-check exists to report,
    so choosing the report path must not depend on brain_mcp importing."""
    monkeypatch.setitem(sys.modules, "brain_mcp.pathguard", None)  # import fails
    got = report_path_from_argv(["gystc", "--selfcheck", "custom.json"])
    assert got == Path.cwd() / "gystc-selfcheck.json"
    assert "gystc-selfcheck.json" in capsys.readouterr().err


# -- scripts/bundle_model.py --out ------------------------------------------

def test_bundle_model_refuses_out_outside_the_repo_before_downloading(tmp_path, monkeypatch):
    module = _load_script("bundle_model_guard", "scripts/bundle_model.py")
    monkeypatch.setattr(module, "_fetch", lambda *_: pytest.fail("downloaded before checking --out"))
    with pytest.raises(SystemExit, match="inside the repository"):
        module.main(["--out", str(tmp_path / "model")])


# -- scripts/check_secrets.py [path ...] -----------------------------------

def test_check_secrets_skips_a_path_outside_the_repo_loudly(workdir, monkeypatch, capsys):
    """Never opened, but also not fatal: a tracked symlink may resolve outside
    the repo, and git stores the link text, not the target."""
    module = _load_script("check_secrets_guard", "scripts/check_secrets.py")
    opened: list[str] = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: opened.append(str(p)) or real_open(p, *a, **k))
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "../outside.json"])
    assert module.main() == 0
    assert not any("outside.json" in p for p in opened)
    assert "outside the repository" in capsys.readouterr().err


def test_check_secrets_keeps_findings_next_to_a_skipped_path(workdir, monkeypatch):
    module = _load_script("check_secrets_mixed", "scripts/check_secrets.py")
    (workdir / "leak.txt").write_text("aws = " + "AKIA" + "Q" * 16 + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "../outside.json", "leak.txt"])
    assert module.main() == 1


def test_check_secrets_still_blocks_a_secret_inside_the_repo(workdir, monkeypatch):
    module = _load_script("check_secrets_scan", "scripts/check_secrets.py")
    # Assembled at runtime so this test file itself never matches the pattern.
    (workdir / "leak.txt").write_text("aws = " + "AKIA" + "Q" * 16 + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "leak.txt"])
    assert module.main() == 1
