"""Step 0 of vault curation: the vault is NOT in git (~1890 irreplaceable notes),
so reversibility is the prerequisite for ANY change. These tests pin the safety
rails: a local-only snapshot repo, secrets (.obsidian) never tracked, no remote,
idempotent, and one revertable commit per curation run.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from brain_mcp.curation.vault_git import ensure_vault_repo, commit_run, head_sha


def _git(vault: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True, encoding="utf-8").stdout


def _make_vault(tmp_path: Path, *, gitignore: bool = True) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Hi\nbody [[other]]\n", encoding="utf-8")
    obs = vault / ".obsidian"
    obs.mkdir()
    (obs / "secret.json").write_text('{"api_key":"SUPERSECRET"}', encoding="utf-8")
    if gitignore:
        (vault / ".gitignore").write_text(".obsidian/\n.trash/\ngraphify-out/\n*.tmp\n", encoding="utf-8")
    return vault


def test_ensure_creates_snapshot_repo(tmp_path):
    vault = _make_vault(tmp_path)
    res = ensure_vault_repo(vault, snapshot_message="snap")
    assert res["status"] == "initialized" and res["commit"]
    assert (vault / ".git").is_dir()
    assert "note.md" in _git(vault, "ls-files")
    log = _git(vault, "log", "--oneline")
    assert log.count("\n") == 1  # exactly one initial commit


def test_obsidian_secrets_never_tracked(tmp_path):
    vault = _make_vault(tmp_path)
    ensure_vault_repo(vault, snapshot_message="snap")
    tracked = _git(vault, "ls-files")
    assert ".obsidian" not in tracked, "secrets (.obsidian) must never be committed"


def test_creates_gitignore_when_missing_to_protect_secrets(tmp_path):
    # No .gitignore present -> the tool MUST create one covering .obsidian before add -A.
    vault = _make_vault(tmp_path, gitignore=False)
    ensure_vault_repo(vault, snapshot_message="snap")
    assert (vault / ".gitignore").exists()
    assert ".obsidian" not in _git(vault, "ls-files")


def test_idempotent(tmp_path):
    vault = _make_vault(tmp_path)
    ensure_vault_repo(vault, snapshot_message="snap")
    res2 = ensure_vault_repo(vault, snapshot_message="snap")
    assert res2["status"] == "already"


def test_never_adds_a_remote(tmp_path):
    vault = _make_vault(tmp_path)
    ensure_vault_repo(vault, snapshot_message="snap")
    assert _git(vault, "remote").strip() == "", "vault repo must be local-only (no remote)"


def test_commit_run_commits_changes_and_is_revertable(tmp_path):
    vault = _make_vault(tmp_path)
    ensure_vault_repo(vault, snapshot_message="snap")
    (vault / "note.md").write_text("# Hi\nEDITED\n", encoding="utf-8")
    sha = commit_run(vault, "curation run 1")
    assert sha and sha == head_sha(vault)
    # revert restores the pre-edit content
    subprocess.run(["git", "-C", str(vault), "revert", "--no-edit", sha],
                   capture_output=True, text=True)
    assert "EDITED" not in (vault / "note.md").read_text(encoding="utf-8")


def test_commit_run_noop_when_clean(tmp_path):
    vault = _make_vault(tmp_path)
    ensure_vault_repo(vault, snapshot_message="snap")
    assert commit_run(vault, "nothing changed") is None
