"""Regression tests for vault_git fixes (audit findings 8, 9):
- 8: is_repo must require the git TOPLEVEL == vault root. A vault merely nested
     inside an enclosing repo must not count — otherwise init/snapshot/secret-
     gitignore are silently skipped and `git add -A` sweeps the parent worktree.
- 8 (gitignore): the required-ignore check must match line-wise, not substring
     (a commented-out '#.obsidian/' must not count as present).
- 9: commit_run with explicit paths stages only those paths.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from brain_mcp.curation.vault_git import ensure_vault_repo, commit_run, is_repo


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, encoding="utf-8").stdout


def _init_outer_repo(tmp_path: Path) -> Path:
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "outer_file.txt").write_text("outer", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=outer, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=outer, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=outer, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=outer, capture_output=True)
    subprocess.run(["git", "commit", "-m", "outer"], cwd=outer, capture_output=True)
    return outer


def test_is_repo_rejects_vault_nested_in_enclosing_repo(tmp_path):
    outer = _init_outer_repo(tmp_path)
    vault = outer / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Hi\n", encoding="utf-8")
    assert is_repo(vault) is False     # nested != the vault's own repo


def test_ensure_vault_repo_inits_embedded_repo_when_nested(tmp_path):
    # With the toplevel check, a nested vault gets its OWN repo (snapshot +
    # secret gitignore) instead of silently riding the parent repo.
    outer = _init_outer_repo(tmp_path)
    vault = outer / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Hi\n", encoding="utf-8")
    obs = vault / ".obsidian"
    obs.mkdir()
    (obs / "secret.json").write_text('{"api_key":"SUPERSECRET"}', encoding="utf-8")

    res = ensure_vault_repo(vault, snapshot_message="snap")
    assert res["status"] == "initialized"
    assert (vault / ".git").is_dir()
    assert is_repo(vault) is True
    assert ".obsidian" not in _git(vault, "ls-files")   # secrets protected
    # nothing of the vault leaked into the OUTER repo's index
    assert "vault" not in _git(outer, "ls-files")


def test_is_repo_true_for_vault_that_is_its_own_repo(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Hi\n", encoding="utf-8")
    ensure_vault_repo(vault, snapshot_message="snap")
    assert is_repo(vault) is True


def test_commented_out_ignore_line_does_not_count_as_present(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Hi\n", encoding="utf-8")
    obs = vault / ".obsidian"
    obs.mkdir()
    (obs / "secret.json").write_text('{"api_key":"SUPERSECRET"}', encoding="utf-8")
    # substring-matching would treat these as 'already ignored'
    (vault / ".gitignore").write_text("#.obsidian/\n!.trash/\n", encoding="utf-8")
    ensure_vault_repo(vault, snapshot_message="snap")
    assert ".obsidian" not in _git(vault, "ls-files"), \
        "commented-out ignore must not satisfy the secret-protection check"


def test_commit_run_with_paths_stages_only_those(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("a\n", encoding="utf-8")
    (vault / ".gitignore").write_text(".obsidian/\n.trash/\ngraphify-out/\n*.tmp\n",
                                      encoding="utf-8")
    ensure_vault_repo(vault, snapshot_message="snap")
    (vault / "a.md").write_text("a EDIT\n", encoding="utf-8")
    (vault / "other.md").write_text("concurrent\n", encoding="utf-8")
    sha = commit_run(vault, "scoped run", paths=["a.md"])
    assert sha
    committed = _git(vault, "show", "--name-only", "--format=", "HEAD")
    assert "a.md" in committed and "other.md" not in committed
    assert "?? other.md" in _git(vault, "status", "--porcelain")


def test_commit_run_with_empty_paths_is_noop(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("a\n", encoding="utf-8")
    ensure_vault_repo(vault, snapshot_message="snap")
    (vault / "dirty.md").write_text("dirty\n", encoding="utf-8")
    assert commit_run(vault, "empty run", paths=[]) is None


def test_commit_run_stages_deletion_of_moved_tracked_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("a\n", encoding="utf-8")
    ensure_vault_repo(vault, snapshot_message="snap")
    (vault / "99 Archiv").mkdir()
    (vault / "a.md").rename(vault / "99 Archiv" / "a.md")
    sha = commit_run(vault, "archive run", paths=["a.md", "99 Archiv/a.md"])
    assert sha
    assert _git(vault, "status", "--porcelain").strip() == ""
