"""The write/apply layer — the only part that mutates the vault. Hard rails:
refuses without a git repo, archives instead of hard-deleting, blocks path
escapes, never clobbers, and commits exactly one revertable snapshot per run."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from brain_mcp.curation.vault_git import ensure_vault_repo
from brain_mcp.curation.apply import (
    apply_archive, apply_edit, apply_create, apply_actions, unified_diff,
)


def _repo_vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    (v / "02 Projekte").mkdir(parents=True)
    (v / "old.md").write_text("# Old\nstale\n", encoding="utf-8")
    (v / "02 Projekte" / "p.md").write_text("# P\nbody\n", encoding="utf-8")
    (v / ".gitignore").write_text(".obsidian/\n", encoding="utf-8")
    ensure_vault_repo(v, snapshot_message="snap")
    return v


def test_archive_moves_not_deletes(tmp_path):
    v = _repo_vault(tmp_path)
    dest = apply_archive(v, "old.md")
    assert not (v / "old.md").exists()                 # original gone
    assert dest.exists() and dest.read_text(encoding="utf-8").startswith("# Old")
    assert "99 Archiv" in dest.as_posix()              # archived, not deleted


def test_archive_refuses_without_git(tmp_path):
    v = tmp_path / "novault"
    (v).mkdir()
    (v / "n.md").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):
        apply_archive(v, "n.md")                       # fail-closed: reversibility required


def test_write_refuses_path_escape(tmp_path):
    v = _repo_vault(tmp_path)
    with pytest.raises(ValueError):
        apply_archive(v, "../escape.md")


def test_apply_edit_changes_content(tmp_path):
    v = _repo_vault(tmp_path)
    apply_edit(v, "02 Projekte/p.md", "# P\nUPDATED\n")
    assert "UPDATED" in (v / "02 Projekte" / "p.md").read_text(encoding="utf-8")


def test_apply_create_new_note_and_no_clobber(tmp_path):
    v = _repo_vault(tmp_path)
    apply_create(v, "05 Referenzen/new.md", "# New\nfrom reconcile\n")
    assert (v / "05 Referenzen" / "new.md").exists()
    with pytest.raises(FileExistsError):
        apply_create(v, "05 Referenzen/new.md", "different")  # never clobber


def test_apply_actions_commits_once(tmp_path):
    v = _repo_vault(tmp_path)

    def _count_commits() -> int:
        out = subprocess.run(["git", "-C", str(v), "rev-list", "--count", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        return int(out)

    before = _count_commits()
    res = apply_actions(v, [
        {"op": "archive", "file": "old.md"},
        {"op": "edit", "file": "02 Projekte/p.md", "new_content": "# P\nEDIT\n"},
    ], run_message="curation run 1")
    assert res["applied"] == 2 and res["commit"]
    assert _count_commits() == before + 1              # exactly one commit for the run


def test_unified_diff_shows_change(tmp_path):
    d = unified_diff("a\nb\n", "a\nB\n", "p.md")
    assert "-b" in d and "+B" in d
