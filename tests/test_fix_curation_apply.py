"""Regression tests for curation apply fixes (audit findings 4, 9, 18, 30):
- 4:  stale 'edit'/'archive' actions (note changed since analyze) are SKIPPED loudly
      instead of clobbering concurrent edits.
- 9:  the run commit stages ONLY the files the run touched — unrelated concurrent
      vault changes stay out, so `git revert <sha>` is surgical.
- 18: a mid-run failure never leaves silent uncommitted half-state: the partial
      mutations are committed as a clearly-marked PARTIAL commit before re-raising.
- 30: apply_archive collision-rename and re-archive edge cases.
Follow-up (diff-review finding 3): re-running a partially-applied proposal must
not crash — missing-source / already-archived / already-created ops become
recorded skips with a reason distinct from stale.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from brain_mcp.curation.vault_git import ensure_vault_repo
from brain_mcp.curation.apply import apply_archive, apply_actions


def _git(vault: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True, encoding="utf-8").stdout


def _repo_vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "old.md").write_text("# Old\nstale\n", encoding="utf-8")
    (v / "p.md").write_text("# P\nbody\n", encoding="utf-8")
    (v / ".gitignore").write_text(".obsidian/\n", encoding="utf-8")
    ensure_vault_repo(v, snapshot_message="snap")
    return v


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- finding 4

def test_stale_edit_is_skipped_and_concurrent_edit_preserved(tmp_path, capsys):
    v = _repo_vault(tmp_path)
    base_hash = _sha(v / "p.md")                       # fingerprint at "analyze" time
    (v / "p.md").write_text("# P\nCONCURRENT EDIT\n", encoding="utf-8")  # review window
    res = apply_actions(v, [
        {"op": "edit", "file": "p.md", "new_content": "# P\nSTALE REWRITE\n",
         "base_hash": base_hash},
    ], run_message="run")
    assert res["applied"] == 0
    assert len(res["skipped"]) == 1 and res["skipped"][0]["file"] == "p.md"
    # the concurrent edit survived — NOT clobbered
    assert "CONCURRENT EDIT" in (v / "p.md").read_text(encoding="utf-8")
    assert "SKIPPED" in capsys.readouterr().err       # loud per-action message


def test_matching_fingerprint_applies_normally(tmp_path):
    v = _repo_vault(tmp_path)
    res = apply_actions(v, [
        {"op": "edit", "file": "p.md", "new_content": "# P\nOK\n",
         "base_hash": _sha(v / "p.md")},
    ], run_message="run")
    assert res["applied"] == 1 and res["skipped"] == []
    assert "OK" in (v / "p.md").read_text(encoding="utf-8")


def test_stale_archive_is_skipped(tmp_path, capsys):
    v = _repo_vault(tmp_path)
    base_hash = _sha(v / "old.md")
    (v / "old.md").write_text("# Old\nREWRITTEN MEANWHILE\n", encoding="utf-8")
    res = apply_actions(v, [
        {"op": "archive", "file": "old.md", "base_hash": base_hash},
    ], run_message="run")
    assert res["applied"] == 0 and len(res["skipped"]) == 1
    assert (v / "old.md").exists()                     # not moved
    assert "SKIPPED" in capsys.readouterr().err


# ---------------------------------------------------------------- finding 9

def test_run_commit_stages_only_touched_files(tmp_path):
    v = _repo_vault(tmp_path)
    # concurrent writer drops an unrelated note between curation commits
    (v / "unrelated.md").write_text("# Unrelated\nuser data\n", encoding="utf-8")
    res = apply_actions(v, [{"op": "archive", "file": "old.md"}], run_message="run")
    assert res["commit"]
    committed = _git(v, "show", "--name-only", "--format=", "HEAD")
    assert "unrelated.md" not in committed             # NOT swept into the run commit
    assert "old.md" in committed
    # still present + untracked, so the advertised `git revert` is surgical
    assert (v / "unrelated.md").exists()
    assert "?? unrelated.md" in _git(v, "status", "--porcelain")


def test_revert_of_run_commit_leaves_unrelated_edit_alone(tmp_path):
    v = _repo_vault(tmp_path)
    (v / "p.md").write_text("# P\nuser edit\n", encoding="utf-8")   # concurrent edit
    res = apply_actions(v, [{"op": "archive", "file": "old.md"}], run_message="run")
    subprocess.run(["git", "-C", str(v), "revert", "--no-edit", res["commit"]],
                   capture_output=True, text=True)
    assert (v / "old.md").exists()                                  # archive reverted
    assert "user edit" in (v / "p.md").read_text(encoding="utf-8")  # edit untouched


# ---------------------------------------------------------------- finding 18

def test_mid_run_failure_commits_partial_state(tmp_path, capsys):
    v = _repo_vault(tmp_path)
    before = _git(v, "rev-list", "--count", "HEAD").strip()
    # trigger: unknown op — a missing file is now a recorded skip (finding 3
    # follow-up), so the hard mid-run failure pin needs a real hard error
    with pytest.raises(ValueError):
        apply_actions(v, [
            {"op": "archive", "file": "old.md"},                   # succeeds
            {"op": "nope", "file": "x.md"},                        # raises
        ], run_message="run 1")
    after = _git(v, "rev-list", "--count", "HEAD").strip()
    assert int(after) == int(before) + 1               # partial state IS committed
    assert "PARTIAL" in _git(v, "log", "-1", "--format=%s")
    # the touched paths are clean — no silent uncommitted half-state
    status = _git(v, "status", "--porcelain")
    assert "old.md" not in status
    assert "ERROR" in capsys.readouterr().err          # surfaced, not silent


def test_mid_run_failure_with_nothing_applied_commits_nothing(tmp_path):
    v = _repo_vault(tmp_path)
    before = _git(v, "rev-list", "--count", "HEAD").strip()
    with pytest.raises(ValueError):
        apply_actions(v, [{"op": "nope", "file": "x.md"}], run_message="run")
    assert _git(v, "rev-list", "--count", "HEAD").strip() == before


def test_next_run_commit_not_polluted_by_failed_run(tmp_path):
    # Finding 18's downstream harm: run N's revert must not drag failed run N-1 along.
    v = _repo_vault(tmp_path)
    with pytest.raises(ValueError):
        apply_actions(v, [
            {"op": "archive", "file": "old.md"},
            {"op": "nope", "file": "x.md"},
        ], run_message="failed run")
    res = apply_actions(v, [
        {"op": "edit", "file": "p.md", "new_content": "# P\nrun2\n"},
    ], run_message="run 2")
    committed = _git(v, "show", "--name-only", "--format=", res["commit"])
    assert "old.md" not in committed and "99 Archiv" not in committed


# ------------------------------------- follow-up finding 3: replayed proposals

def test_replayed_archive_is_recorded_skip_not_crash(tmp_path, capsys):
    """Re-running the SAME proposal after a (partial) apply must not crash with
    an uncaught FileNotFoundError: the already-archived op becomes a recorded
    skip whose reason is distinct from a stale skip."""
    v = _repo_vault(tmp_path)
    actions = [{"op": "archive", "file": "old.md", "base_hash": _sha(v / "old.md")}]
    assert apply_actions(v, actions, run_message="run")["applied"] == 1
    res = apply_actions(v, actions, run_message="run retry")   # replay: must not raise
    assert res["applied"] == 0 and len(res["skipped"]) == 1
    reason = res["skipped"][0]["reason"]
    assert "already archived" in reason                        # distinct reason
    assert "changed since analyze" not in reason               # NOT lumped with stale
    assert "SKIPPED" in capsys.readouterr().err                # loud, not silent


def test_replayed_create_is_recorded_skip_and_never_clobbers(tmp_path, capsys):
    v = _repo_vault(tmp_path)
    actions = [{"op": "create", "file": "new.md", "new_content": "# New\nv1\n"}]
    assert apply_actions(v, actions, run_message="run")["applied"] == 1
    res = apply_actions(v, actions, run_message="run retry")
    assert res["applied"] == 0 and len(res["skipped"]) == 1
    assert "already exists" in res["skipped"][0]["reason"]
    assert (v / "new.md").read_text(encoding="utf-8") == "# New\nv1\n"  # untouched
    assert "SKIPPED" in capsys.readouterr().err


def test_archive_missing_source_without_archived_copy_is_recorded_skip(tmp_path, capsys):
    v = _repo_vault(tmp_path)
    res = apply_actions(v, [{"op": "archive", "file": "ghost.md"}], run_message="run")
    assert res["applied"] == 0 and len(res["skipped"]) == 1
    assert "missing" in res["skipped"][0]["reason"]
    assert "SKIPPED" in capsys.readouterr().err


def test_edit_of_missing_note_is_recorded_skip_not_crash(tmp_path, capsys):
    v = _repo_vault(tmp_path)
    res = apply_actions(v, [{"op": "edit", "file": "missing.md", "new_content": "x"}],
                        run_message="run")
    assert res["applied"] == 0 and len(res["skipped"]) == 1
    assert "missing" in res["skipped"][0]["reason"]
    assert "SKIPPED" in capsys.readouterr().err


def test_replay_mixed_with_new_action_applies_new_without_partial_commit(tmp_path):
    """'PARTIAL (failed mid-run)' stays reserved for real mid-run failures — a
    replayed op next to a fresh one must yield one normal commit."""
    v = _repo_vault(tmp_path)
    apply_actions(v, [{"op": "archive", "file": "old.md"}], run_message="run 1")
    res = apply_actions(v, [
        {"op": "archive", "file": "old.md"},                         # replay -> skip
        {"op": "edit", "file": "p.md", "new_content": "# P\nv2\n"},  # fresh -> applied
    ], run_message="run 2")
    assert res["applied"] == 1 and len(res["skipped"]) == 1
    assert res["commit"]
    assert "PARTIAL" not in _git(v, "log", "-1", "--format=%s")
    assert "v2" in (v / "p.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- finding 30

def test_archive_same_name_twice_keeps_both_copies(tmp_path):
    v = _repo_vault(tmp_path)
    first = apply_archive(v, "old.md")
    (v / "old.md").write_text("# Old\nsecond life\n", encoding="utf-8")  # recreated
    second = apply_archive(v, "old.md")
    assert first.exists() and second.exists()
    assert first != second                              # collision-renamed, no overwrite
    assert "stale" in first.read_text(encoding="utf-8")
    assert "second life" in second.read_text(encoding="utf-8")


def test_archive_same_directory_twice_keeps_both(tmp_path):
    v = _repo_vault(tmp_path)
    (v / "proj").mkdir()
    (v / "proj" / "a.md").write_text("v1", encoding="utf-8")
    first = apply_archive(v, "proj")
    (v / "proj").mkdir()
    (v / "proj" / "a.md").write_text("v2", encoding="utf-8")
    second = apply_archive(v, "proj")
    assert first != second
    assert (first / "a.md").read_text(encoding="utf-8") == "v1"
    assert (second / "a.md").read_text(encoding="utf-8") == "v2"


def test_archive_dotted_directory_name_collision_terminates(tmp_path):
    v = _repo_vault(tmp_path)
    (v / "v1.2").mkdir()
    (v / "v1.2" / "a.md").write_text("one", encoding="utf-8")
    first = apply_archive(v, "v1.2")
    (v / "v1.2").mkdir()
    (v / "v1.2" / "a.md").write_text("two", encoding="utf-8")
    second = apply_archive(v, "v1.2")
    assert first.exists() and second.exists() and first != second
    assert (second / "a.md").read_text(encoding="utf-8") == "two"


def test_archive_refuses_paths_already_under_archive(tmp_path):
    # Guard: re-archiving '99 Archiv' (or anything inside) would nest archives /
    # move the archive into its own subtree.
    v = _repo_vault(tmp_path)
    apply_archive(v, "old.md")                          # creates 99 Archiv/old.md
    with pytest.raises(ValueError):
        apply_archive(v, "99 Archiv/old.md")
    with pytest.raises(ValueError):
        apply_archive(v, "99 Archiv")
