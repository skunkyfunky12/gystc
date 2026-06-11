"""Regression tests for the curate CLI gate (audit findings 11, 4-cli-side):
- 11: the human gate must SHOW what it approves — unified diff for edits, full
      target list for archives, content for creates; --yes prints the same preview.
- 4:  fingerprints from the analyze proposal are honored; stale skips exit nonzero.
Follow-ups (diff-review findings 3 + 10): replaying an applied proposal exits
nonzero with a distinct skip reason instead of crashing, and the whole module
runs hermetically (BRAIN_DATA_DIR) instead of reading the real ~/.gystc.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain_mcp.curation.cli import main


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Finding 10: cmd_analyze reads load_config().exclude_dirs — without
    BRAIN_DATA_DIR that falls back to (and mkdirs!) the developer's real
    ~/.gystc, so ambient config would silently change what these tests
    exercise. Same convention as test_fix_cli_vector.py's cli_env fixture."""
    monkeypatch.setenv("BRAIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("BRAIN_VAULT_PATH", raising=False)


def _vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "p.md").write_text("# P\nold line\n", encoding="utf-8")
    (v / ".gitignore").write_text(".obsidian/\n", encoding="utf-8")
    main(["init", "--vault", str(v)])
    return v


def _write_actions(tmp_path: Path, payload: dict) -> Path:
    prop = tmp_path / "actions.json"
    prop.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return prop


def test_dry_run_shows_unified_diff_for_edits(tmp_path, capsys):
    v = _vault(tmp_path)
    prop = _write_actions(tmp_path, {"actions": [
        {"op": "edit", "file": "p.md", "new_content": "# P\nnew line\n"},
    ]})
    main(["apply", "--vault", str(v), "--proposals", str(prop)])
    out = capsys.readouterr().out
    assert "-old line" in out and "+new line" in out   # an actual diff, not just 'edit p.md'
    assert "old line" in (v / "p.md").read_text(encoding="utf-8")  # dry run wrote nothing


def test_dry_run_lists_archive_directory_contents(tmp_path, capsys):
    v = _vault(tmp_path)
    (v / "proj").mkdir()
    (v / "proj" / "inner1.md").write_text("x", encoding="utf-8")
    (v / "proj" / "inner2.md").write_text("y", encoding="utf-8")
    prop = _write_actions(tmp_path, {"actions": [{"op": "archive", "file": "proj"}]})
    main(["apply", "--vault", str(v), "--proposals", str(prop)])
    out = capsys.readouterr().out
    assert "inner1.md" in out and "inner2.md" in out   # full target list, not a blind one-liner
    assert (v / "proj").exists()


def test_dry_run_shows_create_content(tmp_path, capsys):
    v = _vault(tmp_path)
    prop = _write_actions(tmp_path, {"actions": [
        {"op": "create", "file": "new.md", "new_content": "# New\nCREATED BODY\n"},
    ]})
    main(["apply", "--vault", str(v), "--proposals", str(prop)])
    assert "CREATED BODY" in capsys.readouterr().out


def test_yes_prints_same_preview_before_applying(tmp_path, capsys):
    v = _vault(tmp_path)
    prop = _write_actions(tmp_path, {"actions": [
        {"op": "edit", "file": "p.md", "new_content": "# P\nnew line\n"},
    ]})
    main(["apply", "--vault", str(v), "--proposals", str(prop), "--yes"])
    out = capsys.readouterr().out
    assert "-old line" in out and "+new line" in out   # preview shown on the real run too
    assert "new line" in (v / "p.md").read_text(encoding="utf-8")


def test_apply_honors_proposal_fingerprints_and_exits_nonzero_on_stale(tmp_path, capsys):
    v = _vault(tmp_path)
    # analyze writes the proposal WITH fingerprints
    proposal = tmp_path / "proposal.json"
    main(["analyze", "--vault", str(v), "--out", str(proposal)])
    data = json.loads(proposal.read_text(encoding="utf-8"))
    assert data["fingerprints"]["p.md"]
    actions = {"actions": [{"op": "edit", "file": "p.md", "new_content": "# P\nrewrite\n"}],
               "fingerprints": data["fingerprints"]}
    prop = _write_actions(tmp_path, actions)
    # concurrent edit in the review window
    (v / "p.md").write_text("# P\nCONCURRENT\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["apply", "--vault", str(v), "--proposals", str(prop), "--yes"])
    assert exc.value.code != 0                          # nonzero summary
    assert "CONCURRENT" in (v / "p.md").read_text(encoding="utf-8")  # not clobbered
    assert "SKIPPED" in capsys.readouterr().err


def test_analyze_reads_isolated_config_not_real_home(tmp_path):
    # Finding 10 pin: exclude_dirs must come from THIS test's config.json (via
    # BRAIN_DATA_DIR), proving the CLI never consults the developer's ~/.gystc.
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(
        json.dumps({"exclude_dirs": ["Skip Me"]}), encoding="utf-8")
    v = _vault(tmp_path)
    (v / "Skip Me").mkdir()
    (v / "Skip Me" / "hidden.md").write_text("# Hidden\n", encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    main(["analyze", "--vault", str(v), "--out", str(proposal)])
    fp = json.loads(proposal.read_text(encoding="utf-8"))["fingerprints"]
    assert "p.md" in fp                                   # normal note scanned
    assert not any(k.startswith("Skip Me/") for k in fp)  # isolated exclude honored


def test_apply_replay_skips_cleanly_and_exits_nonzero(tmp_path, capsys):
    """Finding 3 pin: retrying `curate apply` with the SAME proposal after it
    already applied must not crash with FileNotFoundError — replayed ops are
    reported as skips with a distinct reason, and the rerun still exits nonzero
    (honest: this run did not fully apply the proposal)."""
    v = _vault(tmp_path)
    prop = _write_actions(tmp_path, {"actions": [{"op": "archive", "file": "p.md"}]})
    main(["apply", "--vault", str(v), "--proposals", str(prop), "--yes"])  # applies
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:                # no uncaught traceback
        main(["apply", "--vault", str(v), "--proposals", str(prop), "--yes"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "already archived" in err                      # distinct replay reason
    assert "changed since analyze" not in err             # not mislabeled as stale
