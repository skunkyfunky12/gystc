"""Regression tests for curation analyze fixes (audit findings 26, 27, 4-analyze-side):
- 26: unreadable notes are logged loudly and EXCLUDED from dead-link/removal proposals
      (read failure is not evidence of a dead note).
- 27: a note deleted between rglob and stat must not crash analyze.
- 4:  analyze emits per-note content fingerprints so apply can detect staleness.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from brain_mcp.curation.analyze import (
    scan_notes, find_dead_links, find_orphans, find_exact_duplicates, find_stale, analyze,
)


def _vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "A.md").write_text("# A\nlinks to [[Locked]]\n", encoding="utf-8")
    (v / "Locked.md").write_text("# Locked\nexists but unreadable\n", encoding="utf-8")
    return v


def _make_unreadable(monkeypatch, name: str):
    """Simulate an OS-level read failure (lock/permission) for one file."""
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def fake_read_bytes(self):
        if self.name == name:
            raise PermissionError(f"locked: {self}")
        return real_read_bytes(self)

    def fake_read_text(self, *a, **kw):
        if self.name == name:
            raise PermissionError(f"locked: {self}")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(Path, "read_text", fake_read_text)


def test_unreadable_note_is_logged_to_stderr(tmp_path, monkeypatch, capsys):
    v = _vault(tmp_path)
    _make_unreadable(monkeypatch, "Locked.md")
    scan_notes(v)
    err = capsys.readouterr().err
    assert "Locked.md" in err and "WARNING" in err  # no silent skip


def test_unreadable_note_does_not_become_dead_link(tmp_path, monkeypatch):
    # Finding 26: A links [[Locked]]; Locked.md EXISTS but cannot be read.
    # The link must NOT be proposed as dead (read failure is not evidence).
    v = _vault(tmp_path)
    _make_unreadable(monkeypatch, "Locked.md")
    notes = scan_notes(v)
    dead = find_dead_links(notes)
    targets = {c["detail"]["target"] for c in dead}
    assert "Locked" not in targets


def test_unreadable_note_excluded_from_removal_proposals(tmp_path, monkeypatch):
    v = _vault(tmp_path)
    _make_unreadable(monkeypatch, "Locked.md")
    notes = scan_notes(v)
    flagged = {c["file"] for c in
               find_orphans(notes) + find_exact_duplicates(notes) + find_stale(notes)}
    assert "Locked.md" not in flagged


def test_analyze_stats_carry_read_errors(tmp_path, monkeypatch):
    v = _vault(tmp_path)
    _make_unreadable(monkeypatch, "Locked.md")
    result = analyze(v)
    assert result["stats"]["read_errors"] == 1
    assert result["stats"]["notes_scanned"] == 1  # only the readable note


def test_note_deleted_between_rglob_and_stat_does_not_crash(tmp_path, monkeypatch):
    # Finding 27: stat() raced against deletion must be survivable.
    v = _vault(tmp_path)
    real_stat = Path.stat

    def fake_stat(self, *a, **kw):
        if self.name == "Locked.md":
            raise FileNotFoundError(str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", fake_stat)
    notes = scan_notes(v)  # must not raise
    assert any(n["path"] == "A.md" for n in notes)


def test_analyze_emits_fingerprints_for_apply(tmp_path):
    # Finding 4 (analyze side): the plan must carry a content fingerprint per note.
    v = _vault(tmp_path)
    result = analyze(v)
    fp = result["fingerprints"]
    expected = hashlib.sha256((v / "A.md").read_bytes()).hexdigest()
    assert fp["A.md"] == expected
    assert "Locked.md" in fp


def test_extra_skip_dirs_merged(tmp_path):
    # Finding 26 (scope): analyze honors configured exclude_dirs, not only _SKIP_DIRS.
    v = _vault(tmp_path)
    (v / "Excluded").mkdir()
    (v / "Excluded" / "X.md").write_text("# X\n", encoding="utf-8")
    notes = scan_notes(v, extra_skip_dirs=["Excluded"])
    assert all(not n["path"].startswith("Excluded/") for n in notes)
