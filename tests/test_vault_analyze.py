"""Read-only vault analysis: deterministic detectors that produce a REVIEWABLE
candidate list (dead links, orphans, exact duplicates, stale). Never writes."""
from __future__ import annotations

import os
import time
from pathlib import Path

from brain_mcp.curation.analyze import (
    scan_notes, find_dead_links, find_orphans, find_exact_duplicates, find_stale, analyze,
)


def _vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    (v / "02 Projekte").mkdir(parents=True)
    (v / "A.md").write_text("# A\nlinks to [[B]] and [[Ghost]] and [[B#sec|alias]]\n", encoding="utf-8")
    (v / "B.md").write_text("# B\nno links here\n", encoding="utf-8")
    (v / "C.md").write_text("# C\nlonely orphan, nobody links me\n", encoding="utf-8")
    (v / "02 Projekte" / "D.md").write_text("# Dup\nidentical body\n", encoding="utf-8")
    (v / "02 Projekte" / "E.md").write_text("# Dup\nidentical body\n", encoding="utf-8")
    # ignored dirs must be skipped
    (v / ".obsidian").mkdir()
    (v / ".obsidian" / "x.md").write_text("secret note should be ignored", encoding="utf-8")
    return v


def test_scan_skips_ignored_dirs(tmp_path):
    notes = scan_notes(_vault(tmp_path))
    names = {n["name"] for n in notes}
    assert {"a", "b", "c", "d", "e"} <= names
    assert all(".obsidian" not in n["path"] for n in notes)


def test_find_dead_links(tmp_path):
    notes = scan_notes(_vault(tmp_path))
    dead = find_dead_links(notes)
    targets = {c["detail"]["target"] for c in dead}
    assert "Ghost" in targets          # [[Ghost]] has no note
    assert "B" not in targets          # [[B]] and [[B#sec|alias]] resolve to B.md
    assert all(c["tier"] == "yellow" for c in dead)


def test_find_orphans(tmp_path):
    notes = scan_notes(_vault(tmp_path))
    orphans = {c["file"] for c in find_orphans(notes)}
    assert "C.md" in orphans           # nobody links C
    assert "B.md" not in orphans       # A links B


def test_find_exact_duplicates(tmp_path):
    notes = scan_notes(_vault(tmp_path))
    dups = find_exact_duplicates(notes)
    assert dups, "D.md and E.md are identical"
    files = set()
    for c in dups:
        files.add(c["file"])
        files.update(c["detail"]["duplicates"])
    assert {"02 Projekte/D.md", "02 Projekte/E.md"} <= {f.replace("\\", "/") for f in files}
    assert all(c["tier"] == "red" for c in dups)  # archive one copy


def test_find_stale(tmp_path):
    v = _vault(tmp_path)
    old = v / "B.md"
    two_years_ago = time.time() - 730 * 86400
    os.utime(old, (two_years_ago, two_years_ago))
    notes = scan_notes(v)
    stale = {c["file"] for c in find_stale(notes, max_age_days=365)}
    assert "B.md" in stale
    assert "A.md" not in stale         # freshly written


def test_analyze_is_readonly_and_returns_candidates(tmp_path):
    v = _vault(tmp_path)
    before = {p: p.stat().st_mtime for p in v.rglob("*.md")}
    result = analyze(v)
    assert "candidates" in result and "stats" in result
    assert len(result["candidates"]) >= 3  # at least the dead link, orphan, dupes
    after = {p: p.stat().st_mtime for p in v.rglob("*.md")}
    assert before == after, "analyze must not modify any file"
