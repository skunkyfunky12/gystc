"""Read-only vault analysis. Deterministic detectors that surface a reviewable
candidate list. NEVER writes — judgment + apply happen elsewhere, human-gated.

Semantic near-duplicate detection (via the shared daemon's brain_retrieve) is an
optional add-on; these deterministic detectors work without a daemon."""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

# Skip secrets, trash, derived artifacts, git internals, and already-archived notes.
_SKIP_DIRS = {".git", ".obsidian", ".trash", "graphify-out", "99 Archiv"}
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def _link_target(raw: str) -> str:
    # [[Target#heading|alias]] -> "Target"
    return raw.split("|", 1)[0].split("#", 1)[0].strip()


def scan_notes(vault: Path) -> list[dict]:
    vault = Path(vault)
    notes: list[dict] = []
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        links = {t for t in (_link_target(m) for m in _WIKILINK.findall(content)) if t}
        notes.append({
            "path": rel.as_posix(),
            "name": p.stem.lower(),
            "content": content,
            "mtime": p.stat().st_mtime,
            "links": links,
            "word_count": len(content.split()),
        })
    return notes


def _basename(target: str) -> str:
    return target.split("/")[-1].lower()


def find_dead_links(notes: list[dict]) -> list[dict]:
    known = {n["name"] for n in notes}
    out: list[dict] = []
    for n in notes:
        for target in sorted(n["links"]):
            if _basename(target) not in known:
                out.append({
                    "file": n["path"], "kind": "dead_link",
                    "problem": f"Wikilink [[{target}]] resolves to no note",
                    "action": "fix or remove the link",
                    "tier": "yellow", "detail": {"target": target},
                })
    return out


def find_orphans(notes: list[dict]) -> list[dict]:
    inbound = {n["name"]: 0 for n in notes}
    for n in notes:
        for target in n["links"]:
            base = _basename(target)
            if base in inbound and base != n["name"]:
                inbound[base] += 1
    return [
        {"file": n["path"], "kind": "orphan",
         "problem": "No other note links here (orphan)",
         "action": "review — link it in, or leave if intentional",
         "tier": "green", "detail": {}}
        for n in notes if inbound.get(n["name"], 0) == 0
    ]


def find_exact_duplicates(notes: list[dict]) -> list[dict]:
    by_hash: dict[str, list[dict]] = {}
    for n in notes:
        h = hashlib.sha256(n["content"].strip().encode("utf-8")).hexdigest()
        by_hash.setdefault(h, []).append(n)
    out: list[dict] = []
    for group in by_hash.values():
        if len(group) > 1:
            ordered = sorted(group, key=lambda x: x["path"])
            keep, rest = ordered[0], ordered[1:]
            out.append({
                "file": keep["path"], "kind": "exact_duplicate",
                "problem": f"{len(ordered)} notes have identical content",
                "action": "keep one, archive the others",
                "tier": "red",
                "detail": {"duplicates": [g["path"] for g in rest],
                           "all": [g["path"] for g in ordered]},
            })
    return out


def find_stale(notes: list[dict], max_age_days: float = 365.0, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    cutoff = max_age_days * 86400
    return [
        {"file": n["path"], "kind": "stale",
         "problem": f"Not modified in {int((now - n['mtime']) / 86400)} days",
         "action": "review for outdated facts / archive",
         "tier": "green", "detail": {"age_days": int((now - n["mtime"]) / 86400)}}
        for n in notes if (now - n["mtime"]) > cutoff
    ]


def analyze(vault: Path, *, max_age_days: float = 365.0) -> dict:
    """Run all read-only detectors. Returns {candidates, stats}. Writes nothing."""
    notes = scan_notes(vault)
    candidates = (
        find_dead_links(notes)
        + find_exact_duplicates(notes)
        + find_orphans(notes)
        + find_stale(notes, max_age_days=max_age_days)
    )
    kinds = ("dead_link", "exact_duplicate", "orphan", "stale")
    stats = {"notes_scanned": len(notes)}
    stats.update({k: sum(1 for c in candidates if c["kind"] == k) for k in kinds})
    return {"candidates": candidates, "stats": stats}
