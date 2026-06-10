"""Read-only vault analysis. Deterministic detectors that surface a reviewable
candidate list. NEVER writes — judgment + apply happen elsewhere, human-gated.

Semantic near-duplicate detection (via the shared daemon's brain_retrieve) is an
optional add-on; these deterministic detectors work without a daemon."""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# Skip secrets, trash, derived artifacts, git internals, and already-archived notes.
_SKIP_DIRS = {".git", ".obsidian", ".trash", "graphify-out", "99 Archiv"}
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def _link_target(raw: str) -> str:
    # [[Target#heading|alias]] -> "Target"
    return raw.split("|", 1)[0].split("#", 1)[0].strip()


def scan_notes(vault: Path, extra_skip_dirs: Iterable[str] = ()) -> list[dict]:
    vault = Path(vault)
    skip_dirs = _SKIP_DIRS | set(extra_skip_dirs)  # honor configured exclude_dirs too
    notes: list[dict] = []
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        if any(part in skip_dirs for part in rel.parts):
            continue
        try:
            st = p.stat()  # inside the try: the note can vanish between rglob and here
            raw = p.read_bytes()
        except OSError as e:
            # Loud + conservative: a read failure is NOT evidence the note is gone.
            # Keep its name in the universe (unreadable stub) so links pointing at
            # it are never proposed as dead, and exclude it from removal proposals.
            print(f"WARNING: Cannot read {p}: {e}", file=sys.stderr)
            notes.append({
                "path": rel.as_posix(), "name": p.stem.lower(), "content": "",
                "mtime": 0.0, "links": set(), "word_count": 0,
                "sha256": None, "unreadable": True,
            })
            continue
        content = raw.decode("utf-8", errors="replace")
        links = {t for t in (_link_target(m) for m in _WIKILINK.findall(content)) if t}
        notes.append({
            "path": rel.as_posix(),
            "name": p.stem.lower(),
            "content": content,
            "mtime": st.st_mtime,
            "links": links,
            "word_count": len(content.split()),
            # content fingerprint: lets apply detect notes changed since analyze
            "sha256": hashlib.sha256(raw).hexdigest(),
            "unreadable": False,
        })
    return notes


def _basename(target: str) -> str:
    return target.split("/")[-1].lower()


def find_dead_links(notes: list[dict]) -> list[dict]:
    known = {n["name"] for n in notes}  # includes unreadable stubs: their links are NOT dead
    out: list[dict] = []
    for n in notes:
        if n.get("unreadable"):
            continue
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
        # unreadable notes: their inbound/outbound state is unknown — never propose
        for n in notes if inbound.get(n["name"], 0) == 0 and not n.get("unreadable")
    ]


def find_exact_duplicates(notes: list[dict]) -> list[dict]:
    by_hash: dict[str, list[dict]] = {}
    for n in notes:
        if n.get("unreadable"):  # content unknown — cannot be evidence of a dupe
            continue
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
        for n in notes if (now - n["mtime"]) > cutoff and not n.get("unreadable")
    ]


def analyze(vault: Path, *, max_age_days: float = 365.0,
            extra_skip_dirs: Iterable[str] = ()) -> dict:
    """Run all read-only detectors. Returns {candidates, stats, fingerprints}.
    Writes nothing. `fingerprints` (path -> sha256 of analyzed bytes) lets the
    apply step refuse actions whose note changed in the review window."""
    notes = scan_notes(vault, extra_skip_dirs=extra_skip_dirs)
    readable = [n for n in notes if not n.get("unreadable")]
    candidates = (
        find_dead_links(notes)
        + find_exact_duplicates(notes)
        + find_orphans(notes)
        + find_stale(notes, max_age_days=max_age_days)
    )
    kinds = ("dead_link", "exact_duplicate", "orphan", "stale")
    stats = {"notes_scanned": len(readable), "read_errors": len(notes) - len(readable)}
    stats.update({k: sum(1 for c in candidates if c["kind"] == k) for k in kinds})
    return {"candidates": candidates, "stats": stats,
            "fingerprints": {n["path"]: n["sha256"] for n in notes}}
