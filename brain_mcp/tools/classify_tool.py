"""Handlers for brain_classify and brain_classify_feedback MCP tools."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from brain_mcp.indexer.scanner import REGION_TAG_TO_IDX
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.classifier import classify_region
from brain_mcp.tools.recent import REGION_NAMES

_BRAIN_TAG_RE = re.compile(r"#brain/[\w-]+")  # non-capturing, for re.sub

# Derive slug lookup from canonical source (scanner.py)
_IDX_TO_SLUG: dict[int, str] = {idx: slug for slug, idx in REGION_TAG_TO_IDX.items()}


def handle_brain_classify(
    db: BrainDB,
    vault_path: Path | None,
    *,
    title: str | None = None,
    path: str | None = None,
    content: str | None = None,
    batch: bool = False,
    apply: bool = False,
) -> dict:
    """Classify a note or batch-classify all Stammhirn notes."""

    if batch:
        return _batch_classify(db, vault_path, apply=apply)

    # Single note classification
    if not title and not path and not content:
        return {"error": "Provide at least one of: title, path, or content"}

    # Try to find note in DB
    note = None
    if path:
        note = db.get_note_by_path(path)
    elif title:
        note = db.get_note_by_title(title)

    note_title = title if title is not None else (note["title"] if note else "unknown")
    note_content = content if content is not None else (note["content"] if note else "")
    note_path = path if path is not None else (note["path"] if note else None)

    region_idx = classify_region(note_title, note_content or "", path=note_path)

    result = {
        "title": note_title,
        "path": note_path,
        "region_idx": region_idx,
        "region": REGION_NAMES[region_idx],
        "applied": False,
    }

    if apply and note:
        db.update_note_region(note["id"], region_idx)
        _update_file_tag(vault_path, note["path"], region_idx)
        result["applied"] = True

    return result


def _batch_classify(db: BrainDB, vault_path: Path | None, *, apply: bool = False) -> dict:
    """Classify all Stammhirn (region 9) notes."""
    stammhirn_notes = [n for n in db.get_all_notes() if n["region_idx"] == 9]
    moves = []
    unchanged = 0
    file_errors = []

    for note in stammhirn_notes:
        new_idx = classify_region(note["title"], note["content"] or "", path=note["path"])
        if new_idx != 9:
            move = {
                "title": note["title"],
                "path": note["path"],
                "from": REGION_NAMES[9],
                "to": REGION_NAMES[new_idx],
                "to_idx": new_idx,
            }
            moves.append(move)
            if apply:
                db.update_note_region(note["id"], new_idx)
                err = _update_file_tag(vault_path, note["path"], new_idx)
                if err:
                    file_errors.append(err)
        else:
            unchanged += 1

    result = {
        "batch": True,
        "stammhirn_total": len(stammhirn_notes),
        "reclassified": len(moves),
        "unchanged": unchanged,
        "applied": apply,
        "moves": moves[:100],
        "moves_truncated": len(moves) > 100,
    }
    if file_errors:
        result["file_errors"] = file_errors[:20]
    return result


def _update_file_tag(vault_path: Path | None, note_path: str, region_idx: int) -> str | None:
    """Update #brain/ tag in the .md file. Returns error string or None."""
    if vault_path is None:
        return None
    file_path = (vault_path / note_path).resolve()
    # Guard against path traversal (CWE-22)
    if not file_path.is_relative_to(vault_path.resolve()):
        return f"{note_path}: path traversal blocked"
    new_slug = _IDX_TO_SLUG.get(region_idx, "stammhirn")
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if _BRAIN_TAG_RE.search(text):
            text = _BRAIN_TAG_RE.sub(f"#brain/{new_slug}", text, count=1)
        else:
            text = text.rstrip() + f"\n\n#brain/{new_slug}\n"
        file_path.write_text(text, encoding="utf-8")
        return None
    except OSError as e:
        return f"{note_path}: {e}"


def handle_brain_classify_feedback(
    db: BrainDB,
    vault_path: Path | None,
    data_dir: Path,
    *,
    path: str,
    correct_region_idx: int,
    reason: str = "",
) -> dict:
    """Correct a misclassification and log for future improvement."""
    correct_region_idx = int(correct_region_idx)
    if not (0 <= correct_region_idx < 12):
        return {"error": f"correct_region_idx must be 0-11, got {correct_region_idx}"}

    note = db.get_note_by_path(path)
    if not note:
        return {"error": f"Note not found: {path}"}

    old_idx = note["region_idx"]
    db.update_note_region(note["id"], correct_region_idx)

    file_err = _update_file_tag(vault_path, path, correct_region_idx)

    # Log correction to JSONL for future keyword improvement
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": path,
        "title": note["title"],
        "old_region_idx": old_idx,
        "old_region": REGION_NAMES[old_idx] if 0 <= old_idx < len(REGION_NAMES) else f"unknown({old_idx})",
        "correct_region_idx": correct_region_idx,
        "correct_region": REGION_NAMES[correct_region_idx],
        "reason": reason,
    }

    log_path = data_dir / "classification_log.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"WARNING: Could not write classification log: {e}", file=sys.stderr)

    result = {
        "path": path,
        "old_region": REGION_NAMES[old_idx] if 0 <= old_idx < len(REGION_NAMES) else f"unknown({old_idx})",
        "new_region": REGION_NAMES[correct_region_idx],
        "applied": True,
        "logged": True,
    }
    if file_err:
        result["file_error"] = file_err
    return result
