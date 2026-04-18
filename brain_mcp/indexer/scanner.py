from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

_BACKLINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[|#][^\]]*?)?\]\]")
_BRAIN_TAG_RE = re.compile(r"#brain/([\w-]+)")

REGION_TAG_TO_IDX = {
    "praefrontaler-cortex": 0,
    "motorischer-cortex": 1,
    "sensorischer-cortex": 2,
    "hippocampus": 3,
    "kleinhirn": 4,
    "nucleus-accumbens": 5,
    "broca-areal": 6,
    "visueller-cortex": 7,
    "thalamus": 8,
    "stammhirn": 9,
    "basalganglien": 10,
    "amygdala": 11,
}


def compute_content_hash(text: str) -> str:
    """Return a deterministic SHA-256 hex digest for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_note_file(file_path: Path, vault_root: Path, folder_to_region: dict[str, int]) -> dict | None:
    """Parse a single .md file and return a structured note dict, or None on read error."""
    rel = file_path.relative_to(vault_root)
    title = file_path.stem

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # --- region detection ---
    brain_tags = _BRAIN_TAG_RE.findall(text)
    if brain_tags:
        region_idx = REGION_TAG_TO_IDX.get(brain_tags[0], 9)
    else:
        top_folder = rel.parts[0] if len(rel.parts) > 1 else ""
        region_idx = folder_to_region.get(top_folder, 9)

    # --- backlinks ---
    backlinks = _BACKLINK_RE.findall(text)

    # --- metadata ---
    word_count = len(text.split())
    all_tags = list(set(re.findall(r"#[\w/-]+", text)))
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)

    return {
        "path": str(rel).replace("\\", "/"),
        "title": title,
        "content": text,
        "content_hash": compute_content_hash(text),
        "region_idx": region_idx,
        "tags": all_tags[:20],
        "word_count": word_count,
        "created_at": mtime.strftime("%Y-%m-%d"),
        "modified_at": mtime.isoformat(),
        "backlink_titles": [bl.strip() for bl in backlinks],
    }


def scan_vault(vault_path: Path, folder_to_region: dict[str, int]) -> list[dict]:
    """Walk *vault_path*, parse every ``.md`` file, return structured note dicts.

    Parameters
    ----------
    vault_path:
        Root directory of the Obsidian vault.
    folder_to_region:
        Mapping from top-level folder name to region index.  Used as a
        fallback when a note has no ``#brain/<region>`` tag.
    """
    md_files = sorted(vault_path.rglob("*.md"))
    md_files = [f for f in md_files if ".obsidian" not in f.parts]

    notes: list[dict] = []
    for f in md_files:
        note = parse_note_file(f, vault_path, folder_to_region)
        if note is not None:
            notes.append(note)

    return notes
