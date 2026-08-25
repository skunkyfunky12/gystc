from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, UTC
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
    except OSError as e:
        print(f"WARNING: Cannot read {file_path}: {e}", file=sys.stderr)
        return None

    # --- region detection ---
    brain_tags = _BRAIN_TAG_RE.findall(text)
    if brain_tags:
        slug = brain_tags[0]
        if slug not in REGION_TAG_TO_IDX:
            print(f"WARNING: Unrecognized brain tag '#brain/{slug}' in {file_path.name}, defaulting to Stammhirn", file=sys.stderr)
        region_idx = REGION_TAG_TO_IDX.get(slug, 9)
    else:
        top_folder = rel.parts[0] if len(rel.parts) > 1 else ""
        region_idx = folder_to_region.get(top_folder, 9)

    # --- backlinks ---
    backlinks = _BACKLINK_RE.findall(text)

    # --- metadata ---
    word_count = len(text.split())
    all_tags = list(set(re.findall(r"#[\w/-]+", text)))
    stat = file_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    # Real creation time where the platform keeps one: st_birthtime on
    # macOS/BSD, st_ctime on Windows (there it IS the creation time, unlike
    # POSIX where it is the inode-change time). Linux exposes neither through
    # stat(), so mtime stays the best available guess. Never later than mtime:
    # a copied file gets a fresh creation stamp, and a note that reads
    # "created after it was last edited" is worse than a slightly early date.
    birth = getattr(stat, "st_birthtime", None)
    if birth is None and sys.platform == "win32":
        birth = stat.st_ctime
    created = min(datetime.fromtimestamp(birth, tz=UTC), mtime) if birth else mtime

    return {
        "path": str(rel).replace("\\", "/"),
        "title": title,
        "content": text,
        "content_hash": compute_content_hash(text),
        "region_idx": region_idx,
        "tags": all_tags[:20],
        "word_count": word_count,
        "created_at": created.strftime("%Y-%m-%d"),
        "modified_at": mtime.isoformat(),
        "backlink_titles": [bl.strip() for bl in backlinks],
    }


def scan_vault(
    vault_path: Path,
    folder_to_region: dict[str, int],
    exclude_dirs: list[str] | None = None,
) -> list[dict]:
    """Walk *vault_path*, parse every ``.md`` file, return structured note dicts.

    Parameters
    ----------
    vault_path:
        Root directory of the Obsidian vault.
    folder_to_region:
        Mapping from top-level folder name to region index.  Used as a
        fallback when a note has no ``#brain/<region>`` tag.
    exclude_dirs:
        Directory names (relative to the vault) to skip entirely, e.g.
        ``["99 Archiv", "graphify-out"]``.  A file is skipped when any of these
        names appears as a path component anywhere below the vault root.
    """
    # casefold so a config entry typed in a different case still matches on the
    # case-insensitive Windows filesystem (silent non-match would be a footgun).
    excluded = {d.casefold() for d in (exclude_dirs or [])}
    md_files = sorted(vault_path.rglob("*.md"))

    notes: list[dict] = []
    for f in md_files:
        if ".obsidian" in f.parts:
            continue
        if excluded and not excluded.isdisjoint(
            p.casefold() for p in f.relative_to(vault_path).parts
        ):
            continue
        note = parse_note_file(f, vault_path, folder_to_region)
        if note is not None:
            notes.append(note)

    return notes
