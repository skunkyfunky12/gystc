from __future__ import annotations
from brain_mcp.storage.database import BrainDB

REGION_NAMES = [
    "Praefrontaler Cortex", "Motorischer Cortex", "Sensorischer Cortex",
    "Hippocampus", "Kleinhirn", "Nucleus Accumbens", "Broca-Areal",
    "Visueller Cortex", "Thalamus", "Stammhirn", "Basalganglien", "Amygdala",
]
REGION_NAME_TO_IDX = {name: i for i, name in enumerate(REGION_NAMES)}
# Case-insensitive lookup: maps lowercased region name to index
_REGION_NAME_TO_IDX_LOWER = {name.lower(): i for i, name in enumerate(REGION_NAMES)}


def resolve_region_idx(region: str | None) -> int | None:
    """Resolve a region name to its index, case-insensitively. Returns None if not found or input is None."""
    if region is None:
        return None
    # Try exact match first, then case-insensitive
    idx = REGION_NAME_TO_IDX.get(region)
    if idx is not None:
        return idx
    return _REGION_NAME_TO_IDX_LOWER.get(region.lower())

def handle_brain_recent(db: BrainDB, days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    days = max(1, min(days, 365))
    limit = max(1, min(limit, 100))
    region_idx = resolve_region_idx(region)
    rows = db.get_recent_notes(days=days, region_idx=region_idx, limit=limit)
    results = []
    for r in rows:
        results.append({
            "title": r["title"], "path": r["path"],
            "region": REGION_NAMES[r["region_idx"]] if 0 <= r["region_idx"] < 12 else "Stammhirn",
            "modified_at": r["modified_at"], "word_count": r["word_count"],
        })
    return results
