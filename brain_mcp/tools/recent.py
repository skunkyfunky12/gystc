from __future__ import annotations
from brain_mcp.storage.database import BrainDB

REGION_NAMES = [
    "Praefrontaler Cortex", "Motorischer Cortex", "Sensorischer Cortex",
    "Hippocampus", "Kleinhirn", "Nucleus Accumbens", "Broca-Areal",
    "Visueller Cortex", "Thalamus", "Stammhirn", "Basalganglien", "Amygdala",
]
REGION_NAME_TO_IDX = {name: i for i, name in enumerate(REGION_NAMES)}

def handle_brain_recent(db: BrainDB, days: int = 7, region: str | None = None, limit: int = 20) -> list[dict]:
    days = max(1, min(days, 365))
    limit = max(1, min(limit, 100))
    region_idx = REGION_NAME_TO_IDX.get(region) if region else None
    rows = db.get_recent_notes(days=days, region_idx=region_idx, limit=limit)
    results = []
    for r in rows:
        results.append({
            "title": r["title"], "path": r["path"],
            "region": REGION_NAMES[r["region_idx"]] if 0 <= r["region_idx"] < 12 else "Stammhirn",
            "modified_at": r["modified_at"], "word_count": r["word_count"],
        })
    return results
