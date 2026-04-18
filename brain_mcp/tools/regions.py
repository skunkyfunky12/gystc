from __future__ import annotations
import json
import re
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import REGION_NAMES, REGION_NAME_TO_IDX, resolve_region_idx

def handle_brain_regions(
    db: BrainDB,
    action: str,
    region: str | None = None,
    description: str | None = None,
    color: str | None = None,
) -> dict | list[dict]:
    if action == "list":
        return _list_regions(db)
    elif action == "describe":
        if not region:
            return {"error": "region parameter required for describe action"}
        return _describe_region(db, region)
    elif action == "customize":
        if not region:
            return {"error": "region parameter required for customize action"}
        return _customize_region(db, region, description, color)
    else:
        return {"error": f"Unknown action: {action}. Use 'list', 'describe', or 'customize'."}

def _list_regions(db: BrainDB) -> list[dict]:
    regions = db.get_all_regions()
    counts = db.get_region_note_counts()
    return [{"idx": r["idx"], "name": r["name"], "color": r["color"],
             "description": r["description"], "note_count": counts.get(r["idx"], 0),
             "position": json.loads(r["position"])} for r in regions]

def _describe_region(db: BrainDB, region_name: str) -> dict:
    idx = resolve_region_idx(region_name)
    if idx is None:
        return {"error": f"Unknown region: {region_name}. Use brain_regions(action='list')."}
    region = db.get_region(idx)
    counts = db.get_region_note_counts()
    notes = db.execute("SELECT title, path, word_count FROM notes WHERE region_idx=? ORDER BY word_count DESC LIMIT 10", (idx,)).fetchall()
    return {"idx": region["idx"], "name": region["name"], "color": region["color"],
            "description": region["description"], "position": json.loads(region["position"]),
            "note_count": counts.get(idx, 0),
            "top_notes": [{"title": n["title"], "path": n["path"], "word_count": n["word_count"]} for n in notes]}

def _customize_region(db: BrainDB, region_name: str, description: str | None, color: str | None) -> dict:
    idx = resolve_region_idx(region_name)
    if idx is None:
        return {"error": f"Unknown region: {region_name}"}
    # REVIEW FIX: validate hex color
    if color and not re.match(r'^#[0-9A-Fa-f]{6}$', color):
        return {"error": "Color must be a hex color like #FF0000"}
    db.update_region(idx, description=description, color=color)
    return {"updated": True, "region": region_name, "idx": idx}
