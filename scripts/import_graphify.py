"""Import graphify graph.json edges into brain.db.

Usage:
    python scripts/import_graphify.py <graph.json> [--dry-run]

Matches graphify nodes to brain.db notes by title/path, then imports
links as typed edges (semantic, contains, calls, etc.).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_mcp.storage.database import BrainDB


def normalize_label(label: str) -> str:
    """Normalize a graphify node label for matching."""
    return label.strip().lower().replace("_", " ").replace("-", " ")


def build_note_index(db: BrainDB) -> dict[str, int]:
    """Build lookup: normalized title -> note_id. Excludes ambiguous collisions."""
    index: dict[str, int] = {}
    collisions: set[str] = set()
    for note in db.get_all_notes():
        # Index by title (normalized)
        key = normalize_label(note["title"])
        if key in index and index[key] != note["id"]:
            collisions.add(key)
        else:
            index[key] = note["id"]
        # Also index by filename stem from path
        path_stem = Path(note["path"]).stem.lower().replace("_", " ").replace("-", " ")
        if path_stem not in index:
            index[path_stem] = note["id"]
    # Remove ambiguous keys
    for key in collisions:
        index.pop(key, None)
    if collisions:
        print(f"WARNING: {len(collisions)} ambiguous title collisions excluded from matching", file=sys.stderr)
    return index


def import_graphify(graph_path: Path, db: BrainDB, *, dry_run: bool = False) -> dict:
    """Import graphify graph.json into brain.db edges."""
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    links = graph.get("links", [])

    # Build graphify node_id -> label mapping
    gnode_map: dict[str, str] = {}
    for node in nodes:
        nid = node.get("id", "")
        label = node.get("label", node.get("norm_label", nid))
        gnode_map[nid] = label

    # Build brain.db note index
    note_index = build_note_index(db)

    # Match graphify nodes to brain notes
    matched = 0
    unmatched = 0
    gnode_to_note: dict[str, int] = {}
    for nid, label in gnode_map.items():
        key = normalize_label(label)
        if key in note_index:
            gnode_to_note[nid] = note_index[key]
            matched += 1
        else:
            unmatched += 1

    # Import edges
    edges_to_insert: list[tuple] = []
    skipped = 0
    for link in links:
        src = link.get("source", link.get("_src", ""))
        tgt = link.get("target", link.get("_tgt", ""))
        relation = link.get("relation", "related")
        weight = link.get("weight", 1.0)
        conf = link.get("confidence_score", link.get("confidence", None))
        if isinstance(conf, str):
            conf = {"EXTRACTED": 1.0, "HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}.get(conf, 0.5)
        source_file = link.get("source_file", "")

        src_id = gnode_to_note.get(src)
        tgt_id = gnode_to_note.get(tgt)
        if src_id is None or tgt_id is None or src_id == tgt_id:
            skipped += 1
            continue

        edge_type = f"graphify:{relation}"
        edges_to_insert.append((src_id, tgt_id, relation, edge_type, weight, conf, source_file))

    imported = 0
    if not dry_run and edges_to_insert:
        # Atomic: delete old graphify edges + insert new ones in single transaction
        _deleted, imported = db.replace_edges_by_type_prefix("graphify:", edges_to_insert)

    return {
        "graph_nodes": len(nodes),
        "graph_links": len(links),
        "matched_nodes": matched,
        "unmatched_nodes": unmatched,
        "edges_importable": len(edges_to_insert),
        "edges_skipped": skipped,
        "edges_imported": imported,
        "dry_run": dry_run,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_graphify.py <graph.json> [--dry-run]")
        sys.exit(1)

    graph_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if not graph_path.exists():
        print(f"Error: {graph_path} not found")
        sys.exit(1)

    db_path = Path.home() / ".gystc" / "brain.db"
    db = BrainDB(db_path)

    result = import_graphify(graph_path, db, dry_run=dry_run)

    print(f"Graphify Import {'(DRY RUN)' if dry_run else ''}")
    print(f"  Graph: {result['graph_nodes']} nodes, {result['graph_links']} links")
    print(f"  Matched: {result['matched_nodes']} nodes -> brain.db notes")
    print(f"  Unmatched: {result['unmatched_nodes']} nodes (no brain.db match)")
    print(f"  Edges importable: {result['edges_importable']}")
    print(f"  Edges skipped: {result['edges_skipped']} (missing node match or self-loop)")
    print(f"  Edges imported: {result['edges_imported']}")

    db.close()


if __name__ == "__main__":
    main()
