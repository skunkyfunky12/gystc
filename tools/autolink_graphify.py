#!/usr/bin/env python3
"""Auto-link graphify notes to parent documentation.

Reads _autolink.json from a project's graphify folder,
then adds Dokumentation backlinks to graphify notes
and updates Code-Referenzen in main docs.

Usage:
    python autolink_graphify.py <vault_path> <project_subfolder>

Example:
    python autolink_graphify.py "/path/to/vault" "02 Projekte/MeinProjekt"
"""
import json
import re
import sys
from pathlib import Path


def load_config(graphify_dir: Path) -> dict | None:
    config_path = graphify_dir / "_autolink.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.index("---", 3)
    fm_text = text[3:end].strip()
    result = {}
    for line in fm_text.split("\n"):
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, val = line.split(":", 1)
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def match_note_to_doc(note_name: str, source_file: str, rules: list[dict]) -> str | None:
    """Match a graphify note to a parent doc based on patterns."""
    for rule in rules:
        for pattern in rule["patterns"]:
            if pattern.lower() in note_name.lower() or pattern.lower() in source_file.lower():
                return rule["doc"]
    return None


def add_doc_link_to_note(note_path: Path, doc_name: str) -> bool:
    """Add Dokumentation backlink to a graphify note if missing."""
    text = note_path.read_text(encoding="utf-8")
    link = f"[[{doc_name}]]"
    if link in text:
        return False

    # Add after ## Connections section, before tags
    doc_line = f"\n## Dokumentation\n- {link}\n"

    # Find position: after last Connection line, before tags line
    lines = text.split("\n")
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        # Find the tag line at the bottom (e.g. #graphify/code ...)
        if line.startswith("#graphify/") or line.startswith("#brain/"):
            insert_idx = i
            break

    lines.insert(insert_idx, doc_line.strip())
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def update_code_referenzen(doc_path: Path, new_notes: list[str]) -> int:
    """Add new graphify notes to a doc's ## Code-Referenzen section."""
    text = doc_path.read_text(encoding="utf-8")

    # Find existing Code-Referenzen section
    if "## Code-Referenzen" not in text:
        return 0

    # Get already linked notes
    existing = set(re.findall(r"\[\[([^\]]+)\]\]", text.split("## Code-Referenzen")[1].split("\n#")[0]))

    added = 0
    lines = text.split("\n")
    # Find the end of Code-Referenzen (before #brain/ tag or next section)
    ref_start = None
    ref_end = None
    for i, line in enumerate(lines):
        if "## Code-Referenzen" in line:
            ref_start = i
        elif ref_start and (line.startswith("#brain/") or (line.startswith("## ") and i > ref_start)):
            ref_end = i
            break

    if ref_start is None:
        return 0
    if ref_end is None:
        ref_end = len(lines)

    insert_at = ref_end
    for note_name in sorted(new_notes):
        if note_name not in existing and not note_name.startswith("_"):
            lines.insert(insert_at, f"- [[{note_name}]]")
            insert_at += 1
            added += 1

    if added:
        doc_path.write_text("\n".join(lines), encoding="utf-8")
    return added


def main():
    if len(sys.argv) < 3:
        print("Usage: autolink_graphify.py <vault_path> <project_subfolder>")
        sys.exit(1)

    vault = Path(sys.argv[1])
    project = sys.argv[2]
    graphify_dir = vault / project / "graphify"

    config = load_config(graphify_dir)
    if not config:
        print(f"No _autolink.json found in {graphify_dir}")
        sys.exit(1)

    rules = config["rules"]
    doc_folder = vault / config["doc_folder"]

    # Track which notes belong to which docs
    doc_notes: dict[str, list[str]] = {r["doc"]: [] for r in rules}

    # Phase 1: Add Dokumentation links to graphify notes
    linked = 0
    for note_path in sorted(graphify_dir.glob("*.md")):
        if note_path.name.startswith("_"):
            continue

        note_name = note_path.stem
        fm = read_frontmatter(note_path)
        source = fm.get("source_file", "")

        doc = match_note_to_doc(note_name, source, rules)
        if doc:
            if add_doc_link_to_note(note_path, doc):
                linked += 1
            doc_notes[doc].append(note_name)

    # Phase 2: Update Code-Referenzen in main docs
    updated = 0
    for doc_name, notes in doc_notes.items():
        if not notes:
            continue
        doc_path = doc_folder / f"{doc_name}.md"
        if doc_path.exists():
            n = update_code_referenzen(doc_path, notes)
            if n:
                updated += n

    print(f"Autolink: {linked} Dokumentation-Links hinzugefuegt, {updated} Code-Referenzen aktualisiert")


if __name__ == "__main__":
    main()
