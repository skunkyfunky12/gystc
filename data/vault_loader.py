"""Scan an entire Obsidian vault and build a graph (nodes + edges)."""
import re
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

FOLDER_TO_REGION = {
    "00 Index": 8,
    "01 Lucas": 3,
    "02 Projekte": 0,
    "03 Agenten": 6,
    "04 Sessions": 3,
    "05 Referenzen": 2,
}


def load_vault(vault_path: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """Scan vault and return (nodes, edges).

    Each node is a dict with keys: id, title, source_file, region_idx, tags, word_count.
    Edges are (source_idx, target_idx) tuples based on [[backlinks]].
    """
    vault = Path(vault_path)
    md_files = sorted(vault.rglob("*.md"))
    md_files = [f for f in md_files if ".obsidian" not in f.parts]

    title_to_idx: dict[str, int] = {}
    nodes: list[dict] = []

    for f in md_files:
        rel = f.relative_to(vault)
        title = f.stem
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"Warning: skipping {f} — {e}")
            continue

        brain_tags = _BRAIN_TAG_RE.findall(text)
        region_idx = 9
        if brain_tags:
            region_idx = REGION_TAG_TO_IDX.get(brain_tags[0], 9)
        else:
            top_folder = rel.parts[0] if len(rel.parts) > 1 else ""
            region_idx = FOLDER_TO_REGION.get(top_folder, 9)

        backlinks = _BACKLINK_RE.findall(text)
        word_count = len(text.split())
        all_tags = list(set(re.findall(r"#[\w/-]+", text)))
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
        created = mtime.strftime("%Y-%m-%d")

        idx = len(nodes)
        key = title.lower()
        if key not in title_to_idx:
            title_to_idx[key] = idx
        nodes.append({
            "id": str(rel).replace("\\", "/"),
            "title": title,
            "source_file": str(rel).replace("\\", "/"),
            "region_idx": region_idx,
            "tags": all_tags[:5],
            "word_count": word_count,
            "created": created,
            "backlink_titles": [bl.strip() for bl in backlinks],
        })

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for src_idx, node in enumerate(nodes):
        for bl_title in node.get("backlink_titles", []):
            tgt_idx = title_to_idx.get(bl_title.lower())
            if tgt_idx is not None and tgt_idx != src_idx:
                pair = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)

    for node in nodes:
        node.pop("backlink_titles", None)

    print(f"Vault: {len(nodes)} notes, {len(edges)} backlink edges")
    return nodes, edges
