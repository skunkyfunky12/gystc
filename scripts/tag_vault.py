"""Assign #brain/<region> tags to all graphify notes based on their community.

Usage:  python tag_vault.py <vault_path>   (or set GYSTC_VAULT)
"""
import os
import re
import sys
from pathlib import Path

_vault_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GYSTC_VAULT", "")
if not _vault_arg:
    sys.exit("Usage: python tag_vault.py <vault_path>  (or set GYSTC_VAULT)")
VAULT = Path(_vault_arg)

COMMUNITY_TO_REGION = {
    # Visual / UI
    "3D Camera & GL": "visueller-cortex",
    "Brain App UI": "visueller-cortex",
    "GL Scene & Shaders": "visueller-cortex",
    "Web Widget Bridge": "visueller-cortex",
    # Architecture / Planning
    "MCP Server Core": "praefrontaler-cortex",
    "MCP Spec Design": "praefrontaler-cortex",
    "Plans & Specs": "praefrontaler-cortex",
    # Data intake / Search
    "Embedder & Retrieve": "sensorischer-cortex",
    "Recent Notes Tool": "sensorischer-cortex",
    # Write actions
    "Tools: Store & Context": "motorischer-cortex",
    # Infrastructure
    "Storage & DB Layer": "stammhirn",
    "Config System": "stammhirn",
    "GPU & Launch Logs": "stammhirn",
    # Background processing
    "Vault Scanner": "basalganglien",
    "File Watcher": "basalganglien",
    # Precision algorithms
    "Layout & Physics": "kleinhirn",
    "Physics & Dependencies": "kleinhirn",
    "Region Tests": "kleinhirn",
    # Data relay
    "Vector Store & Loader": "thalamus",
    "Obsidian Integration": "thalamus",
    "Region Data": "thalamus",
}
DEFAULT_REGION = "stammhirn"

BRAIN_TAG_RE = re.compile(r"#brain/[\w-]+")


def assign_tag(file_path: Path) -> str | None:
    text = file_path.read_text(encoding="utf-8", errors="replace")

    if BRAIN_TAG_RE.search(text):
        tag = BRAIN_TAG_RE.search(text).group()
        if tag in ("#brain/hub", "#brain/region", "#brain/"):
            pass  # invalid, replace below
        else:
            return None  # already tagged correctly

    community_match = re.search(r'community:\s*"([^"]+)"', text)
    if community_match:
        community = community_match.group(1)
        region = COMMUNITY_TO_REGION.get(community, DEFAULT_REGION)
    else:
        rel = file_path.relative_to(VAULT)
        top_folder = rel.parts[0] if len(rel.parts) > 1 else ""
        folder_map = {
            "00 Index": "thalamus",
            "01 Lucas": "hippocampus",
            "02 Projekte": "praefrontaler-cortex",
            "04 Sessions": "hippocampus",
            "05 Referenzen": "sensorischer-cortex",
        }
        region = folder_map.get(top_folder, DEFAULT_REGION)

    tag = f"#brain/{region}"

    old_bad = BRAIN_TAG_RE.search(text)
    if old_bad:
        text = text.replace(old_bad.group(), tag)
    elif text.rstrip().endswith("---") and text.count("---") >= 2:
        last_fence = text.rfind("---")
        before = text[:last_fence].rstrip()
        after = text[last_fence:]
        tags_match = re.search(r"(tags:\s*\n(?:\s+-\s+.*\n)*)", before)
        if tags_match:
            tags_block = tags_match.group(0)
            new_tags = tags_block.rstrip("\n") + f"\n  - brain/{region}\n"
            text = text.replace(tags_block, new_tags)
        else:
            text = text[:last_fence] + f"tags:\n  - brain/{region}\n" + text[last_fence:]
    else:
        text = text.rstrip() + f"\n\n{tag}\n"

    file_path.write_text(text, encoding="utf-8")
    return tag


def main():
    md_files = sorted(VAULT.rglob("*.md"))
    md_files = [f for f in md_files if ".obsidian" not in f.parts]

    tagged = 0
    skipped = 0
    for f in md_files:
        result = assign_tag(f)
        if result:
            tagged += 1
        else:
            skipped += 1

    print(f"Tagged: {tagged} notes")
    print(f"Skipped (already tagged): {skipped} notes")
    print(f"Total: {tagged + skipped} notes")


if __name__ == "__main__":
    main()
