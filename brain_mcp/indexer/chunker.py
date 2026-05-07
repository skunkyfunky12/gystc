from __future__ import annotations

import hashlib
import re

HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
MIN_CHUNK_WORDS = 50
CHUNK_THRESHOLD_WORDS = 500


def split_into_chunks(content: str, note_title: str) -> list[dict]:
    """Split markdown content at ## and ### headings for precise retrieval.

    Returns empty list if content is below threshold or has no headings.
    """
    if len(content.split()) < CHUNK_THRESHOLD_WORDS:
        return []

    matches = list(HEADING_RE.finditer(content))
    if not matches:
        return []

    raw: list[dict] = []
    if matches[0].start() > 0:
        pre = content[: matches[0].start()].strip()
        if pre:
            raw.append({"heading": note_title, "content": pre})

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if body:
            raw.append({"heading": heading, "content": body})

    if len(raw) < 2:
        return []

    merged: list[dict] = []
    for chunk in raw:
        wc = len(chunk["content"].split())
        if merged and wc < MIN_CHUNK_WORDS:
            merged[-1]["content"] += "\n\n" + chunk["content"]
        else:
            merged.append(chunk)

    if len(merged) > 1 and len(merged[0]["content"].split()) < MIN_CHUNK_WORDS:
        merged[1]["content"] = merged[0]["content"] + "\n\n" + merged[1]["content"]
        merged.pop(0)

    result: list[dict] = []
    for idx, chunk in enumerate(merged):
        chunk_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
        result.append({
            "heading": chunk["heading"],
            "content": chunk["content"],
            "content_hash": chunk_hash,
            "word_count": len(chunk["content"].split()),
            "chunk_idx": idx,
        })
    return result
