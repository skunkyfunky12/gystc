#!/usr/bin/env python3
"""PostToolUse hook: stream Claude's tool activity to the GYSTC Dashboard's live
terminal at http://127.0.0.1:9500.

Fire-and-forget: a short timeout and a blanket except mean this never blocks or
fails a Claude Code turn, and it stays silent when the dashboard isn't running.
Registered by the setup wizard as a PostToolUse hook (matcher "*").
"""
from __future__ import annotations

import json
import os
import sys

ACTIVITY_URL = "http://127.0.0.1:9500"
TIMEOUT = 0.5
_MAX_TEXT = 120


def _read_session_token() -> str | None:
    """Read the dashboard's per-session write token from <data_dir>/activity_token.

    Mirrors brain_mcp.config's data-dir resolution (BRAIN_DATA_DIR env override,
    else ~/.gystc) without importing brain_mcp — this hook runs standalone under
    Claude Code and must stay stdlib-only."""
    try:
        from pathlib import Path
        data_dir = os.environ.get("BRAIN_DATA_DIR")
        token_path = (Path(data_dir) if data_dir else Path.home() / ".gystc") / "activity_token"
        token = token_path.read_text(encoding="utf-8").strip()
        return token or None
    except OSError:
        return None  # dashboard not running this session — POST will be rejected loudly there


def _hl(label: str, term) -> str:
    """`label <span class='hl'>term</span>` — the hl span lights up the matching
    brain node in the dashboard. Returns just the label if term is empty."""
    term = ("" if term is None else str(term)).strip()
    if not term:
        return label
    if len(term) > 60:
        term = term[:60] + "…"
    return f"{label} <span class='hl'>{term}</span>"


def _build_event(payload: dict) -> dict | None:
    """Turn a Claude Code PostToolUse payload into a {tag, text, tagClass} event,
    or None if there's nothing worth showing."""
    tool = (payload.get("tool_name") or "").strip()
    if not tool:
        return None
    ti = payload.get("tool_input") or {}
    short = tool.split("__")[-1]  # mcp__gystc__brain_retrieve -> brain_retrieve

    # GYSTC memory tools — highlight the query/title so the brain node pulses.
    if "brain_" in tool:
        verb = short.replace("brain_", "")
        term = ti.get("query") or ti.get("title") or ti.get("path")
        if not term and isinstance(ti.get("file_paths"), list) and ti["file_paths"]:
            term = ti["file_paths"][0]
        return {"tag": (verb.upper()[:20] or "BRAIN"), "text": _hl(short, term)[:_MAX_TEXT], "tagClass": "tag-mem"}

    # Common Claude Code tools.
    if tool in ("Edit", "Write", "MultiEdit"):
        name = os.path.splitext(os.path.basename(ti.get("file_path", "")))[0]
        return {"tag": tool.upper()[:20], "text": _hl(tool, name)[:_MAX_TEXT], "tagClass": "tag-mem"}
    if tool == "Read":
        name = os.path.splitext(os.path.basename(ti.get("file_path", "")))[0]
        if not name:
            return None
        return {"tag": "READ", "text": _hl("Read", name)[:_MAX_TEXT], "tagClass": "tag-tool"}
    if tool in ("Grep", "Glob"):
        return {"tag": tool.upper(), "text": _hl(tool, ti.get("pattern"))[:_MAX_TEXT], "tagClass": "tag-tool"}
    if tool == "Bash":
        return {"tag": "BASH", "text": _hl("Bash", (ti.get("command") or "")[:50])[:_MAX_TEXT], "tagClass": "tag-tool"}

    return {"tag": (short.upper()[:20] or "TOOL"), "text": short[:_MAX_TEXT], "tagClass": "tag-tool"}


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    event = _build_event(payload)
    if not event:
        return
    try:
        import urllib.request
        headers = {"Content-Type": "application/json"}
        token = _read_session_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            ACTIVITY_URL,
            data=json.dumps(event).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=TIMEOUT).close()
    except Exception:
        pass  # dashboard not running / port closed — silently ignore


if __name__ == "__main__":
    main()
