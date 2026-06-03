"""Tests for the PostToolUse activity-feed hook (feeds the dashboard terminal)."""
from brain_mcp.hooks.activity_feed import _build_event


def test_brain_retrieve_highlights_query():
    e = _build_event({"tool_name": "mcp__gystc__brain_retrieve", "tool_input": {"query": "caching"}})
    assert e["tag"] == "RETRIEVE"
    assert e["tagClass"] == "tag-mem"
    assert "<span class='hl'>caching</span>" in e["text"]


def test_brain_store_highlights_title():
    e = _build_event({"tool_name": "mcp__gystc__brain_store", "tool_input": {"title": "My Note"}})
    assert e["tag"] == "STORE"
    assert e["tagClass"] == "tag-mem"
    assert "<span class='hl'>My Note</span>" in e["text"]


def test_brain_related_uses_path_when_no_title():
    e = _build_event({"tool_name": "mcp__gystc__brain_related", "tool_input": {"path": "02 Projekte/x.md"}})
    assert e["tag"] == "RELATED"
    assert "<span class='hl'>02 Projekte/x.md</span>" in e["text"]


def test_read_uses_filename_stem():
    e = _build_event({"tool_name": "Read", "tool_input": {"file_path": "/a/b/server.py"}})
    assert e["tag"] == "READ"
    assert e["tagClass"] == "tag-tool"
    assert "<span class='hl'>server</span>" in e["text"]


def test_bash_is_tool_class():
    e = _build_event({"tool_name": "Bash", "tool_input": {"command": "x" * 200}})
    assert e["tag"] == "BASH"
    assert e["tagClass"] == "tag-tool"
    assert len(e["text"]) <= 120


def test_generic_tool_fallback():
    e = _build_event({"tool_name": "WebFetch", "tool_input": {"url": "x"}})
    assert e["tag"] == "WEBFETCH"
    assert e["tagClass"] == "tag-tool"


def test_missing_tool_name_returns_none():
    assert _build_event({"tool_input": {}}) is None
    assert _build_event({}) is None
