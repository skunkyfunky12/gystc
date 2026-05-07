import json
from pathlib import Path

from brain_mcp.hooks.session_start_context import get_brain_db_path, build_query, search_and_format
from brain_mcp.storage.database import BrainDB


def test_build_query_with_cwd():
    result = build_query(cwd="/home/user/my-project", git_context=None)
    assert "my-project" in result


def test_build_query_with_git():
    result = build_query(
        cwd="/home/user/repo",
        git_context={"repo": "neural-brain", "branch": "main", "commits": ["fix startup"]},
    )
    assert "neural-brain" in result
    assert "main" in result
    assert "fix startup" in result


def test_search_and_format(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(
        path="proj/note.md", title="Neural Brain Server",
        content="The neural brain MCP server handles vault indexing",
        content_hash="h1", region_idx=0, tags=[], word_count=8,
        created_at="2026-01-01", modified_at="2026-01-01",
    )
    db.close()
    output = search_and_format(tmp_path / "test.db", "neural brain")
    assert "Neural Brain Server" in output
    assert "=== Brain Context ===" in output
    assert "proj/note.md" in output


def test_search_and_format_no_results(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.close()
    output = search_and_format(tmp_path / "test.db", "nonexistent_term_xyz")
    assert output == ""


def test_get_brain_db_path_auto_context_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DATA_DIR", str(tmp_path))
    config = {"auto_context": False}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "brain.db").touch()
    assert get_brain_db_path() is None


def test_get_brain_db_path_auto_context_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DATA_DIR", str(tmp_path))
    config = {"auto_context": True}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "brain.db").touch()
    result = get_brain_db_path()
    assert result is not None
    assert result.name == "brain.db"
