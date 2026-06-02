"""brain_classify apply behaviour — must not silently no-op (2026-06-02 audit)."""
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.classify_tool import handle_brain_classify


def test_classify_apply_errors_when_note_absent(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    # content-only classify with apply=True: nothing in the DB to update
    result = handle_brain_classify(db, None, content="text about routing", apply=True)
    assert result["applied"] is False
    assert "error" in result
    db.close()


def test_classify_apply_succeeds_for_stored_note(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="x.md", title="X", content="auth jwt login session token",
                   content_hash="h1", region_idx=9, tags=[], word_count=5,
                   created_at="2026-01-01", modified_at="2026-01-01")
    result = handle_brain_classify(db, None, path="x.md", apply=True)
    assert result["applied"] is True
    assert "error" not in result
    db.close()
