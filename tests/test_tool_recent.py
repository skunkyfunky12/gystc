from datetime import datetime, timedelta, UTC

from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.recent import handle_brain_recent

def _seed_db(db):
    now = datetime.now(UTC)
    yesterday = (now - timedelta(days=1)).isoformat()
    two_days = (now - timedelta(days=2)).isoformat()
    old = (now - timedelta(days=30)).isoformat()
    db.upsert_note(path="a.md", title="Recent Note", content="c", content_hash="h1",
                   region_idx=3, tags=["#test"], word_count=10,
                   created_at=yesterday[:10], modified_at=yesterday)
    db.upsert_note(path="b.md", title="Old Note", content="c", content_hash="h2",
                   region_idx=0, tags=[], word_count=5,
                   created_at=old[:10], modified_at=old)
    db.upsert_note(path="c.md", title="Also Recent", content="c", content_hash="h3",
                   region_idx=3, tags=[], word_count=8,
                   created_at=two_days[:10], modified_at=two_days)

def test_brain_recent_default(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=7, region=None, limit=20)
    assert len(result) == 2
    assert result[0]["title"] == "Recent Note"
    db.close()

def test_brain_recent_region_filter(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=7, region="Hippocampus", limit=20)
    assert len(result) == 2
    assert all(r["region"] == "Hippocampus" for r in result)
    db.close()

def test_brain_recent_limit(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=365, region=None, limit=1)
    assert len(result) == 1
    db.close()

def test_brain_recent_output_format(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed_db(db)
    result = handle_brain_recent(db, days=7, region=None, limit=20)
    note = result[0]
    assert "title" in note
    assert "path" in note
    assert "region" in note
    assert "modified_at" in note
    assert "word_count" in note
    db.close()
