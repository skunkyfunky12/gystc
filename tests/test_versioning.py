import json
from brain_mcp.storage.database import BrainDB


def _make_note(db, path="a.md", title="Note A", content="Original content",
               content_hash="h1", region_idx=0):
    return db.upsert_note(
        path=path, title=title, content=content, content_hash=content_hash,
        region_idx=region_idx, tags=["#test"], word_count=len(content.split()),
        created_at="2026-01-01", modified_at="2026-01-01",
    )


def test_save_version(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db)
    vid = db.save_version(note_id, "Original content", "Note A", 0,
                          json.dumps(["#test"]), 2, "h1")
    assert vid > 0
    versions = db.get_versions(note_id)
    assert len(versions) == 1
    assert versions[0]["content_hash"] == "h1"
    db.close()


def test_save_version_dedup(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db)
    vid1 = db.save_version(note_id, "content", "T", 0, "[]", 1, "hash_a")
    vid2 = db.save_version(note_id, "content", "T", 0, "[]", 1, "hash_a")
    assert vid1 == vid2
    assert len(db.get_versions(note_id)) == 1
    db.close()


def test_get_version_by_id(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db)
    vid = db.save_version(note_id, "old content", "T", 0, "[]", 2, "h_old")
    version = db.get_version(vid)
    assert version is not None
    assert version["content"] == "old content"
    assert version["content_hash"] == "h_old"
    db.close()


def test_prune_versions(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db)
    for i in range(55):
        db.save_version(note_id, f"v{i}", "T", 0, "[]", 1, f"hash_{i}")
    db.prune_versions(note_id, max_versions=50)
    versions = db.get_versions(note_id, limit=100)
    assert len(versions) == 50
    hashes = [v["content_hash"] for v in versions]
    assert "hash_0" not in hashes
    assert "hash_54" in hashes
    db.close()


def test_upsert_note_saves_version_on_change(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db, content="v1 content", content_hash="h1")
    db.upsert_note(
        path="a.md", title="Note A", content="v2 content", content_hash="h2",
        region_idx=0, tags=["#test"], word_count=2,
        created_at="2026-01-01", modified_at="2026-01-02",
    )
    versions = db.get_versions(note_id)
    assert len(versions) == 1
    assert versions[0]["content_hash"] == "h1"
    assert versions[0]["title"] == "Note A"
    db.close()


def test_upsert_note_no_version_if_unchanged(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db, content_hash="same")
    db.upsert_note(
        path="a.md", title="Note A", content="Original content",
        content_hash="same", region_idx=0, tags=["#test"], word_count=2,
        created_at="2026-01-01", modified_at="2026-01-02",
    )
    versions = db.get_versions(note_id)
    assert len(versions) == 0
    db.close()


def test_delete_note_cascades_versions(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    note_id = _make_note(db)
    db.save_version(note_id, "old", "T", 0, "[]", 1, "h_old")
    db.delete_note("a.md")
    count = db.execute("SELECT COUNT(*) FROM note_versions WHERE note_id=?",
                       (note_id,)).fetchone()[0]
    assert count == 0
    db.close()
