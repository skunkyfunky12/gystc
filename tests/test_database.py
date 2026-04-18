import json
from brain_mcp.storage.database import BrainDB

def test_create_tables(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    names = [r[0] for r in tables]
    assert "notes" in names
    assert "regions" in names
    assert "edges" in names
    assert "schema_version" in names
    db.close()

def test_default_regions_seeded(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    rows = db.execute("SELECT idx, name, color FROM regions ORDER BY idx").fetchall()
    assert len(rows) == 12
    assert rows[0][1] == "Praefrontaler Cortex"
    assert rows[0][2] == "#3498DB"
    assert rows[11][1] == "Amygdala"
    db.close()

def test_upsert_note(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(
        path="test/note.md", title="Test Note", content="Some content here",
        content_hash="abc123", region_idx=3, tags=["#brain/hippocampus"],
        word_count=3, created_at="2026-01-01", modified_at="2026-01-02",
    )
    row = db.get_note_by_path("test/note.md")
    assert row is not None
    assert row["title"] == "Test Note"
    assert row["region_idx"] == 3
    assert json.loads(row["tags"]) == ["#brain/hippocampus"]
    db.close()

def test_upsert_note_updates_existing(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="V1", content="old", content_hash="h1",
                   region_idx=0, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="a.md", title="V2", content="new", content_hash="h2",
                   region_idx=1, tags=["#x"], word_count=2, created_at="2026-01-01", modified_at="2026-01-02")
    row = db.get_note_by_path("a.md")
    assert row["title"] == "V2"
    assert row["content_hash"] == "h2"
    assert row["region_idx"] == 1
    assert row["created_at"] == "2026-01-01"  # MUST preserve original created_at
    db.close()

def test_get_note_by_title(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="x.md", title="Hello World", content="c", content_hash="h",
                   region_idx=0, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    row = db.get_note_by_title("Hello World")
    assert row is not None
    assert row["path"] == "x.md"
    db.close()

def test_upsert_edges(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="A", content="c", content_hash="h1",
                   region_idx=0, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="B", content="c", content_hash="h2",
                   region_idx=1, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    a = db.get_note_by_path("a.md")
    b = db.get_note_by_path("b.md")
    db.upsert_edge(a["id"], b["id"], link_text="B")
    edges = db.get_edges_for_note(a["id"])
    assert len(edges) == 1
    assert edges[0]["target_id"] == b["id"]
    db.close()

def test_fts_search(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="Routing", content="This note explains Express routing middleware",
                   content_hash="h1", region_idx=3, tags=[], word_count=6, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="Auth", content="Authentication uses JWT tokens",
                   content_hash="h2", region_idx=11, tags=[], word_count=5, created_at="2026-01-01", modified_at="2026-01-01")
    results = db.fts_search("routing middleware")
    assert len(results) >= 1
    assert results[0]["path"] == "a.md"
    db.close()

def test_recent_notes(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="old.md", title="Old", content="c", content_hash="h1",
                   region_idx=0, tags=[], word_count=1, created_at="2025-01-01", modified_at="2025-01-01")
    db.upsert_note(path="new.md", title="New", content="c", content_hash="h2",
                   region_idx=0, tags=[], word_count=1, created_at="2026-04-18", modified_at="2026-04-18")
    results = db.get_recent_notes(days=30, limit=10)
    assert results[0]["path"] == "new.md"
    db.close()

def test_region_note_counts(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.upsert_note(path="a.md", title="A", content="c", content_hash="h1",
                   region_idx=3, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="B", content="c", content_hash="h2",
                   region_idx=3, tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01")
    counts = db.get_region_note_counts()
    assert counts.get(3, 0) == 2
    assert counts.get(0, 0) == 0
    db.close()

def test_update_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    db.update_region(0, description="My planning region", color="#FF0000")
    row = db.execute("SELECT description, color FROM regions WHERE idx=0").fetchone()
    assert row[0] == "My planning region"
    assert row[1] == "#FF0000"
    db.close()
