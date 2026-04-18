from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.regions import handle_brain_regions

def _seed(db):
    db.upsert_note(path="a.md", title="A", content="c", content_hash="h1",
                   region_idx=0, tags=[], word_count=100, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="b.md", title="B", content="c", content_hash="h2",
                   region_idx=0, tags=[], word_count=50, created_at="2026-01-01", modified_at="2026-01-01")
    db.upsert_note(path="c.md", title="C", content="c", content_hash="h3",
                   region_idx=3, tags=[], word_count=200, created_at="2026-01-01", modified_at="2026-01-01")

def test_list_regions(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed(db)
    result = handle_brain_regions(db, action="list")
    assert len(result) == 12
    assert result[0]["name"] == "Praefrontaler Cortex"
    assert result[0]["note_count"] == 2
    assert result[3]["note_count"] == 1
    db.close()

def test_describe_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    _seed(db)
    result = handle_brain_regions(db, action="describe", region="Praefrontaler Cortex")
    assert result["name"] == "Praefrontaler Cortex"
    assert result["note_count"] == 2
    assert len(result["top_notes"]) == 2
    assert result["top_notes"][0]["word_count"] >= result["top_notes"][1]["word_count"]
    db.close()

def test_customize_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="customize", region="Praefrontaler Cortex",
                                  description="My custom description", color="#FF0000")
    assert result["updated"] is True
    row = db.get_region(0)
    assert row["description"] == "My custom description"
    assert row["color"] == "#FF0000"
    db.close()

def test_customize_invalid_color(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="customize", region="Praefrontaler Cortex", color="not-a-color")
    assert "error" in result
    db.close()

def test_describe_unknown_region(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="describe", region="Nonexistent")
    assert "error" in result
    db.close()

def test_invalid_action(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    result = handle_brain_regions(db, action="delete")
    assert "error" in result
    db.close()
