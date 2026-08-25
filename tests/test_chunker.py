from brain_mcp.indexer.chunker import split_into_chunks, MIN_CHUNK_WORDS


def _make_long_content(word_count: int, headings: list[str] | None = None) -> str:
    if headings is None:
        headings = ["## Section A", "## Section B", "## Section C"]
    words_per_section = word_count // (len(headings) + 1)
    filler = " ".join(["word"] * words_per_section)
    parts = [f"Intro paragraph.\n{filler}"]
    for h in headings:
        parts.append(f"\n\n{h}\n\n{filler}")
    return "\n".join(parts)


def test_short_note_no_chunks():
    content = "Short note with only a few words."
    result = split_into_chunks(content, "Short Note")
    assert result == []


def test_long_note_splits_at_headings():
    content = _make_long_content(600, ["## Architecture", "## Testing"])
    result = split_into_chunks(content, "My Note")
    assert len(result) >= 2
    headings = [c["heading"] for c in result]
    assert "Architecture" in headings or "My Note" in headings


def test_chunk_has_required_fields():
    content = _make_long_content(600)
    result = split_into_chunks(content, "Test Note")
    assert len(result) > 0
    chunk = result[0]
    assert "heading" in chunk
    assert "content" in chunk
    assert "content_hash" in chunk
    assert "word_count" in chunk
    assert "chunk_idx" in chunk


def test_chunk_idx_sequential():
    content = _make_long_content(800, ["## A", "## B", "## C"])
    result = split_into_chunks(content, "Note")
    indices = [c["chunk_idx"] for c in result]
    assert indices == list(range(len(indices)))


def test_no_headings_returns_empty():
    content = " ".join(["word"] * 600)
    result = split_into_chunks(content, "Plain Note")
    assert result == []


def test_small_first_chunk_merges_forward():
    content = "# Title\nShort intro.\n\n## Big Section\n\n" + " ".join(["word"] * 550)
    result = split_into_chunks(content, "Note")
    if result:
        assert result[0]["word_count"] >= MIN_CHUNK_WORDS


def test_small_chunk_merges_backward():
    big = " ".join(["word"] * 300)
    small = "tiny"
    content = f"## Section 1\n\n{big}\n\n## Section 2\n\n{big}\n\n## Section 3\n\n{small}"
    result = split_into_chunks(content, "Note")
    for chunk in result:
        assert chunk["word_count"] >= MIN_CHUNK_WORDS or len(result) == 1


def test_content_hash_deterministic():
    content = _make_long_content(600)
    r1 = split_into_chunks(content, "Note")
    r2 = split_into_chunks(content, "Note")
    assert [c["content_hash"] for c in r1] == [c["content_hash"] for c in r2]


def test_h3_headings_also_split():
    parts = ["Intro.\n" + " ".join(["word"] * 200)]
    parts.append("### Sub A\n\n" + " ".join(["word"] * 200))
    parts.append("### Sub B\n\n" + " ".join(["word"] * 200))
    content = "\n\n".join(parts)
    result = split_into_chunks(content, "Note")
    assert len(result) >= 2


# -----------------------------------------------------------------------
# DB integration tests
# -----------------------------------------------------------------------
from brain_mcp.storage.database import BrainDB


def _make_db_note(db, path="long.md", title="Long Note"):
    return db.upsert_note(
        path=path, title=title, content="x" * 100,
        content_hash="h_long", region_idx=0, tags=[],
        word_count=600, created_at="2026-01-01", modified_at="2026-01-01",
    )


def test_replace_chunks(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    nid = _make_db_note(db)
    chunks = [
        {"heading": "Intro", "content": "intro text", "content_hash": "ch1",
         "word_count": 2, "chunk_idx": 0},
        {"heading": "Details", "content": "detail text", "content_hash": "ch2",
         "word_count": 2, "chunk_idx": 1},
    ]
    old_faiss = db.replace_chunks(nid, chunks)
    assert old_faiss == []
    stored = db.get_chunks_for_note(nid)
    assert len(stored) == 2
    assert stored[0]["heading"] == "Intro"
    assert stored[1]["chunk_idx"] == 1
    db.close()


def test_replace_chunks_returns_old_faiss(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    nid = _make_db_note(db)
    db.replace_chunks(nid, [
        {"heading": "A", "content": "a", "content_hash": "c1",
         "word_count": 1, "chunk_idx": 0},
    ])
    db.set_chunk_faiss_idx(nid, 0, 42)
    old_faiss = db.replace_chunks(nid, [
        {"heading": "B", "content": "b", "content_hash": "c2",
         "word_count": 1, "chunk_idx": 0},
    ])
    assert old_faiss == [42]
    db.close()


def test_get_chunks_by_faiss_indices(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    nid = _make_db_note(db)
    db.replace_chunks(nid, [
        {"heading": "H", "content": "c", "content_hash": "ch",
         "word_count": 1, "chunk_idx": 0},
    ])
    db.set_chunk_faiss_idx(nid, 0, 99)
    results = db.get_chunks_by_faiss_indices([99])
    assert len(results) == 1
    assert results[0]["heading"] == "H"
    assert results[0]["note_title"] == "Long Note"
    assert results[0]["note_path"] == "long.md"
    db.close()


def test_delete_note_cascades_chunks(tmp_path):
    db = BrainDB(tmp_path / "test.db")
    nid = _make_db_note(db)
    db.replace_chunks(nid, [
        {"heading": "X", "content": "y", "content_hash": "cz",
         "word_count": 1, "chunk_idx": 0},
    ])
    db.delete_note("long.md")
    count = db.execute("SELECT COUNT(*) FROM chunks WHERE note_id=?", (nid,)).fetchone()[0]
    assert count == 0
    db.close()
