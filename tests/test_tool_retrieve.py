# tests/test_tool_retrieve.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.retrieve import handle_brain_retrieve


def _seed(db, vectors, embedder):
    notes = [
        ("routing.md", "Routing", "Express routing handles HTTP requests and maps URLs to handlers", 3),
        ("auth.md", "Auth", "JWT authentication protects API endpoints with bearer tokens", 11),
        ("css.md", "CSS Themes", "Tailwind CSS themes with dark mode and custom color palette", 7),
    ]
    for path, title, content, region_idx in notes:
        note_id = db.upsert_note(
            path=path, title=title, content=content, content_hash=f"h_{path}",
            region_idx=region_idx, tags=[], word_count=len(content.split()),
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(note_id, faiss_ids[0])


def test_retrieve_finds_relevant(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    # MockEmbedder uses hash-based random vectors, so cosine similarity between
    # different texts can be near zero or negative. Use threshold=-1.0 to test
    # retrieval logic independent of embedding quality.
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="HTTP routing middleware", limit=3, threshold=-1.0)
    assert len(result) >= 1
    assert all("title" in r for r in result)
    assert all("similarity" in r for r in result)
    db.close()


def test_retrieve_region_filter(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="routing", region="Hippocampus", limit=10)
    assert all(r["region"] == "Hippocampus" for r in result)
    db.close()


def test_retrieve_respects_limit(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="test", limit=1)
    assert len(result) <= 1
    db.close()


def test_retrieve_output_format(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="routing", limit=3, threshold=-1.0)
    assert len(result) >= 1
    if result:
        r = result[0]
        assert "title" in r
        assert "path" in r
        assert "region" in r
        assert "similarity" in r
        assert "snippet" in r
        assert "word_count" in r
    db.close()


def test_retrieve_empty_index(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_retrieve(db, vectors, mock_embedder, query="anything", limit=10)
    assert result == []
    db.close()
