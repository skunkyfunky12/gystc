# tests/test_tool_related.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.related import handle_brain_related


def _seed(db, vectors, embedder):
    notes = [
        ("a.md", "Routing", "Express routing handles HTTP requests", 3),
        ("b.md", "Auth", "JWT authentication protects endpoints", 11),
        ("c.md", "Middleware", "Middleware chain processes requests in order", 0),
    ]
    for path, title, content, region_idx in notes:
        nid = db.upsert_note(
            path=path, title=title, content=content, content_hash=f"h_{path}",
            region_idx=region_idx, tags=[], word_count=len(content.split()),
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(nid, faiss_ids[0])
    a = db.get_note_by_path("a.md")
    c = db.get_note_by_path("c.md")
    db.upsert_edge(a["id"], c["id"], link_text="Middleware")


def test_related_by_title(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, title="Routing", limit=5)
    assert len(result) >= 1
    assert all(r["title"] != "Routing" for r in result)
    db.close()


def test_related_by_path(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, path="a.md", limit=5)
    assert len(result) >= 1
    db.close()


def test_related_includes_backlink_neighbor(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, title="Routing", limit=5)
    titles = {r["title"] for r in result}
    assert "Middleware" in titles
    db.close()


def test_related_has_relation_type(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_related(db, vectors, mock_embedder, title="Routing", limit=5)
    assert all("relation_type" in r for r in result)
    db.close()


def test_related_unknown_title(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_related(db, vectors, mock_embedder, title="Nonexistent", limit=5)
    assert "error" in result or result == []
    db.close()
