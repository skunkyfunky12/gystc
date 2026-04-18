# tests/test_tool_context.py
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.context import handle_brain_context


def _seed(db, vectors, embedder):
    notes_data = [
        ("routing.md", "Routing", "Express routing handles HTTP request mapping", 3),
        ("auth.md", "Auth", "JWT authentication and bearer token validation", 11),
        ("middleware.md", "Middleware", "Request middleware chain for Express apps", 0),
    ]
    for path, title, content, region_idx in notes_data:
        nid = db.upsert_note(
            path=path, title=title, content=content, content_hash=f"h_{path}",
            region_idx=region_idx, tags=[], word_count=len(content.split()),
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(nid, faiss_ids[0])
    a = db.get_note_by_path("routing.md")
    c = db.get_note_by_path("middleware.md")
    db.upsert_edge(a["id"], c["id"], link_text="Middleware")


def test_context_by_file_paths(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   file_paths=["routing.md"], max_notes=10)
    assert len(result) >= 1
    titles = {r["title"] for r in result}
    assert "Middleware" in titles
    db.close()


def test_context_by_task_description(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   task_description="fixing authentication tokens", max_notes=5)
    assert len(result) >= 1
    db.close()


def test_context_combined(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   file_paths=["routing.md"],
                                   task_description="adding auth middleware",
                                   max_notes=10)
    assert len(result) >= 1
    db.close()


def test_context_has_relevance_reason(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   file_paths=["routing.md"], max_notes=10)
    assert all("relevance_reason" in r for r in result)
    db.close()


def test_context_respects_max_notes(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_context(db, vectors, mock_embedder,
                                   task_description="anything", max_notes=1)
    assert len(result) <= 1
    db.close()


def test_context_no_input(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_context(db, vectors, mock_embedder)
    assert "error" in result or result == []
    db.close()
