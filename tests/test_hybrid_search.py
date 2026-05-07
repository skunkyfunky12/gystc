from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.retrieve import handle_brain_retrieve, _rrf_merge


def _seed(db, vectors, embedder):
    notes = [
        ("routing.md", "Express Routing", "Express routing handles HTTP requests", 3),
        ("auth.md", "JWT Auth", "JWT authentication protects API endpoints", 11),
        ("config.md", "BrainConfig", "BrainConfig class handles configuration loading", 9),
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


def test_rrf_merge_both_sources():
    faiss_ranked = [(1, 0.9), (2, 0.8), (3, 0.7)]
    fts_ranked = [(2, 0), (4, 1), (1, 2)]
    merged = _rrf_merge(faiss_ranked, fts_ranked, k=60)
    ids = [nid for nid, _ in merged]
    assert 2 in ids[:2]
    assert len(merged) == 4


def test_rrf_merge_single_source():
    faiss_ranked = [(1, 0.9), (2, 0.8)]
    merged = _rrf_merge(faiss_ranked, [], k=60)
    assert len(merged) == 2


def test_hybrid_retrieve_finds_fts_only_match(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    _seed(db, vectors, mock_embedder)
    result = handle_brain_retrieve(
        db, vectors, mock_embedder, query="BrainConfig", limit=10, threshold=-1.0,
    )
    assert len(result) >= 1
    titles = [r["title"] for r in result]
    assert "BrainConfig" in titles


def test_hybrid_retrieve_chunk_result(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    note_id = db.upsert_note(
        path="big.md", title="Big Note", content="x" * 100,
        content_hash="h_big", region_idx=0, tags=[],
        word_count=600, created_at="2026-01-01", modified_at="2026-01-01",
    )
    note_vec = mock_embedder.embed(["big note content"])
    note_fids = vectors.add(note_vec)
    db.set_faiss_idx(note_id, note_fids[0])
    db.replace_chunks(note_id, [
        {"heading": "Architecture", "content": "The architecture uses microservices",
         "content_hash": "ch1", "word_count": 5, "chunk_idx": 0},
    ])
    chunk_vec = mock_embedder.embed(["The architecture uses microservices"])
    chunk_fids = vectors.add(chunk_vec)
    db.set_chunk_faiss_idx(note_id, 0, chunk_fids[0])
    result = handle_brain_retrieve(
        db, vectors, mock_embedder, query="microservices architecture",
        limit=10, threshold=-1.0,
    )
    assert len(result) >= 1
    db.close()


def test_hybrid_retrieve_dedup_note_and_chunk(tmp_path, mock_embedder):
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    content = "Same content for both"
    note_id = db.upsert_note(
        path="dup.md", title="Dup Note", content=content,
        content_hash="h_dup", region_idx=0, tags=[],
        word_count=3, created_at="2026-01-01", modified_at="2026-01-01",
    )
    note_vec = mock_embedder.embed([content])
    note_fids = vectors.add(note_vec)
    db.set_faiss_idx(note_id, note_fids[0])
    db.replace_chunks(note_id, [
        {"heading": "S", "content": content, "content_hash": "ch_dup",
         "word_count": 3, "chunk_idx": 0},
    ])
    chunk_vec = mock_embedder.embed([content])
    chunk_fids = vectors.add(chunk_vec)
    db.set_chunk_faiss_idx(note_id, 0, chunk_fids[0])
    result = handle_brain_retrieve(
        db, vectors, mock_embedder, query=content, limit=10, threshold=-1.0,
    )
    paths = [r["path"] for r in result]
    assert paths.count("dup.md") <= 1
    db.close()
