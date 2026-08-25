"""Integration tests: full indexing pipeline and incremental re-index."""
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.scanner import scan_vault


def test_full_index_pipeline(tmp_vault, mock_embedder, tmp_path):
    """End-to-end: scan vault -> populate DB -> embed -> search."""
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)

    notes = scan_vault(tmp_vault, folder_to_region={"Projects": 10})
    title_to_id: dict[str, int] = {}
    for note in notes:
        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id

    for note in notes:
        for bl_title in note.get("backlink_titles", []):
            src_id = title_to_id.get(note["title"])
            tgt_id = title_to_id.get(bl_title)
            if src_id and tgt_id and src_id != tgt_id:
                db.upsert_edge(src_id, tgt_id, link_text=bl_title)

    contents = [n["content"] for n in notes]
    vecs = mock_embedder.embed(contents)
    faiss_ids = vectors.add(vecs)
    for i, note in enumerate(notes):
        nid = title_to_id[note["title"]]
        db.set_faiss_idx(nid, faiss_ids[i])

    assert vectors.size == 3
    assert len(db.get_all_notes()) == 3

    from brain_mcp.tools.retrieve import handle_brain_retrieve
    results = handle_brain_retrieve(db, vectors, mock_embedder, query="routing HTTP", limit=3, threshold=0.0)
    assert len(results) >= 1

    from brain_mcp.tools.recent import handle_brain_recent
    recent = handle_brain_recent(db, days=365, limit=10)
    assert len(recent) == 3

    from brain_mcp.tools.related import handle_brain_related
    related = handle_brain_related(db, vectors, mock_embedder, title="note1", limit=5)
    assert len(related) >= 1

    db.close()


def test_incremental_reindex(tmp_vault, mock_embedder, tmp_path):
    """Only changed files get re-embedded."""
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)

    notes = scan_vault(tmp_vault, folder_to_region={})
    for note in notes:
        existing_hash = db.get_content_hash(note["path"])
        if existing_hash == note["content_hash"]:
            continue
        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        vec = mock_embedder.embed([note["content"]])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(note_id, faiss_ids[0])

    assert vectors.size == 3

    notes2 = scan_vault(tmp_vault, folder_to_region={})
    new_embeds = 0
    for note in notes2:
        existing_hash = db.get_content_hash(note["path"])
        if existing_hash == note["content_hash"]:
            continue
        new_embeds += 1

    assert new_embeds == 0
    db.close()
