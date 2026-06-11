from brain_mcp.indexer.scanner import scan_vault, compute_content_hash


def test_scan_vault_finds_md_files(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    assert len(notes) == 3
    titles = {n["title"] for n in notes}
    assert titles == {"note1", "note2", "project1"}


def test_scan_vault_extracts_brain_tags(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert by_title["note1"]["region_idx"] == 3  # hippocampus
    assert by_title["note2"]["region_idx"] == 0  # praefrontaler-cortex


def test_scan_vault_folder_mapping(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={"Projects": 10})
    by_title = {n["title"]: n for n in notes}
    assert by_title["project1"]["region_idx"] == 10


def test_scan_vault_default_region(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert by_title["project1"]["region_idx"] == 9


def test_scan_vault_backlinks(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert "note2" in by_title["note1"]["backlink_titles"]
    assert "note1" in by_title["note2"]["backlink_titles"]


def test_scan_vault_content_hash(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    assert all("content_hash" in n for n in notes)
    assert all(len(n["content_hash"]) == 64 for n in notes)


def test_scan_vault_content_included(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert "routing" in by_title["note1"]["content"].lower()


def test_compute_content_hash_deterministic():
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    h3 = compute_content_hash("different")
    assert h1 == h2
    assert h1 != h3


def test_scan_vault_skips_obsidian_dir(tmp_vault):
    obs = tmp_vault / ".obsidian"
    obs.mkdir()
    (obs / "config.md").write_text("# Config", encoding="utf-8")
    notes = scan_vault(tmp_vault, folder_to_region={})
    titles = {n["title"] for n in notes}
    assert "config" not in titles


def test_scan_vault_skips_exclude_dirs(tmp_vault):
    """exclude_dirs keeps archived / derived dumps out of the index scan."""
    archiv = tmp_vault / "99 Archiv"
    archiv.mkdir()
    (archiv / "old.md").write_text("# Old archived note", encoding="utf-8")
    nested = archiv / "02 Projekte"
    nested.mkdir()
    (nested / "deep.md").write_text("# Deep archived note", encoding="utf-8")
    gout = tmp_vault / "graphify-out"
    gout.mkdir()
    (gout / "junk.md").write_text("# graphify junk", encoding="utf-8")

    notes = scan_vault(
        tmp_vault, folder_to_region={}, exclude_dirs=["99 Archiv", "graphify-out"]
    )
    titles = {n["title"] for n in notes}
    assert titles == {"note1", "note2", "project1"}  # only the 3 real notes
    assert "old" not in titles
    assert "deep" not in titles
    assert "junk" not in titles


def test_scan_vault_exclude_dirs_none_is_noop(tmp_vault):
    """Omitting exclude_dirs must not change the default scan."""
    assert len(scan_vault(tmp_vault, folder_to_region={})) == 3
    assert len(scan_vault(tmp_vault, folder_to_region={}, exclude_dirs=[])) == 3


# -----------------------------------------------------------------------
# Pipeline integration test
# -----------------------------------------------------------------------
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.indexer.pipeline import index_vault


def test_pipeline_chunks_long_notes(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    filler = " ".join(["word"] * 250)
    long_content = f"Intro.\n{filler}\n\n## Architecture\n\n{filler}\n\n## Testing\n\n{filler}\n#brain/praefrontaler-cortex"
    (vault / "long.md").write_text(long_content, encoding="utf-8")
    (vault / "short.md").write_text("Short note.\n#brain/hippocampus", encoding="utf-8")
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    count = index_vault(db, vectors, mock_embedder, vault, {})
    assert count == 2  # both notes embedded
    chunks = db.get_chunks_for_note(db.get_note_by_path("long.md")["id"])
    assert len(chunks) >= 2
    assert all(c["faiss_idx"] is not None for c in chunks)
    short_chunks = db.get_chunks_for_note(db.get_note_by_path("short.md")["id"])
    assert len(short_chunks) == 0
    assert vectors.size > 2  # notes + chunks
    db.close()


# -----------------------------------------------------------------------
# Prune: notes that vanish from the scan (deleted on disk OR newly excluded)
# must be removed from the DB *and* their vectors removed from FAISS, or a
# rebuild leaves orphan rows with stale faiss_idx -> corrupt search results.
# -----------------------------------------------------------------------

def test_delete_notes_not_in_returns_faiss_and_cascades(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    db.upsert_note(
        path="a.md", title="A", content="aaa", content_hash="h1", region_idx=9,
        tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01T00:00:00",
        faiss_idx=10,
    )
    b = db.upsert_note(
        path="b.md", title="B", content="bbb", content_hash="h2", region_idx=9,
        tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01T00:00:00",
        faiss_idx=20,
    )
    db.replace_chunks(b, [{"heading": "", "content": "bbb", "content_hash": "hc",
                           "word_count": 1, "chunk_idx": 0}])
    db.set_chunk_faiss_idx(b, 0, 99)

    removed = db.delete_notes_not_in({"a.md"})

    assert set(removed) == {20, 99}              # b's note vector + chunk vector
    assert db.get_note_by_path("b.md") is None   # b gone
    assert db.get_note_by_path("a.md") is not None
    assert db.get_chunks_for_note(b) == []       # chunks cascade-deleted
    db.close()


def test_delete_notes_not_in_keeps_everything_when_all_present(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    db.upsert_note(path="a.md", title="A", content="a", content_hash="h1", region_idx=9,
                   tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01T00:00:00")
    assert db.delete_notes_not_in({"a.md"}) == []
    assert db.get_note_by_path("a.md") is not None
    db.close()


def test_index_vault_prunes_newly_excluded_notes(tmp_path, mock_embedder):
    """Re-indexing with a new exclude removes the now-excluded note (DB + FAISS)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "keep.md").write_text("Keep me.\n#brain/hippocampus", encoding="utf-8")
    arch = vault / "99 Archiv"
    arch.mkdir()
    (arch / "drop.md").write_text("Drop me.\n#brain/hippocampus", encoding="utf-8")
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)

    index_vault(db, vectors, mock_embedder, vault, {})          # both indexed
    assert db.get_note_by_path("99 Archiv/drop.md") is not None
    size_before = vectors.size

    index_vault(db, vectors, mock_embedder, vault, {}, exclude_dirs=["99 Archiv"])
    assert db.get_note_by_path("99 Archiv/drop.md") is None     # pruned from DB
    assert db.get_note_by_path("keep.md") is not None
    assert vectors.size < size_before                           # vector removed from FAISS
    db.close()


def test_index_vault_force_rebuild_drops_excluded(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "keep.md").write_text("Keep.\n#brain/hippocampus", encoding="utf-8")
    arch = vault / "99 Archiv"
    arch.mkdir()
    (arch / "drop.md").write_text("Drop.\n#brain/hippocampus", encoding="utf-8")
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)

    index_vault(db, vectors, mock_embedder, vault, {})
    assert db.get_note_count() == 2

    index_vault(db, vectors, mock_embedder, vault, {},
                exclude_dirs=["99 Archiv"], force=True)
    assert db.get_note_count() == 1                             # stale DB row gone
    assert db.get_note_by_path("keep.md") is not None
    db.close()


def test_index_vault_empty_scan_does_not_wipe(tmp_path, mock_embedder):
    """Fail-safe: a scan that returns nothing must NOT prune the whole brain."""
    vault = tmp_path / "vault"
    vault.mkdir()
    sub = vault / "Region"
    sub.mkdir()
    (sub / "a.md").write_text("A.\n#brain/hippocampus", encoding="utf-8")
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)

    index_vault(db, vectors, mock_embedder, vault, {})
    assert db.get_note_count() == 1

    # Exclude the only region -> scan returns nothing -> guard prevents wipe.
    index_vault(db, vectors, mock_embedder, vault, {}, exclude_dirs=["Region"])
    assert db.get_note_count() == 1
    db.close()


def test_scan_vault_exclude_dirs_case_insensitive(tmp_vault):
    """A config entry in a different case must still exclude (Windows FS)."""
    arch = tmp_vault / "99 Archiv"
    arch.mkdir()
    (arch / "old.md").write_text("# Old", encoding="utf-8")
    notes = scan_vault(tmp_vault, folder_to_region={}, exclude_dirs=["99 archiv"])
    assert "old" not in {n["title"] for n in notes}
    assert len(notes) == 3


def test_force_rebuild_does_not_drop_note_vectors_on_id_collision(tmp_path, mock_embedder):
    """C1 regression: after a force reset, note ids restart at 0 and overlap the
    PREVIOUS build's chunk ids. The chunk phase must NOT remove those stale ids,
    or it silently deletes freshly-added note vectors.

    Setup that creates the collision: build one chunked note (note id 0, chunk
    ids 1,2,...), grow with short notes (they take ids after the chunks), then
    force-rebuild -> new note ids 0,1,2 collide with old chunk ids 1,2.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    filler = " ".join(["alpha"] * 250)
    (vault / "n0.md").write_text(
        f"Intro.\n{filler}\n\n## Part\n\n{filler}\n\n## More\n\n{filler}\n#brain/hippocampus",
        encoding="utf-8",
    )
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)

    index_vault(db, vectors, mock_embedder, vault, {}, force=True)   # n0 + its chunks

    (vault / "n1.md").write_text("Short one.\n#brain/hippocampus", encoding="utf-8")
    (vault / "n2.md").write_text("Short two.\n#brain/hippocampus", encoding="utf-8")
    index_vault(db, vectors, mock_embedder, vault, {})              # grow (ids after chunks)

    index_vault(db, vectors, mock_embedder, vault, {}, force=True)  # collide-prone rebuild

    # Every note must still own a live vector (none silently removed).
    for path in ("n0.md", "n1.md", "n2.md"):
        row = db.get_note_by_path(path)
        assert row is not None and row["faiss_idx"] is not None
        assert vectors.reconstruct(row["faiss_idx"]) is not None, f"{path} vector was removed (C1)"
    db.close()
