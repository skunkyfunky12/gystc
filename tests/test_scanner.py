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
