"""brain_store writer vs reader behavior.

The single writer instance persists to brain.db + index.faiss. A read-only
instance (persist=False) still writes the .md file (source of truth) but must
NOT touch brain.db or index.faiss -- the writer's watcher indexes the file.
This is what keeps two coexisting servers from clobbering the shared index.
"""
from __future__ import annotations

from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.store import handle_brain_store
from tests.conftest import MockEmbedder


def test_reader_writes_file_but_not_db_or_index(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "brain.db")
    vectors = VectorStore(dimension=384)
    try:
        res = handle_brain_store(
            db, vectors, MockEmbedder(), vault,
            title="Reader Note", content="content stored from a reader instance",
            region_idx=0, persist=False,
        )
        assert (vault / "Reader Note.md").exists(), "file must still be written"
        assert res["indexed"] is False
        assert res.get("persisted_by") == "primary"
        assert db.get_note_by_path("Reader Note.md") is None, "reader must not write brain.db"
        assert vectors.size == 0, "reader must not write index.faiss"
    finally:
        db.close()


def test_writer_persists_to_db_and_index(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "brain.db")
    vectors = VectorStore(dimension=384)
    try:
        res = handle_brain_store(
            db, vectors, MockEmbedder(), vault,
            title="Writer Note", content="content stored from the writer instance",
            region_idx=0,  # persist defaults to True
        )
        assert (vault / "Writer Note.md").exists()
        assert res["indexed"] is True
        assert "persisted_by" not in res
        assert db.get_note_by_path("Writer Note.md") is not None
        assert vectors.size == 1
    finally:
        db.close()
