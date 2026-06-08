"""The writer's watcher path (_handle_file_change) is now the sole indexer of
reader-written and external (Obsidian) edits, so it must keep CHUNK vectors in
sync -- not just the note-level vector. Otherwise search returns stale snippets
for edited notes and leaks orphaned chunk vectors.
"""
from __future__ import annotations

from brain_mcp.config import BrainConfig
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.server import BrainState, _handle_file_change
from brain_mcp.storage.database import BrainDB
from tests.conftest import MockEmbedder


def _chunked_content(marker: str) -> str:
    # > 500 words across two headings -> splits into chunks.
    sections = []
    for heading in ("Alpha", "Beta"):
        body = " ".join(f"{marker}word{i}" for i in range(300))
        sections.append(f"## {heading}\n{body}")
    return "# Title\n\n" + "\n\n".join(sections)


def _state(tmp_path) -> BrainState:
    vault = tmp_path / "vault"
    vault.mkdir()
    return BrainState(
        config=BrainConfig(data_dir=tmp_path, vault_path=vault),
        db=BrainDB(tmp_path / "brain.db"),
        vectors=VectorStore(dimension=384),
        embedder=MockEmbedder(),
    )


def test_watcher_refreshes_chunks_on_edit_without_leaking(tmp_path):
    state = _state(tmp_path)
    try:
        f = state.config.vault_path / "note.md"
        f.write_text(_chunked_content("aaa"), encoding="utf-8")
        _handle_file_change(state, str(f), "created")

        note = state.db.get_note_by_path("note.md")
        assert note is not None
        chunks1 = state.db.get_chunks_for_note(note["id"])
        assert len(chunks1) >= 2, "long note should produce chunk vectors"
        size_after_create = state.vectors.size

        # Edit the file -> watcher must converge chunk content to the new text and
        # NOT accumulate the old chunk vectors.
        f.write_text(_chunked_content("bbb"), encoding="utf-8")
        _handle_file_change(state, str(f), "modified")

        chunks2 = state.db.get_chunks_for_note(note["id"])
        assert chunks2, "chunks must still exist after edit"
        assert all("bbb" in c["content"] for c in chunks2), "chunk content must converge to the edit"
        assert all("aaa" not in c["content"] for c in chunks2)
        assert state.vectors.size == size_after_create, "old chunk vectors leaked on edit"
    finally:
        state.db.close()


def test_watcher_delete_removes_chunk_vectors(tmp_path):
    state = _state(tmp_path)
    try:
        f = state.config.vault_path / "note.md"
        f.write_text(_chunked_content("aaa"), encoding="utf-8")
        _handle_file_change(state, str(f), "created")
        assert state.vectors.size > 0

        _handle_file_change(state, str(f), "deleted")
        assert state.db.get_note_by_path("note.md") is None
        assert state.vectors.size == 0, "note + chunk vectors must all be removed on delete"
    finally:
        state.db.close()
