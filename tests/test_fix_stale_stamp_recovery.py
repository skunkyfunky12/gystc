"""Regression tests: a failed embed must never leave a note stamped with the
PREVIOUS content's vector (detach-then-embed), and --force must clear stale
stamps before fresh ids are assigned (id collisions on partial failure).

Final-verification round: residuals of audit finding R1#2 ('index --force with
a failing embedder') reproduced empirically by the verifier — the same root
cause lives in four paths: the pipeline's changed-note batch, the --force
rebuild, brain_store, and version rollback. upsert_note keeps faiss_idx via
COALESCE, so 'changed hash + surviving stamp' silently maps a note to its OLD
content's vector forever (the reconcile only re-embeds faiss_idx IS NULL rows).
"""
from __future__ import annotations

from brain_mcp.indexer.pipeline import index_vault
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.tools.store import handle_brain_store
from brain_mcp.tools.versioning import handle_brain_history, handle_brain_rollback
from tests.conftest import MockEmbedder


class FailingEmbedder(MockEmbedder):
    def embed(self, texts):
        raise RuntimeError("model exploded")


class FailFirstEmbedder(MockEmbedder):
    """Fails the FIRST embed call (the notes batch), succeeds afterwards
    (the chunks batch) — the partial-failure shape from the finding."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient OOM")
        return super().embed(texts)


# ---------------------------------------------------------------- pipeline

def test_changed_note_with_failed_embed_stays_retryable(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("Original content.", encoding="utf-8")
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})
    old_idx = db.get_note_by_path("a.md")["faiss_idx"]
    assert old_idx is not None

    (vault / "a.md").write_text("Changed content.", encoding="utf-8")
    index_vault(db, vectors, FailingEmbedder(), vault, {})
    row = db.get_note_by_path("a.md")
    assert row["faiss_idx"] is None, \
        "failed embed must leave the changed row retryable (faiss_idx NULL), " \
        "not permanently mapped to the OLD content's vector"
    assert vectors.reconstruct(old_idx) is None, \
        "the old-content vector must be detached, not kept as a wrong mapping"

    # The next healthy run must actually pick it up again.
    index_vault(db, vectors, mock_embedder, vault, {})
    assert db.get_note_by_path("a.md")["faiss_idx"] is not None


def test_force_partial_failure_cannot_collide_ids(tmp_path, mock_embedder):
    """--force resets the store; if the notes batch fails but the chunks batch
    succeeds, fresh chunk ids 0..K must not collide with stale note stamps."""
    vault = tmp_path / "vault"
    vault.mkdir()
    # >500 words total so split_into_chunks actually produces chunks
    body = "\n\n".join(f"## Sec {i}\n" + ("word " * 200) for i in range(3))
    (vault / "a.md").write_text("# A\n" + body, encoding="utf-8")
    (vault / "b.md").write_text("# B\n" + body, encoding="utf-8")
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})  # healthy first build

    index_vault(db, vectors, FailFirstEmbedder(), vault, {}, force=True)

    note_stamps = set()
    chunk_stamps = set()
    for p in ("a.md", "b.md"):
        row = db.get_note_by_path(p)
        if row["faiss_idx"] is not None:
            note_stamps.add(row["faiss_idx"])
        for c in db.get_chunks_for_note(row["id"]):
            if c["faiss_idx"] is not None:
                chunk_stamps.add(c["faiss_idx"])
    assert chunk_stamps, "test setup: the chunks batch should have succeeded"
    assert not note_stamps & chunk_stamps, \
        f"stale note stamps collide with fresh chunk ids: {note_stamps & chunk_stamps}"
    assert not note_stamps, \
        "the failed note batch must leave NULL stamps (retryable), not stale ids"


# -------------------------------------------------------------- brain_store

def test_brain_store_failed_embed_stays_retryable(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    r1 = handle_brain_store(db, vectors, mock_embedder, vault,
                            title="Note", content="Original.",
                            region_idx=3, persist=True)
    old_idx = db.get_note_by_path(r1["path"])["faiss_idx"]
    assert old_idx is not None

    r2 = handle_brain_store(db, vectors, FailingEmbedder(), vault,
                            title="Note", content="Changed.",
                            region_idx=3, persist=True)
    assert r2["indexed"] is False
    row = db.get_note_by_path(r1["path"])
    assert row["faiss_idx"] is None, \
        "failed store embed must leave the row retryable, not stale-mapped"
    assert vectors.reconstruct(old_idx) is None


# ----------------------------------------------------------------- rollback

def test_rollback_failed_embed_stays_retryable(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    r1 = handle_brain_store(db, vectors, mock_embedder, vault,
                            title="Note", content="Version one.",
                            region_idx=3, persist=True)
    handle_brain_store(db, vectors, mock_embedder, vault,
                       title="Note", content="Version two.",
                       region_idx=3, persist=True)
    path = r1["path"]
    current_idx = db.get_note_by_path(path)["faiss_idx"]
    history = handle_brain_history(db, path=path)
    assert isinstance(history, list) and history, "a version must exist"

    result = handle_brain_rollback(db, vault, path=path,
                                   version_id=history[0]["id"],
                                   vectors=vectors, embedder=FailingEmbedder(),
                                   persist=True)
    assert result.get("rolled_back") is True
    row = db.get_note_by_path(path)
    assert row["faiss_idx"] is None, \
        "failed rollback re-embed must leave the row retryable, not pointing " \
        "at the pre-rollback content's vector"
    assert vectors.reconstruct(current_idx) is None
