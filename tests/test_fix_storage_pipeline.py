# tests/test_fix_storage_pipeline.py
"""Regression tests for the reconcile-prune data-loss findings + commit batching.

Covers:
- Finding 5:  a file that drops out of the scan but still EXISTS on disk (e.g.
  a transient read error / sharing violation) must be SKIPPED by the prune,
  not deleted -- deletion CASCADEs its version history away.
- Finding 10: notes pruned because they moved into an excluded dir (curation
  archive) must keep their version history (tombstone table, re-attached when
  the note returns at the same path).
- Finding 14: reconciling an unchanged vault must not commit (= fsync) once
  per backlink edge.
"""
from __future__ import annotations

import brain_mcp.indexer.pipeline as pipeline
from brain_mcp.indexer.pipeline import index_vault
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB


def _build_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("Alpha note. [[b]]\n#brain/hippocampus", encoding="utf-8")
    (vault / "b.md").write_text("Beta note. [[a]]\n#brain/hippocampus", encoding="utf-8")
    return vault


# -----------------------------------------------------------------------
# Finding 5: transient read errors must not prune (and must be loud)
# -----------------------------------------------------------------------

def test_unreadable_file_is_not_pruned(tmp_path, mock_embedder, monkeypatch, capsys):
    vault = _build_vault(tmp_path)
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})

    # Edit b -> its pre-edit content becomes a version (the data at risk).
    (vault / "b.md").write_text("Beta note CHANGED. [[a]]\n#brain/hippocampus", encoding="utf-8")
    index_vault(db, vectors, mock_embedder, vault, {})
    b_id = db.get_note_by_path("b.md")["id"]
    assert len(db.get_versions(b_id)) == 1

    # Simulate a transient read error: the scan drops b.md although the file
    # is still on disk (scanner returns None on OSError and the path vanishes
    # from scanned_paths).
    real_scan = pipeline.scan_vault

    def flaky_scan(*args, **kwargs):
        return [n for n in real_scan(*args, **kwargs) if n["path"] != "b.md"]

    monkeypatch.setattr(pipeline, "scan_vault", flaky_scan)
    index_vault(db, vectors, mock_embedder, vault, {})

    row = db.get_note_by_path("b.md")
    assert row is not None, "transiently unreadable note was pruned from the DB"
    assert len(db.get_versions(row["id"])) == 1, "version history was destroyed"
    assert "b.md" in capsys.readouterr().err, "skip must be logged, not silent"
    db.close()


def test_deleted_file_is_still_pruned(tmp_path, mock_embedder):
    """Pin the original prune purpose: a file truly gone from disk IS pruned."""
    vault = _build_vault(tmp_path)
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})

    (vault / "b.md").unlink()
    index_vault(db, vectors, mock_embedder, vault, {})
    assert db.get_note_by_path("b.md") is None
    assert db.get_note_by_path("a.md") is not None
    db.close()


# -----------------------------------------------------------------------
# Finding 10: archive -> exclude -> prune must not destroy version history
# -----------------------------------------------------------------------

def test_archived_note_history_survives_prune_and_returns(tmp_path, mock_embedder):
    vault = _build_vault(tmp_path)
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})

    (vault / "b.md").write_text("Beta note v2. [[a]]\n#brain/hippocampus", encoding="utf-8")
    index_vault(db, vectors, mock_embedder, vault, {})
    b_id = db.get_note_by_path("b.md")["id"]
    assert len(db.get_versions(b_id)) == 1

    # Archive: move into an excluded dir -> the row is pruned (intentional)...
    arch = vault / "99 Archiv"
    arch.mkdir()
    (vault / "b.md").rename(arch / "b.md")
    index_vault(db, vectors, mock_embedder, vault, {}, exclude_dirs=["99 Archiv"])
    assert db.get_note_by_path("b.md") is None

    # ...but the version history must survive the ON DELETE CASCADE.
    cnt = db.execute(
        "SELECT COUNT(*) FROM archived_note_versions WHERE path=?", ("b.md",)
    ).fetchone()[0]
    assert cnt == 1, "archiving destroyed brain_versions history"

    # Un-archive: the note returns at the same path -> history re-attaches.
    (arch / "b.md").rename(vault / "b.md")
    index_vault(db, vectors, mock_embedder, vault, {}, exclude_dirs=["99 Archiv"])
    row = db.get_note_by_path("b.md")
    assert row is not None
    assert len(db.get_versions(row["id"])) == 1, "history not re-attached after un-archive"
    db.close()


def test_delete_note_preserves_versions_in_tombstone(tmp_path):
    """Watcher delete/rename path: versions survive in the tombstone table."""
    db = BrainDB(tmp_path / "t.db")
    nid = db.upsert_note(
        path="a.md", title="A", content="v1", content_hash="h1", region_idx=9,
        tags=[], word_count=1, created_at="2026-01-01", modified_at="2026-01-01T00:00:00",
    )
    db.save_version(nid, "v0", "A", 9, "[]", 1, "h0")
    db.delete_note("a.md")
    assert db.get_note_by_path("a.md") is None
    cnt = db.execute(
        "SELECT COUNT(*) FROM archived_note_versions WHERE path=?", ("a.md",)
    ).fetchone()[0]
    assert cnt == 1, "delete_note destroyed version history without tombstoning"
    db.close()


# -----------------------------------------------------------------------
# Finding 14: no per-edge commit storm on an unchanged reconcile
# -----------------------------------------------------------------------

class _CountingConn:
    """Wraps the sqlite connection and counts commit() calls."""

    def __init__(self, real):
        self._real = real
        self.commits = 0

    def commit(self):
        self.commits += 1
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_unchanged_reconcile_does_not_commit_per_edge(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    names = ["n0", "n1", "n2", "n3"]
    for n in names:
        links = " ".join(f"[[{m}]]" for m in names if m != n)
        (vault / f"{n}.md").write_text(
            f"{n} content. {links}\n#brain/hippocampus", encoding="utf-8"
        )
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})  # initial build

    counting = _CountingConn(db._conn)
    db._conn = counting
    index_vault(db, vectors, mock_embedder, vault, {})  # unchanged reconcile
    db._conn = counting._real
    # 4 notes x 3 backlinks = 12 edges; pre-fix this was >= 12 commits.
    assert counting.commits <= 3, (
        f"unchanged reconcile produced {counting.commits} commits -- "
        "edge/note/chunk writes must be batched, not one fsync per row"
    )
    db.close()


def test_upsert_edges_batch_inserts_and_updates(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    a = db.upsert_note(path="a.md", title="A", content="a", content_hash="h1", region_idx=9,
                       tags=[], word_count=1, created_at="c", modified_at="m")
    b = db.upsert_note(path="b.md", title="B", content="b", content_hash="h2", region_idx=9,
                       tags=[], word_count=1, created_at="c", modified_at="m")
    db.upsert_edges([(a, b, "B"), (b, a, "A")])
    assert len(db.get_edges_for_note(a)) == 2

    db.upsert_edges([(a, b, "B-renamed")])  # conflict -> update, not duplicate
    edges = {(e["source_id"], e["target_id"]): e["link_text"] for e in db.get_edges_for_note(a)}
    assert edges[(a, b)] == "B-renamed"
    assert len(edges) == 2
    db.close()


# -----------------------------------------------------------------------
# Review round 2, finding 6: re-embedding a changed note must remove the
# displaced (previous) vector -- it leaked as an orphan before.
# -----------------------------------------------------------------------

def test_reembed_removes_displaced_note_vector(tmp_path, mock_embedder):
    vault = _build_vault(tmp_path)
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    index_vault(db, vectors, mock_embedder, vault, {})
    old_idx = db.get_note_by_path("a.md")["faiss_idx"]
    assert old_idx is not None
    size_before = vectors.size

    (vault / "a.md").write_text(
        "Alpha note EDITED. [[b]]\n#brain/hippocampus", encoding="utf-8"
    )
    index_vault(db, vectors, mock_embedder, vault, {})

    new_idx = db.get_note_by_path("a.md")["faiss_idx"]
    assert new_idx != old_idx
    assert vectors.reconstruct(old_idx) is None, \
        "displaced note vector must be removed (orphan-vector leak)"
    assert vectors.reconstruct(new_idx) is not None
    assert vectors.size == size_before, \
        "store must not grow on re-embed (one in, one out)"
