# tests/test_fix_cli_vector.py
"""Regression tests for the cli-vector audit fixes:

1.  `brain_mcp index` must acquire the WriterLock and abort non-zero when a
    live writer holds it (instead of corrupting the shared faiss_idx mapping).
2.  `brain_mcp index --force` with a failing embedder must NOT overwrite the
    previous good index.faiss with an empty one and must NOT report success.
17. A corrupted index file must be signalled by VectorStore.load() and the
    CLI must clear stale faiss_idx so the next run actually re-embeds.
25. brain_related must reconstruct the stored vector instead of re-embedding,
    batch note lookups, and cap the neighbor fan-out.
"""
import argparse
import hashlib

import numpy as np
import pytest

from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.storage.database import BrainDB
from brain_mcp.storage.file_lock import WriterLock
from brain_mcp.tools.related import handle_brain_related


class DeterministicEmbedder:
    """Working embedder: text hash -> fixed 384-dim unit vector."""

    is_ready = True

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts):
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class FailingEmbedder:
    """Simulates a missing/uncached model (HF_HUB_OFFLINE=1 hard failure)."""

    is_ready = False

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts):
        raise RuntimeError("model not available (offline)")


class ExplodingEmbedder:
    """Asserts that embed() is never reached."""

    is_ready = True

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts):
        raise AssertionError("embed() must not be called")


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "alpha.md").write_text(
        "# Alpha\nRouting tables and HTTP maps.\n[[Beta]]", encoding="utf-8"
    )
    (vault / "beta.md").write_text(
        "# Beta\nAuthentication middleware and JWT tokens.\n[[Alpha]]", encoding="utf-8"
    )
    (vault / "gamma.md").write_text(
        "# Gamma\nPipelines move data between systems.", encoding="utf-8"
    )
    monkeypatch.setenv("BRAIN_DATA_DIR", str(data_dir))
    monkeypatch.delenv("BRAIN_VAULT_PATH", raising=False)
    return data_dir, vault


def _run_index(vault, embedder_factory, monkeypatch, force=False):
    """Invoke cmd_index in-process with a patched embedder backend."""
    import brain_mcp.indexer.embedder as emb_mod

    monkeypatch.setattr(emb_mod, "SentenceTransformerBackend", embedder_factory)
    from brain_mcp.__main__ import cmd_index

    cmd_index(argparse.Namespace(vault=str(vault), force=force))


def _assert_db_index_consistent(data_dir):
    """Every note's faiss_idx must be unique and reconstruct to the embedding
    of its CURRENT content -- i.e. no stale/collided id mappings."""
    emb = DeterministicEmbedder()
    db = BrainDB(data_dir / "brain.db")
    store = VectorStore.load(data_dir / "index.faiss", dimension=384)
    try:
        notes = db.get_all_notes()
        assert notes, "expected indexed notes"
        seen: set[int] = set()
        for row in notes:
            assert row["faiss_idx"] is not None, f"{row['path']}: never re-embedded"
            assert row["faiss_idx"] not in seen, f"{row['path']}: faiss_idx collision"
            seen.add(row["faiss_idx"])
            vec = store.reconstruct(row["faiss_idx"])
            assert vec is not None, f"{row['path']}: faiss_idx points at missing vector"
            expected = emb.embed([row["content"]])[0]
            assert np.allclose(vec, expected, atol=1e-5), f"{row['path']}: mapped to wrong vector"
        chunk_rows = db.execute(
            "SELECT faiss_idx FROM chunks WHERE faiss_idx IS NOT NULL"
        ).fetchall()
        for c in chunk_rows:
            assert c["faiss_idx"] not in seen, "chunk id collides with a note id"
            seen.add(c["faiss_idx"])
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Finding 1: CLI index must respect the WriterLock
# ---------------------------------------------------------------------------

def test_index_aborts_when_writer_lock_held(cli_env, monkeypatch, capsys):
    data_dir, vault = cli_env
    lock = WriterLock(data_dir / "writer.lock")
    assert lock.acquire()
    try:
        with pytest.raises(SystemExit) as exc:
            _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "writer.lock" in err
        # must abort BEFORE touching the shared DB/index
        assert not (data_dir / "brain.db").exists()
        assert not (data_dir / "index.faiss").exists()
    finally:
        lock.release()


def test_index_releases_lock_after_success(cli_env, monkeypatch):
    data_dir, vault = cli_env
    _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
    lock = WriterLock(data_dir / "writer.lock")
    assert lock.acquire(), "lock must be released after a successful run"
    lock.release()


def test_index_releases_lock_after_embed_failure(cli_env, monkeypatch):
    data_dir, vault = cli_env
    with pytest.raises(SystemExit):
        _run_index(vault, lambda name: FailingEmbedder(), monkeypatch, force=True)
    lock = WriterLock(data_dir / "writer.lock")
    assert lock.acquire(), "lock must be released even when indexing fails"
    lock.release()


# ---------------------------------------------------------------------------
# Finding 2: --force with failing embedder must not destroy the good index
# ---------------------------------------------------------------------------

def test_force_with_failing_embedder_keeps_index_and_exits_nonzero(
    cli_env, monkeypatch, capsys
):
    data_dir, vault = cli_env
    _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
    index_path = data_dir / "index.faiss"
    good_bytes = index_path.read_bytes()
    capsys.readouterr()  # drain run-1 output

    with pytest.raises(SystemExit) as exc:
        _run_index(vault, lambda name: FailingEmbedder(), monkeypatch, force=True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Done." not in err, "error path must not report success"
    assert "ERROR" in err
    # the previous good index must survive a failed --force rebuild
    assert index_path.read_bytes() == good_bytes
    # DB stays consistent with the kept index: stamps still resolve correctly
    db = BrainDB(data_dir / "brain.db")
    store = VectorStore.load(index_path, dimension=384)
    try:
        for row in db.get_all_notes():
            if row["faiss_idx"] is not None:
                assert store.reconstruct(row["faiss_idx"]) is not None
    finally:
        db.close()


def test_failed_force_then_recovery_has_no_stale_mapping(cli_env, monkeypatch):
    data_dir, vault = cli_env
    _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
    with pytest.raises(SystemExit):
        _run_index(vault, lambda name: FailingEmbedder(), monkeypatch, force=True)
    # embedder works again -> a plain incremental run must restore consistency
    _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
    _assert_db_index_consistent(data_dir)


# ---------------------------------------------------------------------------
# Finding 17: corrupted-index recovery must actually rebuild
# ---------------------------------------------------------------------------

def test_load_corrupt_sets_flag_and_deletes_file(tmp_path, capsys):
    path = tmp_path / "index.faiss"
    path.write_bytes(b"this is not a faiss index")
    store = VectorStore.load(path, dimension=384)
    assert store.size == 0
    assert store.corrupted_on_load is True
    assert not path.exists()
    assert "WARNING" in capsys.readouterr().err


def test_load_good_index_flag_false(tmp_path):
    fresh = VectorStore(dimension=384)
    assert fresh.corrupted_on_load is False
    vecs = DeterministicEmbedder().embed(["hello world"])
    fresh.add(vecs)
    path = tmp_path / "index.faiss"
    fresh.save(path)
    loaded = VectorStore.load(path, dimension=384)
    assert loaded.corrupted_on_load is False
    missing = VectorStore.load(tmp_path / "missing.faiss", dimension=384)
    assert missing.corrupted_on_load is False


def test_cli_index_rebuilds_after_corruption(cli_env, monkeypatch):
    data_dir, vault = cli_env
    _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
    # corrupt the index on disk (e.g. crash mid-write on an old version)
    (data_dir / "index.faiss").write_bytes(b"garbage" * 128)

    # a plain (non-force) run must clear stale faiss_idx and re-embed all
    _run_index(vault, lambda name: DeterministicEmbedder(), monkeypatch)
    _assert_db_index_consistent(data_dir)


# ---------------------------------------------------------------------------
# Finding 25: brain_related reconstructs, batches, and caps fan-out
# ---------------------------------------------------------------------------

def _seed_related(db, vectors, emb):
    notes = [
        ("a.md", "Routing", "Express routing handles HTTP requests"),
        ("b.md", "Auth", "JWT authentication protects endpoints"),
        ("c.md", "Middleware", "Middleware chain processes requests in order"),
    ]
    for path, title, content in notes:
        nid = db.upsert_note(
            path=path, title=title, content=content, content_hash=f"h_{path}",
            region_idx=0, tags=[], word_count=len(content.split()),
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        fid = vectors.add(emb.embed([content]))[0]
        db.set_faiss_idx(nid, fid)
    a = db.get_note_by_path("a.md")
    c = db.get_note_by_path("c.md")
    db.upsert_edge(a["id"], c["id"], link_text="Middleware")


def test_related_uses_reconstruct_not_reembed(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    _seed_related(db, vectors, DeterministicEmbedder())
    # ExplodingEmbedder fails the test if related re-embeds the source note
    result = handle_brain_related(
        db, vectors, ExplodingEmbedder(), title="Routing", limit=5
    )
    assert isinstance(result, list) and len(result) >= 1
    assert any(r["relation_type"] in ("semantic", "both") for r in result), \
        "semantic candidates must come from the reconstructed stored vector"
    db.close()


def test_related_falls_back_to_embed_when_vector_missing(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    emb = DeterministicEmbedder()
    _seed_related(db, vectors, emb)
    # dangling faiss_idx -> reconstruct returns None -> must fall back to embed
    a = db.get_note_by_path("a.md")
    db.set_faiss_idx(a["id"], 9999)

    calls = []

    class CountingEmbedder(DeterministicEmbedder):
        def embed(self, texts):
            calls.append(texts)
            return super().embed(texts)

    result = handle_brain_related(
        db, vectors, CountingEmbedder(), title="Routing", limit=5
    )
    assert isinstance(result, list)
    assert len(calls) == 1, "must fall back to embedding exactly once"
    db.close()


def test_related_batches_note_lookups(tmp_path):
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    _seed_related(db, vectors, DeterministicEmbedder())

    n1_calls = []
    original = db.get_note_by_id
    db.get_note_by_id = lambda nid: (n1_calls.append(nid), original(nid))[1]

    result = handle_brain_related(
        db, vectors, ExplodingEmbedder(), title="Routing", limit=5
    )
    assert isinstance(result, list) and len(result) >= 1
    assert n1_calls == [], "candidates must be fetched in one batched query"
    db.close()


def test_related_caps_neighbor_fanout(tmp_path, monkeypatch):
    monkeypatch.setattr("brain_mcp.tools.related.MAX_BFS_NEIGHBORS", 3)
    db = BrainDB(tmp_path / "t.db")
    vectors = VectorStore(dimension=384)
    hub_id = db.upsert_note(
        path="hub.md", title="Hub", content="hub of everything", content_hash="h_hub",
        region_idx=0, tags=[], word_count=3,
        created_at="2026-01-01", modified_at="2026-01-01",
    )
    for i in range(10):
        nid = db.upsert_note(
            path=f"n{i}.md", title=f"N{i}", content=f"note {i}", content_hash=f"h_{i}",
            region_idx=0, tags=[], word_count=2,
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        db.upsert_edge(hub_id, nid, link_text=f"N{i}")

    fetched = []
    original = db.get_notes_by_ids
    db.get_notes_by_ids = lambda ids: (fetched.append(len(ids)), original(ids))[1]

    result = handle_brain_related(
        db, vectors, ExplodingEmbedder(), title="Hub", limit=10, semantic=False
    )
    assert isinstance(result, list)
    assert all(n <= 3 for n in fetched), f"neighbor fan-out not capped: {fetched}"
    assert len(result) <= 3
    db.close()
