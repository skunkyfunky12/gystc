import numpy as np
from brain_mcp.indexer.vector_store import VectorStore


def _random_vectors(n, dim=384, seed=42):
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_add_and_search():
    store = VectorStore(dimension=384)
    vecs = _random_vectors(5)
    ids = store.add(vecs)
    assert ids == [0, 1, 2, 3, 4]
    assert store.size == 5
    scores, result_ids = store.search(vecs[0:1], k=3)
    assert result_ids[0][0] == 0
    assert scores[0][0] > 0.99


def test_search_returns_top_k():
    store = VectorStore(dimension=384)
    vecs = _random_vectors(10)
    store.add(vecs)
    scores, result_ids = store.search(vecs[0:1], k=5)
    assert len(result_ids[0]) == 5
    assert result_ids[0][0] == 0


def test_save_and_load(tmp_path):
    store = VectorStore(dimension=384)
    vecs = _random_vectors(5)
    store.add(vecs)
    path = tmp_path / "test.faiss"
    store.save(path)
    store2 = VectorStore.load(path, dimension=384)
    assert store2.size == 5
    scores, result_ids = store2.search(vecs[0:1], k=1)
    assert result_ids[0][0] == 0


def test_empty_store_search():
    store = VectorStore(dimension=384)
    query = _random_vectors(1)
    scores, result_ids = store.search(query, k=5)
    assert len(result_ids[0]) == 0


def test_load_nonexistent_returns_empty(tmp_path):
    path = tmp_path / "missing.faiss"
    store = VectorStore.load(path, dimension=384)
    assert store.size == 0


def test_remove_vectors():
    store = VectorStore(dimension=384)
    vecs = _random_vectors(5)
    ids = store.add(vecs)
    assert store.size == 5
    store.remove([ids[2]])
    assert store.size == 4
    # Removed vector should not appear in search results
    scores, result_ids = store.search(vecs[2:3], k=5)
    assert 2 not in result_ids[0]


def test_atomic_save_creates_no_tmp(tmp_path):
    store = VectorStore(dimension=384)
    vecs = _random_vectors(3)
    store.add(vecs)
    path = tmp_path / "test.faiss"
    store.save(path)
    assert path.exists()
    assert not path.with_suffix('.faiss.tmp').exists()
