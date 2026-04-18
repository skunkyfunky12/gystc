"""Thread-safe FAISS vector store for semantic search.

Uses IndexIDMap2(IndexFlatIP) for stable ID-based addition and removal,
preventing ghost vectors from deleted/edited notes.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Thread-safe FAISS vector store with inner-product similarity.

    Uses IndexIDMap2 wrapping IndexFlatIP so vectors can be added with
    explicit IDs and removed individually -- critical for keeping the
    index consistent when notes are edited or deleted.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        base = faiss.IndexFlatIP(dimension)
        self._index: faiss.Index = faiss.IndexIDMap2(base)
        self._lock = threading.RLock()
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        with self._lock:
            return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray) -> list[int]:
        """Add vectors and return their assigned IDs.

        Vectors are L2-normalised before insertion so that inner-product
        similarity equals cosine similarity.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        with self._lock:
            ids = list(range(self._next_id, self._next_id + len(vectors)))
            id_array = np.array(ids, dtype=np.int64)
            self._index.add_with_ids(vectors, id_array)
            self._next_id += len(vectors)
            return ids

    def remove(self, ids: list[int]) -> None:
        """Remove vectors by their IDs.

        Safe to call with IDs that are not in the index (FAISS ignores them).
        """
        if not ids:
            return
        with self._lock:
            id_array = np.array(ids, dtype=np.int64)
            self._index.remove_ids(id_array)

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Search for the *k* nearest neighbours of each query vector.

        Returns ``(scores, ids)`` where each is a list-of-lists (one
        inner list per query vector).  Empty index returns empty lists.
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        faiss.normalize_L2(query)

        with self._lock:
            if self._index.ntotal == 0:
                n_queries = query.shape[0]
                return [[] for _ in range(n_queries)], [[] for _ in range(n_queries)]

            # Clamp k to available vectors
            effective_k = min(k, self._index.ntotal)
            scores_raw, ids_raw = self._index.search(query, effective_k)

        # Convert to Python lists, filtering out sentinel -1 entries
        all_scores: list[list[float]] = []
        all_ids: list[list[int]] = []
        for i in range(len(scores_raw)):
            row_scores = []
            row_ids = []
            for j in range(len(ids_raw[i])):
                if ids_raw[i][j] != -1:
                    row_scores.append(float(scores_raw[i][j]))
                    row_ids.append(int(ids_raw[i][j]))
            all_scores.append(row_scores)
            all_ids.append(row_ids)

        return all_scores, all_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Atomically save the index to *path*.

        Writes to a temporary file first, then renames -- so a crash
        mid-write cannot corrupt an existing index file.
        """
        path = Path(path)
        tmp = path.with_suffix(".faiss.tmp")
        with self._lock:
            faiss.write_index(self._index, str(tmp))
        os.replace(str(tmp), str(path))

    @classmethod
    def load(cls, path: Path | str, dimension: int = 384) -> VectorStore:
        """Load an index from *path*, or return an empty store if the file
        does not exist or is corrupted."""
        store = cls(dimension=dimension)
        path = Path(path)
        if path.exists():
            try:
                store._index = faiss.read_index(str(path))
                # Recover next_id from the loaded index
                if store._index.ntotal > 0:
                    stored_ids = faiss.vector_to_array(store._index.id_map)
                    store._next_id = int(stored_ids.max()) + 1
            except Exception as e:
                print(
                    f"WARNING: FAISS index corrupted, rebuilding: {e}",
                    file=sys.stderr,
                )
                path.unlink(missing_ok=True)
        return store
