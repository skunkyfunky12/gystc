"""Embedding backend protocol and SentenceTransformer implementation."""

from __future__ import annotations

import sys
import threading
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol that any embedding backend must satisfy."""

    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerBackend:
    """Embedding backend using sentence-transformers with thread-safe model loading."""

    def __init__(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        eager: bool = False,
    ) -> None:
        self._model_name = model_name
        self._model = None
        self._load_lock = threading.RLock()
        self._ready = threading.Event()
        if eager:
            self._load()

    def _load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            import time
            t0 = time.perf_counter()
            print(
                f"Loading embedding model: {self._model_name}...", file=sys.stderr
            )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                elapsed = time.perf_counter() - t0
                print(f"Model loaded in {elapsed:.1f}s.", file=sys.stderr)
            except Exception as exc:
                import traceback
                print(f"ERROR loading embedding model: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            finally:
                self._ready.set()

    def _get_model(self):
        """Thread-safe model accessor. Loads on first call, returns cached reference."""
        with self._load_lock:
            if self._model is None:
                self._load()
            return self._model

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Block until the model is loaded. Returns False on timeout."""
        return self._ready.wait(timeout=timeout)

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded without blocking."""
        return self._ready.is_set() and self._model is not None

    @property
    def dimension(self) -> int:
        """Return embedding dimension. Dynamic when model is loaded, default 384 otherwise."""
        if self._model is not None:
            return self._model.get_embedding_dimension()
        return 384  # default for paraphrase-multilingual-MiniLM-L12-v2

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized embedding vectors."""
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(vectors, dtype=np.float32)
