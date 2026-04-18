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
    """Embedding backend using sentence-transformers with lazy, thread-safe model loading."""

    def __init__(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ) -> None:
        self._model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # double-check after acquiring lock
                return
            print(
                f"Loading embedding model: {self._model_name}...", file=sys.stderr
            )
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            print("Model loaded.", file=sys.stderr)

    @property
    def dimension(self) -> int:
        """Return embedding dimension. Dynamic when model is loaded, default 384 otherwise."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384  # default for paraphrase-multilingual-MiniLM-L12-v2

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized embedding vectors."""
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(vectors, dtype=np.float32)
