"""Cross-encoder re-ranker for post-RRF refinement of search results."""
from __future__ import annotations

import sys
import threading
import time

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 20


class CrossEncoderReRanker:
    """Re-ranks search candidates using a cross-encoder model.

    Loads the model in a background thread on first use.
    If the model isn't ready yet, returns candidates unchanged (graceful fallback).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._load_thread: threading.Thread | None = None

    def start_loading(self) -> None:
        """Kick off background model loading."""
        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            t0 = time.perf_counter()
            print(f"Loading re-ranker model: {self._model_name}...", file=sys.stderr)
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                elapsed = time.perf_counter() - t0
                print(f"Re-ranker loaded in {elapsed:.1f}s.", file=sys.stderr)
                self._ready.set()
            except Exception as exc:
                print(f"ERROR loading re-ranker: {exc}", file=sys.stderr)
                self._ready.set()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._model is not None

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout=timeout) and self._model is not None

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-rank candidates by cross-encoder score. Falls back to original order if model not ready."""
        if not candidates or len(candidates) <= 2:
            return candidates
        if not self.is_ready:
            return candidates

        top = candidates[:RERANK_TOP_N]
        rest = candidates[RERANK_TOP_N:]

        pairs = [[query, c.get("snippet", c.get("title", ""))] for c in top]

        with self._lock:
            if self._model is None:
                return candidates
            scores = self._model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)

        scored = sorted(zip(top, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored] + rest
