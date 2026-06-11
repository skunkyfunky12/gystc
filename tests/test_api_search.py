import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from tests.conftest import MockEmbedder

# Qt is stubbed per-module via the shared conftest fixture (a collection-time
# sys.modules mock would leak into every later test module).
pytestmark = pytest.mark.usefixtures("qt_stubbed")

# data.regions is pure data (no PyQt6 dependency), so import the REAL module.
# Do NOT stub it into sys.modules — that previously leaked a fake (no colors,
# empty community map) into test_regions.py when it ran after this module.


def _seed_db(db, vectors, embedder):
    notes = [
        ("routing.md", "Express Routing", "Express routing handles HTTP requests", 3),
        ("auth.md", "JWT Authentication", "JWT authentication protects API endpoints", 11),
        ("config.md", "BrainConfig", "BrainConfig class handles configuration loading", 9),
    ]
    for path, title, content, region_idx in notes:
        note_id = db.upsert_note(
            path=path, title=title, content=content, content_hash=f"h_{path}",
            region_idx=region_idx, tags=[], word_count=len(content.split()),
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        vec = embedder.embed([content])
        faiss_ids = vectors.add(vec)
        db.set_faiss_idx(note_id, faiss_ids[0])


def test_search_endpoint_returns_results(tmp_path):
    from brain.web_widget import _make_web_handler, _search_state_cache

    db = BrainDB(tmp_path / "brain.db")
    embedder = MockEmbedder()
    vectors = VectorStore(dimension=384)
    _seed_db(db, vectors, embedder)

    _search_state_cache["instance"] = (db, vectors, embedder)

    handler_cls = _make_web_handler(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = json.dumps({"query": "routing", "limit": 10}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/api/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=5)
        data = json.loads(resp.read())
        assert "results" in data
        assert len(data["results"]) >= 1
        titles = [r["title"] for r in data["results"]]
        assert "Express Routing" in titles
    finally:
        server.shutdown()
        _search_state_cache.clear()
        db.close()


def test_search_endpoint_empty_query(tmp_path):
    from brain.web_widget import _make_web_handler, _search_state_cache

    db = BrainDB(tmp_path / "brain.db")
    embedder = MockEmbedder()
    vectors = VectorStore(dimension=384)
    _search_state_cache["instance"] = (db, vectors, embedder)

    handler_cls = _make_web_handler(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = json.dumps({"query": ""}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/api/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=5)
        data = json.loads(resp.read())
        assert "results" in data
        assert len(data["results"]) == 0
    finally:
        server.shutdown()
        _search_state_cache.clear()
        db.close()


def test_search_endpoint_body_limit(tmp_path):
    from brain.web_widget import _make_web_handler, _search_state_cache
    from urllib.error import HTTPError

    db = BrainDB(tmp_path / "brain.db")
    embedder = MockEmbedder()
    vectors = VectorStore(dimension=384)
    _search_state_cache["instance"] = (db, vectors, embedder)

    handler_cls = _make_web_handler(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = b"x" * 9000
        req = Request(
            f"http://127.0.0.1:{port}/api/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)
        assert exc_info.value.code == 413
    finally:
        server.shutdown()
        _search_state_cache.clear()
        db.close()
