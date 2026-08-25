"""Regression tests for the dashboard audit fixes (brain/web_widget.py + brain/web/).

Covers:
- Finding 6:  /api/reindex must respect the MCP single-writer lock (409 when held).
- Finding 20: /api/vault/ path containment must not be a string-prefix check.
- Finding 21: three.js must be vendored locally (no unpkg.com at runtime).
- Finding 29: the activity feed server must reject writes without the session token.
- Follow-up finding 8 (review round 2): the token compare must be constant-time
  (hmac.compare_digest over bytes), matching the daemon's guard.
- Follow-up finding 9 (review round 2): the activity_token file must be written
  owner-only (0o600; best-effort on Windows), since its readability gates writes.
"""
import json
import re
import secrets
import threading
import urllib.parse
import weakref
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from brain_mcp.storage.file_lock import WriterLock

# Qt is stubbed per-module via the shared conftest fixture (a collection-time
# sys.modules mock would leak into every later test module).
pytestmark = pytest.mark.usefixtures("qt_stubbed")

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "brain" / "web"
VENDOR_THREE = WEB_DIR / "vendor" / "three-0.160.0"


def _get(port, path, headers=None):
    req = Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        resp = urlopen(req, timeout=5)
        return resp.status, resp.read()
    except HTTPError as e:
        return e.code, e.read()


def _post(port, path, data=b"", headers=None):
    req = Request(f"http://127.0.0.1:{port}{path}", data=data,
                  headers=headers or {}, method="POST")
    try:
        resp = urlopen(req, timeout=10)
        body = resp.read()
        return resp.status, (json.loads(body) if body else {})
    except HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return e.code, {}


@pytest.fixture
def web_server(tmp_path, monkeypatch):
    """Dashboard API server wired to a temp data dir + vault named 'Claude Brain'
    so a sibling 'Claude Brainx' dir can expose the string-prefix bug."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = tmp_path / "Claude Brain"
    vault.mkdir()
    (vault / "note.md").write_text("# hello\n", encoding="utf-8")
    (data_dir / "config.json").write_text(
        json.dumps({"vault_path": str(vault)}), encoding="utf-8")
    monkeypatch.setenv("BRAIN_DATA_DIR", str(data_dir))
    monkeypatch.delenv("BRAIN_VAULT_PATH", raising=False)

    from brain.web_widget import _make_web_handler
    token = secrets.token_hex(16)  # generated, not literal -- keeps the secret-scan hook quiet
    handler_cls = _make_web_handler(tmp_path, token)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"port": port, "vault": vault, "data_dir": data_dir, "tmp": tmp_path,
           "token": token, "auth": {"Authorization": f"Bearer {token}"}}
    server.shutdown()


# ---------------------------------------------------------------- finding 20

def test_vault_file_inside_vault_served(web_server):
    (web_server["vault"] / "img.md").write_text("content", encoding="utf-8")
    code, body = _get(web_server["port"], "/api/vault/img.md", web_server["auth"])
    assert code == 200
    assert body == b"content"


def test_vault_sibling_prefix_dir_blocked(web_server):
    # 'Claude Brainx' starts with the vault dir name 'Claude Brain' -> a bare
    # str.startswith() containment check serves its files (CWE-22).
    sibling = web_server["tmp"] / "Claude Brainx"
    sibling.mkdir()
    (sibling / "secret.md").write_text("SECRET", encoding="utf-8")
    rel = urllib.parse.quote("../Claude Brainx/secret.md")
    code, body = _get(web_server["port"], f"/api/vault/{rel}", web_server["auth"])
    assert code == 403
    assert b"SECRET" not in body


def test_vault_plain_parent_traversal_blocked(web_server):
    (web_server["tmp"] / "outside.md").write_text("OUTSIDE", encoding="utf-8")
    rel = urllib.parse.quote("../outside.md")
    code, body = _get(web_server["port"], f"/api/vault/{rel}", web_server["auth"])
    assert code == 403
    assert b"OUTSIDE" not in body


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="no symlink support")
def test_vault_symlink_escape_blocked(web_server):
    # A symlink inside the vault pointing outside must not be served.
    target = web_server["tmp"] / "linked-secret.md"
    target.write_text("LINKED", encoding="utf-8")
    link = web_server["vault"] / "innocent.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not permitted (Windows non-admin)")
    code, body = _get(web_server["port"], "/api/vault/innocent.md", web_server["auth"])
    assert code == 403
    assert b"LINKED" not in body


# ----------------------------------------------------------------- finding 6

class _FakeBackend:
    """Stands in for SentenceTransformerBackend so reindex tests stay fast."""

    def __init__(self, model_name=None, **kwargs):
        self._model_name = model_name

    @property
    def dimension(self):
        return 384


@pytest.fixture
def stub_heavy_indexing(monkeypatch):
    import brain_mcp.indexer.embedder as emb_mod
    import brain_mcp.indexer.pipeline as pipe_mod
    monkeypatch.setattr(emb_mod, "SentenceTransformerBackend", _FakeBackend)
    monkeypatch.setattr(pipe_mod, "index_vault", lambda *a, **k: 3)


def test_reindex_409_when_writer_lock_held(web_server, stub_heavy_indexing):
    lock = WriterLock(web_server["data_dir"] / "writer.lock")
    assert lock.acquire()
    try:
        code, data = _post(web_server["port"], "/api/reindex", headers=web_server["auth"])
        assert code == 409
        assert "error" in data
    finally:
        lock.release()


def test_reindex_runs_and_releases_lock_when_free(web_server, stub_heavy_indexing):
    import time
    code, data = _post(web_server["port"], "/api/reindex", headers=web_server["auth"])
    assert code == 200
    assert data.get("indexed") == 3
    # The lock must be released after the reindex finished. The response is
    # written before the handler's finally block runs, so poll briefly.
    lock = WriterLock(web_server["data_dir"] / "writer.lock")
    deadline = time.time() + 5
    acquired = False
    while time.time() < deadline:
        acquired = lock.acquire()
        if acquired:
            break
        time.sleep(0.05)
    assert acquired, "writer.lock was not released after reindex"
    lock.release()


# ---------------------------------------------------------------- finding 29

class _FakeWidget:
    def __init__(self):
        self.received = []

    def push_activity(self, data):
        self.received.append(data)


@pytest.fixture
def activity_server():
    from brain.web_widget import _make_activity_handler
    widget = _FakeWidget()
    token = secrets.token_hex(16)  # generated, not literal — keeps the secret-scan hook quiet
    handler_cls = _make_activity_handler(weakref.ref(widget), token)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"port": port, "widget": widget, "token": token}
    server.shutdown()


def _activity_post(srv, headers):
    body = json.dumps({"tag": "X", "text": "spoof", "tagClass": "tag-tool"}).encode()
    return _post(srv["port"], "/", body, headers)


def test_activity_post_without_token_rejected(activity_server):
    code, _ = _activity_post(activity_server, {"Content-Type": "application/json"})
    assert code == 401
    assert activity_server["widget"].received == []


def test_activity_post_wrong_token_rejected(activity_server):
    code, _ = _activity_post(activity_server, {
        "Content-Type": "application/json",
        "Authorization": "Bearer wrong-token",
    })
    assert code == 401
    assert activity_server["widget"].received == []


def test_activity_post_with_token_accepted(activity_server):
    code, _ = _activity_post(activity_server, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {activity_server['token']}",
    })
    assert code == 200
    assert len(activity_server["widget"].received) == 1
    assert activity_server["widget"].received[0]["text"] == "spoof"


def test_activity_post_with_origin_header_rejected(activity_server):
    # Browser-originated cross-origin POSTs always carry an Origin header;
    # the hook (urllib) never does. Reject them even with a valid token.
    code, _ = _activity_post(activity_server, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {activity_server['token']}",
        "Origin": "http://evil.example",
    })
    assert code == 403
    assert activity_server["widget"].received == []


# ------------------------------------------ follow-up finding 8 (review round 2)

def test_activity_auth_uses_constant_time_compare():
    import inspect
    import brain.web_widget as ww
    src = inspect.getsource(ww._make_activity_handler)
    assert "compare_digest" in src, \
        "token check must be constant-time (hmac.compare_digest), like the daemon's"
    assert "auth !=" not in src, "plain != comparison leaks a timing oracle"


def test_activity_post_non_ascii_auth_header_rejected_not_crashed(activity_server):
    # str-mode compare_digest raises TypeError on non-ASCII input; the handler
    # must compare bytes so a hostile header yields a clean 401, not a crash.
    code, _ = _activity_post(activity_server, {
        "Content-Type": "application/json",
        "Authorization": "Bearer töken",
    })
    assert code == 401
    assert activity_server["widget"].received == []


# ------------------------------------------ follow-up finding 9 (review round 2)

def test_activity_token_file_written_owner_only(tmp_path, monkeypatch):
    import os
    import brain.web_widget as ww

    seen = {}
    real_open = os.open

    def spy_open(path, flags, mode=0o777, *args, **kwargs):
        if Path(path).name == "activity_token":
            seen["mode"] = mode
            seen["flags"] = flags
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(ww.os, "open", spy_open)
    token_path = tmp_path / "activity_token"
    ww._write_activity_token(token_path, "sessiontoken123")
    assert seen["mode"] == 0o600, "token file must be created owner-only"
    assert seen["flags"] & os.O_TRUNC, "a stale token must be truncated"
    assert token_path.read_text(encoding="utf-8") == "sessiontoken123"
    if os.name == "posix":
        assert (token_path.stat().st_mode & 0o777) == 0o600


def test_activity_token_preexisting_world_readable_file_tightened(tmp_path):
    # O_CREAT's mode does not retrofit an existing file: a leftover 0644 token
    # from an old session must be truncated AND re-chmodded to 0600.
    import os
    import brain.web_widget as ww

    token_path = tmp_path / "activity_token"
    token_path.write_text("old-leaky-token", encoding="utf-8")
    if os.name == "posix":
        os.chmod(token_path, 0o644)
    ww._write_activity_token(token_path, "fresh")
    assert token_path.read_text(encoding="utf-8") == "fresh"
    if os.name == "posix":
        assert (token_path.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------- finding 21

def test_index_html_has_no_remote_script_origins():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "unpkg.com" not in html
    assert '"three": "./vendor/three-0.160.0/build/three.module.js"' in html
    assert '"three/addons/": "./vendor/three-0.160.0/examples/jsm/"' in html


def test_vendored_three_core_is_nontrivial():
    three = VENDOR_THREE / "build" / "three.module.js"
    assert three.is_file()
    assert three.stat().st_size > 100_000  # real three.js build is ~1.2 MB


def test_vendored_addon_imports_resolve():
    # Every `three/addons/...` import in brain-app.js must exist locally, and
    # every relative import inside the vendored jsm files must resolve too —
    # otherwise the page breaks offline exactly like the CDN version did.
    app = (WEB_DIR / "brain-app.js").read_text(encoding="utf-8")
    jsm = VENDOR_THREE / "examples" / "jsm"
    addons = re.findall(r"from\s+['\"]three/addons/([^'\"]+)['\"]", app)
    assert addons, "brain-app.js should import three addons"
    for addon in addons:
        assert (jsm / addon).is_file(), f"missing vendored addon: {addon}"
    for f in jsm.rglob("*.js"):
        for spec in re.findall(r"from\s+['\"](\.[^'\"]+)['\"]", f.read_text(encoding="utf-8")):
            resolved = (f.parent / spec).resolve()
            assert resolved.is_file(), f"{f.name}: unresolved import {spec}"


# ------------------------------------- review round 2 (0/5): reindex recovery

def test_reindex_clears_stale_stamps_when_index_lost(web_server, stub_heavy_indexing):
    """No index.faiss on disk but rows still stamped => the stamps are stale
    (corrupt index was deleted, or the writer crashed before saving). The
    dashboard reindex must clear them like the CLI/server writer paths do,
    or nothing re-embeds and fresh ids collide with the stale ones."""
    from brain_mcp.storage.database import BrainDB

    db_path = web_server["data_dir"] / "brain.db"
    db = BrainDB(db_path)
    db.upsert_note(
        path="ghost.md", title="Ghost", content="body", content_hash="h1",
        region_idx=3, tags=[], word_count=1,
        created_at="2026-01-01", modified_at="2026-01-01", faiss_idx=7,
    )
    assert db.has_faiss_stamps()
    db.close()

    code, data = _post(web_server["port"], "/api/reindex", headers=web_server["auth"])
    assert code == 200, data

    db = BrainDB(db_path)
    try:
        assert not db.has_faiss_stamps(), \
            "reindex must clear stale faiss_idx stamps when the index is lost"
    finally:
        db.close()


# ----------------------------------------- ox-alpha review 2026-08-23, finding 4
# /api/stats hardcoded dimension=384 while VectorStore.load() DELETES an index
# whose dimension does not match. With a non-384 model configured, merely
# opening the dashboard (brain-app.js fetches /api/stats on load) destroyed
# index.faiss -- silently, because the caller swallowed everything. The DB kept
# its faiss_idx stamps, so nothing re-embedded afterwards either.

def _write_index(path: Path, dimension: int, count: int = 3) -> None:
    import numpy as np
    from brain_mcp.indexer.vector_store import VectorStore
    store = VectorStore(dimension=dimension)
    store.add(np.ones((count, dimension), dtype=np.float32))
    store.save(path)


def test_stats_does_not_delete_index_with_non_384_dimension(web_server):
    index_path = web_server["data_dir"] / "index.faiss"
    _write_index(index_path, dimension=768)
    before = index_path.read_bytes()

    code, _ = _get(web_server["port"], "/api/stats", web_server["auth"])

    assert code == 200
    assert index_path.exists(), \
        "/api/stats must not delete the FAISS index of a non-384 model"
    assert index_path.read_bytes() == before, "/api/stats must not rewrite the index"


def test_stats_counts_vectors_of_non_384_index(web_server):
    _write_index(web_server["data_dir"] / "index.faiss", dimension=768, count=5)

    code, body = _get(web_server["port"], "/api/stats", web_server["auth"])

    assert code == 200
    assert json.loads(body)["vectors"] == 5


def test_stats_counts_vectors_of_384_index(web_server):
    _write_index(web_server["data_dir"] / "index.faiss", dimension=384, count=2)

    code, body = _get(web_server["port"], "/api/stats", web_server["auth"])

    assert code == 200
    assert json.loads(body)["vectors"] == 2


def test_stats_reports_zero_vectors_when_no_index_exists(web_server):
    # The handler dropped its `index_path.exists()` guard when it stopped using
    # VectorStore.load(); read_index_stats() answers None for a missing file and
    # the panel must still render rather than 500.
    assert not (web_server["data_dir"] / "index.faiss").exists()

    code, body = _get(web_server["port"], "/api/stats", web_server["auth"])

    assert code == 200
    assert json.loads(body)["vectors"] == 0


# ----------------------------------- CI finding 2026-08-25 (macos-latest, run 1)
# A handler that answers 401/403/404 without reading the request body leaves the
# received bytes sitting in the socket. BaseHTTPRequestHandler then closes the
# connection -- and on BSD/macOS closing a socket that still holds unread data
# sends an RST instead of a FIN, so the client raises ConnectionResetError
# instead of seeing the status we just wrote. Windows and Linux tolerate it,
# which is why three long-standing activity-feed tests only went red once CI
# started running the suite on macOS. A rejected hook post must report "401",
# not "connection reset by peer", or debugging points at the network instead of
# at the token. Asserted white-box so it holds on every platform.

def _run_post_with_body(handler_cls, headers, body=b'{"tag":"X","text":"t"}', **attrs):
    """Drive one do_POST against a handler instance with a real body buffer."""
    import io
    handler = object.__new__(handler_cls)
    for name, value in attrs.items():
        setattr(handler, name, value)
    handler.headers = {"Content-Length": str(len(body)), **headers}
    handler.rfile = io.BytesIO(body)
    sent = []
    handler.send_response = lambda code, *a: sent.append(code)
    handler.end_headers = lambda: None
    handler.do_POST()
    return sent, handler.rfile.read()


def test_activity_reject_without_token_drains_body(activity_server):
    from brain.web_widget import _make_activity_handler
    handler_cls = _make_activity_handler(weakref.ref(_FakeWidget()), "irrelevant")

    sent, leftover = _run_post_with_body(handler_cls, {})

    assert sent == [401]
    assert leftover == b"", "unread body -> RST instead of 401 on macOS"


def test_activity_reject_with_origin_drains_body(activity_server):
    from brain.web_widget import _make_activity_handler
    handler_cls = _make_activity_handler(weakref.ref(_FakeWidget()), "irrelevant")

    sent, leftover = _run_post_with_body(handler_cls, {"Origin": "http://evil.example"})

    assert sent == [403]
    assert leftover == b"", "unread body -> RST instead of 403 on macOS"


def test_web_handler_unknown_post_path_drains_body(tmp_path):
    from brain.web_widget import _make_web_handler
    token = secrets.token_hex(16)
    handler_cls = _make_web_handler(tmp_path, token)

    sent, leftover = _run_post_with_body(
        handler_cls, {"Authorization": f"Bearer {token}"}, path="/api/nope")

    assert sent == [404]
    assert leftover == b"", "unread body -> RST instead of 404 on macOS"


# ----------------------------------- ox-alpha review 2026-08-23, finding 1 (HIGH)
# /api/config (POST rewrites the config), /api/reindex, /api/search and
# /api/vault/<path> (serves any .md or attachment in the vault) ran on an
# ephemeral 127.0.0.1 port with no token and no origin check -- while the
# activity feed on the FIXED port 9500 had both. Two consequences: any local
# process could read vault contents over HTTP, and a browser tab could reach the
# state-changing endpoints (the body is parsed as JSON regardless of
# Content-Type, so text/plain avoids the preflight entirely). The dashboard's own
# page gets the token injected by _setup_page, so a browser tab cannot obtain it.

API_PATHS_GET = ["/api/config", "/api/stats", "/api/vault/note.md"]
API_PATHS_POST = ["/api/config", "/api/reindex", "/api/search"]


@pytest.mark.parametrize("path", API_PATHS_GET)
def test_api_get_without_token_rejected(web_server, path):
    code, body = _get(web_server["port"], path)
    assert code == 401
    assert b"hello" not in body, "vault content leaked to an unauthenticated reader"


@pytest.mark.parametrize("path", API_PATHS_POST)
def test_api_post_without_token_rejected(web_server, path):
    code, _ = _post(web_server["port"], path, json.dumps({"log_level": "DEBUG"}).encode())
    assert code == 401


def test_api_post_without_token_does_not_change_config(web_server):
    before = (web_server["data_dir"] / "config.json").read_text(encoding="utf-8")

    code, _ = _post(web_server["port"], "/api/config",
                    json.dumps({"log_level": "DEBUG"}).encode())

    assert code == 401
    assert (web_server["data_dir"] / "config.json").read_text(encoding="utf-8") == before


def test_api_wrong_token_rejected(web_server):
    code, _ = _get(web_server["port"], "/api/stats",
                   {"Authorization": "Bearer not-the-session-token"})
    assert code == 401


def test_api_non_ascii_auth_header_rejected_not_crashed(web_server):
    # str-mode compare_digest raises TypeError on non-ASCII input; comparing
    # bytes must yield a clean 401 instead of a 500.
    code, _ = _get(web_server["port"], "/api/stats", {"Authorization": "Bearer töken"})
    assert code == 401


def test_api_with_token_still_works(web_server):
    code, body = _get(web_server["port"], "/api/stats", web_server["auth"])
    assert code == 200
    assert "notes" in json.loads(body)


def test_api_foreign_origin_rejected_even_with_token(web_server):
    # Defense in depth: a token leaked into a page must still not be usable
    # cross-origin. Same rule the activity feed applies on its fixed port.
    code, _ = _get(web_server["port"], "/api/stats",
                   {**web_server["auth"], "Origin": "http://evil.example"})
    assert code == 403


def test_api_own_origin_accepted(web_server):
    # Chrome sends Origin on same-origin POST, so the dashboard's own requests
    # carry it -- rejecting every Origin outright would break the page itself.
    port = web_server["port"]
    code, _ = _get(port, "/api/stats",
                   {**web_server["auth"], "Origin": f"http://127.0.0.1:{port}"})
    assert code == 200


def test_static_files_served_without_token(web_server):
    # The page shell must load before any script can send a token.
    (web_server["tmp"] / "index.html").write_text("<h1>shell</h1>", encoding="utf-8")
    code, body = _get(web_server["port"], "/index.html")
    assert code == 200
    assert b"shell" in body


def test_page_receives_the_api_token_at_document_creation():
    # If the injection is dropped, every /api/* call 401s and the dashboard
    # comes up empty -- a failure mode no HTTP-level test would notice, because
    # the server would be behaving exactly as asked.
    # Read the file rather than inspect the class: under the Qt stub the widget's
    # base is a MagicMock, so attribute lookup never reaches the real method.
    module_src = (REPO_ROOT / "brain" / "web_widget.py").read_text(
        encoding="utf-8").replace(chr(13), "")   # the repo checks out CRLF
    start = module_src.index("def _setup_page")
    next_method = module_src.find(chr(10) + "    def ", start + 1)
    src = module_src[start:] if next_method < 0 else module_src[start:next_method]
    assert "window.__apiToken" in src, "the page cannot authenticate without the token"
    assert "DocumentCreation" in src, "the token must exist before brain-app.js runs"


def test_dashboard_js_sends_the_token_on_every_api_call():
    import re
    js = (WEB_DIR / "brain-app.js").read_text(encoding="utf-8")
    bare = re.findall(r"(?<![\w.])fetch\(\s*'/api[^']*'", js)
    assert not bare, f"these API calls bypass apiFetch and will 401: {bare}"
    assert "window.__apiToken" in js


# ---------------------------------- ox-alpha review 2026-08-23, finding 3 (part)
# The dashboard kept its own BrainDB + SentenceTransformer + FAISS for search,
# and /api/reindex built a SECOND embedder and vector store next to it -- two
# models in one process while a reindex runs. Sharing is also the more correct
# behaviour: rebuilding into a private copy left the cached store (the one
# search answers from) holding the pre-reindex index until the dashboard
# restarted. Both objects are internally locked (BrainDB: check_same_thread=False
# + RLock, VectorStore: RLock), so the handler thread may share them.

@pytest.fixture
def cached_search_state(monkeypatch, tmp_path):
    """Pre-populate the module cache the way a first search would."""
    import brain.web_widget as ww
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.indexer.vector_store import VectorStore

    db = BrainDB(tmp_path / "cached.db")
    vectors = VectorStore(dimension=384)
    embedder = _FakeBackend()
    monkeypatch.setitem(ww._search_state_cache, "instance", (db, vectors, embedder))
    yield {"db": db, "vectors": vectors, "embedder": embedder}
    db.close()
    ww._search_state_cache.pop("instance", None)


def test_reindex_reuses_the_cached_model_instead_of_loading_a_second(
        web_server, cached_search_state, monkeypatch):
    import brain_mcp.indexer.embedder as emb_mod
    import brain_mcp.indexer.pipeline as pipe_mod

    def _refuse(*args, **kwargs):
        raise AssertionError("reindex loaded a second embedding model")

    monkeypatch.setattr(emb_mod, "SentenceTransformerBackend", _refuse)
    monkeypatch.setattr(pipe_mod, "index_vault", lambda *a, **k: 1)

    code, data = _post(web_server["port"], "/api/reindex", headers=web_server["auth"])

    assert code == 200
    assert data.get("indexed") == 1


def test_reindex_rebuilds_into_the_store_search_answers_from(
        web_server, cached_search_state, monkeypatch):
    import numpy as np
    import brain_mcp.indexer.embedder as emb_mod
    import brain_mcp.indexer.pipeline as pipe_mod

    monkeypatch.setattr(emb_mod, "SentenceTransformerBackend", _FakeBackend)

    def _fake_index(db, vectors, embedder, vault, mapping):
        vectors.add(np.ones((2, 384), dtype=np.float32))
        return 2

    monkeypatch.setattr(pipe_mod, "index_vault", _fake_index)

    code, _ = _post(web_server["port"], "/api/reindex", headers=web_server["auth"])

    assert code == 200
    assert cached_search_state["vectors"].size == 2, \
        "the reindex wrote into a private store; search keeps answering from the old one"


def test_reset_search_state_closes_the_database(tmp_path):
    import brain.web_widget as ww
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.indexer.vector_store import VectorStore

    db = BrainDB(tmp_path / "closed.db")
    ww._search_state_cache["instance"] = (db, VectorStore(dimension=384), _FakeBackend())

    ww._reset_search_state()

    assert "instance" not in ww._search_state_cache
    with pytest.raises(Exception):
        db.get_all_notes()


def test_closing_the_widget_releases_the_cached_database():
    # Instantiating the widget needs a real Qt stack; assert the wiring instead,
    # the same way the token-injection guard does.
    module_src = (REPO_ROOT / "brain" / "web_widget.py").read_text(
        encoding="utf-8").replace(chr(13), "")
    start = module_src.index("def closeEvent")
    nxt = module_src.find(chr(10) + "    def ", start + 1)
    src = module_src[start:] if nxt < 0 else module_src[start:nxt]
    assert "_reset_search_state()" in src


def test_dashboard_js_parses():
    """The token wiring touched brain-app.js, and nothing else in the suite would
    notice a syntax error there -- the dashboard would simply come up blank."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    subprocess.run([node, "--check", str(WEB_DIR / "brain-app.js")], check=True,
                   capture_output=True)
