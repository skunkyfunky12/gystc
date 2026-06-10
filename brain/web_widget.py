"""WebEngine widget that hosts the Three.js brain renderer."""
from __future__ import annotations

import json
import sys
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QUrl, pyqtSlot, pyqtSignal, QObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineScript

from data.regions import COMMUNITY_TO_REGION, REGIONS

_WEB_DIR = Path(__file__).parent / "web"
if not _WEB_DIR.is_dir():
    _WEB_DIR = Path(getattr(sys, "_MEIPASS", "")) / "brain" / "web"
ACTIVITY_PORT = 9500


CONFIG_API_MAX_BODY = 8192
SEARCH_API_MAX_BODY = 8192

# Module-level search state cache (lazy-loaded)
_search_state_cache: dict = {}


def _get_search_state():
    """Return (db, vectors, embedder) tuple, cached after first call."""
    cached = _search_state_cache.get("instance")
    if cached is not None:
        return cached

    from brain_mcp.config import load_config
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.indexer.vector_store import VectorStore
    from brain_mcp.indexer.embedder import SentenceTransformerBackend

    config = load_config()
    db = BrainDB(config.db_path)
    embedder = SentenceTransformerBackend(config.model_name)
    vectors = VectorStore.load(config.index_path, dimension=embedder.dimension)
    threading.Thread(target=embedder._load, daemon=True).start()
    state = (db, vectors, embedder)
    _search_state_cache["instance"] = state
    return state


def _make_web_handler(directory: Path):
    """HTTP handler that serves static files AND a /api/config endpoint."""

    class WebHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            super().end_headers()

        def do_GET(self):
            if self.path == "/api/config":
                self._handle_config_get()
            elif self.path == "/api/stats":
                self._handle_stats_get()
            elif self.path.startswith("/api/vault/"):
                self._handle_vault_file()
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == "/api/config":
                self._handle_config_post()
            elif self.path == "/api/reindex":
                self._handle_reindex()
            elif self.path == "/api/search":
                self._handle_search()
            else:
                self.send_response(404)
                self.end_headers()

        def _handle_config_get(self):
            try:
                from brain_mcp.config import load_config
                config = load_config()
                data = {
                    "vault_path": str(config.vault_path) if config.vault_path else None,
                    "model_name": config.model_name,
                    "auto_index": config.auto_index,
                    "index_on_startup": config.index_on_startup,
                    "folder_to_region": config.folder_to_region,
                    "log_level": config.log_level,
                }
                self._json_response(200, data)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        def _handle_config_post(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length == 0:
                    self._json_response(400, {"error": "Empty request body"})
                    return
                if length > CONFIG_API_MAX_BODY:
                    self._json_response(413, {"error": "Body too large"})
                    return
                body = self.rfile.read(length).decode("utf-8")
                try:
                    updates = json.loads(body)
                except json.JSONDecodeError:
                    self._json_response(400, {"error": "Invalid JSON"})
                    return

                from brain_mcp.config import load_config, save_config
                config = load_config()

                if "auto_index" in updates:
                    config.auto_index = bool(updates["auto_index"])
                if "index_on_startup" in updates:
                    config.index_on_startup = bool(updates["index_on_startup"])
                if "log_level" in updates and updates["log_level"] in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    config.log_level = updates["log_level"]

                save_config(config)
                self._json_response(200, {"ok": True})
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        def _handle_stats_get(self):
            try:
                from brain_mcp.config import load_config
                config = load_config()
                notes_count = 0
                vectors_count = 0
                edges_count = 0
                if config.db_path.exists():
                    import sqlite3
                    conn = sqlite3.connect(str(config.db_path))
                    try:
                        cur = conn.cursor()
                        notes_count = cur.execute("SELECT count(*) FROM notes").fetchone()[0]
                        try:
                            edges_count = cur.execute("SELECT count(*) FROM edges").fetchone()[0]
                        except Exception:
                            pass
                    finally:
                        conn.close()
                if config.index_path.exists():
                    try:
                        from brain_mcp.indexer.vector_store import VectorStore
                        vs = VectorStore.load(config.index_path, dimension=384)
                        vectors_count = vs.size
                    except Exception:
                        pass
                self._json_response(200, {
                    "notes": notes_count,
                    "vectors": vectors_count,
                    "edges": edges_count,
                    "regions": 12,
                })
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        def _handle_reindex(self):
            try:
                from brain_mcp.config import load_config
                from brain_mcp.storage.database import BrainDB
                from brain_mcp.storage.file_lock import WriterLock
                from brain_mcp.indexer.vector_store import VectorStore
                from brain_mcp.indexer.embedder import SentenceTransformerBackend
                from brain_mcp.indexer.pipeline import index_vault
                import time

                config = load_config()
                if config.vault_path is None or not config.vault_path.is_dir():
                    self._json_response(400, {"error": "No vault_path configured"})
                    return
                # Single-writer invariant: only the writer.lock holder may rewrite
                # faiss_idx / persist index.faiss (see brain_mcp/server.py). If an
                # MCP writer instance is live, refuse instead of racing it.
                lock = WriterLock(config.data_dir / "writer.lock")
                if not lock.acquire():
                    print("reindex refused: an MCP writer instance holds writer.lock", file=sys.stderr)
                    self._json_response(409, {
                        "error": "Index is owned by a running GYSTC MCP instance. "
                                 "Close it (or reindex through it) and try again."
                    })
                    return
                try:
                    db = BrainDB(config.db_path)
                    try:
                        embedder = SentenceTransformerBackend(config.model_name)
                        vectors = VectorStore.load(config.index_path, dimension=embedder.dimension)
                        t0 = time.time()
                        count = index_vault(db, vectors, embedder, config.vault_path, config.folder_to_region)
                        vectors.save(config.index_path)
                        elapsed = round(time.time() - t0, 1)
                        self._json_response(200, {"indexed": count, "elapsed": elapsed})
                    finally:
                        db.close()
                finally:
                    lock.release()
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        def _handle_search(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length == 0 or length > SEARCH_API_MAX_BODY:
                    self._json_response(413 if length > SEARCH_API_MAX_BODY else 400,
                                        {"error": "Invalid body size"})
                    return
                body = self.rfile.read(length).decode("utf-8")
                try:
                    params = json.loads(body)
                except json.JSONDecodeError:
                    self._json_response(400, {"error": "Invalid JSON"})
                    return

                query = str(params.get("query", "")).strip()[:1000]
                if not query:
                    self._json_response(200, {"results": []})
                    return

                region = params.get("region")
                limit = max(1, min(int(params.get("limit", 20)), 50))

                from brain_mcp.tools.retrieve import handle_brain_retrieve
                db, vectors, embedder = _get_search_state()
                fts_only = not embedder.is_ready
                results = handle_brain_retrieve(
                    db, vectors, embedder,
                    query=query, region=region, limit=limit,
                    threshold=0.2, fts_only=fts_only,
                )
                if isinstance(results, list) and results and isinstance(results[0], dict) and "error" in results[0]:
                    self._json_response(500, results[0])
                    return
                self._json_response(200, {"results": results, "fts_only": fts_only})
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        _VAULT_MIME = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
            ".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8",
        }

        def _handle_vault_file(self):
            """Serve files from the Obsidian vault (images, attachments)."""
            try:
                from brain_mcp.config import load_config
                config = load_config()
                if not config.vault_path:
                    self._json_response(400, {"error": "No vault configured"})
                    return

                import urllib.parse
                rel = urllib.parse.unquote(self.path[len("/api/vault/"):])

                # Path-traversal guard (CWE-22): resolve() follows symlinks, so a
                # link escaping the vault lands outside vault_root and is rejected.
                # A string-prefix check would wrongly admit sibling dirs like
                # "<vault> Backup" — use real path containment instead.
                vault_root = config.vault_path.resolve()
                try:
                    resolved = (config.vault_path / rel).resolve()
                    inside = resolved.is_relative_to(vault_root)
                except (OSError, ValueError) as exc:
                    print(f"vault file guard: cannot resolve {rel!r}: {exc}", file=sys.stderr)
                    inside = False
                if not inside:
                    self.send_response(403)
                    self.end_headers()
                    return

                if not resolved.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return

                suffix = resolved.suffix.lower()
                mime = self._VAULT_MIME.get(suffix, "application/octet-stream")
                data = resolved.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        def _json_response(self, code, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass  # Suppress HTTP request logs

    return WebHandler


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _start_local_server(directory: Path, port: int = 0) -> tuple[HTTPServer, int]:
    handler = _make_web_handler(directory)
    server = _ThreadingHTTPServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


_MAX_ACTIVITY_BODY = 65536


def _make_activity_handler(widget_ref, token: str):
    """Activity feed handler. The fixed port (9500) is reachable by any local
    process, so writes require the per-session *token* (Authorization: Bearer)
    that the dashboard publishes to <data_dir>/activity_token for the hook.
    Browser-originated requests always carry an Origin header and are rejected
    outright — the legitimate writer (urllib in activity_feed.py) never sends one.
    """

    class ActivityHandler(BaseHTTPRequestHandler):
        _warned_unauthorized = False

        def do_POST(self):
            if self.headers.get("Origin") is not None:
                self.send_response(403)
                self.end_headers()
                return
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                # Warn once per session, not per rejected request (the hook
                # fires on every tool use until it adopts the token file).
                if not ActivityHandler._warned_unauthorized:
                    ActivityHandler._warned_unauthorized = True
                    print(
                        "activity feed: rejected POST without valid session token "
                        "(writers must send 'Authorization: Bearer <data_dir>/activity_token')",
                        file=sys.stderr,
                    )
                self.send_response(401)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_ACTIVITY_BODY:
                self.send_response(413)
                self.end_headers()
                return
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            w = widget_ref()
            if w:
                w.push_activity(data)

        def log_message(self, fmt, *args):
            pass

    return ActivityHandler


class BrainBridge(QObject):
    activity_received = pyqtSignal(str, str, str)

    def __init__(self, nodes, on_node_clicked=None):
        super().__init__()
        self._nodes = nodes
        self.on_node_clicked = on_node_clicked

    @pyqtSlot(int, str)
    def nodeClicked(self, node_id: int, title: str):
        if self.on_node_clicked and 0 <= node_id < len(self._nodes):
            self.on_node_clicked(self._nodes[node_id])


class BrainWebWidget(QWebEngineView):
    _activity_signal = pyqtSignal(str)

    def __init__(self, nodes, edges, positions, parent=None):
        super().__init__(parent)
        self._nodes = nodes
        self._edges = edges
        self._positions = np.asarray(positions, dtype=np.float32)

        self._bridge = BrainBridge(nodes)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)

        self._activity_signal.connect(self._run_js_on_main_thread)

        self._graph_json = self._build_graph_json()

        self._server, self._port = _start_local_server(_WEB_DIR)
        self._start_activity_server()
        self._setup_page()

    def _start_activity_server(self):
        import secrets
        import weakref
        # Per-session write token: published to <data_dir>/activity_token so the
        # local hook (brain_mcp/hooks/activity_feed.py) can read it; any process
        # that cannot read that file cannot inject lines into the terminal.
        token = secrets.token_hex(16)
        try:
            from brain_mcp.config import load_config
            token_path = load_config().data_dir / "activity_token"
            token_path.write_text(token, encoding="utf-8")
        except Exception as exc:
            print(f"activity feed: could not publish session token: {exc}", file=sys.stderr)
        handler_cls = _make_activity_handler(weakref.ref(self), token)
        try:
            self._activity_server = HTTPServer(("127.0.0.1", ACTIVITY_PORT), handler_cls)
            thread = threading.Thread(target=self._activity_server.serve_forever, daemon=True)
            thread.start()
            print(f"Activity server on port {ACTIVITY_PORT}")
        except OSError:
            print(f"Activity port {ACTIVITY_PORT} in use, skipping")

    @pyqtSlot(str)
    def _run_js_on_main_thread(self, js: str):
        self.page().runJavaScript(js)

    @staticmethod
    def _sanitize_html(text: str) -> str:
        import html as _html
        import re
        safe = _html.escape(text, quote=True)
        def _restore_hl(m: re.Match) -> str:
            inner = _html.escape(m.group(1))
            return f"<span class='hl'>{inner}</span>"
        safe = re.sub(
            r"&lt;span class=(?:&#x27;|&#039;|')hl(?:&#x27;|&#039;|')&gt;(.*?)&lt;/span&gt;",
            _restore_hl,
            safe,
        )
        return safe

    def push_activity(self, data: dict):
        tag = data.get("tag", "SYS")[:20]
        text = self._sanitize_html(data.get("text", "")[:500])
        tag_class = data.get("tagClass", "tag-tool")
        if tag_class not in ("tag-tool", "tag-mem", "tag-you", "tag-out"):
            tag_class = "tag-tool"
        js = f"if(window.addActivityLine) window.addActivityLine({json.dumps(tag)}, {json.dumps(text)}, {json.dumps(tag_class)});"
        self._activity_signal.emit(js)

    @property
    def on_node_clicked(self):
        return self._bridge.on_node_clicked

    @on_node_clicked.setter
    def on_node_clicked(self, callback):
        self._bridge.on_node_clicked = callback

    def _build_graph_json(self) -> str:
        nodes_data = []
        for i, n in enumerate(self._nodes):
            if "region_idx" in n:
                region_idx = n["region_idx"]
            else:
                region_idx = COMMUNITY_TO_REGION.get(n.get("community", 0), 9)
            source_file = n.get("source_file", n.get("id", f"Node {i}"))
            title = n.get("title", Path(source_file).stem if source_file else f"Node {i}")
            pos = self._positions[i].tolist()
            nodes_data.append({
                "title": title,
                "source_file": source_file,
                "regionIdx": int(region_idx),
                "pos": pos,
                "tags": [f"#brain/{REGIONS[region_idx]['name'].lower().replace(' ', '-')}"],
                "wordCount": n.get("word_count", 500),
                "created": n.get("created", "2026-01-01"),
            })

        edges_data = [[int(s), int(t)] for s, t in self._edges]
        return json.dumps({"nodes": nodes_data, "edges": edges_data})

    def _setup_page(self):
        page = self.page()
        page.setWebChannel(self._channel)

        script_src = f"""
        window.__graphData = {self._graph_json};
        window.__onNodeClick = function(nodeId, title) {{
            if (window.bridge) {{
                window.bridge.nodeClicked(nodeId, title);
            }}
        }};
        """

        script = QWebEngineScript()
        script.setName("graph-data-injection")
        script.setSourceCode(script_src)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        page.scripts().insert(script)

        webchannel_script = QWebEngineScript()
        webchannel_script.setName("qwebchannel")
        webchannel_script.setSourceCode("""
        var script = document.createElement('script');
        script.src = 'qrc:///qtwebchannel/qwebchannel.js';
        script.onload = function() {
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.bridge = channel.objects.bridge;
            });
        };
        document.head.appendChild(script);
        """)
        webchannel_script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
        webchannel_script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        page.scripts().insert(webchannel_script)

        self.setUrl(QUrl(f"http://127.0.0.1:{self._port}/index.html?v={int(time.time())}"))
