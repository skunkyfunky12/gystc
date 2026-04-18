"""WebEngine widget that hosts the Three.js brain renderer."""
from __future__ import annotations

import json
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QUrl, pyqtSlot, pyqtSignal, QObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineScript

from data.regions import COMMUNITY_TO_REGION, REGIONS

_WEB_DIR = Path(__file__).parent / "web"
ACTIVITY_PORT = 9500


def _start_local_server(directory: Path, port: int = 0) -> tuple[HTTPServer, int]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = HTTPServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


_MAX_ACTIVITY_BODY = 65536


def _make_activity_handler(widget_ref):
    class ActivityHandler(BaseHTTPRequestHandler):
        def do_POST(self):
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
        import weakref
        handler_cls = _make_activity_handler(weakref.ref(self))
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
        import re
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        safe = re.sub(r"&lt;span class=&#039;hl&#039;&gt;", "<span class='hl'>", safe)
        safe = re.sub(r"&lt;span class='hl'&gt;", "<span class='hl'>", safe)
        safe = re.sub(r"&lt;/span&gt;", "</span>", safe)
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

        self.setUrl(QUrl(f"http://127.0.0.1:{self._port}/index.html"))
