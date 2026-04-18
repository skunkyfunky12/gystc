"""WebEngine widget that hosts the Three.js brain renderer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QUrl, pyqtSlot, QObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage

from data.regions import COMMUNITY_TO_REGION, REGIONS


class BrainBridge(QObject):
    """Python object exposed to JavaScript via QWebChannel."""

    def __init__(self, nodes, on_node_clicked=None):
        super().__init__()
        self._nodes = nodes
        self.on_node_clicked = on_node_clicked

    @pyqtSlot(int, str)
    def nodeClicked(self, node_id: int, title: str):
        if self.on_node_clicked and 0 <= node_id < len(self._nodes):
            self.on_node_clicked(self._nodes[node_id])


class BrainWebWidget(QWebEngineView):
    """QWebEngineView that renders the Three.js brain prototype.

    Parameters
    ----------
    nodes:     List of node dicts from graph.json.
    edges:     List of (src, tgt) index pairs.
    positions: (N, 3) float32 array of pre-computed positions.
    parent:    Optional Qt parent widget.
    """

    def __init__(self, nodes, edges, positions, parent=None):
        super().__init__(parent)
        self._nodes = nodes
        self._edges = edges
        self._positions = np.asarray(positions, dtype=np.float32)

        self._bridge = BrainBridge(nodes)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)

        self._graph_json = self._build_graph_json()
        self._setup_page()

    @property
    def on_node_clicked(self):
        return self._bridge.on_node_clicked

    @on_node_clicked.setter
    def on_node_clicked(self, callback):
        self._bridge.on_node_clicked = callback

    def _build_graph_json(self) -> str:
        nodes_data = []
        for i, n in enumerate(self._nodes):
            region_idx = COMMUNITY_TO_REGION.get(n.get("community", 0), 9)
            source_file = n.get("source_file", n.get("id", f"Node {i}"))
            title = Path(source_file).stem if source_file else f"Node {i}"
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

        html_path = Path(__file__).parent / "web" / "index.html"

        # Inject data before page loads via a user script
        script_src = f"""
        window.__graphData = {self._graph_json};
        window.__onNodeClick = function(nodeId, title) {{
            if (window.bridge) {{
                window.bridge.nodeClicked(nodeId, title);
            }}
        }};
        """

        from PyQt6.QtWebEngineCore import QWebEngineScript
        script = QWebEngineScript()
        script.setName("graph-data-injection")
        script.setSourceCode(script_src)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        page.scripts().insert(script)

        # Also inject QWebChannel JS
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

        self.setUrl(QUrl.fromLocalFile(str(html_path.resolve())))
