# tests/test_loader.py
import json
import tempfile
import os
from data.loader import load_graph

def _make_test_graph(path):
    data = {
        "nodes": [
            {"id": "n1", "label": "foo()", "community": 0, "source_file": "/a.ts", "source_location": "L1", "file_type": "code"},
            {"id": "n2", "label": "bar()", "community": 0, "source_file": "/b.ts", "source_location": "L5", "file_type": "code"},
            {"id": "n3", "label": "baz()", "community": 3, "source_file": "/c.ts", "source_location": "L10", "file_type": "code"},
        ],
        "links": [
            {"source": "n1", "target": "n2", "relation": "calls", "confidence": "EXTRACTED", "weight": 1.0},
            {"source": "n2", "target": "n3", "relation": "contains", "confidence": "INFERRED", "weight": 1.0},
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def test_load_graph_returns_nodes_and_edges():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        _make_test_graph(path)
        nodes, edges = load_graph(path)
        assert len(nodes) == 3
        assert len(edges) == 2

def test_node_has_required_fields():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        _make_test_graph(path)
        nodes, _ = load_graph(path)
        n = nodes[0]
        assert n["id"] == "n1"
        assert n["label"] == "foo()"
        assert n["community"] == 0
        assert n["source_file"] == "/a.ts"

def test_edge_indices_are_integers():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.json")
        _make_test_graph(path)
        nodes, edges = load_graph(path)
        for src, tgt in edges:
            assert isinstance(src, int)
            assert isinstance(tgt, int)
            assert 0 <= src < len(nodes)
            assert 0 <= tgt < len(nodes)
