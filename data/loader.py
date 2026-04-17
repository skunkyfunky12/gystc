# data/loader.py
import json


def load_graph(path):
    """Load graphify graph.json, return (nodes_list, edges_as_index_pairs)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    edges = []
    for link in data.get("links", []):
        src = link.get("source") or link.get("_src")
        tgt = link.get("target") or link.get("_tgt")
        if src in id_to_idx and tgt in id_to_idx:
            edges.append((id_to_idx[src], id_to_idx[tgt]))
    return nodes, edges
