import sys
import json
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow

from data.loader import load_graph
from data.brain_layout import assign_initial_positions
from data.regions import REGIONS, COMMUNITY_TO_REGION
from brain.physics import PhysicsSimulation
from brain.web_widget import BrainWebWidget
from integrations.obsidian import open_node_in_obsidian


def _ensure_default_config() -> None:
    config_dir = Path.home() / ".neural-brain"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    if not config_path.exists():
        default = {
            "graph_path": "graphify-out/graph.json",
            "obsidian_api_key": "",
            "vault_path": "",
        }
        config_path.write_text(json.dumps(default, indent=4), encoding="utf-8")


def main():
    _ensure_default_config()

    config_path = Path.home() / ".neural-brain" / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    graph_path = config.get("graph_path", "graphify-out/graph.json")
    api_key = config.get("obsidian_api_key", "")
    vault_path = config.get("vault_path", "")

    nodes, edges = load_graph(graph_path)
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges")

    positions = assign_initial_positions(nodes)

    region_centers = np.array([r["position"] for r in REGIONS], dtype=np.float32)
    node_regions = np.array(
        [COMMUNITY_TO_REGION.get(n.get("community", 0), 9) for n in nodes],
        dtype=np.int32,
    )

    physics = PhysicsSimulation(positions, edges, region_centers, node_regions)
    for _ in range(200):
        physics.tick()
    positions = physics.get_positions_f32()
    print("Physics converged")

    app = QApplication(sys.argv)

    web_widget = BrainWebWidget(nodes, edges, positions)

    if api_key:
        web_widget.on_node_clicked = lambda node: open_node_in_obsidian(node, api_key, vault_path)

    window = QMainWindow()
    window.setWindowTitle("Neural Brain Dashboard")
    window.setStyleSheet("background-color: #05070B;")
    window.setCentralWidget(web_widget)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
