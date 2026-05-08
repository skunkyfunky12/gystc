import os
import sys
import json
from pathlib import Path

# Chromium SharedImageManager flicker fix — must be set before QWebEngine loads
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--in-process-gpu",
)

import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QIcon

from data.loader import load_graph
from data.vault_loader import load_vault
from data.brain_layout import assign_initial_positions
from data.regions import REGIONS, COMMUNITY_TO_REGION
from brain.physics import PhysicsSimulation
from brain.web_widget import BrainWebWidget
from integrations.obsidian import open_node_in_obsidian
from setup_wizard import needs_setup, run_setup


def _find_icon() -> Path | None:
    base = Path(__file__).parent / "assets"
    for ext in (".icns", ".ico", ".svg"):
        p = base / f"gystc-icon{ext}"
        if p.exists():
            return p
    return None


def main():
    app = QApplication(sys.argv)
    icon_path = _find_icon()
    if icon_path:
        app.setWindowIcon(QIcon(str(icon_path)))

    if needs_setup():
        run_setup(app)

    config_path = Path.home() / ".gystc" / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"ERROR: config.json invalid: {e}")
            print(f"Fix: {config_path}")
            sys.exit(1)

    graph_path = config.get("graph_path", "graphify-out/graph.json")
    api_key = config.get("obsidian_api_key", "")
    vault_path = config.get("vault_path", "")

    if vault_path:
        vault_dir = Path(vault_path)
        if not vault_dir.is_dir():
            print(f"ERROR: vault_path '{vault_path}' existiert nicht.")
            print(f"Fix: ~/.gystc/config.json → vault_path korrigieren oder leeren.")
            sys.exit(1)
        nodes, edges = load_vault(vault_path)
    else:
        nodes, edges = load_graph(graph_path)
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges")

    positions = assign_initial_positions(nodes)

    region_centers = np.array([r["position"] for r in REGIONS], dtype=np.float32)
    node_regions = np.array(
        [n.get("region_idx", COMMUNITY_TO_REGION.get(n.get("community", 0), 9)) for n in nodes],
        dtype=np.int32,
    )

    physics = PhysicsSimulation(positions, edges, region_centers, node_regions)
    for _ in range(200):
        physics.tick()
    positions = physics.get_positions_f32()
    print("Physics converged")

    web_widget = BrainWebWidget(nodes, edges, positions)

    if api_key:
        web_widget.on_node_clicked = lambda node: open_node_in_obsidian(node, api_key, vault_path)

    window = QMainWindow()
    window.setWindowTitle("GYSTC Dashboard")
    window.setStyleSheet("background-color: #05070B;")
    window.setCentralWidget(web_widget)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
