import sys
import json
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from data.loader import load_graph
from data.brain_layout import assign_initial_positions
from data.regions import REGIONS, COMMUNITY_TO_REGION
from brain.physics import PhysicsSimulation
from brain.camera import OrbitCamera
from brain.scene import Scene
from brain.window import BrainWindow
from brain.gl_widget import BrainGLWidget
from integrations.obsidian import open_node_in_obsidian


def _ensure_default_config() -> None:
    """Create ~/.neural-brain/config.json with defaults if it does not exist."""
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

    # Load config
    config_path = Path.home() / ".neural-brain" / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    graph_path = config.get("graph_path", "graphify-out/graph.json")
    api_key = config.get("obsidian_api_key", "")
    vault_path = config.get("vault_path", "")

    # Load graph
    nodes, edges = load_graph(graph_path)
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges")

    # Assign initial positions
    positions = assign_initial_positions(nodes)

    # Build region data for physics
    region_centers = np.array([r["position"] for r in REGIONS], dtype=np.float32)
    node_regions = np.array(
        [COMMUNITY_TO_REGION.get(n.get("community", 0), 9) for n in nodes],
        dtype=np.int32,
    )

    # Run physics pre-simulation (200 ticks to converge)
    physics = PhysicsSimulation(positions, edges, region_centers, node_regions)
    for _ in range(200):
        physics.tick()
    positions = physics.get_positions_f32()
    print("Physics converged")

    # Create Qt application
    app = QApplication(sys.argv)

    camera = OrbitCamera()
    scene = Scene(nodes, edges, positions)
    gl_widget = BrainGLWidget(scene, camera, physics, positions, nodes)

    # Wire click -> Obsidian
    if api_key:
        gl_widget.on_node_clicked = lambda node: open_node_in_obsidian(node, api_key, vault_path)

    window = BrainWindow(gl_widget, camera)
    window.setWindowTitle("Neural Brain Dashboard")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
