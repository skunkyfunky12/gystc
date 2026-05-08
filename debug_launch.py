import sys
import traceback
import json
from pathlib import Path

print("Starting debug launch...")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    import numpy as np
    print("Imports OK")

    from data.loader import load_graph
    from data.brain_layout import assign_initial_positions
    from data.regions import REGIONS, COMMUNITY_TO_REGION
    from brain.physics import PhysicsSimulation
    from brain.camera import OrbitCamera
    from brain.scene import Scene
    print("Project imports OK")

    app = QApplication(sys.argv)
    print("QApplication created")

    config = json.loads(
        Path.home().joinpath(".gystc", "config.json").read_text(encoding="utf-8")
    )
    nodes, edges = load_graph(config["graph_path"])
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

    camera = OrbitCamera()
    scene = Scene(nodes, edges, positions)
    print("Scene created")

    from brain.gl_widget import BrainGLWidget
    gl_widget = BrainGLWidget(scene, camera, physics, positions, nodes)
    print("GL Widget created")

    from brain.window import BrainWindow
    window = BrainWindow(gl_widget, camera)
    print("Window created and shown!")

    QTimer.singleShot(5000, app.quit)
    ret = app.exec()
    print(f"App exited with code {ret}")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
