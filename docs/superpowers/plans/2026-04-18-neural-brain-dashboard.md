# Neural Brain Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native PyQt6 desktop app that renders the graphify knowledge graph as a 3D force-directed brain in OpenGL with dark space aesthetics.

**Architecture:** PyQt6 window hosts a QOpenGLWidget that renders nodes (billboard quads with glow shaders) and edges (lines with opacity falloff) in a force-directed layout constrained to brain-shaped region gravity wells. Data flows from graphify graph.json through a physics simulation into GPU vertex buffers.

**Tech Stack:** Python 3.11+, PyQt6, PyOpenGL, GLSL, numpy, scipy, networkx, requests

---

## File Structure

```
neural-brain/
├── main.py                     # QApplication entry point
├── requirements.txt            # Dependencies
├── .gitignore
├── brain/
│   ├── __init__.py
│   ├── window.py               # QMainWindow (maximized, dark, keyboard shortcuts)
│   ├── gl_widget.py            # QOpenGLWidget (render loop, mouse events)
│   ├── camera.py               # OrbitCamera (rotate, zoom, pan, fly-to)
│   ├── scene.py                # Builds GPU buffers from positioned graph data
│   ├── physics.py              # Force-directed simulation (scipy KD-tree)
│   ├── picking.py              # Ray-cast: screen click to node index
│   └── shaders/
│       ├── node.vert / .frag   # Billboard quad with radial glow
│       ├── edge.vert / .frag   # Simple colored line
│       └── star.vert / .frag   # Point-based starfield
├── data/
│   ├── __init__.py
│   ├── loader.py               # Parses graphify graph.json
│   ├── regions.py              # 12 brain regions + C2R mapping
│   └── brain_layout.py         # Initial 3D positions per region center
├── integrations/
│   ├── __init__.py
│   └── obsidian.py             # REST API call to open notes
└── tests/
    ├── __init__.py
    ├── test_regions.py
    ├── test_loader.py
    ├── test_physics.py
    ├── test_camera.py
    ├── test_picking.py
    └── test_obsidian.py
```

---

### Task 1: Project Scaffold + Dependencies

**Files:**
- Create: `main.py`, `requirements.txt`, `.gitignore`
- Create: `brain/__init__.py`, `data/__init__.py`, `integrations/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create project directories**

Run from the Projekte folder:
```
mkdir -p neural-brain/brain/shaders neural-brain/data neural-brain/integrations neural-brain/tests
```

- [ ] **Step 2: Write requirements.txt**

```
PyQt6>=6.6
PyOpenGL>=3.1.7
PyOpenGL-accelerate>=3.1.7
numpy>=1.24
scipy>=1.11
networkx>=3.0
requests>=2.31
pytest>=7.0
```

- [ ] **Step 3: Install dependencies**

```
pip install -r requirements.txt
```

- [ ] **Step 4: Verify imports work**

```python
# Run: python -c "from PyQt6.QtWidgets import QApplication; from OpenGL.GL import *; import numpy; import scipy; print('OK')"
```

- [ ] **Step 5: Create empty __init__.py files and minimal main.py**

main.py:
```python
import sys
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Neural Brain")
    print("Neural Brain starting...")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

All __init__.py files are empty.

- [ ] **Step 6: Init git and commit**

Create .gitignore with: `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`, `dist/`, `build/`

Then init repo and commit all files.

---

### Task 2: Region Definitions + Brain Layout

**Files:**
- Create: `data/regions.py`, `data/brain_layout.py`
- Create: `tests/test_regions.py`

- [ ] **Step 1: Write tests for regions**

```python
# tests/test_regions.py
from data.regions import REGIONS, COMMUNITY_TO_REGION

def test_twelve_regions():
    assert len(REGIONS) == 12

def test_each_region_has_required_fields():
    for r in REGIONS:
        assert "name" in r
        assert "color" in r and len(r["color"]) == 3
        assert "position" in r and len(r["position"]) == 3

def test_community_mapping_covers_all():
    for cid in range(77):
        assert cid in COMMUNITY_TO_REGION
        assert 0 <= COMMUNITY_TO_REGION[cid] < 12
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement regions.py**

12 regions with name, RGB color tuple (0-1 floats), and 3D position. Plus the full COMMUNITY_TO_REGION dict mapping communities 0-76 to region indices 0-11.

Colors (as RGB float tuples):
- Praefrontaler Cortex: (0.204, 0.596, 0.859) at (0, 60, -80)
- Motorischer Cortex: (0.906, 0.298, 0.235) at (-30, 70, -40)
- Sensorischer Cortex: (0.180, 0.800, 0.443) at (30, 70, -20)
- Hippocampus: (0.953, 0.612, 0.071) at (-50, 0, 20)
- Kleinhirn: (0.608, 0.349, 0.714) at (40, -20, 70)
- Nucleus Accumbens: (0.102, 0.737, 0.612) at (0, 10, -20)
- Broca-Areal: (0.902, 0.494, 0.133) at (-40, 30, -50)
- Visueller Cortex: (0.557, 0.267, 0.678) at (30, 20, 70)
- Thalamus: (0.086, 0.627, 0.522) at (0, 20, 0)
- Stammhirn: (0.584, 0.647, 0.651) at (0, -60, 50)
- Basalganglien: (0.827, 0.329, 0.000) at (-15, 15, -5)
- Amygdala: (0.753, 0.224, 0.169) at (-20, -10, -30)

C2R mapping: {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:7, 9:7, 10:10, 11:10, 12:8, 13:8, 14:10, 15:11, 16:5, 17:10, 18:10, 19:11, 20:7, 21:11, 22:9, 23:8, 24:10, 25:10, 26:10, 27:11, 28:7, 29:3, 30:3, 31:3, 32:9, 33:0, 34:0, 35:11, 36:11, 37:9, 38:7, 39:7, 40:9, 41:7, 42:7, 43:9, 44:3, 45:3, 46:3, 47:7, 48:10, 49:5, 50:7, 51:9, 52:9, 53:9, 54:9, 55:9, 56:9, 57:9, 58:9, 59:9, 60:9, 61:0, 62:9, 63:8, 64:9, 65:9, 66:9, 67:0, 68:8, 69:7, 70:3, 71:7, 72:7, 73:3, 74:0, 75:10, 76:8}

- [ ] **Step 4: Implement brain_layout.py**

`assign_initial_positions(nodes)` function: for each node, look up its community via C2R to get region index, take that region's center position, add gaussian scatter (std=15), return numpy float32 array of shape (N, 3).

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Commit**

---

### Task 3: Graph Data Loader

**Files:**
- Create: `data/loader.py`, `tests/test_loader.py`

- [ ] **Step 1: Write tests**

Test with a temporary graph.json containing 3 nodes and 2 links. Verify:
- `load_graph(path)` returns `(nodes_list, edges_as_index_pairs)`
- Nodes have id, label, community, source_file fields
- Edge indices are valid integers into the nodes list

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement loader.py**

`load_graph(path)`: reads JSON, builds id-to-index mapping, converts link source/target IDs to integer index pairs. Handles both `source`/`target` and `_src`/`_tgt` field names.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 4: Force-Directed Physics Engine

**Files:**
- Create: `brain/physics.py`, `tests/test_physics.py`

- [ ] **Step 1: Write tests**

Three tests:
1. Simulation converges (total_displacement < 1.0 after 200 ticks)
2. Connected nodes attract (distance decreases)
3. Region gravity pulls node toward center (distance to center decreases)

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement PhysicsSimulation class**

Constructor takes: positions (Nx3 float32), edges (list of (src, tgt) tuples), region_centers (12x3 float32), node_regions (N int32 array).

Tunable parameters with defaults: center_strength=0.02, repel_strength=5.0, link_strength=0.8, link_distance=30.0, region_gravity=0.15, damping=0.95.

`tick()` method applies four forces:
1. Center gravity: `forces -= positions * center_strength`
2. Region gravity: each node pulled toward its region center
3. Node repulsion via scipy.spatial.cKDTree (query_pairs r=80), inverse-square
4. Link attraction: spring force toward link_distance

Then: `velocities = (velocities + forces) * damping; positions += velocities`

`get_positions_f32()`: returns positions as float32 array.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 5: Orbit Camera

**Files:**
- Create: `brain/camera.py`, `tests/test_camera.py`

- [ ] **Step 1: Write tests**

Test: view matrix is 4x4, zoom changes distance, rotate changes yaw, projection matrix is 4x4.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement OrbitCamera**

State: target (vec3), distance, yaw, pitch, fov=45, near=1, far=5000.

Methods: `rotate(dx, dy)`, `zoom(delta)`, `pan(dx, dy)`, `get_eye_position()`, `get_view_matrix()` (look_at), `get_projection_matrix(w, h)` (perspective), `fly_to(target, distance)`, `reset()`.

Helper functions: `_look_at(eye, center, up)` and `_perspective(fov, aspect, near, far)` — both return 4x4 float32 matrices.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 6: GLSL Shaders

**Files:**
- Create: All 6 shader files in `brain/shaders/`

- [ ] **Step 1: node.vert** — Instanced billboard rendering. Inputs: a_position (vec3, instanced), a_color (vec3, instanced), a_size (float, instanced), a_quad (vec2, per-vertex). Uniforms: u_view, u_proj, u_time. Shader-based drift via sin(u_time + gl_InstanceID). Extract cam_right/cam_up from view matrix for billboarding.

- [ ] **Step 2: node.frag** — Radial glow. White-hot center (smoothstep), fading to region color. Alpha = exp(-dist * 2.5). Discard if dist > 1.0.

- [ ] **Step 3: edge.vert** — Simple transform. Inputs: a_position (vec3), a_color (vec4). Pass through to fragment.

- [ ] **Step 4: edge.frag** — Pass through v_color.

- [ ] **Step 5: star.vert** — Point rendering. gl_PointSize = 1.5. Pass brightness to fragment.

- [ ] **Step 6: star.frag** — White with alpha = brightness * 0.4.

- [ ] **Step 7: Commit**

---

### Task 7: Scene Builder (CPU to GPU Buffers)

**Files:**
- Create: `brain/scene.py`

- [ ] **Step 1: Implement Scene class**

`load_shader(vert_file, frag_file)`: reads GLSL source from shaders/ dir, compiles, links, returns program ID.

`Scene.__init__(nodes, edges, positions)`: stores data, no GL calls yet.

`Scene.init_gl()`: compiles all 3 shader programs, calls buffer builders.

`Scene._build_node_buffers()`: Creates VAO with 4 instanced VBOs (positions, colors, sizes, quad corners). Position buffer is GL_DYNAMIC_DRAW. Colors/sizes from region mapping. Hub nodes get size=9, regular=3.

`Scene._build_edge_buffers()`: Builds vertex array with interleaved position (vec3) + color (vec4 with alpha). Alpha based on edge distance and same/cross-region.

`Scene._build_star_buffers()`: 2000 random points in [-2000, 2000] cube with random brightness.

`Scene.update_positions(positions)`: rebinds position buffer + rebuilds edges.

- [ ] **Step 2: Commit**

---

### Task 8: OpenGL Widget + Render Loop

**Files:**
- Create: `brain/gl_widget.py`

- [ ] **Step 1: Implement BrainGLWidget(QOpenGLWidget)**

`initializeGL()`: set clear color #030508, enable blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), enable GL_PROGRAM_POINT_SIZE, call scene.init_gl(), start QTimer at 16ms (~60fps).

`paintGL()`: clear, compute view/proj from camera, render in order: stars (GL_POINTS), edges (GL_LINES), nodes (GL_TRIANGLE_STRIP instanced). Pass u_time uniform to node shader.

Mouse events: left-drag = camera.rotate, right-drag = camera.pan, scroll = camera.zoom. Left click (no drag) = pick_node + on_node_clicked callback.

`_handle_click(mx, my)`: calls picking.pick_node, if hit calls on_node_clicked(node_dict).

- [ ] **Step 2: Commit**

---

### Task 9: Ray-Casting Picker

**Files:**
- Create: `brain/picking.py`, `tests/test_picking.py`

- [ ] **Step 1: Write tests**

Test: node at origin is picked when clicking screen center. Test: clicking far corner returns -1.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement pick_node(mx, my, w, h, camera, positions, node_radius)**

Unproject screen coords to ray in world space (NDC -> inv_proj -> inv_view). Ray-sphere intersection test against each node position with given radius. Return index of closest hit, or -1.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

---

### Task 10: Obsidian Integration

**Files:**
- Create: `integrations/obsidian.py`, `tests/test_obsidian.py`

- [ ] **Step 1: Write tests**

Test: `build_open_url` URL-encodes spaces. Test: strips vault prefix from absolute paths.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement obsidian.py**

`build_open_url(note_path)`: strips vault prefix if present, URL-encodes, returns `http://127.0.0.1:27123/open/{encoded}`.

`open_in_obsidian(note_path)`: reads API key from `~/.neural-brain/config.json`, POSTs to the URL with Bearer auth. Fails silently if Obsidian not running.

`open_node_in_obsidian(node)`: extracts source_file from node dict, calls open_in_obsidian.

- [ ] **Step 4: Create config file at ~/.neural-brain/config.json** with obsidian_api_key.

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Commit**

---

### Task 11: Main Window + App Entry Point

**Files:**
- Create: `brain/window.py`
- Modify: `main.py`

- [ ] **Step 1: Implement BrainWindow(QMainWindow)**

Dark background stylesheet. Sets BrainGLWidget as central widget. Shows maximized.

keyPressEvent: ESC=close, F=toggle fullscreen, R=reset camera, 1-9/0/-/==fly to region 1-12.

- [ ] **Step 2: Implement full main.py**

Reads config from ~/.neural-brain/config.json (graph_path, obsidian_api_key).
Loads graph.json via loader.
Assigns initial positions via brain_layout.
Runs 200 physics iterations (or until converged).
Creates QApplication, Scene, BrainWindow.
Wires on_node_clicked to open_node_in_obsidian.

- [ ] **Step 3: Run the app and verify**

- [ ] **Step 4: Commit**

---

### Task 12: Smoke Test + Final Verification

- [ ] **Step 1: Run all tests**

All tests should pass: test_regions, test_loader, test_physics, test_camera, test_picking, test_obsidian.

- [ ] **Step 2: Launch and verify visually**

Checklist:
- Dark background with starfield
- Nodes visible with colored glow in brain formation
- Edges connecting nodes (white/gray)
- Orbit camera works (left-drag, scroll, right-drag)
- Number keys fly to regions
- R resets, F toggles fullscreen
- Click node opens in Obsidian
- ESC quits
- No GPU spike (check Task Manager)

- [ ] **Step 3: Final commit**
