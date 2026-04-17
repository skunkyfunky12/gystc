# Neural Brain Dashboard — Design Spec

**Date:** 2026-04-17
**Status:** Approved
**Stack:** Python, PyQt6, QOpenGLWidget, GLSL Shaders

## Purpose

Native desktop application that visualizes the unified knowledge graph (all projects, all knowledge) as a 3D force-directed graph arranged in a brain-shaped topology. Replaces Obsidian Graph View for visualization. Obsidian remains the editor.

## Architecture

```
neural-brain/
├── main.py                  # QApplication entry point
├── brain/
│   ├── window.py            # QMainWindow (fullscreen, dark, frameless)
│   ├── gl_widget.py         # QOpenGLWidget — 3D rendering loop
│   ├── camera.py            # Orbit camera (rotate, zoom, pan)
│   ├── scene.py             # Scene graph: nodes, edges, labels
│   ├── physics.py           # Force-directed layout (scipy KD-tree for repulsion)
│   ├── picking.py           # Ray-casting: 2D mouse coord → 3D node hit detection
│   └── shaders/
│       ├── node.vert        # Node vertex shader
│       ├── node.frag        # Node fragment shader (glow)
│       ├── edge.vert        # Edge vertex shader
│       └── edge.frag        # Edge fragment shader
├── data/
│   ├── loader.py            # graph.json parser
│   ├── brain_layout.py      # 12 region centers in 3D brain shape
│   └── regions.py           # Region definitions (name, color, position)
├── integrations/
│   └── obsidian.py          # REST API: open note on click
└── requirements.txt         # PyQt6, PyOpenGL, networkx, requests, numpy
```

### graph.json Data Format (NetworkX JSON)

**Nodes:**
- `id`: Unique identifier (hashed file path or function name)
- `label`: Display name (filename or function)
- `community`: Integer community assignment (0-76)
- `source_file`: Absolute file path
- `source_location`: Line number (e.g., "L1", "L22")
- `file_type`: "code"

**Links (edges):**
- `source` / `target`: Node IDs
- `relation`: Relationship type (e.g., "contains", "calls")
- `confidence`: "EXTRACTED" or "INFERRED"
- `weight`: Edge weight (1.0)

### Data Flow

1. `loader.py` reads `graphify-out/graph.json` from configured repo paths
2. `regions.py` maps 77 communities to 12 brain regions via C2R dict
3. `brain_layout.py` assigns 3D center positions for each region (brain-shaped ellipsoid)
4. `physics.py` runs force-directed simulation to position nodes
5. `scene.py` builds GPU vertex buffers from positioned nodes/edges
6. `gl_widget.py` renders each frame via GLSL shaders
7. On click: `obsidian.py` calls Obsidian REST API to open the note

### Data Sources

- **graphify** (`graphify-out/graph.json`): code knowledge graph (nodes, edges, communities)
- **Obsidian Vault**: notes, sessions, references (accessed via REST API on port 27123)
- Future: additional repos fed through graphify, merged into one unified graph

## 3D Layout: Force-Directed with Brain Constraint

Like Obsidian's graph view but with regional gravity wells forming a brain shape.

### Physics Forces

| Force | Description |
|-------|-------------|
| **Link attraction** | Connected nodes attract each other (spring force) |
| **Node repulsion** | All nodes repel each other (inverse-square) |
| **Center gravity** | Weak pull toward world origin |
| **Region gravity** | Each node pulled toward its brain region center |

### Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `centerStrength` | 0.02 | Pull toward world center |
| `repelStrength` | 5.0 | Node-to-node repulsion |
| `linkStrength` | 0.8 | Edge attraction |
| `linkDistance` | 30.0 | Ideal edge rest length |
| `regionGravity` | 0.15 | Pull toward region center (brain-shaping force) |
| `damping` | 0.95 | Velocity damping per tick |

### Brain Region Centers (3D Ellipsoid)

Lateral brain view. X = left/right, Y = up/down, Z = front/back.

| Region | Position (x, y, z) | Anatomical Role |
|--------|--------------------|-----------------| 
| Praefrontaler Cortex | (0, 60, -80) | Front/Top — decision layer |
| Motorischer Cortex | (-30, 70, -40) | Top-left — action execution |
| Sensorischer Cortex | (30, 70, -20) | Top-right — data intake |
| Hippocampus | (-50, 0, 20) | Left-mid — spatial navigation |
| Kleinhirn | (40, -20, 70) | Back-bottom — precision algorithms |
| Nucleus Accumbens | (0, 10, -20) | Deep center — reward system |
| Broca-Areal | (-40, 30, -50) | Left-front — language/AI |
| Visueller Cortex | (30, 20, 70) | Back — visual processing |
| Thalamus | (0, 20, 0) | Dead center — data relay |
| Stammhirn | (0, -60, 50) | Bottom-back — infrastructure |
| Basalganglien | (-15, 15, -5) | Deep — background processing |
| Amygdala | (-20, -10, -30) | Front-deep — social/auth |

## Visual Design: Dark Space + Bioluminescence

### Background
- Near-black `#030508`
- Static starfield particles (small white dots at random 3D positions, no animation cost)
- Optional subtle dark blue/purple nebula gradient in far plane

### Nodes
- Billboard quads with radial gradient (computed in fragment shader)
- Core: white/light gray center, thin ring of region color
- Glow: radial alpha falloff in shader, no post-processing needed
- Hub nodes (12 total): 3x larger, stronger glow
- Drift animation: shader-based uniform offset (not CPU position mutation) for organic feel

### Edges
- Thin lines (0.5-1px), white/gray base with subtle region color tint
- Opacity falls off with edge length
- Intra-region: higher opacity (0.15-0.3)
- Inter-region: lower opacity (0.03-0.08), gentle curve

### Labels
- Monospace font, billboard-rendered (always faces camera)
- Only visible for hub nodes (12 region names)
- Light gray at ~20% opacity
- Node name appears as tooltip on hover

### Hover Effect
- Node under cursor: glow intensifies, connected edges brighten
- Node name + metadata shown as overlay text

## Interaction

### Camera (Orbit)
| Input | Action |
|-------|--------|
| Left mouse + drag | Rotate around center |
| Scroll wheel | Zoom in/out |
| Right mouse + drag | Pan |

### Click
| Input | Action |
|-------|--------|
| Left click on node | Open note in Obsidian via REST API |

### Keyboard
| Key | Action |
|-----|--------|
| `R` | Reset camera to default position |
| `F` | Toggle fullscreen |
| `1`-`9`, `0`, `-`, `=` | Fly camera to region 1-12 |
| `ESC` | Quit |

## Obsidian Integration

Open notes via HTTP POST to Local REST API:
```
POST http://127.0.0.1:27123/open/{note_path}
Authorization: Bearer {API_KEY}
```

API key stored in `~/.neural-brain/config.json`.

## Performance Budget

| Metric | Target |
|--------|--------|
| VRAM usage | < 100MB |
| FPS | 60fps at 500 nodes |
| Startup time | < 2s (cached graph) |
| Physics sim | Converges in < 200 iterations |
| Draw calls | < 10 per frame (instanced rendering) |

## Dependencies

```
PyQt6>=6.6
PyOpenGL>=3.1.7
PyOpenGL-accelerate>=3.1.7
numpy>=1.24
scipy>=1.11
networkx>=3.0
requests>=2.31
```

## Not in Scope (v1)

- Agent Orchestra integration (v2)
- Live activity tracking (v2)
- Multi-repo graph merging UI (v2 — for now: config file lists repos)
- Settings UI (v2 — for now: edit config.json)
- Fly-through camera mode (v2)
