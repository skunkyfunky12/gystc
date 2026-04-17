"""Scene builder: compiles shaders, manages VAOs/VBOs, drives all GPU rendering.

Call order:
    scene = Scene(nodes, edges, positions)   # CPU only — no GL calls
    scene.init_gl()                          # after GL context is created
    scene.render(view, proj, time)           # every frame
    scene.update_positions(new_positions)    # after each physics tick
"""

import ctypes
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_LINES,
    GL_LINK_STATUS,
    GL_POINTS,
    GL_STATIC_DRAW,
    GL_TRIANGLE_FAN,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glBufferSubData,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteShader,
    glDrawArrays,
    glDrawArraysInstanced,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribDivisor,
    glVertexAttribPointer,
    GL_COMPILE_STATUS,
)

from data.regions import COMMUNITY_TO_REGION, REGIONS

_SHADER_DIR = Path(__file__).parent / "shaders"


# ---------------------------------------------------------------------------
# Shader compilation helper
# ---------------------------------------------------------------------------

def load_shader(vert_file: str, frag_file: str) -> int:
    """Compile a vertex + fragment shader pair and link into a program.

    Parameters
    ----------
    vert_file:
        Filename (not full path) inside ``brain/shaders/``.
    frag_file:
        Filename (not full path) inside ``brain/shaders/``.

    Returns
    -------
    int
        OpenGL program ID.

    Raises
    ------
    RuntimeError
        If compilation or linking fails, with the driver info log attached.
    """
    vert_src = (_SHADER_DIR / vert_file).read_text(encoding="utf-8")
    frag_src = (_SHADER_DIR / frag_file).read_text(encoding="utf-8")

    def _compile(src: str, shader_type: int) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            raise RuntimeError(f"Shader compile error ({shader_type}):\n{log}")
        return shader

    vert = _compile(vert_src, GL_VERTEX_SHADER)
    frag = _compile(frag_src, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vert)
    glAttachShader(program, frag)
    glLinkProgram(program)

    glDeleteShader(vert)
    glDeleteShader(frag)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"Shader link error:\n{log}")

    return program


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    """Manages all GPU resources required to render the brain visualization.

    Parameters
    ----------
    nodes:
        List of node dicts as returned by ``data.loader.load_graph``.
        Each dict must contain at least a ``"community"`` key (int 0-76).
    edges:
        List of ``(src_idx, tgt_idx)`` integer tuples.
    positions:
        ``(N, 3)`` float32 numpy array of initial node positions.
    """

    def __init__(self, nodes: list, edges: list, positions: np.ndarray) -> None:
        self._nodes = nodes
        self._edges = edges
        self._positions = np.asarray(positions, dtype=np.float32)

        # Populated by init_gl()
        self._prog_node: int = 0
        self._prog_edge: int = 0
        self._prog_star: int = 0

        self._node_vao: int = 0
        self._node_vbo_positions: int = 0   # VBO 1 — updated every tick
        self._node_vbo_colors: int = 0      # VBO 2
        self._node_vbo_sizes: int = 0       # VBO 3
        self._node_vbo_quad: int = 0        # VBO 4
        self._node_count: int = len(nodes)

        self._edge_vao: int = 0
        self._edge_vbo: int = 0
        self._edge_vertex_count: int = len(edges) * 2

        self._star_vao: int = 0
        self._star_vbo: int = 0
        self._star_count: int = 2000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_gl(self) -> None:
        """Compile shaders and upload all static GPU data.

        Must be called **after** the OpenGL context exists.
        """
        self._prog_node = load_shader("node.vert", "node.frag")
        self._prog_edge = load_shader("edge.vert", "edge.frag")
        self._prog_star = load_shader("star.vert", "star.frag")

        self._build_node_buffers()
        self._build_edge_buffers()
        self._build_star_buffers()

    def update_positions(self, positions: np.ndarray) -> None:
        """Push new node positions to the GPU (called after each physics tick).

        Parameters
        ----------
        positions:
            ``(N, 3)`` float32 numpy array with updated world positions.
        """
        self._positions = np.asarray(positions, dtype=np.float32)

        # Update the dynamic position VBO in-place — no reallocation.
        data = self._positions.flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_positions)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Edges reference positions directly, so rebuild them.
        self._build_edge_buffers()

    def render(self, view_matrix: np.ndarray, proj_matrix: np.ndarray, time: float) -> None:
        """Draw stars, edges, then nodes.

        Parameters
        ----------
        view_matrix:
            Column-major 4×4 float32 view matrix.
        proj_matrix:
            Column-major 4×4 float32 projection matrix.
        time:
            Elapsed seconds (passed to node shader for drift animation).
        """
        view = np.asarray(view_matrix, dtype=np.float32)
        proj = np.asarray(proj_matrix, dtype=np.float32)

        # -- Stars ----------------------------------------------------------
        glUseProgram(self._prog_star)
        glUniformMatrix4fv(
            glGetUniformLocation(self._prog_star, "u_view"), 1, GL_TRUE, view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self._prog_star, "u_proj"), 1, GL_TRUE, proj
        )
        glBindVertexArray(self._star_vao)
        glDrawArrays(GL_POINTS, 0, self._star_count)

        # -- Edges ----------------------------------------------------------
        glUseProgram(self._prog_edge)
        glUniformMatrix4fv(
            glGetUniformLocation(self._prog_edge, "u_view"), 1, GL_TRUE, view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self._prog_edge, "u_proj"), 1, GL_TRUE, proj
        )
        glBindVertexArray(self._edge_vao)
        glDrawArrays(GL_LINES, 0, self._edge_vertex_count)

        # -- Nodes (instanced billboards) -----------------------------------
        glUseProgram(self._prog_node)
        glUniformMatrix4fv(
            glGetUniformLocation(self._prog_node, "u_view"), 1, GL_TRUE, view
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self._prog_node, "u_proj"), 1, GL_TRUE, proj
        )
        glUniform1f(glGetUniformLocation(self._prog_node, "u_time"), float(time))
        glBindVertexArray(self._node_vao)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, self._node_count)

        # Clean up state
        glBindVertexArray(0)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # Private buffer builders
    # ------------------------------------------------------------------

    def _build_node_buffers(self) -> None:
        """Create VAO and four VBOs for instanced node rendering."""
        nodes = self._nodes

        # -- Colours (per instance, static) --------------------------------
        colors = np.array(
            [REGIONS[COMMUNITY_TO_REGION[n.get("community", 0)]]["color"] for n in nodes],
            dtype=np.float32,
        )  # shape (N, 3)

        # -- Sizes (per instance, static) ----------------------------------
        # Determine hub node per region: the node with the highest degree.
        degree: dict[int, int] = {}
        for src, tgt in self._edges:
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1

        # For each region, find the node index with the highest degree.
        region_hub: dict[int, tuple[int, int]] = {}  # region_idx -> (best_degree, node_idx)
        for node_idx, n in enumerate(nodes):
            region_idx = COMMUNITY_TO_REGION[n.get("community", 0)]
            deg = degree.get(node_idx, 0)
            if region_idx not in region_hub or deg > region_hub[region_idx][0]:
                region_hub[region_idx] = (deg, node_idx)

        hub_indices = {info[1] for info in region_hub.values()}
        sizes = np.array(
            [9.0 if i in hub_indices else 3.0 for i in range(len(nodes))],
            dtype=np.float32,
        )  # shape (N,)

        # -- Quad corners (per vertex, static) ------------------------------
        quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

        # -- Build VAO ------------------------------------------------------
        self._node_vao = glGenVertexArrays(1)
        glBindVertexArray(self._node_vao)

        vbos = glGenBuffers(4)
        self._node_vbo_positions, self._node_vbo_colors, self._node_vbo_sizes, self._node_vbo_quad = vbos

        # VBO 1 — positions (loc=0, instanced, DYNAMIC)
        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_positions)
        flat_pos = self._positions.flatten()
        glBufferData(GL_ARRAY_BUFFER, flat_pos.nbytes, flat_pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(0, 1)

        # VBO 2 — colors (loc=1, instanced, STATIC)
        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_colors)
        flat_col = colors.flatten()
        glBufferData(GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(1, 1)

        # VBO 3 — sizes (loc=2, instanced, STATIC)
        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_sizes)
        glBufferData(GL_ARRAY_BUFFER, sizes.nbytes, sizes, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(2, 1)

        # VBO 4 — quad corners (loc=3, per-vertex, STATIC)
        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_quad)
        flat_quad = quad.flatten()
        glBufferData(GL_ARRAY_BUFFER, flat_quad.nbytes, flat_quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(3, 0)  # per-vertex, not instanced

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _build_edge_buffers(self) -> None:
        """Build (or rebuild) interleaved [x,y,z,r,g,b,a] edge vertex data."""
        positions = self._positions
        nodes = self._nodes

        vertex_data: list[float] = []

        for src, tgt in self._edges:
            src_region = COMMUNITY_TO_REGION[nodes[src]["community"]]
            tgt_region = COMMUNITY_TO_REGION[nodes[tgt]["community"]]
            same_region = src_region == tgt_region

            if same_region:
                alpha = 0.2
                region_color = REGIONS[src_region]["color"]
                # White with slight region tint: lerp 70 % white + 30 % region color
                r = 0.7 + 0.3 * region_color[0]
                g = 0.7 + 0.3 * region_color[1]
                b = 0.7 + 0.3 * region_color[2]
            else:
                alpha = 0.05
                r, g, b = 1.0, 1.0, 1.0

            sx, sy, sz = positions[src]
            tx, ty, tz = positions[tgt]

            vertex_data.extend([sx, sy, sz, r, g, b, alpha])
            vertex_data.extend([tx, ty, tz, r, g, b, alpha])

        arr = np.array(vertex_data, dtype=np.float32)

        if self._edge_vao == 0:
            # First call — allocate
            self._edge_vao = glGenVertexArrays(1)
            self._edge_vbo = glGenBuffers(1)

        glBindVertexArray(self._edge_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._edge_vbo)

        stride = 7 * 4  # 7 floats × 4 bytes
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)

        # a_position (loc=0): vec3 at byte offset 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)

        # a_color (loc=1): vec4 at byte offset 12
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._edge_vertex_count = len(self._edges) * 2

    def _build_star_buffers(self) -> None:
        """Generate 2000 random background stars and upload to GPU."""
        rng = np.random.default_rng(123)
        positions = rng.uniform(-2000.0, 2000.0, (self._star_count, 3)).astype(np.float32)
        brightness = rng.uniform(0.3, 1.0, self._star_count).astype(np.float32)

        # Interleaved [x, y, z, brightness] — 4 floats per star
        interleaved = np.empty((self._star_count, 4), dtype=np.float32)
        interleaved[:, :3] = positions
        interleaved[:, 3] = brightness
        arr = interleaved.flatten()

        self._star_vao = glGenVertexArrays(1)
        self._star_vbo = glGenBuffers(1)

        glBindVertexArray(self._star_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._star_vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats × 4 bytes

        # a_position (loc=0): vec3 at offset 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)

        # a_brightness (loc=1): float at offset 12
        import ctypes
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
