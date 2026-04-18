"""Scene builder: compiles shaders, manages VAOs/VBOs, drives all GPU rendering."""

import ctypes
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COMPILE_STATUS,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_LINES,
    GL_LINK_STATUS,
    GL_POINTS,
    GL_STATIC_DRAW,
    GL_TRIANGLE_FAN,
    GL_TRUE,
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
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBlendFunc,
    glDepthMask,
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
)

from data.regions import COMMUNITY_TO_REGION, REGIONS

_SHADER_DIR = Path(__file__).parent / "shaders"


def load_shader(vert_file: str, frag_file: str) -> int:
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


class Scene:

    def __init__(self, nodes: list, edges: list, positions: np.ndarray) -> None:
        self._nodes = nodes
        self._edges = edges
        self._positions = np.asarray(positions, dtype=np.float32)

        self._prog_node: int = 0
        self._prog_edge: int = 0
        self._prog_star: int = 0

        self._node_vao: int = 0
        self._node_vbo_positions: int = 0
        self._node_vbo_colors: int = 0
        self._node_vbo_sizes: int = 0
        self._node_vbo_quad: int = 0
        self._node_count: int = len(nodes)

        self._edge_vao: int = 0
        self._edge_vbo: int = 0
        self._edge_vertex_count: int = len(edges) * 2

        self._star_vao: int = 0
        self._star_vbo: int = 0
        self._star_count: int = 4000

        # Cached uniform locations (populated in init_gl)
        self._uloc: dict[str, int] = {}

        # Pre-compute static edge color data for vectorized rebuild
        self._edge_src = np.array([e[0] for e in edges], dtype=np.int32)
        self._edge_tgt = np.array([e[1] for e in edges], dtype=np.int32)
        self._edge_colors = self._compute_edge_colors()

    def _compute_edge_colors(self) -> np.ndarray:
        """Pre-compute RGBA color for each edge (static, doesn't change with positions)."""
        n_edges = len(self._edges)
        colors = np.ones((n_edges, 4), dtype=np.float32)
        for i, (src, tgt) in enumerate(self._edges):
            src_region = COMMUNITY_TO_REGION.get(self._nodes[src].get("community", 0), 9)
            tgt_region = COMMUNITY_TO_REGION.get(self._nodes[tgt].get("community", 0), 9)
            if src_region == tgt_region:
                rc = REGIONS[src_region]["color"]
                colors[i] = [0.5 + 0.5 * rc[0], 0.5 + 0.5 * rc[1], 0.5 + 0.5 * rc[2], 0.35]
            else:
                colors[i] = [0.7, 0.7, 0.8, 0.1]
        return colors

    def init_gl(self) -> None:
        self._prog_node = load_shader("node.vert", "node.frag")
        self._prog_edge = load_shader("edge.vert", "edge.frag")
        self._prog_star = load_shader("star.vert", "star.frag")

        # Cache all uniform locations once
        for prog, name in [
            (self._prog_star, "star_view"), (self._prog_star, "star_proj"),
            (self._prog_star, "star_time"),
            (self._prog_edge, "edge_view"), (self._prog_edge, "edge_proj"),
            (self._prog_node, "node_view"), (self._prog_node, "node_proj"),
            (self._prog_node, "node_time"),
        ]:
            uniform_name = name.split("_", 1)[1]
            self._uloc[name] = glGetUniformLocation(prog, f"u_{uniform_name}")

        self._build_node_buffers()
        self._build_edge_buffers()
        self._build_star_buffers()

    def update_positions(self, positions: np.ndarray) -> None:
        self._positions = np.asarray(positions, dtype=np.float32)

        data = self._positions.flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_positions)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._update_edge_positions()

    def render(self, view_matrix: np.ndarray, proj_matrix: np.ndarray, time: float) -> None:
        view = np.ascontiguousarray(view_matrix, dtype=np.float32)
        proj = np.ascontiguousarray(proj_matrix, dtype=np.float32)

        # Stars — render first, no depth writes (background)
        glDepthMask(False)
        glUseProgram(self._prog_star)
        glUniformMatrix4fv(self._uloc["star_view"], 1, GL_TRUE, view)
        glUniformMatrix4fv(self._uloc["star_proj"], 1, GL_TRUE, proj)
        glUniform1f(self._uloc["star_time"], float(time))
        glBindVertexArray(self._star_vao)
        glDrawArrays(GL_POINTS, 0, self._star_count)
        glDepthMask(True)

        # Edges — alpha-blended pass
        glUseProgram(self._prog_edge)
        glUniformMatrix4fv(self._uloc["edge_view"], 1, GL_TRUE, view)
        glUniformMatrix4fv(self._uloc["edge_proj"], 1, GL_TRUE, proj)
        glBindVertexArray(self._edge_vao)
        glDrawArrays(GL_LINES, 0, self._edge_vertex_count)

        # Nodes — additive blending for bioluminescent glow
        glDepthMask(False)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glUseProgram(self._prog_node)
        glUniformMatrix4fv(self._uloc["node_view"], 1, GL_TRUE, view)
        glUniformMatrix4fv(self._uloc["node_proj"], 1, GL_TRUE, proj)
        glUniform1f(self._uloc["node_time"], float(time))
        glBindVertexArray(self._node_vao)
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, self._node_count)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(True)

        glBindVertexArray(0)
        glUseProgram(0)

    def _build_node_buffers(self) -> None:
        nodes = self._nodes

        colors = np.array(
            [REGIONS[COMMUNITY_TO_REGION.get(n.get("community", 0), 9)]["color"] for n in nodes],
            dtype=np.float32,
        )

        degree: dict[int, int] = {}
        for src, tgt in self._edges:
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1

        region_hub: dict[int, tuple[int, int]] = {}
        for node_idx, n in enumerate(nodes):
            region_idx = COMMUNITY_TO_REGION.get(n.get("community", 0), 9)
            deg = degree.get(node_idx, 0)
            if region_idx not in region_hub or deg > region_hub[region_idx][0]:
                region_hub[region_idx] = (deg, node_idx)

        hub_indices = {info[1] for info in region_hub.values()}
        sizes = np.array(
            [18.0 if i in hub_indices else 6.0 for i in range(len(nodes))],
            dtype=np.float32,
        )

        quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

        self._node_vao = glGenVertexArrays(1)
        glBindVertexArray(self._node_vao)

        vbos = glGenBuffers(4)
        self._node_vbo_positions, self._node_vbo_colors, self._node_vbo_sizes, self._node_vbo_quad = vbos

        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_positions)
        flat_pos = self._positions.flatten()
        glBufferData(GL_ARRAY_BUFFER, flat_pos.nbytes, flat_pos, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(0, 1)

        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_colors)
        flat_col = colors.flatten()
        glBufferData(GL_ARRAY_BUFFER, flat_col.nbytes, flat_col, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(1, 1)

        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_sizes)
        glBufferData(GL_ARRAY_BUFFER, sizes.nbytes, sizes, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(2, 1)

        glBindBuffer(GL_ARRAY_BUFFER, self._node_vbo_quad)
        flat_quad = quad.flatten()
        glBufferData(GL_ARRAY_BUFFER, flat_quad.nbytes, flat_quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribDivisor(3, 0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _build_edge_buffers(self) -> None:
        """Build interleaved edge vertex data and set up VAO (called once in init_gl)."""
        n_edges = len(self._edges)
        # Interleaved: [x,y,z,r,g,b,a] per vertex, 2 vertices per edge
        arr = np.empty((n_edges * 2, 7), dtype=np.float32)

        src_pos = self._positions[self._edge_src]
        tgt_pos = self._positions[self._edge_tgt]
        arr[0::2, :3] = src_pos
        arr[0::2, 3:] = self._edge_colors
        arr[1::2, :3] = tgt_pos
        arr[1::2, 3:] = self._edge_colors
        flat = arr.flatten()

        self._edge_vao = glGenVertexArrays(1)
        self._edge_vbo = glGenBuffers(1)

        glBindVertexArray(self._edge_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._edge_vbo)

        stride = 7 * 4
        glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._edge_vertex_count = n_edges * 2

    def _update_edge_positions(self) -> None:
        """Vectorized edge position update — no Python loop, no VAO re-setup."""
        n_edges = len(self._edges)
        arr = np.empty((n_edges * 2, 7), dtype=np.float32)

        src_pos = self._positions[self._edge_src]
        tgt_pos = self._positions[self._edge_tgt]
        arr[0::2, :3] = src_pos
        arr[0::2, 3:] = self._edge_colors
        arr[1::2, :3] = tgt_pos
        arr[1::2, 3:] = self._edge_colors
        flat = arr.flatten()

        glBindBuffer(GL_ARRAY_BUFFER, self._edge_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, flat.nbytes, flat)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_star_buffers(self) -> None:
        rng = np.random.default_rng(123)
        positions = rng.uniform(-2000.0, 2000.0, (self._star_count, 3)).astype(np.float32)
        brightness = rng.uniform(0.3, 1.0, self._star_count).astype(np.float32)

        interleaved = np.empty((self._star_count, 4), dtype=np.float32)
        interleaved[:, :3] = positions
        interleaved[:, 3] = brightness
        arr = interleaved.flatten()

        self._star_vao = glGenVertexArrays(1)
        self._star_vbo = glGenBuffers(1)

        glBindVertexArray(self._star_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._star_vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)

        stride = 4 * 4

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
