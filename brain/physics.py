"""Force-directed physics simulation for the Neural Brain Dashboard.

Positions N graph nodes in 3D space using four forces per tick:
  1. Global center gravity  — pulls all nodes toward the origin
  2. Region gravity         — pulls each node toward its brain-region center
  3. Node repulsion         — inverse-square repulsion between nearby nodes
  4. Link attraction        — spring force for edges, rest length = link_distance

All heavy math is vectorised with NumPy; scipy.spatial.cKDTree handles
efficient nearest-neighbour pair queries for the repulsion pass.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class PhysicsSimulation:
    """Force-directed physics simulation with brain-region gravity wells.

    Parameters
    ----------
    positions : ndarray, shape (N, 3), float32
        Initial 3-D positions for each node.  Copied internally.
    edges : list of (int, int)
        Source / target index pairs for graph edges.
    region_centers : ndarray, shape (12, 3), float32
        3-D centre of each of the 12 brain regions.
    node_regions : ndarray, shape (N,), int
        Brain-region index (0–11) for every node.
    center_strength : float
        How strongly the origin pulls all nodes (default 0.02).
    repel_strength : float
        Magnitude of the inverse-square repulsion force (default 5.0).
    link_strength : float
        Spring constant for edge attraction (default 0.8).
    link_distance : float
        Rest length of each edge spring in world units (default 30.0).
    region_gravity : float
        Strength of the pull toward each node's region centre (default 0.15).
    damping : float
        Velocity damping factor applied each tick (default 0.95).
    """

    def __init__(
        self,
        positions: np.ndarray,
        edges: list[tuple[int, int]],
        region_centers: np.ndarray,
        node_regions: np.ndarray,
        *,
        center_strength: float = 0.002,
        repel_strength: float = 12.0,
        link_strength: float = 0.3,
        link_distance: float = 80.0,
        region_gravity: float = 0.35,
        damping: float = 0.95,
    ) -> None:
        self._positions = np.array(positions, dtype=np.float64)   # work in f64 for stability
        self._velocities = np.zeros_like(self._positions)

        # Store edges as two parallel index arrays for vectorised ops
        if edges:
            edge_arr = np.array(edges, dtype=np.int32)
            self._src = edge_arr[:, 0]
            self._tgt = edge_arr[:, 1]
        else:
            self._src = np.empty(0, dtype=np.int32)
            self._tgt = np.empty(0, dtype=np.int32)

        self._region_centers = np.array(region_centers, dtype=np.float64)
        self._node_regions = np.array(node_regions, dtype=np.int32)

        # Tunable parameters
        self.center_strength = float(center_strength)
        self.repel_strength = float(repel_strength)
        self.link_strength = float(link_strength)
        self.link_distance = float(link_distance)
        self.region_gravity = float(region_gravity)
        self.damping = float(damping)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the simulation by one time step."""
        pos = self._positions
        forces = np.zeros_like(pos)

        # 1. Global center gravity: pull every node toward origin
        forces -= pos * self.center_strength

        # 2. Region gravity: pull each node toward its region centre
        region_targets = self._region_centers[self._node_regions]   # (N, 3)
        forces += (region_targets - pos) * self.region_gravity

        # 3. Node repulsion via cKDTree (inverse-square, radius = 80)
        if self.repel_strength != 0.0:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=160.0, output_type="ndarray")  # (M, 2) int
            if len(pairs):
                i_idx = pairs[:, 0]
                j_idx = pairs[:, 1]
                delta = pos[i_idx] - pos[j_idx]                       # (M, 3)
                dist2 = np.einsum("ij,ij->i", delta, delta)           # (M,)
                dist2 = np.maximum(dist2, 1e-4)                       # avoid div/0
                scale = self.repel_strength / dist2                   # (M,)
                unit = delta / np.sqrt(dist2)[:, np.newaxis]          # (M, 3)
                force_vec = scale[:, np.newaxis] * unit               # (M, 3)
                # Accumulate: i pushed away from j, j pushed away from i
                np.add.at(forces, i_idx, force_vec)
                np.add.at(forces, j_idx, -force_vec)

        # 4. Link attraction: spring force for each edge
        if self.link_strength != 0.0 and len(self._src):
            delta = pos[self._tgt] - pos[self._src]                   # (E, 3)
            dist = np.linalg.norm(delta, axis=1, keepdims=True)       # (E, 1)
            dist = np.maximum(dist, 1e-6)
            # Spring: pull / push to rest length
            displacement = (dist - self.link_distance) / dist         # (E, 1)
            spring = self.link_strength * displacement * delta        # (E, 3)
            np.add.at(forces, self._src,  spring)
            np.add.at(forces, self._tgt, -spring)

        # Integrate: scale forces by dt for stable Euler integration,
        # then apply damping and update positions
        dt = 0.05
        self._velocities = (self._velocities + forces * dt) * self.damping
        self._positions += self._velocities * dt

    def get_positions_f32(self) -> np.ndarray:
        """Return current node positions as a float32 numpy array (copy)."""
        return self._positions.astype(np.float32)
