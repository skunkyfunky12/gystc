"""Ray-casting picker for 3D node selection.

Converts a 2D mouse position to a world-space ray and tests intersection
with axis-aligned spheres around each graph node.
"""
from __future__ import annotations

import math

import numpy as np


def pick_node(
    mx: int,
    my: int,
    w: int,
    h: int,
    camera,
    positions: np.ndarray,
    node_radius: float = 5.0,
) -> int:
    """Return the index of the closest node hit by a ray through (mx, my).

    Parameters
    ----------
    mx, my:      Mouse position in screen pixels (origin = top-left).
    w, h:        Viewport width and height in pixels.
    camera:      An ``OrbitCamera`` instance.
    positions:   (N, 3) float array of node world-space positions.
    node_radius: Sphere radius used for intersection testing.

    Returns
    -------
    Index of the closest intersected node, or -1 if no node was hit.
    """
    # ------------------------------------------------------------------
    # 1. Screen -> NDC
    # ------------------------------------------------------------------
    ndc_x = (2.0 * mx / w) - 1.0
    ndc_y = 1.0 - (2.0 * my / h)   # Y is flipped (screen Y grows downward)

    # ------------------------------------------------------------------
    # 2. Unproject to world-space ray direction
    # ------------------------------------------------------------------
    proj = camera.get_projection_matrix(w, h).astype(np.float64)
    view = camera.get_view_matrix().astype(np.float64)

    inv_proj = np.linalg.inv(proj)
    inv_view = np.linalg.inv(view)

    # Clip space point on the near plane
    clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)

    # Eye-space direction (undo projection)
    eye_coords = inv_proj @ clip
    eye_coords[2] = -1.0   # we want a direction, pointing into the scene
    eye_coords[3] = 0.0    # direction, not a point

    # World-space direction (undo view)
    world_dir = (inv_view @ eye_coords)[:3]
    length = np.linalg.norm(world_dir)
    if length < 1e-12:
        return -1
    direction = world_dir / length

    ray_origin = camera.get_eye_position().astype(np.float64)

    # ------------------------------------------------------------------
    # 3. Ray-sphere intersection for every node
    # ------------------------------------------------------------------
    best_idx = -1
    best_t = math.inf
    r2 = node_radius * node_radius

    positions = np.asarray(positions, dtype=np.float64)
    for i, pos in enumerate(positions):
        oc = ray_origin - pos
        b = 2.0 * np.dot(direction, oc)
        c = np.dot(oc, oc) - r2
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            continue
        t = (-b - math.sqrt(discriminant)) / 2.0
        if t > 0.0 and t < best_t:
            best_t = t
            best_idx = i

    return best_idx
