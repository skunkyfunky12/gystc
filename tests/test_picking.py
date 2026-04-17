"""Tests for brain.picking — ray-casting node picker (TDD)."""
import numpy as np
import pytest

from brain.picking import pick_node


def test_pick_node_at_origin():
    """A node at the origin is hit when clicking the screen centre."""
    from brain.camera import OrbitCamera

    cam = OrbitCamera()  # target=[0,0,0], distance=300, yaw=0, pitch=20
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # Screen centre of an 800x600 viewport should point straight at the origin.
    result = pick_node(400, 300, 800, 600, cam, positions, node_radius=5.0)
    assert result == 0


def test_pick_miss_returns_negative():
    """Clicking the far corner of the screen should miss the node at origin."""
    from brain.camera import OrbitCamera

    cam = OrbitCamera()
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # Top-left corner (0, 0) points far away from the origin.
    result = pick_node(0, 0, 800, 600, cam, positions, node_radius=5.0)
    assert result == -1
