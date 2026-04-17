"""Tests for brain.camera.OrbitCamera — written first (TDD)."""
import numpy as np
import pytest

from brain.camera import OrbitCamera


def test_view_matrix_is_4x4():
    """get_view_matrix() must return a (4, 4) float32 array."""
    cam = OrbitCamera()
    matrix = cam.get_view_matrix()
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.float32


def test_zoom_changes_distance():
    """zoom(-1) must decrease the camera distance."""
    cam = OrbitCamera()
    initial_distance = cam.distance
    cam.zoom(-1)
    assert cam.distance < initial_distance


def test_rotate_changes_yaw():
    """rotate(10, 0) must change the yaw angle."""
    cam = OrbitCamera()
    initial_yaw = cam.yaw
    cam.rotate(10, 0)
    assert cam.yaw != initial_yaw


def test_projection_matrix_is_4x4():
    """get_projection_matrix(800, 600) must return a (4, 4) float32 array."""
    cam = OrbitCamera()
    matrix = cam.get_projection_matrix(800, 600)
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.float32
