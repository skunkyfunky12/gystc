"""Orbit camera for 3D navigation around the brain graph.

Supports rotate, zoom, and pan. All matrix math uses numpy only.
Angles are stored in degrees and converted to radians internally.
"""
import math
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Standard OpenGL look-at matrix.

    Parameters
    ----------
    eye:    camera position (vec3)
    center: point the camera looks at (vec3)
    up:     world-up direction (vec3)

    Returns
    -------
    4x4 float32 view matrix
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = center - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-12:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f = f / f_norm

    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-12:
        s = np.array([1.0, 0.0, 0.0])
    else:
        s = s / s_norm

    u = np.cross(s, f)

    m = np.array([
        [ s[0],  s[1],  s[2], -np.dot(s, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [  0.0,   0.0,   0.0,             1.0],
    ], dtype=np.float32)
    return m


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard perspective projection matrix (OpenGL convention).

    Parameters
    ----------
    fov_deg: vertical field-of-view in degrees
    aspect:  width / height
    near:    near clipping plane
    far:     far clipping plane

    Returns
    -------
    4x4 float32 projection matrix
    """
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    depth_range = near - far

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / depth_range
    m[2, 3] = (2.0 * far * near) / depth_range
    m[3, 2] = -1.0
    return m


# ---------------------------------------------------------------------------
# OrbitCamera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """Orbit camera that rotates, zooms, and pans around a target point."""

    # Default state
    _DEFAULT_TARGET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    _DEFAULT_DISTANCE = 300.0
    _DEFAULT_YAW = 0.0
    _DEFAULT_PITCH = 20.0
    _DEFAULT_FOV = 45.0
    _DEFAULT_NEAR = 1.0
    _DEFAULT_FAR = 5000.0

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Public state manipulation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore all camera parameters to their defaults."""
        self.target = self._DEFAULT_TARGET.copy()
        self.distance = self._DEFAULT_DISTANCE
        self.yaw = self._DEFAULT_YAW
        self.pitch = self._DEFAULT_PITCH
        self.fov = self._DEFAULT_FOV
        self.near = self._DEFAULT_NEAR
        self.far = self._DEFAULT_FAR

    def rotate(self, dx: float, dy: float) -> None:
        """Adjust yaw by *dx* degrees and pitch by *dy* degrees.

        Pitch is clamped to [-89, 89] to avoid gimbal lock at the poles.
        """
        self.yaw += dx
        self.pitch = max(-89.0, min(89.0, self.pitch + dy))

    def zoom(self, delta: float) -> None:
        """Multiply distance by ``(1 + delta * 0.1)``.

        Distance is clamped to [10, 2000].
        """
        self.distance *= 1.0 + delta * 0.1
        self.distance = max(10.0, min(2000.0, self.distance))

    def pan(self, dx: float, dy: float) -> None:
        """Translate the target in screen-aligned right/up directions.

        Movement is scaled by ``distance * 0.001`` so panning feels
        consistent regardless of how far away the camera is.
        """
        eye = self.get_eye_position()
        forward = self.target - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-12:
            return
        forward = forward / forward_norm

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm

        up = np.cross(right, forward)

        scale = self.distance * 0.001
        self.target += right * (dx * scale) + up * (dy * scale)

    def fly_to(self, target: np.ndarray, distance: float | None = None) -> None:
        """Move the camera target and optionally set a new distance."""
        self.target = np.asarray(target, dtype=np.float64).copy()
        if distance is not None:
            self.distance = float(distance)

    # ------------------------------------------------------------------
    # Matrix generation
    # ------------------------------------------------------------------

    def get_eye_position(self) -> np.ndarray:
        """Compute the eye position from spherical coordinates.

        Returns a numpy vec3 (float64).
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        cos_pitch = math.cos(pitch_rad)
        x = self.distance * cos_pitch * math.sin(yaw_rad)
        y = self.distance * math.sin(pitch_rad)
        z = self.distance * cos_pitch * math.cos(yaw_rad)

        return self.target + np.array([x, y, z], dtype=np.float64)

    def get_view_matrix(self) -> np.ndarray:
        """Return the 4x4 float32 view matrix."""
        eye = self.get_eye_position()
        up = np.array([0.0, 1.0, 0.0])
        return _look_at(eye, self.target, up)

    def get_projection_matrix(self, width: int, height: int) -> np.ndarray:
        """Return the 4x4 float32 perspective projection matrix."""
        aspect = width / max(height, 1)
        return _perspective(self.fov, aspect, self.near, self.far)
