"""OpenGL widget that hosts the 3D brain graph render loop.

Wires together OrbitCamera, Scene, and PhysicsSimulation into a
repaint loop driven by QTimer at ~60 fps.
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROGRAM_POINT_SIZE,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glClearColor,
    glEnable,
    glViewport,
)

from brain.picking import pick_node


class BrainGLWidget(QOpenGLWidget):
    """QOpenGLWidget that renders the brain graph every frame.

    Parameters
    ----------
    scene:     Scene instance (owns shader programs and VAOs).
    camera:    OrbitCamera instance.
    physics:   PhysicsSimulation instance.
    positions: Initial (N, 3) float32 node positions.
    nodes:     List/sequence of node objects indexed by position index.
    parent:    Optional Qt parent widget.
    """

    def __init__(self, scene, camera, physics, positions, nodes, parent=None):
        super().__init__(parent)
        self._scene = scene
        self._camera = camera
        self._physics = physics
        self._positions = positions
        self._nodes = nodes

        self._start_time: float = time.perf_counter()
        self._mouse_pressed: bool = False
        self._last_mouse_pos = None
        self._mouse_button = None
        self._click_pos = None           # for detecting click vs drag
        self.on_node_clicked = None      # callback, set externally

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """Set up GL state and start the render timer."""
        import sys
        try:
            from OpenGL.GL import glGetString, GL_VERSION, GL_RENDERER
            print(f"OpenGL: {glGetString(GL_VERSION)}")
            print(f"GPU: {glGetString(GL_RENDERER)}")
        except Exception as e:
            print(f"Could not query GL info: {e}", file=sys.stderr)

        glClearColor(3 / 255, 5 / 255, 8 / 255, 1.0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)

        try:
            self._scene.init_gl()
            print("Scene GL initialized OK")
        except Exception as e:
            print(f"Scene init_gl FAILED: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60 fps

    def _tick(self) -> None:
        """Advance physics, push new positions to scene, request repaint."""
        self._physics.tick()
        new_pos = self._physics.get_positions_f32()
        self._positions = new_pos
        self._scene.update_positions(new_pos)
        self.update()

    def paintGL(self) -> None:
        """Render one frame."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view = self._camera.get_view_matrix()
        proj = self._camera.get_projection_matrix(self.width(), self.height())
        t = time.perf_counter() - self._start_time
        try:
            self._scene.render(view, proj, t)
        except Exception as e:
            if not getattr(self, '_render_error_logged', False):
                import sys
                print(f"Render error: {e}", file=sys.stderr)
                self._render_error_logged = True

    def resizeGL(self, w: int, h: int) -> None:
        """Update the GL viewport when the widget is resized."""
        glViewport(0, 0, w, h)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._mouse_pressed = True
        self._mouse_button = event.button()
        self._last_mouse_pos = event.position()
        self._click_pos = event.position()   # will be cleared on drag

    def mouseMoveEvent(self, event) -> None:
        if not self._mouse_pressed:
            return

        current = event.position()
        last = self._last_mouse_pos
        dx = current.x() - last.x()
        dy = current.y() - last.y()

        if self._mouse_button == Qt.MouseButton.LeftButton:
            self._camera.rotate(dx * 0.3, dy * 0.3)
        elif self._mouse_button == Qt.MouseButton.RightButton:
            self._camera.pan(-dx, dy)

        self._last_mouse_pos = current
        self._click_pos = None   # moved → no longer a pure click

    def mouseReleaseEvent(self, event) -> None:
        self._mouse_pressed = False
        if self._click_pos is not None:
            # No drag occurred → treat as a click
            if self._mouse_button == Qt.MouseButton.LeftButton:
                pos = event.position()
                self._handle_click(pos.x(), pos.y())

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        self._camera.zoom(delta)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _handle_click(self, mx: float, my: float) -> None:
        """Run ray-cast pick and fire the on_node_clicked callback."""
        hit = pick_node(
            int(mx), int(my),
            self.width(), self.height(),
            self._camera,
            self._positions,
            node_radius=5.0,
        )
        if hit >= 0 and self.on_node_clicked is not None:
            self.on_node_clicked(self._nodes[hit])
