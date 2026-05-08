"""Main application window for the GYSTC Dashboard."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow

from data.regions import REGIONS


# Key-to-region-index mapping for numeric keys 1-9, 0, -, =
_KEY_REGION_MAP = {
    Qt.Key.Key_1: 0,
    Qt.Key.Key_2: 1,
    Qt.Key.Key_3: 2,
    Qt.Key.Key_4: 3,
    Qt.Key.Key_5: 4,
    Qt.Key.Key_6: 5,
    Qt.Key.Key_7: 6,
    Qt.Key.Key_8: 7,
    Qt.Key.Key_9: 8,
    Qt.Key.Key_0: 9,
    Qt.Key.Key_Minus: 10,
    Qt.Key.Key_Equal: 11,
}


class BrainWindow(QMainWindow):
    """Main window that wraps the OpenGL brain widget.

    Parameters
    ----------
    gl_widget: BrainGLWidget instance to use as the central widget.
    camera:    OrbitCamera instance for keyboard-driven navigation.
    """

    def __init__(self, gl_widget, camera) -> None:
        super().__init__()
        self._gl_widget = gl_widget
        self._camera = camera

        self.setWindowTitle("GYSTC Dashboard")
        self.setStyleSheet("background-color: #030508;")
        self.setCentralWidget(gl_widget)
        self.showMaximized()

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts.

        ESC     → close window
        F       → toggle fullscreen / maximized
        R       → reset camera
        1-9     → fly to regions 0-8
        0       → fly to region 9
        - (Minus)   → fly to region 10
        = (Equal)   → fly to region 11
        """
        key = event.key()

        if key == Qt.Key.Key_Escape:
            self.close()

        elif key == Qt.Key.Key_F:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()

        elif key == Qt.Key.Key_R:
            self._camera.reset()

        else:
            idx = _KEY_REGION_MAP.get(key)
            if idx is not None and idx < len(REGIONS):
                target = np.array(REGIONS[idx]["position"], dtype=np.float32)
                self._camera.fly_to(target, distance=150.0)

        super().keyPressEvent(event)
