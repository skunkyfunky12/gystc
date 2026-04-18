"""Minimal OpenGL window test - no scene, no shaders, just a colored background."""
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import glClearColor, glClear, GL_COLOR_BUFFER_BIT


class MinimalGL(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.2, 0.0, 0.3, 1.0)
        print("initializeGL OK - purple background set")

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)


app = QApplication(sys.argv)
win = QMainWindow()
gl = MinimalGL()
win.setCentralWidget(gl)
win.setWindowTitle("Minimal GL Test")
win.resize(800, 600)
win.show()
print("Window shown - you should see a purple window")
sys.exit(app.exec())
