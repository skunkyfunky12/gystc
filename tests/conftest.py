# tests/conftest.py
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_QT_MODS = [
    "PyQt6", "PyQt6.QtCore", "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtWebChannel", "PyQt6.QtWebEngineCore", "PyQt6.QtGui",
    "PyQt6.QtWidgets",
]


@pytest.fixture(scope="module")
def qt_stubbed():
    """Stub PyQt6 in sys.modules so brain.web_widget imports headless.

    Installed around the requesting module's tests only and fully removed
    afterwards (including the mock-flavored brain.web_widget): a module-level
    sys.modules mock runs at collection time and leaks into every later test
    module — it broke the real-Qt setup_wizard tests. Opt in per module with
    `pytestmark = pytest.mark.usefixtures("qt_stubbed")` and import
    brain.web_widget inside tests/fixtures, never at module level."""
    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m in _QT_MODS}
    sys.modules.pop("brain.web_widget", None)
    for m in _QT_MODS:
        sys.modules[m] = MagicMock()
    yield
    for m in _QT_MODS:
        sys.modules.pop(m, None)
    sys.modules.pop("brain.web_widget", None)
    sys.modules.update(saved)


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text(
        "# Hello\nThis is about routing and maps.\n[[note2]]\n#brain/hippocampus",
        encoding="utf-8",
    )
    (vault / "note2.md").write_text(
        "# World\nCSP middleware handles security headers.\n[[note1]]\n#brain/praefrontaler-cortex",
        encoding="utf-8",
    )
    sub = vault / "Projects"
    sub.mkdir()
    (sub / "project1.md").write_text(
        "# Project\nA project about data pipelines.\n[[note1]]",
        encoding="utf-8",
    )
    return vault


@pytest.fixture
def tmp_config_dir(tmp_path, tmp_vault):
    config_dir = tmp_path / ".gystc"
    config_dir.mkdir()
    config = {"vault_path": str(tmp_vault)}
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config_dir


class MockEmbedder:
    """Deterministic embedder for testing. Maps text hash to a fixed 384-dim vector."""

    is_ready = True

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def mock_embedder():
    return MockEmbedder()
