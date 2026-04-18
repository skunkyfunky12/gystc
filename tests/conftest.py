# tests/conftest.py
import json
from pathlib import Path
import numpy as np
import pytest


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
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    config = {"vault_path": str(tmp_vault)}
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config_dir


class MockEmbedder:
    """Deterministic embedder for testing. Maps text hash to a fixed 384-dim vector."""

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def mock_embedder():
    return MockEmbedder()
