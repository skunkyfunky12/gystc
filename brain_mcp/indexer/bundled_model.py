"""Locate an embedding model that ships *with* the application.

GYSTC never reaches the network at runtime: ``SentenceTransformerBackend._load``
sets ``HF_HUB_OFFLINE=1``, so ``SentenceTransformer("<hub-name>")`` resolves from
the local Hugging Face cache or not at all. A pip install on a developer machine
usually has that cache. A packaged release does not -- and the failure is silent:
the load raises, the backend stays unready, and search quietly degrades to
keyword-only for the rest of the process.

The fix is to ship the model files inside the bundle and load them by path.
Offline stays on; there is simply nothing left to fetch.

A bundled directory must never *override* a model the user chose. ``model_name``
is configurable (``BRAIN_MODEL_NAME`` / ``model_name`` in config.json), and a
silent swap would be worse than the bug this module fixes: the vault would be
embedded by a model nobody picked, and if its dimension differs from the index
on disk, ``VectorStore.load`` deletes that index. So the bundle records which
model it contains, and the directory is only used when the recorded name matches
what was asked for.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Written by SentenceTransformer.save(); their presence is what separates a real
# model directory from an empty placeholder that would fail deep inside torch.
_REQUIRED_FILES = ("config.json", "modules.json")

# Written by scripts/bundle_model.py: which model these files actually are.
MARKER_FILE = "gystc-model.json"

ENV_VAR = "GYSTC_MODEL_DIR"


def _has_model_files(path: Path) -> bool:
    try:
        return path.is_dir() and all((path / name).is_file() for name in _REQUIRED_FILES)
    except OSError:
        return False


def write_marker(path: Path, model_name: str) -> Path:
    """Record which model was saved into *path*."""
    marker = path / MARKER_FILE
    marker.write_text(
        json.dumps({"model_name": model_name}, indent=2) + "\n", encoding="utf-8"
    )
    return marker


def read_marker(path: Path) -> str | None:
    """Return the model name recorded in *path*, or ``None`` if unmarked."""
    try:
        data = json.loads((path / MARKER_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get("model_name") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


def candidate_model_dirs() -> list[Path]:
    """Every place a bundled model may live, most specific first."""
    candidates: list[Path] = []

    # PyInstaller onedir: sys._MEIPASS is the _internal directory, and the spec
    # ships the whole assets/ tree, so assets/model travels with it.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "assets" / "model")

    # Running from a source checkout: <repo>/assets/model
    candidates.append(Path(__file__).resolve().parents[2] / "assets" / "model")

    return candidates


def bundled_model_dir(model_name: str | None = None) -> Path | None:
    """Return a usable bundled model directory, or ``None``.

    ``GYSTC_MODEL_DIR`` wins unconditionally -- pointing at a directory is an
    explicit choice. Anything found automatically must identify itself as
    *model_name*; an unmarked or differently-marked directory is ignored, so the
    configured model still decides.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        path = Path(override)
        return path if _has_model_files(path) else None

    for candidate in candidate_model_dirs():
        if not _has_model_files(candidate):
            continue
        if model_name is not None and read_marker(candidate) != model_name:
            continue
        return candidate
    return None


def resolve_model_source(model_name: str) -> str:
    """Return what to hand to ``SentenceTransformer``.

    A matching bundled directory wins over the hub name: it is the only source
    that works with ``HF_HUB_OFFLINE=1`` on a machine that has never seen the
    model. Anything else falls back to the name, which is what a developer
    install with a warm Hugging Face cache expects.
    """
    bundled = bundled_model_dir(model_name)
    return str(bundled) if bundled is not None else model_name
