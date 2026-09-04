"""Headless self-check for the packaged GYSTC Dashboard.

Why this exists: the release build used to install only the GUI dependencies, so
the shipped binary had no faiss, no mcp and no sentence-transformers. The build
still went green -- CI only checked that ``dist/`` existed -- and every search in
the released app answered HTTP 500 ``No module named 'faiss'``. A green build has
to mean the binary can actually search, and only a run of the built artefact can
show that.

``GYSTC Dashboard --selfcheck [report.json]`` exercises the real code path the
dashboard's HTTP API uses (index a throwaway vault, embed it, retrieve it) and
exits non-zero if any step fails. It never touches the user's vault, config,
database or index: everything happens in a temporary directory.

The Windows build is windowed (``console=False``), which leaves ``sys.stdout``
and ``sys.stderr`` as ``None``. Nothing here may assume they are writable, so the
result travels in a JSON report file and in the exit code.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# Third-party packages that must be inside the bundle. Each one was missing from
# the 1.4.3 release build, where their absence surfaced as an HTTP 500 per search.
RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "requests",
    "faiss",
    "httpx",
    "psutil",
    "mcp",
    "torch",
    "transformers",
    "sentence_transformers",
    # The dashboard extra. Both are already imported by main.py before this
    # runs, so naming them here costs nothing and keeps the list honest about
    # what the binary needs.
    "PyQt6",
    "PyQt6.QtWebEngineWidgets",
)

# Declared project dependencies the packaged dashboard deliberately does not
# carry, with the reason. Every dependency belongs in exactly one of the two
# lists (enforced by tests/test_release_bundle.py), so adding one forces a
# decision instead of a silent gap in the check.
NOT_BUNDLED = {
    "watchdog": (
        "only brain_mcp/indexer/watcher.py imports it, and that runs in the MCP "
        "server; the dashboard indexes on demand via /api/reindex and never "
        "starts a file watcher"
    ),
}

# The brain_mcp modules brain/web_widget.py imports inside its request handlers.
BRAIN_MCP_MODULES = (
    "brain_mcp.config",
    "brain_mcp.storage.database",
    "brain_mcp.storage.migrations",
    "brain_mcp.storage.file_lock",
    "brain_mcp.indexer.vector_store",
    "brain_mcp.indexer.embedder",
    "brain_mcp.indexer.chunker",
    "brain_mcp.indexer.scanner",
    "brain_mcp.indexer.pipeline",
    "brain_mcp.tools.retrieve",
    "brain_mcp.tools.recent",
)

_PROBE_TITLE = "Selfcheck Probe Note"
_PROBE_QUERY = "wiederauffindbare Sonde"
_PROBE_BODY = (
    f"# {_PROBE_TITLE}\n\n"
    "Dies ist eine wiederauffindbare Sonde fuer den Selbsttest des Pakets.\n"
)


def _ensure_streams() -> io.StringIO:
    """Give the frozen windowed build writable stdout/stderr, and capture them."""
    sink = io.StringIO()
    if sys.stdout is None:
        sys.stdout = sink
    if sys.stderr is None:
        sys.stderr = sink
    return sink


def _force_offline(scratch: Path) -> None:
    """Cut every route to the Hugging Face hub, before anything imports it.

    ``huggingface_hub`` reads its offline flag into a module constant at import
    time, so setting it later -- as ``SentenceTransformerBackend._load`` does --
    is a no-op once the package is loaded. This must therefore run before the
    first import, or the check would pass on a networked build machine by
    quietly downloading what the bundle is missing.

    ``HF_HOME`` is redirected too: the build step that created the bundle also
    warmed the runner's cache, and a load served from that cache would prove
    nothing about the files inside the artefact.
    """
    empty_home = scratch / "hf-home"
    empty_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(empty_home)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _check_imports(names: tuple[str, ...]) -> list[str]:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # ImportError, but a broken wheel raises others
            missing.append(f"{name}: {exc}")
    return missing


def _run_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    missing = _check_imports(RUNTIME_PACKAGES)
    steps.append({
        "name": "runtime_packages",
        "ok": not missing,
        "detail": "all present" if not missing else "; ".join(missing),
    })

    missing = _check_imports(BRAIN_MCP_MODULES)
    steps.append({
        "name": "brain_mcp_modules",
        "ok": not missing,
        "detail": "all present" if not missing else "; ".join(missing),
    })

    if not all(step["ok"] for step in steps):
        # Without the imports nothing below can run; report what we know.
        steps.append({"name": "embedding_model", "ok": False, "detail": "skipped"})
        steps.append({"name": "index_and_retrieve", "ok": False, "detail": "skipped"})
        return steps

    from brain_mcp.config import DEFAULT_MODEL
    from brain_mcp.indexer.bundled_model import ENV_VAR, bundled_model_dir
    from brain_mcp.indexer.embedder import SentenceTransformerBackend
    from brain_mcp.indexer.pipeline import index_vault
    from brain_mcp.indexer.vector_store import VectorStore
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.tools.retrieve import handle_brain_retrieve

    model_dir = bundled_model_dir(DEFAULT_MODEL)
    if model_dir is None:
        steps.append({
            "name": "embedding_model",
            "ok": False,
            "detail": (
                f"no bundled directory identifying itself as {DEFAULT_MODEL} -- "
                "the packaged app runs with HF_HUB_OFFLINE=1 and cannot fetch "
                f"one (set {ENV_VAR} to test against a local copy)"
            ),
        })
        steps.append({"name": "index_and_retrieve", "ok": False, "detail": "skipped"})
        return steps

    # Pass the hub name, not the directory: this is exactly what the app does,
    # so the check covers the resolution step too, not just the files.
    embedder = SentenceTransformerBackend(DEFAULT_MODEL, eager=True)
    if not embedder.is_ready:
        steps.append({
            "name": "embedding_model",
            "ok": False,
            "detail": f"model at {model_dir} failed to load",
        })
        steps.append({"name": "index_and_retrieve", "ok": False, "detail": "skipped"})
        return steps

    vector = embedder.embed(["probe"])
    dim_ok = vector.shape == (1, embedder.dimension)
    steps.append({
        "name": "embedding_model",
        "ok": bool(dim_ok),
        "detail": f"{model_dir} -> dim {embedder.dimension}, shape {vector.shape}",
    })
    if not dim_ok:
        steps.append({"name": "index_and_retrieve", "ok": False, "detail": "skipped"})
        return steps

    workdir = Path(tempfile.mkdtemp(prefix="gystc-selfcheck-"))
    db = None
    try:
        vault = workdir / "vault"
        vault.mkdir()
        (vault / f"{_PROBE_TITLE}.md").write_text(_PROBE_BODY, encoding="utf-8")

        db = BrainDB(workdir / "brain.db")
        vectors = VectorStore(dimension=embedder.dimension)
        embedded = index_vault(db, vectors, embedder, vault, {})
        results = handle_brain_retrieve(db, vectors, embedder, query=_PROBE_QUERY, limit=5)
        titles = [r.get("title") for r in results if isinstance(r, dict)]
        found = _PROBE_TITLE in titles
        steps.append({
            "name": "index_and_retrieve",
            "ok": bool(found and embedded == 1),
            "detail": f"embedded {embedded} note(s), retrieved {titles}",
        })
    finally:
        if db is not None:
            db.close()
        shutil.rmtree(workdir, ignore_errors=True)

    return steps


def run_selfcheck(report_path: Path | None = None) -> int:
    """Run every check, write a JSON report, return a process exit code."""
    sink = _ensure_streams()
    scratch = Path(tempfile.mkdtemp(prefix="gystc-selfcheck-env-"))
    try:
        _force_offline(scratch)
        steps = _run_steps()
        crash = None
    except Exception:
        steps = []
        crash = traceback.format_exc()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    ok = bool(steps) and all(step["ok"] for step in steps) and crash is None
    report = {
        "ok": ok,
        "executable": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "steps": steps,
        "crash": crash,
        "output": sink.getvalue(),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)

    if report_path is not None:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            # The report is a convenience; the exit code is the contract. Say so
            # rather than turning a passing check into a failure.
            print(f"selfcheck: could not write {report_path}: {exc}", file=sys.stderr)

    print(text, file=sys.stderr)
    return 0 if ok else 1
