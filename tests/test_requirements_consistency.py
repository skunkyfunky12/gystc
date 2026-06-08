"""Pin: `pip install -r requirements.txt` must yield a RUNNABLE MCP server.

Regression for the stale requirements.txt that listed only GUI deps (PyQt6,
PyOpenGL, scipy...) and omitted every core runtime dep (mcp, sentence-transformers,
faiss-cpu, watchdog, psutil) -> a fresh `pip install -r requirements.txt` produced
a non-functional server. pyproject.toml is the single source of truth; this test
ensures requirements.txt either installs the package (pulling those deps) or lists
them all, so the two never silently diverge again.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _core_runtime_deps() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = []
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].strip().lower()
        if name:
            names.append(name)
    return names


def test_requirements_txt_yields_a_runnable_server():
    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        ln.strip() for ln in req_text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    # Installing the local package (any form of `.` / `-e .`, optionally with
    # extras) pulls pyproject's dependencies -> runnable, and keeps a single
    # source of truth.
    def _installs_package(ln: str) -> bool:
        body = ln[2:].strip() if ln.startswith("-e") else ln
        return body == "." or body.startswith(".[")

    if any(_installs_package(ln) for ln in lines):
        return

    haystack = req_text.lower()
    missing = [dep for dep in _core_runtime_deps() if dep not in haystack]
    assert not missing, (
        "requirements.txt would install a non-functional server; missing core "
        f"runtime deps: {missing}. Either list them or install the package "
        "(e.g. `-e .[dashboard]`)."
    )
