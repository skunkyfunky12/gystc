"""Regression tests: the setup wizard must never install from PyPI.

The PyPI name ``gystc`` is not ours (pypi.org/pypi/gystc/json -> 404 on
2026-09-23). _pip_install_gystc used to try ``pip install gystc`` FIRST and
only fall back to the GitHub repo -- whoever registers that name would ship
code to every user who clicks "Install MCP + Hooks" (dependency confusion).
These tests pin the fix: exactly one pip call, against our own repo, pinned
to the release tag the wizard itself was built from.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")  # setup_wizard imports PyQt6 at module level

import setup_wizard
from brain_mcp import __version__

ROOT = Path(setup_wizard.__file__).resolve().parent
EXPECTED_SPEC = f"git+https://github.com/skunkyfunky12/gystc.git@v{__version__}"


class _Recorder:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.calls: list[list[str]] = []
        self._rc = returncode
        self._stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self._rc, stdout="", stderr=self._stderr)


def test_install_uses_only_the_pinned_repo_url(monkeypatch):
    rec = _Recorder(returncode=0)
    monkeypatch.setattr(setup_wizard.subprocess, "run", rec)

    ok, msg = setup_wizard.SetupWizard._pip_install_gystc("python")

    assert ok
    assert rec.calls == [["python", "-m", "pip", "install", EXPECTED_SPEC]]
    assert "GitHub" in msg


def test_failed_install_never_falls_back_to_pypi(monkeypatch):
    rec = _Recorder(returncode=1, stderr="fatal: could not read from remote")
    monkeypatch.setattr(setup_wizard.subprocess, "run", rec)

    ok, msg = setup_wizard.SetupWizard._pip_install_gystc("python")

    assert not ok
    assert "could not read from remote" in msg
    assert len(rec.calls) == 1
    assert rec.calls[0][-1] == EXPECTED_SPEC


def test_no_bare_pypi_name_anywhere_in_the_wizard():
    """The manual-fix hint shown on failure must not point at PyPI either."""
    src = (ROOT / "setup_wizard.py").read_text(encoding="utf-8")
    assert '"install", "gystc"' not in src
    assert "pip install gystc\"" not in src
    assert "pip install gystc\n" not in src
