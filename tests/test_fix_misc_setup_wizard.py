"""Regression tests for audit finding 22 (misc group).

setup_wizard._install_hooks / _write_config used Path.write_text (truncate-
then-write) on ~/.claude.json, ~/.claude/settings.json and ~/.gystc/config.json:
a crash / kill / full disk mid-write would leave the user's PRIMARY Claude
config truncated. These tests pin the fix: every config write goes through
_atomic_write_json (sibling tmp file + os.replace, atomic on Windows/POSIX),
keeps a timestamped .bak of the prior content, and merges into the existing
JSON instead of clobbering unrelated keys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")  # setup_wizard imports PyQt6 at module level

import setup_wizard

ROOT = Path(setup_wizard.__file__).resolve().parent


# ------------------------------------------------------------- helper unit

def test_atomic_write_creates_file_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "config.json"
    setup_wizard._atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))
    # no prior content -> nothing to back up
    assert not list(tmp_path.glob("*.bak"))


def test_atomic_write_backs_up_previous_content(tmp_path):
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"old": True}), encoding="utf-8")
    setup_wizard._atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    baks = list(tmp_path.glob("claude.json.*.bak"))
    assert len(baks) == 1
    assert json.loads(baks[0].read_text(encoding="utf-8")) == {"old": True}


def test_atomic_write_never_truncates_target(tmp_path, monkeypatch):
    """The target must keep its old content right up to the atomic os.replace,
    and survive untouched if the replace fails. The old write_text pattern
    truncated the file BEFORE writing the new bytes."""
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"old": True}), encoding="utf-8")

    seen = {}

    def checking_replace(src, dst, *args, **kwargs):
        src, dst = Path(src), Path(dst)
        # at replace time the target still holds the OLD content...
        seen["target_at_replace"] = json.loads(dst.read_text(encoding="utf-8"))
        # ...and the fully serialized NEW content sits in a SIBLING tmp file
        # (same dir => same filesystem => os.replace is atomic)
        seen["tmp_in_same_dir"] = src.parent == dst.parent
        seen["tmp_content"] = json.loads(src.read_text(encoding="utf-8"))
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", checking_replace)
    with pytest.raises(OSError, match="simulated crash"):
        setup_wizard._atomic_write_json(target, {"new": True})
    monkeypatch.undo()

    assert seen["target_at_replace"] == {"old": True}
    assert seen["tmp_in_same_dir"] is True
    assert seen["tmp_content"] == {"new": True}
    # a failed replace leaves the original target fully intact
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}


# ------------------------------------------------ no raw truncate-writes left

def test_config_writes_route_through_atomic_helper():
    src = (ROOT / "setup_wizard.py").read_text(encoding="utf-8")
    assert "mcp_path.write_text(" not in src
    assert "settings_path.write_text(" not in src
    assert "CONFIG_PATH.write_text(" not in src
    # def + 3 call sites (mcp loop, settings.json, _write_config)
    assert src.count("_atomic_write_json(") >= 4


# ----------------------------------------------------------------- end to end

@pytest.fixture()
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_write_config_merges_and_keeps_backup(tmp_path, monkeypatch, qapp):
    cfg_dir = tmp_path / ".gystc"
    cfg_path = cfg_dir / "config.json"
    cfg_dir.mkdir()
    cfg_path.write_text(
        json.dumps({"model_name": "MiniLM", "vault_path": "old"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(setup_wizard, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(setup_wizard, "CONFIG_PATH", cfg_path)

    wizard = setup_wizard.SetupWizard()
    wizard._vault_input.setText(str(tmp_path / "MyVault"))
    wizard._write_config()

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["model_name"] == "MiniLM"  # MCP-server-owned key preserved
    assert saved["vault_path"] == str(tmp_path / "MyVault")
    baks = list(cfg_dir.glob("config.json.*.bak"))
    assert len(baks) == 1
    assert json.loads(baks[0].read_text(encoding="utf-8"))["vault_path"] == "old"


def test_install_hooks_merges_claude_json_and_settings(tmp_path, monkeypatch, qapp):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    claude_json = tmp_path / ".claude.json"  # the user's primary Claude config
    claude_json.write_text(
        json.dumps({"numStartups": 42, "mcpServers": {"other": {"command": "x"}}}),
        encoding="utf-8",
    )
    settings_path = claude_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"permissions": {"allow": ["Bash"]}}), encoding="utf-8"
    )

    monkeypatch.setattr(setup_wizard, "CLAUDE_DIR", claude_dir)
    monkeypatch.setattr(
        setup_wizard, "MCP_CONFIG_PATHS", [claude_dir / ".mcp.json", claude_json]
    )
    monkeypatch.setattr(
        setup_wizard.SetupWizard,
        "_find_system_python",
        staticmethod(lambda: r"C:\fake\python.exe"),
    )
    monkeypatch.setattr(
        setup_wizard.SetupWizard, "_check_brain_mcp", staticmethod(lambda cmd: True)
    )

    wizard = setup_wizard.SetupWizard()
    wizard._install_hooks()

    # ~/.claude.json: gystc merged in, unrelated keys preserved, .bak created
    data = json.loads(claude_json.read_text(encoding="utf-8"))
    assert data["numStartups"] == 42
    assert data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["gystc"]["args"] == ["-m", "brain_mcp"]
    assert len(list(tmp_path.glob(".claude.json.*.bak"))) == 1

    # settings.json: hooks added, unrelated keys preserved, .bak created
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["permissions"] == {"allow": ["Bash"]}
    assert "SessionStart" in settings["hooks"]
    assert len(list(claude_dir.glob("settings.json.*.bak"))) == 1
