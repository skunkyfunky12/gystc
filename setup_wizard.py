"""First-run setup wizard -- shown when no ~/.gystc/config.json exists."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

IS_MAC = sys.platform == "darwin"

CONFIG_DIR = Path.home() / ".gystc"
CONFIG_PATH = CONFIG_DIR / "config.json"
CLAUDE_DIR = Path.home() / ".claude"
MCP_CONFIG_PATHS = [CLAUDE_DIR / ".mcp.json", Path.home() / ".claude.json"]

_FONT_FAMILY = "-apple-system, 'Helvetica Neue', sans-serif" if IS_MAC else "'Segoe UI', sans-serif"
_MONO_FAMILY = "'SF Mono', Menlo, monospace" if IS_MAC else "'JetBrains Mono', monospace"
_VAULT_PLACEHOLDER = "/Users/.../My Vault" if IS_MAC else "C:\\Users\\...\\My Vault"

STYLE = f"""
QDialog {{ background: #0A0E15; }}
QLabel {{ color: #E6EEFB; font-family: {_FONT_FAMILY}; }}
QLabel#title {{ font-size: 22px; font-weight: bold; color: #5EE9F0; }}
QLabel#subtitle {{ font-size: 12px; color: #A7B4CC; }}
QLabel#section {{ font-size: 13px; font-weight: 600; color: #5EE9F0;
                 margin-top: 16px; }}
QLabel#hint {{ font-size: 11px; color: #5E6A83; }}
QLineEdit {{
    background: #101623; border: 1px solid rgba(180,210,255,0.12);
    border-radius: 6px; padding: 8px 12px;
    color: #E6EEFB; font-family: {_MONO_FAMILY}; font-size: 12px;
}}
QLineEdit:focus {{ border-color: rgba(94,233,240,0.4); }}
QPushButton {{
    background: transparent; border: 1px solid rgba(180,210,255,0.15);
    border-radius: 6px; padding: 8px 16px;
    color: #A7B4CC; font-family: {_MONO_FAMILY}; font-size: 12px;
}}
QPushButton:hover {{ border-color: rgba(94,233,240,0.35); color: #E6EEFB;
                    background: rgba(255,255,255,0.03); }}
QPushButton#primary {{
    border-color: rgba(94,233,240,0.5); color: #5EE9F0;
}}
QPushButton#primary:hover {{
    background: rgba(94,233,240,0.1);
}}
"""


class SetupWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GYSTC Setup")
        self.setFixedSize(560, 520)
        self.setStyleSheet(STYLE)
        self.vault_path = ""
        self.api_key = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(32, 28, 32, 24)

        title = QLabel("GYSTC")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Get Your Shit Together, Claude.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        sec1 = QLabel("1. OBSIDIAN VAULT")
        sec1.setObjectName("section")
        layout.addWidget(sec1)

        vault_row = QHBoxLayout()
        self._vault_input = QLineEdit()
        self._vault_input.setPlaceholderText(_VAULT_PLACEHOLDER)
        vault_row.addWidget(self._vault_input, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_vault)
        vault_row.addWidget(browse_btn)
        layout.addLayout(vault_row)

        hint1 = QLabel("The folder that contains your .obsidian/ directory.")
        hint1.setObjectName("hint")
        layout.addWidget(hint1)

        sec2 = QLabel("2. OBSIDIAN API KEY (optional)")
        sec2.setObjectName("section")
        layout.addWidget(sec2)

        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("Bearer token from Local REST API plugin")
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self._key_input)

        hint2 = QLabel("Enables click-to-open notes in Obsidian. Skip if unsure.")
        hint2.setObjectName("hint")
        layout.addWidget(hint2)

        sec3 = QLabel("3. CLAUDE SETUP")
        sec3.setObjectName("section")
        layout.addWidget(sec3)

        claude_row = QHBoxLayout()

        copy_btn = QPushButton("Copy CLAUDE.md to clipboard")
        copy_btn.clicked.connect(self._copy_claude_template)
        claude_row.addWidget(copy_btn)

        hook_btn = QPushButton("Install MCP + Hooks")
        hook_btn.setObjectName("primary")
        hook_btn.clicked.connect(self._install_hooks)
        claude_row.addWidget(hook_btn)

        layout.addLayout(claude_row)

        hint3 = QLabel(
            "Paste CLAUDE.md into your Claude Code settings so Claude knows your brain.\n"
            "'Install MCP + Hooks' installs the MCP server package, registers it with\n"
            "Claude, and adds the session-start hook. Requires Python 3.11+."
        )
        hint3.setObjectName("hint")
        hint3.setWordWrap(True)
        layout.addWidget(hint3)

        self._status = QLabel("")
        self._status.setObjectName("hint")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.addStretch()
        skip_btn = QPushButton("Skip Setup")
        skip_btn.clicked.connect(self._skip)
        bottom.addWidget(skip_btn)
        start_btn = QPushButton("Save & Launch Dashboard")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._save_and_launch)
        bottom.addWidget(start_btn)
        layout.addLayout(bottom)

    def _browse_vault(self):
        path = QFileDialog.getExistingDirectory(self, "Select Obsidian Vault")
        if path:
            self._vault_input.setText(path)

    def _copy_claude_template(self):
        candidates = [
            Path(__file__).parent / "CLAUDE_TEMPLATE.md",
            Path(getattr(sys, "_MEIPASS", "")) / "CLAUDE_TEMPLATE.md",
        ]
        text = _fallback_claude_template()
        for p in candidates:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                break
        QApplication.clipboard().setText(text)
        self._status.setText("Copied to clipboard. Paste into your CLAUDE.md.")
        self._status.setStyleSheet("color: #22C55E;")

    @staticmethod
    def _find_system_python() -> str:
        if not IS_MAC and not getattr(sys, "frozen", False):
            return sys.executable
        for name in ("python3", "python"):
            found = shutil.which(name)
            if found and ".app/" not in found and "_internal" not in found:
                return found
        return "python3"

    @staticmethod
    def _check_brain_mcp(python_cmd: str) -> bool:
        try:
            r = subprocess.run(
                [python_cmd, "-c", "import brain_mcp; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0 and "ok" in r.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _pip_install_gystc(python_cmd: str) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                [python_cmd, "-m", "pip", "install", "gystc"],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                return True, "installed from PyPI"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        repo_url = "git+https://github.com/skunkyfunky12/gystc.git"
        try:
            r = subprocess.run(
                [python_cmd, "-m", "pip", "install", repo_url],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                return True, "installed from GitHub"
            return False, r.stderr.strip().split("\n")[-1][:200]
        except FileNotFoundError:
            return False, "Python not found"
        except subprocess.TimeoutExpired:
            return False, "install timed out"

    def _install_hooks(self):
        results = []
        python_cmd = self._find_system_python()

        if not self._check_brain_mcp(python_cmd):
            self._status.setText("Installing GYSTC MCP server (this may take a minute)...")
            self._status.setStyleSheet("color: #FBBF24;")
            QApplication.processEvents()

            ok, msg = self._pip_install_gystc(python_cmd)
            if ok:
                results.append(f"MCP package: {msg}")
            else:
                results.append(f"MCP package: FAILED ({msg})")
                self._status.setText(
                    f"Could not install brain_mcp: {msg}\n"
                    f"Manual fix: {python_cmd} -m pip install gystc"
                )
                self._status.setStyleSheet("color: #FF5C7C;")
                return
        else:
            results.append("MCP package: ready")

        mcp_entry = {
            "command": python_cmd,
            "args": ["-m", "brain_mcp"],
            "env": {},
        }

        for mcp_path in MCP_CONFIG_PATHS:
            try:
                existing = {}
                if mcp_path.exists():
                    existing = json.loads(mcp_path.read_text(encoding="utf-8"))
                servers = existing.setdefault("mcpServers", {})
                servers["gystc"] = mcp_entry
                mcp_path.parent.mkdir(parents=True, exist_ok=True)
                mcp_path.write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                results.append(f"MCP config: {mcp_path.name}")
            except Exception as e:
                results.append(f"MCP {mcp_path.name}: {e}")

        hooks_dir = CLAUDE_DIR / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        settings_path = CLAUDE_DIR / "settings.json"
        try:
            settings = {}
            if settings_path.exists():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            hooks = settings.setdefault("hooks", {})
            session_hooks = hooks.setdefault("SessionStart", [])

            already = any(
                "brain_mcp" in h.get("command", "")
                for h in session_hooks
            )
            if not already:
                quoted_py = f'"{python_cmd}"' if " " in python_cmd else python_cmd
                session_hooks.append({
                    "command": f"{quoted_py} -m brain_mcp.hooks.session_start_context",
                    "description": "GYSTC: load vault context on session start",
                })
                settings_path.write_text(
                    json.dumps(settings, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                results.append("Hook: installed")
            else:
                results.append("Hook: already present")
        except Exception as e:
            results.append(f"Hook: {e}")

        self._status.setText(" | ".join(results))
        self._status.setStyleSheet("color: #5EE9F0;")

    def _save_and_launch(self):
        self.vault_path = self._vault_input.text().strip()
        self.api_key = self._key_input.text().strip()
        self._write_config()
        self.accept()

    def _skip(self):
        self._write_config()
        self.accept()

    def reject(self):
        self._write_config()
        super().reject()

    def _write_config(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        vault = self._vault_input.text().strip()
        config = {
            "vault_path": vault if vault else "",
            "obsidian_api_key": self._key_input.text().strip(),
            "graph_path": "graphify-out/graph.json",
        }
        CONFIG_PATH.write_text(
            json.dumps(config, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )


def needs_setup() -> bool:
    return not CONFIG_PATH.exists()


def _find_icon() -> Path | None:
    for base in (Path(__file__).parent / "assets", Path(getattr(sys, "_MEIPASS", "")) / "assets"):
        for ext in (".icns", ".ico", ".svg"):
            p = base / f"gystc-icon{ext}"
            if p.exists():
                return p
    return None


def run_setup(app: QApplication) -> bool:
    icon_path = _find_icon()
    wizard = SetupWizard()
    if icon_path:
        wizard.setWindowIcon(QIcon(str(icon_path)))
    result = wizard.exec()
    return result == QDialog.DialogCode.Accepted


def _fallback_claude_template() -> str:
    return (
        "# GYSTC -- Instructions for Claude\n\n"
        "You have a persistent long-term memory via the GYSTC MCP server.\n"
        "Your vault is your brain -- search it BEFORE answering from general knowledge.\n\n"
        "## Tools: brain_retrieve, brain_store, brain_recent, brain_related, brain_classify, brain_versions\n"
        "## 12 brain regions organize notes by function (not topic).\n"
        "## Search proactively. Store sparingly. Reference paths.\n\n"
        "Full docs: https://gystc.dev\n"
    )
