"""Step 0 of curation: make the vault a LOCAL git repo so every change is
reversible (`git revert`). No remote, ever. Secrets (.obsidian) never tracked."""
from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

# Must be ignored BEFORE we ever `git add -A`. .obsidian holds plugin secrets
# (Local REST API key) + volatile UI state; the rest are trash/derived artifacts.
_REQUIRED_IGNORES = [".obsidian/", ".trash/", "graphify-out/", "*.tmp"]


def _git(vault: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(vault), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=check,
    )


def is_repo(vault: Path) -> bool:
    r = _git(vault, "rev-parse", "--is-inside-work-tree", check=False)
    return r.returncode == 0 and r.stdout.strip() == "true"


def head_sha(vault: Path) -> str | None:
    r = _git(vault, "rev-parse", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _ensure_secret_gitignore(vault: Path) -> None:
    """Guarantee secrets/trash/derived are ignored BEFORE staging anything."""
    gi = vault / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    missing = [p for p in _REQUIRED_IGNORES if p not in existing]
    if not missing:
        return
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    gi.write_text(
        existing + sep
        + "# Added by vault-curation: never track secrets / trash / derived artifacts.\n"
        + "\n".join(missing) + "\n",
        encoding="utf-8",
    )


def _ensure_identity(vault: Path) -> None:
    """Commits need an author. Honor the existing (global) identity; fall back local."""
    if not _git(vault, "config", "user.email", check=False).stdout.strip():
        _git(vault, "config", "user.email", "vault-curation@local")
    if not _git(vault, "config", "user.name", check=False).stdout.strip():
        _git(vault, "config", "user.name", "Vault Curation")
    _git(vault, "config", "commit.gpgsign", "false")  # no signing prompts/hangs


def ensure_vault_repo(vault: Path, *, snapshot_message: str | None = None) -> dict:
    """Make the vault a local git repo with a full snapshot. Idempotent. No remote."""
    vault = Path(vault)
    if is_repo(vault):
        return {"status": "already", "commit": head_sha(vault)}
    _git(vault, "init")
    _ensure_identity(vault)
    _ensure_secret_gitignore(vault)   # protect secrets BEFORE `git add -A`
    _git(vault, "add", "-A")
    msg = snapshot_message or f"Vault-Snapshot vor Curation ({date.today().isoformat()})"
    _git(vault, "commit", "-m", msg)
    return {"status": "initialized", "commit": head_sha(vault)}


def commit_run(vault: Path, message: str) -> str | None:
    """Commit one curation run's changes. Returns the sha, or None if nothing changed.
    Local-only; never pushes."""
    vault = Path(vault)
    _git(vault, "add", "-A")
    if _git(vault, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return None  # nothing staged
    _ensure_identity(vault)
    _git(vault, "commit", "-m", message)
    return head_sha(vault)
