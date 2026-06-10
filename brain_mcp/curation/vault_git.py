"""Step 0 of curation: make the vault a LOCAL git repo so every change is
reversible (`git revert`). No remote, ever. Secrets (.obsidian) never tracked."""
from __future__ import annotations

import subprocess
import sys
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
    """True only if the vault itself is the repo ROOT. A vault merely nested in
    an enclosing repo must NOT count: init/snapshot/secret-gitignore would be
    silently skipped and `git add -A` would sweep the whole parent worktree."""
    r = _git(vault, "rev-parse", "--show-toplevel", check=False)
    top = r.stdout.strip()
    if r.returncode != 0 or not top:
        return False
    try:
        return Path(top).resolve() == Path(vault).resolve()
    except OSError as e:  # unresolvable path: treat as 'not the vault's repo', loudly
        print(f"WARNING: is_repo: cannot resolve {top!r} vs {vault}: {e}", file=sys.stderr)
        return False


def head_sha(vault: Path) -> str | None:
    r = _git(vault, "rev-parse", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _ensure_secret_gitignore(vault: Path) -> bool:
    """Guarantee secrets/trash/derived are ignored BEFORE staging anything.
    Returns True if .gitignore was modified. Matches line-wise (stripped), so a
    commented-out '#.obsidian/' or negated '!.obsidian/' never counts as present."""
    gi = vault / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    existing_lines = {ln.strip() for ln in existing.splitlines()}
    missing = [p for p in _REQUIRED_IGNORES if p not in existing_lines]
    if not missing:
        return False
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    gi.write_text(
        existing + sep
        + "# Added by vault-curation: never track secrets / trash / derived artifacts.\n"
        + "\n".join(missing) + "\n",
        encoding="utf-8",
    )
    return True


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


def commit_run(vault: Path, message: str, paths: list[str] | None = None) -> str | None:
    """Commit one curation run's changes. Returns the sha, or None if nothing changed.
    Local-only; never pushes.

    `paths`: the exact files the run touched. When given, ONLY those are staged —
    unrelated concurrent vault edits stay out, so `git revert <sha>` is surgical.
    None = stage the whole vault (explicit full-snapshot use only)."""
    vault = Path(vault)
    # Re-assert secret protection: the 'already a repo' init path never ran it.
    gi_changed = _ensure_secret_gitignore(vault)
    if paths is None:
        _git(vault, "add", "-A")
    else:
        specs: list[str] = []
        for rel in dict.fromkeys(paths):  # de-dupe, keep order
            # A pathspec matching nothing makes `git add` fail: keep a path only
            # if it exists on disk or is tracked (deletion to stage).
            if (vault / rel).exists() or _git(vault, "ls-files", "--", rel, check=False).stdout.strip():
                specs.append(rel)
            else:
                print(f"WARNING: commit_run: skipping unknown path: {rel}", file=sys.stderr)
        if gi_changed:
            specs.append(".gitignore")
        if not specs:
            return None  # nothing touched
        _git(vault, "add", "-A", "--", *specs)
    if _git(vault, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return None  # nothing staged
    _ensure_identity(vault)
    _git(vault, "commit", "-m", message)
    return head_sha(vault)
