"""The write/apply layer — the ONLY part that mutates the vault. Every entry point
refuses without a git repo (reversibility), blocks path escapes, archives instead
of hard-deleting, never clobbers, and commits one revertable snapshot per run.
Callers must have shown the diff + obtained human approval first."""
from __future__ import annotations

import difflib
import shutil
from pathlib import Path

from brain_mcp.curation.vault_git import is_repo, commit_run

_ARCHIVE = "99 Archiv"


def _require_repo(vault: Path) -> None:
    if not is_repo(Path(vault)):
        raise RuntimeError(
            "Vault is not a git repo — run ensure_vault_repo first. "
            "Reversibility is required before any write."
        )


def _safe_target(vault: Path, rel: str) -> Path:
    vault = Path(vault)
    target = (vault / rel).resolve()
    if not target.is_relative_to(vault.resolve()):
        raise ValueError(f"path escapes vault: {rel}")
    return target


def _atomic_write(target: Path, content: str) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)


def unified_diff(old: str, new: str, path: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def apply_archive(vault: Path, rel: str) -> Path:
    """Move a note OR a whole folder to '99 Archiv/' (preserving structure).
    Never hard-deletes, never overwrites an existing archived entry."""
    _require_repo(vault)
    src = _safe_target(vault, rel)
    if not src.exists():
        raise FileNotFoundError(rel)
    dest = _safe_target(vault, f"{_ARCHIVE}/{rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    final, i = dest, 1
    while final.exists():
        final = dest.with_name(f"{dest.stem}.{i}{dest.suffix}")
        i += 1
    shutil.move(str(src), str(final))
    return final


def apply_edit(vault: Path, rel: str, new_content: str) -> None:
    """Overwrite an existing note. Caller must have shown the diff + gotten OK."""
    _require_repo(vault)
    target = _safe_target(vault, rel)
    if not target.is_file():
        raise FileNotFoundError(rel)
    _atomic_write(target, new_content)


def apply_create(vault: Path, rel: str, content: str) -> None:
    """Create a NEW note (e.g. a reconcile promotion). Never clobbers an existing file."""
    _require_repo(vault)
    target = _safe_target(vault, rel)
    if target.exists():
        raise FileExistsError(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, content)


def apply_actions(vault: Path, actions: list[dict], *, run_message: str) -> dict:
    """Apply a list of CONFIRMED actions, then commit one revertable snapshot.
    actions: [{op:'archive'|'edit'|'create', file, new_content?}]."""
    _require_repo(vault)
    applied = 0
    for a in actions:
        op = a["op"]
        if op == "archive":
            apply_archive(vault, a["file"])
        elif op == "edit":
            apply_edit(vault, a["file"], a["new_content"])
        elif op == "create":
            apply_create(vault, a["file"], a["new_content"])
        else:
            raise ValueError(f"unknown op: {op}")
        applied += 1
    return {"applied": applied, "commit": commit_run(vault, run_message)}
