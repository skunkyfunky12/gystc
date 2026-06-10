"""The write/apply layer — the ONLY part that mutates the vault. Every entry point
refuses without a git repo (reversibility), blocks path escapes, archives instead
of hard-deleting, never clobbers, and commits one revertable snapshot per run.
Callers must have shown the diff + obtained human approval first."""
from __future__ import annotations

import difflib
import hashlib
import shutil
import sys
from pathlib import Path

from brain_mcp.curation.vault_git import is_repo, commit_run

_ARCHIVE = "99 Archiv"


class StaleNoteError(RuntimeError):
    """The note changed on disk after the proposal was analyzed — applying would
    clobber a concurrent edit (optimistic-concurrency check)."""


def _require_repo(vault: Path) -> None:
    if not is_repo(Path(vault)):
        raise RuntimeError(
            "Vault is not a git repo — run ensure_vault_repo first. "
            "Reversibility is required before any write."
        )


def _check_base_hash(target: Path, rel: str, base_hash: str | None) -> None:
    """Verify the file still matches the content the proposal was based on."""
    if base_hash is None or not target.is_file():
        return
    current = hashlib.sha256(target.read_bytes()).hexdigest()
    if current != base_hash:
        raise StaleNoteError(
            f"{rel} changed since analyze "
            f"(now {current[:12]}…, proposal based on {base_hash[:12]}…)"
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


def apply_archive(vault: Path, rel: str, *, base_hash: str | None = None) -> Path:
    """Move a note OR a whole folder to '99 Archiv/' (preserving structure).
    Never hard-deletes, never overwrites an existing archived entry."""
    _require_repo(vault)
    # Re-archiving the archive (or anything inside) would nest '99 Archiv/99 Archiv'
    # or move the archive into its own subtree — refuse with a clear error.
    if Path(rel).parts and Path(rel).parts[0] == _ARCHIVE:
        raise ValueError(f"already under {_ARCHIVE}: {rel}")
    src = _safe_target(vault, rel)
    if not src.exists():
        raise FileNotFoundError(rel)
    _check_base_hash(src, rel, base_hash)
    dest = _safe_target(vault, f"{_ARCHIVE}/{rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    final, i = dest, 1
    while final.exists():
        final = dest.with_name(f"{dest.stem}.{i}{dest.suffix}")
        i += 1
    shutil.move(str(src), str(final))
    return final


def apply_edit(vault: Path, rel: str, new_content: str, *, base_hash: str | None = None) -> None:
    """Overwrite an existing note. Caller must have shown the diff + gotten OK.
    Pass `base_hash` (sha256 of the analyzed bytes) to refuse clobbering a note
    that changed in the analyze->apply window."""
    _require_repo(vault)
    target = _safe_target(vault, rel)
    if not target.is_file():
        raise FileNotFoundError(rel)
    _check_base_hash(target, rel, base_hash)
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
    actions: [{op:'archive'|'edit'|'create', file, new_content?, base_hash?}].

    Stale actions (file changed since analyze, per base_hash) are SKIPPED loudly
    and reported in the result — never silently clobbered. Replays of a
    partially-applied proposal (source already archived/edited away, create
    target already present) are likewise recorded skips with a distinct reason
    instead of crashing the rerun. On a mid-run failure the already-applied
    actions are committed as a clearly-marked PARTIAL snapshot before
    re-raising, so the vault never holds silent half-state."""
    _require_repo(vault)
    vault = Path(vault)
    applied = 0
    skipped: list[dict] = []
    touched: list[str] = []  # exact files this run mutated -> surgical commit
    try:
        for a in actions:
            op = a["op"]
            rel = Path(a["file"]).as_posix()
            try:
                if op == "archive":
                    final = apply_archive(vault, a["file"], base_hash=a.get("base_hash"))
                    touched += [rel, final.relative_to(vault.resolve()).as_posix()]
                elif op == "edit":
                    apply_edit(vault, a["file"], a["new_content"], base_hash=a.get("base_hash"))
                    touched.append(rel)
                elif op == "create":
                    apply_create(vault, a["file"], a["new_content"])
                    touched.append(rel)
                else:
                    raise ValueError(f"unknown op: {op}")
            except StaleNoteError as e:
                print(f"SKIPPED (stale): {e}", file=sys.stderr)
                skipped.append({"file": a.get("file"), "reason": str(e)})
                continue
            applied += 1
    except Exception:
        if touched:
            # Never leave silent uncommitted half-state: snapshot what was done,
            # clearly marked, so the next run's commit stays surgical/revertable.
            try:
                sha = commit_run(vault, f"PARTIAL (failed mid-run): {run_message}",
                                 paths=touched)
                print(f"ERROR: curation run failed after {applied} action(s); "
                      f"partial state committed as {(sha or '-')[:8]}", file=sys.stderr)
            except Exception as ce:
                print(f"ERROR: curation run failed AND the partial-state commit "
                      f"failed too: {ce}", file=sys.stderr)
        raise
    return {"applied": applied, "skipped": skipped,
            "commit": commit_run(vault, run_message, paths=touched)}
