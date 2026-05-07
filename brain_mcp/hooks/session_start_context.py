"""SessionStart hook: inject relevant vault context using FTS5 (no model required)."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def get_brain_db_path() -> Path | None:
    data_dir = os.environ.get("BRAIN_DATA_DIR")
    if data_dir:
        p = Path(data_dir)
    else:
        p = Path.home() / ".neural-brain"

    config_file = p / "config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
            if not config.get("auto_context", True):
                return None
        except (json.JSONDecodeError, OSError):
            pass

    db_path = p / "brain.db"
    return db_path if db_path.exists() else None


def get_git_context() -> dict | None:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if top.returncode != 0:
            return None
        repo = Path(top.stdout.strip()).name
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        branch_name = branch.stdout.strip() if branch.returncode == 0 else ""
        log = subprocess.run(
            ["git", "log", "--oneline", "-5", "--format=%s"],
            capture_output=True, text=True, timeout=5,
        )
        commits = log.stdout.strip().split("\n") if log.returncode == 0 and log.stdout.strip() else []
        return {"repo": repo, "branch": branch_name, "commits": commits}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def build_query(cwd: str, git_context: dict | None) -> str:
    if git_context:
        parts = [git_context["repo"], git_context["branch"]]
        parts.extend(git_context["commits"][:5])
        return " ".join(p for p in parts if p)
    return Path(cwd).name


def search_and_format(db_path: Path, query: str) -> str:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    words = query.split()[:10]
    safe_query = " ".join(f'"{w}"' for w in words if w)
    if not safe_query:
        conn.close()
        return ""
    try:
        results = conn.execute(
            """SELECT n.title, n.path, substr(n.content, 1, 150) AS snippet
               FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid
               WHERE notes_fts MATCH ? ORDER BY rank LIMIT 3""",
            (safe_query,),
        ).fetchall()
    except sqlite3.OperationalError:
        results = []
    conn.close()

    if not results:
        return ""

    lines = ["=== Brain Context ===", "Relevant vault notes for your current work:", ""]
    for i, r in enumerate(results, 1):
        snippet = (r["snippet"] or "").replace("\n", " ").strip()
        lines.append(f"{i}. **{r['title']}** ({r['path']}) -- {snippet}...")
    lines.append("")
    lines.append("Use brain_retrieve for deeper searches.")
    return "\n".join(lines)


def main() -> None:
    db_path = get_brain_db_path()
    if db_path is None:
        return
    cwd = os.getcwd()
    git_ctx = get_git_context()
    query = build_query(cwd, git_ctx)
    output = search_and_format(db_path, query)
    if output:
        print(output)


if __name__ == "__main__":
    main()
