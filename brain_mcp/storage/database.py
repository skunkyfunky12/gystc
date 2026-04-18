# brain_mcp/storage/database.py
from __future__ import annotations
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from brain_mcp.storage.migrations import run_migrations


class BrainDB:
    def __init__(self, db_path: Path | str):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        run_migrations(self._conn)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def close(self) -> None:
        self._conn.close()

    def upsert_note(
        self,
        path: str,
        title: str,
        content: str,
        content_hash: str,
        region_idx: int,
        tags: list[str],
        word_count: int,
        created_at: str,
        modified_at: str,
        faiss_idx: int | None = None,
    ) -> int:
        tags_json = json.dumps(tags)
        with self._lock:
            # Grab old FTS data before upsert so we can remove stale index entry
            old = self._conn.execute("SELECT id, title, content FROM notes WHERE path=?", (path,)).fetchone()
            self._conn.execute(
                """INSERT INTO notes (path, title, content, content_hash, region_idx, tags, word_count, created_at, modified_at, faiss_idx)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                     title=excluded.title, content=excluded.content, content_hash=excluded.content_hash,
                     region_idx=excluded.region_idx, tags=excluded.tags, word_count=excluded.word_count,
                     modified_at=excluded.modified_at, faiss_idx=COALESCE(excluded.faiss_idx, faiss_idx)
                """,
                (path, title, content, content_hash, region_idx, tags_json, word_count, created_at, modified_at, faiss_idx),
            )
            row = self._conn.execute("SELECT id FROM notes WHERE path=?", (path,)).fetchone()
            note_id = row["id"]
            # FTS sync for external-content FTS5:
            # Use special 'delete' command with OLD content, then INSERT new content
            if old is not None:
                self._conn.execute(
                    "INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)",
                    (old["id"], old["title"], old["content"]),
                )
            self._conn.execute("INSERT INTO notes_fts(rowid, title, content) VALUES (?, ?, ?)", (note_id, title, content))
            self._conn.commit()
        return note_id

    def set_faiss_idx(self, note_id: int, faiss_idx: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE notes SET faiss_idx=?, embedded_at=? WHERE id=?",
                               (faiss_idx, datetime.now(timezone.utc).isoformat(), note_id))
            self._conn.commit()

    def get_note_by_path(self, path: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes WHERE path=?", (path,)).fetchone()

    def get_note_by_title(self, title: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes WHERE title=?", (title,)).fetchone()

    def get_note_by_id(self, note_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()

    def get_all_notes(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes ORDER BY id").fetchall()

    def update_note_region(self, note_id: int, region_idx: int) -> None:
        if not (0 <= region_idx < 12):
            raise ValueError(f"region_idx must be 0-11, got {region_idx}")
        with self._lock:
            self._conn.execute("UPDATE notes SET region_idx = ? WHERE id = ?", (region_idx, note_id))
            self._conn.commit()

    def get_notes_by_faiss_indices(self, indices: list[int]) -> list[sqlite3.Row]:
        if not indices:
            return []
        placeholders = ",".join("?" for _ in indices)
        with self._lock:
            return self._conn.execute(
                f"SELECT * FROM notes WHERE faiss_idx IN ({placeholders})", tuple(indices)
            ).fetchall()

    def delete_note(self, path: str) -> None:
        with self._lock:
            row = self.get_note_by_path(path)
            if row:
                self._conn.execute("INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)",
                                   (row["id"], row["title"], row["content"]))
                self._conn.execute("DELETE FROM edges WHERE source_id=? OR target_id=?", (row["id"], row["id"]))
                self._conn.execute("DELETE FROM notes WHERE id=?", (row["id"],))
                self._conn.commit()

    def upsert_edge(self, source_id: int, target_id: int, link_text: str = "",
                    edge_type: str = "backlink", weight: float = 1.0,
                    confidence: float | None = None, source_file: str = "") -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO edges (source_id, target_id, link_text, edge_type, weight, confidence, source_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id) DO UPDATE SET
                     link_text=excluded.link_text, edge_type=excluded.edge_type,
                     weight=excluded.weight, confidence=excluded.confidence,
                     source_file=excluded.source_file""",
                (source_id, target_id, link_text, edge_type, weight, confidence, source_file),
            )
            self._conn.commit()

    def bulk_upsert_edges(self, edges: list[tuple]) -> int:
        """Bulk insert/update edges. Each tuple: (source_id, target_id, link_text, edge_type, weight, confidence, source_file)."""
        with self._lock:
            self._conn.executemany(
                """INSERT INTO edges (source_id, target_id, link_text, edge_type, weight, confidence, source_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id) DO UPDATE SET
                     link_text=excluded.link_text, edge_type=excluded.edge_type,
                     weight=excluded.weight, confidence=excluded.confidence,
                     source_file=excluded.source_file""",
                edges,
            )
            self._conn.commit()
            return len(edges)

    def delete_edges_by_type(self, edge_type: str) -> int:
        """Delete all edges of a specific type. Returns count deleted."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM edges WHERE edge_type = ?", (edge_type,))
            self._conn.commit()
            return cursor.rowcount

    def delete_edges_by_type_prefix(self, prefix: str) -> int:
        """Delete all edges whose type starts with prefix (e.g. 'graphify:')."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM edges WHERE edge_type LIKE ?", (prefix + "%",))
            self._conn.commit()
            return cursor.rowcount

    def get_edge_type_counts(self) -> dict[str, int]:
        """Count edges by type."""
        with self._lock:
            rows = self._conn.execute("SELECT edge_type, COUNT(*) as cnt FROM edges GROUP BY edge_type").fetchall()
            return {r["edge_type"]: r["cnt"] for r in rows}

    def get_edges_for_note(self, note_id: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM edges WHERE source_id=? OR target_id=?", (note_id, note_id)
            ).fetchall()

    def get_neighbor_ids(self, note_id: int, depth: int = 1) -> set[int]:
        with self._lock:
            visited: set[int] = set()
            frontier = {note_id}
            for _ in range(depth):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = self._conn.execute(
                    f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    tuple(frontier) + tuple(frontier),
                ).fetchall()
                next_frontier: set[int] = set()
                for r in rows:
                    next_frontier.add(r["source_id"])
                    next_frontier.add(r["target_id"])
                visited |= frontier
                frontier = next_frontier - visited
            visited |= frontier
            visited.discard(note_id)
            return visited

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Wrap each word in double quotes to escape FTS5 special operators."""
        words = query.split()
        if not words:
            return '""'
        return " ".join(f'"{w}"' for w in words)

    def fts_search(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        safe_query = self._sanitize_fts_query(query)
        with self._lock:
            try:
                return self._conn.execute(
                    """SELECT n.*, rank FROM notes_fts
                       JOIN notes n ON n.id = notes_fts.rowid
                       WHERE notes_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (safe_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

    def get_recent_notes(self, days: int = 7, region_idx: int | None = None, limit: int = 20) -> list[sqlite3.Row]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._lock:
            if region_idx is not None:
                return self._conn.execute(
                    "SELECT * FROM notes WHERE modified_at >= ? AND region_idx = ? ORDER BY modified_at DESC LIMIT ?",
                    (cutoff, region_idx, limit),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM notes WHERE modified_at >= ? ORDER BY modified_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()

    def get_region_note_counts(self) -> dict[int, int]:
        with self._lock:
            rows = self._conn.execute("SELECT region_idx, COUNT(*) as cnt FROM notes GROUP BY region_idx").fetchall()
            return {r["region_idx"]: r["cnt"] for r in rows}

    def get_all_regions(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM regions ORDER BY idx").fetchall()

    def get_region(self, idx: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM regions WHERE idx=?", (idx,)).fetchone()

    def update_region(self, idx: int, description: str | None = None, color: str | None = None) -> None:
        with self._lock:
            if description is not None:
                self._conn.execute("UPDATE regions SET description=? WHERE idx=?", (description, idx))
            if color is not None:
                self._conn.execute("UPDATE regions SET color=? WHERE idx=?", (color, idx))
            self._conn.commit()

    def get_content_hash(self, path: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT content_hash FROM notes WHERE path=?", (path,)).fetchone()
            return row["content_hash"] if row else None
