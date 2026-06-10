# brain_mcp/storage/database.py
from __future__ import annotations
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from brain_mcp.storage.migrations import run_migrations

# Copies a pruned note's version history into the tombstone table BEFORE the
# ON DELETE CASCADE destroys it (archive via curation must never wipe
# brain_versions history). Dedupes on (path, content_hash) so repeated
# prune/re-add cycles don't accumulate duplicates.
_TOMBSTONE_VERSIONS_SQL = """
    INSERT INTO archived_note_versions
        (path, note_id, content_hash, content, title, region_idx,
         tags, word_count, versioned_at, reason, archived_at)
    SELECT ?, nv.note_id, nv.content_hash, nv.content, nv.title,
           nv.region_idx, nv.tags, nv.word_count, nv.versioned_at,
           nv.reason, ?
    FROM note_versions nv
    WHERE nv.note_id = ?
      AND NOT EXISTS (
        SELECT 1 FROM archived_note_versions a
        WHERE a.path = ? AND a.content_hash = nv.content_hash)
"""


class BrainDB:
    def __init__(self, db_path: Path | str):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA wal_autocheckpoint=1000")
        # WAL + synchronous=NORMAL: commits no longer fsync the WAL each time
        # (the default FULL made every per-row commit an fsync -> multi-second
        # startup storms). Still crash-safe; at worst the last commit is lost
        # on POWER loss, and brain.db is derived from the vault on disk.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.RLock()
        self._closed = False
        run_migrations(self._conn)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
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
            old = self._conn.execute(
                "SELECT id, title, content, content_hash, region_idx, tags, word_count FROM notes WHERE path=?",
                (path,),
            ).fetchone()
            if old is not None and old["content_hash"] != content_hash:
                self.save_version(
                    old["id"], old["content"], old["title"], old["region_idx"],
                    old["tags"], old["word_count"], old["content_hash"],
                )
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
            if old is None:
                # A note re-appearing at a previously pruned path (e.g. moved
                # back out of the archive) gets its preserved history back --
                # see delete_notes_not_in / delete_note (tombstone table).
                restored = self._conn.execute(
                    """INSERT INTO note_versions
                           (note_id, content_hash, content, title, region_idx,
                            tags, word_count, versioned_at, reason)
                       SELECT ?, content_hash, content, title, region_idx,
                              tags, word_count, versioned_at, reason
                       FROM archived_note_versions WHERE path=?""",
                    (note_id, path),
                )
                if restored.rowcount > 0:
                    self._conn.execute(
                        "DELETE FROM archived_note_versions WHERE path=?", (path,)
                    )
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

    def save_version(self, note_id: int, content: str, title: str, region_idx: int,
                     tags: str, word_count: int, content_hash: str,
                     reason: str = "") -> int:
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM note_versions WHERE note_id=? AND content_hash=?",
                (note_id, content_hash),
            ).fetchone()
            if existing:
                return existing["id"]
            now = datetime.now(timezone.utc).isoformat()
            cursor = self._conn.execute(
                """INSERT INTO note_versions
                   (note_id, content_hash, content, title, region_idx, tags, word_count, versioned_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (note_id, content_hash, content, title, region_idx, tags, word_count, now, reason),
            )
            self._conn.commit()
            return cursor.lastrowid

    def get_versions(self, note_id: int, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                """SELECT id, content_hash, title, word_count, versioned_at, reason
                   FROM note_versions WHERE note_id=?
                   ORDER BY versioned_at DESC LIMIT ?""",
                (note_id, limit),
            ).fetchall()

    def get_version(self, version_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM note_versions WHERE id=?", (version_id,)
            ).fetchone()

    def prune_versions(self, note_id: int, max_versions: int = 50) -> int:
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM note_versions WHERE note_id=?", (note_id,)
            ).fetchone()[0]
            if count <= max_versions:
                return 0
            to_delete = count - max_versions
            self._conn.execute(
                """DELETE FROM note_versions WHERE id IN (
                    SELECT id FROM note_versions WHERE note_id=?
                    ORDER BY versioned_at ASC LIMIT ?
                )""",
                (note_id, to_delete),
            )
            self._conn.commit()
            return to_delete

    def set_faiss_idx(self, note_id: int, faiss_idx: int) -> int | None:
        """Stamp *faiss_idx* on a note and return the DISPLACED previous stamp
        (or ``None`` if there was none / it is unchanged).

        Read-modify-write under the DB lock: when a watcher re-index and a
        reconcile embedding pass embed the same note concurrently, the losing
        stamp's vector would otherwise stay in FAISS forever (orphan-vector
        leak -- only the id currently in the row is ever passed to remove()).
        Callers must ``vectors.remove()`` a non-None displaced id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT faiss_idx FROM notes WHERE id=?", (note_id,)
            ).fetchone()
            previous = row["faiss_idx"] if row is not None else None
            self._conn.execute("UPDATE notes SET faiss_idx=?, embedded_at=? WHERE id=?",
                               (faiss_idx, datetime.now(timezone.utc).isoformat(), note_id))
            self._conn.commit()
        return previous if previous != faiss_idx else None

    def set_faiss_indices(self, items: list[tuple[int, int]]) -> list[int]:
        """Batch variant of :meth:`set_faiss_idx` (``(note_id, faiss_idx)``):
        one transaction instead of one commit (= WAL write) per note.

        Returns the displaced previous stamps; callers must ``vectors.remove()``
        them or they leak as orphan vectors when a watcher re-index raced this
        batch on the same note (see :meth:`set_faiss_idx`)."""
        if not items:
            return []
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            note_ids = [note_id for note_id, _ in items]
            placeholders = ",".join("?" for _ in note_ids)
            previous = {
                r["id"]: r["faiss_idx"]
                for r in self._conn.execute(
                    f"SELECT id, faiss_idx FROM notes WHERE id IN ({placeholders})",
                    tuple(note_ids),
                ).fetchall()
            }
            self._conn.executemany(
                "UPDATE notes SET faiss_idx=?, embedded_at=? WHERE id=?",
                [(fid, now, note_id) for note_id, fid in items],
            )
            self._conn.commit()
        return [
            prev for note_id, fid in items
            if (prev := previous.get(note_id)) is not None and prev != fid
        ]

    def clear_all_faiss_idx(self) -> None:
        """NULL every stored faiss_idx (notes + chunks) so the pipeline's
        "re-embed faiss_idx IS NULL" reconcile rebuilds them from scratch.

        Needed after index.faiss corruption/loss: rows degrade to "unindexed"
        (recoverable) instead of keeping stale ids that collide with freshly
        assigned ones (wrongly-mapped search results). Callers: the CLI
        (`brain_mcp index` / `config init`) and the server writer paths
        (_start_indexing); the dashboard reindex can reuse it the same way."""
        with self._lock:
            self._conn.execute("UPDATE notes SET faiss_idx=NULL, embedded_at=NULL")
            self._conn.execute("UPDATE chunks SET faiss_idx=NULL")
            self._conn.commit()

    def has_faiss_stamps(self) -> bool:
        """True if any note or chunk row still carries a faiss_idx stamp."""
        with self._lock:
            if self._conn.execute(
                "SELECT 1 FROM notes WHERE faiss_idx IS NOT NULL LIMIT 1"
            ).fetchone() is not None:
                return True
            return self._conn.execute(
                "SELECT 1 FROM chunks WHERE faiss_idx IS NOT NULL LIMIT 1"
            ).fetchone() is not None

    def get_note_by_path(self, path: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes WHERE path=?", (path,)).fetchone()

    def get_note_by_title(self, title: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes WHERE title=?", (title,)).fetchone()

    def get_note_by_id(self, note_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()

    def get_note_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def get_all_notes(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes ORDER BY id").fetchall()

    def update_note_region(self, note_id: int, region_idx: int) -> None:
        if not (0 <= region_idx < 12):
            raise ValueError(f"region_idx must be 0-11, got {region_idx}")
        with self._lock:
            self._conn.execute("UPDATE notes SET region_idx = ? WHERE id = ?", (region_idx, note_id))
            self._conn.commit()

    def get_notes_by_ids(self, note_ids: list[int]) -> list[sqlite3.Row]:
        if not note_ids:
            return []
        placeholders = ",".join("?" for _ in note_ids)
        with self._lock:
            return self._conn.execute(
                f"SELECT * FROM notes WHERE id IN ({placeholders})", tuple(note_ids)
            ).fetchall()

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
                # Preserve version history before the CASCADE destroys it (the
                # watcher fires delete+create on every rename/move).
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(_TOMBSTONE_VERSIONS_SQL, (path, now, row["id"], path))
                self._conn.execute("INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)",
                                   (row["id"], row["title"], row["content"]))
                self._conn.execute("DELETE FROM edges WHERE source_id=? OR target_id=?", (row["id"], row["id"]))
                self._conn.execute("DELETE FROM notes WHERE id=?", (row["id"],))
                self._conn.commit()

    def get_all_note_paths(self) -> list[str]:
        with self._lock:
            return [r["path"] for r in self._conn.execute("SELECT path FROM notes").fetchall()]

    def delete_notes_not_in(self, keep_paths: set[str]) -> list[int]:
        """Delete every note whose path is not in *keep_paths* and return the
        FAISS ids (note-level + chunk-level) the caller must drop from the
        vector store.

        Used by a (re)build to prune notes that vanished from the scan -- e.g.
        deleted on disk or moved into a newly-excluded directory.  Without this,
        a force-rebuild leaves orphan rows whose stale ``faiss_idx`` collide
        with freshly assigned ids -> corrupt search results.  Chunks and edges
        are removed via ``ON DELETE CASCADE``; the external-content FTS mirror
        is kept in sync explicitly (FTS5 does not auto-delete).  Version
        history is NOT destroyed: it is copied into ``archived_note_versions``
        (keyed by path) before the cascade fires and re-attached by
        ``upsert_note`` when a note returns at the same path (un-archive).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path, title, content, faiss_idx FROM notes"
            ).fetchall()
            stale = [r for r in rows if r["path"] not in keep_paths]
            if not stale:
                return []
            now = datetime.now(timezone.utc).isoformat()
            preserved = 0
            faiss_ids: list[int] = []
            for r in stale:
                if r["faiss_idx"] is not None:
                    faiss_ids.append(r["faiss_idx"])
                chunk_rows = self._conn.execute(
                    "SELECT faiss_idx FROM chunks WHERE note_id=? AND faiss_idx IS NOT NULL",
                    (r["id"],),
                ).fetchall()
                faiss_ids.extend(c["faiss_idx"] for c in chunk_rows)
                cur = self._conn.execute(
                    _TOMBSTONE_VERSIONS_SQL, (r["path"], now, r["id"], r["path"])
                )
                if cur.rowcount > 0:
                    preserved += 1
                self._conn.execute(
                    "INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)",
                    (r["id"], r["title"], r["content"]),
                )
            ids = [r["id"] for r in stale]
            placeholders = ",".join("?" for _ in ids)
            self._conn.execute(
                f"DELETE FROM notes WHERE id IN ({placeholders})", tuple(ids)
            )
            self._conn.commit()
            if preserved:
                print(
                    f"reconcile: pruned {len(stale)} notes; version history of "
                    f"{preserved} preserved in archived_note_versions",
                    file=sys.stderr,
                )
            return faiss_ids

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

    def upsert_edges(self, edges: list[tuple[int, int, str]]) -> None:
        """Batch upsert of backlink edges ``(source_id, target_id, link_text)``.

        One transaction/commit for the whole batch -- index_vault used to call
        :meth:`upsert_edge` per backlink, i.e. one WAL commit PER EDGE on every
        writer startup (a multi-second fsync storm on large vaults).
        """
        if not edges:
            return
        with self._lock:
            self._conn.executemany(
                """INSERT INTO edges (source_id, target_id, link_text, edge_type, weight, confidence, source_file)
                   VALUES (?, ?, ?, 'backlink', 1.0, NULL, '')
                   ON CONFLICT(source_id, target_id) DO UPDATE SET
                     link_text=excluded.link_text, edge_type=excluded.edge_type,
                     weight=excluded.weight, confidence=excluded.confidence,
                     source_file=excluded.source_file""",
                edges,
            )
            self._conn.commit()


    def replace_edges_by_type_prefix(self, prefix: str, edges: list[tuple]) -> tuple[int, int]:
        """Atomically delete all edges with prefix and insert new ones in a single transaction.
        Returns (deleted_count, inserted_count)."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM edges WHERE edge_type LIKE ?", (prefix + "%",))
            deleted = cursor.rowcount
            if edges:
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
            return deleted, len(edges)

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
        return " ".join(f'"{w.replace(chr(34), "")}"' for w in words)

    def fts_search(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        safe_query = self._sanitize_fts_query(query)
        for attempt in (1, 2):
            with self._lock:
                try:
                    return self._conn.execute(
                        """SELECT n.*, rank FROM notes_fts
                           JOIN notes n ON n.id = notes_fts.rowid
                           WHERE notes_fts MATCH ?
                           ORDER BY rank LIMIT ?""",
                        (safe_query, limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    # NEVER swallow silently: 'database is locked' or a
                    # missing/corrupt notes_fts table would otherwise read as
                    # "the brain has no memory of this" (FTS is the only
                    # search path while the model loads).
                    if not (attempt == 1 and "locked" in str(exc).lower()):
                        print(f"fts_search failed (query={query!r}): {exc}", file=sys.stderr)
                        return []
                    # fall through: retry once after backing off below
            # Back off OUTSIDE the RLock: the contention is cross-process
            # (in-process callers share this lock and busy_timeout already
            # waited), so sleeping while holding it would stall every other
            # DB call -- including the 'instant' brain_status/brain_recent.
            time.sleep(0.1)
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

    # ------------------------------------------------------------------
    # Chunk methods
    # ------------------------------------------------------------------

    def replace_chunks(self, note_id: int, chunks: list[dict]) -> list[int]:
        """Delete old chunks, insert new ones. Returns old faiss_idx values to remove."""
        with self._lock:
            old_faiss = [
                r["faiss_idx"] for r in
                self._conn.execute(
                    "SELECT faiss_idx FROM chunks WHERE note_id=? AND faiss_idx IS NOT NULL",
                    (note_id,),
                ).fetchall()
            ]
            self._conn.execute("DELETE FROM chunks WHERE note_id=?", (note_id,))
            for chunk in chunks:
                self._conn.execute(
                    """INSERT INTO chunks (note_id, heading, content, content_hash, word_count, chunk_idx)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (note_id, chunk["heading"], chunk["content"], chunk["content_hash"],
                     chunk["word_count"], chunk["chunk_idx"]),
                )
            self._conn.commit()
            return old_faiss

    def set_chunk_faiss_idx(self, note_id: int, chunk_idx: int, faiss_idx: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET faiss_idx=? WHERE note_id=? AND chunk_idx=?",
                (faiss_idx, note_id, chunk_idx),
            )
            self._conn.commit()

    def set_chunk_faiss_indices(self, items: list[tuple[int, int, int]]) -> None:
        """Batch variant of :meth:`set_chunk_faiss_idx`
        (``(note_id, chunk_idx, faiss_idx)``): one commit per batch, not per chunk."""
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE chunks SET faiss_idx=? WHERE note_id=? AND chunk_idx=?",
                [(fid, note_id, chunk_idx) for note_id, chunk_idx, fid in items],
            )
            self._conn.commit()

    def get_chunks_for_note(self, note_id: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM chunks WHERE note_id=? ORDER BY chunk_idx", (note_id,)
            ).fetchall()

    def get_chunks_by_faiss_indices(self, indices: list[int]) -> list[sqlite3.Row]:
        if not indices:
            return []
        placeholders = ",".join("?" for _ in indices)
        with self._lock:
            return self._conn.execute(
                f"""SELECT c.*, n.title AS note_title, n.path AS note_path,
                           n.region_idx AS note_region_idx
                    FROM chunks c JOIN notes n ON c.note_id = n.id
                    WHERE c.faiss_idx IN ({placeholders})""",
                tuple(indices),
            ).fetchall()

    def delete_chunks_for_note(self, note_id: int) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT faiss_idx FROM chunks WHERE note_id=?", (note_id,)
            ).fetchall()
            if not rows:
                # Nothing to delete -> no commit. The reconcile calls this for
                # every chunkless note on every startup; an unconditional
                # commit here was one WAL write per note.
                return []
            old_faiss = [r["faiss_idx"] for r in rows if r["faiss_idx"] is not None]
            self._conn.execute("DELETE FROM chunks WHERE note_id=?", (note_id,))
            self._conn.commit()
            return old_faiss
