"""SQLite storage for indexed conversations and immutable source snapshots."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    files_seen INTEGER NOT NULL DEFAULT 0,
    files_indexed INTEGER NOT NULL DEFAULT 0,
    files_skipped INTEGER NOT NULL DEFAULT 0,
    conversations_indexed INTEGER NOT NULL DEFAULT 0,
    messages_indexed INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    configured_path TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_scan_at TEXT,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    original_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    archive_path TEXT,
    parse_status TEXT NOT NULL,
    error TEXT,
    conversation_id TEXT,
    indexed_at TEXT NOT NULL,
    UNIQUE(source_id, original_path),
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_source_files_fingerprint
    ON source_files(source_id, original_path, size_bytes, mtime_ns);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    source_file_id INTEGER,
    source_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    original_path TEXT NOT NULL,
    archive_path TEXT,
    account_id TEXT,
    source_label TEXT,
    originator TEXT,
    model_provider TEXT,
    model TEXT,
    reasoning_effort TEXT,
    cwd TEXT,
    git_branch TEXT,
    title TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    user_message_count INTEGER NOT NULL DEFAULT 0,
    assistant_message_count INTEGER NOT NULL DEFAULT 0,
    preview TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT,
    FOREIGN KEY(source_file_id) REFERENCES source_files(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_account ON conversations(account_id);
CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source_id);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    event_type TEXT,
    created_at TEXT,
    raw_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(conversation_id, ordinal),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, ordinal);
"""


class Repository:
    """Thread-safe repository used by the scanner and the local HTTP server."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            str(db_path), check_same_thread=False, timeout=30
        )
        self.connection.row_factory = sqlite3.Row
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(SCHEMA)
            self.fts_enabled = self._ensure_fts()
            self.connection.commit()

    def _ensure_fts(self) -> bool:
        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(
                    conversation_id UNINDEXED,
                    message_id UNINDEXED,
                    title,
                    cwd,
                    text,
                    tokenize='unicode61'
                )"""
            )
            return True
        except sqlite3.OperationalError:
            return False

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def upsert_source(self, source_id: str, kind: str, path: str, enabled: bool) -> None:
        with self._lock:
            self.connection.execute(
                """INSERT INTO sources(source_id, kind, configured_path, enabled)
                   VALUES(?, ?, ?, ?)
                   ON CONFLICT(source_id) DO UPDATE SET
                     kind=excluded.kind,
                     configured_path=excluded.configured_path,
                     enabled=excluded.enabled""",
                (source_id, kind, path, int(enabled)),
            )
            self.connection.commit()

    def mark_source_scan(self, source_id: str, scanned_at: str, error: str | None) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE sources SET last_scan_at=?, last_error=? WHERE source_id=?",
                (scanned_at, error, source_id),
            )
            self.connection.commit()

    def existing_file(self, source_id: str, original_path: str) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(
                "SELECT * FROM source_files WHERE source_id=? AND original_path=?",
                (source_id, original_path),
            ).fetchone()

    def upsert_source_file(self, values: dict[str, Any]) -> int:
        columns = [
            "source_id",
            "original_path",
            "size_bytes",
            "mtime_ns",
            "sha256",
            "archive_path",
            "parse_status",
            "error",
            "conversation_id",
            "indexed_at",
        ]
        params = [values.get(column) for column in columns]
        with self._lock:
            self.connection.execute(
                f"""INSERT INTO source_files({','.join(columns)})
                    VALUES({','.join('?' for _ in columns)})
                    ON CONFLICT(source_id, original_path) DO UPDATE SET
                      size_bytes=excluded.size_bytes,
                      mtime_ns=excluded.mtime_ns,
                      sha256=excluded.sha256,
                      archive_path=excluded.archive_path,
                      parse_status=excluded.parse_status,
                      error=excluded.error,
                      conversation_id=excluded.conversation_id,
                      indexed_at=excluded.indexed_at""",
                params,
            )
            row = self.connection.execute(
                "SELECT id FROM source_files WHERE source_id=? AND original_path=?",
                (values["source_id"], values["original_path"]),
            ).fetchone()
            self.connection.commit()
            return int(row[0])

    def replace_conversation(
        self,
        conversation: dict[str, Any],
        messages: Iterable[dict[str, Any]],
    ) -> None:
        message_rows = list(messages)
        with self._lock:
            self.connection.execute(
                "DELETE FROM messages WHERE conversation_id=?",
                (conversation["id"],),
            )
            self.connection.execute(
                "DELETE FROM message_search WHERE conversation_id=?",
                (conversation["id"],),
            ) if self.fts_enabled else None
            columns = [
                "id",
                "source_file_id",
                "source_id",
                "source_kind",
                "original_path",
                "archive_path",
                "account_id",
                "source_label",
                "originator",
                "model_provider",
                "model",
                "reasoning_effort",
                "cwd",
                "git_branch",
                "title",
                "created_at",
                "updated_at",
                "message_count",
                "user_message_count",
                "assistant_message_count",
                "preview",
                "metadata_json",
                "content_hash",
            ]
            self.connection.execute(
                f"""INSERT INTO conversations({','.join(columns)})
                    VALUES({','.join('?' for _ in columns)})
                    ON CONFLICT(id) DO UPDATE SET
                      source_file_id=excluded.source_file_id,
                      source_id=excluded.source_id,
                      source_kind=excluded.source_kind,
                      original_path=excluded.original_path,
                      archive_path=excluded.archive_path,
                      account_id=excluded.account_id,
                      source_label=excluded.source_label,
                      originator=excluded.originator,
                      model_provider=excluded.model_provider,
                      model=excluded.model,
                      reasoning_effort=excluded.reasoning_effort,
                      cwd=excluded.cwd,
                      git_branch=excluded.git_branch,
                      title=excluded.title,
                      created_at=excluded.created_at,
                      updated_at=excluded.updated_at,
                      message_count=excluded.message_count,
                      user_message_count=excluded.user_message_count,
                      assistant_message_count=excluded.assistant_message_count,
                      preview=excluded.preview,
                      metadata_json=excluded.metadata_json,
                      content_hash=excluded.content_hash""",
                [conversation.get(column) for column in columns],
            )
            self.connection.executemany(
                """INSERT INTO messages(
                    conversation_id, ordinal, role, text, event_type, created_at, raw_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        conversation["id"],
                        row["ordinal"],
                        row["role"],
                        row["text"],
                        row.get("event_type"),
                        row.get("created_at"),
                        json.dumps(row.get("raw", {}), ensure_ascii=False),
                    )
                    for row in message_rows
                ],
            )
            if self.fts_enabled:
                self.connection.executemany(
                    "INSERT INTO message_search(conversation_id, message_id, title, cwd, text) VALUES(?, ?, ?, ?, ?)",
                    [
                        (
                            conversation["id"],
                            row["ordinal"],
                            conversation["title"],
                            conversation.get("cwd") or "",
                            row["text"],
                        )
                        for row in message_rows
                    ],
                )
            self.connection.commit()

    def start_scan(self, started_at: str) -> int:
        with self._lock:
            cursor = self.connection.execute(
                "INSERT INTO scan_runs(started_at, status) VALUES(?, 'running')",
                (started_at,),
            )
            self.connection.commit()
            return int(cursor.lastrowid)

    def finish_scan(self, run_id: int, values: dict[str, Any], finished_at: str, status: str) -> None:
        with self._lock:
            self.connection.execute(
                """UPDATE scan_runs SET finished_at=?, status=?, files_seen=?, files_indexed=?,
                   files_skipped=?, conversations_indexed=?, messages_indexed=?, errors_json=?
                   WHERE id=?""",
                (
                    finished_at,
                    status,
                    values.get("files_seen", 0),
                    values.get("files_indexed", 0),
                    values.get("files_skipped", 0),
                    values.get("conversations_indexed", 0),
                    values.get("messages_indexed", 0),
                    json.dumps(values.get("errors", []), ensure_ascii=False),
                    run_id,
                ),
            )
            self.connection.commit()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                """SELECT
                    (SELECT COUNT(*) FROM conversations) AS conversations,
                    (SELECT COUNT(*) FROM messages) AS messages,
                    (SELECT COUNT(DISTINCT account_id) FROM conversations WHERE account_id IS NOT NULL AND account_id <> '') AS accounts,
                    (SELECT COUNT(DISTINCT source_id) FROM conversations) AS sources,
                    (SELECT MAX(finished_at) FROM scan_runs WHERE status IN ('ok','partial')) AS last_scan
                """
            ).fetchone()
            latest = self.connection.execute(
                "SELECT * FROM scan_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            result = dict(row)
            result["latest_scan"] = dict(latest) if latest else None
            return result

    def filters(self) -> dict[str, list[str]]:
        with self._lock:
            accounts = self.connection.execute(
                "SELECT DISTINCT account_id FROM conversations WHERE account_id IS NOT NULL AND account_id <> '' ORDER BY account_id"
            ).fetchall()
            sources = self.connection.execute(
                "SELECT DISTINCT source_label FROM conversations WHERE source_label IS NOT NULL AND source_label <> '' ORDER BY source_label"
            ).fetchall()
            projects = self.connection.execute(
                "SELECT DISTINCT cwd FROM conversations WHERE cwd IS NOT NULL AND cwd <> '' ORDER BY cwd"
            ).fetchall()
            return {
                "accounts": [row[0] for row in accounts],
                "sources": [row[0] for row in sources],
                "projects": [row[0] for row in projects],
            }

    def list_conversations(
        self,
        query: str = "",
        account_id: str = "",
        source_label: str = "",
        cwd: str = "",
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = []
        if account_id:
            conditions.append("c.account_id=?")
            params.append(account_id)
        if source_label:
            conditions.append("c.source_label=?")
            params.append(source_label)
        if cwd:
            conditions.append("c.cwd=?")
            params.append(cwd)
        if query.strip():
            like = f"%{query.strip()}%"
            conditions.append(
                "(c.title LIKE ? OR c.preview LIKE ? OR c.cwd LIKE ? OR EXISTS "
                "(SELECT 1 FROM messages m WHERE m.conversation_id=c.id AND m.text LIKE ?))"
            )
            params.extend([like, like, like, like])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([max(1, min(limit, 200)), max(0, offset)])
        with self._lock:
            rows = self.connection.execute(
                f"""SELECT c.* FROM conversations c {where}
                    ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.id DESC
                    LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            conversation = self.connection.execute(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            if not conversation:
                return None
            messages = self.connection.execute(
                "SELECT ordinal, role, text, event_type, created_at FROM messages WHERE conversation_id=? ORDER BY ordinal",
                (conversation_id,),
            ).fetchall()
            result = dict(conversation)
            result["messages"] = [dict(row) for row in messages]
            return result

    def latest_runs(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM scan_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
