"""A minimal SQLite stand-in for the pinned Hermes ``hermes_state`` surface (v0.21.0).

The base suite has no Hermes, yet the companion's private archive operation is SQL against
Hermes's tables. This module gives it those tables with the exact shapes the pinned Hermes
creates (read from a fresh pinned ``state.db``), and the enumerated private names with the
pinned signatures, so ``archive_voice_rows`` runs for real in CI. Behaviour is the minimum
the operation relies on; the real-Hermes qualification proves the behaviour itself.

Registered as ``sys.modules["hermes_state"]`` by the tests that use it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
);
CREATE TABLE session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""


class SessionTurnLeaseLostError(RuntimeError):
    pass


class CompressionSessionClosedError(RuntimeError):
    pass


class SessionDB:
    _TRANSCRIPT_WRITE_PATIENCE_S = 60.0

    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _execute_write(self, fn: Any, patience_s: Any = None) -> Any:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(self._conn)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return result

    def _check_transcript_write_guards(
        self,
        conn: Any,
        session_id: Any,
        compression_lock_holder: Any,
        turn_lease_holder: Any = None,
        turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False,
        reject_active_compression_lock: bool = False,
    ) -> None:
        if turn_lease_holder:
            lease = conn.execute(
                "SELECT holder FROM session_turn_leases WHERE conversation_id = ?",
                (session_id,),
            ).fetchone()
            if lease is None or lease["holder"] != turn_lease_holder:
                raise SessionTurnLeaseLostError(session_id)
        session = conn.execute(
            "SELECT ended_at, end_reason FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is not None and session["ended_at"] is not None and (
            session["end_reason"] == "compression"
        ):
            raise CompressionSessionClosedError(session_id)

    def _insert_message_rows(self, conn: Any, session_id: Any, messages: Any) -> Any:
        for message in messages:
            metadata = message.get("display_metadata")
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, "
                "platform_message_id, observed, _compressed_summary, active, display_metadata) "
                "VALUES (?, ?, ?, ?, ?, 0, 0, 1, ?)",
                (
                    session_id,
                    message["role"],
                    message["content"],
                    float(message["timestamp"]),
                    message.get("platform_message_id"),
                    None if not metadata else json.dumps(metadata),
                ),
            )
        return len(messages), 0

    def create_session(self, session_id: Any, source: Any, **kwargs: Any) -> Any:
        self._execute_write(
            lambda conn: conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, parent_session_id, started_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, source, kwargs.get("parent_session_id"), time.time()),
            )
        )
        return session_id

    def try_acquire_session_turn_lease(
        self,
        session_id: Any,
        holder: Any,
        *,
        ttl_seconds: float = 300.0,
        patience_s: Any = None,
    ) -> bool:
        def acquire(conn: Any) -> bool:
            now = time.time()
            conn.execute(
                "DELETE FROM session_turn_leases WHERE conversation_id = ? AND expires_at <= ?",
                (session_id, now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO session_turn_leases VALUES (?, ?, ?, ?)",
                (session_id, holder, now, now + ttl_seconds),
            )
            owner = conn.execute(
                "SELECT holder FROM session_turn_leases WHERE conversation_id = ?",
                (session_id,),
            ).fetchone()
            return bool(owner["holder"] == holder)

        return bool(self._execute_write(acquire))

    def refresh_session_turn_lease(
        self, session_id: Any, holder: Any, *, ttl_seconds: float = 300.0
    ) -> bool:
        return bool(
            self._execute_write(
                lambda conn: conn.execute(
                    "UPDATE session_turn_leases SET expires_at = ? "
                    "WHERE conversation_id = ? AND holder = ?",
                    (time.time() + ttl_seconds, session_id, holder),
                ).rowcount
            )
        )

    def release_session_turn_lease(self, session_id: Any, holder: Any) -> None:
        self._execute_write(
            lambda conn: conn.execute(
                "DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                (session_id, holder),
            )
        )
