"""The companion's own SQLite file: one row of fences and progress per voice conversation.

``state.db`` is the only source of truth for what an archive holds. This file holds only
what must survive the companion itself: which session a conversation is bound to, the
fingerprint it last committed (and the one it intends next), quarantine and tombstone
fences, the lease holder, and the review ledger. It never holds transcript text.

Every step is one ``BEGIN IMMEDIATE`` transaction that re-reads the fences it depends on,
and every step is written before the ``state.db`` action it guards (write-ahead), so no
transaction ever spans the two files.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_realtime.companion.integrity import (
    ArchiveRefusal,
    Fingerprint,
    Identity,
    validate_conversation_id,
)

_SCHEMA_VERSION = 1
_CONCRETE_PATH = type(Path())
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_MAX_HOLDER_CHARS = 256
# Categories a durable quarantine may record: the archive no longer matches its evidence.
QUARANTINE_CATEGORIES = frozenset(
    {"mismatch", "missing", "over_cap", "rotated", "recovery", "lineage"}
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_archive (
    conversation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    committed_count INTEGER,
    committed_chain TEXT,
    committed_generation INTEGER,
    committed_seq INTEGER,
    pending_count INTEGER,
    pending_chain TEXT,
    pending_generation INTEGER,
    pending_seq INTEGER,
    quarantine TEXT,
    tombstone INTEGER,
    holder TEXT,
    review_ledger TEXT NOT NULL DEFAULT '{}',
    CHECK ((committed_count IS NULL) = (committed_chain IS NULL)),
    CHECK ((pending_count IS NULL) = (pending_chain IS NULL)),
    CHECK ((committed_generation IS NULL) = (committed_seq IS NULL)),
    CHECK ((pending_generation IS NULL) = (pending_seq IS NULL)),
    CHECK (committed_count IS NOT NULL OR pending_count IS NOT NULL)
) STRICT
"""
_COLUMNS = (
    "conversation_id, session_id, committed_count, committed_chain, committed_generation, "
    "committed_seq, pending_count, pending_chain, pending_generation, pending_seq, "
    "quarantine, tombstone, holder"
)


@dataclass(frozen=True, slots=True)
class Progress:
    """An archive state: its fingerprint, and the last identity it covers (None if empty)."""

    fingerprint: Fingerprint
    cursor: Identity | None

    def __post_init__(self) -> None:
        if type(self.fingerprint) is not Fingerprint:
            raise TypeError("progress fingerprint must be an exact Fingerprint")
        if self.cursor is not None and type(self.cursor) is not Identity:
            raise TypeError("progress cursor must be an exact Identity or None")


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    session_id: str
    committed: Progress | None
    pending: Progress | None
    quarantine: str | None
    tombstone: int | None
    holder: str | None


def _progress(count: Any, chain: Any, generation: Any, seq: Any) -> Progress | None:
    if count is None:
        return None
    cursor = None if generation is None else Identity(generation, seq)
    return Progress(Fingerprint(count, chain), cursor)


def _columns(progress: Progress) -> tuple[int, str, int | None, int | None]:
    if type(progress) is not Progress:
        raise TypeError("progress must be an exact Progress")
    cursor = progress.cursor
    return (
        progress.fingerprint.count,
        progress.fingerprint.chain,
        None if cursor is None else cursor.generation,
        None if cursor is None else cursor.seq,
    )


class CompanionStore:
    """The plugin-store file. Single-threaded: call it from one thread only."""

    def __init__(self, path: Path) -> None:
        if type(path) is not _CONCRETE_PATH:
            raise TypeError("companion store path must be an exact pathlib Path")
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        try:
            self._connection.execute("PRAGMA journal_mode = DELETE")
            self._connection.execute("PRAGMA synchronous = FULL")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, _SCHEMA_VERSION):
                raise RuntimeError("companion store has an unsupported schema version")
            self._connection.execute(_SCHEMA)
            self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def pragma(self, name: str) -> object:
        if name not in ("synchronous", "journal_mode"):
            raise ValueError("only durability pragmas are readable")
        return self._connection.execute(f"PRAGMA {name}").fetchone()[0]

    def _step(self, conversation_id: str, action: Any) -> Any:
        """Run ``action(row)`` inside one ``BEGIN IMMEDIATE`` over this conversation's row."""

        validate_conversation_id(conversation_id)
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM voice_archive WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            result = action(row)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return result

    @staticmethod
    def _record(row: Any) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=row[0],
            session_id=row[1],
            committed=_progress(row[2], row[3], row[4], row[5]),
            pending=_progress(row[6], row[7], row[8], row[9]),
            quarantine=row[10],
            tombstone=row[11],
            holder=row[12],
        )

    def read(self, conversation_id: str) -> ConversationRecord | None:
        return self._step(  # type: ignore[no-any-return]
            conversation_id, lambda row: None if row is None else self._record(row)
        )

    def bind(
        self, conversation_id: str, session_id: str, creation: Progress
    ) -> ConversationRecord:
        """Bind a new conversation to the session it is about to create.

        The creation is recorded as pending, before the session exists, so recovery can
        tell an interrupted creation from a vanished archive.
        """

        if type(session_id) is not str or _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("session id must be 1-128 characters of [A-Za-z0-9_-]")
        pending = _columns(creation)

        def action(row: Any) -> ConversationRecord:
            if row is not None or self._connection.execute(
                "SELECT 1 FROM voice_archive WHERE session_id = ?", (session_id,)
            ).fetchone():
                raise ArchiveRefusal("bound")
            self._connection.execute(
                "INSERT INTO voice_archive (conversation_id, session_id, pending_count, "
                "pending_chain, pending_generation, pending_seq) VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, session_id, *pending),
            )
            return ConversationRecord(
                conversation_id, session_id, None, creation, None, None, None
            )

        return self._step(conversation_id, action)  # type: ignore[no-any-return]

    def set_holder(self, conversation_id: str, holder: str) -> None:
        if type(holder) is not str or not 0 < len(holder) <= _MAX_HOLDER_CHARS:
            raise ValueError("lease holder must be a bounded, non-empty str")

        def action(row: Any) -> None:
            if row is None:
                raise ArchiveRefusal("unbound")
            self._connection.execute(
                "UPDATE voice_archive SET holder = ? WHERE conversation_id = ?",
                (holder, conversation_id),
            )

        self._step(conversation_id, action)

    def begin_pending(self, conversation_id: str, committed: Progress, pending: Progress) -> None:
        """Record the intended next state, if every fence allows it."""

        expected = _columns(committed)
        intended = _columns(pending)

        def action(row: Any) -> None:
            if row is None:
                raise ArchiveRefusal("unbound")
            if row[10] is not None:
                raise ArchiveRefusal("quarantined")
            if row[11] is not None:
                raise ArchiveRefusal("tombstoned")
            if row[6] is not None:
                raise ArchiveRefusal("pending")
            if tuple(row[2:6]) != expected:
                raise ArchiveRefusal("stale")
            self._connection.execute(
                "UPDATE voice_archive SET pending_count = ?, pending_chain = ?, "
                "pending_generation = ?, pending_seq = ? WHERE conversation_id = ?",
                (*intended, conversation_id),
            )

        self._step(conversation_id, action)

    def _resolve_pending(self, conversation_id: str, pending: Progress, promote: bool) -> None:
        intended = _columns(pending)

        def action(row: Any) -> None:
            if row is None:
                raise ArchiveRefusal("unbound")
            if tuple(row[6:10]) != intended:
                raise ArchiveRefusal("stale")
            if promote:
                self._connection.execute(
                    "UPDATE voice_archive SET committed_count = pending_count, "
                    "committed_chain = pending_chain, committed_generation = pending_generation, "
                    "committed_seq = pending_seq WHERE conversation_id = ?",
                    (conversation_id,),
                )
            self._connection.execute(
                "UPDATE voice_archive SET pending_count = NULL, pending_chain = NULL, "
                "pending_generation = NULL, pending_seq = NULL WHERE conversation_id = ?",
                (conversation_id,),
            )

        self._step(conversation_id, action)

    def promote(self, conversation_id: str, pending: Progress) -> None:
        """Make the pending state committed: the archive now provably holds it."""

        self._resolve_pending(conversation_id, pending, promote=True)

    def clear_pending(self, conversation_id: str, pending: Progress) -> None:
        """Drop the pending state: the archive provably still holds the committed one."""

        self._resolve_pending(conversation_id, pending, promote=False)

    def quarantine(self, conversation_id: str, category: str) -> None:
        """Durably fence a conversation. The first reason recorded is kept."""

        if category not in QUARANTINE_CATEGORIES:
            raise ValueError("quarantine category is not recognized")

        def action(row: Any) -> None:
            if row is None:
                raise ArchiveRefusal("unbound")
            self._connection.execute(
                "UPDATE voice_archive SET quarantine = COALESCE(quarantine, ?) "
                "WHERE conversation_id = ?",
                (category, conversation_id),
            )

        self._step(conversation_id, action)
