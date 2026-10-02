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

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_realtime.companion.integrity import (
    MAX_ARCHIVE_ROWS,
    MAX_IDENTITY,
    ArchiveRefusal,
    Fingerprint,
    Identity,
    validate_conversation_id,
)

# Version 1 was a draft without the tied progress and quarantine constraints; version 2
# could not record a "count" quarantine.
_SCHEMA_VERSION = 3
_CONCRETE_PATH = type(Path())
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_MAX_HOLDER_CHARS = 256
# A generous bound on stored conversations: each is one row of fences and progress.
MAX_BOUND_CONVERSATIONS = 4096
# One periodic and one closing identity per archived row fit without eviction.
# Gap-heavy close histories can exceed this bound and are refused fail-closed.
MAX_REVIEW_LEDGER_ENTRIES = 2 * MAX_ARCHIVE_ROWS
_REVIEW_OUTCOMES = frozenset(
    {"reserved", "accepted", "finished", "failed", "cancelled", "unknown"}
)
# Categories a durable quarantine may record: the archive no longer matches its evidence.
QUARANTINE_CATEGORIES = frozenset(
    {"mismatch", "missing", "over_cap", "rotated", "recovery", "lineage", "count"}
)

def _progress_checks(prefix: str) -> str:
    """One progress state's constraints: all or nothing, bounded, and cursor tied to count.

    A state covering no rows has no cursor; a state covering any row has one.
    """

    count, chain = f"{prefix}_count", f"{prefix}_chain"
    generation, seq = f"{prefix}_generation", f"{prefix}_seq"
    return (
        f"CHECK (({count} IS NULL) = ({chain} IS NULL)),\n"
        f"    CHECK (({generation} IS NULL) = ({seq} IS NULL)),\n"
        f"    CHECK ({count} IS NULL OR ({count} = 0) = ({generation} IS NULL)),\n"
        f"    CHECK ({count} IS NULL OR {count} >= 0),\n"
        f"    CHECK ({generation} IS NULL OR ({generation} >= 0 AND {seq} >= 0)),\n"
        f"    CHECK ({chain} IS NULL OR length({chain}) = 64)"
    )


_QUARANTINE_SQL = ", ".join(f"'{category}'" for category in sorted(QUARANTINE_CATEGORIES))
_SCHEMA = f"""
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
    review_ledger TEXT NOT NULL DEFAULT '{{}}',
    {_progress_checks("committed")},
    {_progress_checks("pending")},
    CHECK (committed_count IS NOT NULL OR pending_count IS NOT NULL),
    CHECK (quarantine IS NULL OR quarantine IN ({_QUARANTINE_SQL}))
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
        # Every batch archives at least one row, so a cursor exists exactly when rows do.
        if (self.fingerprint.count == 0) != (self.cursor is None):
            raise ValueError("progress must have a cursor exactly when it covers rows")


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
            # The pragma answers with the mode in force, which need not be the one requested.
            mode = self._connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
            if mode != "delete":
                raise RuntimeError("companion store could not use a rollback journal")
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

    @staticmethod
    def _review_key(request: Any) -> str:
        return (
            f"{request.generation}:{request.seq_from}:"
            f"{request.seq_through}:{int(request.closing)}"
        )

    @staticmethod
    def _decode_review_ledger(raw: object) -> dict[str, dict[str, str]]:
        if type(raw) is not str or len(raw) > 1_048_576:
            raise RuntimeError("review ledger is invalid")
        def distinct(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate review ledger key")
                result[key] = value
            return result
        try:
            ledger = json.loads(raw, object_pairs_hook=distinct)
        except (ValueError, TypeError) as error:
            raise RuntimeError("review ledger is invalid") from error
        if type(ledger) is not dict or len(ledger) > MAX_REVIEW_LEDGER_ENTRIES:
            raise RuntimeError("review ledger is invalid")
        ids: set[str] = set()
        for key, value in ledger.items():
            parts = key.split(":") if type(key) is str else []
            valid_key = (
                len(parts) == 4
                and len(key) <= 80
                and all(1 <= len(part) <= 16 for part in parts)
                and all(part.isascii() and part.isdecimal() for part in parts)
                and all(str(int(part)) == part for part in parts)
                and all(0 <= int(part) <= MAX_IDENTITY for part in parts[:3])
                and int(parts[1]) <= int(parts[2])
                and parts[3] in {"0", "1"}
            )
            if (
                not valid_key or type(value) is not dict
                or set(value) != {"id", "outcome"}
                or type(value["id"]) is not str
                or not re.fullmatch(r"vr_[0-9a-f]{32}", value["id"])
                or value["id"] in ids
                or type(value["outcome"]) is not str
                or value["outcome"] not in _REVIEW_OUTCOMES
            ):
                raise RuntimeError("review ledger is invalid")
            ids.add(value["id"])
        return ledger

    @staticmethod
    def _admission_fences(row: Any) -> None:
        if row is None or row[2] is None:
            raise ArchiveRefusal("unbound")
        if row[10] is not None:
            raise ArchiveRefusal("quarantined")
        if row[11] is not None:
            raise ArchiveRefusal("tombstoned")
        if row[6] is not None:
            raise ArchiveRefusal("pending")

    def _review_ledger(self, conversation_id: str, row: Any) -> dict[str, dict[str, str]]:
        if row is None:
            raise ArchiveRefusal("unbound")
        raw = self._connection.execute(
            "SELECT review_ledger FROM voice_archive WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        return self._decode_review_ledger(raw)

    def find_review(self, request: Any) -> tuple[str, str] | None:
        key = self._review_key(request)
        def action(row: Any) -> tuple[str, str] | None:
            entry = self._review_ledger(request.conversation_id, row).get(key)
            return None if entry is None else (entry["id"], entry["outcome"])
        return self._step(request.conversation_id, action)  # type: ignore[no-any-return]

    def reserve_review(self, request: Any, review_id: str) -> None:
        if type(review_id) is not str or not re.fullmatch(r"vr_[0-9a-f]{32}", review_id):
            raise ValueError("review ID is invalid")
        key = self._review_key(request)

        def action(row: Any) -> None:
            self._admission_fences(row)
            ledger = self._review_ledger(request.conversation_id, row)
            if key in ledger:
                raise ArchiveRefusal("busy")
            if len(ledger) >= MAX_REVIEW_LEDGER_ENTRIES:
                raise ArchiveRefusal("capacity")
            ledger[key] = {"id": review_id, "outcome": "reserved"}
            self._connection.execute(
                "UPDATE voice_archive SET review_ledger = ? WHERE conversation_id = ?",
                (
                    json.dumps(ledger, sort_keys=True, separators=(",", ":")),
                    request.conversation_id,
                ),
            )

        self._step(request.conversation_id, action)

    def accept_review(self, request: Any, review_id: str) -> None:
        key = self._review_key(request)

        def action(row: Any) -> None:
            self._admission_fences(row)
            ledger = self._review_ledger(request.conversation_id, row)
            entry = ledger.get(key)
            if entry != {"id": review_id, "outcome": "reserved"}:
                raise RuntimeError("review reservation changed")
            entry["outcome"] = "accepted"
            self._connection.execute(
                "UPDATE voice_archive SET review_ledger = ? WHERE conversation_id = ?",
                (
                    json.dumps(ledger, sort_keys=True, separators=(",", ":")),
                    request.conversation_id,
                ),
            )

        self._step(request.conversation_id, action)

    def finish_review(self, conversation_id: str, review_id: str, outcome: str) -> None:
        if outcome not in _REVIEW_OUTCOMES or outcome == "accepted":
            raise ValueError("review outcome is invalid")
        def action(row: Any) -> None:
            ledger = self._review_ledger(conversation_id, row)
            for entry in ledger.values():
                if entry["id"] == review_id:
                    entry["outcome"] = outcome
                    self._connection.execute(
                        "UPDATE voice_archive SET review_ledger = ? WHERE conversation_id = ?",
                        (
                            json.dumps(ledger, sort_keys=True, separators=(",", ":")),
                            conversation_id,
                        ),
                    )
                    return
            raise RuntimeError("review identity is absent")
        self._step(conversation_id, action)

    def review_outcome(self, conversation_id: str, review_id: str) -> str | None:
        def action(row: Any) -> str | None:
            ledger = self._review_ledger(conversation_id, row)
            for entry in ledger.values():
                if entry["id"] == review_id:
                    return entry["outcome"]
            return None
        return self._step(conversation_id, action)  # type: ignore[no-any-return]

    def recover_reviews(self) -> None:
        """A prior process's admitted thread cannot be observed; retain unknown evidence."""

        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                "SELECT conversation_id, review_ledger FROM voice_archive"
            )
            for count, (conversation_id, raw) in enumerate(cursor, start=1):
                if count > MAX_BOUND_CONVERSATIONS:
                    raise RuntimeError("review ledger conversation bound exceeded")
                ledger = self._decode_review_ledger(raw)
                if any(entry["outcome"] in {"reserved", "accepted"} for entry in ledger.values()):
                    for entry in ledger.values():
                        if entry["outcome"] in {"reserved", "accepted"}:
                            entry["outcome"] = "unknown"
                    connection.execute(
                        "UPDATE voice_archive SET review_ledger = ? WHERE conversation_id = ?",
                        (
                            json.dumps(ledger, sort_keys=True, separators=(",", ":")),
                            conversation_id,
                        ),
                    )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

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
            (bound,) = self._connection.execute("SELECT COUNT(*) FROM voice_archive").fetchone()
            if bound >= MAX_BOUND_CONVERSATIONS:
                # Fails closed on binding, never at start; forget (M3) prunes.
                raise ArchiveRefusal("conversations")
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
