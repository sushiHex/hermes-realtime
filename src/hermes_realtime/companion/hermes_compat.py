"""Every Hermes name the voice companion relies on, enumerated in one module.

Qualified against Hermes v0.21.0 (29112bef). ``SURFACE`` lists each name with the exact
parameters the pin gives it, and every lookup goes through ``resolve`` or ``_method``, which
refuse any name the tuple does not list: the surface is enforced, not documented. The
qualification (``scripts/qualify_hermes_voice_archive.py``) checks each name and signature
against the pin, and proves the private archive operation behaves exactly like Hermes's own
``append_messages_batch``.

The archive operation is private on purpose: Hermes's public append cannot verify the whole
archive and insert in the same write transaction, and nesting it inside ``_execute_write``
would open a second one.

Hermes is imported lazily, inside functions, so importing this module needs no Hermes.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, NamedTuple

from hermes_realtime.companion.integrity import (
    HEADER_COLUMNS,
    MESSAGE_COLUMNS,
    VOICE_SOURCE,
    ArchivePlan,
    ArchiveRefusal,
    Fingerprint,
    Projection,
    VoiceBatch,
    check_partition,
    plan_archive,
    platform_message_id,
    project,
    voice_metadata,
)

HERMES_MODULE = "hermes_state"


class SurfaceName(NamedTuple):
    """One relied-on name, and its exact parameter names at the pin (None: not callable)."""

    name: str
    parameters: tuple[str, ...] | None


SURFACE: tuple[SurfaceName, ...] = (
    SurfaceName("SessionDB", None),
    SurfaceName("SessionTurnLeaseLostError", None),
    SurfaceName("CompressionSessionClosedError", None),
    SurfaceName("SessionDB._TRANSCRIPT_WRITE_PATIENCE_S", None),
    SurfaceName("SessionDB._execute_write", ("self", "fn", "patience_s")),
    SurfaceName(
        "SessionDB._check_transcript_write_guards",
        (
            "self",
            "conn",
            "session_id",
            "compression_lock_holder",
            "turn_lease_holder",
            "turn_lease_ttl_seconds",
            "reject_active_turn_lease",
            "reject_active_compression_lock",
        ),
    ),
    SurfaceName("SessionDB._insert_message_rows", ("self", "conn", "session_id", "messages")),
    SurfaceName("SessionDB.create_session", ("self", "session_id", "source", "kwargs")),
    SurfaceName(
        "SessionDB.try_acquire_session_turn_lease",
        ("self", "session_id", "holder", "ttl_seconds", "patience_s"),
    ),
    SurfaceName(
        "SessionDB.refresh_session_turn_lease", ("self", "session_id", "holder", "ttl_seconds")
    ),
    SurfaceName("SessionDB.release_session_turn_lease", ("self", "session_id", "holder")),
)
# The table shapes the projection relies on: every messages column (the fingerprint covers
# them all), and the session columns the header reads.
MESSAGE_TABLE_COLUMNS = frozenset({"id", "session_id", *MESSAGE_COLUMNS})
SESSION_TABLE_COLUMNS = frozenset({"id", "message_count", *HEADER_COLUMNS})

_LISTED = frozenset(entry.name for entry in SURFACE)
_ROW_SQL = (
    f"SELECT {', '.join(MESSAGE_COLUMNS)} FROM messages "
    "WHERE session_id = ? ORDER BY id ASC LIMIT ?"
)
_HEADER_SQL = f"SELECT {', '.join(HEADER_COLUMNS)} FROM sessions WHERE id = ?"


class CompatError(LookupError):
    """A Hermes name outside the enumerated surface was requested."""


def _lookup(name: str) -> Any:
    target: Any = importlib.import_module(HERMES_MODULE)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def resolve(name: str) -> Any:
    """Look up one listed Hermes name; an unlisted one is refused before any import."""

    if name not in _LISTED:
        raise CompatError("Hermes name is not on the enumerated surface")
    return _lookup(name)


def _method(db: Any, name: str) -> Any:
    if name not in _LISTED or not name.startswith("SessionDB."):
        raise CompatError("Hermes name is not on the enumerated surface")
    return getattr(db, name.removeprefix("SessionDB."))


def _session_db(db: Any) -> Any:
    if type(db) is not resolve("SessionDB"):
        raise TypeError("the companion binds only an exact Hermes SessionDB")
    return db


def check_surface() -> tuple[str, ...]:
    """Names that are missing, or whose parameters differ from the pin; empty when current."""

    failures: list[str] = []
    for entry in SURFACE:
        try:
            target = _lookup(entry.name)
        except (ImportError, AttributeError):
            failures.append(f"missing:{entry.name}")
            continue
        if entry.parameters is not None:
            try:
                parameters = tuple(inspect.signature(target).parameters)
            except (TypeError, ValueError):
                parameters = ()
            if parameters != entry.parameters:
                failures.append(f"signature:{entry.name}")
    return tuple(failures)


def check_shapes(db: Any) -> tuple[str, ...]:
    """Tables whose columns differ from what the projection relies on; empty when current."""

    def read(conn: Any) -> tuple[frozenset[str], frozenset[str]]:
        messages = frozenset(row[1] for row in conn.execute("PRAGMA table_info(messages)"))
        sessions = frozenset(row[1] for row in conn.execute("PRAGMA table_info(sessions)"))
        return messages, sessions

    messages, sessions = _method(_session_db(db), "SessionDB._execute_write")(read)
    failures: list[str] = []
    if messages != MESSAGE_TABLE_COLUMNS:
        failures.append("shape:messages")
    if SESSION_TABLE_COLUMNS - sessions:
        failures.append("shape:sessions")
    return tuple(failures)


def _read_projection(conn: Any, session_id: str, cap: int) -> Projection | None:
    header = conn.execute(_HEADER_SQL, (session_id,)).fetchone()
    if header is None:
        return None
    # At most cap + 1 rows, inactive and compacted included, in Hermes's own read order.
    rows = conn.execute(_ROW_SQL, (session_id, cap + 1)).fetchall()
    return project(
        dict(zip(HEADER_COLUMNS, tuple(header), strict=True)),
        [dict(zip(MESSAGE_COLUMNS, tuple(row), strict=True)) for row in rows],
        cap,
    )


def read_projection(db: Any, session_id: str, cap: int) -> Projection | None:
    """One consistent read of a session's projection, serialized with every writer."""

    return _method(_session_db(db), "SessionDB._execute_write")(  # type: ignore[no-any-return]
        lambda conn: _read_projection(conn, session_id, cap)
    )


def create_voice_session(db: Any, session_id: str) -> None:
    """Create the voice session.

    Hermes's creation is an upsert that keeps an existing row's fields. The caller verifies
    the created session against the expected genesis fingerprint, so a session that already
    held this id, with any other header or any row, is quarantined rather than adopted.
    """

    _method(_session_db(db), "SessionDB.create_session")(session_id, source=VOICE_SOURCE)


def acquire_lease(db: Any, session_id: str, holder: str, ttl_seconds: float) -> bool:
    acquire = _method(_session_db(db), "SessionDB.try_acquire_session_turn_lease")
    return acquire(session_id, holder, ttl_seconds=ttl_seconds) is True


def refresh_lease(db: Any, session_id: str, holder: str, ttl_seconds: float) -> bool:
    refresh = _method(_session_db(db), "SessionDB.refresh_session_turn_lease")
    return refresh(session_id, holder, ttl_seconds=ttl_seconds) is True


def release_lease(db: Any, session_id: str, holder: str) -> None:
    _method(_session_db(db), "SessionDB.release_session_turn_lease")(session_id, holder)


def archive_voice_rows(
    db: Any,
    session_id: str,
    holder: str,
    batch: VoiceBatch,
    expected_committed: Fingerprint,
    expected_pending: Fingerprint,
    *,
    conversation_id: str,
    cap: int,
    lease_ttl_seconds: float,
) -> ArchivePlan:
    """Verify the whole archive and append only the missing rows, in one write transaction.

    A batch whose rows and gaps do not partition its range is refused whole, before any
    transaction. Inside Hermes's ``BEGIN IMMEDIATE``: the lease guard with this holder; a
    projection of every row, which must equal ``expected_committed`` (or ``expected_pending``:
    already applied); an identity and full-content check of the batch; the insert of only
    the missing rows; the ``message_count`` update Hermes's own append makes; and a re-read
    that must equal ``expected_pending``. Any refusal raises inside the transaction, which
    rolls it back, so a refused batch never mutates the archive.
    """

    rows = check_partition(batch).rows
    db = _session_db(db)
    execute_write = _method(db, "SessionDB._execute_write")
    guard = _method(db, "SessionDB._check_transcript_write_guards")
    insert = _method(db, "SessionDB._insert_message_rows")
    patience = _method(db, "SessionDB._TRANSCRIPT_WRITE_PATIENCE_S")
    lease_lost = resolve("SessionTurnLeaseLostError")
    rotated = resolve("CompressionSessionClosedError")

    def archive(conn: Any) -> ArchivePlan:
        try:
            guard(
                conn,
                session_id,
                None,
                turn_lease_holder=holder,
                turn_lease_ttl_seconds=lease_ttl_seconds,
            )
        except lease_lost:
            raise ArchiveRefusal("lease_lost") from None
        except rotated:
            raise ArchiveRefusal("rotated") from None
        plan = plan_archive(
            _read_projection(conn, session_id, cap),
            conversation_id,
            rows,
            expected_committed,
            expected_pending,
            cap,
        )
        if plan.inserts:
            messages = [
                {
                    "role": row.role,
                    "content": row.text,
                    "timestamp": row.timestamp,
                    "platform_message_id": platform_message_id(conversation_id, row.identity),
                    "display_metadata": voice_metadata(row),
                }
                for row in plan.inserts
            ]
            insert(conn, session_id, messages)
            conn.execute(
                "UPDATE sessions SET message_count = message_count + ? WHERE id = ?",
                (len(messages), session_id),
            )
        after = _read_projection(conn, session_id, cap)
        if after is None or after.fingerprint() != expected_pending:
            raise ArchiveRefusal("drift")
        return plan

    return execute_write(archive, patience_s=patience)  # type: ignore[no-any-return]


class HermesArchivePort:
    """The archive port over one exact Hermes ``SessionDB``."""

    def __init__(self, db: Any) -> None:
        self._db = _session_db(db)

    def create_session(self, session_id: str) -> None:
        create_voice_session(self._db, session_id)

    def acquire_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool:
        return acquire_lease(self._db, session_id, holder, ttl_seconds)

    def refresh_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool:
        return refresh_lease(self._db, session_id, holder, ttl_seconds)

    def release_lease(self, session_id: str, holder: str) -> None:
        release_lease(self._db, session_id, holder)

    def read_projection(self, session_id: str, cap: int) -> Projection | None:
        return read_projection(self._db, session_id, cap)

    def archive_rows(
        self,
        session_id: str,
        holder: str,
        conversation_id: str,
        batch: VoiceBatch,
        expected_committed: Fingerprint,
        expected_pending: Fingerprint,
        cap: int,
        lease_ttl_seconds: float,
    ) -> ArchivePlan:
        return archive_voice_rows(
            self._db,
            session_id,
            holder,
            batch,
            expected_committed,
            expected_pending,
            conversation_id=conversation_id,
            cap=cap,
            lease_ttl_seconds=lease_ttl_seconds,
        )
