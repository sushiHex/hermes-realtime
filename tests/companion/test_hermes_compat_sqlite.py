"""The real ``hermes_compat`` SQL, run in the base suite against Hermes's exact table shapes.

``tests/support/hermes_state_stand_in.py`` stands in for the pinned ``hermes_state``: the
private operation, the projection read and its ordering, the ``message_count`` update and
the post-insert re-read all run here, in CI, without Hermes.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.companion.archive import OpenReport, VoiceArchive
from hermes_realtime.companion.hermes_compat import (
    HERMES_MODULE,
    HermesArchivePort,
    archive_voice_rows,
    check_shapes,
    check_surface,
    durability_level,
    read_projection,
)
from hermes_realtime.companion.integrity import (
    EXPECTED_HEADER,
    MAX_ARCHIVE_ROWS,
    VOICE_SOURCE,
    ArchiveRefusal,
    Identity,
    VoiceBatch,
    VoiceRow,
    expected_after,
    genesis,
)
from hermes_realtime.companion.store import CompanionStore

_STAND_IN_PATH = Path(__file__).resolve().parents[1] / "support" / "hermes_state_stand_in.py"
_SPEC = importlib.util.spec_from_file_location("hermes_state_stand_in", _STAND_IN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
hermes_state_stand_in = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hermes_state_stand_in)

_CONVERSATION = "conv"


@pytest.fixture
def hermes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    monkeypatch.setitem(sys.modules, HERMES_MODULE, hermes_state_stand_in)
    db = hermes_state_stand_in.SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CompanionStore]:
    opened = CompanionStore(tmp_path / "companion.db")
    try:
        yield opened
    finally:
        opened.close()


def _rows(start: int, stop: int, *, gap: tuple[int, int] | None = None) -> tuple[VoiceRow, ...]:
    return tuple(
        VoiceRow(
            identity=Identity(0, seq),
            role="user" if seq % 2 == 0 else "assistant",
            text=f"row {seq}",
            interrupted=seq % 4 == 3,
            timestamp=1_700_000_000.0 + seq,
            gap_before=gap if seq == start else None,
        )
        for seq in range(start, stop)
    )


def _batch(start: int, stop: int) -> VoiceBatch:
    return VoiceBatch(0, start, stop - 1, _rows(start, stop))


def _raw(tmp_path: Path, sql: str, *parameters: object) -> list[tuple[Any, ...]]:
    with contextlib.closing(sqlite3.connect(tmp_path / "state.db")) as raw:
        return [tuple(row) for row in raw.execute(sql, parameters).fetchall()]


async def _open(store: CompanionStore, hermes: Any) -> VoiceArchive:
    archive = VoiceArchive(store, HermesArchivePort(hermes))
    assert await archive.open(_CONVERSATION) == OpenReport(recovery="created")
    return archive


def test_the_stand_in_matches_the_enumerated_surface_and_shapes(hermes: Any) -> None:
    assert check_surface() == ()
    assert check_shapes(hermes) == ()
    assert durability_level(hermes) == 2


def test_the_sessions_shape_is_bound_by_equality(hermes: Any) -> None:
    hermes._conn.execute("ALTER TABLE sessions ADD COLUMN voice_extra TEXT")
    assert check_shapes(hermes) == ("shape:sessions",)


def test_the_messages_shape_is_bound_by_equality(hermes: Any) -> None:
    hermes._conn.execute("ALTER TABLE messages ADD COLUMN voice_extra TEXT")
    assert check_shapes(hermes) == ("shape:messages",)


@pytest.mark.asyncio
async def test_the_private_operation_appends_in_order_and_counts(
    store: CompanionStore, hermes: Any, tmp_path: Path
) -> None:
    archive = await _open(store, hermes)
    try:
        first = await archive.archive(_CONVERSATION, _batch(0, 4))
        gapped = VoiceBatch(0, 4, 7, (*_rows(6, 7, gap=(4, 5)), *_rows(7, 8)))
        second = await archive.archive(_CONVERSATION, gapped)
        retried = await archive.archive(_CONVERSATION, gapped)
    finally:
        await archive.close()
    assert (first.inserted, second.inserted, retried.inserted) == (4, 2, 0)
    record = store.read(_CONVERSATION)
    assert record is not None and record.committed is not None and record.pending is None
    rows = (*_rows(0, 4), *gapped.rows)
    assert record.committed.fingerprint == expected_after(
        genesis(EXPECTED_HEADER), _CONVERSATION, rows
    )
    identities = _raw(
        tmp_path, "SELECT platform_message_id FROM messages WHERE session_id = ? ORDER BY id",
        record.session_id,
    )
    assert [identity for (identity,) in identities] == [
        f"voice:conv:0:{seq}" for seq in (0, 1, 2, 3, 6, 7)
    ]
    assert _raw(tmp_path, "SELECT source, message_count FROM sessions WHERE id = ?",
                record.session_id) == [(VOICE_SOURCE, 6)]


@pytest.mark.asyncio
async def test_storage_that_drifts_is_refused_inside_the_transaction(
    store: CompanionStore, hermes: Any, tmp_path: Path
) -> None:
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, _batch(0, 2))
        original = hermes._insert_message_rows

        def drifting(conn: Any, session_id: Any, messages: Any) -> Any:
            result = original(conn, session_id, messages)
            conn.execute("UPDATE messages SET display_kind = 'drift'")
            return result

        hermes._insert_message_rows = drifting
        before = _raw(tmp_path, "SELECT * FROM messages ORDER BY id")
        with pytest.raises(ArchiveRefusal) as refusal:
            await archive.archive(_CONVERSATION, _batch(2, 4))
        assert refusal.value.category == "drift"
        assert _raw(tmp_path, "SELECT * FROM messages ORDER BY id") == before
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_lost_lease_and_a_child_session_are_refused(
    store: CompanionStore, hermes: Any
) -> None:
    archive = await _open(store, hermes)
    record = store.read(_CONVERSATION)
    assert record is not None and record.committed is not None
    committed = record.committed.fingerprint
    try:
        with pytest.raises(ArchiveRefusal) as refusal:
            archive_voice_rows(hermes, record.session_id, "pid=1:someone-else", _batch(0, 1),
                               committed, committed, conversation_id=_CONVERSATION,
                               cap=MAX_ARCHIVE_ROWS, lease_ttl_seconds=300.0)
        assert refusal.value.category == "lease_lost"
        hermes.create_session("branch", "cli", parent_session_id=record.session_id)
        with pytest.raises(ArchiveRefusal) as refusal:
            read_projection(hermes, record.session_id, MAX_ARCHIVE_ROWS)
        assert refusal.value.category == "lineage"
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_stale_message_count_is_quarantined_before_any_write(
    store: CompanionStore, hermes: Any, tmp_path: Path
) -> None:
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, _batch(0, 2))
        record = store.read(_CONVERSATION)
        assert record is not None
        # A foreign UPDATE of the counter alone: every row stays exactly as archived.
        hermes._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (record.session_id,),
        ))
        before = _raw(tmp_path, "SELECT * FROM messages ORDER BY id")
        with pytest.raises(ArchiveRefusal) as refusal:
            await archive.archive(_CONVERSATION, _batch(2, 4))
        assert refusal.value.category == "count"
        assert _raw(tmp_path, "SELECT * FROM messages ORDER BY id") == before
        assert _raw(tmp_path, "SELECT message_count FROM sessions WHERE id = ?",
                    record.session_id) == [(3,)]
    finally:
        await archive.close()
    record = store.read(_CONVERSATION)
    assert record is not None and record.quarantine == "count"
    reopened = VoiceArchive(store, HermesArchivePort(hermes))
    with pytest.raises(ArchiveRefusal) as refusal:
        await reopened.open(_CONVERSATION)
    assert refusal.value.category == "quarantined"


@pytest.mark.asyncio
async def test_a_hermes_below_full_synchronous_is_not_made_ready(
    store: CompanionStore, hermes: Any
) -> None:
    hermes._conn.execute("PRAGMA synchronous = NORMAL")
    assert durability_level(hermes) == 1
    archive = VoiceArchive(store, HermesArchivePort(hermes))
    with pytest.raises(ArchiveRefusal) as refusal:
        await archive.open(_CONVERSATION)
    assert refusal.value.category == "durability"
    assert archive.ready(_CONVERSATION) is False
