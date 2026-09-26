from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from hermes_realtime.companion.integrity import (
    EXPECTED_HEADER,
    ArchiveRefusal,
    Fingerprint,
    Identity,
    genesis,
)
from hermes_realtime.companion.store import CompanionStore, ConversationRecord, Progress

_GENESIS = Progress(genesis(EXPECTED_HEADER), None)
_FIRST = Progress(Fingerprint(2, "1" * 64), Identity(0, 1))
_SECOND = Progress(Fingerprint(4, "2" * 64), Identity(0, 3))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CompanionStore]:
    opened = CompanionStore(tmp_path / "companion.db")
    try:
        yield opened
    finally:
        opened.close()


def _committed(store: CompanionStore) -> CompanionStore:
    store.bind("conv", "session_1", _GENESIS)
    store.promote("conv", _GENESIS)
    return store


def _category(error: pytest.ExceptionInfo[ArchiveRefusal]) -> str:
    return error.value.category


def test_a_bound_conversation_records_creation_as_pending(store: CompanionStore) -> None:
    record = store.bind("conv", "session_1", _GENESIS)
    assert record == ConversationRecord(
        conversation_id="conv",
        session_id="session_1",
        committed=None,
        pending=_GENESIS,
        quarantine=None,
        tombstone=None,
        holder=None,
    )
    assert store.read("conv") == record
    assert store.read("other") is None


def test_a_conversation_or_session_binds_once(store: CompanionStore) -> None:
    store.bind("conv", "session_1", _GENESIS)
    with pytest.raises(ArchiveRefusal) as refusal:
        store.bind("conv", "session_2", _GENESIS)
    assert _category(refusal) == "bound"
    with pytest.raises(ArchiveRefusal) as refusal:
        store.bind("conv2", "session_1", _GENESIS)
    assert _category(refusal) == "bound"


def test_pending_is_promoted_to_committed_with_its_cursor(store: CompanionStore) -> None:
    _committed(store)
    store.begin_pending("conv", _GENESIS, _FIRST)
    record = store.read("conv")
    assert record is not None and record.committed == _GENESIS and record.pending == _FIRST
    store.promote("conv", _FIRST)
    record = store.read("conv")
    assert record is not None and record.committed == _FIRST and record.pending is None


def test_pending_can_be_cleared_only_by_its_own_value(store: CompanionStore) -> None:
    _committed(store)
    store.begin_pending("conv", _GENESIS, _FIRST)
    with pytest.raises(ArchiveRefusal) as refusal:
        store.clear_pending("conv", _SECOND)
    assert _category(refusal) == "stale"
    with pytest.raises(ArchiveRefusal) as refusal:
        store.promote("conv", _SECOND)
    assert _category(refusal) == "stale"
    store.clear_pending("conv", _FIRST)
    record = store.read("conv")
    assert record is not None and record.committed == _GENESIS and record.pending is None


def test_a_second_pending_write_is_refused(store: CompanionStore) -> None:
    _committed(store)
    store.begin_pending("conv", _GENESIS, _FIRST)
    with pytest.raises(ArchiveRefusal) as refusal:
        store.begin_pending("conv", _GENESIS, _FIRST)
    assert _category(refusal) == "pending"


def test_pending_requires_the_committed_progress_it_was_computed_from(
    store: CompanionStore,
) -> None:
    _committed(store)
    with pytest.raises(ArchiveRefusal) as refusal:
        store.begin_pending("conv", _FIRST, _SECOND)
    assert _category(refusal) == "stale"
    record = store.read("conv")
    assert record is not None and record.pending is None


def test_pending_is_refused_for_a_quarantined_conversation(store: CompanionStore) -> None:
    _committed(store)
    store.quarantine("conv", "mismatch")
    with pytest.raises(ArchiveRefusal) as refusal:
        store.begin_pending("conv", _GENESIS, _FIRST)
    assert _category(refusal) == "quarantined"


def test_pending_is_refused_for_a_tombstoned_conversation(
    store: CompanionStore, tmp_path: Path
) -> None:
    _committed(store)
    # Forget is not wired in this milestone; the column exists and the fence reads it.
    with contextlib.closing(sqlite3.connect(tmp_path / "companion.db")) as raw:
        raw.execute("UPDATE voice_archive SET tombstone = 0 WHERE conversation_id = 'conv'")
        raw.commit()
    with pytest.raises(ArchiveRefusal) as refusal:
        store.begin_pending("conv", _GENESIS, _FIRST)
    assert _category(refusal) == "tombstoned"


def test_an_unbound_conversation_is_refused(store: CompanionStore) -> None:
    for step in (
        lambda: store.begin_pending("conv", _GENESIS, _FIRST),
        lambda: store.promote("conv", _FIRST),
        lambda: store.clear_pending("conv", _FIRST),
        lambda: store.quarantine("conv", "mismatch"),
        lambda: store.set_holder("conv", "pid=1:voice=conv:boot=x"),
    ):
        with pytest.raises(ArchiveRefusal) as refusal:
            step()
        assert _category(refusal) == "unbound"


def test_quarantine_is_durable_first_reason_wins_and_keeps_pending(tmp_path: Path) -> None:
    path = tmp_path / "companion.db"
    store = CompanionStore(path)
    try:
        _committed(store)
        store.begin_pending("conv", _GENESIS, _FIRST)
        store.quarantine("conv", "missing")
        store.quarantine("conv", "mismatch")
    finally:
        store.close()
    reopened = CompanionStore(path)
    try:
        record = reopened.read("conv")
        assert record is not None
        assert record.quarantine == "missing"
        assert record.pending == _FIRST
    finally:
        reopened.close()


def test_quarantine_categories_are_closed(store: CompanionStore) -> None:
    _committed(store)
    with pytest.raises(ValueError):
        store.quarantine("conv", "because")


def test_the_holder_is_recorded(store: CompanionStore) -> None:
    _committed(store)
    store.set_holder("conv", "pid=1:voice=conv:boot=abc")
    record = store.read("conv")
    assert record is not None and record.holder == "pid=1:voice=conv:boot=abc"


def test_the_store_holds_fences_and_progress_never_transcript_text(tmp_path: Path) -> None:
    CompanionStore(tmp_path / "companion.db").close()
    with contextlib.closing(sqlite3.connect(tmp_path / "companion.db")) as raw:
        columns = [row[1] for row in raw.execute("PRAGMA table_info(voice_archive)")]
        tables = [
            row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
    assert tables == ["voice_archive"]
    assert columns == [
        "conversation_id",
        "session_id",
        "committed_count",
        "committed_chain",
        "committed_generation",
        "committed_seq",
        "pending_count",
        "pending_chain",
        "pending_generation",
        "pending_seq",
        "quarantine",
        "tombstone",
        "holder",
        "review_ledger",
    ]


def test_the_store_commits_durably(store: CompanionStore) -> None:
    assert store.pragma("synchronous") == 2
    assert store.pragma("journal_mode") == "delete"


@pytest.mark.parametrize("version", [1, 2, 99])
def test_a_store_of_another_schema_version_is_refused(tmp_path: Path, version: int) -> None:
    # Version 1 lacks the tied CHECK constraints; version 2 cannot record "count".
    path = tmp_path / "companion.db"
    with contextlib.closing(sqlite3.connect(path)) as raw:
        raw.execute(f"PRAGMA user_version = {version}")
        raw.commit()
    with pytest.raises(RuntimeError):
        CompanionStore(path)


def test_a_journal_mode_that_does_not_take_is_refused() -> None:
    # SQLite answers "memory" to a DELETE request on an in-memory database: nothing durable.
    with pytest.raises(RuntimeError):
        CompanionStore(Path(":memory:"))


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE voice_archive SET committed_count = 0",
        "UPDATE voice_archive SET committed_generation = NULL, committed_seq = NULL",
        "UPDATE voice_archive SET quarantine = 'because'",
        "UPDATE voice_archive SET committed_chain = 'short'",
        "UPDATE voice_archive SET committed_seq = -1",
    ],
    ids=["count-without-rows", "rows-without-cursor", "quarantine", "chain", "negative"],
)
def test_the_schema_refuses_inconsistent_progress(tmp_path: Path, statement: str) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    try:
        _committed(store)
        store.begin_pending("conv", _GENESIS, _FIRST)
        store.promote("conv", _FIRST)
    finally:
        store.close()
    with (
        contextlib.closing(sqlite3.connect(tmp_path / "companion.db")) as raw,
        pytest.raises(sqlite3.IntegrityError),
    ):
        raw.execute(statement)


def test_progress_ties_its_cursor_to_its_count() -> None:
    with pytest.raises(ValueError):
        Progress(Fingerprint(0, "0" * 64), Identity(0, 0))
    with pytest.raises(ValueError):
        Progress(Fingerprint(2, "0" * 64), None)


def test_store_arguments_are_exact(tmp_path: Path, store: CompanionStore) -> None:
    with pytest.raises(TypeError):
        CompanionStore(str(tmp_path / "other.db"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.bind("conv", "session_1", (_GENESIS,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.bind("conv:bad", "session_1", _GENESIS)
    with pytest.raises(TypeError):
        Progress(_GENESIS.fingerprint, (0, 1))  # type: ignore[arg-type]
