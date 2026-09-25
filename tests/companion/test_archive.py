from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hermes_realtime.companion.archive import ArchiveAck, OpenReport, VoiceArchive
from hermes_realtime.companion.integrity import (
    EXPECTED_HEADER,
    VOICE_SOURCE,
    ArchivePlan,
    ArchiveRefusal,
    Fingerprint,
    Identity,
    Projection,
    VoiceRow,
    expected_row_values,
    genesis,
    plan_archive,
    project,
)
from hermes_realtime.companion.store import CompanionStore, Progress

_CONVERSATION = "conv"
_TTL = 3.0


class FakeHermes:
    """An in-memory stand-in for the enumerated Hermes surface, with injectable faults."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}
        self.rows: dict[str, list[dict[str, object]]] = {}
        self.lease: dict[str, str] = {}
        self.calls: list[str] = []
        self.refresh_result: bool | Exception = True
        self.drift: dict[str, object] = {}
        self.after_insert: Callable[[], None] | None = None
        self.on_acquire: Callable[[str], None] | None = None

    def only(self) -> str:
        (session_id,) = self.sessions
        return session_id

    def create_session(self, session_id: str) -> None:
        self.calls.append("create_session")
        if session_id in self.sessions:
            raise ArchiveRefusal("session_exists")
        self.sessions[session_id] = {
            "parent_session_id": None,
            "source": VOICE_SOURCE,
            "ended_at": None,
            "end_reason": None,
        }
        self.rows[session_id] = []

    def acquire_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool:
        self.calls.append("acquire_lease")
        if self.on_acquire is not None:
            self.on_acquire(holder)
        if self.lease.get(session_id, holder) != holder:
            return False
        self.lease[session_id] = holder
        return True

    def refresh_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool:
        self.calls.append("refresh_lease")
        if isinstance(self.refresh_result, Exception):
            raise self.refresh_result
        return self.refresh_result and self.lease.get(session_id) == holder

    def release_lease(self, session_id: str, holder: str) -> None:
        self.calls.append("release_lease")
        if self.lease.get(session_id) == holder:
            del self.lease[session_id]

    def _project(self, session_id: str, cap: int) -> Projection | None:
        header = self.sessions.get(session_id)
        if header is None:
            return None
        return project(header, self.rows[session_id][: cap + 1], cap)

    def read_projection(self, session_id: str, cap: int) -> Projection | None:
        self.calls.append("read_projection")
        return self._project(session_id, cap)

    def archive_rows(
        self,
        session_id: str,
        holder: str,
        conversation_id: str,
        rows: tuple[VoiceRow, ...],
        expected_committed: Fingerprint,
        expected_pending: Fingerprint,
        cap: int,
        lease_ttl_seconds: float,
    ) -> ArchivePlan:
        self.calls.append("archive_rows")
        if self.lease.get(session_id) != holder:
            raise ArchiveRefusal("lease_lost")
        header = self.sessions.get(session_id)
        if header is not None and header["end_reason"] == "compression":
            raise ArchiveRefusal("rotated")
        plan = plan_archive(
            self._project(session_id, cap),
            conversation_id,
            rows,
            expected_committed,
            expected_pending,
            cap,
        )
        before = list(self.rows[session_id])
        for row in plan.inserts:
            self.rows[session_id].append(expected_row_values(conversation_id, row) | self.drift)
        if self.after_insert is not None:
            self.after_insert()
        after = self._project(session_id, cap)
        if after is None or after.fingerprint() != expected_pending:
            self.rows[session_id] = before
            raise ArchiveRefusal("drift")
        return plan


def _rows(start: int, stop: int, *, generation: int = 0, text: str = "row") -> tuple[VoiceRow, ...]:
    return tuple(
        VoiceRow(
            identity=Identity(generation, seq),
            role="user" if seq % 2 == 0 else "assistant",
            text=f"{text} {seq}",
            interrupted=seq % 4 == 3,
            timestamp=1_700_000_000.0 + seq,
        )
        for seq in range(start, stop)
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CompanionStore]:
    opened = CompanionStore(tmp_path / "companion.db")
    try:
        yield opened
    finally:
        opened.close()


def _archive(
    store: CompanionStore,
    hermes: FakeHermes,
    *,
    cap: int = 4096,
    ttl: float = _TTL,
    sleep: Callable[[float], object] | None = None,
    max_conversations: int = 16,
) -> VoiceArchive:
    extra: dict[str, object] = {} if sleep is None else {"sleep": sleep}
    return VoiceArchive(
        store,
        hermes,
        cap=cap,
        lease_ttl_seconds=ttl,
        max_conversations=max_conversations,
        **extra,  # type: ignore[arg-type]
    )


def _markers(output: str, prefix: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(prefix))
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


async def _open(store: CompanionStore, hermes: FakeHermes, **options: object) -> VoiceArchive:
    archive = _archive(store, hermes, **options)  # type: ignore[arg-type]
    await archive.open(_CONVERSATION)
    return archive


async def _refused(awaitable: object) -> str:
    with pytest.raises(ArchiveRefusal) as refusal:
        await awaitable  # type: ignore[misc]
    return refusal.value.category


def _stored(store: CompanionStore) -> tuple[Progress | None, Progress | None, str | None]:
    record = store.read(_CONVERSATION)
    assert record is not None
    return record.committed, record.pending, record.quarantine


# --- opening -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opening_creates_the_session_under_a_fresh_recorded_lease(
    store: CompanionStore, tmp_path: Path
) -> None:
    hermes = FakeHermes()
    archive = _archive(store, hermes)
    seen_holders: list[str | None] = []

    def on_acquire(holder: str) -> None:
        # Hermes runs on a worker thread; read what the store has durably recorded.
        with contextlib.closing(sqlite3.connect(tmp_path / "companion.db")) as raw:
            (recorded,) = raw.execute("SELECT holder FROM voice_archive").fetchone()
        seen_holders.append(recorded)

    hermes.on_acquire = on_acquire
    try:
        report = await archive.open(_CONVERSATION)
        assert report == OpenReport(recovery="created")
        assert archive.ready(_CONVERSATION) is True
        holder = archive.holder(_CONVERSATION)
        assert re.fullmatch(rf"pid={os.getpid()}:voice=conv:boot=[0-9a-f]{{32}}", holder)
        # The holder is durable before it is taken, and it is the one Hermes records.
        assert seen_holders == [holder]
        assert hermes.lease == {hermes.only(): holder}
        committed, pending, quarantine = _stored(store)
        assert committed == Progress(genesis(EXPECTED_HEADER), None)
        assert pending is None and quarantine is None
        assert hermes.calls == ["acquire_lease", "read_projection", "create_session",
                                "read_projection"]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_restarted_companion_never_reuses_a_holder(store: CompanionStore) -> None:
    hermes = FakeHermes()
    first = await _open(store, hermes)
    stale = first.holder(_CONVERSATION)
    await first.close()
    second = await _open(store, hermes)
    try:
        assert second.holder(_CONVERSATION) != stale
        record = store.read(_CONVERSATION)
        assert record is not None and record.holder == second.holder(_CONVERSATION)
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_a_quarantined_conversation_is_refused_before_any_lease(
    store: CompanionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    await archive.close()
    store.quarantine(_CONVERSATION, "mismatch")
    hermes.calls.clear()
    reopened = _archive(store, hermes)
    assert await _refused(reopened.open(_CONVERSATION)) == "quarantined"
    assert hermes.calls == []
    assert reopened.ready(_CONVERSATION) is False
    assert _markers(capsys.readouterr().out, "[voice-archive-open] ") == [
        {"refusal": "quarantined", "version": 1}
    ]


@pytest.mark.asyncio
async def test_a_tombstoned_conversation_is_refused_before_any_lease(
    store: CompanionStore, tmp_path: Path
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    await archive.close()
    with contextlib.closing(sqlite3.connect(tmp_path / "companion.db")) as raw:
        raw.execute("UPDATE voice_archive SET tombstone = 0")
        raw.commit()
    hermes.calls.clear()
    assert await _refused(_archive(store, hermes).open(_CONVERSATION)) == "tombstoned"
    assert hermes.calls == []


@pytest.mark.asyncio
async def test_a_lease_held_elsewhere_is_refused_before_verification(
    store: CompanionStore,
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    await archive.close()
    hermes.lease[hermes.only()] = "pid=1:foreign"
    hermes.calls.clear()
    reopened = _archive(store, hermes)
    assert await _refused(reopened.open(_CONVERSATION)) == "lease_held"
    assert hermes.calls == ["acquire_lease"]
    assert reopened.ready(_CONVERSATION) is False


@pytest.mark.asyncio
async def test_a_conversation_opens_once(store: CompanionStore) -> None:
    archive = await _open(store, FakeHermes())
    try:
        assert await _refused(archive.open(_CONVERSATION)) == "bound"
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_conversations_are_bounded(store: CompanionStore) -> None:
    archive = _archive(store, FakeHermes(), max_conversations=1)
    try:
        await archive.open("one")
        assert await _refused(archive.open("two")) == "conversations"
    finally:
        await archive.close()


# --- archiving -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_batch_is_written_ahead_committed_and_acknowledged(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        ack = await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        assert ack == ArchiveAck(_CONVERSATION, 0, 3, inserted=4, already_applied=False)
        committed, pending, _ = _stored(store)
        projection = hermes._project(hermes.only(), 4096)
        assert projection is not None
        assert committed == Progress(projection.fingerprint(), Identity(0, 3))
        assert pending is None
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_sequential_retries_verify_and_never_append(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        for _ in range(3):
            ack = await archive.archive(_CONVERSATION, 0, _rows(0, 4))
            assert ack.inserted == 0 and ack.already_applied is True
        assert len(hermes.rows[hermes.only()]) == 4
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_concurrent_retries_leave_one_record_per_identity(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        acks = await asyncio.gather(
            *(archive.archive(_CONVERSATION, 0, _rows(0, 4)) for _ in range(4))
        )
        assert sorted(ack.inserted for ack in acks) == [0, 0, 0, 4]
        identities = [row["platform_message_id"] for row in hermes.rows[hermes.only()]]
        assert len(identities) == len(set(identities)) == 4
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_an_overlapping_batch_appends_only_its_new_rows(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        ack = await archive.archive(_CONVERSATION, 0, _rows(2, 6))
        assert ack.inserted == 2 and ack.seq_through == 5
        assert len(hermes.rows[hermes.only()]) == 6
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_conflicting_payload_is_refused_without_mutation(
    store: CompanionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        before = list(hermes.rows[hermes.only()])
        stored = _stored(store)
        capsys.readouterr()
        conflicting = _rows(0, 4, text="other")
        assert await _refused(archive.archive(_CONVERSATION, 0, conflicting)) == "conflict"
        assert hermes.rows[hermes.only()] == before
        assert _stored(store) == stored
        assert _markers(capsys.readouterr().out, "[voice-archive] ") == [
            {"refusal": "conflict", "rows": 4, "version": 1}
        ]
        # A refused payload does not fence the conversation.
        assert (await archive.archive(_CONVERSATION, 0, _rows(4, 6))).inserted == 2
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_batch_with_a_gap_is_refused_before_anything_is_written(
    store: CompanionStore,
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        stored = _stored(store)
        hermes.calls.clear()
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(5, 6))) == "identity"
        assert hermes.calls == []
        assert _stored(store) == stored
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_batch_past_the_cap_is_refused_and_the_conversation_stays_ready(
    store: CompanionStore,
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes, cap=5)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        stored = _stored(store)
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == "capacity"
        assert _stored(store) == stored
        assert len(hermes.rows[hermes.only()]) == 4
        assert archive.ready(_CONVERSATION) is True
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_vanished_store_record_is_refused(store: CompanionStore, tmp_path: Path) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        with contextlib.closing(sqlite3.connect(tmp_path / "companion.db")) as raw:
            raw.execute("DELETE FROM voice_archive")
            raw.commit()
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(0, 1))) == "unbound"
        assert hermes.rows[hermes.only()] == []
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_archiving_requires_an_open_conversation(store: CompanionStore) -> None:
    archive = _archive(store, FakeHermes())
    assert await _refused(archive.archive(_CONVERSATION, 0, _rows(0, 1))) == "not_ready"


# --- foreign mutation and quarantine -------------------------------------------------------


def _tamper_content(hermes: FakeHermes) -> None:
    rows = hermes.rows[hermes.only()]
    rows[1] = rows[1] | {"content": "changed"}


def _tamper_order(hermes: FakeHermes) -> None:
    rows = hermes.rows[hermes.only()]
    rows[0], rows[2] = rows[2], rows[0]


def _append_foreign(hermes: FakeHermes) -> None:
    hermes.rows[hermes.only()].append(expected_row_values("foreign", _rows(0, 1)[0]))


def _delete(hermes: FakeHermes) -> None:
    session_id = hermes.only()
    del hermes.sessions[session_id]
    hermes.rows[session_id] = []


def _rotate(hermes: FakeHermes) -> None:
    hermes.sessions[hermes.only()] |= {"ended_at": 5.0, "end_reason": "compression"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "category"),
    [
        (_tamper_content, "mismatch"),
        (_tamper_order, "mismatch"),
        (_append_foreign, "mismatch"),
        (_delete, "missing"),
        (_rotate, "rotated"),
    ],
    ids=["content", "order", "append", "delete", "rotate"],
)
async def test_a_foreign_mutation_is_quarantined_durably_before_the_refusal(
    store: CompanionStore,
    tamper: Callable[[FakeHermes], None],
    category: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        tamper(hermes)
        snapshot = {key: list(value) for key, value in hermes.rows.items()}
        capsys.readouterr()
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == category
        assert hermes.rows == snapshot
        assert _stored(store)[2] == category
        assert archive.ready(_CONVERSATION) is False
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == "not_ready"
        assert _markers(capsys.readouterr().out, "[voice-archive] ") == [
            {"refusal": category, "rows": 2, "version": 1},
            {"refusal": "not_ready", "rows": 2, "version": 1},
        ]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_an_unpersistable_quarantine_fences_the_whole_companion(
    store: CompanionStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        _tamper_content(hermes)

        def fail(conversation_id: str, category: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(store, "quarantine", fail)
        capsys.readouterr()
        with pytest.raises(RuntimeError):
            await archive.archive(_CONVERSATION, 0, _rows(4, 6))
        assert archive.ready(_CONVERSATION) is False
        assert await _refused(archive.open("another")) == "fenced"
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == "fenced"
        markers = _markers(capsys.readouterr().out, "[voice-archive] ")
        assert markers[0] == {"failure": "RuntimeError", "rows": 2, "version": 1}
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_lost_lease_refuses_clears_pending_and_fences(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        stored = _stored(store)
        hermes.lease[hermes.only()] = "pid=1:foreign"
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == "lease_lost"
        assert _stored(store) == stored
        assert archive.ready(_CONVERSATION) is False
        assert len(hermes.rows[hermes.only()]) == 4
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_storage_that_drifts_from_the_prediction_is_refused_and_fences(
    store: CompanionStore,
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        stored = _stored(store)
        hermes.drift = {"display_kind": "voice"}
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == "drift"
        assert _stored(store) == stored
        assert len(hermes.rows[hermes.only()]) == 4
        assert archive.ready(_CONVERSATION) is False
    finally:
        await archive.close()


# --- crash recovery ------------------------------------------------------------------------


def _restart(hermes: FakeHermes) -> None:
    """The old process died: its lease is reclaimable, as Hermes does for a dead PID."""
    hermes.lease.clear()


@pytest.mark.asyncio
async def test_an_unknown_outcome_keeps_pending_and_fences(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))

        def crash() -> None:
            raise RuntimeError("connection reset after commit")

        hermes.after_insert = crash
        with pytest.raises(RuntimeError):
            await archive.archive(_CONVERSATION, 0, _rows(4, 6))
        _, pending, _ = _stored(store)
        assert pending is not None and pending.cursor == Identity(0, 5)
        assert archive.ready(_CONVERSATION) is False
    finally:
        await archive.close()


async def _crash_with_pending(store: CompanionStore, hermes: FakeHermes, applied: bool) -> None:
    archive = await _open(store, hermes)
    await archive.archive(_CONVERSATION, 0, _rows(0, 4))

    def crash() -> None:
        if not applied:
            hermes.rows[hermes.only()] = hermes.rows[hermes.only()][:4]
        raise RuntimeError("process died")

    hermes.after_insert = crash
    with pytest.raises(RuntimeError):
        await archive.archive(_CONVERSATION, 0, _rows(4, 6))
    hermes.after_insert = None
    await archive.close()
    _restart(hermes)


@pytest.mark.asyncio
async def test_recovery_clears_pending_the_archive_never_reached(store: CompanionStore) -> None:
    hermes = FakeHermes()
    await _crash_with_pending(store, hermes, applied=False)
    archive = _archive(store, hermes)
    try:
        assert await archive.open(_CONVERSATION) == OpenReport(recovery="cleared")
        committed, pending, _ = _stored(store)
        assert pending is None and committed is not None and committed.cursor == Identity(0, 3)
        assert (await archive.archive(_CONVERSATION, 0, _rows(4, 6))).inserted == 2
        assert len(hermes.rows[hermes.only()]) == 6
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_recovery_promotes_pending_the_archive_already_holds(store: CompanionStore) -> None:
    hermes = FakeHermes()
    await _crash_with_pending(store, hermes, applied=True)
    archive = _archive(store, hermes)
    try:
        assert await archive.open(_CONVERSATION) == OpenReport(recovery="promoted")
        committed, pending, _ = _stored(store)
        assert pending is None and committed is not None and committed.cursor == Identity(0, 5)
        ack = await archive.archive(_CONVERSATION, 0, _rows(4, 6))
        assert ack.inserted == 0 and ack.already_applied is True
        assert len(hermes.rows[hermes.only()]) == 6
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_recovery_quarantines_an_archive_matching_neither(store: CompanionStore) -> None:
    hermes = FakeHermes()
    await _crash_with_pending(store, hermes, applied=False)
    _append_foreign(hermes)
    assert await _refused(_archive(store, hermes).open(_CONVERSATION)) == "recovery"
    assert _stored(store)[2] == "recovery"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "category"),
    [(_tamper_content, "mismatch"), (_delete, "missing")],
    ids=["content", "delete"],
)
async def test_startup_verifies_against_committed_and_never_recreates(
    store: CompanionStore, tamper: Callable[[FakeHermes], None], category: str
) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    await archive.archive(_CONVERSATION, 0, _rows(0, 4))
    await archive.close()
    tamper(hermes)
    hermes.calls.clear()
    reopened = _archive(store, hermes)
    assert await _refused(reopened.open(_CONVERSATION)) == category
    assert "create_session" not in hermes.calls
    assert _stored(store)[2] == category
    assert reopened.ready(_CONVERSATION) is False
    # A failed open does not keep the lease it took.
    assert hermes.lease == {}


@pytest.mark.asyncio
async def test_startup_quarantines_an_archive_over_the_cap(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes, cap=4)
    await archive.archive(_CONVERSATION, 0, _rows(0, 4))
    await archive.close()
    _append_foreign(hermes)
    assert await _refused(_archive(store, hermes, cap=4).open(_CONVERSATION)) == "over_cap"
    assert _stored(store)[2] == "over_cap"


@pytest.mark.asyncio
async def test_an_interrupted_creation_is_completed(store: CompanionStore) -> None:
    hermes = FakeHermes()
    store.bind(_CONVERSATION, "voice_session", Progress(genesis(EXPECTED_HEADER), None))
    hermes.create_session("voice_session")
    hermes.calls.clear()
    archive = _archive(store, hermes)
    try:
        assert await archive.open(_CONVERSATION) == OpenReport(recovery="promoted")
        assert "create_session" not in hermes.calls
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_a_creation_colliding_with_a_foreign_session_is_quarantined(
    store: CompanionStore,
) -> None:
    hermes = FakeHermes()
    store.bind(_CONVERSATION, "voice_session", Progress(genesis(EXPECTED_HEADER), None))
    hermes.create_session("voice_session")
    hermes.sessions["voice_session"]["source"] = "cli"
    assert await _refused(_archive(store, hermes).open(_CONVERSATION)) == "recovery"
    assert _stored(store)[2] == "recovery"


# --- lease refresh -------------------------------------------------------------------------


class _Clock:
    """A sleep that records its intervals and lets each wait pass at once."""

    def __init__(self) -> None:
        self.intervals: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.intervals.append(seconds)
        await asyncio.sleep(0)


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_the_lease_is_refreshed_every_third_of_its_ttl(store: CompanionStore) -> None:
    hermes = FakeHermes()
    clock = _Clock()
    archive = await _open(store, hermes, sleep=clock)
    try:
        await _until(lambda: hermes.calls.count("refresh_lease") >= 3)
        assert set(clock.intervals) == {_TTL / 3}
        assert archive.ready(_CONVERSATION) is True
    finally:
        await archive.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "cause"),
    [(False, "lost"), (OSError("disk I/O error"), "raised")],
    ids=["false", "raised"],
)
async def test_a_failed_refresh_fences_all_work(
    store: CompanionStore,
    result: bool | Exception,
    cause: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hermes = FakeHermes()
    clock = _Clock()
    archive = await _open(store, hermes, sleep=clock)
    try:
        await archive.archive(_CONVERSATION, 0, _rows(0, 4))
        hermes.refresh_result = result
        await _until(lambda: not archive.ready(_CONVERSATION))
        assert await _refused(archive.archive(_CONVERSATION, 0, _rows(4, 6))) == "not_ready"
        assert len(hermes.rows[hermes.only()]) == 4
        assert _markers(capsys.readouterr().out, "[voice-archive-lease] ") == [
            {"fence": cause, "version": 1}
        ]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_closing_releases_the_lease_and_stops_work(store: CompanionStore) -> None:
    hermes = FakeHermes()
    archive = await _open(store, hermes)
    await archive.close()
    assert hermes.lease == {}
    assert archive.ready(_CONVERSATION) is False
    assert await _refused(archive.archive(_CONVERSATION, 0, _rows(0, 1))) == "not_ready"


def test_archive_options_are_exact_and_bounded(store: CompanionStore) -> None:
    hermes = FakeHermes()
    for options in (
        {"cap": 0},
        {"cap": 4097},
        {"lease_ttl_seconds": 0.0},
        {"lease_ttl_seconds": float("inf")},
        {"max_conversations": 0},
    ):
        with pytest.raises(ValueError):
            VoiceArchive(store, hermes, **options)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VoiceArchive(store, hermes, cap=True)
    with pytest.raises(TypeError):
        VoiceArchive(object(), hermes)  # type: ignore[arg-type]
