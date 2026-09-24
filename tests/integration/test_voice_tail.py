from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.integration import run_record as run_record_module
from hermes_realtime.integration import voice_tail as voice_tail_module
from hermes_realtime.integration.voice_tail import (
    VoiceTailWriter,
    max_voice_tail_bytes,
    parse_voice_tail,
    voice_tail_bytes,
)
from hermes_realtime.speech import Transcript

_MARKER = "[voice-tail] "
_LOCK_MARKER = "[voice-tail-lock] "


def _rows(*rows: tuple[str, str, bool], prior_work: bool = False) -> DurableConversation:
    return DurableConversation(
        messages=tuple(
            ConversationMessage(role, text, interrupted) for role, text, interrupted in rows
        ),
        prior_work=prior_work,
    )


def _markers(output: str, prefix: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(prefix)]


def _writer(path: Path, **kwargs: float) -> VoiceTailWriter:
    return VoiceTailWriter(path, **kwargs)


def _parse(raw: bytes) -> DurableConversation | None:
    return parse_voice_tail(raw, max_messages=16, max_item_chars=64)


def _orphan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(b"orphaned plaintext")
    return Path(temporary)


def test_the_tail_is_sorted_compact_versioned_json() -> None:
    tail = _rows(("user", "Hi é", False), ("assistant", "Cut", True), prior_work=True)

    assert voice_tail_bytes(tail) == (
        b'{"messages":[{"interrupted":false,"role":"user","text":"Hi \\u00e9"},'
        b'{"interrupted":true,"role":"assistant","text":"Cut"}],"prior_work":true,"version":1}'
    )
    assert _parse(voice_tail_bytes(tail)) == tail
    assert _parse(voice_tail_bytes(_rows())) == _rows()


def test_the_byte_bound_admits_a_worst_case_tail_at_the_stores_bounds() -> None:
    # One astral character is one str character but twelve escaped bytes.
    tail = DurableConversation(
        messages=tuple(
            ConversationMessage("assistant", "\U0001f600" * 8, interrupted=True) for _ in range(3)
        ),
        prior_work=False,
    )
    raw = voice_tail_bytes(tail)
    bound = max_voice_tail_bytes(3, 8)

    def parse(data: bytes) -> DurableConversation | None:
        return parse_voice_tail(data, max_messages=3, max_item_chars=8)

    assert len(raw) <= bound
    assert parse(raw) == tail
    assert parse(b" " * (bound + 1)) is None
    assert parse(raw + b" " * (bound + 1 - len(raw))) is None
    assert parse(raw + b" " * (bound - len(raw))) == tail


def _document(**overrides: object) -> bytes:
    document: dict[str, object] = {
        "messages": [{"interrupted": False, "role": "user", "text": "Hi"}],
        "prior_work": False,
        "version": 1,
    }
    document.update(overrides)
    return json.dumps(document).encode("utf-8")


def _row_document(**overrides: object) -> bytes:
    row: dict[str, object] = {"interrupted": False, "role": "user", "text": "Hi"}
    row.update(overrides)
    return _document(messages=[row])


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe{}", id="not-utf8"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(b"[" * 5_000 + b"]" * 5_000, id="nested-past-the-recursion-limit"),
        pytest.param(
            b'{"messages":[],"prior_work":false,"version":1,"version":1}',
            id="duplicate-top-level-key",
        ),
        pytest.param(
            b'{"messages":[{"interrupted":false,"role":"user","role":"user","text":"Hi"}],'
            b'"prior_work":false,"version":1}',
            id="duplicate-row-key",
        ),
        pytest.param(_document(extra=1), id="extra-top-level-key"),
        pytest.param(
            json.dumps({"prior_work": False, "version": 1}).encode(), id="missing-messages"
        ),
        pytest.param(
            json.dumps({"messages": [], "prior_work": False}).encode(), id="missing-version"
        ),
        pytest.param(
            json.dumps({"messages": [], "version": 1}).encode(), id="missing-prior-work"
        ),
        pytest.param(_document(prior_work=0), id="prior-work-not-a-boolean"),
        pytest.param(_document(version=2), id="unknown-version"),
        pytest.param(_document(version="1"), id="string-version"),
        pytest.param(_document(version=1.0), id="float-version"),
        pytest.param(_document(version=True), id="boolean-version"),
        pytest.param(_document(messages={}), id="messages-not-a-list"),
        pytest.param(_document(messages=["row"]), id="row-not-an-object"),
        pytest.param(_row_document(extra=1), id="extra-row-key"),
        pytest.param(
            _document(messages=[{"role": "user", "text": "Hi"}]),
            id="missing-row-key",
        ),
        pytest.param(_row_document(role=1), id="role-not-a-string"),
        pytest.param(_row_document(text=["Hi"]), id="text-not-a-string"),
        pytest.param(_row_document(interrupted=0), id="interrupted-not-a-boolean"),
        pytest.param(_row_document(role="system"), id="unknown-role"),
        pytest.param(_row_document(interrupted=True), id="interrupted-user-row"),
        pytest.param(_row_document(text="   "), id="blank-text"),
        pytest.param(_row_document(text="said deleg_private"), id="private-run-token"),
        pytest.param(_row_document(text="lone \ud800 surrogate"), id="not-utf8-encodable"),
        pytest.param(
            _document(messages=[{"interrupted": False, "role": "user", "text": "Hi"}] * 17),
            id="more-rows-than-max-messages",
        ),
    ],
)
def test_every_malformed_class_is_refused(raw: bytes) -> None:
    assert _parse(raw) is None


@pytest.mark.asyncio
async def test_open_restores_the_tail_before_anything_else_and_reports_only_a_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "state" / "voice-tail-v1.json"
    tail = _rows(
        ("user", "Earlier question", False),
        ("assistant", "Earlier answer", True),
        prior_work=True,
    )
    run_record_module.write_run_record(path, voice_tail_bytes(tail))
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)

    await writer.open(store)
    try:
        assert store.snapshot().messages == tail.messages
        assert store.snapshot().revision == 1
        assert store.snapshot().terminal_task_count == 1
        assert store.snapshot().active_tasks == ()
        output = capsys.readouterr().out
        assert _markers(output, _MARKER) == [_MARKER + '{"restored":2,"version":1}']
        assert "Earlier" not in output
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_open_without_a_tail_starts_empty_and_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)

    await writer.open(store)
    await writer.close()

    assert store.snapshot().messages == ()
    assert store.snapshot().revision == 0
    assert _markers(capsys.readouterr().out, _MARKER) == []
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"{not json", id="unparseable"),
        pytest.param(
            voice_tail_bytes(_rows(("user", "x" * 65, False))),
            id="outside-the-stores-bounds",
        ),
    ],
)
async def test_a_refused_tail_starts_a_fresh_conversation_with_one_marker(
    raw: bytes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    path.write_bytes(raw)
    writer = _writer(path)
    store = ConversationContextStore(max_item_chars=64, on_change=writer.update)

    await writer.open(store)
    try:
        assert store.snapshot().messages == ()
        assert store.snapshot().revision == 0
        assert path.read_bytes() == raw
        output = capsys.readouterr().out
        assert _markers(output, _MARKER) == [_MARKER + '{"refusal":"malformed","version":1}']
        assert "x" * 65 not in output

        store.record_user_transcript(Transcript(text="Fresh start", final=True))
    finally:
        await writer.close()

    assert _parse(path.read_bytes()) == _rows(("user", "Fresh start", False))
    assert _markers(capsys.readouterr().out, _MARKER) == []


@pytest.mark.asyncio
async def test_open_removes_orphaned_plaintext_temporaries_after_locking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    orphan = _orphan(path)
    writer = _writer(path)

    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        assert not orphan.exists()
    finally:
        await writer.close()


@pytest.mark.asyncio
async def test_a_scanner_held_orphan_never_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    tail = _rows(("user", "Kept", False))
    run_record_module.write_run_record(path, voice_tail_bytes(tail))
    held = _orphan(path)
    real_unlink = Path.unlink

    def scanner_held(self: Path, missing_ok: bool = False) -> None:
        if self == held:
            raise PermissionError(13, "sharing violation")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", scanner_held)
    writer = _writer(path)
    store = ConversationContextStore(on_change=writer.update)

    with caplog.at_level(logging.WARNING, logger=run_record_module.__name__):
        await writer.open(store)
    await writer.close()

    assert store.snapshot().messages == tail.messages
    assert held.exists()
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings == ["orphaned temporary could not be removed (PermissionError)"]


@pytest.mark.asyncio
async def test_an_unreadable_tail_fails_start_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    tail = _rows(("user", "Kept", False))
    run_record_module.write_run_record(path, voice_tail_bytes(tail))

    def sharing_violation(_path: Path, _max_bytes: int) -> bytes | None:
        raise PermissionError(13, "sharing violation")

    monkeypatch.setattr(voice_tail_module, "read_run_record", sharing_violation)
    failed = _writer(path)
    failed_store = ConversationContextStore(on_change=failed.update)
    with pytest.raises(PermissionError):
        await failed.open(failed_store)
    await failed.close()
    monkeypatch.undo()

    assert failed_store.snapshot().revision == 0
    assert _markers(capsys.readouterr().out, _MARKER) == []
    retry = _writer(path)
    retry_store = ConversationContextStore(on_change=retry.update)
    await retry.open(retry_store)
    await retry.close()
    assert retry_store.snapshot().messages == tail.messages


@pytest.mark.asyncio
async def test_a_second_host_cannot_open_a_live_hosts_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "voice-tail-v1.json"
    tail = _rows(("user", "Owned", False))
    run_record_module.write_run_record(path, voice_tail_bytes(tail))
    first = _writer(path)
    await first.open(ConversationContextStore(on_change=first.update))
    in_flight = _orphan(path)
    capsys.readouterr()
    second = _writer(path)
    second_store = ConversationContextStore(on_change=second.update)

    try:
        with pytest.raises(RuntimeError, match="another host holds the voice tail"):
            await second.open(second_store)

        assert second_store.snapshot().revision == 0
        assert in_flight.exists()
        output = capsys.readouterr().out
        assert _markers(output, _LOCK_MARKER) == [_LOCK_MARKER + '{"cause":"held","version":1}']
        assert _markers(output, _MARKER) == []
        await second.close()
        assert _parse(path.read_bytes()) == tail
    finally:
        await first.close()
    in_flight.unlink()

    third = _writer(path)
    third_store = ConversationContextStore(on_change=third.update)
    await third.open(third_store)
    await third.close()
    assert third_store.snapshot().messages == tail.messages


class _GatedWrites:
    """Replace the atomic write with one that records bytes and can block or fail."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.written: list[bytes] = []
        self.failures: list[BaseException] = []
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        monkeypatch.setattr(voice_tail_module, "write_run_record", self._write)

    def _write(self, path: Path, data: bytes) -> None:
        self.started.set()
        self.release.wait(timeout=5)
        if self.failures:
            raise self.failures.pop(0)
        self.written.append(data)
        run_record_module.write_run_record(path, data)


async def _until(condition: object, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():  # type: ignore[operator]
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_updates_coalesce_and_the_latest_snapshot_always_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    first = _rows(("user", "one", False))
    writes.release.clear()
    writer.update(first)
    await asyncio.to_thread(writes.started.wait, 2)

    for text in ("two", "three", "four"):
        writer.update(_rows(("user", text, False)))
    writes.release.set()
    await _until(lambda: len(writes.written) == 2)
    await asyncio.sleep(0.05)

    latest = voice_tail_bytes(_rows(("user", "four", False)))
    assert writes.written == [voice_tail_bytes(first), latest]
    await writer.close()
    assert len(writes.written) == 2
    assert path.read_bytes() == latest


def test_update_only_stores_the_latest_snapshot_synchronously(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "voice-tail-v1.json")

    writer.update(_rows(("user", "one", False)))
    writer.update(_rows(("user", "two", False)))

    assert not (tmp_path / "voice-tail-v1.json").exists()
    with pytest.raises(TypeError):
        writer.update(_rows(("user", "x", False)).messages)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_sharing_violation_backs_off_and_retries_with_the_latest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    writes = _GatedWrites(monkeypatch)
    writes.failures.extend([PermissionError(13, "sharing violation"), PermissionError(13, "x")])
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path, initial_backoff_seconds=0.01, max_backoff_seconds=0.02)
    await writer.open(ConversationContextStore(on_change=writer.update))

    with caplog.at_level(logging.WARNING, logger=voice_tail_module.__name__):
        writer.update(_rows(("user", "private words", False)))
        await _until(lambda: len(writes.failures) == 0)
        writer.update(_rows(("user", "newer words", False)))
        await _until(lambda: len(writes.written) >= 1)
        await asyncio.sleep(0.05)

    assert writes.written[-1] == voice_tail_bytes(_rows(("user", "newer words", False)))
    stale = voice_tail_bytes(_rows(("user", "private words", False)))
    assert all(data != stale for data in writes.written[1:])
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("words" not in record.getMessage() for record in warnings)
    assert all(str(tmp_path) not in record.getMessage() for record in warnings)
    await writer.close()
    assert path.read_bytes() == voice_tail_bytes(_rows(("user", "newer words", False)))


@pytest.mark.asyncio
async def test_a_non_os_write_failure_is_retried_and_never_kills_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    writes = _GatedWrites(monkeypatch)
    writes.failures.append(ValueError("serializer bug"))
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path, initial_backoff_seconds=0.01, max_backoff_seconds=0.02)
    await writer.open(ConversationContextStore(on_change=writer.update))

    with caplog.at_level(logging.WARNING, logger=voice_tail_module.__name__):
        writer.update(_rows(("user", "one", False)))
        await _until(lambda: len(writes.written) == 1)
    writer.update(_rows(("user", "two", False)))
    await _until(lambda: len(writes.written) == 2)
    await writer.close()

    assert path.read_bytes() == voice_tail_bytes(_rows(("user", "two", False)))
    assert [record.getMessage() for record in caplog.records] == [
        "voice tail write failed (ValueError); retrying in 0.01 s"
    ]


@pytest.mark.asyncio
async def test_the_backoff_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _GatedWrites(monkeypatch)
    writes.failures.extend(PermissionError(13, "sharing violation") for _ in range(8))
    delays: list[float] = []

    async def recording_backoff(delay: float) -> None:
        delays.append(delay)
        await asyncio.sleep(0)

    writer = _writer(tmp_path / "voice-tail-v1.json")
    monkeypatch.setattr(writer, "_wait_backoff", recording_backoff)
    await writer.open(ConversationContextStore(on_change=writer.update))

    writer.update(_rows(("user", "one", False)))
    await _until(lambda: len(writes.written) == 1)
    # A success resets the backoff for the next failure.
    writes.failures.append(PermissionError(13, "sharing violation"))
    writer.update(_rows(("user", "two", False)))
    await _until(lambda: len(writes.written) == 2)
    await writer.close()

    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0, 0.05]


@pytest.mark.asyncio
async def test_close_flushes_the_final_dirty_snapshot_and_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writes.release.clear()
    writer.update(_rows(("assistant", "Cut", False)))
    await asyncio.to_thread(writes.started.wait, 2)
    writer.update(_rows(("assistant", "Cut", True)))

    closing = asyncio.create_task(writer.close())
    await asyncio.sleep(0.02)
    writes.release.set()
    await asyncio.wait_for(closing, timeout=2)

    assert writes.written[-1] == voice_tail_bytes(_rows(("assistant", "Cut", True)))
    assert path.read_bytes() == voice_tail_bytes(_rows(("assistant", "Cut", True)))
    next_owner = _writer(path)
    await next_owner.open(ConversationContextStore(on_change=next_owner.update))
    await next_owner.close()


@pytest.mark.asyncio
async def test_a_failed_final_write_backs_off_fully_and_retries_before_close_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    delays: list[float] = []
    writer = _writer(path, initial_backoff_seconds=0.05, max_backoff_seconds=0.05)
    real_backoff = writer._wait_backoff

    async def recording_backoff(delay: float) -> None:
        delays.append(delay)
        started = asyncio.get_running_loop().time()
        await real_backoff(delay)
        delays.append(round(asyncio.get_running_loop().time() - started, 2))

    monkeypatch.setattr(writer, "_wait_backoff", recording_backoff)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writes.failures.append(PermissionError(13, "sharing violation"))
    writer.update(_rows(("assistant", "Final", True)))

    await asyncio.wait_for(writer.close(), timeout=2)

    assert writes.written == [voice_tail_bytes(_rows(("assistant", "Final", True)))]
    assert delays[0] == 0.05
    # Close never cuts the backoff short.
    assert delays[1] >= 0.04
    assert path.read_bytes() == voice_tail_bytes(_rows(("assistant", "Final", True)))


@pytest.mark.asyncio
async def test_a_close_timeout_keeps_the_tail_owned_and_a_retried_close_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path, close_timeout_seconds=0.05)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writes.release.clear()
    writer.update(_rows(("user", "slow", False)))
    await asyncio.to_thread(writes.started.wait, 2)

    with pytest.raises(RuntimeError, match="close timeout"):
        await writer.close()

    contender = _writer(path)
    with pytest.raises(RuntimeError, match="another host holds the voice tail"):
        await contender.open(ConversationContextStore(on_change=contender.update))
    writes.release.set()
    await writer.close()

    assert path.read_bytes() == voice_tail_bytes(_rows(("user", "slow", False)))
    successor = _writer(path)
    await successor.open(ConversationContextStore(on_change=successor.update))
    await successor.close()


@pytest.mark.asyncio
async def test_close_waits_for_an_in_flight_write_so_the_newest_lands_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[bytes] = []
    first_started = threading.Event()
    release_first = threading.Event()
    first_finished = threading.Event()

    def write(path: Path, data: bytes) -> None:
        first = not first_started.is_set()
        if first:
            first_started.set()
            release_first.wait(timeout=5)
        written.append(data)
        run_record_module.write_run_record(path, data)
        if first:
            first_finished.set()

    monkeypatch.setattr(voice_tail_module, "write_run_record", write)
    path = tmp_path / "voice-tail-v1.json"
    writer = _writer(path)
    await writer.open(ConversationContextStore(on_change=writer.update))
    writer.update(_rows(("user", "stale", False)))
    await asyncio.to_thread(first_started.wait, 2)
    writer.update(_rows(("user", "newest", False)))

    closing = asyncio.create_task(writer.close())
    await asyncio.sleep(0.05)
    release_first.set()
    await asyncio.wait_for(closing, timeout=2)
    # Every write has landed before the assertions, whichever order they took.
    assert await asyncio.to_thread(first_finished.wait, 2)

    assert written[-1] == voice_tail_bytes(_rows(("user", "newest", False)))
    assert path.read_bytes() == voice_tail_bytes(_rows(("user", "newest", False)))


@pytest.mark.asyncio
async def test_close_without_a_change_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    writer = _writer(tmp_path / "voice-tail-v1.json")
    await writer.open(ConversationContextStore(on_change=writer.update))

    await writer.close()
    await writer.close()

    assert writes.written == []


@pytest.mark.asyncio
async def test_a_writer_that_never_opened_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = _GatedWrites(monkeypatch)
    writer = _writer(tmp_path / "voice-tail-v1.json")
    writer.update(_rows(("user", "before open", False)))

    await writer.close()

    assert writes.written == []
    assert not (tmp_path / "voice-tail-v1.json").exists()


@pytest.mark.asyncio
async def test_open_is_one_shot_and_requires_an_exact_store(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "voice-tail-v1.json")
    with pytest.raises(TypeError, match="store"):
        await writer.open(object())  # type: ignore[arg-type]
    await writer.open(ConversationContextStore(on_change=writer.update))
    try:
        with pytest.raises(RuntimeError, match="once"):
            await writer.open(ConversationContextStore())
    finally:
        await writer.close()


@pytest.mark.parametrize("path", ["voice-tail-v1.json", b"voice-tail-v1.json", 1])
def test_the_tail_path_must_be_an_exact_path(path: object) -> None:
    with pytest.raises(TypeError, match="path"):
        VoiceTailWriter(path)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"close_timeout_seconds": 0.0},
        {"close_timeout_seconds": -1.0},
        {"close_timeout_seconds": float("inf")},
        {"close_timeout_seconds": float("nan")},
    ],
)
def test_the_close_timeout_is_finite_and_positive(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="close timeout"):
        VoiceTailWriter(Path("voice-tail-v1.json"), **kwargs)
