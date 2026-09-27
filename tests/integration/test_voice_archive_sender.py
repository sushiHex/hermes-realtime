"""The realtime archive sender: one batch in flight, the cursor moves only on an ack."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.integration.bridge import BridgeProtocolError
from hermes_realtime.integration.run_record import read_run_record
from hermes_realtime.integration.voice_archive import VoiceArchiveSender
from hermes_realtime.integration.voice_tail import VoiceTailWriter, parse_voice_tail
from hermes_realtime.protocol import (
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
)
from hermes_realtime.speech import Transcript

_MARKER = "[voice-archive-send] "


class _Link:
    def __init__(self, companion: _Companion) -> None:
        self.companion = companion
        self.capabilities = companion.capabilities
        self.closed = False

    async def archive(self, event: VoiceArchiveEvent) -> Any:
        return await self.companion.answer(event)

    async def close(self) -> None:
        self.closed = True


class _Companion:
    """A scripted companion: answers ack by default; ``script`` overrides per send."""

    def __init__(self, capabilities: frozenset[str] = frozenset({"voice_archive"})) -> None:
        self.capabilities = capabilities
        self.sent: list[VoiceArchiveEvent] = []
        self.script: list[str] = []
        self.connects = 0
        self.links: list[_Link] = []
        self.hang = asyncio.Event()

    async def connect(self) -> _Link:
        self.connects += 1
        link = _Link(self)
        self.links.append(link)
        return link

    async def answer(self, event: VoiceArchiveEvent) -> Any:
        self.sent.append(event)
        action = self.script.pop(0) if self.script else "ack"
        fields = {
            "conversation_id": event.conversation_id,
            "generation": event.generation,
            "seq_from": event.seq_from,
            "seq_through": event.seq_through,
        }
        if action == "ack":
            return VoiceArchiveAckEvent(type="voice_archive_ack", **fields)
        if action == "wrong_ack":
            return VoiceArchiveAckEvent(
                type="voice_archive_ack", **(fields | {"seq_through": event.seq_through + 1})
            )
        if action == "refuse":
            return VoiceArchiveRefusedEvent(
                type="voice_archive_refused", category="quarantined", **fields
            )
        if action in ("not_ready", "lease_held", "conversations"):
            return VoiceArchiveRefusedEvent(type="voice_archive_refused", category=action, **fields)
        if action == "novel":
            # A category neither set lists (a newer companion): built past validation.
            return VoiceArchiveRefusedEvent.model_construct(
                protocol_version="0.2", type="voice_archive_refused", category="novel", **fields
            )
        if action == "drop":
            raise BridgeProtocolError("the companion closed before answering")
        if action == "hang":
            await self.hang.wait()
        raise AssertionError(action)


async def _no_sleep(_delay: float) -> None:
    await asyncio.sleep(0)


async def _setup(
    tmp_path: Path, companion: _Companion, **options: Any
) -> tuple[VoiceTailWriter, ConversationContextStore, VoiceArchiveSender]:
    writer = VoiceTailWriter(
        tmp_path / "tail.json", max_outbox_rows=16, max_batch_rows=2,
        conversation_ids=lambda: "conv",
    )
    store = ConversationContextStore(max_item_chars=64, on_change=writer.update)
    await writer.open(store)
    sender = VoiceArchiveSender(
        writer, companion.connect, reply_timeout=options.pop("reply_timeout", 5.0),
        sleep=options.pop("sleep", _no_sleep),
    )
    sender.start()
    return writer, store, sender


async def _until(condition: Any, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.005)


def _say(store: ConversationContextStore, *texts: str) -> None:
    for text in texts:
        store.record_user_transcript(Transcript(text=text, final=True))


def _cursor(tmp_path: Path) -> object:
    raw = read_run_record(tmp_path / "tail.json", 1 << 20)
    if raw is None:
        return "no tail yet"
    tail = parse_voice_tail(raw, max_messages=16, max_item_chars=64, max_outbox_rows=16)
    assert tail is not None and tail.archive is not None
    return tail.archive.cursor


def _markers(output: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(_MARKER))
        for line in output.splitlines()
        if line.startswith(_MARKER)
    ]


@pytest.mark.asyncio
async def test_batches_leave_one_at_a_time_and_the_cursor_moves_only_on_an_ack(
    tmp_path: Path,
) -> None:
    companion = _Companion()
    writer, store, sender = await _setup(tmp_path, companion)
    try:
        _say(store, "One", "Two", "Three")
        await _until(lambda: len(companion.sent) == 2)
        await _until(lambda: _cursor(tmp_path) == 2)
        assert [(e.seq_from, e.seq_through) for e in companion.sent] == [(0, 1), (2, 2)]
        assert [row.text for row in companion.sent[0].rows] == ["One", "Two"]
        assert companion.connects == 1
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_an_unknown_outcome_resends_the_frozen_batch_unchanged(tmp_path: Path) -> None:
    companion = _Companion()
    companion.script = ["drop", "wrong_ack", "ack"]
    writer, store, sender = await _setup(tmp_path, companion)
    try:
        _say(store, "One")
        await _until(lambda: len(companion.sent) == 3)
        await _until(lambda: _cursor(tmp_path) == 0)
        assert companion.sent[0] == companion.sent[1] == companion.sent[2]
        # Each unknown outcome drops its connection: no late answer can be misread.
        assert companion.connects == 3
        assert all(link.closed for link in companion.links[:2])
        assert (sender.sent, sender.resent) == (3, 2)
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_no_answer_within_the_timeout_is_an_unknown_outcome(tmp_path: Path) -> None:
    companion = _Companion()
    companion.script = ["hang"]
    writer, store, sender = await _setup(tmp_path, companion, reply_timeout=0.05)
    try:
        _say(store, "One")
        await _until(lambda: len(companion.sent) == 2)
        await _until(lambda: _cursor(tmp_path) == 0)
        assert companion.sent[0] == companion.sent[1]
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_a_refusal_fences_archiving_with_one_marker_and_keeps_the_outbox(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    companion = _Companion()
    companion.script = ["refuse"]
    writer, store, sender = await _setup(tmp_path, companion)
    try:
        _say(store, "One")
        await _until(lambda: sender.fence is not None)
        _say(store, "Two", "Three")
        await asyncio.sleep(0.05)
        assert len(companion.sent) == 1
        assert sender.fence == "quarantined"
        assert _cursor(tmp_path) is None
        output = capsys.readouterr().out
        assert _markers(output) == [{"fence": "quarantined", "version": 1}]
        assert "One" not in output
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_a_transient_refusal_retries_the_frozen_batch_unchanged_with_one_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    delays: list[float] = []

    async def recording(delay: float) -> None:
        delays.append(delay)
        await asyncio.sleep(0)

    companion = _Companion()
    companion.script = ["not_ready", "lease_held", "conversations", "ack", "not_ready", "ack"]
    writer, store, sender = await _setup(tmp_path, companion, sleep=recording)
    try:
        _say(store, "One")
        await _until(lambda: _cursor(tmp_path) == 0)
        assert companion.sent[:4] == [companion.sent[0]] * 4
        assert sender.fence is None
        _say(store, "Two")
        await _until(lambda: _cursor(tmp_path) == 1)
        assert sender.fence is None
        # Bounded backoff between retries, reset by the acknowledgment.
        assert delays == [0.5, 1.0, 2.0, 0.5]
        # One marker per episode: the first refusal of each run of transient refusals.
        assert _markers(capsys.readouterr().out) == [
            {"transient": "not_ready", "version": 1},
            {"transient": "not_ready", "version": 1},
        ]
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_a_refusal_category_in_neither_set_fails_closed_to_a_fence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    companion = _Companion()
    companion.script = ["novel"]
    writer, store, sender = await _setup(tmp_path, companion)
    try:
        _say(store, "One")
        await _until(lambda: sender.fence is not None)
        await asyncio.sleep(0.05)
        assert sender.fence == "novel"
        assert len(companion.sent) == 1
        assert _markers(capsys.readouterr().out) == [{"fence": "novel", "version": 1}]
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_no_voice_event_is_sent_unless_the_companion_advertises_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    companion = _Companion(capabilities=frozenset())
    writer, store, sender = await _setup(tmp_path, companion)
    try:
        _say(store, "One")
        await _until(lambda: companion.connects >= 2)
        assert companion.sent == []
        assert all(link.closed for link in companion.links[:-1])
        assert {"refusal": "capability", "version": 1} in _markers(capsys.readouterr().out)
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_archiving_never_blocks_speech_or_the_tail(tmp_path: Path) -> None:
    companion = _Companion()
    companion.script = ["hang"]
    writer, store, sender = await _setup(tmp_path, companion, reply_timeout=3600.0)
    try:
        _say(store, "One")
        await _until(lambda: len(companion.sent) == 1)
        loop = asyncio.get_running_loop()
        started = loop.time()
        _say(store, *(f"Row {index}" for index in range(10)))
        assert loop.time() - started < 0.5
        assert len(store.snapshot().messages) == 11

        def tail_holds_everything() -> bool:
            raw = read_run_record(tmp_path / "tail.json", 1 << 20)
            tail = parse_voice_tail(
                raw or b"", max_messages=16, max_item_chars=64, max_outbox_rows=16
            )
            return tail is not None and len(tail.conversation.messages) == 11

        await _until(tail_holds_everything)
    finally:
        await sender.close()
        await writer.close()


@pytest.mark.asyncio
async def test_close_stops_the_sender_even_mid_exchange(tmp_path: Path) -> None:
    companion = _Companion()
    companion.script = ["hang"]
    writer, store, sender = await _setup(tmp_path, companion, reply_timeout=3600.0)
    _say(store, "One")
    await _until(lambda: len(companion.sent) == 1)

    await asyncio.wait_for(sender.close(), timeout=2)
    await sender.close()
    await writer.close()

    assert companion.links[-1].closed


@pytest.mark.asyncio
async def test_connection_failures_back_off_within_a_bound(tmp_path: Path) -> None:
    delays: list[float] = []

    async def recording(delay: float) -> None:
        delays.append(delay)
        await asyncio.sleep(0)

    attempts = 0

    async def refusing() -> Any:
        nonlocal attempts
        attempts += 1
        raise ConnectionRefusedError("no companion")

    writer = VoiceTailWriter(tmp_path / "tail.json", conversation_ids=lambda: "conv")
    store = ConversationContextStore(max_item_chars=64, on_change=writer.update)
    await writer.open(store)
    sender = VoiceArchiveSender(
        writer, refusing, initial_backoff_seconds=0.5, max_backoff_seconds=2.0, sleep=recording
    )
    sender.start()
    try:
        _say(store, "One")
        await _until(lambda: attempts >= 5)
        assert delays[:5] == [0.5, 1.0, 2.0, 2.0, 2.0]
    finally:
        await sender.close()
        await writer.close()


def test_the_sender_options_are_bounded() -> None:
    writer = VoiceTailWriter(Path("tail.json"))

    async def connect() -> Any:
        raise AssertionError

    for options in (
        {"reply_timeout": 0.0},
        {"reply_timeout": float("inf")},
        {"initial_backoff_seconds": 0.0},
        {"initial_backoff_seconds": 5.0, "max_backoff_seconds": 1.0},
    ):
        with pytest.raises(ValueError):
            VoiceArchiveSender(writer, connect, **options)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VoiceArchiveSender(object(), connect)  # type: ignore[arg-type]
