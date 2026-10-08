"""Unit tests for the LiveKit transport boundary."""

from __future__ import annotations

import asyncio
import json
import math
import struct
from types import SimpleNamespace
from typing import Any, cast

import jwt
import pytest
from livekit import api, rtc

from hermes_realtime.livekit import LiveKitConnection, LiveKitRoomPeer
from hermes_realtime.speech.types import AudioFrame, SpeechChunk


def _connection() -> LiveKitConnection:
    return LiveKitConnection(
        "ws://127.0.0.1:7880",
        "test-key",
        "synthetic-unit-test-secret-32-bytes",
    )


def test_speech_chunk_duration_is_bounded_by_the_persistent_queue() -> None:
    frame = AudioFrame(
        pcm=b"\x00\x00" * (48_000 * 348 // 100),
        sample_rate_hz=48_000,
        channels=1,
    )

    LiveKitRoomPeer._validate_speech_chunk_duration(frame)

    oversized_samples = (
        48_000 * (LiveKitRoomPeer._MAX_SPEECH_QUEUE_MS + 1) // 1_000
    )
    oversized = AudioFrame(
        pcm=b"\x00\x00" * oversized_samples,
        sample_rate_hz=48_000,
        channels=1,
    )
    with pytest.raises(ValueError, match="queue duration"):
        LiveKitRoomPeer._validate_speech_chunk_duration(oversized)


def test_connection_repr_hides_api_secret() -> None:
    sensitive_value = "do-not-log-this"
    connection = LiveKitConnection(
        url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret=sensitive_value,
    )

    assert sensitive_value not in repr(connection)


def test_token_is_short_lived_room_scoped_and_audio_only() -> None:
    secret = "local-test-secret-at-least-32-bytes"
    connection = LiveKitConnection("ws://127.0.0.1:7880", "test-key", secret)

    token = connection.token(identity="peer", room_name="room-a")
    claims = api.TokenVerifier("test-key", secret).verify(token)
    raw_claims = jwt.decode(token, secret, algorithms=["HS256"], issuer="test-key")

    assert claims.identity == "peer"
    assert claims.video is not None
    assert claims.video.room_join is True
    assert claims.video.room == "room-a"
    assert claims.video.can_subscribe is True
    assert claims.video.can_publish_data is False
    assert claims.video.can_update_own_metadata is False
    assert claims.video.can_publish_sources == ["microphone"]
    assert 0 < raw_claims["exp"] - raw_claims["nbf"] <= 300


@pytest.mark.asyncio
async def test_subscribed_audio_track_queue_is_bounded() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    capacity = peer._subscribed_audio.maxsize
    assert capacity > 0
    track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)

    for _ in range(capacity + 5):
        peer._room.emit(
            "track_subscribed",
            cast(Any, track),
            cast(Any, SimpleNamespace(name="track")),
            cast(Any, SimpleNamespace(identity="remote")),
        )

    assert peer._subscribed_audio.qsize() == capacity
    peer._room.emit("disconnected", cast(Any, None))
    assert peer._subscribed_audio.empty()


@pytest.mark.asyncio
async def test_subscribed_audio_overflow_retains_newest_track() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    capacity = peer._subscribed_audio.maxsize
    track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)

    for index in range(capacity):
        peer._room.emit(
            "track_subscribed",
            cast(Any, track),
            cast(Any, SimpleNamespace(name=f"old-{index}")),
            cast(Any, SimpleNamespace(identity="remote")),
        )
    peer._room.emit(
        "track_subscribed",
        cast(Any, track),
        cast(Any, SimpleNamespace(name="expected-newest")),
        cast(Any, SimpleNamespace(identity="remote")),
    )

    queued_names = [
        peer._subscribed_audio.get_nowait()[1]
        for _ in range(peer._subscribed_audio.qsize())
    ]
    assert "old-0" not in queued_names
    assert "expected-newest" in queued_names


@pytest.mark.asyncio
async def test_bound_remote_identity_rejects_other_participant_tracks_before_queueing() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="worker")
    track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)
    peer.bind_remote_identity("browser_user")

    peer._room.emit(
        "track_subscribed",
        cast(Any, track),
        cast(Any, SimpleNamespace(name="attacker-track")),
        cast(Any, SimpleNamespace(identity="other_user")),
    )
    peer._room.emit(
        "track_subscribed",
        cast(Any, track),
        cast(Any, SimpleNamespace(name="user-track")),
        cast(Any, SimpleNamespace(identity="browser_user")),
    )

    identity, track_name, _ = peer._subscribed_audio.get_nowait()
    assert identity == "browser_user"
    assert track_name == "user-track"
    assert peer._subscribed_audio.empty()


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _QueueSource(_Closable):
    def __init__(self) -> None:
        super().__init__()
        self.clear_count = 0

    def clear_queue(self) -> None:
        self.clear_count += 1


class _FailingRoom:
    async def disconnect(self) -> None:
        raise RuntimeError("provider disconnect failed")


class _SuccessfulRoom:
    async def disconnect(self) -> None:
        return None


class _ConnectFailureRoom:
    def __init__(self, *, cleanup_fails_once: bool = False) -> None:
        self.cleanup_attempts = 0
        self.cleanup_fails_once = cleanup_fails_once

    async def connect(self, _url: str, _token: str) -> None:
        raise RuntimeError("connect failed")

    async def disconnect(self) -> None:
        self.cleanup_attempts += 1
        if self.cleanup_fails_once and self.cleanup_attempts == 1:
            raise RuntimeError("connect cleanup failed")


@pytest.mark.asyncio
async def test_connect_failure_closes_the_partial_room_and_peer() -> None:
    connection = _connection()
    peer = LiveKitRoomPeer(connection, identity="peer")
    room = _ConnectFailureRoom()
    peer._room = cast(Any, room)

    with pytest.raises(RuntimeError, match="connect failed"):
        await peer.connect("room")

    assert room.cleanup_attempts == 1
    assert not peer.connected
    with pytest.raises(RuntimeError, match="closed"):
        await peer.connect("room")


@pytest.mark.asyncio
async def test_failed_connect_cleanup_remains_retryable() -> None:
    connection = _connection()
    peer = LiveKitRoomPeer(connection, identity="peer")
    room = _ConnectFailureRoom(cleanup_fails_once=True)
    peer._room = cast(Any, room)

    with pytest.raises(BaseExceptionGroup, match="connect and cleanup failed"):
        await peer.connect("room")

    assert room.cleanup_attempts == 1
    await peer.disconnect()
    assert room.cleanup_attempts == 2
    assert not peer.connected


@pytest.mark.asyncio
async def test_disconnect_keeps_failed_room_cleanup_retryable() -> None:
    connection = _connection()
    peer = LiveKitRoomPeer(connection, identity="peer")
    stream = _Closable()
    source = _Closable()
    peer._audio_stream = cast(Any, stream)
    peer._audio_source = cast(Any, source)
    peer._room = cast(Any, _FailingRoom())
    peer._connected = True

    with pytest.raises(RuntimeError, match="provider disconnect failed"):
        await peer.disconnect()

    assert stream.closed
    assert source.closed
    assert peer._audio_stream is None
    assert peer._audio_source is None
    assert peer._connected

    peer._room = cast(Any, _SuccessfulRoom())
    await peer.disconnect()

    assert not peer._connected


@pytest.mark.asyncio
async def test_sdk_disconnected_event_updates_state_and_retains_local_cleanup() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    source = _Closable()
    peer._audio_source = cast(Any, source)
    peer._local_track = cast(Any, object())
    peer._publication_sid = "TR_departed"
    peer._connected = True

    peer._room.emit("disconnected", cast(Any, None))

    assert not peer.connected
    assert peer._closed
    assert peer._audio_source is source

    await peer.disconnect()

    assert source.closed
    assert peer._audio_source is None
    assert peer._local_track is None
    assert peer._publication_sid is None


@pytest.mark.asyncio
async def test_clear_audio_queue_discards_buffered_pcm() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    source = _QueueSource()
    peer._audio_source = cast(Any, source)
    peer._connected = True

    await peer.clear_audio_queue()

    assert source.clear_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout_seconds", "error_type"),
    [
        (True, TypeError),
        ("1", TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
async def test_peer_rejects_invalid_timeout_values(
    timeout_seconds: object,
    error_type: type[BaseException],
) -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    peer._connected = True

    with pytest.raises(error_type):
        await peer.clear_audio_queue(timeout_seconds=cast(Any, timeout_seconds))


class _FailingPublisher:
    async def publish_track(self, _track: object, _options: object) -> None:
        raise RuntimeError("publication failed")


class _PublishFailureRoom:
    local_participant = _FailingPublisher()


@pytest.mark.asyncio
async def test_publish_failure_does_not_retain_an_unpublished_source() -> None:
    connection = _connection()
    peer = LiveKitRoomPeer(connection, identity="peer")
    peer._room = cast(Any, _PublishFailureRoom())
    peer._connected = True
    frame = AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)

    with pytest.raises(RuntimeError, match="publication failed"):
        await peer.publish_audio(frame)

    assert peer._audio_source is None


class _DelayedPublisher:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.unpublished: list[str] = []
        self.publish_count = 0

    async def publish_track(self, _track: object, _options: object) -> object:
        self.publish_count += 1
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(sid="TR_delayed")

    async def unpublish_track(self, track_sid: str) -> None:
        self.unpublished.append(track_sid)


class _DelayedPublicationRoom:
    def __init__(self) -> None:
        self.local_participant = _DelayedPublisher()
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


@pytest.mark.asyncio
async def test_cancelled_publication_remains_owned_for_disconnect_cleanup() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    room = _DelayedPublicationRoom()
    peer._room = cast(Any, room)
    peer._connected = True
    frame = AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)

    publish_task = asyncio.create_task(peer.publish_audio(frame))
    await room.local_participant.started.wait()
    publish_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publish_task

    room.local_participant.release.set()
    await peer.disconnect()

    assert room.local_participant.unpublished == ["TR_delayed"]
    assert room.disconnect_calls == 1
    assert peer._audio_source is None


@pytest.mark.asyncio
async def test_cancelled_speech_prepare_remains_owned_for_exact_cleanup() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    room = _DelayedPublicationRoom()
    peer._room = cast(Any, room)
    peer._connected = True
    chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="hello",
        audio=AudioFrame(
            pcm=b"\x00\x00" * 480,
            sample_rate_hz=48_000,
            channels=1,
        ),
    )

    prepare_task = asyncio.create_task(peer.prepare_speech_chunk(chunk))
    await room.local_participant.started.wait()
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task

    room.local_participant.release.set()
    await peer.cancel_speech_chunk(chunk)

    assert room.local_participant.unpublished == ["TR_delayed"]
    assert peer._audio_source is None


@pytest.mark.asyncio
async def test_sequential_speech_chunks_reuse_one_publication_until_disconnect() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    room = _DelayedPublicationRoom()
    room.local_participant.release.set()
    peer._room = cast(Any, room)
    peer._connected = True
    first = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="first",
        audio=AudioFrame(
            pcm=b"\x00\x00" * 480,
            sample_rate_hz=48_000,
            channels=1,
        ),
    )
    second = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_002",
        text="second",
        audio=first.audio,
    )

    first_stream = await peer.prepare_speech_chunk(first)
    await peer.finish_speech_chunk(first)
    second_stream = await peer.prepare_speech_chunk(second)
    await peer.cancel_speech_chunk(second)

    assert first_stream == second_stream
    assert room.local_participant.publish_count == 1
    assert room.local_participant.unpublished == []

    await peer.disconnect()
    assert room.local_participant.unpublished == ["TR_delayed"]


@pytest.mark.asyncio
async def test_disconnect_waits_for_in_flight_publication() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    room = _DelayedPublicationRoom()
    peer._room = cast(Any, room)
    peer._connected = True
    frame = AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)

    publish_task = asyncio.create_task(peer.publish_audio(frame))
    await room.local_participant.started.wait()
    disconnect_task = asyncio.create_task(peer.disconnect())
    await asyncio.sleep(0)

    assert not disconnect_task.done()

    room.local_participant.release.set()
    await publish_task
    await disconnect_task

    assert room.local_participant.unpublished == ["TR_delayed"]
    assert peer._audio_source is None
    assert not peer.connected


@pytest.mark.asyncio
async def test_publish_rechecks_connection_when_disconnect_wins_lock() -> None:
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    room = _DelayedPublicationRoom()
    room.local_participant.release.set()
    peer._room = cast(Any, room)
    peer._connected = True
    frame = AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)

    await peer._publish_lock.acquire()
    disconnect_task = asyncio.create_task(peer.disconnect())
    await asyncio.sleep(0)
    publish_task = asyncio.create_task(peer.publish_audio(frame))
    await asyncio.sleep(0)
    peer._publish_lock.release()

    await disconnect_task
    with pytest.raises(RuntimeError, match="not connected"):
        await publish_task

    assert room.local_participant.unpublished == []
    assert peer._audio_source is None


_RATE = 48_000


def _ms(value: float) -> int:
    return round(_RATE * value / 1000)


def _word(ms: float, amplitude: int = 12_000) -> list[int]:
    return [
        round(amplitude * math.sin(2 * math.pi * 220 * index / _RATE))
        for index in range(_ms(ms))
    ]


def _gap(ms: float) -> list[int]:
    return [0] * _ms(ms)


def _speech(turn_id: str, chunk_id: str, samples: list[int]) -> SpeechChunk:
    return SpeechChunk(
        turn_id=turn_id,
        chunk_id=chunk_id,
        text=chunk_id,
        audio=AudioFrame(
            pcm=struct.pack(f"<{len(samples)}h", *samples),
            sample_rate_hz=_RATE,
            channels=1,
        ),
    )


class _PlayoutClock:
    """Fake playout clock: the native queue advances exactly with elapsed time."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []
        self.block: asyncio.Event | None = None
        # Per-sleep early wakes, as the 15.6 ms loop clock can produce.
        self.early: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.block is not None:
            await self.block.wait()
        early = self.early.pop(0) if self.early else 0.0
        self.now += seconds - early + 1e-9


class _PlayoutSource(_Closable):
    def __init__(self, clock: _PlayoutClock) -> None:
        super().__init__()
        self.clock = clock
        self.events: list[str] = []
        self.captured: list[list[int]] = []
        self.started_at: float | None = None
        self.cleared_at: float | None = None
        self.fail_capture_after = -1

    def clear_queue(self) -> None:
        self.events.append("clear")
        self.cleared_at = self.clock.now

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if len(self.captured) == self.fail_capture_after:
            raise RuntimeError("capture failed")
        self.events.append("capture")
        if self.started_at is None:
            self.started_at = self.clock.now
        self.captured.append(list(frame.data))

    async def wait_for_playout(self) -> None:
        self.events.append("playout")

    def played_until(self) -> int:
        """Samples handed to the track before the queue was cleared."""

        assert self.started_at is not None and self.cleared_at is not None
        return round((self.cleared_at - self.started_at) * _RATE)


async def _published_peer(
    samples: list[int],
    *,
    played_ms: float,
) -> tuple[LiveKitRoomPeer, _PlayoutSource, _PlayoutClock, SpeechChunk]:
    clock = _PlayoutClock()
    source = _PlayoutSource(clock)
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    room = _DelayedPublicationRoom()
    room.local_participant.release.set()
    peer._room = cast(Any, room)
    peer._connected = True
    peer._audio_source = cast(Any, source)
    peer._publication_sid = "TR_speech"
    peer._active_speech_track_name = "hermes-speech-test"
    peer._clock = clock
    peer._sleep = clock.sleep
    chunk = _speech("turn_001", "chunk_001", samples)
    await peer.prepare_speech_chunk(chunk)
    await peer.publish_speech_chunk(chunk)
    clock.now += played_ms / 1000
    return peer, source, clock, chunk


def _fade_source(samples: list[int], captured: list[int]) -> tuple[int, list[int]]:
    """The captured fade's PCM before padding: (length, samples)."""

    length = _ms(15)
    assert len(captured) % (_RATE // 100) == 0
    assert all(sample == 0 for sample in captured[length:])
    return length, captured[:length]


def _stop_evidence(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    marker = "[speech-stop] "
    return [
        json.loads(line[len(marker) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(marker)
    ]


@pytest.mark.asyncio
async def test_interrupted_speech_stops_in_the_next_word_gap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = _word(150) + _gap(60) + _word(400)
    peer, source, clock, chunk = await _published_peer(samples, played_ms=20)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    # The tail, then the fade drained on the playout clock: 2 blocks + 1 tick.
    assert clock.sleeps == [pytest.approx(0.130), pytest.approx(0.030)]
    assert source.events == ["capture", "clear", "capture"]
    stop = source.played_until()
    assert _ms(150) <= stop <= _ms(210) - _ms(15)
    _length, fade = _fade_source(samples, source.captured[1])
    assert all(sample == 0 for sample in fade)
    assert _stop_evidence(capsys) == [
        {
            "clamped": False,
            "faded": True,
            "late_ms": 0,
            "mode": "word",
            "outcome": "gap",
            "stop_block": 15,
            "tail_ms": 130,
        }
    ]
    assert peer._active_speech_chunk_key is None


@pytest.mark.asyncio
async def test_interrupted_speech_without_a_gap_fades_at_the_cap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = _word(1_000)
    peer, source, clock, chunk = await _published_peer(samples, played_ms=100)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert clock.sleeps == [pytest.approx(0.300), pytest.approx(0.030)]
    stop = source.played_until()
    assert stop == _ms(400)
    length, fade = _fade_source(samples, source.captured[1])
    assert fade[-1] == 0
    assert abs(fade[0] - samples[stop]) <= 12_000 * math.pi / length
    assert _stop_evidence(capsys) == [
        {
            "clamped": False,
            "faded": True,
            "late_ms": 0,
            "mode": "word",
            "outcome": "cap",
            "stop_block": 40,
            "tail_ms": 300,
        }
    ]


@pytest.mark.asyncio
async def test_interrupted_speech_ending_inside_the_cap_plays_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = _word(200)
    peer, source, clock, chunk = await _published_peer(samples, played_ms=100)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert clock.sleeps == [pytest.approx(0.100)]
    assert source.events == ["capture", "clear"]
    assert _stop_evidence(capsys) == [
        {
            "clamped": True,
            "faded": False,
            "late_ms": 0,
            "mode": "word",
            "outcome": "end",
            "stop_block": 20,
            "tail_ms": 100,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_word", "mode", "outcome"),
    [(True, "word", "end"), (False, "hard", "immediate")],
)
async def test_a_clock_past_the_chunk_end_counts_as_fully_played(
    capsys: pytest.CaptureFixture[str],
    finish_word: bool,
    mode: str,
    outcome: str,
) -> None:
    peer, source, clock, chunk = await _published_peer(_word(200), played_ms=500)

    await peer.cancel_speech_chunk(chunk, finish_word=finish_word)

    assert clock.sleeps == []
    assert source.events == ["capture", "clear"]
    assert _stop_evidence(capsys) == [
        {
            "clamped": True,
            "faded": False,
            "late_ms": 0,
            "mode": mode,
            "outcome": outcome,
            "stop_block": 20,
            "tail_ms": 0,
        }
    ]
    assert peer._active_speech_chunk_key is None


@pytest.mark.asyncio
async def test_the_tail_deadline_is_the_planned_stop_on_the_playout_clock() -> None:
    samples = _word(150) + _gap(60) + _word(400)
    # 23 ms played: the snapped position says 20 ms, the clock says 23 ms.
    peer, source, clock, chunk = await _published_peer(samples, played_ms=23)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert clock.sleeps[0] == pytest.approx(0.127)
    assert source.played_until() == _ms(150)


@pytest.mark.asyncio
async def test_a_late_wake_is_recorded_as_how_far_past_the_planned_stop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = _word(150) + _gap(60) + _word(400)
    peer, _source, clock, chunk = await _published_peer(samples, played_ms=20)
    clock.early = [-0.012]

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    evidence = _stop_evidence(capsys)[0]
    assert (evidence["late_ms"], evidence["stop_block"]) == (10, 16)


@pytest.mark.asyncio
async def test_early_timer_wakes_never_cut_or_drain_before_the_deadline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = _word(150) + _gap(60) + _word(400)
    peer, source, clock, chunk = await _published_peer(samples, played_ms=20)
    clock.early = [0.012, 0.0, 0.012]

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert clock.sleeps == [
        pytest.approx(0.130),
        pytest.approx(0.012),
        pytest.approx(0.030),
        pytest.approx(0.012),
    ]
    assert source.played_until() == _ms(150)
    assert _stop_evidence(capsys)[0]["late_ms"] == 0


@pytest.mark.asyncio
async def test_hard_stop_is_immediate_with_only_the_fade(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = _word(150) + _gap(60) + _word(400)
    peer, source, clock, chunk = await _published_peer(samples, played_ms=20)

    await peer.cancel_speech_chunk(chunk)

    assert clock.sleeps == [pytest.approx(0.030)]
    assert source.events == ["capture", "clear", "capture"]
    assert source.played_until() == _ms(20)
    length, fade = _fade_source(samples, source.captured[1])
    assert fade[-1] == 0
    assert abs(fade[0] - samples[_ms(20)]) <= 12_000 * math.pi / length
    assert _stop_evidence(capsys) == [
        {
            "clamped": False,
            "faded": True,
            "late_ms": 0,
            "mode": "hard",
            "outcome": "immediate",
            "stop_block": 2,
            "tail_ms": 0,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("played_ms", "expected_ms"), [(103, 100), (107, 110)])
async def test_fade_starts_at_the_nearest_native_ten_ms_block(
    played_ms: float,
    expected_ms: int,
) -> None:
    samples = list(range(-15_000, 15_000, 1))[: _ms(500)]
    peer, source, _clock, chunk = await _published_peer(samples, played_ms=played_ms)

    await peer.cancel_speech_chunk(chunk)

    _length, fade = _fade_source(samples, source.captured[1])
    assert abs(fade[0] - samples[_ms(expected_ms)]) <= 1


@pytest.mark.asyncio
async def test_unpublished_speech_cancel_only_clears() -> None:
    clock = _PlayoutClock()
    source = _PlayoutSource(clock)
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    peer._connected = True
    peer._audio_source = cast(Any, source)
    peer._publication_sid = "TR_speech"
    peer._active_speech_track_name = "hermes-speech-test"
    peer._clock = clock
    peer._sleep = clock.sleep
    chunk = _speech("turn_001", "chunk_001", _word(200))
    await peer.prepare_speech_chunk(chunk)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert source.events == ["clear"]
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_an_abandoned_chunks_clock_never_positions_another_chunk() -> None:
    clock = _PlayoutClock()
    source = _PlayoutSource(clock)
    peer = LiveKitRoomPeer(_connection(), identity="peer")
    peer._connected = True
    peer._audio_source = cast(Any, source)
    peer._publication_sid = "TR_speech"
    peer._active_speech_track_name = "hermes-speech-test"
    peer._clock = clock
    peer._sleep = clock.sleep
    # A chunk abandoned without release (as disconnect does) left its clock.
    peer._speech_playout = (("turn_000", "chunk_000"), clock.now - 0.1)
    chunk = _speech("turn_001", "chunk_001", _word(1_000))
    await peer.prepare_speech_chunk(chunk)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert source.events == ["clear"]
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_a_released_chunks_clock_is_retired_with_it() -> None:
    peer, source, clock, chunk = await _published_peer(_word(200), played_ms=50)
    await peer.finish_speech_chunk(chunk)
    await peer.prepare_speech_chunk(chunk)

    await peer.cancel_speech_chunk(chunk, finish_word=True)

    assert source.events == ["capture", "playout", "clear"]
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_cancelled_word_tail_still_clears_and_releases_the_chunk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    peer, source, clock, chunk = await _published_peer(_word(1_000), played_ms=10)
    clock.block = asyncio.Event()

    task = asyncio.create_task(peer.cancel_speech_chunk(chunk, finish_word=True))
    while not clock.sleeps:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert source.events == ["capture", "clear"]
    assert _stop_evidence(capsys) == [
        {"faded": False, "late_ms": 0, "mode": "word", "outcome": "cap", "tail_ms": 300}
    ]
    # Released: a retry plays no stale fade, and the next chunk is admitted.
    clock.block = None
    await peer.cancel_speech_chunk(chunk, finish_word=True)
    assert source.events == ["capture", "clear"]
    following = _speech("turn_002", "chunk_002", _word(100))
    await peer.prepare_speech_chunk(following)


@pytest.mark.asyncio
async def test_a_failed_fade_still_releases_the_chunk() -> None:
    peer, source, _clock, chunk = await _published_peer(_word(1_000), played_ms=10)
    source.fail_capture_after = 1

    with pytest.raises(RuntimeError, match="capture failed"):
        await peer.cancel_speech_chunk(chunk)

    assert source.events == ["capture", "clear"]
    assert peer._active_speech_chunk_key is None
    assert peer._speech_playout is None


@pytest.mark.asyncio
async def test_next_speech_waits_for_the_word_tail_without_overlap() -> None:
    peer, source, clock, chunk = await _published_peer(_word(1_000), played_ms=10)
    clock.block = asyncio.Event()
    following = _speech("turn_002", "chunk_002", _word(100))

    tail = asyncio.create_task(peer.cancel_speech_chunk(chunk, finish_word=True))
    while not clock.sleeps:
        await asyncio.sleep(0)
    next_prepare = asyncio.create_task(peer.prepare_speech_chunk(following))
    for _ in range(5):
        await asyncio.sleep(0)

    assert not next_prepare.done()
    clock.block.set()
    await tail
    await next_prepare
    await peer.publish_speech_chunk(following)

    assert source.events == ["capture", "clear", "capture", "capture"]


@pytest.mark.asyncio
async def test_cancel_rejects_a_non_boolean_finish_word() -> None:
    peer, _source, _clock, chunk = await _published_peer(_word(100), played_ms=10)

    with pytest.raises(TypeError, match="finish_word"):
        await peer.cancel_speech_chunk(chunk, finish_word=cast(Any, 1))


@pytest.mark.asyncio
async def test_peer_cannot_reconnect_after_disconnect() -> None:
    connection = _connection()
    peer = LiveKitRoomPeer(connection, identity="peer")
    peer._room = cast(Any, _SuccessfulRoom())
    peer._connected = True

    await peer.disconnect()

    with pytest.raises(RuntimeError, match="closed"):
        await peer.connect("another-room")
