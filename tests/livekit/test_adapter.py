"""Unit tests for the LiveKit transport boundary."""

from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_peer_cannot_reconnect_after_disconnect() -> None:
    connection = _connection()
    peer = LiveKitRoomPeer(connection, identity="peer")
    peer._room = cast(Any, _SuccessfulRoom())
    peer._connected = True

    await peer.disconnect()

    with pytest.raises(RuntimeError, match="closed"):
        await peer.connect("another-room")
