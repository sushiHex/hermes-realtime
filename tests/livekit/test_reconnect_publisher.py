from __future__ import annotations

import asyncio
from types import MethodType

import pytest

from hermes_realtime.livekit import LiveKitRoomPeer, ReconnectSafeLiveKitAudioPublisher
from hermes_realtime.speech import AudioFrame, SpeechChunk


def _chunk() -> SpeechChunk:
    return SpeechChunk(
        turn_id="turn_1",
        chunk_id="chunk_1",
        text="hello",
        audio=AudioFrame(pcm=b"\x01\x00" * 480, sample_rate_hz=48_000, channels=1),
    )


def _peer_probe() -> tuple[LiveKitRoomPeer, list[str]]:
    peer = object.__new__(LiveKitRoomPeer)
    events: list[str] = []

    async def prepare(
        self: LiveKitRoomPeer,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str:
        del self, chunk, timeout_seconds
        events.append("prepare")
        return "track_1"

    async def publish(
        self: LiveKitRoomPeer,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, chunk, timeout_seconds
        events.append("publish")

    async def finish(
        self: LiveKitRoomPeer,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, chunk, timeout_seconds
        events.append("finish")

    async def cancel(
        self: LiveKitRoomPeer,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, chunk, timeout_seconds
        events.append("cancel")

    peer.prepare_speech_chunk = MethodType(prepare, peer)  # type: ignore[method-assign]
    peer.publish_speech_chunk = MethodType(publish, peer)  # type: ignore[method-assign]
    peer.finish_speech_chunk = MethodType(finish, peer)  # type: ignore[method-assign]
    peer.cancel_speech_chunk = MethodType(cancel, peer)  # type: ignore[method-assign]
    return peer, events


@pytest.mark.asyncio
async def test_reconnect_safe_publisher_pins_each_chunk_to_bound_peer() -> None:
    peer, events = _peer_probe()
    publisher = ReconnectSafeLiveKitAudioPublisher()
    chunk = _chunk()

    await publisher.bind(peer)
    assert await publisher.prepare_speech_chunk(chunk) == "track_1"
    await publisher.publish_speech_chunk(chunk)
    with pytest.raises(RuntimeError, match="active speech chunks"):
        await publisher.unbind(peer)
    await publisher.finish_speech_chunk(chunk)
    await publisher.unbind(peer)

    assert events == ["prepare", "publish", "finish"]
    with pytest.raises(RuntimeError, match="not bound"):
        await publisher.prepare_speech_chunk(chunk)


@pytest.mark.asyncio
async def test_reconnect_safe_publisher_cancel_releases_chunk_authority() -> None:
    peer, events = _peer_probe()
    publisher = ReconnectSafeLiveKitAudioPublisher()
    chunk = _chunk()

    await publisher.bind(peer)
    await publisher.prepare_speech_chunk(chunk)
    await publisher.cancel_speech_chunk(chunk)
    await publisher.unbind(peer)

    assert events == ["prepare", "cancel"]


@pytest.mark.asyncio
async def test_cancelled_prepare_retains_exact_peer_authority_for_cleanup() -> None:
    peer, events = _peer_probe()
    prepare_started = asyncio.Event()
    never = asyncio.Event()

    async def prepare(
        self: LiveKitRoomPeer,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str:
        del self, chunk, timeout_seconds
        events.append("prepare-side-effect")
        prepare_started.set()
        await never.wait()
        return "track_unreachable"

    peer.prepare_speech_chunk = MethodType(prepare, peer)  # type: ignore[method-assign]
    publisher = ReconnectSafeLiveKitAudioPublisher()
    chunk = _chunk()
    await publisher.bind(peer)

    preparing = asyncio.create_task(publisher.prepare_speech_chunk(chunk))
    await asyncio.wait_for(prepare_started.wait(), timeout=1)
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing

    await publisher.cancel_speech_chunk(chunk)
    await publisher.unbind(peer)
    assert events == ["prepare-side-effect", "cancel"]
