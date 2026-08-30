"""Opt-in end-to-end test against the native local LiveKit server."""

from __future__ import annotations

import asyncio
import math
import os
import struct
import uuid
from collections.abc import AsyncIterator, Callable

import pytest

from hermes_realtime.conversation import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.livekit import (
    LiveKitConnection,
    LiveKitPCMDeliveryConfirmation,
    LiveKitRoomPeer,
    LiveKitSpeechPlayback,
)
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    SpeechChunk,
    Transcript,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_LIVEKIT_LOCAL") != "1",
        reason="set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit",
    ),
]


def _local_connection() -> LiveKitConnection:
    return LiveKitConnection(
        url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
        api_secret=os.getenv("LIVEKIT_API_SECRET", "local-" + ("x" * 32)),
    )


def _tone_frame(*, frequency_hz: float = 440.0) -> AudioFrame:
    sample_rate_hz = 48_000
    samples_per_channel = 480
    samples = [
        round(12_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate_hz))
        for index in range(samples_per_channel)
    ]
    return AudioFrame(
        pcm=struct.pack(f"<{len(samples)}h", *samples),
        sample_rate_hz=sample_rate_hz,
        channels=1,
    )


class _BlockedIncrementalInference:
    def __init__(self) -> None:
        self.release_completion = asyncio.Event()
        self.completed = False

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "First audible answer."
        await self.release_completion.wait()
        yield "Second audible answer."
        self.completed = True

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class _ToneSynthesizer:
    def __init__(self) -> None:
        self._chunk_number = 0

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self._chunk_number += 1
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"livekit_chunk_{self._chunk_number}",
            text=text,
            audio=_tone_frame(frequency_hz=440.0 + self._chunk_number * 40.0),
        )

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class _TracingPCMConfirmation:
    def __init__(self, subscriber: LiveKitRoomPeer) -> None:
        self._confirmation = LiveKitPCMDeliveryConfirmation(
            subscriber,
            timeout_seconds=10,
            max_frames=150,
        )
        self.confirmed_chunks: list[str] = []
        self.first_confirmation = asyncio.Event()

    async def prepare(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        await self._confirmation.prepare(chunk, stream_identity, is_valid=is_valid)

    async def confirm(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        await self._confirmation.confirm(chunk, is_valid=is_valid)
        self.confirmed_chunks.append(chunk.text)
        self.first_confirmation.set()

    async def cancel(self, turn_id: str) -> None:
        await self._confirmation.cancel(turn_id)


class _GatedChunkReceiver:
    def __init__(self, peer: LiveKitRoomPeer, blocked_turn_id: str) -> None:
        self._peer = peer
        self._blocked_turn_id = blocked_turn_id
        self._active_turn_id: str | None = None
        self._blocked_once = False
        self.receive_started = asyncio.Event()
        self.release = asyncio.Event()

    async def prepare_receive_speech_chunk(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        self._active_turn_id = chunk.turn_id
        await self._peer.prepare_receive_speech_chunk(
            chunk,
            stream_identity,
            timeout_seconds=timeout_seconds,
        )

    async def receive_audio(self, *, timeout_seconds: float) -> AudioFrame:
        if self._active_turn_id == self._blocked_turn_id and not self._blocked_once:
            self._blocked_once = True
            self.receive_started.set()
            async with asyncio.timeout(timeout_seconds):
                await self.release.wait()
        return await self._peer.receive_audio(timeout_seconds=timeout_seconds)


@pytest.mark.asyncio
async def test_two_peers_exchange_pcm_audio_through_local_livekit() -> None:
    room_name = f"hermes-local-{uuid.uuid4().hex}"
    connection = _local_connection()
    publisher = LiveKitRoomPeer(connection, identity="publisher")
    subscriber = LiveKitRoomPeer(connection, identity="subscriber")

    async with asyncio.timeout(20):
        try:
            await subscriber.connect(room_name)
            await publisher.connect(room_name)

            async def publish_tone() -> None:
                frame = _tone_frame()
                for _ in range(100):
                    await publisher.publish_audio(frame)

            async def receive_non_silent_frame() -> AudioFrame:
                for _ in range(150):
                    frame = await subscriber.receive_audio(timeout_seconds=10)
                    samples = struct.unpack(f"<{len(frame.pcm) // 2}h", frame.pcm)
                    if max(abs(sample) for sample in samples) > 500:
                        return frame
                raise AssertionError("subscriber received no non-silent PCM frame")

            received, _ = await asyncio.gather(receive_non_silent_frame(), publish_tone())

            assert received.sample_rate_hz == 48_000
            assert received.channels == 1
            assert len(received.pcm) % 2 == 0
            assert subscriber.remote_identities == frozenset({"publisher"})
            assert publisher.remote_identities == frozenset({"subscriber"})

            await publisher.disconnect()
            while subscriber.remote_identities:
                await asyncio.sleep(0.01)
            assert not publisher.connected

            await subscriber.disconnect()
            assert not subscriber.connected
        finally:
            await publisher.disconnect()
            await subscriber.disconnect()


async def test_streaming_loop_confirms_livekit_pcm_before_inference_completes() -> None:
    room_name = f"hermes-streaming-{uuid.uuid4().hex}"
    connection = _local_connection()
    publisher = LiveKitRoomPeer(connection, identity="streaming-publisher")
    subscriber = LiveKitRoomPeer(connection, identity="streaming-subscriber")
    inference = _BlockedIncrementalInference()
    context = ConversationContextStore()
    ledger = DeliveredSpeechLedger()
    response: asyncio.Task[None] | None = None
    loop: StreamingSpeechLoop | None = None

    async with asyncio.timeout(30):
        try:
            await subscriber.connect(room_name)
            await publisher.connect(room_name)
            confirmation = _TracingPCMConfirmation(subscriber)
            playback = LiveKitSpeechPlayback(
                publisher=publisher,
                confirmation=confirmation,
            )
            loop = StreamingSpeechLoop(
                context=context,
                foreground=ForegroundTurnCoordinator(),
                inference=inference,
                synthesizer=_ToneSynthesizer(),
                playback=playback,
                ledger=ledger,
            )
            response = asyncio.create_task(
                loop.respond(
                    "turn_livekit_001",
                    Transcript(text="Can you hear this?", final=True),
                )
            )

            await confirmation.first_confirmation.wait()
            assert inference.completed is False

            inference.release_completion.set()
            await response

            assert confirmation.confirmed_chunks == [
                "First audible answer.",
                "Second audible answer.",
            ]
            assert [message.text for message in context.snapshot().messages] == [
                "Can you hear this?",
                "First audible answer.",
                "Second audible answer.",
            ]
            assert ledger.pending() == ()
            assert ledger.retained_chunk_count == 0
        finally:
            if response is not None and not response.done() and loop is not None:
                await loop.cancel()
            await publisher.disconnect()
            await subscriber.disconnect()


@pytest.mark.asyncio
async def test_cancelled_chunk_track_cannot_confirm_replacement() -> None:
    room_name = f"hermes-replacement-{uuid.uuid4().hex}"
    connection = _local_connection()
    publisher = LiveKitRoomPeer(connection, identity="replacement-publisher")
    subscriber = LiveKitRoomPeer(connection, identity="replacement-subscriber")

    async with asyncio.timeout(30):
        try:
            await subscriber.connect(room_name)
            await publisher.connect(room_name)
            receiver = _GatedChunkReceiver(subscriber, "turn_cancelled")
            confirmation = LiveKitPCMDeliveryConfirmation(receiver, timeout_seconds=10)
            playback = LiveKitSpeechPlayback(
                publisher=publisher,
                confirmation=confirmation,
            )
            cancelled = SpeechChunk(
                turn_id="turn_cancelled",
                chunk_id="chunk_cancelled",
                text="cancelled",
                audio=_tone_frame(frequency_hz=520.0),
            )
            replacement = SpeechChunk(
                turn_id="turn_cancelled",
                chunk_id="chunk_cancelled",
                text="replacement incarnation",
                audio=_tone_frame(frequency_hz=820.0),
            )
            cancelled_task = asyncio.create_task(
                playback.play(cancelled, is_valid=lambda: True)
            )
            await receiver.receive_started.wait()

            await playback.cancel("turn_cancelled")
            assert cancelled_task.done()

            await playback.play(replacement, is_valid=lambda: True)
            with pytest.raises(asyncio.CancelledError):
                await cancelled_task
        finally:
            await publisher.disconnect()
            await subscriber.disconnect()
