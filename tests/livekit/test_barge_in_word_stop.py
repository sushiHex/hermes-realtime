"""A barge-in drives the real playback and peer all the way to a word-boundary stop."""

from __future__ import annotations

import asyncio
import json
import math
import struct
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit import rtc

from hermes_realtime.conversation import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.livekit import (
    LiveKitConnection,
    LiveKitRoomPeer,
    LiveKitSpeechPlayback,
    ReconnectSafeLiveKitAudioPublisher,
)
from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk, Transcript

RATE = 48_000


def _ms(value: float) -> int:
    return round(RATE * value / 1000)


def _speech_pcm() -> bytes:
    """A word, a 60 ms gap, then another word."""

    word = [round(12_000 * math.sin(2 * math.pi * 220 * i / RATE)) for i in range(_ms(150))]
    samples = word + [0] * _ms(60) + word + word
    return struct.pack(f"<{len(samples)}h", *samples)


class _Clock:
    def __init__(self) -> None:
        self.now = 50.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds + 1e-9
        await asyncio.sleep(0)


class _Source:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.events: list[tuple[str, float]] = []

    def clear_queue(self) -> None:
        self.events.append(("clear", self.clock.now))

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.events.append(("capture", self.clock.now))

    async def wait_for_playout(self) -> None:
        self.events.append(("playout", self.clock.now))

    async def aclose(self) -> None:
        return None


class _Inference:
    async def _stream(self) -> AsyncIterator[str]:
        yield "One answer."

    def stream(self, snapshot: ConversationContextSnapshot, *, turn_id: str) -> AsyncIterator[str]:
        del snapshot, turn_id
        return self._stream()

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class _Synthesizer:
    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id="chunk_001",
            text=text,
            audio=AudioFrame(pcm=_speech_pcm(), sample_rate_hz=RATE, channels=1),
        )

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class _BlockingConfirmation:
    def __init__(self) -> None:
        self.confirming = asyncio.Event()
        self.released = asyncio.Event()

    async def prepare(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        del chunk, stream_identity, is_valid

    async def confirm(self, chunk: SpeechChunk, *, is_valid: Callable[[], bool]) -> None:
        del chunk
        self.confirming.set()
        await self.released.wait()
        if not is_valid():
            raise asyncio.CancelledError

    async def cancel(self, turn_id: str) -> None:
        del turn_id
        self.released.set()


@pytest.mark.asyncio
async def test_barge_in_reaches_the_peer_as_a_word_boundary_stop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = _Clock()
    source = _Source(clock)
    peer = LiveKitRoomPeer(
        LiveKitConnection("ws://127.0.0.1:7880", "test-key", "synthetic-unit-test-secret-32-bytes"),
        identity="worker",
    )
    peer._room = cast(Any, SimpleNamespace(local_participant=SimpleNamespace()))
    peer._connected = True
    peer._audio_source = cast(Any, source)
    peer._publication_sid = "TR_speech"
    peer._active_speech_track_name = "hermes-speech-test"
    peer._clock = clock
    peer._sleep = clock.sleep
    publisher = ReconnectSafeLiveKitAudioPublisher()
    await publisher.bind(peer)
    confirmation = _BlockingConfirmation()
    context = ConversationContextStore()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=_Inference(),
        synthesizer=_Synthesizer(),
        playback=LiveKitSpeechPlayback(publisher=publisher, confirmation=confirmation),
        ledger=DeliveredSpeechLedger(),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(confirmation.confirming.wait(), timeout=1)
    published_at = source.events[0][1]
    clock.now += 0.020

    assert await loop.cancel_if_active("turn_001") is True
    with pytest.raises(asyncio.CancelledError):
        await response

    # The chunk played on to the gap at 150 ms, then the queue was cut and faded.
    kinds = [kind for kind, _ in source.events]
    assert kinds == ["capture", "clear", "capture"]
    cut_at = source.events[1][1] - published_at
    assert cut_at == pytest.approx(0.150, abs=1e-6)
    evidence = [
        json.loads(line.removeprefix("[speech-stop] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[speech-stop] ")
    ]
    assert evidence == [
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
    # Nothing of the interrupted chunk is heard, and the peer is free again.
    assert [message.text for message in context.snapshot().messages] == ["Question?"]
    assert peer._active_speech_chunk_key is None
    await publisher.unbind(peer)
