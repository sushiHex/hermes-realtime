"""Voice readiness is announced only once the readiness cue no longer suppresses input (#574)."""

from __future__ import annotations

import asyncio

import pytest

from hermes_realtime import host_launcher
from hermes_realtime.speech import AudioFrame, SpeechChunk

_CHUNK = SpeechChunk(
    turn_id="readiness_1_1",
    chunk_id="readiness_1_1",
    text="Ready.",
    audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=48_000, channels=1),
)


class _Conversation:
    def __init__(self, events: list[str], *, suppress_error: BaseException | None = None,
                 resume_error: BaseException | None = None) -> None:
        self.events = events
        self.suppressed = False
        self._suppress_error = suppress_error
        self._resume_error = resume_error

    async def suppress_audio_input(self, **kwargs: object) -> object:
        if self._suppress_error is not None:
            raise self._suppress_error
        self.suppressed = True
        self.events.append("suppressed")
        return object()

    async def resume_audio_input(self, token: object) -> bool:
        await asyncio.sleep(0)
        if self._resume_error is not None:
            raise self._resume_error
        self.suppressed = False
        self.events.append("released")
        return True


class _Playback:
    def __init__(self, events: list[str], *, error: BaseException | None = None,
                 hold: asyncio.Event | None = None) -> None:
        self.events = events
        self._error = error
        self._hold = hold
        self.started = asyncio.Event()

    async def play(self, chunk: SpeechChunk, *, is_valid: object) -> None:
        self.events.append("playing")
        self.started.set()
        if self._hold is not None:
            await self._hold.wait()
        if self._error is not None:
            raise self._error
        self.events.append("played")


def _ready_recorder(events: list[str], conversation: _Conversation):
    def ready() -> None:
        # The announcement must never be visible while this cue still suppresses input.
        assert not conversation.suppressed
        events.append("ready")

    return ready


async def _play(conversation: _Conversation, playback: _Playback, ready) -> None:
    await host_launcher._play_readiness_cue(
        conversation=conversation,
        playback=playback,
        chunk=_CHUNK,
        participant_identity="participant",
        generation=1,
        is_valid=lambda: True,
        on_input_ready=ready,
    )


@pytest.mark.asyncio
async def test_ready_is_announced_after_the_cue_releases_input() -> None:
    events: list[str] = []
    conversation = _Conversation(events)
    playback = _Playback(events)

    await _play(conversation, playback, _ready_recorder(events, conversation))

    assert events == ["suppressed", "playing", "played", "released", "ready"]


@pytest.mark.asyncio
async def test_failed_cue_playback_still_announces_ready_after_release() -> None:
    events: list[str] = []
    conversation = _Conversation(events)
    playback = _Playback(events, error=RuntimeError("synthetic cue failure"))

    with pytest.raises(RuntimeError, match="synthetic cue failure"):
        await _play(conversation, playback, _ready_recorder(events, conversation))

    assert events == ["suppressed", "playing", "released", "ready"]


@pytest.mark.asyncio
async def test_cue_that_cannot_suppress_input_does_not_strand_readiness() -> None:
    events: list[str] = []
    conversation = _Conversation(events, suppress_error=RuntimeError("already suppressed"))
    playback = _Playback(events)

    with pytest.raises(RuntimeError, match="already suppressed"):
        await _play(conversation, playback, _ready_recorder(events, conversation))

    assert events == ["ready"]


@pytest.mark.asyncio
async def test_cancelled_cue_releases_input_without_announcing_ready() -> None:
    events: list[str] = []
    conversation = _Conversation(events)
    playback = _Playback(events, hold=asyncio.Event())
    operation = asyncio.create_task(
        _play(conversation, playback, _ready_recorder(events, conversation))
    )
    await asyncio.wait_for(playback.started.wait(), 1)

    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert events == ["suppressed", "playing", "released"]


class _Speech:
    def __init__(self, active_turn_id: str | None = None) -> None:
        self.active_turn_id = active_turn_id


async def _run(*, speech: _Speech, gate: asyncio.Lock, admitted: bool, events: list[str]) -> None:
    async def play() -> None:
        events.append("played")

    await host_launcher._run_readiness_cue(
        playback_gate=gate,
        speech=speech,
        admit=lambda: admitted,
        play=play,
        announce_ready=lambda: events.append("ready"),
    )


@pytest.mark.asyncio
async def test_admitted_cue_leaves_the_announcement_to_its_playback() -> None:
    events: list[str] = []
    await _run(speech=_Speech(), gate=asyncio.Lock(), admitted=True, events=events)
    assert events == ["played"]


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", ["speech_active", "gate_busy", "not_admitted"])
async def test_skipped_cue_announces_ready_at_once(skip: str) -> None:
    events: list[str] = []
    gate = asyncio.Lock()
    if skip == "gate_busy":
        await gate.acquire()
    await _run(
        speech=_Speech("turn_1" if skip == "speech_active" else None),
        gate=gate,
        admitted=skip != "not_admitted",
        events=events,
    )
    assert events == ["ready"]


@pytest.mark.asyncio
async def test_cue_whose_release_fails_does_not_announce_ready() -> None:
    events: list[str] = []
    conversation = _Conversation(events, resume_error=RuntimeError("release failed"))
    playback = _Playback(events)

    with pytest.raises(RuntimeError, match="release failed"):
        await _play(conversation, playback, _ready_recorder(events, conversation))

    assert events == ["suppressed", "playing", "played"]
