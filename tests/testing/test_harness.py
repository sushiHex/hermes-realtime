import asyncio
from collections.abc import AsyncIterator

import pytest

from hermes_realtime.conversation import TurnState
from hermes_realtime.speech import (
    AudioFrame,
    SpeechChunk,
    Transcript,
    VoiceActivity,
)
from hermes_realtime.testing import InMemoryConversationHarness


class ScriptedVAD:
    def __init__(self, activities: list[VoiceActivity]) -> None:
        self._activities = iter(activities)

    def process(self, frame: AudioFrame) -> VoiceActivity:
        del frame
        return next(self._activities)


class ScriptedSTT:
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.cancelled = False

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        self.frames.append(frame)
        return ()

    async def finish_utterance(self) -> Transcript:
        return Transcript(text="Could you compare them?", final=True)

    async def cancel(self) -> None:
        self.cancelled = True


class FailingCancelSTT(ScriptedSTT):
    async def cancel(self) -> None:
        await super().cancel()
        raise ValueError("transcriber cancellation failed")


class BlockingCancelSTT(ScriptedSTT):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()
        self.cancel_count = 0

    async def cancel(self) -> None:
        self.cancel_started.set()
        await self.release_cancel.wait()
        self.cancel_count += 1
        await super().cancel()


class BlockingPushCancelSTT(BlockingCancelSTT):
    def __init__(self) -> None:
        super().__init__()
        self.push_started = asyncio.Event()

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        self.push_started.set()
        return ()


class BlockingPushSTT(ScriptedSTT):
    def __init__(self) -> None:
        super().__init__()
        self.push_started = asyncio.Event()
        self.release_push = asyncio.Event()

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        self.push_started.set()
        await self.release_push.wait()
        return ()


class BlockingSecondPushSTT(ScriptedSTT):
    def __init__(self) -> None:
        super().__init__()
        self.push_count = 0
        self.second_push_started = asyncio.Event()
        self.release_second_push = asyncio.Event()

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        self.push_count += 1
        if self.push_count == 2:
            self.second_push_started.set()
            await self.release_second_push.wait()
        return ()


class ScriptedTTS:
    def __init__(self) -> None:
        self.cancelled_turns: list[str] = []

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text
        for number, spoken_text in enumerate(
            ("Initial answer.", " I am checking that.", " I will report back."), start=1
        ):
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id=f"chunk_{number:03d}",
                text=spoken_text,
                audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=24_000, channels=1),
            )

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)


class FailingCancelTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.closed_turns: list[str] = []

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text
        try:
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id="chunk_001",
                text="Partially synthesized.",
                audio=AudioFrame(
                    pcm=b"\x00\x00", sample_rate_hz=24_000, channels=1
                ),
            )
        finally:
            self.closed_turns.append(turn_id)

    async def cancel(self, turn_id: str) -> None:
        await super().cancel(turn_id)
        raise RuntimeError("provider cancellation failed")


class YieldingCancelTTS(ScriptedTTS):
    async def cancel(self, turn_id: str) -> None:
        await asyncio.sleep(0)
        await super().cancel(turn_id)


class BlockingCancelTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(self, turn_id: str) -> None:
        self.cancel_started.set()
        await self.release_cancel.wait()
        await super().cancel(turn_id)


class BlockingSynthesisTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.closed_turns: list[str] = []

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text
        try:
            self.started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            self.closed_turns.append(turn_id)


class BlockingCleanupTTS(BlockingSynthesisTTS):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(self, turn_id: str) -> None:
        self.cancel_started.set()
        await self.release_cancel.wait()
        await super().cancel(turn_id)


class SlowClosingSynthesisTTS(BlockingSynthesisTTS):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text
        try:
            self.started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            self.close_started.set()
            await self.release_close.wait()
            self.closed_turns.append(turn_id)


class CancellationFailingSynthesisTTS(BlockingSynthesisTTS):
    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        self.started.set()
        try:
            await asyncio.Event().wait()
            yield  # pragma: no cover
        except asyncio.CancelledError:
            raise RuntimeError("provider failed during cancellation") from None


class CloseFailIterator:
    def __aiter__(self) -> "CloseFailIterator":
        return self

    async def __anext__(self) -> SpeechChunk:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        raise RuntimeError("iterator close failed")


class SlowCloseIterator(CloseFailIterator):
    def __init__(self) -> None:
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.closed = False

    async def aclose(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        self.closed = True


class SlowCloseTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.iterator = SlowCloseIterator()

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        return self.iterator


class MultiFailureTTS(ScriptedTTS):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        return CloseFailIterator()

    async def cancel(self, turn_id: str) -> None:
        del turn_id
        raise asyncio.CancelledError("provider cancel failed")


class SynchronousFailTTS(ScriptedTTS):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        raise RuntimeError("provider start failed")


class FailingSecondStartTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.start_count = 0

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.start_count += 1
        if self.start_count == 2:
            raise RuntimeError("provider start failed")
        return super().synthesize(text, turn_id)


class FailingSynthesisTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.closed_turns: list[str] = []

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text
        try:
            raise RuntimeError("provider synthesis failed")
            yield  # pragma: no cover
        finally:
            self.closed_turns.append(turn_id)


class WrongTurnTTS(ScriptedTTS):
    def __init__(self) -> None:
        super().__init__()
        self.closed_turns: list[str] = []

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text
        try:
            yield SpeechChunk(
                turn_id="wrong_turn",
                chunk_id="chunk_001",
                text="Wrong turn.",
                audio=AudioFrame(
                    pcm=b"\x00\x00", sample_rate_hz=24_000, channels=1
                ),
            )
        finally:
            self.closed_turns.append(turn_id)


@pytest.mark.asyncio
async def test_response_admission_waits_for_audio_state_transaction() -> None:
    stt = BlockingSecondPushSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)
    harness.join("user_001")
    await harness.receive_audio("user_001", frame)
    reception = asyncio.create_task(harness.receive_audio("user_001", frame))
    await asyncio.wait_for(stt.second_push_started.wait(), timeout=1)

    response = asyncio.create_task(harness.begin_response("turn_001", "Response"))
    await asyncio.sleep(0)
    assert response.done() is False

    stt.release_second_push.set()
    await asyncio.wait_for(reception, timeout=1)
    await asyncio.wait_for(response, timeout=1)
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.RESPONDING


@pytest.mark.asyncio
async def test_streaming_turn_can_be_interrupted_without_cancelling_background_work() -> None:
    stt = ScriptedSTT()
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED]),
        stt=stt,
        tts=tts,
    )
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)

    harness.join("user_001")
    assert await harness.receive_audio("user_001", frame) == ()
    transcripts = await harness.receive_audio("user_001", frame)
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "A deliberately longer response")

    assert await harness.synthesize_next() is True
    assert await harness.play_next() == "Initial answer."
    assert await harness.synthesize_next() is True

    await harness.cancel_speech()

    assert transcripts == (Transcript(text="Could you compare them?", final=True),)
    assert harness.delivered_text("turn_001") == "Initial answer."
    assert harness.queued_speech == ()
    assert tts.cancelled_turns == ["turn_001"]
    assert harness.active_background_work == frozenset({"task_001"})
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.turn_state_history == (
        TurnState.IDLE,
        TurnState.LISTENING,
        TurnState.TRANSCRIBING,
        TurnState.RESPONDING,
        TurnState.SPEAKING,
        TurnState.INTERRUPTED,
    )
    assert harness.events == (
        "participant.joined:user_001",
        "user.speech.started:user_001",
        "user.speech.ended:user_001",
        "user.transcript.final:Could you compare them?",
        "assistant.response.started:turn_001",
        "assistant.speaking.started:turn_001",
        "assistant.playback.interrupted:turn_001",
        "assistant.speaking.stopped:turn_001",
    )


@pytest.mark.asyncio
async def test_spoken_barge_in_cancels_foreground_but_not_background_work() -> None:
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    harness.join("user_001")
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True
    assert harness.start_next_playback() is not None
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)

    assert await harness.receive_audio("user_001", frame) == ()

    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.LISTENING
    assert harness.queued_speech == ()
    assert harness.delivered_text("turn_001") == ""
    assert tts.cancelled_turns == ["turn_001"]
    assert harness.active_background_work == frozenset({"task_001"})


@pytest.mark.asyncio
async def test_failed_spoken_barge_in_still_restores_listening_state() -> None:
    tts = FailingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    harness.join("user_001")
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True
    assert harness.start_next_playback() is not None
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)

    with pytest.raises(RuntimeError, match="provider cancellation failed"):
        await harness.receive_audio("user_001", frame)

    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.LISTENING
    assert harness.queued_speech == ()
    assert harness.active_background_work == frozenset({"task_001"})


@pytest.mark.asyncio
async def test_started_but_unconfirmed_playback_is_not_counted_as_delivered() -> None:
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    started = harness.start_next_playback()
    assert started is not None
    assert started.text == "Initial answer."

    await harness.cancel_speech()

    assert harness.delivered_text("turn_001") == ""
    assert harness.queued_speech == ()
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED


@pytest.mark.asyncio
async def test_speech_and_turn_cancellation_have_independent_scopes() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=ScriptedTTS(),
    )
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    await harness.cancel_speech()

    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.active_background_work == frozenset({"task_001"})

    await harness.cancel_turn()

    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE
    assert harness.active_background_work == frozenset({"task_001"})


@pytest.mark.asyncio
async def test_cancel_settles_running_synthesis_before_closing_iterator() -> None:
    tts = BlockingSynthesisTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    synthesis_call = asyncio.create_task(harness.synthesize_next())
    await asyncio.wait_for(tts.started.wait(), timeout=1)

    await asyncio.wait_for(harness.cancel_speech(), timeout=1)

    assert await asyncio.wait_for(synthesis_call, timeout=1) is False
    assert tts.closed_turns == ["turn_001"]
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_caller_cancelled_synthesis_releases_owned_iterator() -> None:
    tts = BlockingSynthesisTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    synthesis_call = asyncio.create_task(harness.synthesize_next())
    await asyncio.wait_for(tts.started.wait(), timeout=1)

    synthesis_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await synthesis_call

    assert tts.cancelled_turns == ["turn_001"]
    assert tts.closed_turns == ["turn_001"]
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_cancel_caller_cancellation_waits_for_iterator_shutdown() -> None:
    tts = SlowClosingSynthesisTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    synthesis_call = asyncio.create_task(harness.synthesize_next())
    await asyncio.wait_for(tts.started.wait(), timeout=1)
    cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.close_started.wait(), timeout=1)

    cancellation.cancel()
    await asyncio.sleep(0)
    assert cancellation.done() is False

    tts.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancellation, timeout=1)
    assert await asyncio.wait_for(synthesis_call, timeout=1) is False
    assert tts.closed_turns == ["turn_001"]
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_synthesis_caller_cancellation_is_not_swallowed_by_speech_cancel() -> None:
    tts = SlowClosingSynthesisTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    synthesis_call = asyncio.create_task(harness.synthesize_next())
    await asyncio.wait_for(tts.started.wait(), timeout=1)
    cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.close_started.wait(), timeout=1)

    synthesis_call.cancel()
    tts.release_close.set()
    await asyncio.wait_for(cancellation, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(synthesis_call, timeout=1)


@pytest.mark.asyncio
async def test_cancelled_synthesis_waiter_releases_cancellation_markers() -> None:
    tts = BlockingCleanupTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    synthesis_call = asyncio.create_task(harness.synthesize_next())
    await asyncio.wait_for(tts.started.wait(), timeout=1)
    cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.cancel_started.wait(), timeout=1)

    synthesis_call.cancel()
    synthesis_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await synthesis_call
    assert harness._speech_cancelled_tasks == set()
    assert harness._speech_cancel_requested_tasks == set()

    tts.release_cancel.set()
    await asyncio.wait_for(cancellation, timeout=1)


@pytest.mark.asyncio
async def test_provider_failure_is_not_swallowed_by_speech_cancel() -> None:
    tts = CancellationFailingSynthesisTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    synthesis_call = asyncio.create_task(harness.synthesize_next())
    await asyncio.wait_for(tts.started.wait(), timeout=1)

    results = await asyncio.gather(
        harness.cancel_speech(), synthesis_call, return_exceptions=True
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("provider failed during cancellation" in str(result) for result in results)
    assert harness.turn_state is TurnState.INTERRUPTED


@pytest.mark.asyncio
async def test_cancel_caller_cancellation_waits_for_aclose() -> None:
    tts = SlowCloseTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.iterator.close_started.wait(), timeout=1)

    cancellation.cancel()
    await asyncio.sleep(0)
    assert cancellation.done() is False

    tts.iterator.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancellation, timeout=1)
    assert tts.iterator.closed is True
    assert harness.turn_state is TurnState.INTERRUPTED


@pytest.mark.asyncio
async def test_cancel_preserves_provider_and_iterator_close_failures() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=MultiFailureTTS(),
    )
    await harness.begin_response("turn_001", "Response")

    with pytest.raises(BaseExceptionGroup) as captured:
        await harness.cancel_speech()

    assert {type(error) for error in captured.value.exceptions} == {
        asyncio.CancelledError,
        RuntimeError,
    }
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_reused_turn_clears_delivery_before_synthesis_then_cancellation() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=ScriptedTTS(),
    )
    await harness.begin_response("turn_001", "First incarnation")
    while await harness.synthesize_next():
        pass
    while await harness.play_next() is not None:
        pass
    assert harness.delivered_text("turn_001") != ""

    await harness.begin_response("turn_001", "Replacement")
    assert harness.delivered_text("turn_001") == ""
    await harness.cancel_turn()
    assert harness.delivered_text("turn_001") == ""


@pytest.mark.asyncio
async def test_reused_turn_clears_delivery_before_synchronous_start_failure() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=FailingSecondStartTTS(),
    )
    await harness.begin_response("turn_001", "First incarnation")
    while await harness.synthesize_next():
        pass
    while await harness.play_next() is not None:
        pass
    assert harness.delivered_text("turn_001") != ""

    with pytest.raises(RuntimeError, match="provider start failed"):
        await harness.begin_response("turn_001", "Replacement")
    assert harness.delivered_text("turn_001") == ""
    assert harness.turn_state is TurnState.IDLE


@pytest.mark.asyncio
async def test_stale_playback_confirmation_cannot_deliver_reused_ids() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=ScriptedTTS(),
    )
    await harness.begin_response("turn_001", "First incarnation")
    assert await harness.synthesize_next() is True
    stale_receipt = harness.start_next_playback()
    assert stale_receipt is not None
    await harness.cancel_turn()

    await harness.begin_response("turn_001", "Second incarnation")
    assert await harness.synthesize_next() is True
    current_receipt = harness.start_next_playback()
    assert current_receipt is not None
    assert current_receipt.chunk_id == stale_receipt.chunk_id

    with pytest.raises(KeyError, match="unknown playback receipt"):
        harness.confirm_playback_delivered(stale_receipt)
    assert harness.delivered_text("turn_001") == ""
    assert harness.confirm_playback_delivered(current_receipt) == "Initial answer."


@pytest.mark.asyncio
async def test_synchronous_synthesis_start_failure_rolls_back_turn() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=SynchronousFailTTS(),
    )

    with pytest.raises(RuntimeError, match="provider start failed"):
        await harness.begin_response("turn_001", "Response")

    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE
    assert harness.queued_speech == ()
    assert "assistant.response.started:turn_001" not in harness.events
    assert TurnState.RECOVERING in harness.turn_state_history


@pytest.mark.asyncio
async def test_repeated_speech_cancellation_is_idempotent() -> None:
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    await harness.cancel_speech()
    await harness.cancel_speech()

    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.queued_speech == ()
    assert tts.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_turn_cancellation_waits_for_in_progress_speech_cleanup() -> None:
    tts = BlockingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True
    speech_cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.cancel_started.wait(), timeout=1)

    turn_cancellation = asyncio.create_task(harness.cancel_turn())
    await asyncio.sleep(0)
    assert turn_cancellation.done() is False
    assert harness.active_turn_id == "turn_001"

    tts.release_cancel.set()
    await asyncio.wait_for(speech_cancellation, timeout=1)
    await asyncio.wait_for(turn_cancellation, timeout=1)
    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE


@pytest.mark.asyncio
async def test_concurrent_turn_cancellation_is_idempotent() -> None:
    tts = BlockingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    first = asyncio.create_task(harness.cancel_turn())
    second = asyncio.create_task(harness.cancel_turn())
    await asyncio.wait_for(tts.cancel_started.wait(), timeout=1)

    tts.release_cancel.set()
    await asyncio.gather(first, second)

    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE
    assert tts.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_stale_turn_cancellation_does_not_cancel_new_turn() -> None:
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "First")
    stale_generation = harness._turn_generation
    await harness.cancel_turn()
    await harness.begin_response("turn_001", "Second incarnation")

    await harness._cancel_turn_serialized(stale_generation)

    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.RESPONDING
    assert tts.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_cancelled_turn_caller_still_joins_speech_cleanup() -> None:
    tts = BlockingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    speech_cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.cancel_started.wait(), timeout=1)
    turn_cancellation = asyncio.create_task(harness.cancel_turn())
    await asyncio.sleep(0)

    turn_cancellation.cancel()
    await asyncio.sleep(0)
    assert turn_cancellation.done() is False
    assert harness.active_turn_id == "turn_001"

    tts.release_cancel.set()
    await asyncio.wait_for(speech_cancellation, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(turn_cancellation, timeout=1)
    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE


@pytest.mark.asyncio
async def test_playback_cannot_advance_while_cancellation_is_in_progress() -> None:
    tts = BlockingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True
    assert await harness.synthesize_next() is True
    started = harness.start_next_playback()
    assert started is not None
    cancellation = asyncio.create_task(harness.cancel_speech())
    await asyncio.wait_for(tts.cancel_started.wait(), timeout=1)

    assert await harness.synthesize_next() is False
    assert harness.start_next_playback() is None
    with pytest.raises(RuntimeError, match="playback is not active"):
        harness.confirm_playback_delivered(started)

    tts.release_cancel.set()
    await asyncio.wait_for(cancellation, timeout=1)
    assert harness.delivered_text("turn_001") == ""
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_concurrent_speech_cancellation_is_idempotent() -> None:
    tts = YieldingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    await asyncio.gather(harness.cancel_speech(), harness.cancel_speech())

    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.queued_speech == ()
    assert tts.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_cancel_restores_state_and_closes_iterator_when_provider_fails() -> None:
    tts = FailingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    with pytest.raises(RuntimeError, match="provider cancellation failed"):
        await harness.cancel_speech()

    assert tts.closed_turns == ["turn_001"]
    assert harness.queued_speech == ()
    assert harness.active_background_work == frozenset({"task_001"})
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert harness.turn_state_history == (
        TurnState.IDLE,
        TurnState.RESPONDING,
        TurnState.INTERRUPTED,
        TurnState.RECOVERING,
        TurnState.INTERRUPTED,
    )
    await harness.cancel_turn()
    await harness.begin_response("turn_002", "Another response")


@pytest.mark.asyncio
async def test_synthesis_failure_recovers_and_releases_iterator() -> None:
    tts = FailingSynthesisTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")

    with pytest.raises(RuntimeError, match="provider synthesis failed"):
        await harness.synthesize_next()

    assert tts.cancelled_turns == ["turn_001"]
    assert tts.closed_turns == ["turn_001"]
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert TurnState.RECOVERING in harness.turn_state_history
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_wrong_turn_chunk_recovers_and_releases_iterator() -> None:
    tts = WrongTurnTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")

    with pytest.raises(ValueError, match="unexpected turn_id"):
        await harness.synthesize_next()

    assert tts.cancelled_turns == ["turn_001"]
    assert tts.closed_turns == ["turn_001"]
    assert harness.active_turn_id == "turn_001"
    assert harness.turn_state is TurnState.INTERRUPTED
    assert TurnState.RECOVERING in harness.turn_state_history
    assert harness.queued_speech == ()


@pytest.mark.asyncio
async def test_disconnect_blocks_audio_admission_during_stt_cleanup() -> None:
    stt = BlockingPushCancelSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)
    harness.join("user_001")
    disconnection = asyncio.create_task(harness.disconnect("user_001"))
    await asyncio.wait_for(stt.cancel_started.wait(), timeout=1)

    reception = asyncio.create_task(harness.receive_audio("user_001", frame))
    await asyncio.sleep(0)
    assert stt.push_started.is_set() is False

    stt.release_cancel.set()
    await asyncio.wait_for(disconnection, timeout=1)
    with pytest.raises(RuntimeError, match="disconnect is in progress"):
        await asyncio.wait_for(reception, timeout=1)
    assert harness.turn_state is TurnState.IDLE
    assert "user.speech.started:user_001" not in harness.events


@pytest.mark.asyncio
async def test_stale_disconnect_does_not_remove_same_identity_rejoin() -> None:
    stt = BlockingCancelSTT()
    stt.release_cancel.set()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    harness.join("user_001")
    stale_generation = harness._participant_generations["user_001"]
    await harness.disconnect("user_001")
    harness.join("user_001")

    await harness._disconnect_serialized("user_001", stale_generation)

    assert harness.participants == frozenset({"user_001"})
    assert stt.cancel_count == 1


@pytest.mark.asyncio
async def test_queued_disconnect_blocks_playback_start_and_confirmation() -> None:
    stt = BlockingPushSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)
    harness.join("user_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True
    assert await harness.synthesize_next() is True
    receipt = harness.start_next_playback()
    assert receipt is not None

    reception = asyncio.create_task(harness.receive_audio("user_001", frame))
    await asyncio.wait_for(stt.push_started.wait(), timeout=1)
    disconnection = asyncio.create_task(harness.disconnect("user_001"))
    await asyncio.sleep(0)

    assert harness.start_next_playback() is None
    with pytest.raises(RuntimeError, match="disconnect is in progress"):
        harness.confirm_playback_delivered(receipt)

    stt.release_push.set()
    await asyncio.wait_for(reception, timeout=1)
    await asyncio.wait_for(disconnection, timeout=1)
    assert harness.delivered_text("turn_001") == ""


@pytest.mark.asyncio
async def test_queued_disconnect_blocks_participant_admission() -> None:
    stt = BlockingSecondPushSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED, VoiceActivity.SILENCE]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)
    harness.join("user_001")
    await harness.receive_audio("user_001", frame)
    reception = asyncio.create_task(harness.receive_audio("user_001", frame))
    await asyncio.wait_for(stt.second_push_started.wait(), timeout=1)
    disconnection = asyncio.create_task(harness.disconnect("user_001"))
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="disconnect is in progress"):
        harness.join("user_002")

    stt.release_second_push.set()
    await asyncio.wait_for(reception, timeout=1)
    await asyncio.wait_for(disconnection, timeout=1)
    assert harness.participants == frozenset()


@pytest.mark.asyncio
async def test_disconnect_blocks_join_until_stt_cleanup_finishes() -> None:
    stt = BlockingCancelSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    harness.join("user_001")
    disconnection = asyncio.create_task(harness.disconnect("user_001"))
    await asyncio.wait_for(stt.cancel_started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="disconnect is in progress"):
        harness.join("user_002")

    stt.release_cancel.set()
    await asyncio.wait_for(disconnection, timeout=1)
    assert harness.participants == frozenset()


@pytest.mark.asyncio
async def test_disconnect_blocks_new_response_until_stt_cleanup_finishes() -> None:
    stt = BlockingCancelSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    harness.join("user_001")
    await harness.begin_response("turn_001", "First")
    disconnection = asyncio.create_task(harness.disconnect("user_001"))
    await asyncio.wait_for(stt.cancel_started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="disconnect is in progress"):
        await harness.begin_response("turn_002", "Second")

    stt.release_cancel.set()
    await asyncio.wait_for(disconnection, timeout=1)
    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE
    assert harness.participants == frozenset()


@pytest.mark.asyncio
async def test_concurrent_disconnect_is_idempotent() -> None:
    stt = BlockingCancelSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=ScriptedTTS(),
    )
    harness.join("user_001")
    first = asyncio.create_task(harness.disconnect("user_001"))
    second = asyncio.create_task(harness.disconnect("user_001"))
    await asyncio.wait_for(stt.cancel_started.wait(), timeout=1)

    stt.release_cancel.set()
    await asyncio.gather(first, second)

    assert stt.cancel_count == 1
    assert harness.participants == frozenset()
    assert harness.turn_state is TurnState.IDLE


@pytest.mark.asyncio
async def test_disconnect_preserves_speech_and_stt_cleanup_failures() -> None:
    stt = FailingCancelSTT()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=FailingCancelTTS(),
    )
    harness.join("user_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    with pytest.raises(BaseExceptionGroup) as captured:
        await harness.disconnect("user_001")

    assert {type(error) for error in captured.value.exceptions} == {
        RuntimeError,
        ValueError,
    }
    assert harness.participants == frozenset()
    assert harness.active_turn_id is None
    assert harness.turn_state is TurnState.IDLE
    assert stt.cancelled is True


@pytest.mark.asyncio
async def test_disconnect_finishes_cleanup_when_speech_cancellation_fails() -> None:
    stt = ScriptedSTT()
    tts = FailingCancelTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=tts,
    )
    harness.join("user_001")
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    with pytest.raises(RuntimeError, match="provider cancellation failed"):
        await harness.disconnect("user_001")

    assert stt.cancelled is True
    assert tts.closed_turns == ["turn_001"]
    assert harness.participants == frozenset()
    assert harness.queued_speech == ()
    assert harness.active_background_work == frozenset({"task_001"})
    assert harness.events[-1] == "participant.left:user_001"


@pytest.mark.asyncio
async def test_completed_stream_closes_turn_and_allows_the_next_response() -> None:
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=ScriptedSTT(),
        tts=tts,
    )
    await harness.begin_response("turn_001", "Response")
    spoken: list[str] = []

    while await harness.synthesize_next():
        chunk_text = await harness.play_next()
        assert chunk_text is not None
        spoken.append(chunk_text)

    await harness.begin_response("turn_002", "Another response")
    second_spoken: list[str] = []
    while await harness.synthesize_next():
        chunk_text = await harness.play_next()
        assert chunk_text is not None
        second_spoken.append(chunk_text)

    assert "".join(spoken) == "Initial answer. I am checking that. I will report back."
    assert second_spoken == spoken
    assert harness.delivered_text("turn_001") == "".join(spoken)
    assert harness.delivered_text("turn_002") == "".join(second_spoken)
    assert tts.cancelled_turns == []
    assert harness.events == (
        "assistant.response.started:turn_001",
        "assistant.speaking.started:turn_001",
        "assistant.response.completed:turn_001",
        "assistant.speaking.stopped:turn_001",
        "assistant.response.started:turn_002",
        "assistant.speaking.started:turn_002",
        "assistant.response.completed:turn_002",
        "assistant.speaking.stopped:turn_002",
    )


@pytest.mark.asyncio
async def test_disconnect_stops_media_but_preserves_background_work() -> None:
    stt = ScriptedSTT()
    tts = ScriptedTTS()
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SILENCE]),
        stt=stt,
        tts=tts,
    )
    harness.join("user_001")
    harness.start_background_work("task_001")
    await harness.begin_response("turn_001", "Response")
    assert await harness.synthesize_next() is True

    await harness.disconnect("user_001")

    assert harness.participants == frozenset()
    assert harness.queued_speech == ()
    assert harness.active_background_work == frozenset({"task_001"})
    assert stt.cancelled is True
    assert tts.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_disconnect_while_listening_restores_idle_state() -> None:
    harness = InMemoryConversationHarness(
        vad=ScriptedVAD([VoiceActivity.SPEECH_STARTED]),
        stt=ScriptedSTT(),
        tts=ScriptedTTS(),
    )
    harness.join("user_001")
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)
    await harness.receive_audio("user_001", frame)
    assert harness.turn_state is TurnState.LISTENING

    await harness.disconnect("user_001")

    assert harness.turn_state is TurnState.IDLE
    assert harness.participants == frozenset()
