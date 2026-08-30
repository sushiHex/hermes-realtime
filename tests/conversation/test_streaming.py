import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from types import MethodType
from typing import cast

import pytest

from hermes_realtime.conversation import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationInferenceRequest,
    ConversationPromptUpdate,
    ForegroundOutputBackpressure,
    ForegroundTurnCoordinator,
    PrivateRunDisclosureError,
    StreamingSpeechLoop,
    TaskTerminalOutcome,
    UpdateDecision,
    UpdateDecisionKind,
)
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    DurationBoundedSynthesizer,
    SpeechChunk,
    Transcript,
)


class BlockingIncrementalInference:
    def __init__(self) -> None:
        self.first_segment_emitted = asyncio.Event()
        self.release_second_segment = asyncio.Event()
        self.completed = False
        self.snapshots: list[ConversationContextSnapshot] = []

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        self.snapshots.append(snapshot)
        yield "First answer."
        self.first_segment_emitted.set()
        await self.release_second_segment.wait()
        yield "Second answer."
        self.completed = True

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class CancellationRacingFailureInference:
    def __init__(self) -> None:
        self.release_failure = asyncio.Event()
        self.failure_ready = asyncio.Event()
        self.raise_failure = asyncio.Event()

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "First answer."
        await self.release_failure.wait()
        self.failure_ready.set()
        with suppress(asyncio.CancelledError):
            await self.raise_failure.wait()
        raise RuntimeError("producer boom")

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class MutatedSnapshotContextStore(ConversationContextStore):
    def snapshot(self) -> ConversationContextSnapshot:
        snapshot = super().snapshot()
        object.__setattr__(snapshot.messages[0], "text", "deleg_private_001")
        return snapshot


class InferenceCallProbe:
    def __init__(self) -> None:
        self.stream_called = False

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        self.stream_called = True
        return self._empty()

    async def _empty(self) -> AsyncIterator[str]:
        if False:
            yield ""

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class RequestRecordingInference:
    def __init__(self) -> None:
        self.requests: list[ConversationInferenceRequest] = []

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del turn_id
        if type(snapshot) is not ConversationInferenceRequest:
            raise TypeError("expected exact ConversationInferenceRequest")
        self.requests.append(snapshot)
        yield "I will mention the update now."

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class CreatingFailureInference(InferenceCallProbe):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled_turns: list[str] = []

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        self.stream_called = True
        raise RuntimeError("inference creation failed")

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)


class SegmentSynthesizer:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.cancelled_turns: list[str] = []

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.texts.append(text)
        chunk_number = len(self.texts)
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"chunk_{chunk_number}",
            text=text,
            audio=AudioFrame(
                pcm=bytes((chunk_number, 0)),
                sample_rate_hz=16_000,
                channels=1,
            ),
        )

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)


def test_streaming_loop_installs_duration_bound_only_when_explicitly_enabled() -> None:
    provider = SegmentSynthesizer()
    bounded = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=InferenceCallProbe(),
        synthesizer=provider,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        max_speech_chunk_duration_seconds=3.0,
    )
    legacy = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=InferenceCallProbe(),
        synthesizer=provider,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )

    assert isinstance(bounded._synthesizer, DurationBoundedSynthesizer)
    assert legacy._synthesizer is provider


class DuplicateChunkSynthesizer(SegmentSynthesizer):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._duplicate_stream(text, turn_id)

    async def _duplicate_stream(
        self,
        text: str,
        turn_id: str,
    ) -> AsyncIterator[SpeechChunk]:
        del text
        for chunk_text in ("First subchunk.", "Duplicate subchunk."):
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id="chunk_duplicate",
                text=chunk_text,
                audio=AudioFrame(
                    pcm=b"\x01\x00",
                    sample_rate_hz=16_000,
                    channels=1,
                ),
            )


class TwoChunkSegmentSynthesizer(SegmentSynthesizer):
    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.texts.append(text)
        synthesis_number = len(self.texts)
        for chunk_number, chunk_text in enumerate(
            ("First subchunk.", "Second subchunk."),
            start=1,
        ):
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id=f"multi_{synthesis_number}_{chunk_number}",
                text=chunk_text,
                audio=AudioFrame(
                    pcm=bytes((chunk_number, 0)),
                    sample_rate_hz=16_000,
                    channels=1,
                ),
            )


class CreatingFailureSynthesizer(SegmentSynthesizer):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        raise RuntimeError("synthesis creation failed")


class IterationFailureSynthesizer(SegmentSynthesizer):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._failing_stream(text, turn_id)

    async def _failing_stream(
        self,
        text: str,
        turn_id: str,
    ) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        if False:
            yield cast(SpeechChunk, object())
        raise RuntimeError("synthesis iteration failed")


class WrongTurnSynthesizer(SegmentSynthesizer):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._wrong_turn_stream(text, turn_id)

    async def _wrong_turn_stream(
        self,
        text: str,
        turn_id: str,
    ) -> AsyncIterator[SpeechChunk]:
        self.texts.append(text)
        del turn_id
        yield SpeechChunk(
            turn_id="turn_other",
            chunk_id="chunk_wrong_turn",
            text=text,
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )


class MalformedChunkSynthesizer(SegmentSynthesizer):
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._malformed_stream(text, turn_id)

    async def _malformed_stream(
        self,
        text: str,
        turn_id: str,
    ) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        yield cast(SpeechChunk, object())


class MutatedChunkSynthesizer(SegmentSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.trap_touched = False

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._mutated_stream(text, turn_id)

    async def _mutated_stream(
        self,
        text: str,
        turn_id: str,
    ) -> AsyncIterator[SpeechChunk]:
        owner = self

        class TrapString(str):
            def __eq__(self, other: object) -> bool:
                del other
                owner.trap_touched = True
                raise AssertionError("hostile equality executed")

            def __ne__(self, other: object) -> bool:
                del other
                owner.trap_touched = True
                raise AssertionError("hostile inequality executed")

            __hash__ = str.__hash__

        chunk = SpeechChunk(
            turn_id=turn_id,
            chunk_id="chunk_mutated",
            text=text,
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )
        object.__setattr__(chunk, "turn_id", TrapString(turn_id))
        yield chunk


class CloseTrackingTextIterator:
    def __init__(self) -> None:
        self._yielded = False
        self.close_count = 0

    def __aiter__(self) -> "CloseTrackingTextIterator":
        return self

    async def __anext__(self) -> str:
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return "Answer."

    async def aclose(self) -> None:
        self.close_count += 1


class CloseTrackingInference:
    def __init__(self) -> None:
        self.iterator = CloseTrackingTextIterator()
        self.cancelled_turns: list[str] = []

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        return self.iterator

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)


class NonClosingTextIterator(CloseTrackingTextIterator):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def aclose(self) -> None:
        self.close_count += 1
        self.close_started.set()
        while not self.release_close.is_set():
            try:
                await self.release_close.wait()
            except asyncio.CancelledError:
                continue


class NonClosingInference(CloseTrackingInference):
    iterator: NonClosingTextIterator

    def __init__(self) -> None:
        super().__init__()
        self.iterator = NonClosingTextIterator()


class BlockingSynthesisIterator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_count = 0

    def __aiter__(self) -> "BlockingSynthesisIterator":
        return self

    async def __anext__(self) -> SpeechChunk:
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_count += 1


class BlockingSynthesisSynthesizer(SegmentSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.iterator = BlockingSynthesisIterator()

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        return self.iterator


class CloseTrackingChunkIterator:
    def __init__(self, chunk: SpeechChunk) -> None:
        self._chunk: SpeechChunk | None = chunk
        self.close_count = 0

    def __aiter__(self) -> "CloseTrackingChunkIterator":
        return self

    async def __anext__(self) -> SpeechChunk:
        if self._chunk is None:
            raise StopAsyncIteration
        chunk = self._chunk
        self._chunk = None
        return chunk

    async def aclose(self) -> None:
        self.close_count += 1


class CloseTrackingSynthesizer(SegmentSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.iterator: CloseTrackingChunkIterator | None = None

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.iterator = CloseTrackingChunkIterator(
            SpeechChunk(
                turn_id=turn_id,
                chunk_id="chunk_close_tracking",
                text=text,
                audio=AudioFrame(
                    pcm=b"\x01\x00",
                    sample_rate_hz=16_000,
                    channels=1,
                ),
            )
        )
        return self.iterator


class RecordingPlayback:
    def __init__(self) -> None:
        self.chunks: list[SpeechChunk] = []
        self.first_chunk_played = asyncio.Event()
        self.cancelled_turns: list[str] = []

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.chunks.append(chunk)
        self.first_chunk_played.set()

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)


class PerTurnInference:
    def __init__(self) -> None:
        self.cancelled_turns: list[str] = []

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot
        yield f"Answer for {turn_id}."

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)


class NonReturningCancelInference(PerTurnInference):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)
        self.cancel_started.set()
        while not self.release_cancel.is_set():
            try:
                await self.release_cancel.wait()
            except asyncio.CancelledError:
                continue


class EagerTwoSegmentInference(PerTurnInference):
    def __init__(self) -> None:
        super().__init__()
        self.second_segment_emitted = asyncio.Event()

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "First answer."
        self.second_segment_emitted.set()
        yield "Second answer."


class EagerOneSegmentInference(PerTurnInference):
    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "Complete answer."


class MatchingTwoChunkInference(PerTurnInference):
    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "First subchunk.Second subchunk."


class EagerThreeSegmentInference(PerTurnInference):
    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        for index in range(1, 4):
            yield f"Segment {index}."


class PrefetchProbeSynthesizer(SegmentSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.second_synthesized = asyncio.Event()
        self.third_synthesis_started = asyncio.Event()

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.texts.append(text)
        chunk_number = len(self.texts)
        if chunk_number == 2:
            self.second_synthesized.set()
        elif chunk_number == 3:
            self.third_synthesis_started.set()
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"prefetch_{chunk_number}",
            text=text,
            audio=AudioFrame(
                pcm=bytes((chunk_number, 0)),
                sample_rate_hz=16_000,
                channels=1,
            ),
        )


class CancellationResistantInference(PerTurnInference):
    def __init__(self) -> None:
        super().__init__()
        self.owner_cancelled = asyncio.Event()
        self.release_owner = asyncio.Event()

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "First answer."
        while not self.release_owner.is_set():
            try:
                await self.release_owner.wait()
            except asyncio.CancelledError:
                self.owner_cancelled.set()


class BlockingAfterFirstInference(PerTurnInference):
    def __init__(self) -> None:
        super().__init__()
        self.owner_cancelled = asyncio.Event()

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "First answer."
        try:
            await asyncio.Event().wait()
        finally:
            self.owner_cancelled.set()


class OversizedInference(PerTurnInference):
    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "oversized"


class ThreeSegmentInference(PerTurnInference):
    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        for segment in ("One.", "Two.", "Three."):
            yield segment


class BurstInference(PerTurnInference):
    def __init__(self, playback_started: asyncio.Event) -> None:
        super().__init__()
        self._playback_started = playback_started

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "Segment 0."
        await self._playback_started.wait()
        for index in range(1, 8):
            yield f"Segment {index}."


class BlockingFirstPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.first_play_started = asyncio.Event()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.chunks.append(chunk)
        if chunk.turn_id == "turn_001":
            self.first_play_started.set()
            await asyncio.Future()


class BlockingSecondSubchunkPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.second_subchunk_started = asyncio.Event()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.chunks.append(chunk)
        if chunk.turn_id == "turn_001" and chunk.chunk_id == "multi_1_2":
            self.second_subchunk_started.set()
            await asyncio.Future()


class BlockingOriginalAndReplacementPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.original_started = asyncio.Event()
        self.replacement_started = asyncio.Event()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.chunks.append(chunk)
        if chunk.turn_id == "turn_001":
            self.original_started.set()
            await asyncio.Future()
        if chunk.turn_id == "turn_new":
            self.replacement_started.set()
            await asyncio.Future()


class PresentationBlockingPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.play_started = asyncio.Event()
        self.release_play = asyncio.Event()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.chunks.append(chunk)
        self.play_started.set()
        await self.release_play.wait()
        if not is_valid():
            raise asyncio.CancelledError


class DelayedTransportPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.first_play_entered = asyncio.Event()
        self.first_play_cancelled = asyncio.Event()
        self.release_first_play = asyncio.Event()
        self.transport_writes: list[str] = []

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if chunk.turn_id == "turn_001":
            self.first_play_entered.set()
            try:
                await self.release_first_play.wait()
            except asyncio.CancelledError:
                self.first_play_cancelled.set()
                await self.release_first_play.wait()
        if not is_valid():
            raise asyncio.CancelledError
        self.transport_writes.append(chunk.turn_id)


class NonReturningPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.play_started = asyncio.Event()
        self.release_play = asyncio.Event()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        del chunk, is_valid
        self.play_started.set()
        while not self.release_play.is_set():
            try:
                await self.release_play.wait()
            except asyncio.CancelledError:
                continue


class CancellationResistantPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.play_started = asyncio.Event()
        self.play_cancelled = asyncio.Event()
        self.release_play = asyncio.Event()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.chunks.append(chunk)
        self.play_started.set()
        while not self.release_play.is_set():
            try:
                await self.release_play.wait()
            except asyncio.CancelledError:
                self.play_cancelled.set()

    async def cancel(self, turn_id: str) -> None:
        self.cancelled_turns.append(turn_id)
        self.cancel_started.set()
        await self.release_cancel.wait()


class FailingPlayback(RecordingPlayback):
    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        del chunk, is_valid
        raise RuntimeError("playback failed")


@pytest.mark.asyncio
async def test_interrupt_update_uses_foreground_delivery_without_inference() -> None:
    context = ConversationContextStore()
    inference = InferenceCallProbe()
    synthesizer = SegmentSynthesizer()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    decision = UpdateDecision(
        sequence=1,
        completion=TaskTerminalOutcome(
            task_id="task_weather",
            status="completed",
            summary="Rain starts in ten minutes.",
        ),
        kind=UpdateDecisionKind.INTERRUPT.value,
        text="Take an umbrella; rain starts in ten minutes.",
    )

    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "update_1")
    await loop._announce_update(authority, operation, "update_1", decision)

    assert inference.stream_called is False
    assert synthesizer.texts == ["Take an umbrella; rain starts in ten minutes."]
    assert [chunk.text for chunk in playback.chunks] == [
        "Take an umbrella; rain starts in ten minutes."
    ]
    assert [(message.role, message.text) for message in context.snapshot().messages] == [
        ("assistant", "Take an umbrella; rain starts in ten minutes.")
    ]
    assert loop._finish_update_operation(authority, operation) is True


@pytest.mark.asyncio
async def test_mention_update_is_framed_for_next_real_user_turn_only() -> None:
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_prior",
        run_id="deleg_prior",
        objective="Finish prior background work.",
    )
    context.record_task_completed(task_id="task_prior", run_id="deleg_prior")
    inference = RequestRecordingInference()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    mention = UpdateDecision(
        sequence=7,
        completion=TaskTerminalOutcome(
            task_id="task_report",
            status="completed",
            summary="The report is ready.",
        ),
        kind=UpdateDecisionKind.MENTION_NEXT.value,
        text="The requested report is ready.",
    )

    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_mention")
    await loop._respond_with_updates(
        authority,
        operation,
        "turn_mention",
        Transcript(text="What should we do next?", final=True),
        (mention,),
    )

    assert len(inference.requests) == 1
    assert inference.requests[0].terminal_task_count == 1
    assert inference.requests[0].updates == (
        ConversationPromptUpdate(
            sequence=7,
            task_id="task_report",
            status="completed",
            text="The requested report is ready.",
        ),
    )
    assert [(message.role, message.text) for message in inference.requests[0].context.messages] == [
        ("user", "What should we do next?")
    ]
    assert all(
        message.text != "The requested report is ready."
        for message in context.snapshot().messages
    )
    assert loop._finish_update_operation(authority, operation) is True


def test_update_delivery_proof_is_scoped_to_operation_incarnation() -> None:
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=InferenceCallProbe(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    authority = loop._bind_update_executor(object())
    first = loop._begin_update_operation(authority, "reused_turn")
    replacement = loop._begin_update_operation(authority, "reused_turn")

    loop._mark_update_operation_delivered(replacement)

    assert loop._finish_update_operation(authority, first) is False
    assert loop._finish_update_operation(authority, replacement) is True


def test_prompt_update_rejects_hostile_status_before_comparison() -> None:
    class HostileStatus(str):
        equality_calls = 0

        def __eq__(self, other: object) -> bool:
            del other
            type(self).equality_calls += 1
            raise AssertionError("hostile status comparison executed")

    with pytest.raises(TypeError, match="status"):
        ConversationPromptUpdate(
            sequence=1,
            task_id="task_safe",
            status=HostileStatus("completed"),
            text="The requested report is ready.",
        )

    assert HostileStatus.equality_calls == 0


@pytest.mark.asyncio
async def test_partial_transcript_is_rejected_before_inference_or_context() -> None:
    context = ConversationContextStore()
    inference = InferenceCallProbe()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )

    with pytest.raises(ValueError, match="only final transcripts"):
        await loop.respond("turn_001", Transcript(text="Partial", final=False))

    assert inference.stream_called is False
    assert context.snapshot().messages == ()
    assert playback.chunks == []


@pytest.mark.asyncio
async def test_inference_creation_failure_settles_owner_before_return() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    inference = CreatingFailureInference()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )

    with pytest.raises(RuntimeError, match="inference creation failed"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert inference.cancelled_turns == ["turn_001"]
    assert foreground.active_task_count == 0
    assert playback.chunks == []
    assert [message.text for message in context.snapshot().messages] == ["Question?"]


@pytest.mark.asyncio
async def test_assistant_partial_is_visible_before_delivery_confirmed_final() -> None:
    context = ConversationContextStore()
    playback = PresentationBlockingPlayback()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RequestRecordingInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: observed.append((kind, data)),
    )

    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.play_started.wait(), timeout=1)

    assert observed == [
        ("first_foreground_token", {}),
        (
            "assistant_text_generated",
            {
                "role": "assistant",
                "text": "I will mention the update now.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "segmentId": "segment_1",
            },
        ),
        ("first_playable_audio", {}),
        (
            "transcript_partial",
            {
                "role": "assistant",
                "text": "I will mention the update now.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_1",
                "segmentId": "segment_1",
                "segmentTextOffsetUtf16": 0,
            },
        ),
    ]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "I will mention the update now.",
    ]

    playback.release_play.set()
    await asyncio.wait_for(response, timeout=1)

    assert observed[-2] == (
        "transcript_final",
        {
            "role": "assistant",
            "text": "I will mention the update now.",
            "turnId": "turn_001",
            "turnGeneration": 1,
            "chunkId": "chunk_1",
            "segmentId": "segment_1",
        },
    )
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "I will mention the update now.",
    ]


@pytest.mark.asyncio
async def test_cancelled_assistant_turn_projects_exact_terminal_identity() -> None:
    playback = PresentationBlockingPlayback()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=RequestRecordingInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: observed.append((kind, data)),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.play_started.wait(), timeout=1)

    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response

    assert observed[-1] == (
        "assistant_turn_interrupted",
        {"turnId": "turn_001", "turnGeneration": 1},
    )


@pytest.mark.asyncio
async def test_conditional_cancel_preserves_stale_owner_and_revokes_exact_owner() -> None:
    playback = PresentationBlockingPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=RequestRecordingInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    response = asyncio.create_task(
        loop.respond("turn_002", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.play_started.wait(), timeout=1)

    assert loop.active_turn_id == "turn_002"
    assert await loop.cancel_if_active("turn_001") is False
    assert loop.active_turn_id == "turn_002"
    assert loop.foreground_active is True
    assert response.done() is False

    assert await loop.cancel_if_active("turn_002") is True
    with pytest.raises(asyncio.CancelledError):
        await response
    assert loop.active_turn_id is None
    assert loop.foreground_active is False


@pytest.mark.asyncio
async def test_generated_segment_capture_runs_after_publication_before_context_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
    )

    trace: list[object] = []
    context = ConversationContextStore()
    original_record = context.record_assistant_generation

    def record_context(text: str) -> None:
        trace.append(("context", text))
        original_record(text)

    monkeypatch.setattr(context, "record_assistant_generation", record_context)
    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000161")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000162")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(
        lease,
        "logical_session_id",
        "00000000-0000-4000-8000-000000000163",
    )

    def admit_generated(
        self: EvidenceAdmissionControllerV1,
        supplied_lease: object,
        text: str,
    ) -> AppendDisposition:
        del self
        assert supplied_lease is lease
        trace.append(("evidence", text))
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        admit_generated,
        raising=False,
    )
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=EagerOneSegmentInference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_evidence_generated")

    await loop._respond_with_updates(
        authority,
        operation,
        "turn_evidence_generated",
        Transcript(text="Question?", final=True),
        (),
        lease,
    )

    assert trace == [
        ("evidence", "Complete answer."),
        ("context", "Complete answer."),
    ]


@pytest.mark.asyncio
async def test_generated_capture_failure_cannot_change_context_or_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
    )

    context = ConversationContextStore()
    playback = RecordingPlayback()
    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000164")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000165")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(
        lease,
        "logical_session_id",
        "00000000-0000-4000-8000-000000000166",
    )

    def fail_capture(
        self: EvidenceAdmissionControllerV1,
        supplied_lease: object,
        text: str,
    ) -> None:
        del self, supplied_lease, text
        raise RuntimeError("injected generated capture failure")

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        fail_capture,
    )
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=EagerOneSegmentInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_capture_failure")

    await loop._respond_with_updates(
        authority,
        operation,
        "turn_capture_failure",
        Transcript(text="Question?", final=True),
        (),
        lease,
    )

    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "Complete answer.",
    ]
    assert len(playback.chunks) == 1


@pytest.mark.asyncio
async def test_success_terminal_order_is_transport_context_snapshot_close_settle_observe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.conversation import streaming as streaming_module
    from hermes_realtime.evidence import (
        AppendDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
    )

    trace: list[str] = []

    class CheckpointChannel:
        async def emit(self, checkpoint: str) -> None:
            trace.append(checkpoint)

    monkeypatch.setattr(
        streaming_module,
        "_current_qualification_checkpoint_channel",
        lambda: CheckpointChannel(),
        raising=False,
    )
    context = ConversationContextStore()
    ledger = DeliveredSpeechLedger()
    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000181")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000182")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000183")

    original_confirm = ledger.mark_delivered_confirmed
    original_text = ledger.confirmed_text
    original_context = context.record_assistant_delivery
    original_snapshot = ledger.snapshot_turn_counts
    original_close = ledger.close_turn

    def mark_confirmed(receipt: object) -> object:
        trace.append("confirmation")
        return original_confirm(receipt)  # type: ignore[arg-type]

    def confirmed_text(confirmation: object, delivery_admission: object) -> str:
        trace.append("confirmed_text")
        return original_text(confirmation, delivery_admission)  # type: ignore[arg-type]

    def admit_transport(self: object, *args: object, **kwargs: object) -> AppendDisposition:
        del self, args, kwargs
        trace.append("capture")
        return AppendDisposition.ADMITTED

    def record_context(**kwargs: object) -> str:
        trace.append("context")
        return original_context(**kwargs)  # type: ignore[arg-type]

    def snapshot_counts(turn_id: str) -> object:
        trace.append("snapshot")
        return original_snapshot(turn_id)

    def close_turn(turn_id: str) -> None:
        trace.append("close")
        original_close(turn_id)

    def settle(self: object, *args: object, **kwargs: object) -> AppendDisposition:
        del self, args, kwargs
        trace.append("settle")
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(ledger, "mark_delivered_confirmed", mark_confirmed)
    monkeypatch.setattr(ledger, "confirmed_text", confirmed_text)
    monkeypatch.setattr(context, "record_assistant_delivery", record_context)
    monkeypatch.setattr(ledger, "snapshot_turn_counts", snapshot_counts)
    monkeypatch.setattr(ledger, "close_turn", close_turn)
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_transport_confirmed_full",
        admit_transport,
    )
    monkeypatch.setattr(EvidenceAdmissionControllerV1, "settle_completed", settle)
    monkeypatch.setattr(
        StreamingSpeechLoop,
        "_observe_terminal_settlement",
        lambda *_args, **_kwargs: True,
    )

    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=EagerOneSegmentInference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=ledger,
        evidence_admission=admission,
        observer=lambda kind, data: trace.append(kind),
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_terminal_order")
    await loop._respond_with_updates(
        authority,
        operation,
        "turn_terminal_order",
        Transcript(text="Question?", final=True),
        (),
        lease,
    )

    critical = [
        item
        for item in trace
        if item
        in {
            "confirmation",
            "confirmed_text",
            "capture",
            "context",
            "snapshot",
            "close",
            "settle",
            "host_response_completed_before_shutdown",
            "assistant_turn_completed",
        }
    ]
    assert critical == [
        "confirmation",
        "confirmed_text",
        "capture",
        "context",
        "confirmed_text",
        "confirmed_text",
        "snapshot",
        "close",
        "settle",
        "host_response_completed_before_shutdown",
        "assistant_turn_completed",
    ]

    checkpoint_count = trace.count("host_response_completed_before_shutdown")
    await loop.respond("turn_without_evidence", Transcript(text="Question?", final=True))
    assert trace.count("host_response_completed_before_shutdown") == checkpoint_count

    def fail_settlement(self: object, *args: object, **kwargs: object) -> AppendDisposition:
        del self, args, kwargs
        raise RuntimeError("injected settlement failure")

    monkeypatch.setattr(EvidenceAdmissionControllerV1, "settle_completed", fail_settlement)
    failed_operation = loop._begin_update_operation(authority, "turn_failed_settlement")
    await loop._respond_with_updates(
        authority,
        failed_operation,
        "turn_failed_settlement",
        Transcript(text="Question?", final=True),
        (),
        lease,
    )
    assert trace.count("host_response_completed_before_shutdown") == checkpoint_count

    monkeypatch.setattr(EvidenceAdmissionControllerV1, "settle_completed", settle)

    def fail_observation(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("injected settlement observation failure")

    monkeypatch.setattr(
        StreamingSpeechLoop,
        "_observe_terminal_settlement",
        fail_observation,
    )
    failed_observation = loop._begin_update_operation(authority, "turn_failed_observation")
    await loop._respond_with_updates(
        authority,
        failed_observation,
        "turn_failed_observation",
        Transcript(text="Question?", final=True),
        (),
        lease,
    )
    assert trace.count("host_response_completed_before_shutdown") == checkpoint_count


@pytest.mark.asyncio
async def test_close_failure_after_context_commit_records_failure_before_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    ledger = DeliveredSpeechLedger()
    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    cause = object.__new__(TerminalCauseCapabilityV1)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000221")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000222")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000223")
    object.__setattr__(lease, "terminal_cause", cause)
    causes: list[TerminalReason] = []
    settlements: list[bool] = []
    original_close = ledger.close_turn
    close_calls = 0

    def close_turn(turn_id: str) -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("injected ledger close failure")
        original_close(turn_id)

    monkeypatch.setattr(ledger, "close_turn", close_turn)
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_transport_confirmed_full",
        lambda self, *args, **kwargs: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        lambda self, capability, reason: causes.append(reason) or CauseDisposition.RECORDED,
    )
    def settle_terminal(
        self: object,
        supplied_lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self, supplied_lease
        settlements.append(bool(kwargs["assistant_delivery_context_recorded"]))
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )

    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=EagerOneSegmentInference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=ledger,
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_close_failure")

    with pytest.raises(RuntimeError, match="injected ledger close failure"):
        await loop._respond_with_updates(
            authority,
            operation,
            "turn_close_failure",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )

    assert causes == [TerminalReason.LEDGER_CLOSE_FAILED]
    assert settlements == [True]
    assert close_calls == 2


@pytest.mark.asyncio
async def test_generated_observer_failure_does_not_leave_orphaned_assistant_context() -> None:
    context = ConversationContextStore()

    def reject_generated(kind: str, data: dict[str, str | int | bool | None]) -> None:
        del data
        if kind == "assistant_text_generated":
            raise RuntimeError("projection rejected generated text")

    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=EagerOneSegmentInference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        observer=reject_generated,
    )

    with pytest.raises(RuntimeError, match="projection rejected generated text"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert [message.text for message in context.snapshot().messages] == ["Question?"]


@pytest.mark.asyncio
async def test_interruption_preserves_every_generated_segment_in_context_and_projection() -> None:
    context = ConversationContextStore()
    playback = PresentationBlockingPlayback()
    inference = EagerTwoSegmentInference()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: observed.append((kind, data)),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.play_started.wait(), timeout=1)
    await asyncio.wait_for(inference.second_segment_emitted.wait(), timeout=1)

    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response

    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "First answer.",
        "Second answer.",
    ]
    generated = [event for event in observed if event[0] == "assistant_text_generated"]
    assert generated == [
        (
            "assistant_text_generated",
            {
                "role": "assistant",
                "text": "First answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "segmentId": "segment_1",
            },
        ),
        (
            "assistant_text_generated",
            {
                "role": "assistant",
                "text": "Second answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "segmentId": "segment_2",
            },
        ),
    ]
    assert observed[-1][0] == "assistant_turn_interrupted"


@pytest.mark.asyncio
async def test_interrupted_evidence_response_retains_canonical_replay_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
    )

    context = ConversationContextStore()
    inference = EagerTwoSegmentInference()
    playback = BlockingFirstPlayback()
    captured: list[str] = []
    admission = object.__new__(EvidenceAdmissionControllerV1)

    def capture_generated(
        self: EvidenceAdmissionControllerV1,
        lease: EvidenceTurnLease,
        text: str,
    ) -> AppendDisposition:
        del self, lease
        captured.append(text)
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        capture_generated,
    )
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    evidence_turn_id = str(UUID(int=1_271, version=4))
    evidence_lease = object.__new__(EvidenceTurnLease)
    object.__setattr__(evidence_lease, "evidence_turn_id", evidence_turn_id)
    object.__setattr__(evidence_lease, "binding_id", str(UUID(int=1_272, version=4)))
    object.__setattr__(evidence_lease, "binding_generation", 1)
    object.__setattr__(
        evidence_lease,
        "logical_session_id",
        str(UUID(int=1_273, version=4)),
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_001")

    response = asyncio.create_task(
        loop._respond_with_updates(
            authority,
            operation,
            "turn_001",
            Transcript(text="Question?", final=True),
            (),
            evidence_lease,
        )
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)
    await asyncio.wait_for(inference.second_segment_emitted.wait(), timeout=1)
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response
    loop._finish_update_operation(authority, operation)

    replay = loop._eligible_replay_identity()
    assert replay is not None
    assert replay.replay_of_evidence_turn_id == evidence_turn_id
    assert replay.replay_generation == 1
    assert replay.replay_of_evidence_turn_id != "turn_001"
    assert captured == ["First answer.", "Second answer."]

    assert await loop.resume_interrupted(evidence_lease) is True
    assert captured == ["First answer.", "Second answer."]


@pytest.mark.asyncio
async def test_resume_interrupted_replays_unconfirmed_suffix_without_new_inference() -> None:
    context = ConversationContextStore()
    inference = EagerTwoSegmentInference()
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )

    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)
    await asyncio.wait_for(inference.second_segment_emitted.wait(), timeout=1)

    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    original_settle_consumer = loop._settle_consumer

    async def gated_settle_consumer(
        _self: StreamingSpeechLoop,
        consumer: asyncio.Task[None],
    ) -> None:
        cleanup_entered.set()
        await release_cleanup.wait()
        await original_settle_consumer(consumer)

    loop._settle_consumer = MethodType(  # type: ignore[method-assign]
        gated_settle_consumer,
        loop,
    )
    cancellation = asyncio.create_task(loop.cancel())
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1)

    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "mention_during_resume_window")
    decision = UpdateDecision(
        sequence=1,
        completion=TaskTerminalOutcome(
            task_id="task_during_resume_window",
            status="completed",
            summary="Background result ready.",
        ),
        kind=UpdateDecisionKind.MENTION_NEXT.value,
        text="Background result ready.",
    )
    assert (
        await loop._announce_idle_update(
            authority,
            operation,
            "mention_during_resume_window",
            decision,
        )
        is False
    )
    assert loop._finish_update_operation(authority, operation) is False

    release_cleanup.set()
    await cancellation
    loop._settle_consumer = original_settle_consumer  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await response

    def reject_inference(*args: object, **kwargs: object) -> AsyncIterator[str]:
        del args, kwargs
        raise AssertionError("resume must not invoke inference")

    inference.stream = reject_inference  # type: ignore[method-assign]
    assert await loop.resume_interrupted() is True
    assert await loop.resume_interrupted() is False

    assert [chunk.text for chunk in playback.chunks] == [
        "First answer.",
        "First answer.",
        "Second answer.",
    ]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "First answer.",
        "Second answer.",
    ]


@pytest.mark.asyncio
async def test_resume_replays_publication_when_later_speech_subchunk_was_interrupted() -> None:
    context = ConversationContextStore()
    inference = MatchingTwoChunkInference()
    playback = BlockingSecondSubchunkPlayback()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=TwoChunkSegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: observed.append((kind, data)),
    )

    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.second_subchunk_started.wait(), timeout=1)
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response

    def reject_inference(*args: object, **kwargs: object) -> AsyncIterator[str]:
        del args, kwargs
        raise AssertionError("resume must not invoke inference")

    inference.stream = reject_inference  # type: ignore[method-assign]
    assert await loop.resume_interrupted() is True

    assert [chunk.text for chunk in playback.chunks] == [
        "First subchunk.",
        "Second subchunk.",
        "Second subchunk.",
    ]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "First subchunk.Second subchunk.",
    ]
    assert [event for event in observed if event[0] == "assistant_text_generated"] == [
        (
            "assistant_text_generated",
            {
                "role": "assistant",
                "text": "First subchunk.Second subchunk.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "segmentId": "segment_1",
            },
        )
    ]
    partials = [data for kind, data in observed if kind == "transcript_partial"]
    assert [
        (data["turnId"], data["turnGeneration"], data["segmentId"])
        for data in partials
    ] == [("turn_001", 1, "segment_1")] * 3
    assert [data["segmentTextOffsetUtf16"] for data in partials] == [
        0,
        len("First subchunk."),
        len("First subchunk."),
    ]
    lifecycle_events = [
        data
        for kind, data in observed
        if kind in {"assistant_turn_completed", "assistant_turn_interrupted"}
    ]
    assert lifecycle_events
    assert {
        (data["turnId"], data["turnGeneration"]) for data in lifecycle_events
    } == {("turn_001", 1)}


@pytest.mark.asyncio
async def test_repeated_interruption_preserves_presentation_and_cumulative_audio_prefix() -> None:
    class RepeatedInterruptionPlayback(RecordingPlayback):
        def __init__(self) -> None:
            super().__init__()
            self.second_chunk_attempts = 0
            self.first_interruption = asyncio.Event()
            self.second_interruption = asyncio.Event()

        async def play(
            self,
            chunk: SpeechChunk,
            *,
            is_valid: Callable[[], bool],
        ) -> None:
            if not is_valid():
                raise asyncio.CancelledError
            self.chunks.append(chunk)
            if chunk.text != "Second subchunk.":
                return
            self.second_chunk_attempts += 1
            if self.second_chunk_attempts == 1:
                self.first_interruption.set()
                await asyncio.Future()
            if self.second_chunk_attempts == 2:
                self.second_interruption.set()
                await asyncio.Future()

    playback = RepeatedInterruptionPlayback()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=MatchingTwoChunkInference(),
        synthesizer=TwoChunkSegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: observed.append((kind, data)),
    )

    original = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_interruption.wait(), timeout=1)
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original

    first_resume = asyncio.create_task(loop.resume_interrupted())
    await asyncio.wait_for(playback.second_interruption.wait(), timeout=1)
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_resume

    assert await loop.resume_interrupted() is True
    assert [chunk.text for chunk in playback.chunks] == [
        "First subchunk.",
        "Second subchunk.",
        "Second subchunk.",
        "Second subchunk.",
    ]
    lifecycle = [
        data
        for kind, data in observed
        if kind in {"transcript_partial", "assistant_turn_completed", "assistant_turn_interrupted"}
    ]
    assert lifecycle
    assert {(data["turnId"], data["turnGeneration"]) for data in lifecycle} == {
        ("turn_001", 1)
    }


@pytest.mark.asyncio
async def test_resume_does_not_cancel_newer_response_admitted_during_race() -> None:
    inference = EagerOneSegmentInference()
    playback = BlockingOriginalAndReplacementPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    original = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Original?", final=True))
    )
    await asyncio.wait_for(playback.original_started.wait(), timeout=1)
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original

    resume_waiting = asyncio.Event()
    release_resume = asyncio.Event()
    original_run_turn = loop._run_turn

    async def gated_run_turn(self: StreamingSpeechLoop, turn_id: str, **kwargs: object) -> bool:
        del self
        if turn_id.startswith("resume_"):
            resume_waiting.set()
            await release_resume.wait()
        return await original_run_turn(turn_id, **kwargs)  # type: ignore[arg-type]

    loop._run_turn = MethodType(gated_run_turn, loop)  # type: ignore[method-assign]
    resume = asyncio.create_task(loop.resume_interrupted())
    await asyncio.wait_for(resume_waiting.wait(), timeout=1)

    replacement = asyncio.create_task(
        loop.respond("turn_new", Transcript(text="Replacement?", final=True))
    )
    await asyncio.wait_for(playback.replacement_started.wait(), timeout=1)
    release_resume.set()

    assert await resume is False
    assert replacement.done() is False
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement


@pytest.mark.asyncio
async def test_assistant_projection_carries_stable_turn_and_chunk_identity() -> None:
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=EagerTwoSegmentInference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: observed.append((kind, data)),
    )

    await loop.respond("turn_001", Transcript(text="Question?", final=True))

    transcript_events = [
        event for event in observed if event[0] in {"transcript_partial", "transcript_final"}
    ]
    assert transcript_events == [
        (
            "transcript_partial",
            {
                "role": "assistant",
                "text": "First answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_1",
                "segmentId": "segment_1",
                "segmentTextOffsetUtf16": 0,
            },
        ),
        (
            "transcript_final",
            {
                "role": "assistant",
                "text": "First answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_1",
                "segmentId": "segment_1",
            },
        ),
        (
            "transcript_partial",
            {
                "role": "assistant",
                "text": "Second answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_2",
                "segmentId": "segment_2",
                "segmentTextOffsetUtf16": 0,
            },
        ),
        (
            "transcript_final",
            {
                "role": "assistant",
                "text": "Second answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_2",
                "segmentId": "segment_2",
            },
        ),
    ]
    assert observed[-1] == (
        "assistant_turn_completed",
        {"turnId": "turn_001", "turnGeneration": 1},
    )


@pytest.mark.asyncio
async def test_final_transcript_streams_first_audio_before_inference_completes() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    inference = BlockingIncrementalInference()
    synthesizer = SegmentSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []

    def observe(kind: str, data: dict[str, str | int | bool | None]) -> None:
        observed.append((kind, data))

    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
        observer=observe,
    )

    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_chunk_played.wait(), timeout=1)

    assert inference.completed is False
    assert response.done() is False
    assert observed[:3] == [
        ("first_foreground_token", {}),
        (
            "assistant_text_generated",
            {
                "role": "assistant",
                "text": "First answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "segmentId": "segment_1",
            },
        ),
        ("first_playable_audio", {}),
    ]
    assert [chunk.text for chunk in playback.chunks] == ["First answer."]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "First answer.",
    ]

    inference.release_second_segment.set()
    await asyncio.wait_for(response, timeout=1)

    assert inference.completed is True
    assert synthesizer.texts == ["First answer.", "Second answer."]
    assert [event for event in observed if event[0] != "assistant_text_generated"] == [
        ("first_foreground_token", {}),
        ("first_playable_audio", {}),
        (
            "transcript_partial",
            {
                "role": "assistant",
                "text": "First answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_1",
                "segmentId": "segment_1",
                "segmentTextOffsetUtf16": 0,
            },
        ),
        (
            "transcript_final",
            {
                "role": "assistant",
                "text": "First answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_1",
                "segmentId": "segment_1",
            },
        ),
        (
            "transcript_partial",
            {
                "role": "assistant",
                "text": "Second answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_2",
                "segmentId": "segment_2",
                "segmentTextOffsetUtf16": 0,
            },
        ),
        (
            "transcript_final",
            {
                "role": "assistant",
                "text": "Second answer.",
                "turnId": "turn_001",
                "turnGeneration": 1,
                "chunkId": "chunk_2",
                "segmentId": "segment_2",
            },
        ),
        (
            "assistant_turn_completed",
            {"turnId": "turn_001", "turnGeneration": 1},
        ),
    ]
    assert [chunk.text for chunk in playback.chunks] == [
        "First answer.",
        "Second answer.",
    ]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "First answer.",
        "Second answer.",
    ]
    assert foreground.active_task_count == 0
    assert ledger.retained_chunk_count == 0


@pytest.mark.asyncio
async def test_replacement_cancels_providers_and_releases_undelivered_speech() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    synthesizer = SegmentSynthesizer()
    playback = BlockingFirstPlayback()
    ledger = DeliveredSpeechLedger()
    inference = PerTurnInference()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )

    first = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="First question?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    await asyncio.wait_for(
        loop.respond("turn_002", Transcript(text="Second question?", final=True)),
        timeout=1,
    )
    with pytest.raises(asyncio.CancelledError):
        await first

    assert inference.cancelled_turns == ["turn_001"]
    assert synthesizer.cancelled_turns == ["turn_001"]
    assert playback.cancelled_turns == ["turn_001"]
    assert ledger.retained_chunk_count == 0
    assert ledger.pending() == ()
    assert await loop.resume_interrupted() is False
    assert [message.text for message in context.snapshot().messages] == [
        "First question?",
        "Answer for turn_001.",
        "Second question?",
        "Answer for turn_002.",
    ]


@pytest.mark.asyncio
async def test_evidence_backed_replacement_records_cause_before_old_turn_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    playback = BlockingFirstPlayback()
    ledger = DeliveredSpeechLedger()
    admission = object.__new__(EvidenceAdmissionControllerV1)
    first_lease = object.__new__(EvidenceTurnLease)
    second_lease = object.__new__(EvidenceTurnLease)
    first_cause = object.__new__(TerminalCauseCapabilityV1)
    second_cause = object.__new__(TerminalCauseCapabilityV1)
    for lease, suffix, cause in (
        (first_lease, "201", first_cause),
        (second_lease, "211", second_cause),
    ):
        object.__setattr__(
            lease,
            "evidence_turn_id",
            f"00000000-0000-4000-8000-000000000{suffix}",
        )
        object.__setattr__(
            lease,
            "binding_id",
            f"00000000-0000-4000-8000-000000000{int(suffix) + 1}",
        )
        object.__setattr__(lease, "binding_generation", 1)
        object.__setattr__(
            lease,
            "logical_session_id",
            f"00000000-0000-4000-8000-000000000{int(suffix) + 2}",
        )
        object.__setattr__(lease, "terminal_cause", cause)

    causes: list[tuple[object, TerminalReason]] = []
    settlements: list[tuple[object, int, int]] = []

    def record_cause(
        self: object,
        capability: object,
        reason: TerminalReason,
    ) -> CauseDisposition:
        del self
        causes.append((capability, reason))
        return CauseDisposition.RECORDED

    def settle_terminal(
        self: object,
        lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self
        settlements.append(
            (
                lease,
                int(kwargs["queued_chunk_count"]),
                int(kwargs["started_chunk_count"]),
            )
        )
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_transport_confirmed_full",
        lambda self, *args, **kwargs: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        record_cause,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_completed",
        lambda self, *args, **kwargs: AppendDisposition.ADMITTED,
    )

    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=ledger,
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    first_operation = loop._begin_update_operation(authority, "turn_001")
    first = asyncio.create_task(
        loop._respond_with_updates(
            authority,
            first_operation,
            "turn_001",
            Transcript(text="First question?", final=True),
            (),
            first_lease,
        )
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    second_operation = loop._begin_update_operation(authority, "turn_002")
    await asyncio.wait_for(
        loop._respond_with_updates(
            authority,
            second_operation,
            "turn_002",
            Transcript(text="Second question?", final=True),
            (),
            second_lease,
        ),
        timeout=1,
    )
    with pytest.raises(asyncio.CancelledError):
        await first

    assert causes == [(first_cause, TerminalReason.RESPONSE_REPLACED)]
    assert settlements == [(first_lease, 1, 1)]


@pytest.mark.asyncio
async def test_evidence_backed_conditional_cancel_settles_as_barge_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    cause = object.__new__(TerminalCauseCapabilityV1)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000231")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000232")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000233")
    object.__setattr__(lease, "terminal_cause", cause)
    reasons: list[TerminalReason] = []
    settlements: list[tuple[int, int]] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        lambda self, capability, reason: reasons.append(reason)
        or CauseDisposition.RECORDED,
    )

    def settle_terminal(
        self: object,
        supplied_lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self, supplied_lease
        settlements.append(
            (
                int(kwargs["queued_chunk_count"]),
                int(kwargs["started_chunk_count"]),
            )
        )
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_001")
    response = asyncio.create_task(
        loop._respond_with_updates(
            authority,
            operation,
            "turn_001",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    assert await loop.cancel_if_active("turn_001") is True
    with pytest.raises(asyncio.CancelledError):
        await response

    assert reasons == [TerminalReason.BARGE_IN]
    assert settlements == [(1, 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancel_method", "expected_reason"),
    (
        ("cancel", "STOP_SPEAKING"),
        ("cancel_for_binding_close", "BINDING_CLOSED"),
        ("cancel_for_retention_expiry", "RETENTION_EXPIRED"),
    ),
)
async def test_evidence_backed_owned_cancel_settles_with_exact_reason(
    monkeypatch: pytest.MonkeyPatch,
    cancel_method: str,
    expected_reason: str,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    cause = object.__new__(TerminalCauseCapabilityV1)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000241")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000242")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000243")
    object.__setattr__(lease, "terminal_cause", cause)
    reasons: list[TerminalReason] = []
    settlements: list[tuple[int, int]] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        lambda self, capability, reason: reasons.append(reason)
        or CauseDisposition.RECORDED,
    )

    def settle_terminal(
        self: object,
        supplied_lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self, supplied_lease
        settlements.append(
            (
                int(kwargs["queued_chunk_count"]),
                int(kwargs["started_chunk_count"]),
            )
        )
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_001")
    response = asyncio.create_task(
        loop._respond_with_updates(
            authority,
            operation,
            "turn_001",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    await getattr(loop, cancel_method)()
    with pytest.raises(asyncio.CancelledError):
        await response

    assert reasons == [TerminalReason[expected_reason]]
    assert settlements == [(1, 1)]


@pytest.mark.asyncio
async def test_evidence_backed_loop_close_settles_as_host_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    cause = object.__new__(TerminalCauseCapabilityV1)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000251")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000252")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000253")
    object.__setattr__(lease, "terminal_cause", cause)
    reasons: list[TerminalReason] = []
    settlements: list[tuple[int, int]] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        lambda self, capability, reason: reasons.append(reason)
        or CauseDisposition.RECORDED,
    )

    def settle_terminal(
        self: object,
        supplied_lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self, supplied_lease
        settlements.append(
            (
                int(kwargs["queued_chunk_count"]),
                int(kwargs["started_chunk_count"]),
            )
        )
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_001")
    response = asyncio.create_task(
        loop._respond_with_updates(
            authority,
            operation,
            "turn_001",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    await loop.close()
    with pytest.raises(asyncio.CancelledError):
        await response

    assert reasons == [TerminalReason.HOST_SHUTDOWN]
    assert settlements == [(1, 1)]


@pytest.mark.asyncio
async def test_evidence_backed_response_task_cancel_settles_as_caller_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    cause = object.__new__(TerminalCauseCapabilityV1)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000261")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000262")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000263")
    object.__setattr__(lease, "terminal_cause", cause)
    reasons: list[TerminalReason] = []
    settlements: list[tuple[int, int]] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        lambda self, capability, reason: reasons.append(reason)
        or CauseDisposition.RECORDED,
    )

    def settle_terminal(
        self: object,
        supplied_lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self, supplied_lease
        settlements.append(
            (
                int(kwargs["queued_chunk_count"]),
                int(kwargs["started_chunk_count"]),
            )
        )
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_001")
    response = asyncio.create_task(
        loop._respond_with_updates(
            authority,
            operation,
            "turn_001",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    response.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response

    assert reasons == [TerminalReason.CALLER_CANCELLED]
    assert settlements == [(1, 1)]


@pytest.mark.asyncio
async def test_grouped_synthesis_failures_record_provider_before_terminal_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )

    class GroupedFailureStream:
        def __aiter__(self):
            return self

        async def __anext__(self) -> SpeechChunk:
            raise RuntimeError("synthesis iteration failed")

        async def aclose(self) -> None:
            raise ValueError("synthesis close failed")

    class GroupedFailureSynthesizer(SegmentSynthesizer):
        def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
            del text, turn_id
            return cast(AsyncIterator[SpeechChunk], GroupedFailureStream())

    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    cause = object.__new__(TerminalCauseCapabilityV1)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000271")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000272")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000273")
    object.__setattr__(lease, "terminal_cause", cause)
    trace: list[str] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )

    def record_cause(
        self: object,
        capability: object,
        reason: TerminalReason,
    ) -> CauseDisposition:
        del self, capability
        trace.append(f"cause:{reason.value}")
        return CauseDisposition.RECORDED

    def settle_terminal(
        self: object,
        supplied_lease: object,
        **kwargs: object,
    ) -> AppendDisposition:
        del self, supplied_lease, kwargs
        trace.append("settle")
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        record_cause,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        settle_terminal,
    )
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=GroupedFailureSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_grouped")

    with pytest.raises(BaseExceptionGroup) as captured:
        await loop._respond_with_updates(
            authority,
            operation,
            "turn_grouped",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )

    flattened: list[BaseException] = []

    def flatten(error: BaseException) -> None:
        if isinstance(error, BaseExceptionGroup):
            for nested in error.exceptions:
                flatten(nested)
        else:
            flattened.append(error)

    flatten(captured.value)
    assert any(str(error) == "synthesis iteration failed" for error in flattened)
    assert any(str(error) == "synthesis close failed" for error in flattened)
    assert trace == [f"cause:{TerminalReason.PROVIDER_FAILED.value}", "settle"]


@pytest.mark.asyncio
async def test_inference_continues_into_bounded_queue_while_playback_is_blocked() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator(output_capacity=2)
    inference = EagerTwoSegmentInference()
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    observed_second_segment = False
    try:
        await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)
        await asyncio.wait_for(inference.second_segment_emitted.wait(), timeout=0.05)
        observed_second_segment = True
    finally:
        await loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response

    assert observed_second_segment is True


@pytest.mark.asyncio
async def test_synthesis_prefetches_exactly_one_chunk_during_current_playback() -> None:
    synthesizer = PrefetchProbeSynthesizer()
    playback = PresentationBlockingPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(output_capacity=3),
        inference=EagerThreeSegmentInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    try:
        await asyncio.wait_for(playback.play_started.wait(), timeout=1)
        await asyncio.wait_for(synthesizer.second_synthesized.wait(), timeout=1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(synthesizer.third_synthesis_started.wait(), timeout=0.05)
        assert [chunk.text for chunk in playback.chunks] == ["Segment 1."]
    finally:
        playback.release_play.set()
        await asyncio.wait_for(response, timeout=1)

    assert synthesizer.texts == ["Segment 1.", "Segment 2.", "Segment 3."]
    assert [chunk.text for chunk in playback.chunks] == [
        "Segment 1.",
        "Segment 2.",
        "Segment 3.",
    ]


@pytest.mark.asyncio
async def test_cancel_caller_cancellation_waits_for_cleanup_and_is_preserved() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    inference = PerTurnInference()
    synthesizer = SegmentSynthesizer()
    playback = CancellationResistantPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.play_started.wait(), timeout=1)

    cancellation = asyncio.create_task(loop.cancel())
    await asyncio.wait_for(playback.play_cancelled.wait(), timeout=1)
    cancellation.cancel()
    playback.release_play.set()
    await asyncio.wait_for(playback.cancel_started.wait(), timeout=1)
    assert cancellation.done() is False

    playback.release_cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancellation, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await response

    assert inference.cancelled_turns == ["turn_001"]
    assert synthesizer.cancelled_turns == ["turn_001"]
    assert playback.cancelled_turns == ["turn_001"]
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert foreground.active_task_count == 0


@pytest.mark.asyncio
async def test_qualification_trace_records_cleanup_for_each_independent_foreground_cancellation(
) -> None:
    """Each cancelled foreground turn has its own cleanup outcome in owner evidence."""

    from hermes_realtime.evidence import TerminalReason
    from tests.support.qualification import InProcessQualificationComposition

    class BlockingPlaybackWithSecondCleanupFailure(RecordingPlayback):
        def __init__(self) -> None:
            super().__init__()
            self.started: dict[str, asyncio.Event] = {}
            self.cancel_count = 0

        async def play(
            self,
            chunk: SpeechChunk,
            *,
            is_valid: Callable[[], bool],
        ) -> None:
            if not is_valid():
                raise asyncio.CancelledError
            self.chunks.append(chunk)
            self.started.setdefault(chunk.turn_id, asyncio.Event()).set()
            await asyncio.Future()

        async def cancel(self, turn_id: str) -> None:
            self.cancelled_turns.append(turn_id)
            self.cancel_count += 1
            if self.cancel_count == 2:
                raise RuntimeError("second foreground cleanup failed")

    qualification = InProcessQualificationComposition()
    playback = BlockingPlaybackWithSecondCleanupFailure()
    with qualification.wire():
        loop = StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=playback,
            ledger=DeliveredSpeechLedger(),
        )

    first = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="First?", final=True))
    )
    await asyncio.wait_for(
        playback.started.setdefault("turn_001", asyncio.Event()).wait(), timeout=1
    )
    await loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(
        loop.respond("turn_002", Transcript(text="Second?", final=True))
    )
    await asyncio.wait_for(
        playback.started.setdefault("turn_002", asyncio.Event()).wait(), timeout=1
    )
    with pytest.raises(BaseExceptionGroup):
        await loop.close()
    with pytest.raises(BaseExceptionGroup):
        await second

    facts = [
        (record.kind.value, getattr(record, "reason", getattr(record, "succeeded", None)))
        for record in qualification.trace.records()
        if record.kind.value in {"cancellation", "foreground_cleanup"}
    ]

    first_cancellation = facts.index(("cancellation", TerminalReason.STOP_SPEAKING))
    second_cancellation = facts.index(("cancellation", TerminalReason.HOST_SHUTDOWN))
    assert first_cancellation < second_cancellation
    assert ("foreground_cleanup", True) in facts[first_cancellation + 1 : second_cancellation]
    assert ("foreground_cleanup", False) in facts[second_cancellation + 1 :]


@pytest.mark.asyncio
async def test_authority_revocation_unlocks_provider_cleanup_and_close_supersedes_waiter() -> None:
    class CleanupBarrierPlayback(BlockingFirstPlayback):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_entered = asyncio.Event()
            self.release_cancel = asyncio.Event()
            self.loop: StreamingSpeechLoop | None = None

        async def cancel(self, turn_id: str) -> None:
            self.cancelled_turns.append(turn_id)
            assert self.loop is not None
            assert not self.loop._lifecycle_lock.locked()
            self.cancel_entered.set()
            await self.release_cancel.wait()

    foreground = ForegroundTurnCoordinator()
    playback = CleanupBarrierPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=foreground,
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    playback.loop = loop
    first = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="First?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    cancellation = asyncio.create_task(loop.cancel())
    await asyncio.wait_for(playback.cancel_entered.wait(), timeout=1)
    assert loop.active_turn_id is None
    assert foreground.active_turn_id is None
    replacement = asyncio.create_task(
        loop.respond("turn_002", Transcript(text="Second?", final=True))
    )
    await asyncio.sleep(0)
    assert replacement.done() is False

    closer = asyncio.create_task(loop.close())
    playback.release_cancel.set()
    await asyncio.wait_for(cancellation, timeout=1)
    await asyncio.wait_for(closer, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(replacement, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=1)
    with pytest.raises(RuntimeError, match="streaming speech loop is closed"):
        await loop.respond("turn_003", Transcript(text="Late?", final=True))


@pytest.mark.asyncio
async def test_respond_cancellation_preserves_racing_producer_failure() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    inference = CancellationRacingFailureInference()
    playback = BlockingFirstPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=ledger,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)
    inference.release_failure.set()
    await asyncio.wait_for(inference.failure_ready.wait(), timeout=1)

    response.cancel()
    inference.raise_failure.set()
    with pytest.raises(BaseExceptionGroup) as captured:
        await response

    def flatten(error: BaseException) -> list[BaseException]:
        if isinstance(error, BaseExceptionGroup):
            return [item for nested in error.exceptions for item in flatten(nested)]
        return [error]

    failures = flatten(captured.value)
    assert any(isinstance(error, asyncio.CancelledError) for error in failures)
    assert any(
        isinstance(error, RuntimeError) and str(error) == "producer boom"
        for error in failures
    )
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert foreground.active_task_count == 0
    assert not any(
        task.get_name().startswith("foreground-speech-monitor:")
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


@pytest.mark.asyncio
async def test_oversized_inference_segment_fails_before_tts_or_playback() -> None:
    context = ConversationContextStore()
    inference = OversizedInference()
    synthesizer = SegmentSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
        max_segment_chars=4,
    )

    with pytest.raises(ValueError, match="segment.*capacity"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert synthesizer.texts == []
    assert playback.chunks == []
    assert inference.cancelled_turns == ["turn_001"]
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0


def test_segment_character_bound_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="max_segment_chars.*exact integer"):
        StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=RecordingPlayback(),
            ledger=DeliveredSpeechLedger(),
            max_segment_chars=True,
        )


def test_segment_character_bound_rejects_zero() -> None:
    with pytest.raises(ValueError, match="max_segment_chars.*positive"):
        StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=RecordingPlayback(),
            ledger=DeliveredSpeechLedger(),
            max_segment_chars=0,
        )


def test_segment_character_bound_rejects_unsupported_maximum() -> None:
    with pytest.raises(ValueError, match="max_segment_chars.*supported maximum"):
        StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=RecordingPlayback(),
            ledger=DeliveredSpeechLedger(),
            max_segment_chars=65_537,
        )


@pytest.mark.asyncio
async def test_segment_count_capacity_invalidates_queued_segments_before_tts() -> None:
    context = ConversationContextStore()
    inference = ThreeSegmentInference()
    synthesizer = SegmentSynthesizer()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        max_segments=2,
    )

    with pytest.raises(ValueError, match="segment count capacity"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert synthesizer.texts == []
    assert playback.chunks == []
    assert inference.cancelled_turns == ["turn_001"]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "One.",
        "Two.",
    ]


def test_segment_count_bound_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="max_segments.*exact integer"):
        StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=RecordingPlayback(),
            ledger=DeliveredSpeechLedger(),
            max_segments=True,
        )


def test_segment_count_bound_rejects_zero() -> None:
    with pytest.raises(ValueError, match="max_segments.*positive"):
        StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=RecordingPlayback(),
            ledger=DeliveredSpeechLedger(),
            max_segments=0,
        )


def test_segment_count_bound_rejects_unsupported_maximum() -> None:
    with pytest.raises(ValueError, match="max_segments.*supported maximum"):
        StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=PerTurnInference(),
            synthesizer=SegmentSynthesizer(),
            playback=RecordingPlayback(),
            ledger=DeliveredSpeechLedger(),
            max_segments=4097,
        )


@pytest.mark.asyncio
async def test_output_backpressure_cancels_blocked_playback_without_external_stop() -> None:
    context = ConversationContextStore()
    synthesizer = SegmentSynthesizer()
    playback = BlockingFirstPlayback()
    inference = BurstInference(playback.first_play_started)
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(output_capacity=1),
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    try:
        await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)
        with pytest.raises(ForegroundOutputBackpressure):
            await asyncio.wait_for(asyncio.shield(response), timeout=0.1)
    finally:
        if not response.done():
            await loop.cancel()

    assert inference.cancelled_turns == ["turn_001"]
    assert synthesizer.cancelled_turns == ["turn_001"]
    assert playback.cancelled_turns == ["turn_001"]
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0


@pytest.mark.asyncio
async def test_tts_chunk_for_wrong_turn_is_rejected_before_ledger_or_playback() -> None:
    context = ConversationContextStore()
    synthesizer = WrongTurnSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )

    with pytest.raises(ValueError, match="speech chunk turn_id"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert playback.chunks == []
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "Answer for turn_001.",
    ]
    assert synthesizer.cancelled_turns == ["turn_001"]


@pytest.mark.asyncio
async def test_duplicate_chunk_id_is_rejected_before_second_playback() -> None:
    context = ConversationContextStore()
    ledger = DeliveredSpeechLedger()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=DuplicateChunkSynthesizer(),
        playback=playback,
        ledger=ledger,
    )

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert [chunk.text for chunk in playback.chunks] == ["First subchunk."]
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "Answer for turn_001.",
    ]
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0


@pytest.mark.asyncio
async def test_synthesis_creation_failure_settles_without_playback() -> None:
    foreground = ForegroundTurnCoordinator()
    synthesizer = CreatingFailureSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=foreground,
        inference=PerTurnInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )

    with pytest.raises(RuntimeError, match="synthesis creation failed"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert synthesizer.cancelled_turns == ["turn_001"]
    assert playback.chunks == []
    assert ledger.pending() == ()
    assert foreground.active_task_count == 0


@pytest.mark.asyncio
async def test_synthesis_iteration_failure_settles_without_playback() -> None:
    foreground = ForegroundTurnCoordinator()
    synthesizer = IterationFailureSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=foreground,
        inference=PerTurnInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )

    with pytest.raises(RuntimeError, match="synthesis iteration failed"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert synthesizer.cancelled_turns == ["turn_001"]
    assert playback.chunks == []
    assert ledger.pending() == ()
    assert foreground.active_task_count == 0


@pytest.mark.asyncio
async def test_playback_failure_releases_started_but_unconfirmed_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import (
        AppendDisposition,
        CauseDisposition,
        EvidenceAdmissionControllerV1,
        EvidenceTurnLease,
        TerminalCauseCapabilityV1,
        TerminalReason,
    )
    from hermes_realtime.evidence.models import (
        SettledTerminalOutcomeV1,
        TerminalDisposition,
    )
    from hermes_realtime.production_observation import _new_observation_channel

    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    playback = FailingPlayback()
    ledger = DeliveredSpeechLedger()
    causes: list[TerminalReason] = []
    settlements: list[tuple[int, int, bool]] = []
    admission = object.__new__(EvidenceAdmissionControllerV1)
    lease = object.__new__(EvidenceTurnLease)
    object.__setattr__(lease, "evidence_turn_id", "00000000-0000-4000-8000-000000000191")
    object.__setattr__(lease, "binding_id", "00000000-0000-4000-8000-000000000192")
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", "00000000-0000-4000-8000-000000000193")
    object.__setattr__(lease, "terminal_cause", object.__new__(TerminalCauseCapabilityV1))
    observations, recorder = _new_observation_channel()

    def record_cause(self: object, capability: object, cause: TerminalReason) -> CauseDisposition:
        del self, capability
        causes.append(cause)
        return CauseDisposition.RECORDED

    def settle(self: object, lease: object, **kwargs: object) -> AppendDisposition:
        del self, lease
        settlements.append(
            (
                int(kwargs["queued_chunk_count"]),
                int(kwargs["started_chunk_count"]),
                bool(kwargs["assistant_delivery_context_recorded"]),
            )
        )
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda self, lease, text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "record_terminal_cause",
        record_cause,
    )
    monkeypatch.setattr(EvidenceAdmissionControllerV1, "settle_terminal", settle)
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settled_terminal_outcome",
        lambda _self, supplied_lease: (
            SettledTerminalOutcomeV1(
                terminal_disposition=TerminalDisposition.FAILED,
                terminal_reason=TerminalReason.TRANSPORT_FAILED,
                context_committed=False,
            )
            if supplied_lease is lease
            else None
        ),
    )
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=ledger,
        evidence_admission=admission,
        production_observation_recorder=recorder,
    )
    authority = loop._bind_update_executor(object())
    operation = loop._begin_update_operation(authority, "turn_001")

    with pytest.raises(RuntimeError, match="playback failed"):
        await loop._respond_with_updates(
            authority,
            operation,
            "turn_001",
            Transcript(text="Question?", final=True),
            (),
            lease,
        )

    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert causes == [TerminalReason.TRANSPORT_FAILED]
    assert settlements == [(1, 1, False)]
    assert [record.kind.value for record in observations.records()] == ["terminal_settled"]
    [settled] = observations.records()
    assert settled.context_committed is False
    assert foreground.active_task_count == 0
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "Answer for turn_001.",
    ]




@pytest.mark.asyncio
async def test_nonreturning_provider_cancel_is_bounded_and_releases_ledger() -> None:
    inference = NonReturningCancelInference()
    foreground = ForegroundTurnCoordinator()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=foreground,
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=FailingPlayback(),
        ledger=ledger,
        cleanup_timeout_seconds=0.02,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )

    await asyncio.wait_for(inference.cancel_started.wait(), timeout=1)
    await asyncio.sleep(0.1)
    assert response.done()
    with pytest.raises(BaseExceptionGroup, match="streaming response and cleanup"):
        await response

    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert foreground.active_task_count == 0
    inference.release_cancel.set()
    await asyncio.sleep(0)
    await loop.cancel()


@pytest.mark.asyncio
async def test_malformed_tts_value_is_rejected_before_field_access() -> None:
    context = ConversationContextStore()
    synthesizer = MalformedChunkSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )

    with pytest.raises(TypeError, match="exact SpeechChunk"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert playback.chunks == []
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "Answer for turn_001.",
    ]


@pytest.mark.asyncio
async def test_close_invalidates_active_speech_and_rejects_new_transcripts() -> None:
    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator()
    inference = PerTurnInference()
    synthesizer = SegmentSynthesizer()
    playback = BlockingFirstPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    await loop.close()
    with pytest.raises(asyncio.CancelledError):
        await response
    with pytest.raises(RuntimeError, match="streaming speech loop is closed"):
        await loop.respond("turn_002", Transcript(text="Too late?", final=True))

    async def rejected_owner(_: object) -> None:
        return None

    with pytest.raises(RuntimeError, match="foreground turn coordinator is closed"):
        await foreground.start("turn_003", rejected_owner)

    assert inference.cancelled_turns == ["turn_001"]
    assert synthesizer.cancelled_turns == ["turn_001"]
    assert playback.cancelled_turns == ["turn_001"]
    assert foreground.active_task_count == 0
    assert ledger.pending() == ()
    assert ledger.retained_chunk_count == 0
    assert [message.text for message in context.snapshot().messages] == [
        "Question?",
        "Answer for turn_001.",
    ]


@pytest.mark.asyncio
async def test_cancel_during_foreground_drain_still_settles_playback() -> None:
    foreground = ForegroundTurnCoordinator(drain_timeout=1)
    inference = CancellationResistantInference()
    playback = BlockingFirstPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=foreground,
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)
    cancellation = asyncio.create_task(loop.cancel())
    await asyncio.wait_for(inference.owner_cancelled.wait(), timeout=1)

    cancellation.cancel()
    inference.release_owner.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancellation, timeout=1)
        assert response.done() is True
        with pytest.raises(asyncio.CancelledError):
            await response
    finally:
        if not response.done():
            await loop.cancel()

    assert playback.cancelled_turns == ["turn_001"]
    assert foreground.active_task_count == 0


@pytest.mark.asyncio
async def test_mutated_chunk_is_rejected_before_hostile_comparison() -> None:
    synthesizer = MutatedChunkSynthesizer()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )

    with pytest.raises(TypeError, match="turn_id.*exact built-in string"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert synthesizer.trap_touched is False
    assert playback.chunks == []


@pytest.mark.asyncio
async def test_success_closes_provider_iterators_exactly_once() -> None:
    inference = CloseTrackingInference()
    synthesizer = CloseTrackingSynthesizer()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synthesizer,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )

    await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert inference.iterator.close_count == 1
    assert synthesizer.iterator is not None
    assert synthesizer.iterator.close_count == 1


@pytest.mark.asyncio
async def test_consumer_failure_cancels_blocked_inference_owner() -> None:
    foreground = ForegroundTurnCoordinator()
    inference = BlockingAfterFirstInference()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=foreground,
        inference=inference,
        synthesizer=MalformedChunkSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )

    with pytest.raises(TypeError, match="exact SpeechChunk"):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert inference.owner_cancelled.is_set()
    assert foreground.active_task_count == 0


@pytest.mark.asyncio
async def test_replacement_prevents_delayed_stale_transport_start() -> None:
    playback = DelayedTransportPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    first = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="First?", final=True))
    )
    await asyncio.wait_for(playback.first_play_entered.wait(), timeout=1)
    replacement = asyncio.create_task(
        loop.respond("turn_002", Transcript(text="Second?", final=True))
    )
    await asyncio.wait_for(playback.first_play_cancelled.wait(), timeout=1)

    playback.release_first_play.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await replacement

    assert playback.transport_writes == ["turn_002"]


@pytest.mark.asyncio
async def test_mutated_snapshot_is_revalidated_before_inference_side_effects() -> None:
    inference = InferenceCallProbe()
    synthesizer = SegmentSynthesizer()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=MutatedSnapshotContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )

    with pytest.raises(PrivateRunDisclosureError):
        await loop.respond("turn_001", Transcript(text="Question?", final=True))

    assert inference.stream_called is False
    assert synthesizer.texts == []
    assert playback.chunks == []


@pytest.mark.asyncio
async def test_cancel_fails_closed_within_bound_when_playback_ignores_cancellation() -> None:
    playback = NonReturningPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=SegmentSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        cleanup_timeout_seconds=0.02,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.play_started.wait(), timeout=1)

    cancellation = asyncio.create_task(loop.cancel())
    await asyncio.sleep(0.1)
    assert cancellation.done()
    with pytest.raises(TimeoutError, match="speech consumer cleanup"):
        await cancellation

    playback.release_play.set()
    with pytest.raises(asyncio.CancelledError):
        await response
    await loop.cancel()


@pytest.mark.asyncio
async def test_explicit_cancel_preserves_all_provider_cleanup_failures() -> None:
    class FailingCancelInference(PerTurnInference):
        async def cancel(self, turn_id: str) -> None:
            del turn_id
            raise RuntimeError("inference cancel failed")

    class FailingCancelSynthesizer(SegmentSynthesizer):
        async def cancel(self, turn_id: str) -> None:
            del turn_id
            raise RuntimeError("synthesis cancel failed")

    class FailingCancelPlayback(BlockingFirstPlayback):
        async def cancel(self, turn_id: str) -> None:
            del turn_id
            raise RuntimeError("playback cancel failed")

    playback = FailingCancelPlayback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=FailingCancelInference(),
        synthesizer=FailingCancelSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(playback.first_play_started.wait(), timeout=1)

    with pytest.raises(BaseExceptionGroup) as captured:
        await loop.cancel()
    rendered = repr(captured.value)
    assert "inference cancel failed" in rendered
    assert "playback cancel failed" in rendered
    assert "synthesis cancel failed" in rendered

    with pytest.raises(BaseExceptionGroup):
        await response


@pytest.mark.asyncio
async def test_replacement_during_blocked_synthesis_anext_fails_closed() -> None:
    synthesizer = BlockingSynthesisSynthesizer()
    playback = RecordingPlayback()
    ledger = DeliveredSpeechLedger()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=synthesizer,
        playback=playback,
        ledger=ledger,
        cleanup_timeout_seconds=0.02,
    )
    first = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )
    await asyncio.wait_for(synthesizer.iterator.started.wait(), timeout=1)

    replacement = asyncio.create_task(
        loop.respond("turn_002", Transcript(text="Replacement?", final=True))
    )
    await asyncio.sleep(0.1)
    assert replacement.done()
    with pytest.raises(TimeoutError, match="speech consumer cleanup"):
        await replacement

    assert playback.chunks == []
    assert ledger.pending() == ()
    synthesizer.iterator.release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await loop.cancel()
    assert synthesizer.iterator.close_count == 1


@pytest.mark.asyncio
async def test_nonreturning_iterator_close_fails_closed_within_bound() -> None:
    inference = NonClosingInference()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        cleanup_timeout_seconds=0.02,
    )
    response = asyncio.create_task(
        loop.respond("turn_001", Transcript(text="Question?", final=True))
    )

    await asyncio.wait_for(inference.iterator.close_started.wait(), timeout=1)
    await asyncio.sleep(0.1)
    assert response.done()
    with pytest.raises(TimeoutError, match="iterator cleanup"):
        await response

    orphaned_publication_waits = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and "ForegroundTurnCoordinator.next_publication" in repr(task.get_coro())
    ]
    assert orphaned_publication_waits == []

    with pytest.raises(RuntimeError, match="provider cleanup is still pending"):
        await loop.respond("turn_002", Transcript(text="Again?", final=True))
    with pytest.raises(RuntimeError, match="provider cleanup is still pending"):
        await loop.close()

    inference.iterator.release_close.set()
    await asyncio.sleep(0)
    await loop.close()
