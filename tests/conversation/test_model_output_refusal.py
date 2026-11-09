"""A refused model output fails its turn, never the binding (#470)."""

import asyncio
import json
from types import MethodType
from typing import Any

import pytest
from test_streaming import (
    ContextProbePlayback,
    PerTurnInference,
    SentenceSliceSynthesizer,
)
from test_worker import FinalOnPushTranscriber, SilenceVad, action_executor_probe

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.streaming import ModelOutputRefused
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.evidence import UserTurnAuthorityV1
from hermes_realtime.providers.ollama import OllamaDecisionRefused
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript


class ForeignModelOutputRefused(ModelOutputRefused):
    """A subclass cannot claim the exact non-terminal disposition."""


def test_refusal_requires_an_exact_allowlisted_category() -> None:
    refusal = ModelOutputRefused("call_imitation")
    assert refusal.category == "call_imitation"
    with pytest.raises(ValueError):
        ModelOutputRefused("private_authority")
    with pytest.raises(TypeError):
        ModelOutputRefused(type("PrivateStr", (str,), {})("call_imitation"))
    with pytest.raises(TypeError):
        ModelOutputRefused(None)  # type: ignore[arg-type]


def _worker_failing_first_response(
    error: BaseException,
) -> tuple[
    ConversationSessionWorker,
    list[str],
]:
    return _worker_with_outcomes(error)


def _worker_with_outcomes(
    *outcomes: BaseException | None,
) -> tuple[
    ConversationSessionWorker,
    list[str],
]:
    """Each response raises its scripted error, or succeeds for None or once exhausted."""
    executor, _, _ = action_executor_probe()
    answered: list[str] = []
    script = list(outcomes)

    async def respond(
        self: ConversationUpdateExecutor,
        turn_id: str,
        transcript: Transcript,
        authority: UserTurnAuthorityV1 | None = None,
        *,
        reservation: Any | None = None,
    ) -> None:
        del turn_id, authority
        if reservation is not None:
            self._operation_scheduler.release(reservation)
        outcome = script.pop(0) if script else None
        if outcome is not None:
            answered.append("refused:" + transcript.text)
            raise outcome
        answered.append(transcript.text)

    executor.respond = MethodType(respond, executor)  # type: ignore[method-assign]
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
    )
    worker.start()
    return worker, answered


async def _submit(worker: ConversationSessionWorker, sequence: int, text: str) -> None:
    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=sequence,
        text=text,
    )


@pytest.mark.asyncio
async def test_refused_turn_leaves_the_next_utterance_answered_on_the_same_binding() -> None:
    worker, answered = _worker_failing_first_response(ModelOutputRefused("call_imitation"))

    await _submit(worker, 1, "first utterance")
    await worker.wait_for_responses()
    await _submit(worker, 2, "next utterance")
    await worker.wait_for_responses()

    assert answered == ["refused:first utterance", "next utterance"]
    await worker.close()


def _refused() -> ModelOutputRefused:
    return ModelOutputRefused("call_imitation")


async def _turns(worker: ConversationSessionWorker, count: int) -> None:
    for sequence in range(1, count + 1):
        await _submit(worker, sequence, f"utterance {sequence}")
        await worker.wait_for_responses()


@pytest.mark.asyncio
async def test_two_consecutive_refusals_keep_the_binding() -> None:
    worker, answered = _worker_with_outcomes(_refused(), _refused())

    await _turns(worker, 3)

    assert answered == ["refused:utterance 1", "refused:utterance 2", "utterance 3"]
    await worker.close()


@pytest.mark.asyncio
async def test_a_third_consecutive_refusal_ends_the_binding_loudly() -> None:
    third = _refused()
    worker, answered = _worker_with_outcomes(_refused(), _refused(), third)

    await _turns(worker, 2)
    await _submit(worker, 3, "utterance 3")
    with pytest.raises(ModelOutputRefused) as settled:
        await worker.wait_for_responses()
    assert settled.value is third
    with pytest.raises(ModelOutputRefused) as rejected:
        await _submit(worker, 4, "utterance 4")
    assert rejected.value is third

    assert answered == [f"refused:utterance {n}" for n in (1, 2, 3)]
    object.__setattr__(worker, "_terminal_error", None)
    await worker.close()


@pytest.mark.asyncio
async def test_a_completed_response_resets_the_refusal_count() -> None:
    worker, answered = _worker_with_outcomes(
        _refused(),
        _refused(),
        None,
        _refused(),
        _refused(),
    )

    await _turns(worker, 6)

    assert answered == [
        "refused:utterance 1",
        "refused:utterance 2",
        "utterance 3",
        "refused:utterance 4",
        "refused:utterance 5",
        "utterance 6",
    ]
    await worker.close()


@pytest.mark.parametrize(
    "error",
    [
        OllamaDecisionRefused("private_authority"),
        RuntimeError("synthetic provider failure"),
        ForeignModelOutputRefused("call_imitation"),
        ExceptionGroup(
            "refusal and close failed",
            [ModelOutputRefused("call_imitation"), RuntimeError("synthetic close failure")],
        ),
    ],
    ids=["private_authority", "runtime", "subclass", "grouped_with_close"],
)
@pytest.mark.asyncio
async def test_other_response_errors_still_end_the_binding(error: BaseException) -> None:
    worker, answered = _worker_failing_first_response(error)

    await _submit(worker, 1, "first utterance")
    with pytest.raises(type(error)) as settled:
        await worker.wait_for_responses()
    assert settled.value is error
    with pytest.raises(type(error)) as rejected:
        await _submit(worker, 2, "next utterance")
    assert rejected.value is error

    assert answered == ["refused:first utterance"]
    object.__setattr__(worker, "_terminal_error", None)
    await worker.close()


class _SlowCloseRefusalStream:
    """Refuse after a heard segment and a second one, then hold aclose open.

    While aclose is held the producer has failed but its result is still pending,
    so the consumer is revoked before the producer finishes.
    """

    def __init__(self, loop_ref: list[StreamingSpeechLoop], playback: ContextProbePlayback):
        self._loop_ref = loop_ref
        self._playback = playback
        self._step = 0
        self.closing = asyncio.Event()

    def __aiter__(self) -> "_SlowCloseRefusalStream":
        return self

    async def __anext__(self) -> str:
        self._step += 1
        if self._step == 1:
            return _FIRST
        if self._step == 2:
            while not self._playback.chunks:
                await asyncio.sleep(0)
            return "Second part."
        raise ModelOutputRefused("call_imitation")

    async def aclose(self) -> None:
        self.closing.set()
        loop = self._loop_ref[0]
        for _ in range(1000):
            if loop._active_consumer is None:
                break
            await asyncio.sleep(0)
        assert loop._active_consumer is None, "consumer was not revoked during aclose"


class _SlowCloseInference(PerTurnInference):
    def __init__(self, stream: _SlowCloseRefusalStream) -> None:
        super().__init__()
        self.stream_object = stream

    def stream(self, snapshot, *, turn_id):  # type: ignore[no-untyped-def, override]
        del snapshot, turn_id
        return self.stream_object


@pytest.mark.asyncio
async def test_refusal_revoked_before_the_producer_finishes_leaves_no_resumable_speech() -> None:
    context = ConversationContextStore()
    playback = ContextProbePlayback(context)
    loop_ref: list[StreamingSpeechLoop] = []
    stream = _SlowCloseRefusalStream(loop_ref, playback)
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=_SlowCloseInference(stream),
        synthesizer=SentenceSliceSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    loop_ref.append(loop)
    try:
        # Revocation cancels the producer inside its held aclose, so the refusal
        # surfaces grouped with that cancellation; only resumability is asserted here.
        with pytest.raises(BaseExceptionGroup) as failed:
            await asyncio.wait_for(
                loop.respond("turn_001", Transcript(text="Question?", final=True)),
                timeout=2,
            )
        assert any(type(error) is ModelOutputRefused for error in failed.value.exceptions)
        assert stream.closing.is_set()
        assert loop._resumable_speech is None
        assert await loop.announce("after_refusal", "I am still here.") is True
    finally:
        await loop.close()


class _FailOnCuePlayback(ContextProbePlayback):
    """Hold the first chunk, then fail playback itself when cued."""

    def __init__(self, context: ConversationContextStore) -> None:
        super().__init__(context)
        self.playing = asyncio.Event()
        self.fail = asyncio.Event()

    async def play(self, chunk, *, is_valid):  # type: ignore[no-untyped-def, override]
        self.chunks.append(chunk)
        self.playing.set()
        await self.fail.wait()
        raise RuntimeError("synthetic playback failure")


@pytest.mark.asyncio
async def test_consumer_failure_after_a_refusal_leaves_no_resumable_speech() -> None:
    """The consumer fails on its own while the refused producer is still cleaning up."""
    context = ConversationContextStore()
    playback = _FailOnCuePlayback(context)

    class Inference(PerTurnInference):
        async def _stream(self, snapshot, *, turn_id):  # type: ignore[no-untyped-def, override]
            del snapshot, turn_id
            yield _FIRST
            await asyncio.wait_for(playback.playing.wait(), timeout=1)
            playback.fail.set()
            raise ModelOutputRefused("call_imitation")

    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SentenceSliceSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    try:
        with pytest.raises(BaseException) as failed:
            await asyncio.wait_for(
                loop.respond("turn_001", Transcript(text="Question?", final=True)),
                timeout=2,
            )
        assert not isinstance(failed.value, asyncio.CancelledError)
        assert loop._resumable_speech is None
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_refused_acknowledgment_continuation_leaves_no_resumable_speech() -> None:
    from test_work_tools import _surface

    from hermes_realtime.conversation.streaming import AcceptedStartInferenceRequest
    from hermes_realtime.conversation.task_proposals import (
        ConversationTaskProposals,
        HostTaskProposalRequest,
    )

    context = ConversationContextStore()
    surface, _, _ = _surface(context=context)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    playback = ContextProbePlayback(context, block=(("turn_auto", "slice_1_1"),))

    class Inference:
        async def stream(self, snapshot, *, turn_id):  # type: ignore[no-untyped-def]
            del turn_id
            if type(snapshot) is not AcceptedStartInferenceRequest:
                yield HostTaskProposalRequest("Inspect the release")
                return
            yield _FIRST
            await asyncio.wait_for(playback.blocked.wait(), timeout=1)
            yield "Second part."
            raise ModelOutputRefused("unverified_claim")

        async def cancel(self, turn_id):  # type: ignore[no-untyped-def]
            pass

    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SentenceSliceSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    try:
        with pytest.raises(ModelOutputRefused):
            await asyncio.wait_for(
                loop.respond("turn_auto", Transcript(text="Inspect release", final=True)),
                timeout=2,
            )
        assert playback.blocked.is_set()
        assert loop._resumable_speech is None
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.parametrize(
    "error,kind",
    [
        (ModelOutputRefused("call_imitation"), "refused"),
        (ForeignModelOutputRefused("call_imitation"), "other"),
    ],
    ids=["exact", "subclass"],
)
def test_producer_failure_marker_names_only_an_exact_refusal(capsys, error, kind) -> None:
    with pytest.raises(type(error)):
        StreamingSpeechLoop._raise_iteration_or_close_error(
            "unused",
            error,
            None,
            producer_after_text=True,
        )
    assert json.loads(capsys.readouterr().out.removeprefix("[foreground-failure] ")) == {
        "version": 1,
        "stage": "producer",
        "failure_kind": kind,
        "text_phase": "after_text",
    }


_FIRST = "This first sentence is comfortably long enough."


class _RefusedAfterTextInference(PerTurnInference):
    """Publish one segment, then a second, then refuse the rest of the turn."""

    def __init__(self, playback: ContextProbePlayback) -> None:
        super().__init__()
        self._playback = playback

    async def _stream(self, snapshot, *, turn_id):  # type: ignore[no-untyped-def, override]
        del snapshot, turn_id
        # Long enough to publish at once instead of coalescing with the next segment.
        yield _FIRST
        await asyncio.wait_for(self._playback.blocked.wait(), timeout=1)
        yield "Second part."
        raise ModelOutputRefused("call_imitation")


@pytest.mark.asyncio
async def test_half_published_refusal_leaves_no_resumable_speech_and_announce_works() -> None:
    context = ConversationContextStore()
    playback = ContextProbePlayback(context, block=(("turn_001", "slice_1_1"),))
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=_RefusedAfterTextInference(playback),
        synthesizer=SentenceSliceSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    try:
        with pytest.raises(ModelOutputRefused):
            await asyncio.wait_for(
                loop.respond("turn_001", Transcript(text="Question?", final=True)),
                timeout=2,
            )
        assert loop.foreground_active is False
        assert loop._resumable_speech is None
        assert await loop.resume_interrupted() is False

        assert await loop.announce("after_refusal", "I am still here.") is True
        assert [chunk.text for chunk in playback.chunks] == [_FIRST, "I am still here."]
    finally:
        await loop.close()
