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
    executor, _, _ = action_executor_probe()
    answered: list[str] = []

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
        if not answered:
            answered.append("refused:" + transcript.text)
            raise error
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
