"""A selected automatic start that is not accepted gets one fixed host sentence (#353)."""

import asyncio
from dataclasses import dataclass

import pytest

from hermes_realtime.conversation import (
    ActiveTaskIdentityError,
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
    TaskDispatchOutcome,
    WorkStartResult,
)
from hermes_realtime.conversation.streaming import AcceptedStartInferenceRequest
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
    unaccepted_start_speech,
)
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import (
    PresentationBlockingPlayback,
    RecordingPlayback,
    SegmentSynthesizer,
)
from tests.conversation.test_work_tools import _surface

NOT_RUNNING = "No new background task is running for that request."
UNAVAILABLE = "Background task control is unavailable, so that task was not started."
SENTINEL = "SENTINEL reason from Hermes 7f3a"
OBJECTIVE = "Inspect the release OBJECTIVE-9c1d"


class Inference:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def stream(self, snapshot, *, turn_id):
        self.calls.append(snapshot)
        if type(snapshot) is AcceptedStartInferenceRequest:
            yield "I have started the work."
        else:
            yield HostTaskProposalRequest(OBJECTIVE)

    async def cancel(self, turn_id):
        pass


def _loop(context, proposals, inference, *, playback=None, foreground=None):
    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground or ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synth,
        playback=playback or RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    return loop, synth


def _proposals(surface, **options):
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic", **options,
    )
    proposals.bind("participant", 1)
    return proposals


def _host_messages(context):
    return [m for m in context.snapshot().messages if m.source == "host_task_status"]


def _assert_one_host_sentence(context, synth, expected):
    assert synth.texts == [expected]
    heard = _host_messages(context)
    assert [(m.role, m.text) for m in heard] == [("assistant", expected)]
    assert context.snapshot().messages[-1].source == "host_task_status"
    texts = synth.texts + [m.text for m in context.snapshot().messages]
    assert not any(SENTINEL in text for text in texts)
    assert not any("OBJECTIVE-9c1d" in text for text in synth.texts)


async def _respond(loop):
    await loop.respond("turn_auto", Transcript(text="Inspect release", final=True))


@pytest.mark.asyncio
async def test_hermes_rejection_speaks_one_host_sentence_without_reason_text():
    # Mutation: drop the unaccepted branch (silent) or speak result.reason.
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, accepted=False)
    controller.dispatch_outcome = TaskDispatchOutcome(
        task_id="task_release_check", accepted=False, reason=SENTINEL,
    )
    inference = Inference()
    loop, synth = _loop(context, _proposals(surface), inference)
    try:
        await _respond(loop)
        assert len(controller.dispatches) == 1 and len(inference.calls) == 1
        _assert_one_host_sentence(context, synth, NOT_RUNNING)
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_unavailable_control_speaks_the_unavailable_sentence_without_dispatch():
    # Mutation: map unavailable to the general sentence, or drop the task_id guard.
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context)
    await surface.close()
    loop, synth = _loop(context, _proposals(surface), Inference())
    try:
        await _respond(loop)
        assert controller.dispatches == []
        _assert_one_host_sentence(context, synth, UNAVAILABLE)
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_duplicate_objective_says_no_new_task_without_denying_running_work():
    context = ConversationContextStore()
    context.record_task_accepted(task_id="task_existing", run_id="deleg_1", objective=OBJECTIVE)
    surface, controller, _ = _surface(context=context)
    loop, synth = _loop(context, _proposals(surface), Inference())
    try:
        await _respond(loop)
        assert controller.dispatches == []
        _assert_one_host_sentence(context, synth, NOT_RUNNING)
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_capacity_refusal_speaks_the_general_sentence():
    context = ConversationContextStore(max_active_tasks=1)
    context.record_task_accepted(task_id="task_other", run_id="deleg_1", objective="Other work")
    surface, controller, _ = _surface(context=context)
    loop, synth = _loop(context, _proposals(surface), Inference())
    try:
        await _respond(loop)
        assert controller.dispatches == []
        _assert_one_host_sentence(context, synth, NOT_RUNNING)
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_expired_selection_never_dispatched_speaks_the_general_sentence():
    # submit_for_ack returns None only when the one-use consume refused before dispatch.
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context)
    ticks = iter((0.0, 1000.0, 1000.0, 1000.0))
    loop, synth = _loop(context, _proposals(surface, clock=lambda: next(ticks)), Inference())
    try:
        await _respond(loop)
        assert controller.dispatches == []
        _assert_one_host_sentence(context, synth, NOT_RUNNING)
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_accepted_then_inactive_never_claims_it_did_not_start():
    # Hermes accepted the run; a pending cancel then found it no longer active. The
    # result is rejected with a task identity, so "not started" would be false here.
    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, dispatch_gate=gate)

    async def inactive(task_id, *, reason=None):
        controller.cancellations.append((task_id, reason))
        raise ActiveTaskIdentityError("task is not active")

    controller.request_cancel = inactive
    loop, synth = _loop(context, _proposals(surface), Inference())
    operation = asyncio.create_task(_respond(loop))
    try:
        await controller.dispatch_started.wait()
        cancel = asyncio.create_task(surface.cancel_active_work(invocation_id="cancel_1"))
        await asyncio.sleep(0)
        gate.set()
        await operation
        await cancel
        assert len(controller.cancellations) == 1
        _assert_one_host_sentence(context, synth, NOT_RUNNING)
    finally:
        gate.set()
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True], ids=["active", "cancelling"])
async def test_accepted_start_has_no_host_refusal_sentence(cancelled):
    # Mutation: speak the sentence for every result, or for accepted cancelling.
    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, dispatch_gate=gate)
    inference = Inference()
    loop, synth = _loop(context, _proposals(surface), inference)
    operation = asyncio.create_task(_respond(loop))
    try:
        await controller.dispatch_started.wait()
        cancel = None
        if cancelled:
            cancel = asyncio.create_task(surface.cancel_active_work(invocation_id="cancel_1"))
            await asyncio.sleep(0)
        gate.set()
        await operation
        if cancel is not None:
            await cancel
        assert _host_messages(context) == []
        assert NOT_RUNNING not in synth.texts and UNAVAILABLE not in synth.texts
        assert synth.texts == ([] if cancelled else ["I have started the work."])
    finally:
        gate.set()
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_superseded_lease_stays_silent_after_rejection():
    # Mutation: drop the lease.is_current guard before the host sentence.
    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, dispatch_gate=gate, accepted=False)
    foreground = ForegroundTurnCoordinator()
    loop, synth = _loop(context, _proposals(surface), Inference(), foreground=foreground)
    operation = asyncio.create_task(_respond(loop))
    try:
        await controller.dispatch_started.wait()
        foreground._is_current = lambda _lease: False
        gate.set()
        await operation
        assert synth.texts == [] and _host_messages(context) == []
    finally:
        gate.set()
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_undelivered_host_sentence_stays_out_of_history():
    context = ConversationContextStore()
    surface, _controller, _ = _surface(context=context, accepted=False)
    playback = PresentationBlockingPlayback()
    loop, synth = _loop(context, _proposals(surface), Inference(), playback=playback)
    operation = asyncio.create_task(_respond(loop))
    try:
        await asyncio.wait_for(playback.play_started.wait(), 1)
        assert synth.texts == [NOT_RUNNING]
        await loop.cancel()
        playback.release_play.set()
        await asyncio.gather(operation, return_exceptions=True)
        assert _host_messages(context) == []
    finally:
        playback.release_play.set()
        await loop.close()
        await surface.close()


def test_sentence_is_selected_only_by_typed_category():
    unavailable = "work control unavailable; restart required"
    assert unaccepted_start_speech(
        WorkStartResult(accepted=True, state="active", task_id="task_a")) is None
    assert unaccepted_start_speech(
        WorkStartResult(accepted=True, state="cancelling", task_id="task_a")) is None
    assert unaccepted_start_speech(None) == NOT_RUNNING
    assert unaccepted_start_speech(
        WorkStartResult(accepted=False, state="rejected", reason=unavailable)) == UNAVAILABLE
    # A Hermes rejection carrying the host's unavailable words is still a Hermes rejection.
    assert unaccepted_start_speech(WorkStartResult(
        accepted=False, state="rejected", task_id="task_a", reason=unavailable,
    )) == NOT_RUNNING
    assert unaccepted_start_speech(WorkStartResult(
        accepted=False, state="rejected", task_id="task_a", reason=SENTINEL,
    )) == NOT_RUNNING


@dataclass(frozen=True, slots=True)
class _Subclass(WorkStartResult):
    pass


@pytest.mark.parametrize("value", [
    _Subclass(accepted=False, state="rejected", reason=SENTINEL), "rejected", False, object(),
])
def test_sentence_selection_refuses_inexact_results(value):
    # Mutation: replace the exact-type check with isinstance.
    with pytest.raises(TypeError):
        unaccepted_start_speech(value)
