"""Standing host authorization changes admission, never model or tool authority."""
import asyncio

import pytest

from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.work_tools import WorkCommandAdmission
from tests.conversation.test_task_proposals import Surface


def make(policy="automatic"):
    surface = Surface()
    events = []
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *event: events.append(event), start_policy=policy,
    )
    proposals.bind("participant", 1)
    return proposals, surface, events


@pytest.mark.asyncio
async def test_automatic_uses_exact_existing_admission_without_confirmation_projection():
    proposals, surface, events = make()
    objective = "Inspect **this exact** release and retain its constraints."
    assert await proposals.submit(HostTaskProposalRequest(objective)) is None
    assert len(surface.calls) == 1 and surface.calls[0][0] == objective
    assert surface.calls[0][1].startswith("automatic_proposal_")
    assert events == [] and not proposals.has_pending
    assert await proposals.decide(participant="participant", generation=1,
        proposal_id=surface.calls[0][1].removeprefix("automatic_"), objective=objective,
        decision="confirm") is WorkCommandAdmission.REFUSED
    assert len(surface.calls) == 1


@pytest.mark.asyncio
async def test_confirm_policy_still_waits_for_exact_decision():
    proposals, surface, events = make("confirm")
    proposal = await proposals.submit(HostTaskProposalRequest("Inspect release"))
    assert proposal is not None and proposals.has_pending
    assert surface.calls == [] and events[0][1]["status"] == "pending"


@pytest.mark.asyncio
async def test_automatic_unbound_request_never_admits():
    proposals, surface, events = make()
    proposals.close_binding()
    with pytest.raises(PermissionError):
        await proposals.submit(HostTaskProposalRequest("Inspect release"))
    assert surface.calls == [] and events == []


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "policy", [None, True, 1, "", "auto", StringSubclass("automatic"), object()],
)
def test_policy_rejects_unknown_or_non_string_values(policy):
    with pytest.raises((TypeError, ValueError)):
        make(policy)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_automatic_stream_reports_only_real_ack_after_one_silent_submission(accepted):
    from hermes_realtime.conversation import (
        ConversationContextStore,
        ForegroundTurnCoordinator,
        StreamingSpeechLoop,
    )
    from hermes_realtime.conversation.streaming import AcceptedStartInferenceRequest
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
    from tests.conversation.test_proposal_speech import ProposalInference
    from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
    from tests.conversation.test_work_tools import _surface

    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, events = _surface(context=context, dispatch_gate=gate, accepted=accepted)
    proposals = ConversationTaskProposals(surface=surface, observer=lambda *_: None,
        start_policy="automatic")
    proposals.bind("participant", 1)
    synth = SegmentSynthesizer()
    class Inference(ProposalInference):
        async def stream(self, snapshot, *, turn_id):
            if type(snapshot) is AcceptedStartInferenceRequest:
                yield "The work is underway."
            else:
                yield HostTaskProposalRequest("Inspect **the exact** release")
    loop = StreamingSpeechLoop(context=context, foreground=ForegroundTurnCoordinator(),
        inference=Inference(), synthesizer=synth, playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(), task_proposals=proposals)
    response = asyncio.create_task(
        loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
    )
    try:
        await controller.dispatch_started.wait()
        assert synth.texts == [] and events == [] and len(controller.dispatches) == 1
        assert context.snapshot().active_tasks == ()
    finally:
        gate.set()
        await response
        await surface.close()
    expected = {"status": "active" if accepted else "rejected", "taskId": "task_release_check"}
    if not accepted:
        expected["reason"] = "Hermes declined the request"
    assert events == [("task_state", expected)]
    assert controller.cancellations == [] and len(controller.dispatches) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("health", ["closed", "uncertain"])
async def test_automatic_unavailable_surface_projects_refusal_without_dispatch(health):
    from hermes_realtime.conversation.work_tools import WorkControlHealth
    from tests.conversation.test_work_tools import _surface

    surface, controller, events = _surface()
    if health == "closed":
        await surface.close()
    else:
        surface._health = WorkControlHealth.UNCERTAIN
    proposals = ConversationTaskProposals(surface=surface,
        observer=lambda kind, data: events.append((kind, data)), start_policy="automatic")
    proposals.bind("participant", 1)
    try:
        await proposals.submit(HostTaskProposalRequest("Inspect exact release"))
        assert controller.dispatches == [] and not proposals.has_pending
        assert events == [("task_state", {"status": "rejected", "taskId": None,
            "reason": "Task control is unavailable. Restart the session before trying again."})]
    finally:
        await surface.close()


@pytest.mark.asyncio
async def test_automatic_ack_wait_reports_unavailable_without_dispatch():
    from tests.conversation.test_work_tools import _surface

    surface, controller, events = _surface()
    await surface.close()
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda kind, data: events.append((kind, data)),
        start_policy="automatic",
    )
    proposals.bind("participant", 1)
    result = await proposals.submit_for_ack(HostTaskProposalRequest("Inspect release"))
    assert result is not None and result.accepted is False
    assert controller.dispatches == []
    assert events == [("task_state", {"status": "rejected", "taskId": None,
        "reason": "Task control is unavailable. Restart the session before trying again."})]
