"""Host confirmation is the only route from a model proposal to work admission."""

from __future__ import annotations

import pytest

from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.work_tools import WorkCommandAdmission


class Surface:
    max_objective_chars = 1024

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def submit_start_command(self, *, objective: str, invocation_id: str):
        self.calls.append((objective, invocation_id))
        return WorkCommandAdmission.ADMITTED


def owner():
    surface = Surface()
    events = []
    now = [1.0]
    proposals = ConversationTaskProposals(
        surface=surface,
        observer=lambda kind, data: events.append((kind, data)),
        clock=lambda: now[0],
        id_factory=lambda: "proposal_12345678",
    )
    proposals.bind("participant", 1)
    return proposals, surface, events, now


@pytest.mark.asyncio
async def test_proposal_and_readback_never_dispatch_until_exact_confirmation():
    proposals, surface, events, _now = owner()
    objective = "Inspect **the exact** release; never normalize it."
    request = HostTaskProposalRequest(objective, "model_call")
    proposal = proposals.offer(request)
    assert surface.calls == []
    assert events[-1][1]["objective"] == objective
    assert proposal.objective == objective
    assert objective in proposal.readback
    admission = await proposals.decide(
        participant="participant", generation=1, proposal_id=proposal.proposal_id,
        objective=objective, decision="confirm",
    )
    assert admission is WorkCommandAdmission.ADMITTED
    assert surface.calls == [(objective, "confirmed_proposal_12345678")]
    assert all(event[1].get("status") != "active" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"participant": "other"}, {"generation": 2}, {"proposal_id": "proposal_87654321"},
    {"objective": "Inspect release"}, {"decision": "approve"},
])
async def test_confirmation_binding_rejects_each_mismatch(change):
    proposals, surface, _events, _now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect exact release", "call"))
    values = dict(participant="participant", generation=1, proposal_id=proposal.proposal_id,
                  objective=proposal.objective, decision="confirm")
    values.update(change)
    with pytest.raises((PermissionError, ValueError)):
        await proposals.decide(**values)
    assert surface.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["declined", "expired", "user_turn", "rebind", "speech"])
async def test_retired_proposal_cannot_dispatch(retirement):
    proposals, surface, _events, now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release", "call"))
    if retirement == "declined":
        await proposals.decide(participant="participant", generation=1,
            proposal_id=proposal.proposal_id, objective=proposal.objective, decision="decline")
    elif retirement == "expired":
        now[0] += 61
    elif retirement == "rebind":
        proposals.bind("participant", 2)
    else:
        proposals.retire(retirement)
    with pytest.raises(PermissionError):
        await proposals.decide(participant="participant", generation=1,
            proposal_id=proposal.proposal_id, objective=proposal.objective, decision="confirm")
    assert surface.calls == []


@pytest.mark.asyncio
async def test_confirmation_is_consumed_before_dispatch_await_and_replay_refused():
    import asyncio

    proposals, surface, _events, _now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release", "call"))
    entered = asyncio.Event()
    release = asyncio.Event()
    original = surface.submit_start_command

    async def delayed(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)

    surface.submit_start_command = delayed
    values = dict(participant="participant", generation=1, proposal_id=proposal.proposal_id,
                  objective=proposal.objective, decision="confirm")
    first = asyncio.create_task(proposals.decide(**values))
    await entered.wait()
    with pytest.raises(PermissionError):
        await proposals.decide(**values)
    release.set()
    await first
    assert len(surface.calls) == 1


def test_model_invocation_replay_and_replacement_are_refused():
    proposals, surface, _events, _now = owner()
    request = HostTaskProposalRequest("Inspect release", "call")
    proposals.offer(request)
    with pytest.raises(PermissionError):
        proposals.offer(request)
    with pytest.raises(PermissionError):
        proposals.offer(HostTaskProposalRequest("Another proposal", "other_call"))
    assert surface.calls == []
