"""An assent admitted before an await cannot confirm a later proposal or binding."""

import asyncio

import pytest

from hermes_realtime.conversation.commands import ConversationTaskCommandRouter
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.work_tools import WorkCommandAdmission
from tests.conversation.test_proposal_capture import CommandSurface


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["retire", "rebind", "card_confirm", "card_decline", "replace"])
async def test_recording_await_cannot_upgrade_stale_assent_or_confirm_replacement(change):
    surface, entered, release = CommandSurface(), asyncio.Event(), asyncio.Event()
    proposals = ConversationTaskProposals(surface=surface, observer=lambda *_: None)
    proposals.bind("participant", 1)
    first = proposals.offer(HostTaskProposalRequest("First exact objective"))
    proposals.typed_input_started(participant="participant", generation=1)

    async def record(text):
        entered.set()
        await release.wait()

    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda *_: None,
        reserve_observer_capacity=lambda: None,
        task_proposals=proposals,
        validate_user_input=lambda _: None,
        record_user_input=record,
    )
    operation = asyncio.create_task(router.route("yes"))
    await entered.wait()
    replacement = None
    if change == "rebind":
        proposals.bind("participant", 2)
    elif change in {"card_confirm", "card_decline"}:
        await proposals.decide(
            participant="participant",
            generation=1,
            proposal_id=first.proposal_id,
            objective=first.objective,
            decision="confirm" if change == "card_confirm" else "decline",
        )
    else:
        proposals.retire("intervening_input")
        if change == "replace":
            replacement = proposals.offer(HostTaskProposalRequest("New exact objective"))
            proposals.typed_input_started(participant="participant", generation=1)
    release.set()
    assert await operation is True
    assert len(surface.calls) == (1 if change == "card_confirm" else 0)
    if replacement is not None:
        assert proposals.confirmation_target("yes") is replacement
        assert (
            await proposals.consume_input(
                "yes",
                expected_proposal=replacement,
            )
            is WorkCommandAdmission.ADMITTED
        )
        assert surface.calls == [(replacement.objective, "confirmed_" + replacement.proposal_id)]
