"""Host confirmation is the only route from a model proposal to work admission."""

from __future__ import annotations

import pytest

from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.work_tools import WorkCommandAdmission, WorkStartResult


class TextSubclass(str):
    pass


class Surface:
    max_objective_chars = 1024

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def submit_start_command(self, *, objective: str, invocation_id: str):
        self.calls.append((objective, invocation_id))
        return WorkCommandAdmission.ADMITTED

    async def start_work(self, *, objective: str, invocation_id: str):
        self.calls.append((objective, invocation_id))
        return WorkStartResult(accepted=False, state="rejected", reason="synthetic refusal")


def owner():
    surface = Surface()
    events = []
    now = [1.0]
    proposals = ConversationTaskProposals(
        surface=surface,
        observer=lambda kind, data: events.append((kind, data)),
        clock=lambda: now[0],
    )
    proposals.bind("participant", 1)
    return proposals, surface, events, now


@pytest.mark.asyncio
async def test_proposal_and_readback_never_dispatch_until_exact_confirmation():
    proposals, surface, events, _now = owner()
    objective = "Inspect **the exact** release; never normalize it."
    request = HostTaskProposalRequest(objective)
    proposal = proposals.offer(request)
    assert surface.calls == []
    assert events[-1][1]["objective"] == objective
    assert proposal.objective == objective
    assert objective in proposal.readback
    admission = await proposals.decide(
        participant="participant",
        generation=1,
        proposal_id=proposal.proposal_id,
        objective=objective,
        decision="confirm",
    )
    assert admission is WorkCommandAdmission.ADMITTED
    assert surface.calls == [(objective, "confirmed_" + proposal.proposal_id)]
    assert all(event[1].get("status") != "active" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"participant": "other"},
        {"generation": 2},
        {"proposal_id": "proposal_87654321"},
        {"objective": "Inspect release"},
        {"decision": "approve"},
    ],
)
async def test_confirmation_binding_rejects_each_mismatch(change):
    proposals, surface, _events, _now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect exact release"))
    values = dict(
        participant="participant",
        generation=1,
        proposal_id=proposal.proposal_id,
        objective=proposal.objective,
        decision="confirm",
    )
    values.update(change)
    if change.get("decision") == "approve":
        with pytest.raises(ValueError):
            await proposals.decide(**values)
    else:
        assert await proposals.decide(**values) is WorkCommandAdmission.REFUSED
    assert surface.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["declined", "expired", "user_turn", "rebind", "speech"])
async def test_retired_proposal_cannot_dispatch(retirement):
    proposals, surface, _events, now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    if retirement == "declined":
        await proposals.decide(
            participant="participant",
            generation=1,
            proposal_id=proposal.proposal_id,
            objective=proposal.objective,
            decision="decline",
        )
    elif retirement == "expired":
        now[0] += 61
    elif retirement == "rebind":
        proposals.bind("participant", 2)
    else:
        proposals.retire(retirement)
    assert (
        await proposals.decide(
            participant="participant",
            generation=1,
            proposal_id=proposal.proposal_id,
            objective=proposal.objective,
            decision="confirm",
        )
        is WorkCommandAdmission.REFUSED
    )
    assert surface.calls == []


@pytest.mark.asyncio
async def test_confirmation_is_consumed_before_dispatch_await_and_replay_refused():
    import asyncio

    proposals, surface, _events, _now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    entered = asyncio.Event()
    release = asyncio.Event()
    original = surface.submit_start_command

    async def delayed(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)

    surface.submit_start_command = delayed
    values = dict(
        participant="participant",
        generation=1,
        proposal_id=proposal.proposal_id,
        objective=proposal.objective,
        decision="confirm",
    )
    first = asyncio.create_task(proposals.decide(**values))
    await entered.wait()
    try:
        assert not proposals.has_pending
        assert await proposals.decide(**values) is WorkCommandAdmission.REFUSED
    finally:
        release.set()
        await first
    assert len(surface.calls) == 1


def test_model_invocation_replay_and_replacement_are_refused():
    proposals, surface, _events, _now = owner()
    request = HostTaskProposalRequest("Inspect release")
    proposals.offer(request)
    with pytest.raises(PermissionError):
        proposals.offer(request)
    with pytest.raises(PermissionError):
        proposals.offer(HostTaskProposalRequest("Another proposal"))
    assert surface.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "affirmative", ["yes", "Yeah!", "yep.", "sure", "okay", "go ahead", "do it", "please do"]
)
async def test_spoken_confirmation_requires_complete_readback_and_clean_capture(affirmative):
    proposals, surface, _events, _now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.capture_started(participant="participant", generation=1, clean=True)
    assert proposals.is_confirmation(affirmative)
    assert (
        await proposals.consume_input(affirmative, expected_proposal=proposal)
        is WorkCommandAdmission.ADMITTED
    )
    assert surface.calls == [(proposal.objective, "confirmed_" + proposal.proposal_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocker",
    [
        "partial",
        "normalization",
        "unknown_projector",
        "early",
        "noisy",
        "unsolicited_speech",
        "delayed_stt",
    ],
)
async def test_ineligible_assent_never_upgrades_and_retains_retryable_proposal(blocker):
    proposals, surface, events, _now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect **release**"))
    if blocker == "normalization":
        proposals._speech_projection = lambda text: text.replace("*", "")
    elif blocker == "unknown_projector":
        proposals._speech_projection = None
    if blocker in {"early", "delayed_stt"}:
        proposals.capture_started(participant="participant", generation=1, clean=True)
    proposals.complete_readback(
        proposal,
        proposal.readback[:-1] if blocker == "partial" else proposal.readback,
        queued_chunks=1,
        confirmed_chunks=1,
    )
    if blocker not in {"early", "delayed_stt"}:
        proposals.capture_started(participant="participant", generation=1, clean=blocker != "noisy")
    if blocker == "unsolicited_speech":
        proposals.speech_started()
    assert (
        await proposals.consume_input("yes", expected_proposal=proposal)
        is WorkCommandAdmission.REFUSED
    )
    assert surface.calls == []
    assert events[-1][1]["status"] == "pending"
    assert events[-1][1]["reason"] == (
        "use_confirmation_card"
        if blocker in {"partial", "normalization", "unknown_projector", "unsolicited_speech"}
        else "spoken_confirmation_not_ready"
    )
    # Explicit review of the unchanged full-text card remains available.
    await proposals.decide(
        participant="participant",
        generation=1,
        proposal_id=proposal.proposal_id,
        objective=proposal.objective,
        decision="confirm",
    )
    assert len(surface.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        "not yes",
        "yes but change the release",
        "I said yes yesterday",
        "yes yes",
        "please do other work",
        "sure thing",
        "yes\nyes",
    ],
)
def test_confirmation_grammar_matches_only_the_whole_utterance(text):
    proposals, _surface, _events, _now = owner()
    assert not proposals.is_confirmation(text)


@pytest.mark.parametrize("text", ["'yes'", '"yes"', "(yes)", "[yes]", "yes:", "-yes-"])
def test_confirmation_grammar_refuses_quoted_or_wrapped_assent(text):
    proposals, _surface, _events, _now = owner()
    assert not proposals.is_confirmation(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["microphone", "typed"])
@pytest.mark.parametrize(
    "participant,generation",
    [
        ("participant", True),
        (b"participant", 1),
        (TextSubclass("participant"), 1),
    ],
    ids=["bool_generation", "bytes_participant", "str_subclass"],
)
async def test_input_capture_binding_requires_exact_primitive_types(
    source, participant, generation
):
    proposals, surface, _events, _now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    if source == "microphone":
        proposals.capture_started(participant=participant, generation=generation, clean=True)
    else:
        proposals.typed_input_started(participant=participant, generation=generation)
    await proposals.consume_input("yes", expected_proposal=proposal)
    assert surface.calls == []


def test_unrelated_input_retires_proposal_without_cancelling_work():
    proposals, surface, events, _now = owner()
    proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.intervening_input()
    assert surface.calls == []
    assert events[-1][1]["status"] == "retired"


@pytest.mark.asyncio
@pytest.mark.parametrize("queued,confirmed", [(0, 0), (2, 1), (1, 2), (True, 1), (1, True)])
async def test_readback_requires_nonzero_exact_transport_chunk_counts(queued, confirmed):
    proposals, surface, _events, _now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.complete_readback(
        proposal, proposal.readback, queued_chunks=queued, confirmed_chunks=confirmed
    )
    proposals.capture_started(participant="participant", generation=1, clean=True)
    await proposals.consume_input("yes", expected_proposal=proposal)
    assert surface.calls == []


@pytest.mark.asyncio
async def test_readback_completion_cannot_rearm_after_foreign_speech_or_extend_twice():
    proposals, surface, _events, now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    now[0] += 30
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.speech_started()
    now[0] += 20
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.capture_started(participant="participant", generation=1, clean=True)
    await proposals.consume_input("yes", expected_proposal=proposal)
    assert surface.calls == []
    now[0] += 41
    assert (
        await proposals.decide(
            participant="participant",
            generation=1,
            proposal_id=proposal.proposal_id,
            objective=proposal.objective,
            decision="confirm",
        )
        is WorkCommandAdmission.REFUSED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("participant,generation", [("other", 1), ("participant", 2)])
async def test_stale_capture_cannot_bind_a_current_proposal(participant, generation):
    proposals, surface, _events, _now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.capture_started(participant=participant, generation=generation, clean=True)
    await proposals.consume_input("yes", expected_proposal=proposal)
    assert surface.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("speech", ["foreign", "late_own_chunk"])
async def test_speech_after_capture_invalidates_even_late_same_proposal_audio(speech):
    proposals, surface, _events, _now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.capture_started(participant="participant", generation=1, clean=True)
    proposals.speech_started(proposal if speech == "late_own_chunk" else None)
    assert (
        await proposals.consume_input("yes", expected_proposal=proposal)
        is WorkCommandAdmission.REFUSED
    )
    assert surface.calls == []


@pytest.mark.asyncio
async def test_expired_readback_completion_cannot_revive_proposal():
    proposals, surface, _events, now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    now[0] += 61
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.capture_started(participant="participant", generation=1, clean=True)
    assert (
        await proposals.consume_input("yes", expected_proposal=proposal)
        is WorkCommandAdmission.REFUSED
    )
    assert surface.calls == []


@pytest.mark.asyncio
async def test_spoken_assent_after_expiry_is_a_consumed_refusal_not_a_worker_failure():
    proposals, surface, _events, now = owner()
    proposals._speech_projection = lambda text: text
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    proposals.capture_started(participant="participant", generation=1, clean=True)
    now[0] += 61
    assert (
        await proposals.consume_input("yes", expected_proposal=proposal)
        is WorkCommandAdmission.REFUSED
    )
    assert surface.calls == []


@pytest.mark.asyncio
async def test_idempotent_bind_does_not_retire_or_change_current_proposal():
    proposals, surface, _events, _now = owner()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    proposals.bind("participant", 1)
    await proposals.decide(
        participant="participant",
        generation=1,
        proposal_id=proposal.proposal_id,
        objective=proposal.objective,
        decision="confirm",
    )
    assert surface.calls == [(proposal.objective, "confirmed_" + proposal.proposal_id)]


def test_proposal_ids_never_reuse_and_no_hidden_256_proposal_limit():
    proposals, _surface, _events, _now = owner()
    identities = set()
    for _ordinal in range(300):
        proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
        assert proposal.proposal_id not in identities
        identities.add(proposal.proposal_id)
        proposals.retire("synthetic_turn")
    assert len(identities) == 300


@pytest.mark.parametrize("character", ["x", "漢", "\U0001f642", '"'])
def test_shared_proposal_bound_counts_unicode_code_points_without_truncating(character):
    from hermes_realtime.conversation.task_proposals import MAX_TASK_PROPOSAL_OBJECTIVE_CHARS

    assert MAX_TASK_PROPOSAL_OBJECTIVE_CHARS == 1024
    proposals, _surface, _events, _now = owner()
    exact = character * 1024
    proposal = proposals.offer(HostTaskProposalRequest(exact))
    assert proposal.objective == exact
    with pytest.raises(ValueError):
        HostTaskProposalRequest(character * 1025)


def test_card_projection_failure_never_installs_an_actionable_proposal():
    proposals, surface, _events, _now = owner()

    def fail_projection(*_):
        raise RuntimeError("synthetic projection failure")

    proposals._observer = fail_projection
    with pytest.raises(RuntimeError, match="synthetic projection failure"):
        proposals.offer(HostTaskProposalRequest("Inspect release"))
    assert not proposals.has_pending
    assert surface.calls == []
