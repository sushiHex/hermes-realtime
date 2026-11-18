"""Confirmed natural cancel with several running tasks (#237, owner decision 2026-11-18).

With exactly one running task the existing route stays: the host froze that task and a
selection cancels exactly it. With several, the model may only nominate one of the choices
the host froze when the user's final input was admitted. A nomination never cancels: the host
speaks its own readback of that exact task, and only a fresh spoken affirmative of that
completed readback cancels it, through the same exact-task command path as the task card.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationWorkControlSurface,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
    TaskCancelOutcome,
    TaskDispatchOutcome,
)
from hermes_realtime.conversation.natural_cancel import (
    CANCEL_NOT_ACCEPTED,
    CANCEL_TARGET_ENDED,
    CancelChoice,
    HostCancelRequest,
)
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.work_tools import WorkCommandAdmission
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_natural_cancel import StatusProbe
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer

_READBACK_A = "Stop background task. Inspect alpha?"
_READBACK_B = "Stop background task. Inspect beta?"


class ExactProbe(StatusProbe):
    """Accepts every cancel for the task it names; records each controller call."""

    async def request_cancel(self, task_id, *, reason=None):
        self.cancellations.append((task_id, reason))
        self.cancel_requested.add(task_id)
        return TaskCancelOutcome(task_id=task_id, accepted=True)


def _accept(context: ConversationContextStore, name: str) -> None:
    context.record_task_accepted(
        task_id=f"task_{name}", run_id=f"deleg_private_{name}", objective=f"Inspect {name}",
    )


def _end(context: ConversationContextStore, name: str) -> None:
    context.record_task_completed(task_id=f"task_{name}", run_id=f"deleg_private_{name}")


class Nominations:
    """Yields one scripted selection per user turn and records every request."""

    def __init__(self, *selections: object) -> None:
        self.calls: list[object] = []
        self._selections = list(selections)

    async def stream(self, snapshot, *, turn_id):
        self.calls.append(snapshot)
        if type(snapshot) is not ConversationInferenceRequest:
            yield "Okay, stopping that."
            return
        selection = self._selections.pop(0)
        if callable(selection) and not isinstance(selection, HostCancelRequest):
            selection = selection()
        yield selection

    async def cancel(self, turn_id):
        pass


def _owner(*names: str, inference: Nominations, foreground=None, max_segment_chars=4096):
    context = ConversationContextStore()
    for name in names:
        _accept(context, name)
    controller = ExactProbe(
        context=context,
        dispatch_outcome=TaskDispatchOutcome(task_id="task_started", accepted=True),
        cancel_outcome=TaskCancelOutcome(task_id="task_alpha", accepted=True),
    )
    surface = ConversationWorkControlSurface(
        controller=controller, context=context,
        observer=lambda *_: None, reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "local_1",
    )
    events: list[tuple[str, dict[str, object]]] = []
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda kind, data: events.append((kind, dict(data))),
        speech_projection=lambda text: text,
    )
    proposals.bind("participant", 1)
    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground or ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_cancellation=surface,
        task_proposals=proposals,
        max_segment_chars=max_segment_chars,
    )
    return loop, context, controller, proposals, synth, events


async def _say(loop, proposals, text: str, turn: str, *, clean: bool = True) -> None:
    """The spoken route in the worker's order: speech onset latches the capture, then the
    final consumes a pending confirmation, and anything else is an ordinary turn."""
    proposals.capture_started(participant="participant", generation=1, clean=clean)
    target = proposals.confirmation_target(text)
    if target is None:
        if proposals.has_pending:
            proposals.intervening_input()
        await loop.respond(turn, Transcript(text=text, final=True))
        return
    await proposals.consume_input(text, expected_proposal=target)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


# ------------------------------------------------------------ first RED (Astra 9107)


@pytest.mark.asyncio
async def test_wrong_valid_nomination_mutates_nothing_until_the_right_task_is_affirmed():
    inference = Nominations(
        HostCancelRequest(target="choice_2"),  # the user meant alpha; the model picked beta
        HostCancelRequest(target="choice_1"),
    )
    loop, context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )

    await _say(loop, proposals, "Stop the alpha one.", "turn_1")
    request = inference.calls[0]
    assert type(request) is ConversationInferenceRequest
    assert request.cancel_offered is False
    assert request.cancel_choices == (
        CancelChoice("choice_1", "Inspect alpha"), CancelChoice("choice_2", "Inspect beta"),
    )
    assert synth.texts == [_READBACK_B]
    assert controller.cancellations == []

    await _say(loop, proposals, "No, the other one.", "turn_2")
    await _settle()
    assert controller.cancellations == []
    assert synth.texts == [_READBACK_B, _READBACK_A]

    await _say(loop, proposals, "yes", "turn_3")
    await _settle()
    assert controller.cancellations == [("task_alpha", "user requested task cancellation")]
    assert {task.task_id for task in context.snapshot().active_tasks} == {
        "task_alpha", "task_beta",
    }
    assert not proposals.has_pending


# ------------------------------------------------------------------ the freeze


@pytest.mark.asyncio
async def test_sole_task_keeps_the_immediate_route_and_offers_no_choices():
    inference = Nominations(HostCancelRequest())
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", inference=inference,
    )
    await _say(loop, proposals, "Stop that.", "turn_1")
    await _settle()
    assert inference.calls[0].cancel_offered is True
    assert inference.calls[0].cancel_choices == ()
    assert controller.cancellations == [("task_alpha", "user requested task cancellation")]
    assert not proposals.has_pending


@pytest.mark.asyncio
async def test_choices_require_a_proposal_owner():
    context = ConversationContextStore()
    _accept(context, "alpha")
    _accept(context, "beta")
    controller = ExactProbe(
        context=context,
        dispatch_outcome=TaskDispatchOutcome(task_id="task_started", accepted=True),
        cancel_outcome=TaskCancelOutcome(task_id="task_alpha", accepted=True),
    )
    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=lambda *_: None,
        reserve_observer_capacity=lambda: None, utterance_id_factory=lambda: "local_1",
    )
    inference = Nominations("Use the task card to stop one.")
    loop = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=SegmentSynthesizer(), playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(), task_cancellation=surface,
    )
    await loop.respond("turn_1", Transcript(text="Stop one.", final=True))
    assert inference.calls[0].cancel_choices == ()
    assert inference.calls[0].cancel_offered is False


@pytest.mark.asyncio
async def test_indistinguishable_objectives_offer_no_choices():
    inference = Nominations("Use the task cards.")
    loop, context, _controller, proposals, _synth, _events = _owner(inference=inference)
    context.record_task_accepted(task_id="task_one", run_id="deleg_private_1",
                                 objective="Inspect release")
    context.record_task_accepted(task_id="task_two", run_id="deleg_private_2",
                                 objective="inspect release ")
    await _say(loop, proposals, "Stop the release one.", "turn_1")
    assert inference.calls[0].cancel_choices == ()


@pytest.mark.asyncio
async def test_objectives_that_read_back_identically_offer_no_choices():
    inference = Nominations("Use the task cards.")
    loop, context, _controller, proposals, _synth, _events = _owner(inference=inference)
    context.record_task_accepted(task_id="task_one", run_id="deleg_private_1",
                                 objective="Inspect alpha.")
    context.record_task_accepted(task_id="task_two", run_id="deleg_private_2",
                                 objective="Inspect alpha")
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    assert inference.calls[0].cancel_choices == ()


@pytest.mark.asyncio
async def test_more_tasks_than_the_choice_bound_offer_no_choices():
    inference = Nominations("Use the task cards.")
    loop, _context, _controller, proposals, _synth, _events = _owner(
        "a1", "a2", "a3", "a4", "a5", inference=inference,
    )
    await _say(loop, proposals, "Stop one.", "turn_1")
    assert inference.calls[0].cancel_choices == ()


@pytest.mark.asyncio
async def test_a_task_set_change_after_admission_withdraws_the_choices_never_downgrades():
    """Beta ends between admission and the request: no choice and no immediate sole cancel."""
    inference = Nominations("Which one?")
    loop, context, controller, _proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    frozen = loop.freeze_natural_cancel_target()
    assert frozen == ("task_alpha", "task_beta")
    _end(context, "beta")
    await loop._run_turn(
        "turn_1", transcript=Transcript(text="Stop it.", final=True), announcement=None,
        updates=(), update_operation=None, cancel_target=frozen,
    )
    assert inference.calls[0].cancel_choices == ()
    assert inference.calls[0].cancel_offered is False
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_a_card_cancel_after_admission_withdraws_the_choices():
    """The context is unchanged, but the controller now reports a task stopping."""
    inference = Nominations("Which one?")
    loop, _context, controller, _proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    frozen = loop.freeze_natural_cancel_target()
    assert frozen == ("task_alpha", "task_beta")
    controller.cancel_requested.add("task_beta")
    await loop._run_turn(
        "turn_1", transcript=Transcript(text="Stop it.", final=True), announcement=None,
        updates=(), update_operation=None, cancel_target=frozen,
    )
    assert inference.calls[0].cancel_choices == ()
    assert inference.calls[0].cancel_offered is False


@pytest.mark.asyncio
async def test_a_readback_that_would_not_fit_one_segment_offers_no_choices():
    inference = Nominations("Use the task cards.")
    loop, _context, _controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference, max_segment_chars=len(_READBACK_A) - 1,
    )
    await _say(loop, proposals, "Stop one.", "turn_1")
    assert inference.calls[0].cancel_choices == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["set_changed", "indistinguishable", "readback_capacity"])
async def test_each_withdrawal_of_the_choices_leaves_one_bounded_evidence_line(category, capsys):
    inference = Nominations("Use the task cards.")
    loop, context, controller, proposals, _synth, _events = _owner(
        inference=inference,
        max_segment_chars=(len(_READBACK_A) - 1) if category == "readback_capacity" else 4096,
    )
    second = "Inspect alpha." if category == "indistinguishable" else "Inspect beta"
    context.record_task_accepted(task_id="task_one", run_id="deleg_private_1",
                                 objective="Inspect alpha")
    context.record_task_accepted(task_id="task_two", run_id="deleg_private_2",
                                 objective=second)
    frozen = loop.freeze_natural_cancel_target()
    assert frozen == ("task_one", "task_two")
    if category == "set_changed":
        controller.cancel_requested.add("task_two")
    capsys.readouterr()
    await loop._run_turn(
        "turn_1", transcript=Transcript(text="Stop one.", final=True), announcement=None,
        updates=(), update_operation=None, cancel_target=frozen,
    )
    assert inference.calls[0].cancel_choices == ()
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("[cancel-choice-refusal] ")]
    assert lines == [
        '[cancel-choice-refusal] {"category":"' + category + '","tasks":2}'
    ]


@pytest.mark.asyncio
async def test_a_pending_start_withdraws_the_choices():
    gate = asyncio.Event()
    inference = Nominations("Which one?")
    loop, context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    surface = loop._task_cancellation
    controller.dispatch_gate = gate
    start = asyncio.create_task(surface.start_work(objective="Inspect gamma",
                                                   invocation_id="start_1"))
    await controller.dispatch_started.wait()
    try:
        await _say(loop, proposals, "Stop one.", "turn_1")
        assert inference.calls[0].cancel_choices == ()
    finally:
        gate.set()
        await start


# ------------------------------------------------------------- nomination shapes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [HostCancelRequest(), HostCancelRequest(target="choice_3")],
)
async def test_an_unoffered_nomination_shape_has_zero_effect(selection):
    inference = Nominations(selection)
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    with pytest.raises(ValueError, match="host speech refused"):
        await _say(loop, proposals, "Stop one.", "turn_1")
    await _settle()
    assert controller.cancellations == []
    assert not proposals.has_pending
    assert synth.texts == []


@pytest.mark.asyncio
async def test_a_sole_offer_refuses_a_choice_nomination():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", inference=inference,
    )
    with pytest.raises(ValueError, match="host speech refused"):
        await _say(loop, proposals, "Stop it.", "turn_1")
    await _settle()
    assert controller.cancellations == []


@pytest.mark.parametrize("target", ["", "choice_0", "choice_5", "task_alpha", "all", 1])
def test_the_selection_label_is_bounded_and_exact(target):
    with pytest.raises((TypeError, ValueError)):
        HostCancelRequest(target=target)


# ---------------------------------------------------------------- confirmation


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["no", "don't", "No, the other one.", "wait"])
async def test_any_non_affirmative_reply_retires_the_nomination(reply):
    inference = Nominations(HostCancelRequest(target="choice_1"), "Okay.")
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    await _say(loop, proposals, reply, "turn_2")
    await _say(loop, proposals, "yes", "turn_3") if proposals.has_pending else None
    await _settle()
    assert controller.cancellations == []
    assert not proposals.has_pending


@pytest.mark.asyncio
async def test_replayed_affirmative_cancels_once():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    target = proposals.confirmation_target("yes")
    assert target is not None
    first = await proposals.consume_input("yes", expected_proposal=target)
    second = await proposals.consume_input("yes", expected_proposal=target)
    await _settle()
    assert first is WorkCommandAdmission.ADMITTED
    assert second is WorkCommandAdmission.REFUSED
    assert controller.cancellations == [("task_alpha", "user requested task cancellation")]


@pytest.mark.asyncio
async def test_typed_affirmative_retires_the_question_and_continues_as_a_turn():
    """No silent dead end: the typed "yes" is an ordinary turn, and the model may ask again."""
    inference = Nominations(
        HostCancelRequest(target="choice_1"), HostCancelRequest(target="choice_1"),
    )
    loop, _context, controller, proposals, synth, events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.typed_input_started(participant="participant", generation=1)
    assert proposals.confirmation_target("yes") is None
    assert not proposals.has_pending
    await loop.respond("turn_2", Transcript(text="yes", final=True))
    await _settle()
    assert controller.cancellations == []
    # The ordinary turn could only ask again; the user hears the question a second time.
    assert synth.texts == [_READBACK_A, _READBACK_A]
    # No start card is ever published for a cancel question.
    assert all(kind != "task_proposal" for kind, _data in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unclean", "rebind", "expired", "speech", "close"])
async def test_confirmation_requires_clean_current_binding_and_live_readback(change):
    inference = Nominations(HostCancelRequest(target="choice_1"), "Okay.")
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    if change == "rebind":
        proposals.bind("participant", 2)
    elif change == "expired":
        proposals._deadline = 0.0
    elif change == "speech":
        proposals.speech_started()
    elif change == "close":
        proposals.close_binding()
    await _say(loop, proposals, "yes", "turn_2", clean=change != "unclean")
    await _settle()
    assert controller.cancellations == []
    assert not proposals.has_pending


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["rebind", "close", "newer_speech", "newer_typed", "expired", "card_cancel"],
)
async def test_the_confirming_authority_is_rechecked_at_the_locked_commit_point(change):
    """Astra 9177: the consumed question waits for the work surface's admission lock; a
    rebind, close, newer input or expiry while it waits must cancel nothing."""
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    surface = loop._task_cancellation
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    target = proposals.confirmation_target("yes")
    assert target is not None
    await surface._lock.acquire()
    try:
        consume = asyncio.create_task(proposals.consume_input("yes", expected_proposal=target))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not consume.done()  # waiting for the admission lock
        if change == "rebind":
            proposals.bind("participant", 2)
        elif change == "close":
            proposals.close_binding()
        elif change == "newer_speech":
            proposals.capture_started(participant="participant", generation=1, clean=True)
        elif change == "newer_typed":
            proposals.typed_input_started(participant="participant", generation=1)
        elif change == "card_cancel":
            controller.cancel_requested.add("task_beta")
        else:
            proposals._clock = lambda: float("inf")
    finally:
        surface._lock.release()
    assert await consume is WorkCommandAdmission.REFUSED
    await _settle()
    assert controller.cancellations == []


def _router(loop, proposals):
    from hermes_realtime.conversation.commands import ConversationTaskCommandRouter
    from hermes_realtime.evidence import CommandDisposition, InputSource
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(owner_generation=71)
    owner.activate_binding(
        binding_id="00000000-0000-4000-8000-0000000002be",
        binding_generation=1,
        consent_epoch_id="00000000-0000-4000-8000-0000000002bf",
        logical_session_id="00000000-0000-4000-8000-0000000002c0",
    )
    accepted: list[object] = []
    router_events: list[tuple[str, dict[str, object]]] = []
    router = ConversationTaskCommandRouter(
        surface=loop._task_cancellation,
        observer=lambda kind, data: router_events.append((kind, dict(data))),
        reserve_observer_capacity=lambda: None,
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=lambda authority: (
            accepted.append(authority) or CommandDisposition.ADMITTED
        ),
        task_proposals=proposals,
    )

    def mint():
        return owner.mint_final_input(
            source=InputSource.MICROPHONE, input_incarnation=1, media_incarnation=1,
            typed_sequence=None,
        )

    router.events = router_events
    return router, mint, accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "rebind", "newer_speech"])
async def test_route_spoken_carries_the_confirming_authority_to_the_locked_commit(change):
    """The real spoken route: a rebind or newer speech while the confirmed cancel waits for
    the admission lock cancels nothing; otherwise exactly the named task is cancelled."""
    from hermes_realtime.evidence import CommandRoutingOutcome

    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    router, mint, accepted = _router(loop, proposals)
    surface = loop._task_cancellation
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    await surface._lock.acquire()
    try:
        routed = asyncio.create_task(router.route_spoken("yes", mint()))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not routed.done()  # the consumed question waits for the admission lock
        assert not proposals.has_pending
        if change == "rebind":
            proposals.bind("participant", 2)
        elif change == "newer_speech":
            proposals.capture_started(participant="participant", generation=1, clean=True)
    finally:
        surface._lock.release()
    result = await routed
    await _settle()
    if change is None:
        assert result.outcome is CommandRoutingOutcome.ACCEPTED
        assert len(accepted) == 1
        assert controller.cancellations == [("task_alpha", "user requested task cancellation")]
    else:
        assert result.outcome is CommandRoutingOutcome.REJECTED
        assert accepted == []
        assert controller.cancellations == []
        # Superseded is a refusal of this command, never reported as lost task control.
        assert router.events == []


@pytest.mark.asyncio
async def test_an_uninterrupted_confirmation_still_commits_after_waiting_for_the_lock():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    surface = loop._task_cancellation
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    target = proposals.confirmation_target("yes")
    await surface._lock.acquire()
    try:
        consume = asyncio.create_task(proposals.consume_input("yes", expected_proposal=target))
        for _ in range(20):
            await asyncio.sleep(0)
    finally:
        surface._lock.release()
    assert await consume is WorkCommandAdmission.ADMITTED
    await _settle()
    assert controller.cancellations == [("task_alpha", "user requested task cancellation")]


@pytest.mark.asyncio
async def test_speech_between_latch_and_consume_retires_the_question():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    target = proposals.confirmation_target("yes")
    assert target is not None
    proposals.speech_started()  # e.g. a task update spoken while the "yes" is recorded
    assert await proposals.consume_input("yes", expected_proposal=target) is (
        WorkCommandAdmission.REFUSED
    )
    assert not proposals.has_pending
    await _settle()
    assert controller.cancellations == []
    assert all(kind != "task_proposal" for kind, _data in _events)


@pytest.mark.asyncio
async def test_an_interrupted_readback_retires_the_question():
    class FailedPlayback(RecordingPlayback):
        async def play(self, chunk, *, is_valid=None):
            raise RuntimeError("synthetic playback failure")

    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, events = _owner(
        "alpha", "beta", inference=inference,
    )
    loop._playback = FailedPlayback()
    with pytest.raises(RuntimeError, match="synthetic playback failure"):
        await _say(loop, proposals, "Stop alpha.", "turn_1")
    assert not proposals.has_pending
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_no_question_is_held_when_the_turn_has_no_segment_left():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    loop._max_segments = 0
    with pytest.raises(ValueError, match="capacity"):
        await _say(loop, proposals, "Stop alpha.", "turn_1")
    assert not proposals.has_pending
    assert synth.texts == []
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_a_retired_cancel_question_leaves_an_evidence_line(capsys):
    inference = Nominations(HostCancelRequest(target="choice_1"), "Okay.")
    loop, _context, _controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    capsys.readouterr()
    await _say(loop, proposals, "no", "turn_2")
    assert '"category":"intervening_input"' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_target_that_ends_while_pending_retires_and_yes_cancels_nothing():
    inference = Nominations(HostCancelRequest(target="choice_1"), "It already finished.")
    loop, context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    _end(context, "alpha")
    # A successor with the same objective is a different task; "yes" never moves to it.
    context.record_task_accepted(task_id="task_alpha2", run_id="deleg_private_alpha2",
                                 objective="Inspect alpha")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    assert proposals.confirmation_target("yes") is None
    assert not proposals.has_pending
    await _say(loop, proposals, "yes", "turn_2")
    await _settle()
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_a_target_that_ends_between_consume_and_admission_is_refused_as_stale():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, context, controller, proposals, _synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    proposals.capture_started(participant="participant", generation=1, clean=True)
    target = proposals.confirmation_target("yes")
    assert target is not None
    _end(context, "alpha")
    assert await proposals.consume_input("yes", expected_proposal=target) is (
        WorkCommandAdmission.REFUSED
    )
    await _settle()
    assert controller.cancellations == []
    # Refused by the question's own recheck, before any work-surface intake.
    assert not any(key.startswith("confirmed_cancel_") for key in loop._task_cancellation._bindings)


@pytest.mark.asyncio
async def test_a_task_ending_after_nomination_speaks_the_ended_sentence_not_a_readback():
    def end_alpha_then_nominate():
        _end(context, "alpha")
        return HostCancelRequest(target="choice_1")

    inference = Nominations(end_alpha_then_nominate)
    loop, context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    await _settle()
    assert synth.texts == [CANCEL_TARGET_ENDED]
    assert not proposals.has_pending
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_another_cancellation_in_flight_refuses_the_readback():
    def cancel_beta_by_card_then_nominate():
        controller.cancel_requested.add("task_beta")
        return HostCancelRequest(target="choice_1")

    inference = Nominations(cancel_beta_by_card_then_nominate)
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    assert synth.texts == [CANCEL_NOT_ACCEPTED]
    assert not proposals.has_pending


@pytest.mark.asyncio
async def test_a_proposal_already_pending_refuses_the_readback_and_keeps_that_proposal():
    def pend_a_start_then_nominate():
        proposals.offer(HostTaskProposalRequest("Inspect gamma"))
        return HostCancelRequest(target="choice_1")

    inference = Nominations(pend_a_start_then_nominate)
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    assert synth.texts == [CANCEL_NOT_ACCEPTED]
    assert proposals._pending is not None and proposals._pending.cancel_task_id is None
    assert controller.cancellations == []


def test_private_material_withdraws_the_choices_with_the_route():
    from hermes_realtime.conversation.context import ActiveTaskSummary, ConversationMessage
    from hermes_realtime.conversation.turn import ForegroundTurn, turn_routes

    turn = ForegroundTurn(ConversationInferenceRequest(
        revision=1, messages=(ConversationMessage("user", "Stop one."),),
        active_tasks=(ActiveTaskSummary("task_alpha", "Inspect alpha"),
                      ActiveTaskSummary("task_beta", "Inspect beta")),
        cancel_choices=(CancelChoice("choice_1", "Inspect alpha"),
                        CancelChoice("choice_2", "Inspect beta")),
    ))
    offered = turn_routes(turn, work_bound=True, private_material=False,
                          lookup_attempted=False, start_requires_admission=False)
    assert offered.cancel_work is True and len(offered.cancel_choices) == 2
    private = turn_routes(turn, work_bound=True, private_material=True,
                          lookup_attempted=False, start_requires_admission=False)
    assert private.cancel_work is False and private.cancel_choices == ()


# ------------------------------------------------------------ start / cancel isolation


@pytest.mark.asyncio
async def test_a_start_card_decision_cannot_confirm_a_pending_cancel():
    inference = Nominations(HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, _synth, events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop alpha.", "turn_1")
    pending = proposals._pending
    assert pending is not None
    assert await proposals.decide(
        participant="participant", generation=1, proposal_id=pending.proposal_id,
        objective=pending.objective, decision="confirm",
    ) is WorkCommandAdmission.REFUSED
    await _settle()
    assert controller.cancellations == []
    assert controller.dispatches == []
    # A spoken cancel never publishes a start card.
    assert all(kind != "task_proposal" for kind, _data in events)


@pytest.mark.asyncio
async def test_a_pending_start_proposal_is_not_replaced_by_a_cancel_and_yes_starts_it_only():
    inference = Nominations(HostTaskProposalRequest("Inspect gamma"))
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Please inspect gamma.", "turn_1")
    assert synth.texts == [
        "Proposed background task. Inspect gamma Shall I send this task?"
    ]
    with pytest.raises(PermissionError):
        proposals.offer_cancel(task_id="task_alpha", objective="Inspect alpha")
    await _say(loop, proposals, "yes", "turn_2")
    await _settle()
    assert controller.cancellations == []
    assert [objective for objective, _ in controller.dispatches] == ["Inspect gamma"]
