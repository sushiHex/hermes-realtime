"""Host-targeted natural cancel (#237): the host freezes the target, the model only selects.

The model never names a task. The host offers the route only when exactly one task is
running and no start awaits acknowledgment, freezes that task when the user's final input
is admitted, and cancels exactly it. Speech about the cancellation is either a fixed host
sentence for a refusal or a tool-free continuation grounded in the accepted cancel fact.
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
    WorkCancelResult,
)
from hermes_realtime.conversation.context import (
    ActiveTaskSummary,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.conversation.natural_cancel import (
    CANCEL_NOT_ACCEPTED,
    CANCEL_TARGET_ENDED,
    CANCEL_UNAVAILABLE,
    HostCancelRequest,
    unaccepted_cancel_speech,
)
from hermes_realtime.conversation.streaming import (
    AcceptedCancelInferenceRequest,
    AcceptedStartInferenceRequest,
    ConversationInferenceRequest,
)
from hermes_realtime.conversation.task_status import TaskStatusSnapshot
from hermes_realtime.conversation.turn import ForegroundTurn, turn_routes
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
from tests.conversation.test_work_tools import ControllerProbe

_ACK = "Okay, I'm stopping that now."


class StatusProbe(ControllerProbe):
    """The test controller plus the authoritative status view the real controller has."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pending = 0
        self.cancel_requested: set[str] = set()

    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        self.pending += 1
        try:
            return await super().dispatch(objective=objective, utterance_id=utterance_id)
        finally:
            self.pending -= 1

    async def request_cancel(
        self, task_id: str, *, reason: str | None = None,
    ) -> TaskCancelOutcome:
        self.cancel_requested.add(task_id)
        return await super().request_cancel(task_id, reason=reason)

    def task_status(self) -> TaskStatusSnapshot:
        active = {task.task_id for task in self.context.snapshot().active_tasks}
        cancelling = len(active & self.cancel_requested)
        return TaskStatusSnapshot(
            pending=self.pending, active=len(active) - cancelling, cancelling=cancelling,
        )


def _accept(context: ConversationContextStore, name: str) -> None:
    context.record_task_accepted(
        task_id=f"task_{name}", run_id=f"deleg_private_{name}", objective=f"Inspect {name}",
    )


def _surface(
    context: ConversationContextStore, *, cancel_accepted: bool = True,
    dispatch_gate: asyncio.Event | None = None,
) -> tuple[ConversationWorkControlSurface, StatusProbe, list[tuple[str, dict[str, object]]]]:
    controller = StatusProbe(
        context=context,
        dispatch_outcome=TaskDispatchOutcome(task_id="task_started", accepted=True),
        cancel_outcome=TaskCancelOutcome(
            task_id="task_a", accepted=cancel_accepted,
            reason=None if cancel_accepted else "Hermes declined cancellation",
        ),
        dispatch_gate=dispatch_gate,
    )
    events: list[tuple[str, dict[str, object]]] = []
    surface = ConversationWorkControlSurface(
        controller=controller, context=context,
        observer=lambda kind, data: events.append((kind, dict(data))),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "local_1",
    )
    return surface, controller, events


class Inference:
    """Records every request; selects cancel on the first, acknowledges on the second."""

    def __init__(self, *, select: object = None, before_select=None, after_select=None) -> None:
        self.calls: list[object] = []
        self._select = HostCancelRequest() if select is None else select
        self._before_select = before_select
        self._after_select = after_select

    async def stream(self, snapshot, *, turn_id):
        self.calls.append(snapshot)
        if type(snapshot) is AcceptedCancelInferenceRequest:
            yield _ACK
            return
        if self._before_select is not None:
            self._before_select()
        yield self._select
        if self._after_select is not None:
            self._after_select()

    async def cancel(self, turn_id):
        pass


def _loop(context, surface, inference, *, foreground=None):
    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground or ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_cancellation=surface,
    )
    return loop, synth


async def _respond(loop, text: str = "Stop that.") -> None:
    await loop.respond("turn_stop", Transcript(text=text, final=True))


# ------------------------------------------------------------------------ the freeze


@pytest.mark.asyncio
@pytest.mark.parametrize("tasks", [0, 1, 2])
async def test_target_exists_only_with_exactly_one_running_task(tasks: int) -> None:
    context = ConversationContextStore(max_active_tasks=3)
    for name in ("a", "b")[:tasks]:
        _accept(context, name)
    surface, _controller, _events = _surface(context)
    try:
        assert surface.natural_cancel_target() == ("task_a" if tasks == 1 else None)
    finally:
        await surface.close()


@pytest.mark.asyncio
async def test_pending_start_withdraws_the_target() -> None:
    gate = asyncio.Event()
    context = ConversationContextStore(max_active_tasks=3)
    _accept(context, "a")
    surface, controller, _events = _surface(context, dispatch_gate=gate)
    start = asyncio.create_task(
        surface.start_work(objective="Inspect more", invocation_id="start_pending")
    )
    try:
        await controller.dispatch_started.wait()
        assert surface.natural_cancel_target() is None
        gate.set()
        await start
        # Two running tasks afterwards: still no single target.
        assert surface.natural_cancel_target() is None
    finally:
        gate.set()
        await surface.close()


@pytest.mark.asyncio
async def test_a_task_already_cancelling_is_not_a_target() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, _controller, _events = _surface(context)
    try:
        assert (await surface.cancel_work(task_id="task_a", invocation_id="card_cancel")).accepted
        assert surface.natural_cancel_target() is None
    finally:
        await surface.close()


@pytest.mark.asyncio
async def test_unavailable_status_or_closed_control_has_no_target() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, _events = _surface(context)
    controller.task_status = lambda: TaskStatusSnapshot(available=False)  # type: ignore[method-assign]
    assert surface.natural_cancel_target() is None
    del controller.task_status
    await surface.close()
    assert surface.natural_cancel_target() is None


@pytest.mark.asyncio
async def test_context_and_controller_must_agree_on_the_one_task() -> None:
    # The controller reports one running task while the context holds two.
    context = ConversationContextStore(max_active_tasks=3)
    _accept(context, "a")
    _accept(context, "b")
    surface, controller, _events = _surface(context)
    controller.task_status = lambda: TaskStatusSnapshot(active=1)  # type: ignore[method-assign]
    try:
        assert surface.natural_cancel_target() is None
    finally:
        await surface.close()


class _Owner:
    """A cancellation owner that names whatever target it is given."""

    def __init__(self, target: str | None) -> None:
        self.target = target
        self.cancels: list[str] = []

    def natural_cancel_target(self) -> str | None:
        return self.target

    async def cancel_work(
        self, *, task_id: str, invocation_id: str, still_current=None,
    ) -> WorkCancelResult:
        self.cancels.append(task_id)
        return WorkCancelResult(accepted=True, state="cancelling", task_id=task_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(("tasks", "target"), [(2, "task_a"), (1, "task_other"), (0, "task_a")])
async def test_loop_freezes_only_a_target_that_is_this_snapshots_one_task(tasks, target) -> None:
    context = ConversationContextStore(max_active_tasks=3)
    for name in ("a", "b")[:tasks]:
        _accept(context, name)
    owner = _Owner(target)
    inference = Inference()
    loop, _synth = _loop(context, owner, inference)
    try:
        with pytest.raises(ValueError, match="unadvertised_cancel"):
            await _respond(loop)
        assert inference.calls[0].cancel_offered is False
        assert owner.cancels == []
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_an_answer_to_a_heard_proposal_readback_offers_no_cancel() -> None:
    # "No, cancel that" after a readback is about the proposal, not the running task.
    context = ConversationContextStore()
    context.restore(DurableConversation(messages=(
        ConversationMessage("user", "Look into b."),
        ConversationMessage(
            "assistant", "Proposed background task. Inspect b. Shall I send this task?",
            source="host_task_proposal",
        ),
    ), prior_work=False))
    _accept(context, "a")
    surface, controller, _events = _surface(context)
    inference = Inference()
    loop, _synth = _loop(context, surface, inference)
    try:
        with pytest.raises(ValueError, match="unadvertised_cancel"):
            await _respond(loop, "No, cancel that.")
        assert inference.calls[0].cancel_offered is False
        assert controller.cancellations == []
    finally:
        await loop.close()
        await surface.close()


# ------------------------------------------------------------------ the advertisement


@pytest.mark.asyncio
@pytest.mark.parametrize(("tasks", "offered"), [(0, False), (1, True), (2, False)])
async def test_route_is_offered_on_the_user_turn_only_with_one_task(tasks, offered) -> None:
    # Mutation: offer the route whenever any task is active (advertise with two tasks).
    context = ConversationContextStore(max_active_tasks=3)
    for name in ("a", "b")[:tasks]:
        _accept(context, name)
    surface, controller, _events = _surface(context)
    inference = Inference(select="Just chatting.")
    loop, _synth = _loop(context, surface, inference)
    try:
        await _respond(loop, "How is it going?")
        (request,) = inference.calls
        assert type(request) is ConversationInferenceRequest
        assert request.cancel_offered is offered
        assert controller.cancellations == []
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_route_is_not_offered_while_a_start_awaits_acknowledgment() -> None:
    gate = asyncio.Event()
    context = ConversationContextStore(max_active_tasks=3)
    _accept(context, "a")
    surface, controller, _events = _surface(context, dispatch_gate=gate)
    start = asyncio.create_task(
        surface.start_work(objective="Inspect more", invocation_id="start_pending")
    )
    inference = Inference(select="Just chatting.")
    loop, _synth = _loop(context, surface, inference)
    try:
        await controller.dispatch_started.wait()
        await _respond(loop)
        assert inference.calls[0].cancel_offered is False
    finally:
        gate.set()
        await start
        await loop.close()
        await surface.close()


def test_request_cannot_offer_the_route_without_exactly_one_active_task() -> None:
    message = (ConversationMessage("user", "Stop that."),)
    one = (ActiveTaskSummary("task_a", "Inspect a"),)
    two = (*one, ActiveTaskSummary("task_b", "Inspect b"))
    assert ConversationInferenceRequest(
        revision=1, messages=message, active_tasks=one, cancel_offered=True,
    ).cancel_offered is True
    for tasks in ((), two):
        with pytest.raises(ValueError, match="exactly one active task"):
            ConversationInferenceRequest(
                revision=1, messages=message, active_tasks=tasks, cancel_offered=True,
            )
    with pytest.raises(TypeError, match="exact boolean"):
        ConversationInferenceRequest(
            revision=1, messages=message, active_tasks=one, cancel_offered=1,  # type: ignore[arg-type]
        )


def test_non_user_turns_never_route_cancellation() -> None:
    message = (ConversationMessage("user", "Stop that."),)
    one = (ActiveTaskSummary("task_a", "Inspect a"),)
    offered = ConversationInferenceRequest(
        revision=1, messages=message, active_tasks=one, cancel_offered=True,
    )
    plain = offered.context
    acknowledged = AcceptedStartInferenceRequest(
        revision=1, messages=message, active_tasks=one, task_id="task_a",
    )
    cancelling = AcceptedCancelInferenceRequest(
        revision=1, messages=message, active_tasks=one, task_id="task_a",
    )
    ends_assistant = ConversationInferenceRequest(
        revision=1, messages=(*message, ConversationMessage("assistant", "Okay.")),
        active_tasks=one, cancel_offered=True,
    )

    def routes(request, **overrides):
        options = {"work_bound": True, "private_material": False, "lookup_attempted": False,
                   "start_requires_admission": False, **overrides}
        return turn_routes(ForegroundTurn(request), **options)

    assert routes(offered).cancel_work is True
    assert routes(offered, work_bound=False).cancel_work is False
    assert routes(offered, private_material=True).cancel_work is False
    # A lookup spends only the start, as it does for status.
    assert routes(offered, lookup_attempted=True).cancel_work is True
    for request in (plain, acknowledged, cancelling, ends_assistant):
        assert routes(request).cancel_work is False
        assert routes(request).task_status is False


# ---------------------------------------------------------------------- the execution


@pytest.mark.asyncio
async def test_selection_cancels_exactly_the_frozen_task_and_acknowledges_from_the_fact() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, events = _surface(context)
    inference = Inference()
    loop, synth = _loop(context, surface, inference)
    try:
        await _respond(loop)
        assert controller.cancellations == [("task_a", "user requested task cancellation")]
        assert events == [("task_state", {"status": "cancelling", "taskId": "task_a"})]
        first, second = inference.calls
        assert first.cancel_offered is True
        assert type(second) is AcceptedCancelInferenceRequest
        assert second.task_id == "task_a"
        assert synth.texts == [_ACK]
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_target_frozen_at_admission_refuses_stale_and_never_touches_a_new_task() -> None:
    # A ends and B starts between admission and selection. Mutation: resolve the target
    # when the model selects (B gets cancelled).
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, events = _surface(context)

    def swap() -> None:
        context.record_task_completed(
            task_id="task_a", run_id="deleg_private_a",
            result_status="completed", result_text="Done.",
        )
        _accept(context, "b")

    inference = Inference(before_select=swap)
    loop, synth = _loop(context, surface, inference)
    try:
        await _respond(loop)
        assert controller.cancellations == []
        assert [task.task_id for task in context.snapshot().active_tasks] == ["task_b"]
        assert events == [("task_state", {
            "status": "rejected", "taskId": "task_a", "reason": "task is not active",
        })]
        assert len(inference.calls) == 1
        assert synth.texts == [CANCEL_TARGET_ENDED]
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tasks", [0, 2])
async def test_forced_selection_when_unadvertised_has_zero_effect(tasks: int) -> None:
    context = ConversationContextStore(max_active_tasks=3)
    for name in ("a", "b")[:tasks]:
        _accept(context, name)
    surface, controller, events = _surface(context)
    inference = Inference()
    loop, synth = _loop(context, surface, inference)
    try:
        with pytest.raises(ValueError, match="host speech refused: unadvertised_cancel"):
            await _respond(loop)
        assert controller.cancellations == [] and events == []
        assert len(inference.calls) == 1 and synth.texts == []
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_forced_selection_without_a_cancellation_owner_has_zero_effect() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, _events = _surface(context)
    inference = Inference()
    loop, _synth = _loop(context, None, inference)
    try:
        with pytest.raises(ValueError, match="unadvertised_cancel"):
            await _respond(loop)
        assert inference.calls[0].cancel_offered is False
        assert controller.cancellations == []
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_rejected_cancel_speaks_a_fixed_sentence_and_no_model_acknowledgment() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, _events = _surface(context, cancel_accepted=False)
    inference = Inference()
    loop, synth = _loop(context, surface, inference)
    try:
        await _respond(loop)
        assert controller.cancellations == [("task_a", "user requested task cancellation")]
        assert len(inference.calls) == 1
        assert synth.texts == [CANCEL_NOT_ACCEPTED]
        assert "Hermes declined" not in " ".join(synth.texts)
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_superseded_turn_never_cancels() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, _events = _surface(context)
    foreground = ForegroundTurnCoordinator()

    def supersede() -> None:
        foreground._is_current = lambda _lease: False  # type: ignore[method-assign]

    inference = Inference(before_select=supersede)
    loop, synth = _loop(context, surface, inference, foreground=foreground)
    try:
        await _respond(loop)
        assert controller.cancellations == []
        assert synth.texts == []
    finally:
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded", [True, False], ids=["superseded", "control"])
async def test_a_turn_superseded_while_admission_waits_cancels_nothing(superseded) -> None:
    # The lease is checked before the cancel is awaited; the admission lock is the
    # commit point. Hold the lock, supersede the turn, release: no cancel may commit.
    # Mutation: drop the lease re-read under the admission lock.
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, events = _surface(context)
    foreground = ForegroundTurnCoordinator()
    inference = Inference()
    loop, synth = _loop(context, surface, inference, foreground=foreground)
    try:
        await surface._lock.acquire()
        turn = asyncio.create_task(_respond(loop))
        async with asyncio.timeout(1):
            while not getattr(surface._lock, "_waiters", None):
                await asyncio.sleep(0)
        if superseded:
            foreground._is_current = lambda _lease: False  # type: ignore[method-assign]
        surface._lock.release()
        async with asyncio.timeout(1):
            await turn
        if superseded:
            assert controller.cancellations == [] and events == []
            assert len(inference.calls) == 1 and synth.texts == []
            assert surface.health == "open"
        else:
            assert controller.cancellations == [("task_a", "user requested task cancellation")]
            assert synth.texts == [_ACK]
    finally:
        if surface._lock.locked():
            surface._lock.release()
        await loop.close()
        await surface.close()


@pytest.mark.asyncio
async def test_surface_admits_nothing_once_the_callers_condition_fails() -> None:
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, events = _surface(context)
    try:
        result = await surface.cancel_work(
            task_id="task_a", invocation_id="natural_cancel_x", still_current=lambda: False,
        )
        assert result == WorkCancelResult(
            accepted=False, state="rejected", task_id="task_a",
            reason="request superseded before admission",
        )
        assert controller.cancellations == [] and events == []
        assert surface.health == "open"
        assert "natural_cancel_x" not in surface._bindings
        with pytest.raises(TypeError, match="still_current"):
            await surface.cancel_work(
                task_id="task_a", invocation_id="natural_cancel_y", still_current=True,  # type: ignore[arg-type]
            )
    finally:
        await surface.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["at_selection", "after_selection"])
async def test_a_superseded_turn_never_reaches_the_cancellation_owner(when) -> None:
    # The owner here ignores still_current, so only the loop's own lease guards stand
    # between a superseded turn and the call. Mutation: drop the guard at selection
    # (at_selection) or the one before the call (after_selection).
    context = ConversationContextStore()
    _accept(context, "a")
    owner = _Owner("task_a")
    foreground = ForegroundTurnCoordinator()

    def supersede() -> None:
        foreground._is_current = lambda _lease: False  # type: ignore[method-assign]

    resumed: list[bool] = []
    inference = (
        Inference(before_select=supersede, after_select=lambda: resumed.append(True))
        if when == "at_selection" else Inference(after_select=supersede)
    )
    loop, synth = _loop(context, owner, inference, foreground=foreground)
    try:
        await _respond(loop)
        assert owner.cancels == [] and synth.texts == []
        # A superseded selection also stops consuming the stream at once.
        assert resumed == []
    finally:
        await loop.close()


def test_refusal_sentence_is_chosen_by_typed_category_only() -> None:
    assert unaccepted_cancel_speech(
        WorkCancelResult(accepted=True, state="cancelling", task_id="task_a")
    ) is None
    assert unaccepted_cancel_speech(WorkCancelResult(
        accepted=False, state="rejected", reason="work control unavailable; restart required",
    )) == CANCEL_UNAVAILABLE
    assert unaccepted_cancel_speech(WorkCancelResult(
        accepted=False, state="rejected", task_id="task_a", reason="task is not active",
    )) == CANCEL_TARGET_ENDED
    assert unaccepted_cancel_speech(WorkCancelResult(
        accepted=False, state="rejected", task_id="task_a", reason="SENTINEL 7f3a",
    )) == CANCEL_NOT_ACCEPTED
    for sentence in (CANCEL_UNAVAILABLE, CANCEL_TARGET_ENDED, CANCEL_NOT_ACCEPTED):
        assert "stopped" not in sentence
    with pytest.raises(TypeError):
        unaccepted_cancel_speech(None)  # type: ignore[arg-type]


def test_cancel_selection_carries_nothing() -> None:
    assert HostCancelRequest() == HostCancelRequest()
    with pytest.raises(TypeError):
        HostCancelRequest("task_a")  # type: ignore[call-arg]
