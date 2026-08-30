from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from hermes_realtime.conversation import (
    ActiveTaskIdentityError,
    ConversationContextStore,
    ConversationWorkControlSurface,
    PrivateRunDisclosureError,
    TaskCancelOutcome,
    TaskDispatchOutcome,
    WorkCancelResult,
    WorkStartResult,
)


@dataclass
class ControllerProbe:
    context: ConversationContextStore
    dispatch_outcome: TaskDispatchOutcome
    cancel_outcome: TaskCancelOutcome
    dispatch_gate: asyncio.Event | None = None

    def __post_init__(self) -> None:
        self.dispatches: list[tuple[str, str]] = []
        self.cancellations: list[tuple[str, str | None]] = []
        self.dispatch_started = asyncio.Event()
        self._run_number = 0

    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        self.dispatches.append((objective, utterance_id))
        self.dispatch_started.set()
        if self.dispatch_gate is not None:
            await self.dispatch_gate.wait()
        outcome = self.dispatch_outcome
        if outcome.accepted:
            self._run_number += 1
            self.context.record_task_accepted(
                task_id=outcome.task_id,
                run_id=f"deleg_private_{self._run_number}",
                objective=objective,
            )
        return outcome

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        self.cancellations.append((task_id, reason))
        return self.cancel_outcome


def _surface(
    *,
    context: ConversationContextStore | None = None,
    accepted: bool = True,
    dispatch_gate: asyncio.Event | None = None,
    observer: Any = None,
    reserve: Any = None,
    max_invocations: int = 32,
    close_drain_timeout_ms: int = 1000,
) -> tuple[
    ConversationWorkControlSurface,
    ControllerProbe,
    list[tuple[str, dict[str, str | int | bool | None]]],
]:
    context = context or ConversationContextStore()
    dispatch = TaskDispatchOutcome(
        task_id="task_release_check",
        accepted=accepted,
        reason=None if accepted else "Hermes declined the request",
    )
    controller = ControllerProbe(
        context=context,
        dispatch_outcome=dispatch,
        cancel_outcome=TaskCancelOutcome(
            task_id="task_release_check",
            accepted=True,
        ),
        dispatch_gate=dispatch_gate,
    )
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=observer or (lambda kind, data: events.append((kind, data))),
        reserve_observer_capacity=reserve or (lambda: None),
        utterance_id_factory=lambda: "local_1",
        max_invocations=max_invocations,
        close_drain_timeout_ms=close_drain_timeout_ms,
    )
    return surface, controller, events


def test_context_exposes_configured_work_limits_read_only() -> None:
    context = ConversationContextStore(max_item_chars=37, max_active_tasks=3)

    assert context.max_item_chars == 37
    assert context.max_active_tasks == 3


def test_results_are_frozen_exact_and_private_safe() -> None:
    result = WorkStartResult(
        accepted=True,
        state="active",
        task_id="task_public",
    )

    with pytest.raises(AttributeError):
        result.state = "rejected"  # type: ignore[misc]
    with pytest.raises(TypeError, match="exact built-in boolean"):
        WorkStartResult(accepted=1, state="active", task_id="task_public")  # type: ignore[arg-type]
    with pytest.raises(PrivateRunDisclosureError):
        WorkCancelResult(
            accepted=False,
            state="rejected",
            reason="leaked deleg_private_1",
        )
    with pytest.raises(PrivateRunDisclosureError):
        WorkCancelResult(
            accepted=False,
            state="rejected",
            reason="leaked prefixdeleg_private_1",
        )


@pytest.mark.asyncio
async def test_start_validates_and_reserves_before_dispatch() -> None:
    context = ConversationContextStore(max_item_chars=37)
    surface, controller, _events = _surface(
        context=context,
        reserve=lambda: (_ for _ in ()).throw(RuntimeError("capacity exhausted")),
    )

    assert surface.max_objective_chars == 37
    with pytest.raises(ValueError, match="configured maximum"):
        await surface.start_work(objective="x" * 38, invocation_id="call_1")
    with pytest.raises(PrivateRunDisclosureError):
        await surface.start_work(
            objective="inspect deleg_private_1",
            invocation_id="call_2",
        )
    with pytest.raises(TypeError, match="exact built-in string"):
        await surface.start_work(
            objective=type("Text", (str,), {})("inspect"),
            invocation_id="call_subclass",
        )
    with pytest.raises(RuntimeError, match="capacity"):
        await surface.start_work(objective="inspect", invocation_id="call_3")
    assert controller.dispatches == []


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", (True, False))
async def test_start_projects_only_authoritative_dispatch_result(accepted: bool) -> None:
    surface, controller, events = _surface(accepted=accepted)

    result = await surface.start_work(
        objective="Inspect the release evidence",
        invocation_id="call_start",
    )

    assert controller.dispatches == [("Inspect the release evidence", "utterance_local_1")]
    assert result == WorkStartResult(
        accepted=accepted,
        state="active" if accepted else "rejected",
        task_id="task_release_check",
        reason=None if accepted else "Hermes declined the request",
    )
    assert events == [
        (
            "task_state",
            (
                {"status": "active", "taskId": "task_release_check"}
                if accepted
                else {
                    "reason": "Hermes declined the request",
                    "status": "rejected",
                    "taskId": "task_release_check",
                }
            ),
        )
    ]


@pytest.mark.asyncio
async def test_duplicate_invocation_coalesces_and_caller_cancel_does_not_orphan() -> None:
    gate = asyncio.Event()
    surface, controller, events = _surface(dispatch_gate=gate)

    first = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="same_call")
    )
    await controller.dispatch_started.wait()
    replay = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="same_call")
    )
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    gate.set()

    assert await replay == WorkStartResult(
        accepted=True,
        state="active",
        task_id="task_release_check",
    )
    assert len(controller.dispatches) == 1
    assert events == [("task_state", {"status": "active", "taskId": "task_release_check"})]


@pytest.mark.asyncio
async def test_distinct_start_invocations_admit_multiple_active_tasks() -> None:
    context = ConversationContextStore(max_active_tasks=3)
    surface, controller, events = _surface(context=context)
    task_ids = iter(("task_first", "task_second"))

    async def dispatch(*, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        controller.dispatches.append((objective, utterance_id))
        task_id = next(task_ids)
        context.record_task_accepted(
            task_id=task_id,
            run_id=f"deleg_private_{task_id}",
            objective=objective,
        )
        return TaskDispatchOutcome(task_id=task_id, accepted=True)

    controller.dispatch = dispatch  # type: ignore[method-assign]

    first = await surface.start_work(objective="First objective", invocation_id="call_first")
    second = await surface.start_work(objective="Second objective", invocation_id="call_second")

    assert first == WorkStartResult(accepted=True, state="active", task_id="task_first")
    assert second == WorkStartResult(accepted=True, state="active", task_id="task_second")
    assert [task.task_id for task in context.snapshot().active_tasks] == [
        "task_first",
        "task_second",
    ]
    assert events == [
        ("task_state", {"status": "active", "taskId": "task_first"}),
        ("task_state", {"status": "active", "taskId": "task_second"}),
    ]


@pytest.mark.asyncio
async def test_distinct_invocation_rejects_matching_active_objective() -> None:
    context = ConversationContextStore(max_active_tasks=3)
    dispatches: list[str] = []

    class DistinctTaskController:
        async def dispatch(
            self,
            *,
            objective: str,
            utterance_id: str,
        ) -> TaskDispatchOutcome:
            del utterance_id
            dispatches.append(objective)
            task_id = f"task_distinct_{len(dispatches)}"
            context.record_task_accepted(
                task_id=task_id,
                run_id=f"deleg_distinct_{len(dispatches)}",
                objective=objective,
            )
            return TaskDispatchOutcome(accepted=True, task_id=task_id)

        async def request_cancel(self, *, task_id: str, reason: str) -> TaskCancelOutcome:
            del reason
            return TaskCancelOutcome(accepted=True, task_id=task_id)

    surface = ConversationWorkControlSurface(
        controller=DistinctTaskController(),  # type: ignore[arg-type]
        context=context,
        observer=lambda _event_type, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "distinct_objective",
    )
    try:
        first = await surface.start_work(
            objective="Inspect   the Release evidence",
            invocation_id="invocation_distinct_objective_1",
        )
        duplicate = await surface.start_work(
            objective="inspect the release EVIDENCE",
            invocation_id="invocation_distinct_objective_2",
        )

        assert first.accepted is True
        assert duplicate == WorkStartResult(
            accepted=False,
            state="rejected",
            reason="matching work is already active",
        )
        assert dispatches == ["Inspect   the Release evidence"]
        assert surface.health == "open"
    finally:
        await surface.close()


@pytest.mark.asyncio
async def test_distinct_start_invocations_dispatch_concurrently_before_ack() -> None:
    context = ConversationContextStore(max_active_tasks=3)
    surface, controller, _events = _surface(context=context)
    release = asyncio.Event()

    async def dispatch(*, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        controller.dispatches.append((objective, utterance_id))
        await release.wait()
        task_id = "task_first" if objective == "First objective" else "task_second"
        context.record_task_accepted(
            task_id=task_id,
            run_id=f"deleg_private_{task_id}",
            objective=objective,
        )
        return TaskDispatchOutcome(task_id=task_id, accepted=True)

    controller.dispatch = dispatch  # type: ignore[method-assign]
    first = asyncio.create_task(
        surface.start_work(objective="First objective", invocation_id="call_first")
    )
    second = asyncio.create_task(
        surface.start_work(objective="Second objective", invocation_id="call_second")
    )

    for _ in range(10):
        if len(controller.dispatches) == 2:
            break
        await asyncio.sleep(0)
    assert [objective for objective, _utterance_id in controller.dispatches] == [
        "First objective",
        "Second objective",
    ]

    release.set()
    results = await asyncio.gather(first, second)
    assert {result.task_id for result in results} == {"task_first", "task_second"}


@pytest.mark.asyncio
async def test_simultaneous_pending_starts_compete_atomically_for_last_slot() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    release = asyncio.Event()
    surface, controller, _events = _surface(context=context, dispatch_gate=release)

    first = asyncio.create_task(
        surface.start_work(objective="First objective", invocation_id="call_first")
    )
    await asyncio.wait_for(controller.dispatch_started.wait(), timeout=1)
    second = asyncio.create_task(
        surface.start_work(objective="Second objective", invocation_id="call_second")
    )
    await asyncio.sleep(0)

    assert len(controller.dispatches) == 1
    rejected = await asyncio.wait_for(second, timeout=1)
    assert rejected == WorkStartResult(
        accepted=False,
        state="rejected",
        reason="background task capacity exhausted",
    )
    assert len(surface._pending_starts) == 1

    release.set()
    accepted = await asyncio.wait_for(first, timeout=1)
    assert accepted.accepted is True
    assert surface._pending_starts == {}
    assert len(controller.dispatches) == 1


@pytest.mark.asyncio
async def test_conflicting_invocation_fails_closed_without_second_side_effect() -> None:
    surface, controller, _events = _surface()
    await surface.start_work(objective="First objective", invocation_id="same_call")

    conflict = await surface.start_work(
        objective="Different objective",
        invocation_id="same_call",
    )
    later = await surface.cancel_active_work(invocation_id="later_call")

    assert conflict.reason == "work control unavailable; restart required"
    assert later.reason == "work control unavailable; restart required"
    assert surface.health == "uncertain"
    assert len(controller.dispatches) == 1
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_cancel_active_rejects_zero_and_requires_identity_for_multiple() -> None:
    context = ConversationContextStore(max_active_tasks=3)
    surface, controller, events = _surface(context=context)

    empty = await surface.cancel_active_work(invocation_id="cancel_empty")
    assert empty == WorkCancelResult(
        accepted=False,
        state="rejected",
        reason="no active work",
    )
    assert controller.cancellations == []
    assert events[-1] == (
        "task_state",
        {"reason": "no active work", "status": "rejected", "taskId": None},
    )

    context.record_task_accepted(
        task_id="task_one",
        run_id="deleg_private_one",
        objective="one",
    )
    context.record_task_accepted(
        task_id="task_two",
        run_id="deleg_private_two",
        objective="two",
    )
    multiple = await surface.cancel_active_work(invocation_id="cancel_multiple")
    assert multiple.reason == "multiple active tasks; specify task_id"
    assert surface.health == "open"
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_exact_cancel_isolated_to_one_of_multiple_active_tasks() -> None:
    context = ConversationContextStore(max_active_tasks=3)
    for number in ("one", "two"):
        context.record_task_accepted(
            task_id=f"task_{number}",
            run_id=f"deleg_private_{number}",
            objective=number,
        )
    surface, controller, events = _surface(context=context)
    controller.cancel_outcome = TaskCancelOutcome(task_id="task_two", accepted=True)

    result = await surface.cancel_work(
        task_id="task_two",
        invocation_id="cancel_second",
    )

    assert result == WorkCancelResult(
        accepted=True,
        state="cancelling",
        task_id="task_two",
    )
    assert controller.cancellations == [("task_two", "user requested task cancellation")]
    assert events == [("task_state", {"status": "cancelling", "taskId": "task_two"})]


@pytest.mark.asyncio
async def test_cancel_active_targets_exact_single_public_task() -> None:
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_release_check",
        run_id="deleg_private_run",
        objective="Inspect evidence",
    )
    surface, controller, events = _surface(context=context)

    result = await surface.cancel_active_work(invocation_id="cancel_one")

    assert result == WorkCancelResult(
        accepted=True,
        state="cancelling",
        task_id="task_release_check",
    )
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]
    assert events == [("task_state", {"status": "cancelling", "taskId": "task_release_check"})]


@pytest.mark.asyncio
async def test_pending_start_cancel_waits_for_ack_and_cancels_exact_acceptance() -> None:
    gate = asyncio.Event()
    surface, controller, events = _surface(dispatch_gate=gate)
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="start_pending")
    )
    await controller.dispatch_started.wait()
    cancel = asyncio.create_task(surface.cancel_active_work(invocation_id="cancel_pending"))
    await asyncio.sleep(0)
    assert controller.cancellations == []

    gate.set()

    assert await start == WorkStartResult(
        accepted=True,
        state="cancelling",
        task_id="task_release_check",
    )
    assert await cancel == WorkCancelResult(
        accepted=True,
        state="cancelling",
        task_id="task_release_check",
    )
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]
    assert events == [("task_state", {"status": "cancelling", "taskId": "task_release_check"})]


@pytest.mark.asyncio
async def test_repeated_stop_joins_one_task_during_pending_active_overlap() -> None:
    dispatch_gate = asyncio.Event()
    cancel_entered = asyncio.Event()
    release_cancel = asyncio.Event()
    surface, controller, _events = _surface(dispatch_gate=dispatch_gate)
    original_cancel = controller.request_cancel

    async def slow_cancel(
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        cancel_entered.set()
        await release_cancel.wait()
        return await original_cancel(task_id, reason=reason)

    controller.request_cancel = slow_cancel  # type: ignore[method-assign]
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="start_overlap")
    )
    await controller.dispatch_started.wait()
    first_stop = asyncio.create_task(
        surface.cancel_active_work(invocation_id="cancel_overlap_first")
    )
    dispatch_gate.set()
    await asyncio.wait_for(cancel_entered.wait(), timeout=1)

    second_stop = asyncio.create_task(
        surface.cancel_active_work(invocation_id="cancel_overlap_second")
    )
    await asyncio.sleep(0)
    assert not second_stop.done()
    release_cancel.set()

    first_result, second_result, start_result = await asyncio.gather(
        first_stop,
        second_stop,
        start,
    )
    assert first_result.accepted is True
    assert second_result == first_result
    assert start_result.state == "cancelling"
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]


@pytest.mark.asyncio
async def test_pending_start_cancel_propagates_authoritative_rejection_without_stop() -> None:
    gate = asyncio.Event()
    surface, controller, events = _surface(accepted=False, dispatch_gate=gate)
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="start_pending")
    )
    await controller.dispatch_started.wait()
    cancel = asyncio.create_task(surface.cancel_active_work(invocation_id="cancel_pending"))
    gate.set()

    assert (await start).accepted is False
    assert (await cancel).accepted is False
    assert controller.cancellations == []
    assert events == [
        (
            "task_state",
            {
                "reason": "Hermes declined the request",
                "status": "rejected",
                "taskId": "task_release_check",
            },
        )
    ]


@pytest.mark.asyncio
async def test_post_ack_projection_failure_rolls_back_without_poisoning_on_success() -> None:
    def fail_projection(
        _kind: str,
        _data: dict[str, str | int | bool | None],
    ) -> None:
        raise RuntimeError("projection failed")

    surface, controller, _events = _surface(observer=fail_projection)

    with pytest.raises(RuntimeError, match="projection failed"):
        await surface.start_work(
            objective="Inspect evidence",
            invocation_id="rollback_call",
        )

    assert controller.cancellations == [("task_release_check", "task state projection failed")]
    assert surface.health == "open"


@pytest.mark.asyncio
async def test_post_ack_projection_and_rollback_failure_preserves_both() -> None:
    def fail_projection(
        _kind: str,
        _data: dict[str, str | int | bool | None],
    ) -> None:
        raise RuntimeError("projection failed")

    surface, controller, _events = _surface(observer=fail_projection)
    controller.cancel_outcome = TaskCancelOutcome(
        task_id="task_release_check",
        accepted=False,
        reason="rollback rejected",
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await surface.start_work(
            objective="Inspect evidence",
            invocation_id="rollback_call",
        )

    assert [str(error) for error in raised.value.exceptions] == [
        "projection failed",
        "task projection rollback was rejected",
    ]
    assert surface.health == "uncertain"


@pytest.mark.asyncio
async def test_completed_replay_retention_evicts_without_poisoning() -> None:
    surface, controller, _events = _surface(max_invocations=1)

    first = await surface.cancel_active_work(invocation_id="first_call")
    replay = await surface.cancel_active_work(invocation_id="first_call")
    second = await surface.cancel_active_work(invocation_id="second_call")

    assert first == replay
    assert second.reason == "no active work"
    assert surface.health == "open"
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_close_is_cancellation_resistant_and_permanently_closes_admission() -> None:
    gate = asyncio.Event()
    surface, controller, _events = _surface(dispatch_gate=gate)
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="start_call")
    )
    await controller.dispatch_started.wait()
    close = asyncio.create_task(surface.close())
    await asyncio.sleep(0)

    assert surface.health == "closed"
    rejected = await surface.cancel_active_work(invocation_id="stale_call")
    assert rejected.reason == "work control unavailable; restart required"
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close

    gate.set()
    await start
    await surface.close()
    assert surface.health == "closed"


@pytest.mark.asyncio
async def test_mutated_private_controller_result_is_rejected_on_reconstruction() -> None:
    surface, controller, _events = _surface(accepted=False)
    object.__setattr__(
        controller.dispatch_outcome,
        "reason",
        "private prefixdeleg_private_12345678",
    )

    with pytest.raises(PrivateRunDisclosureError):
        await surface.start_work(
            objective="Inspect evidence",
            invocation_id="private_result",
        )
    assert surface.health == "uncertain"


@pytest.mark.asyncio
async def test_unreconciled_dispatch_failure_closes_later_work_admission() -> None:
    surface, controller, _events = _surface()
    dispatch_calls = 0

    async def fail_dispatch(*, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        del objective, utterance_id
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise OSError("connection lost after request write")

    controller.dispatch = fail_dispatch  # type: ignore[method-assign]

    with pytest.raises(OSError, match="connection lost"):
        await surface.start_work(
            objective="Inspect evidence",
            invocation_id="ambiguous_dispatch",
        )
    later = await surface.start_work(
        objective="Do not retry",
        invocation_id="later_dispatch",
    )

    assert dispatch_calls == 1
    assert later.reason == "work control unavailable; restart required"
    assert surface.health == "uncertain"


@pytest.mark.asyncio
async def test_cancel_completion_race_is_bounded_and_keeps_surface_open() -> None:
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_release_check",
        run_id="deleg_private_run",
        objective="Inspect evidence",
    )
    surface, controller, events = _surface(context=context)

    async def completed_before_cancel(
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        del task_id, reason
        raise ActiveTaskIdentityError("task is no longer active")

    controller.request_cancel = completed_before_cancel  # type: ignore[method-assign]

    result = await surface.cancel_active_work(invocation_id="cancel_race")

    assert result == WorkCancelResult(
        accepted=False,
        state="rejected",
        task_id="task_release_check",
        reason="task is not active",
    )
    assert surface.health == "open"
    assert events == [
        (
            "task_state",
            {
                "reason": "task is not active",
                "status": "rejected",
                "taskId": "task_release_check",
            },
        )
    ]


@pytest.mark.asyncio
async def test_pending_cancel_completion_race_is_bounded_and_keeps_surface_open() -> None:
    gate = asyncio.Event()
    surface, controller, events = _surface(dispatch_gate=gate)

    async def completed_before_cancel(
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        del task_id, reason
        raise ActiveTaskIdentityError("task is no longer active")

    controller.request_cancel = completed_before_cancel  # type: ignore[method-assign]
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="pending_race_start")
    )
    await controller.dispatch_started.wait()
    cancel = asyncio.create_task(surface.cancel_active_work(invocation_id="pending_race_cancel"))
    await asyncio.sleep(0)
    gate.set()

    start_result = await start
    cancel_result = await cancel

    assert start_result == WorkStartResult(
        accepted=False,
        state="rejected",
        task_id="task_release_check",
        reason="task is not active",
    )
    assert cancel_result == WorkCancelResult(
        accepted=False,
        state="rejected",
        task_id="task_release_check",
        reason="task is not active",
    )
    assert surface.health == "open"
    assert events == [
        (
            "task_state",
            {
                "reason": "task is not active",
                "status": "rejected",
                "taskId": "task_release_check",
            },
        )
    ]


@pytest.mark.asyncio
async def test_run_prefixed_rejection_reason_is_bounded_and_keeps_surface_open() -> None:
    surface, controller, _events = _surface(accepted=False)
    controller.dispatch_outcome = TaskDispatchOutcome(
        task_id="task_release_check",
        accepted=False,
        reason="run_capacity_exhausted",
    )

    result = await surface.start_work(
        objective="Inspect evidence",
        invocation_id="run_reason",
    )

    assert result.reason == "run_capacity_exhausted"
    assert surface.health == "open"


@pytest.mark.asyncio
async def test_cancelled_dispatch_settles_pending_cancel_and_close() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    surface, controller, _events = _surface()

    async def cancelled_dispatch(*, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        del objective, utterance_id
        started.set()
        await release.wait()
        raise asyncio.CancelledError

    controller.dispatch = cancelled_dispatch  # type: ignore[method-assign]
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="cancelled_start")
    )
    await started.wait()
    cancel = asyncio.create_task(surface.cancel_active_work(invocation_id="cancel_cancelled_start"))
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await start
    async with asyncio.timeout(1):
        cancel_result = await cancel
        await surface.close()
    assert cancel_result.reason == "work control unavailable; restart required"


@pytest.mark.asyncio
async def test_close_timeout_is_bounded_and_retryable() -> None:
    gate = asyncio.Event()
    surface, controller, _events = _surface(
        dispatch_gate=gate,
        close_drain_timeout_ms=10,
    )
    start = asyncio.create_task(
        surface.start_work(objective="Inspect evidence", invocation_id="blocked_start")
    )
    await controller.dispatch_started.wait()

    with pytest.raises(TimeoutError):
        await surface.close()

    gate.set()
    await start
    async with asyncio.timeout(1):
        await surface.close()
