import asyncio
from datetime import UTC, datetime, timedelta, tzinfo

import pytest

from hermes_realtime.conversation import (
    ActiveTaskCapacityError,
    ActiveTaskIdentityError,
    ActiveTaskSummary,
    ConversationContextStore,
    ConversationTaskController,
    PrivateRunDisclosureError,
    TaskCancelOutcome,
    TaskDispatchOutcome,
    TaskTerminalOutcome,
)
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
)


class HostileTZ(tzinfo):
    calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta:
        type(self).calls += 1
        raise RuntimeError("hostile timezone invoked")

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        return "hostile"


class TaskSessionStub:
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        raise AssertionError(request)

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        raise AssertionError(request)

    async def next_update(self) -> WorkCompletedEvent:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        return None


class ClosingTaskSession(TaskSessionStub):
    def __init__(self) -> None:
        self.update_waiting = asyncio.Event()

    async def next_update(self) -> WorkCompletedEvent:
        self.update_waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class CancellationResistantCloseSession(TaskSessionStub):
    def __init__(self) -> None:
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.cancellations = 0

    async def close(self) -> None:
        self.close_started.set()
        while not self.allow_close.is_set():
            try:
                await self.allow_close.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
        self.close_finished.set()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_abandon_controller_close() -> None:
    session = CancellationResistantCloseSession()
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=lambda: "unused",
        close_drain_timeout_ms=50,
    )
    caller = asyncio.create_task(controller.close())
    await session.close_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    session.allow_close.set()
    await controller.close()

    assert session.close_finished.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await controller.dispatch(objective="must not dispatch", utterance_id="utterance_001")


@pytest.mark.asyncio
async def test_close_has_finite_drain_for_cancellation_resistant_session() -> None:
    session = CancellationResistantCloseSession()
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=lambda: "unused",
        close_drain_timeout_ms=5,
    )

    await asyncio.wait_for(controller.close(), timeout=0.2)

    assert session.cancellations == 1
    with pytest.raises(RuntimeError, match="closed"):
        await controller.dispatch(objective="closed", utterance_id="utterance_001")
    session.allow_close.set()
    await asyncio.wait_for(session.close_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_close_wakes_every_blocked_terminal_consumer() -> None:
    session = ClosingTaskSession()
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=lambda: "unused",
    )
    consumers = [asyncio.create_task(controller.next_completion()) for _ in range(3)]
    await session.update_waiting.wait()
    await asyncio.sleep(0)

    await controller.close()

    results = await asyncio.wait_for(
        asyncio.gather(*consumers, return_exceptions=True),
        timeout=0.2,
    )
    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("closed" in str(result) for result in results)


@pytest.mark.asyncio
async def test_close_wakes_blocked_terminal_consumer() -> None:
    session = ClosingTaskSession()
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=lambda: "unused",
    )
    consumer = asyncio.create_task(controller.next_completion())
    await session.update_waiting.wait()

    await controller.close()

    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(consumer, timeout=0.2)


@pytest.mark.parametrize(
    ("status", "summary", "reason"),
    (
        ("completed", None, None),
        ("completed", "done", "unexpected"),
        ("failed", "unexpected", "failed"),
        ("failed", None, None),
        ("interrupted", None, None),
    ),
)
def test_terminal_outcome_requires_status_specific_evidence(
    status: str,
    summary: str | None,
    reason: str | None,
) -> None:
    with pytest.raises(ValueError, match="evidence"):
        TaskTerminalOutcome(
            task_id="task_good",
            status=status,
            summary=summary,
            reason=reason,
        )


@pytest.mark.asyncio
async def test_task_control_inputs_reject_private_and_subclass_text_before_transport() -> None:
    class HostileText(str):
        pass

    dispatch_session = AcceptingTaskSession()
    dispatch_controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=dispatch_session,
        session_id="session_001",
        id_factory=iter((
            "hostile",
            "dispatch_hostile",
            "private",
            "dispatch_private",
        )).__next__,
    )
    with pytest.raises(TypeError, match="exact built-in string"):
        await dispatch_controller.dispatch(
            objective=HostileText("hostile"),
            utterance_id="utterance_001",
        )
    with pytest.raises(PrivateRunDisclosureError):
        await dispatch_controller.dispatch(
            objective="safe",
            utterance_id="deleg_private_001",
        )
    assert dispatch_session.requests == []
    await dispatch_controller.close()

    cancel_context = ConversationContextStore()
    cancel_session = AcceptingCancelSession()
    cancel_controller = ConversationTaskController(
        context=cancel_context,
        session=cancel_session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather", "cancel_001")).__next__,
    )
    await cancel_controller.dispatch(
        objective="check weather",
        utterance_id="utterance_001",
    )
    with pytest.raises(PrivateRunDisclosureError):
        await cancel_controller.request_cancel(
            "task_weather",
            reason="deleg_private_001",
        )
    assert cancel_session.cancel_calls == 0
    await cancel_controller.close()


def test_controller_rejects_noncanonical_session_and_operation_limits() -> None:
    with pytest.raises(ValueError, match="session_id"):
        ConversationTaskController(
            context=ConversationContextStore(),
            session=TaskSessionStub(),
            session_id="bad session",
            id_factory=lambda: "unused",
        )
    with pytest.raises(TypeError, match="max_protocol_operations"):
        ConversationTaskController(
            context=ConversationContextStore(),
            session=TaskSessionStub(),
            session_id="session_001",
            id_factory=lambda: "unused",
            max_protocol_operations=True,
        )


def test_interrupted_terminal_status_is_model_safe() -> None:
    assert TaskTerminalOutcome(
        task_id="task_good",
        status="interrupted",
        reason="canceled by user",
    ).status == "interrupted"


def test_task_control_outcomes_enforce_exact_model_visible_boundaries() -> None:
    class HostileString(str):
        pass

    with pytest.raises(TypeError, match="task_id"):
        TaskDispatchOutcome(task_id=HostileString("task_bad"), accepted=True)
    with pytest.raises(TypeError, match="accepted"):
        TaskCancelOutcome(task_id="task_good", accepted=1)  # type: ignore[arg-type]
    with pytest.raises(PrivateRunDisclosureError):
        TaskCancelOutcome(
            task_id="task_good",
            accepted=False,
            reason="failed for deleg_private_001",
        )
    with pytest.raises(PrivateRunDisclosureError):
        TaskTerminalOutcome(
            task_id="task_good",
            status="completed",
            summary="completed deleg_private_001",
        )


class AcceptingTaskSession(TaskSessionStub):
    def __init__(self) -> None:
        self.requests: list[WorkDispatchRequestedEvent] = []

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        self.requests.append(request)
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_001",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_private_001",
            ),
        )


class LateAckDuringCloseSession(TaskSessionStub):
    def __init__(self) -> None:
        self.dispatch_started = asyncio.Event()
        self.release_ack = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        self.dispatch_started.set()
        await self.release_ack.wait()
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_late",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_late_001",
            ),
        )

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()


@pytest.mark.asyncio
async def test_late_ack_during_close_cannot_commit_active_authority() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    session = LateAckDuringCloseSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("late", "dispatch_late")).__next__,
        close_drain_timeout_ms=50,
    )
    dispatch = asyncio.create_task(
        controller.dispatch(objective="late ack", utterance_id="utterance_001")
    )
    await session.dispatch_started.wait()
    close = asyncio.create_task(controller.close())
    await session.close_started.wait()

    session.release_ack.set()
    with pytest.raises(RuntimeError, match="closed before dispatch acknowledgment"):
        await asyncio.wait_for(dispatch, timeout=0.2)

    assert context.snapshot().active_tasks == ()
    assert "task_late" in controller._pending_dispatches
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_probe", "reservation must remain")

    session.release_close.set()
    await asyncio.wait_for(close, timeout=0.2)


class DuplicateRunDispatchSession(TaskSessionStub):
    def __init__(self) -> None:
        self.dispatch_calls = 0

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        self.dispatch_calls += 1
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id=f"evt_ack_{self.dispatch_calls}",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_duplicate_001",
            ),
        )


@pytest.mark.asyncio
async def test_accepted_ack_commit_failure_faults_before_another_transport() -> None:
    context = ConversationContextStore()
    session = DuplicateRunDispatchSession()
    ids = iter((
        "first",
        "dispatch_first",
        "second",
        "dispatch_second",
        "third",
        "dispatch_third",
    ))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    await controller.dispatch(objective="first", utterance_id="utterance_001")
    with pytest.raises(ActiveTaskIdentityError, match="run identity"):
        await controller.dispatch(objective="second", utterance_id="utterance_002")
    with pytest.raises(ActiveTaskIdentityError, match="run identity"):
        await controller.dispatch(objective="third", utterance_id="utterance_003")

    assert session.dispatch_calls == 2
    await controller.close()


class AcceptingCancelSession(TaskSessionStub):
    def __init__(self) -> None:
        self.cancel_calls = 0

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_cancel_setup",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_private_001",
            ),
        )

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_calls += 1
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id="evt_cancel_ack",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            request_event_id=request.event_id,
            scope=CancelScope.TASK,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=True,
                signaled_run_ids=["deleg_private_001"],
            ),
        )


class MutatingDispatchRequestSession(AcceptingCancelSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(request, "task_id", "task_mutated_after_dispatch")
        object.__setattr__(request, "sequence", acknowledgment.sequence)
        return acknowledgment


class MutatingCancelRequestSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(request, "task_id", "task_mutated_after_cancel")
        object.__setattr__(request, "event_id", "evt_mutated_after_cancel")
        object.__setattr__(request, "sequence", acknowledgment.sequence)
        return acknowledgment


@pytest.mark.asyncio
async def test_dispatch_settlement_uses_preawait_request_authority_snapshot() -> None:
    context = ConversationContextStore()
    controller = ConversationTaskController(
        context=context,
        session=MutatingDispatchRequestSession(),
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )

    outcome = await controller.dispatch(
        objective="check weather",
        utterance_id="utterance_001",
    )

    assert outcome == TaskDispatchOutcome(task_id="task_weather", accepted=True)
    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


@pytest.mark.asyncio
async def test_cancel_settlement_uses_preawait_request_authority_snapshot() -> None:
    context = ConversationContextStore()
    controller = ConversationTaskController(
        context=context,
        session=MutatingCancelRequestSession(),
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather", "cancel_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    outcome = await controller.request_cancel(task_id="task_weather")

    assert outcome == TaskCancelOutcome(task_id="task_weather", accepted=True)
    assert "task_weather" in controller._cancel_operations
    await controller.close()


class LateCancelAckDuringCloseSession(AcceptingCancelSession):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_ack = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_started.set()
        await self.release_ack.wait()
        return await super().cancel(request)

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()


@pytest.mark.asyncio
async def test_late_cancel_ack_during_close_cannot_report_acceptance() -> None:
    context = ConversationContextStore()
    session = LateCancelAckDuringCloseSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather", "cancel_weather")).__next__,
        close_drain_timeout_ms=50,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    cancellation = asyncio.create_task(controller.request_cancel("task_weather"))
    await session.cancel_started.wait()
    close = asyncio.create_task(controller.close())
    await session.close_started.wait()

    session.release_ack.set()
    with pytest.raises(RuntimeError, match="closed before cancellation acknowledgment"):
        await asyncio.wait_for(cancellation, timeout=0.2)

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    assert "task_weather" in controller._cancel_operations

    session.release_close.set()
    await asyncio.wait_for(close, timeout=0.2)


class ExtraRunCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(
            acknowledgment.payload,
            "signaled_run_ids",
            ["deleg_private_001", "deleg_unrelated_001"],
        )
        return acknowledgment


@pytest.mark.asyncio
async def test_cancel_ack_with_extra_run_authority_faults_generation() -> None:
    context = ConversationContextStore()
    session = ExtraRunCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001", "other", "dispatch_other"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    with pytest.raises(RuntimeError, match="exact active run authority"):
        await controller.request_cancel("task_weather")
    with pytest.raises(RuntimeError, match="exact active run authority"):
        await controller.dispatch(objective="must fail closed", utterance_id="utterance_002")

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


class FailingCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_calls += 1
        raise RuntimeError("cancel transport failed")


class DelayedCancelSession(AcceptingCancelSession):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_started.set()
        await self.release_cancel.wait()
        return await super().cancel(request)


class MalformedCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(acknowledgment.payload, "accepted", 1)
        return acknowledgment


class MutatedOuterCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(acknowledgment, "type", "control.cancel")
        return acknowledgment


@pytest.mark.asyncio
async def test_mutated_cancel_outer_event_faults_without_acceptance() -> None:
    context = ConversationContextStore()
    controller = ConversationTaskController(
        context=context,
        session=MutatedOuterCancelSession(),
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather", "cancel_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    with pytest.raises(RuntimeError, match="event type"):
        await controller.request_cancel("task_weather")

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    assert "task_weather" in controller._cancel_operations
    await controller.close()


class MalformedRejectedCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(acknowledgment.payload, "accepted", False)
        object.__setattr__(acknowledgment.payload, "reason", "rejected")
        return acknowledgment


class MalformedAcceptedReasonCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(acknowledgment.payload, "reason", "contradictory")
        return acknowledgment


class MalformedAcceptedEmptyCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(acknowledgment.payload, "signaled_run_ids", [])
        return acknowledgment


class MalformedRejectedMissingReasonCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        acknowledgment = await super().cancel(request)
        object.__setattr__(acknowledgment.payload, "accepted", False)
        object.__setattr__(acknowledgment.payload, "signaled_run_ids", [])
        return acknowledgment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_type",
    (
        MalformedRejectedCancelSession,
        MalformedAcceptedReasonCancelSession,
        MalformedAcceptedEmptyCancelSession,
        MalformedRejectedMissingReasonCancelSession,
    ),
)
async def test_cancel_ack_semantic_contradiction_faults_without_misreporting(
    session_type: type[AcceptingCancelSession],
) -> None:
    context = ConversationContextStore()
    session = session_type()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather", "cancel_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    with pytest.raises(RuntimeError, match="cancellation acknowledgment evidence"):
        await controller.request_cancel("task_weather")

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    assert "task_weather" in controller._cancel_operations
    await controller.close()


@pytest.mark.asyncio
async def test_cancel_transport_failure_is_shared_and_leaves_task_active() -> None:
    context = ConversationContextStore()
    session = FailingCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="transport failed"):
            await controller.request_cancel("task_weather")

    assert session.cancel_calls == 1
    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_orphan_cancel_settlement() -> None:
    context = ConversationContextStore()
    session = DelayedCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    caller = asyncio.create_task(controller.request_cancel("task_weather"))
    await session.cancel_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    session.release_cancel.set()
    outcome = await controller.request_cancel("task_weather")

    assert outcome == TaskCancelOutcome(task_id="task_weather", accepted=True)
    assert session.cancel_calls == 1
    assert context.snapshot().active_tasks[0].task_id == "task_weather"
    await controller.close()


@pytest.mark.asyncio
async def test_malformed_cancel_ack_faults_before_integer_truthiness() -> None:
    context = ConversationContextStore()
    session = MalformedCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001", "other", "dispatch_other"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    with pytest.raises(TypeError, match="acceptance"):
        await controller.request_cancel("task_weather")
    with pytest.raises(TypeError, match="acceptance"):
        await controller.dispatch(objective="must fail closed", utterance_id="utterance_002")

    assert context.snapshot().active_tasks[0].task_id == "task_weather"
    await controller.close()


class RejectingCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_calls += 1
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id=f"evt_cancel_rejected_{self.cancel_calls}",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            request_event_id=request.event_id,
            scope=CancelScope.TASK,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=False,
                reason="not signaled",
            ),
        )


@pytest.mark.asyncio
async def test_generated_event_identity_cannot_be_reused_after_rejection() -> None:
    session = RejectingCancelSession()
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather", "cancel_same", "cancel_same")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    assert (await controller.request_cancel("task_weather")).accepted is False

    with pytest.raises(RuntimeError, match="event identity.*reused"):
        await controller.request_cancel("task_weather")

    assert session.cancel_calls == 1
    await controller.close()


@pytest.mark.asyncio
async def test_generation_operation_budget_stops_rejected_cancel_retries() -> None:
    session = RejectingCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001"))
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        max_protocol_operations=2,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    assert (await controller.request_cancel("task_weather")).accepted is False

    with pytest.raises(RuntimeError, match="operation budget exhausted"):
        await controller.request_cancel("task_weather")

    assert session.cancel_calls == 1
    await controller.close()


class WrongRunCancelSession(AcceptingCancelSession):
    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_calls += 1
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id="evt_cancel_wrong_run",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            request_event_id=request.event_id,
            scope=CancelScope.TASK,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=True,
                signaled_run_ids=["deleg_other_001"],
            ),
        )


@pytest.mark.asyncio
async def test_cancel_ack_without_active_run_faults_generation() -> None:
    context = ConversationContextStore()
    session = WrongRunCancelSession()
    ids = iter((
        "weather",
        "dispatch_weather",
        "cancel_001",
        "other",
        "dispatch_other",
    ))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    with pytest.raises(RuntimeError, match="exact active run authority"):
        await controller.request_cancel("task_weather")
    with pytest.raises(RuntimeError, match="exact active run authority"):
        await controller.dispatch(objective="must fail closed", utterance_id="utterance_002")

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


@pytest.mark.asyncio
async def test_accepted_cancellation_does_not_remove_active_task() -> None:
    context = ConversationContextStore()
    session = AcceptingCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 34, tzinfo=UTC),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    outcome = await controller.request_cancel("task_weather", reason="no longer needed")

    assert outcome == TaskCancelOutcome(task_id="task_weather", accepted=True)
    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    assert "deleg_private_001" not in repr(outcome)
    await controller.close()


@pytest.mark.asyncio
async def test_duplicate_cancellation_shares_one_authoritative_operation() -> None:
    context = ConversationContextStore()
    session = AcceptingCancelSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 34, tzinfo=UTC),
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")

    first, second = await asyncio.gather(
        controller.request_cancel("task_weather"),
        controller.request_cancel("task_weather"),
    )

    assert first == second
    assert session.cancel_calls == 1
    assert context.snapshot().active_tasks[0].task_id == "task_weather"
    await controller.close()


class UpdatingTaskSession(TaskSessionStub):
    def __init__(self, update: WorkCompletedEvent) -> None:
        self.update = update
        self.release_update = asyncio.Event()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_update",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id=self.update.run_id,
            ),
        )

    async def next_update(self) -> WorkCompletedEvent:
        await self.release_update.wait()
        self.release_update.clear()
        return self.update


class ExplodingStr(str):
    calls = 0

    def __str__(self) -> str:
        type(self).calls += 1
        raise RuntimeError("subclass method invoked")


@pytest.mark.asyncio
async def test_terminal_text_subclass_is_rejected_before_subclass_methods() -> None:
    timestamp = datetime(2026, 7, 20, 23, 35, tzinfo=UTC)
    update = WorkCompletedEvent(
        type="work.completed",
        event_id="evt_completed_hostile",
        session_id="session_001",
        sequence=4,
        timestamp=timestamp,
        task_id="task_hostile",
        run_id="deleg_private_001",
        payload=WorkCompletedPayload(
            status=WorkTerminalStatus.COMPLETED,
            summary="initial",
        ),
    )
    object.__setattr__(update.payload, "summary", ExplodingStr("hostile"))
    session = UpdatingTaskSession(update)
    controller = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=iter(("hostile", "dispatch_hostile")).__next__,
        clock=lambda: timestamp,
    )
    await controller.dispatch(objective="hostile", utterance_id="utterance_001")
    session.release_update.set()
    ExplodingStr.calls = 0

    with pytest.raises(TypeError, match="summary"):
        await controller.next_completion()

    assert ExplodingStr.calls == 0
    await controller.close()


class PreAckTerminalSession(TaskSessionStub):
    def __init__(self) -> None:
        self.request: WorkDispatchRequestedEvent | None = None
        self.dispatch_started = asyncio.Event()
        self.release_ack = asyncio.Event()
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()
        self.update_returned = asyncio.Event()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        self.request = request
        self.dispatch_started.set()
        await self.release_ack.wait()
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_preack",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_preack_001",
            ),
        )

    async def next_update(self) -> WorkCompletedEvent:
        update = await self.updates.get()
        self.update_returned.set()
        return update


class RejectedPreAckTerminalSession(PreAckTerminalSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment.payload, "accepted", False)
        object.__setattr__(acknowledgment.payload, "run_id", None)
        object.__setattr__(acknowledgment.payload, "reason", "rejected")
        return acknowledgment


class CancelCompletionRaceSession(AcceptingCancelSession):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel_ack = asyncio.Event()
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        self.cancel_calls += 1
        self.cancel_started.set()
        await self.release_cancel_ack.wait()
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id="evt_cancel_race_ack",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            request_event_id=request.event_id,
            scope=CancelScope.TASK,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=True,
                signaled_run_ids=["deleg_private_001"],
            ),
        )

    async def next_update(self) -> WorkCompletedEvent:
        return await self.updates.get()


class QueuedUpdateSession(AcceptingCancelSession):
    def __init__(self) -> None:
        super().__init__()
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()

    async def next_update(self) -> WorkCompletedEvent:
        return await self.updates.get()


def terminal_event(
    *,
    task_id: str = "task_weather",
    run_id: str = "deleg_private_001",
    session_id: str = "session_001",
    event_id: str = "evt_completed_001",
    sequence: int = 4,
) -> WorkCompletedEvent:
    return WorkCompletedEvent(
        type="work.completed",
        event_id=event_id,
        session_id=session_id,
        sequence=sequence,
        timestamp=datetime(2026, 7, 20, 23, 35, tzinfo=UTC),
        task_id=task_id,
        run_id=run_id,
        payload=WorkCompletedPayload(
            status=WorkTerminalStatus.COMPLETED,
            summary="Finished",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("update", "message"),
    (
        (terminal_event(task_id="task_other"), "active task authority"),
        (terminal_event(run_id="deleg_other_001"), "active task authority"),
        (terminal_event(session_id="session_previous"), "does not belong"),
    ),
)
async def test_mismatched_or_stale_terminal_fails_without_mutation(
    update: WorkCompletedEvent,
    message: str,
) -> None:
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    await session.updates.put(update)

    with pytest.raises(RuntimeError, match=message):
        await controller.next_completion()

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


class MultiRunQueuedSession(TaskSessionStub):
    def __init__(self) -> None:
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id=f"evt_ack_{request.task_id}",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id=f"deleg_{request.task_id.removeprefix('task_')}",
            ),
        )

    async def next_update(self) -> WorkCompletedEvent:
        return await self.updates.get()


@pytest.mark.asyncio
async def test_reconnect_generation_rejects_old_terminal_even_when_task_run_recur() -> None:
    old_context = ConversationContextStore()
    old_session = QueuedUpdateSession()
    old_controller = ConversationTaskController(
        context=old_context,
        session=old_session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await old_controller.dispatch(
        objective="old weather",
        utterance_id="utterance_old",
    )
    await old_controller.close()

    new_context = ConversationContextStore()
    new_session = QueuedUpdateSession()
    new_controller = ConversationTaskController(
        context=new_context,
        session=new_session,
        session_id="session_002",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await new_controller.dispatch(
        objective="new weather",
        utterance_id="utterance_new",
    )
    await new_session.updates.put(terminal_event(session_id="session_001"))

    with pytest.raises(RuntimeError, match="does not belong"):
        await new_controller.next_completion()

    assert new_context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "new weather"),
    )
    await new_controller.close()


@pytest.mark.asyncio
async def test_terminal_output_backpressure_faults_without_retiring_undelivered_task() -> None:
    context = ConversationContextStore()
    session = MultiRunQueuedSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("first", "dispatch_first", "second", "dispatch_second")).__next__,
        max_terminal_updates=1,
    )
    await controller.dispatch(objective="first", utterance_id="utterance_001")
    await controller.dispatch(objective="second", utterance_id="utterance_002")
    await session.updates.put(
        terminal_event(task_id="task_first", run_id="deleg_first", event_id="evt_first")
    )
    await session.updates.put(
        terminal_event(
            task_id="task_second",
            run_id="deleg_second",
            event_id="evt_second",
            sequence=5,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert (await controller.next_completion()).task_id == "task_first"
    with pytest.raises(RuntimeError, match="capacity"):
        await controller.next_completion()
    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_second", "second"),
    )
    await controller.close()


@pytest.mark.asyncio
async def test_mutated_terminal_outer_event_faults_without_retiring_authority() -> None:
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    update = terminal_event()
    object.__setattr__(update, "type", "control.cancel")
    await session.updates.put(update)

    with pytest.raises(RuntimeError, match="event type"):
        await controller.next_completion()

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


@pytest.mark.asyncio
async def test_hostile_terminal_timezone_is_rejected_without_execution() -> None:
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    update = terminal_event()
    object.__setattr__(
        update,
        "timestamp",
        datetime(2026, 7, 21, 3, 18, tzinfo=HostileTZ()),
    )
    HostileTZ.calls = 0
    await session.updates.put(update)

    with pytest.raises(TypeError, match="timezone"):
        await controller.next_completion()

    assert HostileTZ.calls == 0
    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("update", "message"),
    (
        (terminal_event(sequence=0, event_id="evt_stale_terminal"), "sequence"),
        (terminal_event(sequence=2, event_id="evt_ack_cancel_setup"), "identity"),
    ),
)
async def test_stale_or_reused_inbound_terminal_cannot_retire_active_authority(
    update: WorkCompletedEvent,
    message: str,
) -> None:
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    await session.updates.put(update)

    with pytest.raises(RuntimeError, match=message):
        await controller.next_completion()

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check weather"),
    )
    await controller.close()


@pytest.mark.asyncio
async def test_duplicate_terminal_is_delivered_once_then_faults_generation() -> None:
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    update = terminal_event()
    await session.updates.put(update)
    await session.updates.put(update.model_copy(update={"event_id": "evt_duplicate"}))

    assert (await controller.next_completion()).task_id == "task_weather"
    with pytest.raises(RuntimeError, match="sequence was reused"):
        await controller.next_completion()
    assert context.snapshot().active_tasks == ()
    await controller.close()


@pytest.mark.asyncio
async def test_completion_clears_cancel_idempotency_without_waiting_for_ack() -> None:
    context = ConversationContextStore()
    timestamp = datetime(2026, 7, 20, 23, 35, tzinfo=UTC)
    session = CancelCompletionRaceSession()
    ids = iter(("weather", "dispatch_weather", "cancel_001"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: timestamp,
    )
    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    cancel_caller = asyncio.create_task(controller.request_cancel("task_weather"))
    await session.cancel_started.wait()
    await session.updates.put(
        WorkCompletedEvent(
            type="work.completed",
            event_id="evt_completed_cancel_race",
            session_id="session_001",
            sequence=4,
            timestamp=timestamp,
            task_id="task_weather",
            run_id="deleg_private_001",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.COMPLETED,
                summary="Completed while cancellation was in flight",
            ),
        )
    )

    assert (await controller.next_completion()).task_id == "task_weather"
    assert controller._cancel_operations == {}
    session.release_cancel_ack.set()
    assert (await cancel_caller).accepted is True
    await controller.close()


@pytest.mark.asyncio
async def test_terminal_before_rejected_ack_faults_generation() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    session = RejectedPreAckTerminalSession()
    ids = iter(("preack", "dispatch_preack", "other", "dispatch_other"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
    )
    dispatch = asyncio.create_task(
        controller.dispatch(objective="preack work", utterance_id="utterance_001")
    )
    await session.dispatch_started.wait()
    await session.updates.put(
        terminal_event(task_id="task_preack", run_id="deleg_private_001")
    )
    await session.update_returned.wait()
    session.release_ack.set()

    with pytest.raises(RuntimeError, match="terminal evidence.*rejected"):
        await dispatch
    with pytest.raises(RuntimeError, match="terminal evidence.*rejected"):
        await controller.dispatch(objective="must fail closed", utterance_id="utterance_002")

    assert context.snapshot().active_tasks == ()
    assert "task_preack" in controller._pending_dispatches
    assert "task_preack" in controller._preack_terminals
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_probe", "contradictory authority stays retained")
    await controller.close()


@pytest.mark.asyncio
async def test_preack_terminal_must_follow_its_dispatch_ack_sequence() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    session = PreAckTerminalSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("preack", "dispatch_preack")).__next__,
    )
    dispatch = asyncio.create_task(
        controller.dispatch(objective="preack work", utterance_id="utterance_001")
    )
    await session.dispatch_started.wait()
    await session.updates.put(
        terminal_event(
            task_id="task_preack",
            run_id="deleg_preack_001",
            event_id="evt_temporally_stale",
            sequence=0,
        )
    )
    await session.update_returned.wait()
    session.release_ack.set()

    with pytest.raises(RuntimeError, match="dispatch acknowledgment sequence"):
        await dispatch

    assert context.snapshot().active_tasks == ()
    assert "task_preack" in controller._pending_dispatches
    assert "task_preack" in controller._preack_terminals
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_probe", "authority retained")
    await controller.close()


@pytest.mark.asyncio
async def test_terminal_before_ack_is_retained_and_settled_exactly_once() -> None:
    context = ConversationContextStore()
    session = PreAckTerminalSession()
    ids = iter(("preack", "dispatch_preack"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 35, tzinfo=UTC),
    )
    await controller._lock.acquire()
    controller.start()
    caller = asyncio.create_task(
        controller.dispatch(objective="race terminal", utterance_id="utterance_001")
    )
    controller._lock.release()
    await session.dispatch_started.wait()
    assert session.request is not None
    await session.updates.put(
        WorkCompletedEvent(
            type="work.completed",
            event_id="evt_completed_preack",
            session_id="session_001",
            sequence=2,
            timestamp=session.request.timestamp,
            task_id=session.request.task_id,
            run_id="deleg_preack_001",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.COMPLETED,
                summary="Finished before local ack commit",
            ),
        )
    )
    await session.update_returned.wait()
    async with controller._lock:
        assert len(controller._preack_terminals) == 1
    session.release_ack.set()

    assert (await caller).accepted is True
    update = await controller.next_completion()

    assert update.task_id == "task_preack"
    assert context.snapshot().active_tasks == ()
    await controller.close()


@pytest.mark.asyncio
async def test_matching_terminal_update_removes_task_without_exposing_run_id() -> None:
    context = ConversationContextStore()
    timestamp = datetime(2026, 7, 20, 23, 35, tzinfo=UTC)
    session = UpdatingTaskSession(
        WorkCompletedEvent(
            type="work.completed",
            event_id="evt_completed_001",
            session_id="session_001",
            sequence=4,
            timestamp=timestamp,
            task_id="task_weather",
            run_id="deleg_private_001",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.COMPLETED,
                summary="Forecast retrieved",
            ),
        )
    )
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
        clock=lambda: timestamp,
    )

    await controller.dispatch(objective="check weather", utterance_id="utterance_001")
    session.release_update.set()
    update = await controller.next_completion()

    assert update == TaskTerminalOutcome(
        task_id="task_weather",
        status="completed",
        summary="Forecast retrieved",
    )
    assert context.snapshot().active_tasks == ()
    assert "deleg_private_001" not in repr(update)
    await controller.close()


class RejectingTaskSession(TaskSessionStub):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_rejected",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=False,
                reason="policy rejected dispatch",
            ),
        )


@pytest.mark.asyncio
async def test_rejected_dispatch_releases_pending_capacity_without_projection() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    ids = iter(("rejected", "dispatch_rejected"))
    controller = ConversationTaskController(
        context=context,
        session=RejectingTaskSession(),
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 31, tzinfo=UTC),
    )

    outcome = await controller.dispatch(objective="reject me", utterance_id="utterance_001")

    assert outcome == TaskDispatchOutcome(
        task_id="task_rejected",
        accepted=False,
        reason="policy rejected dispatch",
    )
    assert context.snapshot().active_tasks == ()
    context.prepare_task("task_after_rejection", "capacity released")


class NonFollowingDispatchAckSession(AcceptingTaskSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment, "sequence", request.sequence)
        return acknowledgment


@pytest.mark.asyncio
async def test_dispatch_ack_sequence_must_follow_request_without_releasing_reservation() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    controller = ConversationTaskController(
        context=context,
        session=NonFollowingDispatchAckSession(),
        session_id="session_001",
        id_factory=iter(("stale", "dispatch_stale")).__next__,
    )

    with pytest.raises(RuntimeError, match="dispatch acknowledgment sequence"):
        await controller.dispatch(objective="stale ack", utterance_id="utterance_001")

    assert context.snapshot().active_tasks == ()
    assert "task_stale" in controller._pending_dispatches
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_probe", "reservation retained")
    await controller.close()


class MutatedOuterDispatchSession(AcceptingTaskSession):
    def __init__(self, field: str, value: object) -> None:
        super().__init__()
        self._field = field
        self._value = value

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment, self._field, self._value)
        return acknowledgment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("type", "work.dispatch.requested"),
        ("protocol_version", "9.9"),
        ("event_id", "invalid event id"),
        ("sequence", True),
        ("timestamp", datetime(2026, 7, 21, 2, 20)),
    ),
)
async def test_mutated_dispatch_outer_event_faults_without_releasing_reservation(
    field: str,
    value: object,
) -> None:
    context = ConversationContextStore(max_active_tasks=1)
    controller = ConversationTaskController(
        context=context,
        session=MutatedOuterDispatchSession(field, value),
        session_id="session_001",
        id_factory=iter(("outer", "dispatch_outer")).__next__,
    )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        await controller.dispatch(objective="outer mutation", utterance_id="utterance_001")

    assert context.snapshot().active_tasks == ()
    assert "task_outer" in controller._pending_dispatches
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_probe", "reservation retained")
    await controller.close()


class FailingTaskSession(TaskSessionStub):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        del request
        raise RuntimeError("transport failed")


class MalformedAcceptedSession(TaskSessionStub):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_malformed",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_private_001",
            ),
        )
        object.__setattr__(acknowledgment.payload, "accepted", 1)
        return acknowledgment


class MalformedRejectedDispatchSession(AcceptingTaskSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment.payload, "accepted", False)
        object.__setattr__(acknowledgment.payload, "reason", "rejected")
        return acknowledgment


class MalformedAcceptedReasonDispatchSession(AcceptingTaskSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment.payload, "reason", "contradictory")
        return acknowledgment


class MalformedAcceptedMissingRunDispatchSession(AcceptingTaskSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment.payload, "run_id", None)
        return acknowledgment


class MalformedRejectedMissingReasonDispatchSession(AcceptingTaskSession):
    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        acknowledgment = await super().dispatch(request)
        object.__setattr__(acknowledgment.payload, "accepted", False)
        object.__setattr__(acknowledgment.payload, "run_id", None)
        return acknowledgment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_type",
    (
        MalformedRejectedDispatchSession,
        MalformedRejectedMissingReasonDispatchSession,
        MalformedAcceptedReasonDispatchSession,
        MalformedAcceptedMissingRunDispatchSession,
    ),
)
async def test_dispatch_semantic_contradiction_faults_without_releasing_reservation(
    session_type: type[AcceptingTaskSession],
) -> None:
    context = ConversationContextStore(max_active_tasks=1)
    controller = ConversationTaskController(
        context=context,
        session=session_type(),
        session_id="session_001",
        id_factory=iter(("malformed", "dispatch_malformed")).__next__,
    )

    with pytest.raises(RuntimeError, match="dispatch acknowledgment evidence"):
        await controller.dispatch(objective="malformed", utterance_id="utterance_001")

    assert context.snapshot().active_tasks == ()
    assert "task_malformed" in controller._pending_dispatches
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_other", "reservation retained")
    await controller.close()


@pytest.mark.asyncio
async def test_malformed_ack_fails_before_truthiness_or_context_projection() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    controller = ConversationTaskController(
        context=context,
        session=MalformedAcceptedSession(),
        session_id="session_001",
        id_factory=iter(("malformed", "dispatch_malformed")).__next__,
        clock=lambda: datetime(2026, 7, 20, 23, 32, tzinfo=UTC),
    )

    with pytest.raises(TypeError, match="acceptance"):
        await controller.dispatch(objective="malformed", utterance_id="utterance_001")

    assert context.snapshot().active_tasks == ()
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_other", "reservation retained")
    await controller.close()


@pytest.mark.asyncio
async def test_ambiguous_transport_failure_retains_capacity_and_faults_controller() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    ids = iter(("failed", "dispatch_failed"))
    controller = ConversationTaskController(
        context=context,
        session=FailingTaskSession(),
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 32, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        await controller.dispatch(objective="fail", utterance_id="utterance_001")

    assert context.snapshot().active_tasks == ()
    with pytest.raises(ActiveTaskCapacityError, match="capacity"):
        context.prepare_task("task_after_failure", "capacity remains reserved")
    with pytest.raises(RuntimeError, match="transport failed"):
        await controller.dispatch(objective="cannot continue", utterance_id="utterance_002")
    await controller.close()


class HoldingAcceptingTaskSession(TaskSessionStub):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        self.started.set()
        await self.release.wait()
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack_holding",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_holding_001",
            ),
        )


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_orphan_dispatch_settlement() -> None:
    context = ConversationContextStore()
    session = HoldingAcceptingTaskSession()
    ids = iter(("holding", "dispatch_holding"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 33, tzinfo=UTC),
    )
    caller = asyncio.create_task(
        controller.dispatch(objective="finish after cancellation", utterance_id="utterance_001")
    )
    await session.started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    session.release.set()
    await controller.close()

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_holding", "finish after cancellation"),
    )


@pytest.mark.asyncio
async def test_dispatch_projects_task_only_after_authoritative_acceptance() -> None:
    context = ConversationContextStore()
    session = AcceptingTaskSession()
    ids = iter(("task_001", "dispatch_001"))
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 7, 20, 23, 30, tzinfo=UTC),
    )

    outcome = await controller.dispatch(
        objective="Check tomorrow's weather",
        utterance_id="utterance_001",
    )

    assert outcome == TaskDispatchOutcome(task_id="task_task_001", accepted=True)
    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_task_001", "Check tomorrow's weather"),
    )
    assert session.requests[0].task_id == "task_task_001"
    assert "deleg_private_001" not in repr(outcome)
    assert "deleg_private_001" not in repr(context.snapshot())
