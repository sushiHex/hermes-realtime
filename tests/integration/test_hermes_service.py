import asyncio
from datetime import UTC, datetime
from threading import Event, Thread

import pytest

from hermes_realtime.integration import (
    HermesDispatchCommand,
    HermesDispatchRejected,
    HermesIntegrationService,
    SessionBinding,
    SessionBindings,
)
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelEvent,
    WorkDispatchRequestedEvent,
)


def test_session_binding_is_idempotent_for_reconnect() -> None:
    bindings = SessionBindings()

    first = bindings.bind("participant_001", "session_001")
    second = bindings.bind("participant_001", "session_001")

    assert first == second
    assert bindings.session_for("participant_001") == "session_001"


def test_session_binding_rejects_participant_rebind() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")

    with pytest.raises(ValueError, match="already bound"):
        bindings.bind("participant_001", "session_002")


def test_session_binding_rejects_session_hijack() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")

    with pytest.raises(ValueError, match="already bound"):
        bindings.bind("participant_002", "session_001")


def test_released_session_binding_can_be_reassigned() -> None:
    bindings = SessionBindings()
    original = bindings.bind("participant_001", "session_001")

    released = bindings.release(original)

    assert released == original
    assert bindings.session_for("participant_001") is None
    assert bindings.bind("participant_002", "session_001").session_id == "session_001"


def test_binding_lookup_and_release_normalize_participant_id() -> None:
    bindings = SessionBindings()
    original = bindings.bind("  participant_001  ", "session_001")

    assert bindings.session_for("  participant_001  ") == "session_001"
    assert bindings.release(original) == original
    assert bindings.session_for("participant_001") is None


def test_stale_binding_release_cannot_remove_rebound_generation() -> None:
    bindings = SessionBindings()
    original = bindings.bind("participant_001", "session_001")
    assert bindings.release(original) == original
    replacement = bindings.bind("participant_001", "session_001")

    assert bindings.release(original) is None
    assert bindings.binding_for("participant_001") is replacement


def test_binding_admission_serializes_release_and_rebind() -> None:
    bindings = SessionBindings()
    original = bindings.bind("participant_001", "session_001")
    release_started = Event()
    release_completed = Event()

    def release_and_rebind() -> None:
        release_started.set()
        bindings.release(original)
        bindings.bind("participant_002", "session_001")
        release_completed.set()

    with bindings.admission(original) as active:
        assert active is True
        releaser = Thread(target=release_and_rebind)
        releaser.start()
        assert release_started.wait(timeout=1)
        assert release_completed.wait(timeout=0.05) is False

    releaser.join(timeout=1)
    assert release_completed.is_set()
    assert bindings.session_for("participant_002") == "session_001"


class BlockingDispatcher:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.commands: list[HermesDispatchCommand] = []

    async def dispatch(self, command: HermesDispatchCommand) -> str:
        self.commands.append(command)
        self.started.set()
        await self.release.wait()
        return "run_001"


class UniqueBlockingDispatcher(BlockingDispatcher):
    async def dispatch(self, command: HermesDispatchCommand) -> str:
        self.commands.append(command)
        self.started.set()
        await self.release.wait()
        return f"run_{command.task_id}"


@pytest.mark.asyncio
async def test_dispatch_acknowledges_only_after_hermes_returns_run_id() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = BlockingDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )
    request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Inspect the current Hermes plugin API"},
    )

    pending = asyncio.create_task(service.dispatch("participant_001", request))
    await dispatcher.started.wait()
    assert not pending.done()

    dispatcher.release.set()
    acknowledgment = await pending

    assert dispatcher.commands == [
        HermesDispatchCommand(
            session_id="session_001",
            task_id="task_001",
            objective="Inspect the current Hermes plugin API",
            durability="ephemeral",
        )
    ]
    assert acknowledgment.event_id == "evt_ack_001"
    assert acknowledgment.session_id == "session_001"
    assert acknowledgment.sequence == 8
    assert acknowledgment.task_id == "task_001"
    assert acknowledgment.payload.accepted is True
    assert acknowledgment.payload.run_id == "run_001"


@pytest.mark.asyncio
async def test_unbound_participant_cannot_reach_hermes_dispatcher() -> None:
    dispatcher = BlockingDispatcher()
    dispatcher.release.set()
    service = HermesIntegrationService(
        bindings=SessionBindings(),
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )
    request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Inspect the current Hermes plugin API"},
    )

    with pytest.raises(PermissionError, match="not bound"):
        await service.dispatch("participant_001", request)

    assert dispatcher.commands == []


class RejectingDispatcher:
    async def dispatch(self, command: HermesDispatchCommand) -> str:
        del command
        raise HermesDispatchRejected("dispatcher capacity exhausted")


@pytest.mark.asyncio
async def test_hermes_rejection_returns_rejected_acknowledgment() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=RejectingDispatcher(),
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )
    request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Inspect the current Hermes plugin API"},
    )

    acknowledgment = await service.dispatch("participant_001", request)

    assert acknowledgment.payload.accepted is False
    assert acknowledgment.payload.run_id is None
    assert acknowledgment.payload.reason == "dispatcher capacity exhausted"


@pytest.mark.asyncio
async def test_duplicate_task_dispatches_only_once() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = BlockingDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )
    request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Inspect the current Hermes plugin API"},
    )

    first = asyncio.create_task(service.dispatch("participant_001", request))
    second = asyncio.create_task(service.dispatch("participant_001", request))
    await dispatcher.started.wait()
    dispatcher.release.set()

    first_ack, second_ack = await asyncio.gather(first, second)

    assert len(dispatcher.commands) == 1
    assert first_ack == second_ack


class ImmediateDispatcher:
    def __init__(self) -> None:
        self.commands: list[HermesDispatchCommand] = []

    async def dispatch(self, command: HermesDispatchCommand) -> str:
        self.commands.append(command)
        return f"run_{len(self.commands):03d}"


@pytest.mark.asyncio
async def test_completed_dispatch_idempotency_cache_is_bounded() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        max_retained_dispatches=1,
    )

    def request(task_id: str, sequence: int) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_{task_id}",
            session_id="session_001",
            sequence=sequence,
            timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
            task_id=task_id,
            utterance_id=f"utterance_{task_id}",
            payload={"objective": f"Run {task_id}"},
        )

    await service.dispatch("participant_001", request("task_001", 1))
    await service.dispatch("participant_001", request("task_002", 3))
    repeated = await service.dispatch("participant_001", request("task_001", 5))

    assert len(dispatcher.commands) == 3
    assert repeated.payload.run_id == "run_003"


@pytest.mark.asyncio
async def test_reassigned_session_does_not_share_prior_generation_cache() -> None:
    bindings = SessionBindings()
    original = bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )

    def request() -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id="evt_request_001",
            session_id="session_001",
            sequence=1,
            timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
            task_id="task_001",
            utterance_id="utterance_001",
            payload={"objective": "Generation-specific work"},
        )

    first = await service.dispatch("participant_001", request())
    bindings.release(original)
    bindings.bind("participant_002", "session_001")

    second = await service.dispatch("participant_002", request())

    assert first.payload.run_id == "run_001"
    assert second.payload.run_id == "run_002"
    assert len(dispatcher.commands) == 2


class ObservingSessionBindings(SessionBindings):
    def __init__(self) -> None:
        super().__init__()
        self.binding_observed = asyncio.Event()

    def binding_for(self, participant_id: str) -> SessionBinding | None:
        binding = super().binding_for(participant_id)
        self.binding_observed.set()
        return binding


@pytest.mark.asyncio
async def test_released_binding_cannot_pass_delayed_dispatch_admission() -> None:
    bindings = ObservingSessionBindings()
    original = bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )
    request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=1,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Must not cross binding generations"},
    )

    await service._dispatch_lock.acquire()
    delayed = asyncio.create_task(service.dispatch("participant_001", request))
    await bindings.binding_observed.wait()
    bindings.release(original)
    bindings.bind("participant_002", "session_001")
    service._dispatch_lock.release()

    with pytest.raises(PermissionError, match="released"):
        await delayed
    assert dispatcher.commands == []


@pytest.mark.asyncio
async def test_reused_task_id_rejects_conflicting_objective() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )

    def request(objective: str, sequence: int) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_{sequence}",
            session_id="session_001",
            sequence=sequence,
            timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
            task_id="task_001",
            utterance_id=f"utterance_{sequence}",
            payload={"objective": objective},
        )

    await service.dispatch("participant_001", request("First objective", 1))

    with pytest.raises(ValueError, match="conflicting"):
        await service.dispatch("participant_001", request("Different objective", 3))

    assert len(dispatcher.commands) == 1


@pytest.mark.asyncio
async def test_concurrent_acknowledgments_have_unique_session_sequences() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    event_ids = iter(("evt_ack_001", "evt_ack_002"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )

    def request(task_id: str) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_{task_id}",
            session_id="session_001",
            sequence=7,
            timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
            task_id=task_id,
            utterance_id=f"utterance_{task_id}",
            payload={"objective": f"Run {task_id}"},
        )

    acknowledgments = await asyncio.gather(
        service.dispatch("participant_001", request("task_001")),
        service.dispatch("participant_001", request("task_002")),
    )

    assert sorted(ack.sequence for ack in acknowledgments) == [8, 9]


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_shared_hermes_dispatch() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = BlockingDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
    )
    request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Continue after transport cancellation"},
    )

    disconnected_caller = asyncio.create_task(
        service.dispatch("participant_001", request)
    )
    await dispatcher.started.wait()
    disconnected_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnected_caller

    dispatcher.release.set()
    acknowledgment = await service.dispatch("participant_001", request)

    assert len(dispatcher.commands) == 1
    assert acknowledgment.payload.run_id == "run_001"


@pytest.mark.asyncio
async def test_dispatch_admission_rejects_work_above_active_limit() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = BlockingDispatcher()
    event_ids = iter(("evt_rejected_001", "evt_ack_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        max_active_dispatches=1,
    )

    def request(task_id: str, sequence: int) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_{task_id}",
            session_id="session_001",
            sequence=sequence,
            timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
            task_id=task_id,
            utterance_id=f"utterance_{task_id}",
            payload={"objective": f"Run {task_id}"},
        )

    accepted = asyncio.create_task(
        service.dispatch("participant_001", request("task_001", 1))
    )
    await dispatcher.started.wait()

    rejected = await service.dispatch("participant_001", request("task_002", 2))

    assert rejected.payload.accepted is False
    assert rejected.payload.reason == "Hermes dispatch capacity exhausted"
    assert len(dispatcher.commands) == 1

    dispatcher.release.set()
    assert (await accepted).payload.run_id == "run_001"


@pytest.mark.asyncio
async def test_completed_retention_is_trimmed_without_later_admission() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = UniqueBlockingDispatcher()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        event_id_factory=lambda: "evt_ack_001",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        max_retained_dispatches=1,
        max_active_dispatches=3,
    )

    def request(index: int) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_request_{index}",
            session_id="session_001",
            sequence=index,
            timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
            task_id=f"task_{index}",
            utterance_id=f"utterance_{index}",
            payload={"objective": f"Run task {index}"},
        )

    pending = [
        asyncio.create_task(service.dispatch("participant_001", request(index)))
        for index in range(1, 4)
    ]
    while len(dispatcher.commands) < 3:
        await asyncio.sleep(0)
    dispatcher.release.set()
    await asyncio.gather(*pending)
    await asyncio.sleep(0)

    assert len(service._dispatch_tasks) == 1
    assert len(service._dispatch_fingerprints) == 1


class RecordingCanceller:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    async def cancel(self, run_id: str) -> bool:
        self.run_ids.append(run_id)
        return True


def cancel_request(
    *,
    scope: CancelScope = CancelScope.TASK,
    task_id: str | None = "task_001",
) -> ControlCancelEvent:
    return ControlCancelEvent(
        type="control.cancel",
        event_id="evt_cancel_001",
        session_id="session_001",
        sequence=9,
        timestamp=datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
        task_id=task_id,
        payload={"scope": scope, "reason": "User interrupted the request"},
    )


@pytest.mark.asyncio
async def test_task_cancel_waits_for_run_id_then_signals_exact_delegation() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = BlockingDispatcher()
    canceller = RecordingCanceller()
    event_ids = iter(("evt_ack_001", "evt_cancel_ack_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        canceller=canceller,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 4, 2, tzinfo=UTC),
    )
    dispatch_request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Run until interrupted"},
    )

    dispatch = asyncio.create_task(
        service.dispatch("participant_001", dispatch_request)
    )
    await dispatcher.started.wait()
    cancellation = asyncio.create_task(
        service.cancel("participant_001", cancel_request())
    )
    await asyncio.sleep(0)
    assert not cancellation.done()

    dispatcher.release.set()
    acknowledgment, cancel_ack = await asyncio.gather(dispatch, cancellation)

    assert acknowledgment.payload.run_id == "run_001"
    assert canceller.run_ids == ["run_001"]
    assert cancel_ack.payload.accepted is True
    assert cancel_ack.payload.signaled_run_ids == ["run_001"]
    assert cancel_ack.request_event_id == "evt_cancel_001"
    assert cancel_ack.sequence > acknowledgment.sequence


@pytest.mark.asyncio
async def test_rebound_generation_cannot_cancel_prior_generation_run() -> None:
    bindings = SessionBindings()
    original = bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    canceller = RecordingCanceller()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        canceller=canceller,
        event_id_factory=lambda: "evt_001",
        clock=lambda: datetime(2026, 7, 19, 4, 2, tzinfo=UTC),
    )
    dispatch_request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Old generation work"},
    )
    await service.dispatch("participant_001", dispatch_request)
    assert bindings.release(original) is original
    bindings.bind("participant_001", "session_001")

    cancel_ack = await service.cancel("participant_001", cancel_request())

    assert cancel_ack.payload.accepted is False
    assert canceller.run_ids == []


@pytest.mark.asyncio
async def test_duplicate_cancellation_event_signals_hermes_only_once() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    canceller = RecordingCanceller()
    event_ids = iter(("evt_dispatch_ack", "evt_cancel_ack"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        canceller=canceller,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 4, 2, tzinfo=UTC),
    )
    dispatch_request = WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Cancelable work"},
    )
    await service.dispatch("participant_001", dispatch_request)

    first, duplicate = await asyncio.gather(
        service.cancel("participant_001", cancel_request()),
        service.cancel("participant_001", cancel_request()),
    )

    assert first == duplicate
    assert canceller.run_ids == ["run_001"]


@pytest.mark.asyncio
async def test_active_run_remains_cancellable_after_retry_cache_pruning() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    dispatcher = ImmediateDispatcher()
    canceller = RecordingCanceller()
    event_index = 0

    def event_id() -> str:
        nonlocal event_index
        event_index += 1
        return f"evt_{event_index:03d}"

    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=dispatcher,
        canceller=canceller,
        event_id_factory=event_id,
        clock=lambda: datetime(2026, 7, 19, 4, 2, tzinfo=UTC),
        max_retained_dispatches=1,
    )

    def request(task_id: str, sequence: int) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_request_{task_id}",
            session_id="session_001",
            sequence=sequence,
            timestamp=datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
            task_id=task_id,
            utterance_id=f"utterance_{task_id}",
            payload={"objective": f"Run {task_id}"},
        )

    await service.dispatch("participant_001", request("task_001", 1))
    await service.dispatch("participant_001", request("task_002", 2))
    await asyncio.sleep(0)
    cancellation = cancel_request(task_id="task_001")

    acknowledgment = await service.cancel("participant_001", cancellation)

    assert acknowledgment.payload.signaled_run_ids == ["run_001"]
    assert canceller.run_ids == ["run_001"]

    await service.mark_terminal("run_001")

    assert "run_001" not in service._run_bindings


@pytest.mark.asyncio
async def test_terminal_evidence_releases_active_dispatch_capacity() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    event_index = 0

    def event_id() -> str:
        nonlocal event_index
        event_index += 1
        return f"evt_{event_index:03d}"

    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=event_id,
        clock=lambda: datetime(2026, 7, 19, 4, 2, tzinfo=UTC),
        max_active_dispatches=1,
    )

    def request(task_id: str, sequence: int) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            event_id=f"evt_request_{task_id}",
            session_id="session_001",
            sequence=sequence,
            timestamp=datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
            task_id=task_id,
            utterance_id=f"utterance_{task_id}",
            payload={"objective": f"Run {task_id}"},
        )

    first = await service.dispatch("participant_001", request("task_001", 1))
    blocked = await service.dispatch("participant_001", request("task_002", 2))
    assert first.payload.run_id == "run_001"
    assert blocked.payload.accepted is False

    await service.mark_terminal("run_001")
    admitted = await service.dispatch("participant_001", request("task_002", 2))

    assert admitted.payload.run_id == "run_002"
