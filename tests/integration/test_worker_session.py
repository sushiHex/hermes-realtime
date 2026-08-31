import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any, cast

import pytest

from hermes_realtime.conversation import (
    ActiveTaskCapacityError,
    ActiveTaskSummary,
    ConversationContextStore,
    ConversationTaskController,
    TaskCancelOutcome,
    TaskDispatchOutcome,
    TaskTerminalOutcome,
)
from hermes_realtime.integration import (
    BridgeProtocolError,
    EventSequencer,
    HermesCompletionRouter,
    HermesDispatchCommand,
    HermesIntegrationService,
    LocalHermesBridgeClient,
    LocalHermesBridgeServer,
    RealtimeHermesSession,
    SessionBindings,
)
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    ControlCancelPayload,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkDispatchRequestedPayload,
    WorkTerminalStatus,
)


class HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise RuntimeError("hostile string equality invoked")

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise RuntimeError("hostile string inequality invoked")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise RuntimeError("hostile string hashing invoked")


class HostileInt(int):
    calls = 0

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise RuntimeError("hostile integer comparison invoked")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise RuntimeError("hostile integer comparison invoked")


class HostileTZ(tzinfo):
    calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta:
        type(self).calls += 1
        raise RuntimeError("hostile timezone invoked")

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        return "hostile"


class ImmediateDispatcher:
    async def dispatch(self, command: HermesDispatchCommand) -> str:
        return f"deleg_{command.task_id}"


class RecordingCanceller:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    async def cancel(self, run_id: str) -> bool:
        self.run_ids.append(run_id)
        return True


class HoldingClient:
    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release = asyncio.Event()

    async def receive(self) -> object:
        await self.release.wait()
        raise BridgeProtocolError("holding client released")

    async def send(
        self,
        event: object,
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None:
        del event, admission_guard
        self.send_started.set()
        await self.release.wait()


class CancellationResistantReceiveClient:
    def __init__(self) -> None:
        self.receive_started = asyncio.Event()
        self.receive_cancelled = asyncio.Event()
        self.release_receive = asyncio.Event()

    async def receive(self) -> object:
        self.receive_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.receive_cancelled.set()
            await self.release_receive.wait()
            raise
        raise AssertionError("receive unexpectedly resumed")

    async def send(
        self,
        event: object,
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None:
        del event, admission_guard
        raise AssertionError("unexpected send")


@pytest.mark.asyncio
async def test_worker_close_is_owned_and_wakes_consumer_before_resistant_drain() -> None:
    client = CancellationResistantReceiveClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
        close_drain_timeout_ms=10,
    )
    worker.start()
    consumer = asyncio.create_task(worker.next_update())
    await client.receive_started.wait()
    close_caller = asyncio.create_task(worker.close())
    await client.receive_cancelled.wait()

    close_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_caller
    with pytest.raises(BridgeProtocolError, match="closed"):
        await asyncio.wait_for(consumer, timeout=0.2)

    second_close = asyncio.create_task(worker.close())
    client.release_receive.set()
    await asyncio.wait_for(second_close, timeout=0.2)
    assert worker._pending_dispatches == {}
    assert worker._pending_cancellations == {}


class LateTerminalCancellationResistantClient:
    def __init__(self) -> None:
        self.receive_started = asyncio.Event()
        self.receive_cancelled = asyncio.Event()
        self.release_terminal = asyncio.Event()
        self.terminal_returned = asyncio.Event()

    async def receive(self) -> object:
        self.receive_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.receive_cancelled.set()
            await self.release_terminal.wait()
            self.terminal_returned.set()
            return WorkCompletedEvent(
                type="work.completed",
                event_id="evt_late_terminal",
                session_id="session_001",
                sequence=1,
                timestamp=datetime(2026, 7, 21, 2, 30, tzinfo=UTC),
                task_id="task_late",
                run_id="deleg_late",
                payload=WorkCompletedPayload(
                    status=WorkTerminalStatus.COMPLETED,
                    summary="late terminal",
                ),
            )
        raise AssertionError("receive unexpectedly resumed")

    async def send(
        self,
        event: object,
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None:
        del event, admission_guard
        raise AssertionError("unexpected send")


@pytest.mark.asyncio
async def test_worker_drops_terminal_returned_by_detached_receiver_after_close() -> None:
    client = LateTerminalCancellationResistantClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
        max_pending_updates=2,
        close_drain_timeout_ms=10,
    )
    worker.start()
    await client.receive_started.wait()
    await worker.close()

    client.release_terminal.set()
    await client.terminal_returned.wait()
    await asyncio.sleep(0)

    for _ in range(2):
        with pytest.raises(BridgeProtocolError, match="closed"):
            await worker.next_update()
    assert worker._updates.qsize() == 1


class AcknowledgingBlockedSendClient:
    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.release_late_ack = asyncio.Event()
        self.late_ack_queued = asyncio.Event()
        self.ack_queued = asyncio.Event()
        self.incoming: asyncio.Queue[object] = asyncio.Queue()
        self.late_tasks: set[asyncio.Task[None]] = set()

    async def receive(self) -> object:
        return await self.incoming.get()

    async def send(
        self,
        event: object,
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None:
        if admission_guard is not None:
            admission_guard()
        assert isinstance(event, WorkDispatchRequestedEvent)
        self.send_started.set()
        try:
            await self.release_send.wait()
        except asyncio.CancelledError:
            late = asyncio.create_task(self._publish_late_ack(event))
            self.late_tasks.add(late)
            late.add_done_callback(self.late_tasks.discard)
            raise
        await self._publish_ack(event)

    async def _publish_late_ack(self, event: WorkDispatchRequestedEvent) -> None:
        await self.release_late_ack.wait()
        await self._publish_ack(event)
        self.late_ack_queued.set()

    async def _publish_ack(self, event: WorkDispatchRequestedEvent) -> None:
        await self.incoming.put(
            WorkDispatchAcknowledgedEvent(
                type="work.dispatch.acknowledged",
                event_id=f"ack_{event.event_id}",
                session_id=event.session_id,
                sequence=event.sequence + 1,
                timestamp=event.timestamp,
                task_id=event.task_id,
                payload=WorkDispatchAcknowledgedPayload(
                    accepted=True,
                    run_id=f"deleg_{event.task_id}",
                ),
            )
        )
        self.ack_queued.set()


@pytest.mark.asyncio
async def test_caller_cancellation_during_send_preserves_ack_correlation() -> None:
    client = AcknowledgingBlockedSendClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
    )
    caller = asyncio.create_task(worker.dispatch(dispatch_request()))
    await client.send_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    client.release_late_ack.set()
    client.release_send.set()
    await client.ack_queued.wait()

    second = dispatch_request().model_copy(
        update={
            "event_id": "evt_request_002",
            "sequence": 3,
            "task_id": "task_002",
        }
    )
    acknowledgment = await asyncio.wait_for(worker.dispatch(second), timeout=1)

    assert acknowledgment.task_id == "task_002"
    assert worker._terminal_error is None
    assert worker._pending_dispatches == {}
    await worker.close()


@pytest.mark.asyncio
async def test_close_retains_ambiguous_transmitted_dispatch_authority() -> None:
    client = AcknowledgingBlockedSendClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
        close_drain_timeout_ms=10,
    )
    context = ConversationContextStore(max_active_tasks=1)
    controller = ConversationTaskController(
        context=context,
        session=worker,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
        close_drain_timeout_ms=20,
    )
    dispatch = asyncio.create_task(
        controller.dispatch(objective="check weather", utterance_id="utterance_001")
    )
    await client.send_started.wait()

    await controller.close()
    result = (await asyncio.gather(dispatch, return_exceptions=True))[0]

    assert isinstance(result, BaseException)
    assert "task_weather" in controller._pending_dispatches
    with pytest.raises(ActiveTaskCapacityError):
        context.prepare_task("task_probe", "must remain capacity blocked")

    client.release_late_ack.set()
    client.release_send.set()
    await client.late_ack_queued.wait()


def dispatch_request() -> WorkDispatchRequestedEvent:
    return WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=1,
        timestamp=datetime(2026, 7, 19, 6, 0, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload=WorkDispatchRequestedPayload(
            objective="Perform exact background work"
        ),
    )


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    (
        ("max_pending_updates", True, TypeError),
        ("max_pending_updates", 4097, ValueError),
        ("max_pending_operations", False, TypeError),
        ("max_pending_operations", 1025, ValueError),
        ("close_drain_timeout_ms", True, TypeError),
        ("close_drain_timeout_ms", 60_001, ValueError),
    ),
)
def test_worker_session_rejects_noncanonical_or_unbounded_limits(
    keyword: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        RealtimeHermesSession(
            client=cast(LocalHermesBridgeClient, object()),
            session_id="session_001",
            **{keyword: cast(int, value)},
        )


@pytest.mark.asyncio
async def test_worker_session_bounds_pending_operations_before_transport() -> None:
    client = HoldingClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
        max_pending_operations=1,
    )
    first = asyncio.create_task(worker.dispatch(dispatch_request()))
    await asyncio.wait_for(client.send_started.wait(), timeout=1)
    second_request = dispatch_request().model_copy(
        update={"event_id": "evt_request_002", "task_id": "task_002"}
    )

    with pytest.raises(BridgeProtocolError, match="capacity"):
        await worker.dispatch(second_request)
    assert len(worker._pending_dispatches) == 1
    assert len(worker._pending_cancellations) == 0

    client.release.set()
    with pytest.raises(BridgeProtocolError, match="holding client released"):
        await first
    await worker.close()


@pytest.mark.asyncio
async def test_task_controller_runs_over_real_local_bridge_session() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    canceller = RecordingCanceller()
    event_ids = iter(("evt_ack_001", "evt_cancel_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        canceller=canceller,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 6, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 6, 2, tzinfo=UTC),
    )
    context = ConversationContextStore()

    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client, RealtimeHermesSession(
        client=client,
        session_id="session_001",
    ) as worker:
        controller = ConversationTaskController(
            context=context,
            session=worker,
            session_id="session_001",
            id_factory=iter(("weather", "dispatch_weather", "cancel_weather")).__next__,
            clock=lambda: datetime(2026, 7, 19, 6, 3, tzinfo=UTC),
        )
        dispatch = await controller.dispatch(
            objective="check weather",
            utterance_id="utterance_001",
        )
        cancellation = await controller.request_cancel(
            "task_weather",
            reason="no longer needed",
        )
        assert context.snapshot().active_tasks == (
            ActiveTaskSummary("task_weather", "check weather"),
        )

        await server.complete(
            "deleg_task_weather",
            status="completed",
            summary="Weather task finished.",
        )
        terminal = await asyncio.wait_for(controller.next_completion(), timeout=1)

        assert dispatch == TaskDispatchOutcome(task_id="task_weather", accepted=True)
        assert cancellation == TaskCancelOutcome(task_id="task_weather", accepted=True)
        assert terminal == TaskTerminalOutcome(
            task_id="task_weather",
            status="completed",
            summary="Weather task finished.",
        )
        assert context.snapshot().active_tasks == ()
        assert canceller.run_ids == ["deleg_task_weather"]
        assert "deleg_" not in repr((dispatch, cancellation, terminal))
        await controller.close()


@pytest.mark.asyncio
async def test_worker_session_receives_proactive_terminal_update() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 6, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 6, 2, tzinfo=UTC),
    )

    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client, RealtimeHermesSession(
        client=client,
        session_id="session_001",
    ) as worker:
        acknowledgment = await worker.dispatch(dispatch_request())
        await server.complete(
            "deleg_task_001",
            status="completed",
            summary="The background work finished.",
        )
        update = await asyncio.wait_for(worker.next_update(), timeout=1)

        assert acknowledgment.payload.run_id == "deleg_task_001"
        assert isinstance(update, WorkCompletedEvent)
        assert update.run_id == acknowledgment.payload.run_id


@pytest.mark.asyncio
async def test_worker_session_propagates_exact_task_cancellation() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    canceller = RecordingCanceller()
    event_ids = iter(("evt_ack_001", "evt_cancel_ack_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        canceller=canceller,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 6, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: "evt_unused",
        clock=lambda: datetime(2026, 7, 19, 6, 2, tzinfo=UTC),
    )

    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client, RealtimeHermesSession(
        client=client,
        session_id="session_001",
    ) as worker:
        await worker.dispatch(dispatch_request())
        cancellation = ControlCancelEvent(
            type="control.cancel",
            event_id="evt_cancel_001",
            session_id="session_001",
            sequence=3,
            timestamp=datetime(2026, 7, 19, 6, 3, tzinfo=UTC),
            task_id="task_001",
            payload=ControlCancelPayload(
                scope=CancelScope.TASK,
                reason="user interruption",
            ),
        )

        acknowledgment = await worker.cancel(cancellation)

        assert acknowledgment.payload.accepted is True
        assert acknowledgment.payload.signaled_run_ids == ["deleg_task_001"]
        assert canceller.run_ids == ["deleg_task_001"]


@pytest.mark.asyncio
async def test_worker_session_rejects_cross_session_request_before_transport() -> None:
    class FailingClient:
        async def receive(self) -> object:
            raise AssertionError("unexpected receive")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del admission_guard
            raise AssertionError(f"unexpected transport call: {event}")

    worker = RealtimeHermesSession(
        client=FailingClient(),
        session_id="session_other",
    )

    with pytest.raises(PermissionError, match="session"):
        await worker.dispatch(dispatch_request())


@pytest.mark.asyncio
async def test_worker_session_rejects_work_after_receiver_failure() -> None:
    class BrokenClient:
        def __init__(self) -> None:
            self.send_calls = 0

        async def receive(self) -> object:
            raise BridgeProtocolError("synthetic receive failure")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            if admission_guard is not None:
                admission_guard()
            del event
            self.send_calls += 1

    client = BrokenClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
    )
    worker.start()
    await asyncio.sleep(0)

    with pytest.raises(BridgeProtocolError, match="synthetic receive failure"):
        await asyncio.wait_for(worker.dispatch(dispatch_request()), timeout=0.1)

    assert client.send_calls == 0


@pytest.mark.asyncio
async def test_receiver_failure_wins_over_dispatch_already_waiting_for_lock() -> None:
    receive_started = asyncio.Event()
    release_failure = asyncio.Event()

    class RacingClient:
        def __init__(self) -> None:
            self.send_calls = 0

        async def receive(self) -> object:
            receive_started.set()
            await release_failure.wait()
            raise BridgeProtocolError("synthetic racing receive failure")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            if admission_guard is not None:
                admission_guard()
            del event
            self.send_calls += 1

    client = RacingClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
    )
    await worker._lock.acquire()
    worker.start()
    await receive_started.wait()
    dispatch_task = asyncio.create_task(worker.dispatch(dispatch_request()))
    await asyncio.sleep(0)
    release_failure.set()
    await asyncio.sleep(0)
    worker._lock.release()

    with pytest.raises(BridgeProtocolError, match="synthetic racing receive failure"):
        await dispatch_task
    assert client.send_calls == 0
    await worker.close()


@pytest.mark.asyncio
async def test_receiver_failure_rejects_send_waiting_on_transport_lock() -> None:
    release_failure = asyncio.Event()
    transport_lock = asyncio.Lock()

    class TransportQueuedClient:
        def __init__(self) -> None:
            self.send_calls = 0

        async def receive(self) -> object:
            await release_failure.wait()
            raise BridgeProtocolError("synthetic transport-queued failure")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del event
            async with transport_lock:
                if admission_guard is not None:
                    admission_guard()
                self.send_calls += 1

    client = TransportQueuedClient()
    worker = RealtimeHermesSession(
        client=client,
        session_id="session_001",
    )
    await transport_lock.acquire()
    worker.start()
    dispatch_task = asyncio.create_task(worker.dispatch(dispatch_request()))
    await asyncio.sleep(0)
    release_failure.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if worker._terminal_error is not None:
            break
    transport_lock.release()

    with pytest.raises(BridgeProtocolError, match="transport-queued failure"):
        await dispatch_task
    assert client.send_calls == 0
    await worker.close()


@pytest.mark.asyncio
async def test_close_wakes_blocked_terminal_update_waiter() -> None:
    class IdleClient:
        async def receive(self) -> object:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del event, admission_guard

    worker = RealtimeHermesSession(
        client=IdleClient(),
        session_id="session_001",
    )
    worker.start()
    waiter = asyncio.create_task(worker.next_update())
    await asyncio.sleep(0)

    await worker.close()

    with pytest.raises(BridgeProtocolError, match="closed"):
        await asyncio.wait_for(waiter, timeout=0.1)


@pytest.mark.asyncio
async def test_worker_session_fails_closed_when_terminal_buffer_is_full() -> None:
    events = [
        WorkCompletedEvent(
            type="work.completed",
            event_id=f"evt_terminal_{sequence}",
            session_id="session_001",
            sequence=sequence,
            timestamp=datetime(2026, 7, 19, 6, sequence, tzinfo=UTC),
            task_id=f"task_{sequence}",
            run_id=f"deleg_task_{sequence}",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.COMPLETED,
                summary="done",
            ),
        )
        for sequence in (1, 2)
    ]

    class FloodClient:
        async def receive(self) -> object:
            if events:
                return events.pop(0)
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            if admission_guard is not None:
                admission_guard()
            del event

    worker = RealtimeHermesSession(
        client=FloodClient(),
        session_id="session_001",
        max_pending_updates=1,
    )
    worker.start()
    for _ in range(10):
        await asyncio.sleep(0)
        if worker._terminal_error is not None:
            break

    with pytest.raises(BridgeProtocolError, match="buffer"):
        await worker.next_update()
    await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ("session_id", "sequence", "timestamp", "task_id", "request_event_id"),
)
async def test_worker_rejects_hostile_mutated_fields_before_operations(
    field: str,
) -> None:
    if field == "request_event_id":
        event: object = ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id="evt_hostile_cancel_ack",
            session_id="session_001",
            sequence=1,
            timestamp=datetime(2026, 7, 21, 3, 5, tzinfo=UTC),
            request_event_id="evt_cancel_request",
            scope=CancelScope.TASK,
            task_id="task_hostile",
            payload=ControlCancelAcknowledgedPayload(
                accepted=True,
                signaled_run_ids=["deleg_hostile"],
            ),
        )
        hostile: object = HostileStr("evt_cancel_request")
    elif field == "task_id":
        event = WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_hostile_dispatch_ack",
            session_id="session_001",
            sequence=1,
            timestamp=datetime(2026, 7, 21, 3, 5, tzinfo=UTC),
            task_id="task_hostile",
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_hostile",
            ),
        )
        hostile = HostileStr("task_hostile")
    else:
        event = WorkCompletedEvent(
            type="work.completed",
            event_id="evt_hostile_terminal",
            session_id="session_001",
            sequence=1,
            timestamp=datetime(2026, 7, 21, 3, 5, tzinfo=UTC),
            task_id="task_hostile",
            run_id="deleg_hostile",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.COMPLETED,
                summary="done",
            ),
        )
        if field == "sequence":
            hostile = HostileInt(1)
        elif field == "timestamp":
            hostile = datetime(2026, 7, 21, 3, 5, tzinfo=HostileTZ())
        else:
            hostile = HostileStr("hostile")
    object.__setattr__(event, field, hostile)

    class HostileClient:
        def __init__(self) -> None:
            self.returned = False

        async def receive(self) -> object:
            if self.returned:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            self.returned = True
            return event

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del event, admission_guard
            raise AssertionError("unexpected send")

    HostileStr.calls = 0
    HostileInt.calls = 0
    HostileTZ.calls = 0
    worker = RealtimeHermesSession(
        client=HostileClient(),
        session_id="session_001",
    )
    pending: asyncio.Future[Any] | None = None
    if field == "task_id":
        pending = asyncio.get_running_loop().create_future()
        worker._pending_dispatches["task_hostile"] = pending
    elif field == "request_event_id":
        pending = asyncio.get_running_loop().create_future()
        worker._pending_cancellations["evt_cancel_request"] = pending
    worker.start()

    with pytest.raises(TypeError, match="exact"):
        await asyncio.wait_for(worker.next_update(), timeout=0.2)

    assert HostileStr.calls == 0
    assert HostileInt.calls == 0
    assert HostileTZ.calls == 0
    if pending is not None:
        pending.exception()
    await worker.close()


@pytest.mark.asyncio
async def test_worker_uses_validated_correlation_snapshot_after_lock_wait() -> None:
    acknowledgment = WorkDispatchAcknowledgedEvent(
        type="work.dispatch.acknowledged",
        event_id="evt_snapshot_ack",
        session_id="session_001",
        sequence=1,
        timestamp=datetime(2026, 7, 21, 3, 8, tzinfo=UTC),
        task_id="task_snapshot",
        payload=WorkDispatchAcknowledgedPayload(
            accepted=True,
            run_id="deleg_snapshot",
        ),
    )

    class SnapshotMutationClient:
        def __init__(self) -> None:
            self.returned = asyncio.Event()

        async def receive(self) -> object:
            if self.returned.is_set():
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            self.returned.set()
            return acknowledgment

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del event, admission_guard
            raise AssertionError("unexpected send")

    client = SnapshotMutationClient()
    worker = RealtimeHermesSession(client=client, session_id="session_001")
    pending = asyncio.get_running_loop().create_future()
    worker._pending_dispatches["task_snapshot"] = pending
    await worker._lock.acquire()
    worker.start()
    await client.returned.wait()
    await asyncio.sleep(0)
    object.__setattr__(acknowledgment, "task_id", HostileStr("task_snapshot"))
    HostileStr.calls = 0
    worker._lock.release()

    result = await asyncio.wait_for(pending, timeout=0.2)

    assert result.task_id == "task_snapshot"
    assert HostileStr.calls == 0
    await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "dispatch_session",
        "dispatch_task",
        "cancel_session",
        "cancel_event",
        "cancel_task",
    ),
)
async def test_worker_rejects_hostile_outbound_fields_before_operations(
    case: str,
) -> None:
    class NoTransportClient:
        async def receive(self) -> object:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del event, admission_guard
            raise AssertionError("transport must not be reached")

    worker = RealtimeHermesSession(
        client=NoTransportClient(),
        session_id="session_001",
    )
    if case.startswith("dispatch"):
        request: object = dispatch_request()
        field = "session_id" if case == "dispatch_session" else "task_id"
    else:
        request = ControlCancelEvent(
            type="control.cancel",
            event_id="evt_cancel_hostile",
            session_id="session_001",
            sequence=2,
            timestamp=datetime(2026, 7, 21, 3, 10, tzinfo=UTC),
            task_id="task_001",
            payload=ControlCancelPayload(
                scope=CancelScope.TASK,
                reason="stop",
            ),
        )
        field = {
            "cancel_session": "session_id",
            "cancel_event": "event_id",
            "cancel_task": "task_id",
        }[case]
    object.__setattr__(request, field, HostileStr("hostile"))
    HostileStr.calls = 0

    with pytest.raises(TypeError, match="exact"):
        if isinstance(request, WorkDispatchRequestedEvent):
            await worker.dispatch(request)
        else:
            await worker.cancel(cast(ControlCancelEvent, request))

    assert HostileStr.calls == 0
    await worker.close()


def test_worker_rejects_hostile_session_authority_without_execution() -> None:
    class NoTransportClient:
        async def receive(self) -> object:
            raise AssertionError("unreachable")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            del event, admission_guard
            raise AssertionError("unreachable")

    HostileStr.calls = 0

    with pytest.raises(TypeError, match="session_id.*exact"):
        RealtimeHermesSession(
            client=NoTransportClient(),
            session_id=HostileStr("session_001"),
        )

    assert HostileStr.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ("dispatch", "cancel"))
async def test_failed_send_cleans_original_correlation_after_transport_mutation(
    operation_name: str,
) -> None:
    class MutatingFailureClient:
        def __init__(self) -> None:
            self.send_started = asyncio.Event()
            self.release = asyncio.Event()
            self.retained: object | None = None

        async def receive(self) -> object:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(
            self,
            event: object,
            *,
            admission_guard: Callable[[], None] | None = None,
        ) -> None:
            if admission_guard is not None:
                admission_guard()
            self.retained = event
            self.send_started.set()
            await self.release.wait()
            raise BridgeProtocolError("send failed")

    client = MutatingFailureClient()
    worker = RealtimeHermesSession(client=client, session_id="session_001")
    operation: asyncio.Task[Any]
    if operation_name == "dispatch":
        operation = asyncio.create_task(worker.dispatch(dispatch_request()))
        original_key = "task_001"
        field = "task_id"
    else:
        request = ControlCancelEvent(
            type="control.cancel",
            event_id="evt_cancel_mutation",
            session_id="session_001",
            sequence=2,
            timestamp=datetime(2026, 7, 21, 3, 20, tzinfo=UTC),
            task_id="task_001",
            payload=ControlCancelPayload(scope=CancelScope.TASK, reason="stop"),
        )
        operation = asyncio.create_task(worker.cancel(request))
        original_key = "evt_cancel_mutation"
        field = "event_id"
    await client.send_started.wait()
    assert client.retained is not None
    object.__setattr__(client.retained, field, "mutated_after_send")
    client.release.set()

    with pytest.raises(BridgeProtocolError, match="send failed"):
        await operation

    if operation_name == "dispatch":
        assert original_key not in worker._pending_dispatches
    else:
        assert original_key not in worker._pending_cancellations
    await worker.close()
