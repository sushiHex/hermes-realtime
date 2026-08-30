import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from hermes_realtime.integration import (
    BridgeAuthenticationError,
    BridgeProtocolError,
    EventSequencer,
    HermesCompletionRouter,
    HermesDispatchCommand,
    HermesIntegrationService,
    HermesPluginRuntime,
    HermesRunCompletion,
    LocalHermesBridgeClient,
    LocalHermesBridgeServer,
    SessionBindings,
)
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelEvent,
    ProtocolEvent,
    WorkCompletedEvent,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchRequestedEvent,
)


class ImmediateDispatcher:
    async def dispatch(self, command: HermesDispatchCommand) -> str:
        return f"deleg_{command.task_id}"


class RecordingCanceller:
    async def cancel(self, run_id: str) -> bool:
        del run_id
        return True


class BlockingCancelSendServer(LocalHermesBridgeServer):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.cancel_send_started = asyncio.Event()
        self.release_cancel_send = asyncio.Event()

    async def _send_bound(self, route: object, event: object) -> bool:
        if isinstance(event, ControlCancelAcknowledgedEvent):
            self.cancel_send_started.set()
            await self.release_cancel_send.wait()
        return await super()._send_bound(route, event)  # type: ignore[arg-type]


class HookContext:
    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}
        self.interrupt_calls: list[str] = []

    def dispatch_tool(self, name: str, args: dict[str, object]) -> str:
        assert name == "delegate_task"
        raise AssertionError(args)

    def dispatch_reserved_delegation(
        self, args: dict[str, object], delegation_id: str
    ) -> str:
        return json.dumps(
            {
                "status": "dispatched",
                "delegation_id": delegation_id,
            }
        )

    def register_hook(self, name: str, callback: object) -> None:
        self.hooks[name] = callback

    def interrupt_delegation(self, delegation_id: str) -> bool:
        self.interrupt_calls.append(delegation_id)
        return True


class ImmediateCompletionHookContext(HookContext):
    def __init__(self) -> None:
        super().__init__()
        self.before_completion: Callable[[], None] | None = None

    def dispatch_tool(self, name: str, args: dict[str, object]) -> str:
        assert name == "delegate_task"
        raise AssertionError(args)

    def dispatch_reserved_delegation(
        self, args: dict[str, object], delegation_id: str
    ) -> str:
        if self.before_completion is not None:
            self.before_completion()
        callback = self.hooks["subagent_stop"]
        assert callable(callback)
        callback(
            delegation_id=delegation_id,
            child_status="completed",
            child_summary="Completed before dispatch returned.",
        )
        return json.dumps({"status": "dispatched", "delegation_id": delegation_id})


def request() -> WorkDispatchRequestedEvent:
    return WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id="evt_request_001",
        session_id="session_001",
        sequence=4,
        timestamp=datetime(2026, 7, 19, 5, 0, tzinfo=UTC),
        task_id="task_001",
        utterance_id="utterance_001",
        payload={"objective": "Compare the two deployment options"},
    )


def bridge_components() -> tuple[HermesIntegrationService, HermesCompletionRouter]:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_completed_001"))
    return (
        HermesIntegrationService(
            bindings=bindings,
            dispatcher=ImmediateDispatcher(),
            event_id_factory=lambda: next(event_ids),
            clock=lambda: datetime(2026, 7, 19, 5, 1, tzinfo=UTC),
            sequencer=sequencer,
        ),
        HermesCompletionRouter(
            sequencer=sequencer,
            event_id_factory=lambda: next(event_ids),
            clock=lambda: datetime(2026, 7, 19, 5, 2, tzinfo=UTC),
        ),
    )


@pytest.mark.asyncio
async def test_loopback_bridge_streams_acknowledgment_and_completion() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        ) as client:
        await client.send(request())
        acknowledgment = await client.receive()

        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)
        assert acknowledgment.payload.run_id == "deleg_task_001"

        await server.complete(
            "deleg_task_001",
            status="completed",
            summary="Option B is safer and canary deployment is recommended.",
        )
        completion = await client.receive()

        assert isinstance(completion, WorkCompletedEvent)
        assert completion.task_id == "task_001"
        assert completion.run_id == "deleg_task_001"
        assert completion.sequence > acknowledgment.sequence
        assert completion.payload.summary.startswith("Option B")


@pytest.mark.asyncio
async def test_loopback_bridge_suppresses_duplicate_terminal_callbacks() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        acknowledgment = await client.receive()
        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)

        first = await server.complete("deleg_task_001", status="completed", summary="Done")
        duplicate = await server.complete(
            "deleg_task_001",
            status="completed",
            summary="Done",
        )

        assert isinstance(first, WorkCompletedEvent)
        assert duplicate is None
        assert await client.receive() == first


@pytest.mark.asyncio
async def test_reconnect_retry_replays_acknowledgment_and_terminal_event() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server:
        first_client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        )
        async with first_client:
            await first_client.send(request())
            first_ack = await first_client.receive()
            assert isinstance(first_ack, WorkDispatchAcknowledgedEvent)
            first_terminal = await server.complete(
                "deleg_task_001",
                status="completed",
                summary="Completed before reconnect.",
            )
            assert await first_client.receive() == first_terminal

        second_client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        )
        async with second_client:
            await second_client.send(request())
            replayed_ack = await second_client.receive()
            replayed_terminal = await second_client.receive()

        assert replayed_ack == first_ack
        assert replayed_terminal == first_terminal


@pytest.mark.asyncio
async def test_bridge_suppresses_terminal_output_after_binding_rebind() -> None:
    bindings = SessionBindings()
    original = bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 2, tzinfo=UTC),
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
    ) as client:
        await client.send(request())
        acknowledgment = await client.receive()
        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)
        assert bindings.release(original) is original
        bindings.bind("participant_001", "session_001")

        completion = await server.complete(
            "deleg_task_001",
            status="completed",
            summary="Finished stale work.",
        )

        assert completion is None


@pytest.mark.asyncio
async def test_loopback_bridge_rejects_invalid_authentication_token() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="correct-test-token-with-sufficient-entropy",
    ) as server:
        with pytest.raises(BridgeAuthenticationError):
            await LocalHermesBridgeClient.connect(
                host=server.host,
                port=server.port,
                token="incorrect-test-token",
                participant_id="participant_001",
            )


@pytest.mark.asyncio
async def test_non_ascii_authentication_token_is_rejected_without_loop_error() -> None:
    service, completions = bridge_components()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        async with LocalHermesBridgeServer(
            service=service,
            completions=completions,
            token="correct-test-token-with-sufficient-entropy",
        ) as server:
            reader, writer = await asyncio.open_connection(server.host, server.port)
            writer.write(
                json.dumps(
                    {
                        "token": "é" * 24,
                        "participant_id": "participant_001",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
            await writer.drain()

            assert json.loads(await reader.readline()) == {"ok": False}
            assert await reader.readline() == b""
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not loop_errors


@pytest.mark.asyncio
async def test_plugin_hook_streams_completion_across_bridge() -> None:
    context = HookContext()
    runtime = HermesPluginRuntime(context)  # type: ignore[arg-type]
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_cancel_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=runtime.dispatcher,
        canceller=runtime.dispatcher,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 2, tzinfo=UTC),
    )

    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        completion_source=runtime,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        acknowledgment = await client.receive()
        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)

        await client.send(
            ControlCancelEvent(
                type="control.cancel",
                event_id="evt_cancel_001",
                session_id="session_001",
                sequence=6,
                timestamp=datetime(2026, 7, 19, 5, 1, 30, tzinfo=UTC),
                task_id="task_001",
                payload={"scope": CancelScope.TASK, "reason": "user interruption"},
            )
        )
        cancel_ack = await client.receive()
        assert isinstance(cancel_ack, ControlCancelAcknowledgedEvent)
        assert cancel_ack.payload.signaled_run_ids == [acknowledgment.payload.run_id]
        assert context.interrupt_calls == [acknowledgment.payload.run_id]

        callback = context.hooks["subagent_stop"]
        assert callable(callback)
        await asyncio.to_thread(
            callback,
            delegation_id=acknowledgment.payload.run_id,
            child_status="interrupted",
            child_summary="The exact Hermes run stopped.",
        )
        completion = await client.receive()

        assert isinstance(completion, WorkCompletedEvent)
        assert completion.run_id == acknowledgment.payload.run_id
        assert completion.payload.status == "interrupted"
        assert completion.payload.reason == "The exact Hermes run stopped."


@pytest.mark.asyncio
async def test_bridge_buffers_terminal_hook_that_precedes_dispatch_acknowledgment() -> None:
    context = ImmediateCompletionHookContext()
    runtime = HermesPluginRuntime(context)  # type: ignore[arg-type]
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=runtime.dispatcher,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 2, tzinfo=UTC),
    )

    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        completion_source=runtime,
        token="test-token-with-sufficient-entropy",
        max_pending_completions=2,
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        context.before_completion = lambda: (
            server._on_completion(
                HermesRunCompletion(
                    run_id="deleg_unrelated_001",
                    status="completed",
                    summary="unrelated",
                    reason=None,
                )
            ),
            server._on_completion(
                HermesRunCompletion(
                    run_id="deleg_unrelated_002",
                    status="completed",
                    summary="unrelated",
                    reason=None,
                )
            ),
        )
        await client.send(request())
        acknowledgment = await client.receive()
        completion = await client.receive()

        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)
        assert isinstance(completion, WorkCompletedEvent)
        assert completion.run_id == acknowledgment.payload.run_id
        assert completion.payload.summary == "Completed before dispatch returned."
        assert await service.active_run_ids() == frozenset()
        with server._completion_ingress_lock:
            unrelated = {
                run_id
                for run_id in server._completion_ingress
                if run_id.startswith("deleg_unrelated_")
            }
        assert len(unrelated) <= 2


@pytest.mark.asyncio
async def test_bridge_refuses_non_loopback_bind_address() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
        host="0.0.0.0",
    )

    with pytest.raises(ValueError, match="loopback"):
        await server.start()


@pytest.mark.asyncio
async def test_bridge_times_out_unauthenticated_connection() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
        authentication_timeout=0.01,
    ) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        try:
            assert await asyncio.wait_for(reader.read(), timeout=0.5) == b""
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_bridge_counts_unauthenticated_connections_toward_limit() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
        authentication_timeout=1,
        max_connections=1,
    ) as server:
        first_reader, first_writer = await asyncio.open_connection(server.host, server.port)
        for _ in range(10):
            if server._connections:
                break
            await asyncio.sleep(0)
        assert len(server._connections) == 1
        second_reader, second_writer = await asyncio.open_connection(server.host, server.port)
        try:
            assert await asyncio.wait_for(second_reader.read(), timeout=0.5) == b""
            assert not first_reader.at_eof()
        finally:
            first_writer.close()
            second_writer.close()
            await asyncio.gather(
                first_writer.wait_closed(),
                second_writer.wait_closed(),
                return_exceptions=True,
            )


@pytest.mark.asyncio
async def test_bridge_close_terminates_unauthenticated_socket() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
        authentication_timeout=10,
    )
    await server.start()
    reader, writer = await asyncio.open_connection(server.host, server.port)
    for _ in range(10):
        if server._connections:
            break
        await asyncio.sleep(0)
    assert server._connections

    await asyncio.wait_for(server.close(), timeout=0.5)

    assert await asyncio.wait_for(reader.read(), timeout=0.5) == b""
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_bridge_rejects_oversized_raw_handshake() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        writer.write(b"x" * (64 * 1024 + 1) + b"\n")
        await writer.drain()

        assert await asyncio.wait_for(reader.read(), timeout=0.5) == b""
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_bridge_close_cancels_stuck_completion_tasks() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    )
    await server.start()
    stuck = asyncio.create_task(asyncio.Event().wait())
    server._completion_tasks.add(stuck)  # type: ignore[arg-type]

    await asyncio.wait_for(server.close(), timeout=0.5)

    assert stuck.cancelled()


@pytest.mark.asyncio
async def test_disconnected_known_run_records_terminal_and_releases_capacity() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        )
        await client.send(request())
        acknowledgment = await client.receive()
        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)
        await client.close()
        for _ in range(10):
            if not server._routes:
                break
            await asyncio.sleep(0)
        assert not server._routes

        delivered = await server.complete(
            "deleg_task_001",
            status="completed",
            summary="Completed while disconnected.",
        )

        assert delivered is None
        assert await completions.completed("deleg_task_001") is not None
        assert "deleg_task_001" not in service._active_runs.values()


@pytest.mark.asyncio
async def test_cancel_ack_is_written_before_later_terminal_sequence() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_cancel_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        canceller=RecordingCanceller(),
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 1, tzinfo=UTC),
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 2, tzinfo=UTC),
    )
    async with BlockingCancelSendServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        await client.receive()
        await client.send(
            ControlCancelEvent(
                type="control.cancel",
                event_id="evt_cancel_001",
                session_id="session_001",
                sequence=6,
                timestamp=datetime(2026, 7, 19, 5, 1, 30, tzinfo=UTC),
                task_id="task_001",
                payload={"scope": CancelScope.TASK, "reason": "user interruption"},
            )
        )
        await server.cancel_send_started.wait()
        completion_task = asyncio.create_task(
            server.complete(
                "deleg_task_001",
                status="interrupted",
                reason="Interrupted exactly.",
            )
        )
        await asyncio.sleep(0)
        server.release_cancel_send.set()

        first = await client.receive()
        second = await client.receive()
        await completion_task

        assert isinstance(first, ControlCancelAcknowledgedEvent)
        assert isinstance(second, WorkCompletedEvent)
        assert first.sequence < second.sequence


@pytest.mark.asyncio
async def test_bridge_bounds_oversized_hermes_completion_summary() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        await client.receive()

        await server.complete(
            "deleg_task_001",
            status="failed",
            summary="\x00" * 9000,
            reason="\x00" * 9000,
        )
        completion = await client.receive()

        assert isinstance(completion, WorkCompletedEvent)
        assert completion.payload.summary == "\x00" * 4096
        assert completion.payload.reason == "\x00" * 4096


@pytest.mark.asyncio
async def test_same_connection_retry_resequences_retained_evidence() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        first_ack = await client.receive()
        await server.complete(
            "deleg_task_001",
            status="completed",
            summary="done",
        )
        first_terminal = await client.receive()

        await client.send(request())
        replayed_ack = await client.receive()
        replayed_terminal = await client.receive()

        assert first_ack.sequence < first_terminal.sequence
        assert first_terminal.sequence < replayed_ack.sequence
        assert replayed_ack.sequence < replayed_terminal.sequence
        assert replayed_ack.event_id == first_ack.event_id
        assert replayed_terminal.event_id == first_terminal.event_id


@pytest.mark.asyncio
async def test_cancelled_completion_claim_can_be_retried() -> None:
    service, completions = bridge_components()
    async with LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    ) as server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        await client.receive()
        route = server._routes["deleg_task_001"]
        await route.connection.event_lock.acquire()
        completion_task = asyncio.create_task(
            server.complete(
                "deleg_task_001",
                status="completed",
                summary="done",
            )
        )
        for _ in range(10):
            if "deleg_task_001" in server._completing_runs:
                break
            await asyncio.sleep(0)
        completion_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await completion_task
        route.connection.event_lock.release()

        assert "deleg_task_001" not in server._completing_runs
        await server.complete(
            "deleg_task_001",
            status="completed",
            summary="done",
        )
        assert isinstance(await client.receive(), WorkCompletedEvent)


@pytest.mark.asyncio
async def test_terminal_eviction_rejects_stale_acknowledgment_retry() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(f"evt_{index}" for index in range(20))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 1, tzinfo=UTC),
        max_retained_dispatches=2,
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 5, 2, tzinfo=UTC),
        max_retained_completions=1,
    )
    second_request = request().model_copy(
        update={
            "event_id": "evt_request_002",
            "sequence": 10,
            "task_id": "task_002",
            "utterance_id": "utterance_002",
        }
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
    ) as client:
        await client.send(request())
        await client.receive()
        await server.complete("deleg_task_001", status="completed", summary="one")
        await client.receive()

        await client.send(second_request)
        await client.receive()
        await server.complete("deleg_task_002", status="completed", summary="two")
        await client.receive()

        await client.send(request())
        expired = await client.receive()

        assert isinstance(expired, WorkDispatchAcknowledgedEvent)
        assert not expired.payload.accepted
        assert expired.payload.reason is not None
        assert "retention expired" in expired.payload.reason


@pytest.mark.asyncio
async def test_close_rejects_handler_that_registers_after_shutdown_snapshot() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    )

    class DelayedWriter:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

    writer = DelayedWriter()
    reader = asyncio.StreamReader()
    await server._state_lock.acquire()
    close_task = asyncio.create_task(server.close())
    await asyncio.sleep(0)
    handler_task = asyncio.create_task(
        server._handle_connection(  # type: ignore[arg-type]
            reader,
            writer,
            server._server_generation,
        )
    )
    await asyncio.sleep(0)
    server._state_lock.release()

    await close_task
    await asyncio.wait_for(handler_task, timeout=0.1)

    assert writer.closed
    assert not server._connections
    assert not server._handler_tasks


@pytest.mark.asyncio
async def test_cancellation_waits_for_terminal_capacity_release() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    )
    async with server, await LocalHermesBridgeClient.connect(
        host=server.host,
        port=server.port,
        token="test-token-with-sufficient-entropy",
        participant_id="participant_001",
    ) as client:
        await client.send(request())
        acknowledgment = await client.receive()
        assert isinstance(acknowledgment, WorkDispatchAcknowledgedEvent)
        assert acknowledgment.payload.run_id is not None

        await service._dispatch_lock.acquire()
        completion_task = asyncio.create_task(
            server.complete(
                acknowledgment.payload.run_id,
                status="completed",
                summary="done",
            )
        )
        try:
            assert isinstance(await client.receive(), WorkCompletedEvent)
            completion_task.cancel()
            await asyncio.sleep(0)
            assert not completion_task.done()
        finally:
            service._dispatch_lock.release()

        with pytest.raises(asyncio.CancelledError):
            await completion_task
        assert not service._active_runs


@pytest.mark.asyncio
async def test_close_serializes_with_start_before_listener_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    )
    real_start_server = asyncio.start_server
    listener_created = asyncio.Event()
    release_start = asyncio.Event()

    async def delayed_start_server(*args: object, **kwargs: object) -> asyncio.Server:
        listener = await real_start_server(*args, **kwargs)  # type: ignore[arg-type]
        listener_created.set()
        await release_start.wait()
        return listener

    monkeypatch.setattr(asyncio, "start_server", delayed_start_server)
    start_task = asyncio.create_task(server.start())
    await listener_created.wait()
    close_task = asyncio.create_task(server.close())
    await asyncio.sleep(0)
    close_returned_before_start = close_task.done()
    release_start.set()
    await start_task
    await close_task
    listener_remained = server._server is not None
    if listener_remained:
        await server.close()

    assert not close_returned_before_start
    assert not listener_remained


@pytest.mark.asyncio
async def test_closed_bridge_instance_cannot_be_restarted() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    )
    await server.start()
    await server.close()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await server.start()


@pytest.mark.asyncio
async def test_completion_hook_ingress_is_bounded_and_deduplicated() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
        max_pending_completions=2,
    )
    await server.start()
    try:
        for index in range(100):
            server._on_completion(
                HermesRunCompletion(
                    run_id=f"unrelated_{index}",
                    status="completed",
                    summary=None,
                    reason=None,
                )
            )
        server._on_completion(
            HermesRunCompletion(
                run_id="unrelated_0",
                status="completed",
                summary=None,
                reason=None,
            )
        )

        assert len(server._completion_ingress) == 2
        await asyncio.sleep(0)
        assert len(server._completion_tasks) <= 2
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_binding_admission_permission_error_is_handled_at_socket_boundary() -> None:
    service, completions = bridge_components()

    async def reject_dispatch(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PermissionError("participant binding was released before dispatch admission")

    service.dispatch = reject_dispatch  # type: ignore[method-assign]
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        async with LocalHermesBridgeServer(
            service=service,
            completions=completions,
            token="test-token-with-sufficient-entropy",
        ) as server, await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        ) as client:
            await client.send(request())
            with pytest.raises(BridgeProtocolError, match="closed"):
                await client.receive()
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not loop_errors


@pytest.mark.asyncio
async def test_active_pre_ack_terminal_survives_unrelated_pending_flood() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
        max_pending_completions=2,
    )
    acknowledgment = await service.dispatch("participant_001", request())
    run_id = acknowledgment.payload.run_id
    assert run_id == "deleg_task_001"

    await server.start()
    try:
        server._on_completion(
            HermesRunCompletion("unrelated_1", "completed", None, None)
        )
        server._on_completion(
            HermesRunCompletion("unrelated_2", "completed", None, None)
        )
        server._on_completion(
            HermesRunCompletion(run_id, "completed", "authoritative", None)
        )

        assert run_id in server._completion_ingress
        assert (
            sum(key.startswith("unrelated_") for key in server._completion_ingress) == 1
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        server._on_completion(
            HermesRunCompletion("unrelated_3", "completed", None, None)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert run_id in server._pending
        assert sum(key.startswith("unrelated_") for key in server._pending) <= 2

        await completions.track(acknowledgment)
        async with server._state_lock:
            status, summary, reason = server._pending.pop(run_id)
        assert await server.complete(
            run_id,
            status=status,
            summary=summary,
            reason=reason,
        ) is None

        terminal = await completions.completed(run_id)
        assert terminal is not None
        assert terminal.payload.summary == "authoritative"
        assert await service.active_run_ids() == frozenset()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_pending_terminal_survives_acknowledgment_transport_failure() -> None:
    service, completions = bridge_components()
    server = LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token="test-token-with-sufficient-entropy",
    )
    acknowledgment = await service.dispatch("participant_001", request())
    run_id = acknowledgment.payload.run_id
    assert run_id == "deleg_task_001"
    assert await server.complete(run_id, status="completed", summary="authoritative") is None

    await server.start()
    original_send_bound = server._send_bound
    failed = False

    async def fail_first_ack(route: object, event: ProtocolEvent) -> bool:
        nonlocal failed
        if isinstance(event, WorkDispatchAcknowledgedEvent) and not failed:
            failed = True
            raise ConnectionError("forced acknowledgment transport failure")
        return await original_send_bound(route, event)  # type: ignore[arg-type]

    server._send_bound = fail_first_ack  # type: ignore[method-assign]
    try:
        first = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        )
        async with first:
            await first.send(request())
            with pytest.raises(BridgeProtocolError, match="closed"):
                await first.receive()
        assert run_id in server._pending

        server._send_bound = original_send_bound  # type: ignore[method-assign]
        second = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token="test-token-with-sufficient-entropy",
            participant_id="participant_001",
        )
        async with second:
            await second.send(request())
            replayed_ack = await second.receive()
            replayed_terminal = await second.receive()

        assert isinstance(replayed_ack, WorkDispatchAcknowledgedEvent)
        assert isinstance(replayed_terminal, WorkCompletedEvent)
        assert replayed_terminal.run_id == run_id
        assert replayed_terminal.payload.summary == "authoritative"
        assert run_id not in server._pending
        assert await service.active_run_ids() == frozenset()
    finally:
        await server.close()
