"""Worker-facing correlated session over the local Hermes bridge."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, Self, cast
from zoneinfo import ZoneInfo

from pydantic_core import TzInfo

from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    ControlCancelPayload,
    Durability,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkDispatchRequestedPayload,
    WorkTerminalStatus,
)

from .bridge import BridgeProtocolError

_MAX_IDENTIFIER_CHARS = 128
_MAX_TEXT_CHARS = 1024
_MAX_SEQUENCE = (1 << 63) - 1


def _require_exact_string(value: Any, field: str, *, max_chars: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be an exact built-in string")
    if not value or len(value) > max_chars:
        raise BridgeProtocolError(f"{field} must be non-empty and bounded")
    return value


def _require_optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = _require_exact_string(value, field, max_chars=_MAX_TEXT_CHARS)
    if not text.strip():
        raise BridgeProtocolError(f"{field} must contain non-whitespace text")
    return text


def _require_safe_timestamp(value: Any) -> datetime:
    if type(value) is not datetime:
        raise TypeError("timestamp must be an exact datetime")
    zone = value.tzinfo
    if type(zone) not in (timezone, ZoneInfo, TzInfo):
        raise TypeError("timestamp timezone must use an exact trusted implementation")
    if value.utcoffset() is None:
        raise BridgeProtocolError("event timestamp must be timezone-aware")
    return value


class _HermesBridgeClient(Protocol):
    async def send(
        self,
        event: WorkDispatchRequestedEvent | ControlCancelEvent,
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None: ...

    async def receive(self) -> Any: ...


class RealtimeHermesSession:
    """Correlate bridge replies and push terminal updates to a conversation worker."""

    def __init__(
        self,
        *,
        client: _HermesBridgeClient,
        session_id: str,
        max_pending_updates: int = 256,
        max_pending_operations: int = 64,
        close_drain_timeout_ms: int = 1000,
    ) -> None:
        if type(max_pending_updates) is not int:
            raise TypeError("max_pending_updates must be an exact integer")
        if not 1 <= max_pending_updates <= 4096:
            raise ValueError("max_pending_updates must be between 1 and 4096")
        if type(max_pending_operations) is not int:
            raise TypeError("max_pending_operations must be an exact integer")
        if not 1 <= max_pending_operations <= 1024:
            raise ValueError("max_pending_operations must be between 1 and 1024")
        if type(close_drain_timeout_ms) is not int:
            raise TypeError("close_drain_timeout_ms must be an exact integer")
        if not 1 <= close_drain_timeout_ms <= 60_000:
            raise ValueError("close_drain_timeout_ms must be between 1 and 60000")
        self._client = client
        self._session_id = _require_exact_string(
            session_id,
            "session_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        self._receiver: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._pending_dispatches: dict[
            str, asyncio.Future[WorkDispatchAcknowledgedEvent]
        ] = {}
        self._pending_cancellations: dict[
            str, asyncio.Future[ControlCancelAcknowledgedEvent]
        ] = {}
        self._updates: asyncio.Queue[WorkCompletedEvent | BaseException] = asyncio.Queue(
            maxsize=max_pending_updates
        )
        self._last_sequence = -1
        self._terminal_error: BaseException | None = None
        self._closed = False
        self._max_pending_operations = max_pending_operations
        self._owned_operations: set[asyncio.Task[Any]] = set()
        self._close_drain_timeout = close_drain_timeout_ms / 1000
        self._close_operation: asyncio.Task[None] | None = None
        self._detached_tasks: set[asyncio.Task[Any]] = set()

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def start(self) -> None:
        """Start the single reader that owns all inbound bridge correlation."""

        self._raise_if_unavailable()
        if self._receiver is None:
            self._receiver = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        """Close through one caller-cancellation-resistant owned operation."""

        operation = self._close_operation
        if operation is None:
            operation = asyncio.create_task(
                self._close_owned(),
                name=f"realtime-hermes-close:{self._session_id}",
            )
            self._close_operation = operation
            operation.add_done_callback(self._consume_close_operation)
        await asyncio.shield(operation)

    async def _close_owned(self) -> None:
        self._closed = True
        error = BridgeProtocolError("realtime Hermes session closed")
        await self._set_terminal_error(error)
        assert self._terminal_error is not None
        self._publish_terminal_error(self._terminal_error)
        receiver, self._receiver = self._receiver, None
        tasks = tuple(self._owned_operations)
        if receiver is not None:
            tasks = (*tasks, receiver)
        await self._bounded_cancel_and_drain(tasks)

    async def _bounded_cancel_and_drain(
        self,
        tasks: tuple[asyncio.Task[Any], ...],
    ) -> None:
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=self._close_drain_timeout)
        self._retrieve_task_failures(done)
        for task in pending:
            self._detached_tasks.add(task)
            task.add_done_callback(self._consume_detached_task)

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        """Send work and await the acknowledgment correlated by task_id."""

        request = self._validate_dispatch_request(request)
        self._authorize_session(request.session_id)
        self.start()
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._raise_if_unavailable()
            if request.task_id in self._pending_dispatches:
                raise ValueError("a dispatch with this task_id is already pending")
            if (
                len(self._pending_dispatches) + len(self._pending_cancellations)
                >= self._max_pending_operations
            ):
                raise BridgeProtocolError("worker pending operation capacity exhausted")
            future: asyncio.Future[WorkDispatchAcknowledgedEvent] = loop.create_future()
            self._pending_dispatches[request.task_id] = future
        operation = asyncio.create_task(
            self._settle_dispatch(request, request.task_id, future)
        )
        self._owned_operations.add(operation)
        operation.add_done_callback(self._consume_operation)
        return await asyncio.shield(operation)

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        """Send scoped cancellation and await its exact request acknowledgment."""

        request = self._validate_cancel_request(request)
        self._authorize_session(request.session_id)
        self.start()
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._raise_if_unavailable()
            if request.event_id in self._pending_cancellations:
                raise ValueError("this cancellation event is already pending")
            if (
                len(self._pending_dispatches) + len(self._pending_cancellations)
                >= self._max_pending_operations
            ):
                raise BridgeProtocolError("worker pending operation capacity exhausted")
            future: asyncio.Future[ControlCancelAcknowledgedEvent] = loop.create_future()
            self._pending_cancellations[request.event_id] = future
        operation = asyncio.create_task(
            self._settle_cancellation(request, request.event_id, future)
        )
        self._owned_operations.add(operation)
        operation.add_done_callback(self._consume_operation)
        return await asyncio.shield(operation)

    async def next_update(self) -> WorkCompletedEvent:
        """Wait for the next proactively delivered terminal work update."""

        self._raise_if_unavailable()
        update = await self._updates.get()
        if isinstance(update, BaseException):
            if self._terminal_error is update and not self._updates.full():
                self._updates.put_nowait(update)
            raise update
        return update

    def _authorize_session(self, session_id: str) -> None:
        if session_id != self._session_id:
            raise PermissionError("event does not belong to this Hermes session")

    @staticmethod
    def _validate_dispatch_request(request: Any) -> WorkDispatchRequestedEvent:
        if type(request) is not WorkDispatchRequestedEvent:
            raise TypeError("dispatch request must use its exact event class")
        event_type = _require_exact_string(request.type, "event type", max_chars=64)
        if event_type != "work.dispatch.requested":
            raise BridgeProtocolError("dispatch request type contradicted its class")
        protocol_version = _require_exact_string(
            request.protocol_version,
            "protocol version",
            max_chars=16,
        )
        if protocol_version != "0.1":
            raise BridgeProtocolError("dispatch protocol version is unsupported")
        event_id = _require_exact_string(
            request.event_id,
            "event_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        session_id = _require_exact_string(
            request.session_id,
            "session_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        if type(request.sequence) is not int:
            raise TypeError("sequence must be an exact built-in integer")
        sequence = request.sequence
        if not 0 <= sequence <= _MAX_SEQUENCE:
            raise BridgeProtocolError("event sequence must be non-negative and bounded")
        timestamp = _require_safe_timestamp(request.timestamp)
        task_id = _require_exact_string(
            request.task_id,
            "task_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        utterance_id = _require_exact_string(
            request.utterance_id,
            "utterance_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        if type(request.payload) is not WorkDispatchRequestedPayload:
            raise TypeError("dispatch payload must use its exact model")
        payload = request.payload
        objective = _require_exact_string(
            payload.objective,
            "objective",
            max_chars=_MAX_TEXT_CHARS,
        )
        if not objective.strip():
            raise BridgeProtocolError("objective must contain non-whitespace text")
        if type(payload.durability) is not Durability:
            raise TypeError("durability must use the exact protocol enum")
        progress_reporting = _require_exact_string(
            payload.progress_reporting,
            "progress_reporting",
            max_chars=32,
        )
        if progress_reporting not in ("none", "material_only", "all"):
            raise BridgeProtocolError("progress_reporting is unsupported")
        completion_reporting = _require_exact_string(
            payload.completion_reporting,
            "completion_reporting",
            max_chars=32,
        )
        if completion_reporting not in ("proactive", "on_request"):
            raise BridgeProtocolError("completion_reporting is unsupported")
        progress_mode = cast(
            Literal["none", "material_only", "all"],
            progress_reporting,
        )
        completion_mode = cast(
            Literal["proactive", "on_request"],
            completion_reporting,
        )
        return WorkDispatchRequestedEvent(
            type="work.dispatch.requested",
            protocol_version="0.1",
            event_id=event_id,
            session_id=session_id,
            sequence=sequence,
            timestamp=timestamp,
            task_id=task_id,
            utterance_id=utterance_id,
            payload=WorkDispatchRequestedPayload(
                objective=objective,
                durability=payload.durability,
                progress_reporting=progress_mode,
                completion_reporting=completion_mode,
            ),
        )

    @staticmethod
    def _validate_cancel_request(request: Any) -> ControlCancelEvent:
        if type(request) is not ControlCancelEvent:
            raise TypeError("cancellation request must use its exact event class")
        event_type = _require_exact_string(request.type, "event type", max_chars=64)
        if event_type != "control.cancel":
            raise BridgeProtocolError("cancellation request type contradicted its class")
        protocol_version = _require_exact_string(
            request.protocol_version,
            "protocol version",
            max_chars=16,
        )
        if protocol_version != "0.1":
            raise BridgeProtocolError("cancellation protocol version is unsupported")
        event_id = _require_exact_string(
            request.event_id,
            "event_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        session_id = _require_exact_string(
            request.session_id,
            "session_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        if type(request.sequence) is not int:
            raise TypeError("sequence must be an exact built-in integer")
        sequence = request.sequence
        if not 0 <= sequence <= _MAX_SEQUENCE:
            raise BridgeProtocolError("event sequence must be non-negative and bounded")
        timestamp = _require_safe_timestamp(request.timestamp)
        if request.task_id is None:
            task_id = None
        else:
            task_id = _require_exact_string(
                request.task_id,
                "task_id",
                max_chars=_MAX_IDENTIFIER_CHARS,
            )
        if type(request.payload) is not ControlCancelPayload:
            raise TypeError("cancellation payload must use its exact model")
        payload = request.payload
        if type(payload.scope) is not CancelScope:
            raise TypeError("cancellation scope must use the exact protocol enum")
        reason = _require_optional_text(payload.reason, "cancellation reason")
        if payload.scope is CancelScope.TASK and task_id is None:
            raise BridgeProtocolError("task cancellation requires task_id")
        if payload.scope is not CancelScope.TASK and task_id is not None:
            raise BridgeProtocolError("non-task cancellation forbids task_id")
        return ControlCancelEvent(
            type="control.cancel",
            protocol_version="0.1",
            event_id=event_id,
            session_id=session_id,
            sequence=sequence,
            timestamp=timestamp,
            task_id=task_id,
            payload=ControlCancelPayload(scope=payload.scope, reason=reason),
        )

    async def _settle_dispatch(
        self,
        request: WorkDispatchRequestedEvent,
        correlation_task_id: str,
        future: asyncio.Future[WorkDispatchAcknowledgedEvent],
    ) -> WorkDispatchAcknowledgedEvent:
        try:
            self._raise_if_unavailable()
            await self._client.send(
                request,
                admission_guard=self._raise_if_unavailable,
            )
            return await asyncio.shield(future)
        except BaseException:
            async with self._lock:
                self._pending_dispatches.pop(correlation_task_id, None)
            if future.done() and not future.cancelled():
                future.exception()
            raise

    async def _settle_cancellation(
        self,
        request: ControlCancelEvent,
        correlation_event_id: str,
        future: asyncio.Future[ControlCancelAcknowledgedEvent],
    ) -> ControlCancelAcknowledgedEvent:
        try:
            self._raise_if_unavailable()
            await self._client.send(
                request,
                admission_guard=self._raise_if_unavailable,
            )
            return await asyncio.shield(future)
        except BaseException:
            async with self._lock:
                self._pending_cancellations.pop(correlation_event_id, None)
            if future.done() and not future.cancelled():
                future.exception()
            raise

    def _consume_operation(self, operation: asyncio.Task[Any]) -> None:
        self._owned_operations.discard(operation)
        if not operation.cancelled():
            operation.exception()

    @staticmethod
    def _retrieve_task_failures(tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            if not task.cancelled():
                task.exception()

    def _consume_detached_task(self, task: asyncio.Task[Any]) -> None:
        self._detached_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _consume_close_operation(operation: asyncio.Task[None]) -> None:
        if not operation.cancelled():
            operation.exception()

    async def _receive_loop(self) -> None:
        try:
            while True:
                event = await self._client.receive()
                if self._closed:
                    return
                event = self._validate_incoming(event)
                if type(event) is WorkDispatchAcknowledgedEvent:
                    async with self._lock:
                        dispatch_future = self._pending_dispatches.pop(
                            event.task_id,
                            None,
                        )
                    if dispatch_future is None:
                        raise BridgeProtocolError(
                            "dispatch acknowledgment had no pending task"
                        )
                    dispatch_future.set_result(event)
                elif type(event) is ControlCancelAcknowledgedEvent:
                    async with self._lock:
                        cancel_future = self._pending_cancellations.pop(
                            event.request_event_id,
                            None,
                        )
                    if cancel_future is None:
                        raise BridgeProtocolError(
                            "cancellation acknowledgment had no pending request"
                        )
                    cancel_future.set_result(event)
                elif type(event) is WorkCompletedEvent:
                    if self._updates.full():
                        raise BridgeProtocolError(
                            "terminal update buffer exhausted before the worker consumed it"
                        )
                    self._updates.put_nowait(event)
                else:
                    raise BridgeProtocolError("worker received an unsupported event")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._set_terminal_error(exc)
            self._publish_terminal_error(exc)

    def _validate_incoming(self, event: Any) -> Any:
        event_class = type(event)
        if event_class is WorkDispatchAcknowledgedEvent:
            expected_type = "work.dispatch.acknowledged"
        elif event_class is ControlCancelAcknowledgedEvent:
            expected_type = "control.cancel.acknowledged"
        elif event_class is WorkCompletedEvent:
            expected_type = "work.completed"
        else:
            raise BridgeProtocolError("worker received an unsupported exact event class")
        event_type = _require_exact_string(event.type, "event type", max_chars=64)
        if event_type != expected_type:
            raise BridgeProtocolError("event type contradicted its exact event class")
        protocol_version = _require_exact_string(
            event.protocol_version,
            "protocol version",
            max_chars=16,
        )
        if protocol_version != "0.1":
            raise BridgeProtocolError("event protocol version is unsupported")
        event_id = _require_exact_string(
            event.event_id,
            "event_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        session_id = _require_exact_string(
            event.session_id,
            "session_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        if type(event.sequence) is not int:
            raise TypeError("sequence must be an exact built-in integer")
        sequence = event.sequence
        if not 0 <= sequence <= _MAX_SEQUENCE:
            raise BridgeProtocolError("event sequence must be non-negative and bounded")
        timestamp = _require_safe_timestamp(event.timestamp)

        if event_class is WorkDispatchAcknowledgedEvent:
            self._validate_dispatch_acknowledgment(event)
            canonical: Any = WorkDispatchAcknowledgedEvent(
                type="work.dispatch.acknowledged",
                protocol_version="0.1",
                event_id=event_id,
                session_id=session_id,
                sequence=sequence,
                timestamp=timestamp,
                task_id=event.task_id,
                payload=WorkDispatchAcknowledgedPayload(
                    accepted=event.payload.accepted,
                    run_id=event.payload.run_id,
                    reason=event.payload.reason,
                ),
            )
        elif event_class is ControlCancelAcknowledgedEvent:
            self._validate_cancel_acknowledgment(event)
            canonical = ControlCancelAcknowledgedEvent(
                type="control.cancel.acknowledged",
                protocol_version="0.1",
                event_id=event_id,
                session_id=session_id,
                sequence=sequence,
                timestamp=timestamp,
                request_event_id=event.request_event_id,
                scope=event.scope,
                task_id=event.task_id,
                payload=ControlCancelAcknowledgedPayload(
                    accepted=event.payload.accepted,
                    signaled_run_ids=list(event.payload.signaled_run_ids),
                    reason=event.payload.reason,
                ),
            )
        else:
            self._validate_terminal(event)
            canonical = WorkCompletedEvent(
                type="work.completed",
                protocol_version="0.1",
                event_id=event_id,
                session_id=session_id,
                sequence=sequence,
                timestamp=timestamp,
                task_id=event.task_id,
                run_id=event.run_id,
                payload=WorkCompletedPayload(
                    status=event.payload.status,
                    summary=event.payload.summary,
                    reason=event.payload.reason,
                ),
            )

        self._authorize_session(session_id)
        if sequence <= self._last_sequence:
            raise BridgeProtocolError("bridge event sequence was not strictly increasing")
        self._last_sequence = sequence
        return canonical

    @staticmethod
    def _validate_dispatch_acknowledgment(event: Any) -> None:
        _require_exact_string(event.task_id, "task_id", max_chars=_MAX_IDENTIFIER_CHARS)
        if type(event.payload) is not WorkDispatchAcknowledgedPayload:
            raise TypeError("dispatch acknowledgment payload must use its exact model")
        payload = event.payload
        if type(payload.accepted) is not bool:
            raise TypeError("dispatch accepted must be an exact built-in boolean")
        run_id = (
            None
            if payload.run_id is None
            else _require_exact_string(
                payload.run_id,
                "run_id",
                max_chars=_MAX_IDENTIFIER_CHARS,
            )
        )
        reason = _require_optional_text(payload.reason, "dispatch reason")
        if payload.accepted and (run_id is None or reason is not None):
            raise BridgeProtocolError("accepted dispatch evidence is contradictory")
        if not payload.accepted and (run_id is not None or reason is None):
            raise BridgeProtocolError("rejected dispatch evidence is contradictory")

    @staticmethod
    def _validate_cancel_acknowledgment(event: Any) -> None:
        _require_exact_string(
            event.request_event_id,
            "request_event_id",
            max_chars=_MAX_IDENTIFIER_CHARS,
        )
        if type(event.scope) is not CancelScope:
            raise TypeError("cancellation scope must use the exact protocol enum")
        if event.task_id is not None:
            _require_exact_string(
                event.task_id,
                "task_id",
                max_chars=_MAX_IDENTIFIER_CHARS,
            )
        if event.scope is CancelScope.TASK and event.task_id is None:
            raise BridgeProtocolError("task cancellation acknowledgment requires task_id")
        if event.scope is not CancelScope.TASK and event.task_id is not None:
            raise BridgeProtocolError("non-task cancellation acknowledgment forbids task_id")
        if type(event.payload) is not ControlCancelAcknowledgedPayload:
            raise TypeError("cancellation acknowledgment payload must use its exact model")
        payload = event.payload
        if type(payload.accepted) is not bool:
            raise TypeError("cancellation accepted must be an exact built-in boolean")
        if type(payload.signaled_run_ids) is not list:
            raise TypeError("signaled_run_ids must be an exact built-in list")
        if len(payload.signaled_run_ids) > 256:
            raise BridgeProtocolError("signaled_run_ids exceeded its finite bound")
        run_ids = tuple(
            _require_exact_string(
                run_id,
                "signaled run_id",
                max_chars=_MAX_IDENTIFIER_CHARS,
            )
            for run_id in payload.signaled_run_ids
        )
        if len(set(run_ids)) != len(run_ids):
            raise BridgeProtocolError("signaled_run_ids contained duplicates")
        reason = _require_optional_text(payload.reason, "cancellation reason")
        if payload.accepted and (not run_ids or reason is not None):
            raise BridgeProtocolError("accepted cancellation evidence is contradictory")
        if not payload.accepted and (run_ids or reason is None):
            raise BridgeProtocolError("rejected cancellation evidence is contradictory")

    @staticmethod
    def _validate_terminal(event: Any) -> None:
        _require_exact_string(event.task_id, "task_id", max_chars=_MAX_IDENTIFIER_CHARS)
        _require_exact_string(event.run_id, "run_id", max_chars=_MAX_IDENTIFIER_CHARS)
        if type(event.payload) is not WorkCompletedPayload:
            raise TypeError("terminal payload must use its exact model")
        payload = event.payload
        if type(payload.status) is not WorkTerminalStatus:
            raise TypeError("terminal status must use the exact protocol enum")
        summary = _require_optional_text(payload.summary, "terminal summary")
        reason = _require_optional_text(payload.reason, "terminal reason")
        if payload.status is WorkTerminalStatus.COMPLETED:
            if summary is None or reason is not None:
                raise BridgeProtocolError("completed terminal evidence is contradictory")
        elif summary is not None or reason is None:
            raise BridgeProtocolError("non-completed terminal evidence is contradictory")

    def _raise_if_unavailable(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._closed:
            raise BridgeProtocolError("realtime Hermes session closed")

    async def _set_terminal_error(self, exc: BaseException) -> None:
        # Publish failure before contending for the admission lock. A dispatch
        # already queued on that lock must observe the failed receiver and must
        # not send work that can no longer be acknowledged.
        if self._terminal_error is None:
            self._terminal_error = exc
        self._fail_pending(self._terminal_error)

    def _publish_terminal_error(self, exc: BaseException) -> None:
        while not self._updates.empty():
            self._updates.get_nowait()
        self._updates.put_nowait(exc)

    def _fail_pending(self, exc: BaseException) -> None:
        dispatches = tuple(self._pending_dispatches.values())
        cancellations = tuple(self._pending_cancellations.values())
        self._pending_dispatches.clear()
        self._pending_cancellations.clear()
        for future in dispatches:
            if not future.done():
                future.set_exception(exc)
        for cancellation_future in cancellations:
            if not cancellation_future.done():
                cancellation_future.set_exception(exc)
