"""Narrow authoritative task controls for one conversation generation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic_core import TzInfo

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

from .context import ConversationContextStore, PrivateRunDisclosureError, TaskAdmission

_TASK_ID_PATTERN = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_RUN_ID_PATTERN = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_PRIVATE_RUN_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*"
)
_MAX_TASK_ID_CHARS = 128
_MAX_OUTCOME_TEXT_CHARS = 1024
_MAX_PROTOCOL_SEQUENCE = (1 << 63) - 1
_TERMINAL_STATUSES = frozenset(("completed", "failed", "interrupted"))


def _validate_task_id(task_id: str) -> None:
    if type(task_id) is not str:
        raise TypeError("task_id must be an exact built-in string")
    if len(task_id) > _MAX_TASK_ID_CHARS or _TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError("task_id must use the bounded task_ namespace")
    if _PRIVATE_RUN_TOKEN_PATTERN.search(task_id) is not None:
        raise PrivateRunDisclosureError("model-visible text contains a private run token")


def _validate_run_id(run_id: str) -> None:
    if type(run_id) is not str:
        raise TypeError("run_id must be an exact built-in string")
    if len(run_id) > _MAX_TASK_ID_CHARS or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must use the bounded deleg_ namespace")


def _validate_protocol_identifier(value: str, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact built-in string")
    if len(value) > _MAX_TASK_ID_CHARS or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical protocol identifier")


def _validate_event_envelope(event: Any, expected_type: str) -> None:
    if type(event.type) is not str:
        raise TypeError("event type must be an exact built-in string")
    if event.type != expected_type:
        raise RuntimeError("event type contradicts its exact event class")
    if type(event.protocol_version) is not str:
        raise TypeError("protocol version must be an exact built-in string")
    if event.protocol_version != "0.1":
        raise RuntimeError("event protocol version is unsupported")
    _validate_protocol_identifier(event.event_id, "event_id")
    if type(event.sequence) is not int:
        raise TypeError("event sequence must be an exact built-in integer")
    if event.sequence < 0 or event.sequence > _MAX_PROTOCOL_SEQUENCE:
        raise ValueError("event sequence must be non-negative and bounded")
    if type(event.timestamp) is not datetime:
        raise TypeError("event timestamp must be an exact datetime")
    zone = event.timestamp.tzinfo
    if type(zone) not in (timezone, ZoneInfo, TzInfo):
        raise TypeError("event timezone must use an exact trusted implementation")
    if event.timestamp.utcoffset() is None:
        raise ValueError("event timestamp must be timezone-aware")


def _validate_outcome_text(value: str | None, field: str) -> None:
    if value is None:
        return
    if type(value) is not str:
        raise TypeError(f"{field} must be an exact built-in string or None")
    if not value.strip() or len(value) > _MAX_OUTCOME_TEXT_CHARS:
        raise ValueError(f"{field} must be non-empty and bounded")
    if _PRIVATE_RUN_TOKEN_PATTERN.search(value) is not None:
        raise PrivateRunDisclosureError("model-visible text contains a private run token")


class TaskControlSession(Protocol):
    """Existing bridge operations used by the conversation task controller."""

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent: ...

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent: ...

    async def next_update(self) -> WorkCompletedEvent: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskDispatchOutcome:
    """Model-safe dispatch outcome without the private Hermes run handle."""

    task_id: str
    accepted: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_task_id(self.task_id)
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact built-in boolean")
        _validate_outcome_text(self.reason, "reason")
        if self.accepted and self.reason is not None:
            raise ValueError("accepted dispatch cannot include a rejection reason")


@dataclass(frozen=True, slots=True)
class TaskCancelOutcome:
    """Model-safe cancellation outcome without private run handles."""

    task_id: str
    accepted: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_task_id(self.task_id)
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact built-in boolean")
        _validate_outcome_text(self.reason, "reason")
        if self.accepted and self.reason is not None:
            raise ValueError("accepted cancellation cannot include a rejection reason")


@dataclass(frozen=True, slots=True)
class TaskTerminalOutcome:
    """Model-safe terminal update without the private Hermes run handle."""

    task_id: str
    status: str
    summary: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_task_id(self.task_id)
        if type(self.status) is not str:
            raise TypeError("status must be an exact built-in string")
        if self.status not in _TERMINAL_STATUSES:
            raise ValueError("status must be a terminal task status")
        _validate_outcome_text(self.summary, "summary")
        _validate_outcome_text(self.reason, "reason")
        if self.status == "completed":
            if self.summary is None or self.reason is not None:
                raise ValueError("completed status requires summary-only evidence")
        elif self.reason is None or self.summary is not None:
            raise ValueError("non-completed status requires reason-only evidence")


TaskOperationOutcome = TaskDispatchOutcome | TaskCancelOutcome


@dataclass(frozen=True, slots=True)
class _PendingDispatch:
    admission: TaskAdmission


@dataclass(frozen=True, slots=True)
class _TerminalEvidence:
    task_id: str
    run_id: str
    event_sequence: int
    outcome: TaskTerminalOutcome


class ConversationTaskController:
    """Project bridge-grounded task state into one conversation context."""

    def __init__(
        self,
        *,
        context: ConversationContextStore,
        session: TaskControlSession,
        session_id: str,
        id_factory: Callable[[], str],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_terminal_updates: int = 8,
        close_drain_timeout_ms: int = 1000,
        max_protocol_operations: int = 4096,
    ) -> None:
        if type(context) is not ConversationContextStore:
            raise TypeError("context must be an exact ConversationContextStore")
        _validate_protocol_identifier(session_id, "session_id")
        if not callable(id_factory) or not callable(clock):
            raise TypeError("id_factory and clock must be callable")
        if type(max_terminal_updates) is not int:
            raise TypeError("max_terminal_updates must be an exact integer")
        if not 1 <= max_terminal_updates <= 256:
            raise ValueError("max_terminal_updates must be between 1 and 256")
        if type(close_drain_timeout_ms) is not int:
            raise TypeError("close_drain_timeout_ms must be an exact integer")
        if not 1 <= close_drain_timeout_ms <= 60_000:
            raise ValueError("close_drain_timeout_ms must be between 1 and 60000")
        if type(max_protocol_operations) is not int:
            raise TypeError("max_protocol_operations must be an exact integer")
        if not 1 <= max_protocol_operations <= 1_000_000:
            raise ValueError("max_protocol_operations must be between 1 and 1000000")
        self._context = context
        self._session = session
        self._session_id = session_id
        self._id_factory = id_factory
        self._clock = clock
        self._sequence = 0
        self._max_protocol_operations = max_protocol_operations
        self._used_event_ids: set[str] = set()
        self._used_inbound_event_ids: set[str] = set()
        self._used_inbound_sequences: set[int] = set()
        self._max_inbound_sequence = -1
        self._terminal_sequences: dict[str, int] = {}
        self._owned_operations: set[asyncio.Task[TaskOperationOutcome]] = set()
        self._lock = asyncio.Lock()
        self._pending_dispatches: dict[str, _PendingDispatch] = {}
        self._active_run_ids: dict[str, str] = {}
        self._cancel_operations: dict[str, asyncio.Task[TaskCancelOutcome]] = {}
        self._preack_terminals: dict[str, _TerminalEvidence] = {}
        self._terminal_updates: asyncio.Queue[TaskTerminalOutcome | BaseException] = (
            asyncio.Queue(maxsize=max_terminal_updates)
        )
        self._terminal_loop: asyncio.Task[None] | None = None
        self._terminal_error: BaseException | None = None
        self._close_drain_timeout = close_drain_timeout_ms / 1000
        self._close_operation: asyncio.Task[None] | None = None
        self._detached_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    def start(self) -> None:
        """Start the single controller-owned terminal consumer."""

        if self._terminal_error is not None and self._terminal_updates.empty():
            raise self._terminal_error
        if self._closed:
            raise RuntimeError("conversation task controller is closed")
        if self._terminal_loop is None:
            self._terminal_loop = asyncio.create_task(
                self._consume_terminals(),
                name=f"conversation-task-terminals:{self._session_id}",
            )

    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        async with self._lock:
            self._raise_if_unavailable()
            self._raise_if_operation_budget_exhausted()
            _validate_protocol_identifier(utterance_id, "utterance_id")
            if _PRIVATE_RUN_TOKEN_PATTERN.search(utterance_id) is not None:
                raise PrivateRunDisclosureError(
                    "model-visible text contains a private run token"
                )
            task_id = self._next_identifier("task_")
            admission = self._context.prepare_task(task_id, objective)
            try:
                event_id = self._next_identifier("evt_")
                self._claim_event_id_locked(event_id)
                timestamp = self._clock()
                request = WorkDispatchRequestedEvent(
                    type="work.dispatch.requested",
                    event_id=event_id,
                    session_id=self._session_id,
                    sequence=self._sequence,
                    timestamp=timestamp,
                    task_id=task_id,
                    utterance_id=utterance_id,
                    payload=WorkDispatchRequestedPayload(objective=objective),
                )
            except BaseException:
                self._context.discard_task(admission)
                raise
            self._sequence += 1
            self._pending_dispatches[task_id] = _PendingDispatch(admission=admission)
            self.start()
            operation = asyncio.create_task(
                self._settle_dispatch(request),
                name=f"conversation-task-dispatch:{task_id}",
            )
            self._owned_operations.add(operation)
            operation.add_done_callback(self._consume_operation)
        return await asyncio.shield(operation)

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        async with self._lock:
            self._raise_if_unavailable()
            _validate_task_id(task_id)
            _validate_outcome_text(reason, "reason")
            existing = self._cancel_operations.get(task_id)
            if existing is not None:
                operation = existing
            else:
                self._context.require_active_task(task_id)
                run_id = self._active_run_ids.get(task_id)
                if run_id is None:
                    raise RuntimeError("task is not owned by this controller generation")
                self._raise_if_operation_budget_exhausted()
                event_id = self._next_identifier("evt_")
                self._claim_event_id_locked(event_id)
                request = ControlCancelEvent(
                    type="control.cancel",
                    event_id=event_id,
                    session_id=self._session_id,
                    sequence=self._sequence,
                    timestamp=self._clock(),
                    task_id=task_id,
                    payload=ControlCancelPayload(scope=CancelScope.TASK, reason=reason),
                )
                self._sequence += 1
                operation = asyncio.create_task(
                    self._settle_cancel(request, expected_run_id=run_id),
                    name=f"conversation-task-cancel:{task_id}",
                )
                self._cancel_operations[task_id] = operation
                self._owned_operations.add(operation)
                operation.add_done_callback(self._consume_operation)
        return await asyncio.shield(operation)

    async def next_completion(self) -> TaskTerminalOutcome:
        """Return the next controller-validated public terminal update."""

        self.start()
        if self._terminal_error is not None and self._terminal_updates.empty():
            raise self._terminal_error
        update = await self._terminal_updates.get()
        if isinstance(update, BaseException):
            if self._terminal_error is update and not self._terminal_updates.full():
                self._terminal_updates.put_nowait(update)
            raise update
        return update

    async def close(self) -> None:
        """Close through one caller-cancellation-resistant bounded operation."""

        operation = self._close_operation
        if operation is None:
            operation = asyncio.create_task(
                self._close_owned(),
                name=f"conversation-task-close:{self._session_id}",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    async def _close_owned(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._fail_locked(RuntimeError("conversation task controller is closed"))
            self._closed = True
            terminal_loop, self._terminal_loop = self._terminal_loop, None
        session_close = asyncio.create_task(
            self._session.close(),
            name=f"conversation-task-session-close:{self._session_id}",
        )
        await self._bounded_cancel_and_drain((session_close,))
        if terminal_loop is not None:
            terminal_loop.cancel()
            await self._bounded_cancel_and_drain((terminal_loop,))
        await self._bounded_cancel_and_drain(tuple(self._owned_operations))

    async def _bounded_cancel_and_drain(
        self,
        tasks: tuple[asyncio.Task[Any], ...],
    ) -> None:
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=self._close_drain_timeout)
        self._retrieve_task_failures(done)
        if not pending:
            return
        for task in pending:
            task.cancel()
        done, pending = await asyncio.wait(pending, timeout=self._close_drain_timeout)
        self._retrieve_task_failures(done)
        for task in pending:
            self._detached_tasks.add(task)
            task.add_done_callback(self._consume_detached_task)

    async def _settle_dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> TaskDispatchOutcome:
        task_id = request.task_id
        request_sequence = request.sequence
        try:
            acknowledgment = await self._session.dispatch(request)
        except BaseException as exc:
            async with self._lock:
                if not self._closed:
                    self._fail_locked(exc)
            raise
        async with self._lock:
            if self._closed:
                raise RuntimeError(
                    "controller closed before dispatch acknowledgment could settle"
                )
            pending = self._pending_dispatches.get(task_id)
            if pending is None:
                error = RuntimeError("dispatch acknowledgment had no pending reservation")
                self._fail_locked(error)
                raise error
            try:
                self._validate_dispatch_acknowledgment(acknowledgment)
            except BaseException as exc:
                self._fail_locked(exc)
                raise
            if (
                acknowledgment.task_id != task_id
                or acknowledgment.session_id != self._session_id
            ):
                error = RuntimeError("dispatch acknowledgment does not match its request")
                self._fail_locked(error)
                raise error
            preack = self._preack_terminals.get(task_id)
            if acknowledgment.sequence <= request_sequence or (
                preack is not None
                and preack.event_sequence <= acknowledgment.sequence
            ):
                error = RuntimeError(
                    "dispatch acknowledgment sequence contradicted request or terminal ordering"
                )
                self._fail_locked(error)
                raise error
            self._claim_inbound_event_locked(
                acknowledgment,
                reordered_after=None if preack is None else preack.event_sequence,
            )
            if not acknowledgment.payload.accepted:
                if preack is not None:
                    error = RuntimeError(
                        "terminal evidence contradicted a rejected dispatch acknowledgment"
                    )
                    self._fail_locked(error)
                    raise error
                self._context.discard_task(pending.admission)
                del self._pending_dispatches[task_id]
                return TaskDispatchOutcome(
                    task_id=task_id,
                    accepted=False,
                    reason=acknowledgment.payload.reason,
                )
            run_id = acknowledgment.payload.run_id
            if run_id is None:
                error = RuntimeError(
                    "accepted dispatch acknowledgment omitted its run identity"
                )
                self._fail_locked(error)
                raise error
            try:
                self._context.record_reserved_task_accepted(
                    admission=pending.admission,
                    run_id=run_id,
                )
                del self._pending_dispatches[task_id]
                self._active_run_ids[task_id] = run_id
                preack = self._preack_terminals.pop(task_id, None)
                if preack is not None:
                    self._settle_terminal_locked(preack)
            except BaseException as exc:
                self._fail_locked(exc)
                raise
            return TaskDispatchOutcome(task_id=task_id, accepted=True)

    @staticmethod
    def _validate_dispatch_acknowledgment(
        acknowledgment: WorkDispatchAcknowledgedEvent,
    ) -> None:
        if type(acknowledgment) is not WorkDispatchAcknowledgedEvent:
            raise TypeError("dispatch acknowledgment must be exact")
        _validate_event_envelope(
            acknowledgment,
            "work.dispatch.acknowledged",
        )
        if type(acknowledgment.session_id) is not str:
            raise TypeError("dispatch acknowledgment session identity must be exact")
        if type(acknowledgment.task_id) is not str:
            raise TypeError("dispatch acknowledgment task identity must be exact")
        if type(acknowledgment.payload) is not WorkDispatchAcknowledgedPayload:
            raise TypeError("dispatch acknowledgment payload must be exact")
        if type(acknowledgment.payload.accepted) is not bool:
            raise TypeError("dispatch acceptance must be an exact boolean")
        run_id = acknowledgment.payload.run_id
        reason = acknowledgment.payload.reason
        if run_id is not None:
            _validate_run_id(run_id)
        _validate_outcome_text(reason, "reason")
        if acknowledgment.payload.accepted:
            if run_id is None or reason is not None:
                raise RuntimeError(
                    "dispatch acknowledgment evidence contradicts acceptance"
                )
        elif run_id is not None or reason is None:
            raise RuntimeError(
                "dispatch acknowledgment evidence contradicts rejection"
            )

    @staticmethod
    def _validate_cancel_acknowledgment(
        acknowledgment: ControlCancelAcknowledgedEvent,
    ) -> None:
        if type(acknowledgment) is not ControlCancelAcknowledgedEvent:
            raise TypeError("cancellation acknowledgment must be exact")
        _validate_event_envelope(
            acknowledgment,
            "control.cancel.acknowledged",
        )
        if type(acknowledgment.session_id) is not str:
            raise TypeError("cancellation acknowledgment session identity must be exact")
        if type(acknowledgment.request_event_id) is not str:
            raise TypeError("cancellation request identity must be exact")
        if type(acknowledgment.task_id) is not str:
            raise TypeError("cancellation task identity must be exact")
        if type(acknowledgment.scope) is not CancelScope:
            raise TypeError("cancellation scope must be exact")
        if type(acknowledgment.payload) is not ControlCancelAcknowledgedPayload:
            raise TypeError("cancellation acknowledgment payload must be exact")
        if type(acknowledgment.payload.accepted) is not bool:
            raise TypeError("cancellation acceptance must be an exact boolean")
        if type(acknowledgment.payload.signaled_run_ids) is not list:
            raise TypeError("signaled run identities must use an exact list")
        signaled_run_ids = acknowledgment.payload.signaled_run_ids
        for run_id in signaled_run_ids:
            _validate_run_id(run_id)
        reason = acknowledgment.payload.reason
        _validate_outcome_text(reason, "reason")
        if acknowledgment.payload.accepted:
            if not signaled_run_ids or reason is not None:
                raise RuntimeError(
                    "cancellation acknowledgment evidence contradicts acceptance"
                )
        elif signaled_run_ids or reason is None:
            raise RuntimeError(
                "cancellation acknowledgment evidence contradicts rejection"
            )

    async def _settle_cancel(
        self,
        request: ControlCancelEvent,
        *,
        expected_run_id: str,
    ) -> TaskCancelOutcome:
        task_id = request.task_id
        if task_id is None:
            raise RuntimeError("task cancellation request omitted its task identity")
        request_event_id = request.event_id
        request_sequence = request.sequence
        acknowledgment = await self._session.cancel(request)
        async with self._lock:
            if self._closed:
                raise RuntimeError(
                    "controller closed before cancellation acknowledgment could settle"
                )
            try:
                self._validate_cancel_acknowledgment(acknowledgment)
            except BaseException as exc:
                self._fail_locked(exc)
                raise
            if (
                acknowledgment.session_id != self._session_id
                or acknowledgment.request_event_id != request_event_id
                or acknowledgment.scope is not CancelScope.TASK
                or acknowledgment.task_id != task_id
            ):
                error = RuntimeError(
                    "cancellation acknowledgment does not match its request"
                )
                self._fail_locked(error)
                raise error
            if acknowledgment.sequence <= request_sequence:
                error = RuntimeError(
                    "cancellation acknowledgment sequence did not follow its request"
                )
                self._fail_locked(error)
                raise error
            self._claim_inbound_event_locked(
                acknowledgment,
                reordered_after=self._terminal_sequences.get(task_id),
            )
            if not acknowledgment.payload.accepted:
                self._cancel_operations.pop(task_id, None)
                return TaskCancelOutcome(
                    task_id=task_id,
                    accepted=False,
                    reason=acknowledgment.payload.reason,
                )
            signaled_run_ids = acknowledgment.payload.signaled_run_ids
            if len(signaled_run_ids) != 1 or signaled_run_ids[0] != expected_run_id:
                error = RuntimeError(
                    "cancellation acknowledgment did not contain exact active run authority"
                )
                self._fail_locked(error)
                raise error
            return TaskCancelOutcome(task_id=task_id, accepted=True)

    async def _consume_terminals(self) -> None:
        try:
            while True:
                update = await self._session.next_update()
                async with self._lock:
                    if self._closed:
                        return
                    self._accept_terminal_locked(update)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            async with self._lock:
                if not self._closed:
                    self._fail_locked(exc)

    def _accept_terminal_locked(self, update: WorkCompletedEvent) -> None:
        if type(update) is not WorkCompletedEvent:
            raise TypeError("terminal update must be an exact WorkCompletedEvent")
        _validate_event_envelope(update, "work.completed")
        if type(update.session_id) is not str or update.session_id != self._session_id:
            raise RuntimeError("terminal update does not belong to this session")
        if type(update.payload) is not WorkCompletedPayload:
            raise TypeError("terminal payload must be exact")
        if type(update.payload.status) is not WorkTerminalStatus:
            raise TypeError("terminal status must be exact")
        _validate_task_id(update.task_id)
        _validate_run_id(update.run_id)
        summary = update.payload.summary
        reason = update.payload.reason
        _validate_outcome_text(summary, "summary")
        _validate_outcome_text(reason, "reason")
        evidence = _TerminalEvidence(
            task_id=update.task_id,
            run_id=update.run_id,
            event_sequence=update.sequence,
            outcome=TaskTerminalOutcome(
                task_id=update.task_id,
                status=update.payload.status.value,
                summary=summary,
                reason=reason,
            ),
        )
        self._claim_inbound_event_locked(update)
        if evidence.task_id in self._pending_dispatches:
            if evidence.task_id in self._preack_terminals:
                raise RuntimeError("pending dispatch received multiple terminal updates")
            self._preack_terminals[evidence.task_id] = evidence
            return
        self._settle_terminal_locked(evidence)

    def _settle_terminal_locked(self, evidence: _TerminalEvidence) -> None:
        active_run_id = self._active_run_ids.get(evidence.task_id)
        if active_run_id is None or active_run_id != evidence.run_id:
            raise RuntimeError("terminal update does not match active task authority")
        if self._terminal_updates.full():
            raise RuntimeError("terminal outcome capacity exhausted")
        self._context.record_task_completed(
            task_id=evidence.task_id,
            run_id=evidence.run_id,
        )
        self._terminal_sequences[evidence.task_id] = evidence.event_sequence
        del self._active_run_ids[evidence.task_id]
        self._cancel_operations.pop(evidence.task_id, None)
        self._terminal_updates.put_nowait(evidence.outcome)

    def _fail_locked(self, exc: BaseException) -> None:
        if self._terminal_error is None:
            self._terminal_error = exc
        if self._terminal_updates.empty():
            self._terminal_updates.put_nowait(self._terminal_error)

    def _raise_if_unavailable(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._closed:
            raise RuntimeError("conversation task controller is closed")

    def _claim_inbound_event_locked(
        self,
        event: Any,
        *,
        reordered_after: int | None = None,
    ) -> None:
        event_id = event.event_id
        sequence = event.sequence
        if event_id in self._used_inbound_event_ids:
            error = RuntimeError("inbound event identity was reused")
            self._fail_locked(error)
            raise error
        if sequence in self._used_inbound_sequences:
            error = RuntimeError("inbound event sequence was reused")
            self._fail_locked(error)
            raise error
        if sequence <= self._max_inbound_sequence and (
            reordered_after is None or sequence >= reordered_after
        ):
            error = RuntimeError("inbound event sequence was stale")
            self._fail_locked(error)
            raise error
        self._used_inbound_event_ids.add(event_id)
        self._used_inbound_sequences.add(sequence)
        self._max_inbound_sequence = max(self._max_inbound_sequence, sequence)

    def _claim_event_id_locked(self, event_id: str) -> None:
        if event_id in self._used_event_ids:
            error = RuntimeError("generated event identity was reused")
            self._fail_locked(error)
            raise error
        self._used_event_ids.add(event_id)

    def _raise_if_operation_budget_exhausted(self) -> None:
        if self._sequence >= self._max_protocol_operations:
            raise RuntimeError("conversation task operation budget exhausted")

    @staticmethod
    def _retrieve_task_failures(tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            if not task.cancelled():
                task.exception()

    def _consume_detached_task(self, task: asyncio.Task[Any]) -> None:
        self._detached_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _consume_operation(self, operation: asyncio.Task[TaskOperationOutcome]) -> None:
        self._owned_operations.discard(operation)
        if not operation.cancelled():
            operation.exception()

    def _next_identifier(self, prefix: str) -> str:
        suffix = self._id_factory()
        if type(suffix) is not str:
            raise TypeError("generated identifier suffix must be an exact built-in string")
        value = f"{prefix}{suffix}"
        _validate_protocol_identifier(value, "generated identifier")
        return value
