"""Provider-neutral authoritative controls for conversation background work."""

from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar, cast

from .context import (
    ActiveTaskIdentityError,
    ConversationContextStore,
    PrivateRunDisclosureError,
)
from .tasks import TaskCancelOutcome, TaskDispatchOutcome

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_TASK_ID = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_PRIVATE_AUTHORITY = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*")
_MAX_IDENTIFIER_CHARS = 128
_MAX_REASON_CHARS = 1024
_UNAVAILABLE_REASON = "work control unavailable; restart required"
_NO_ACTIVE_REASON = "no active work"
_CAPACITY_REASON = "background task capacity exhausted"
_AMBIGUOUS_ACTIVE_REASON = "multiple active tasks; specify task_id"
_DUPLICATE_OBJECTIVE_REASON = "matching work is already active"
_PublicValue = str | int | bool | None
_Observer = Callable[[str, dict[str, _PublicValue]], None]
_StartProjectionGate = Callable[["WorkStartResult"], Awaitable[None]]


class WorkControlHealth(StrEnum):
    """Admission health for one non-resettable work-control generation."""

    OPEN = "open"
    UNCERTAIN = "uncertain"
    CLOSED = "closed"


def _validate_exact_text(value: str, field: str, *, maximum: int) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be an exact built-in string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{field} exceeds its configured maximum")
    if _PRIVATE_AUTHORITY.search(value) is not None:
        raise PrivateRunDisclosureError("model-visible text contains a private authority token")


def _validate_identifier(value: str, field: str) -> None:
    _validate_exact_text(value, field, maximum=_MAX_IDENTIFIER_CHARS)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical protocol identifier")


def _validate_task_id(value: str) -> None:
    _validate_identifier(value, "task_id")
    if _TASK_ID.fullmatch(value) is None:
        raise ValueError("task_id must use the bounded task_ namespace")


def _validate_reason(value: str | None) -> None:
    if value is None:
        return
    _validate_exact_text(value, "reason", maximum=_MAX_REASON_CHARS)


def _objective_identity(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True, slots=True)
class WorkStartResult:
    """A reconstructively validated, model-safe start result."""

    accepted: bool
    state: str
    task_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact built-in boolean")
        if type(self.state) is not str:
            raise TypeError("state must be an exact built-in string")
        if self.state not in ("active", "cancelling", "rejected"):
            raise ValueError("start state is unsupported")
        if self.task_id is not None:
            _validate_task_id(self.task_id)
        _validate_reason(self.reason)
        if self.accepted:
            if (
                self.state not in ("active", "cancelling")
                or self.task_id is None
                or self.reason is not None
            ):
                raise ValueError("accepted start result is contradictory")
        elif self.state != "rejected" or self.reason is None:
            raise ValueError("rejected start result requires a bounded reason")


@dataclass(frozen=True, slots=True)
class WorkCancelResult:
    """A reconstructively validated, model-safe cancellation result."""

    accepted: bool
    state: str
    task_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact built-in boolean")
        if type(self.state) is not str:
            raise TypeError("state must be an exact built-in string")
        if self.state not in ("cancelling", "rejected"):
            raise ValueError("cancel state is unsupported")
        if self.task_id is not None:
            _validate_task_id(self.task_id)
        _validate_reason(self.reason)
        if self.accepted:
            if self.state != "cancelling" or self.task_id is None or self.reason is not None:
                raise ValueError("accepted cancel result is contradictory")
        elif self.state != "rejected" or self.reason is None:
            raise ValueError("rejected cancel result requires a bounded reason")


class _TaskController(Protocol):
    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome: ...

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome: ...


@dataclass(slots=True)
class _PendingStart:
    invocation_id: str
    objective_identity: str
    cancel_after_ack: bool
    cancel_result: asyncio.Future[WorkCancelResult]
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class _InvocationBinding:
    semantics: tuple[str, ...]
    operation: asyncio.Task[Any]


_ResultT = TypeVar("_ResultT", WorkStartResult, WorkCancelResult)


class _ProjectionRolledBack(BaseException):
    def __init__(self, error: BaseException) -> None:
        self.error = error


class ConversationWorkControlSurface:
    """Own validation, replay, projection, and rollback around task authority."""

    def __init__(
        self,
        *,
        controller: _TaskController,
        context: ConversationContextStore,
        observer: _Observer,
        reserve_observer_capacity: Callable[[], None],
        utterance_id_factory: Callable[[], str] | None = None,
        start_projection_gate: _StartProjectionGate | None = None,
        max_invocations: int = 256,
        close_drain_timeout_ms: int = 5000,
    ) -> None:
        if not callable(getattr(controller, "dispatch", None)):
            raise TypeError("controller must provide dispatch()")
        if not callable(getattr(controller, "request_cancel", None)):
            raise TypeError("controller must provide request_cancel()")
        if type(context) is not ConversationContextStore:
            raise TypeError("context must be an exact ConversationContextStore")
        if not callable(observer):
            raise TypeError("observer must be callable")
        if not callable(reserve_observer_capacity):
            raise TypeError("reserve_observer_capacity must be callable")
        if utterance_id_factory is not None and not callable(utterance_id_factory):
            raise TypeError("utterance_id_factory must be callable")
        if start_projection_gate is not None and not callable(start_projection_gate):
            raise TypeError("start_projection_gate must be callable")
        if type(max_invocations) is not int:
            raise TypeError("max_invocations must be an exact built-in integer")
        if not 1 <= max_invocations <= 4096:
            raise ValueError("max_invocations must be between 1 and 4096")
        if type(close_drain_timeout_ms) is not int:
            raise TypeError("close_drain_timeout_ms must be an exact built-in integer")
        if not 1 <= close_drain_timeout_ms <= 60000:
            raise ValueError("close_drain_timeout_ms must be between 1 and 60000")
        self._controller = controller
        self._context = context
        self._observer = observer
        self._reserve = reserve_observer_capacity
        self._utterance_id_factory = utterance_id_factory or (lambda: secrets.token_hex(12))
        self._start_projection_gate = start_projection_gate
        self._max_invocations = max_invocations
        self._close_drain_timeout_seconds = close_drain_timeout_ms / 1000.0
        self._health = WorkControlHealth.OPEN
        self._lock = asyncio.Lock()
        self._bindings: dict[str, _InvocationBinding] = {}
        self._owned_operations: set[asyncio.Task[Any]] = set()
        self._pending_starts: dict[str, _PendingStart] = {}
        self._close_operation: asyncio.Task[None] | None = None

    @property
    def health(self) -> WorkControlHealth:
        return self._health

    @property
    def max_objective_chars(self) -> int:
        return self._context.max_item_chars

    @property
    def can_cancel_work(self) -> bool:
        return bool(self._pending_starts) or bool(self._context.snapshot().active_tasks)

    @staticmethod
    def _logical_work_count(
        active_tasks: tuple[Any, ...],
        pending_starts: tuple[_PendingStart, ...],
    ) -> int:
        active_task_ids = {task.task_id for task in active_tasks}
        return len(active_task_ids) + sum(
            pending.task_id is None or pending.task_id not in active_task_ids
            for pending in pending_starts
        )

    async def start_work(
        self,
        *,
        objective: str,
        invocation_id: str,
    ) -> WorkStartResult:
        self._validate_objective(objective)
        _validate_identifier(invocation_id, "invocation_id")
        semantics = ("start", objective)
        async with self._lock:
            unavailable = self._unavailable_start_locked()
            if unavailable is not None:
                return unavailable
            existing = self._bindings.get(invocation_id)
            if existing is not None:
                if existing.semantics != semantics:
                    self._mark_uncertain_locked()
                    return self._unavailable_start()
                operation = cast(asyncio.Task[WorkStartResult], existing.operation)
            else:
                if not self._admit_binding_locked():
                    return self._unavailable_start()
                active_tasks = self._context.snapshot().active_tasks
                objective_identity = _objective_identity(objective)
                matching_work = any(
                    _objective_identity(task.objective) == objective_identity
                    for task in active_tasks
                ) or any(
                    pending.objective_identity == objective_identity
                    for pending in self._pending_starts.values()
                )
                if matching_work:
                    self._reserve()
                    operation = self._create_operation_locked(
                        self._project_start(
                            WorkStartResult(
                                accepted=False,
                                state="rejected",
                                reason=_DUPLICATE_OBJECTIVE_REASON,
                            )
                        ),
                        name=f"conversation-work-start-duplicate:{invocation_id}",
                    )
                elif (
                    self._logical_work_count(
                        active_tasks,
                        tuple(self._pending_starts.values()),
                    )
                    >= self._context.max_active_tasks
                ):
                    self._reserve()
                    operation = self._create_operation_locked(
                        self._project_start(
                            WorkStartResult(
                                accepted=False,
                                state="rejected",
                                reason=_CAPACITY_REASON,
                            )
                        ),
                        name=f"conversation-work-start-rejected:{invocation_id}",
                    )
                else:
                    utterance_id = self._new_utterance_id()
                    self._reserve()
                    pending = _PendingStart(
                        invocation_id=invocation_id,
                        objective_identity=objective_identity,
                        cancel_after_ack=False,
                        cancel_result=asyncio.get_running_loop().create_future(),
                    )
                    self._pending_starts[invocation_id] = pending
                    operation = self._create_operation_locked(
                        self._settle_start(
                            objective=objective,
                            utterance_id=utterance_id,
                            pending=pending,
                        ),
                        name=f"conversation-work-start:{invocation_id}",
                    )
                self._bindings[invocation_id] = _InvocationBinding(
                    semantics=semantics,
                    operation=operation,
                )
        return self._copy_start(await asyncio.shield(operation))

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        _validate_identifier(invocation_id, "invocation_id")
        return await self._cancel_work(
            invocation_id=invocation_id,
            expected_task_id=None,
            reason="user requested task cancellation",
            semantic_kind="cancel_active",
        )

    async def cancel_work(
        self,
        *,
        task_id: str,
        invocation_id: str,
        reason: str = "user requested task cancellation",
    ) -> WorkCancelResult:
        """Cancel an exact public task for the deterministic command fallback."""

        _validate_task_id(task_id)
        _validate_identifier(invocation_id, "invocation_id")
        _validate_reason(reason)
        return await self._cancel_work(
            invocation_id=invocation_id,
            expected_task_id=task_id,
            reason=reason,
            semantic_kind="cancel_exact",
        )

    async def close(self) -> None:
        operation = self._close_operation
        if self._close_needs_retry(operation):
            async with self._lock:
                operation = self._close_operation
                if self._close_needs_retry(operation):
                    self._health = WorkControlHealth.CLOSED
                    operation = asyncio.create_task(
                        self._close_owned(),
                        name="conversation-work-control-close",
                    )
                    self._close_operation = operation
        assert operation is not None
        await asyncio.shield(operation)

    @staticmethod
    def _close_needs_retry(operation: asyncio.Task[None] | None) -> bool:
        if operation is None:
            return True
        if not operation.done():
            return False
        return operation.cancelled() or operation.exception() is not None

    async def _cancel_work(
        self,
        *,
        invocation_id: str,
        expected_task_id: str | None,
        reason: str,
        semantic_kind: str,
    ) -> WorkCancelResult:
        semantics = (
            semantic_kind,
            "" if expected_task_id is None else expected_task_id,
            reason,
        )
        async with self._lock:
            unavailable = self._unavailable_cancel_locked()
            if unavailable is not None:
                return unavailable
            existing = self._bindings.get(invocation_id)
            if existing is not None:
                if existing.semantics != semantics:
                    self._mark_uncertain_locked()
                    return self._unavailable_cancel()
                operation = cast(asyncio.Task[WorkCancelResult], existing.operation)
            else:
                if not self._admit_binding_locked():
                    return self._unavailable_cancel()
                active_tasks = self._context.snapshot().active_tasks
                pending_starts = tuple(self._pending_starts.values())
                if expected_task_id is not None:
                    matching = tuple(
                        task for task in active_tasks if task.task_id == expected_task_id
                    )
                    if not matching:
                        operation = self._new_cancel_rejection_locked(
                            invocation_id,
                            task_id=expected_task_id,
                            reason="task is not active",
                        )
                    else:
                        self._reserve()
                        operation = self._create_operation_locked(
                            self._settle_cancel(task_id=expected_task_id, reason=reason),
                            name=f"conversation-work-cancel:{invocation_id}",
                        )
                elif self._logical_work_count(active_tasks, pending_starts) == 0:
                    operation = self._new_cancel_rejection_locked(
                        invocation_id,
                        task_id=None,
                        reason=_NO_ACTIVE_REASON,
                    )
                elif self._logical_work_count(active_tasks, pending_starts) > 1:
                    operation = self._new_cancel_rejection_locked(
                        invocation_id,
                        task_id=None,
                        reason=_AMBIGUOUS_ACTIVE_REASON,
                    )
                elif pending_starts:
                    pending = pending_starts[0]
                    pending.cancel_after_ack = True
                    operation = self._create_operation_locked(
                        self._await_pending_cancel(pending),
                        name=f"conversation-work-pending-cancel:{invocation_id}",
                    )
                else:
                    task_id = active_tasks[0].task_id
                    self._reserve()
                    operation = self._create_operation_locked(
                        self._settle_cancel(task_id=task_id, reason=reason),
                        name=f"conversation-work-cancel:{invocation_id}",
                    )
                self._bindings[invocation_id] = _InvocationBinding(
                    semantics=semantics,
                    operation=operation,
                )
        return self._copy_cancel(await asyncio.shield(operation))

    async def _settle_start(
        self,
        *,
        objective: str,
        utterance_id: str,
        pending: _PendingStart,
    ) -> WorkStartResult:
        try:
            dispatch = await self._controller.dispatch(
                objective=objective,
                utterance_id=utterance_id,
            )
            if type(dispatch) is not TaskDispatchOutcome:
                raise TypeError("task controller returned the wrong dispatch outcome")
        except asyncio.CancelledError:
            async with self._lock:
                self._mark_uncertain_locked()
            self._settle_pending_cancel_result(
                pending,
                self._unavailable_cancel(),
            )
            raise
        except BaseException:
            async with self._lock:
                self._mark_uncertain_locked()
            self._settle_pending_cancel_result(
                pending,
                self._unavailable_cancel(),
            )
            raise
        try:
            if not dispatch.accepted:
                start_result = WorkStartResult(
                    accepted=False,
                    state="rejected",
                    task_id=dispatch.task_id,
                    reason=dispatch.reason or "Hermes rejected task dispatch",
                )
                self._settle_pending_cancel_result(
                    pending,
                    WorkCancelResult(
                        accepted=False,
                        state="rejected",
                        task_id=dispatch.task_id,
                        reason=dispatch.reason or "Hermes rejected task dispatch",
                    ),
                )
                return await self._project_start(start_result)

            async with self._lock:
                pending.task_id = dispatch.task_id
                cancel_after_ack = pending.cancel_after_ack
                if not cancel_after_ack:
                    self._remove_pending_start_locked(pending)
            if cancel_after_ack:
                try:
                    cancellation = await self._request_cancel(
                        dispatch.task_id,
                        reason="user requested task cancellation",
                    )
                except ActiveTaskIdentityError:
                    cancel_result = WorkCancelResult(
                        accepted=False,
                        state="rejected",
                        task_id=dispatch.task_id,
                        reason="task is not active",
                    )
                    self._settle_pending_cancel_result(pending, cancel_result)
                    return await self._project_start(
                        WorkStartResult(
                            accepted=False,
                            state="rejected",
                            task_id=dispatch.task_id,
                            reason="task is not active",
                        )
                    )
                except BaseException:
                    async with self._lock:
                        self._mark_uncertain_locked()
                    self._settle_pending_cancel_result(
                        pending,
                        self._unavailable_cancel(),
                    )
                    raise
                cancel_result = self._cancel_result(cancellation)
                self._settle_pending_cancel_result(pending, cancel_result)
                if cancel_result.accepted:
                    return await self._project_start(
                        WorkStartResult(
                            accepted=True,
                            state="cancelling",
                            task_id=dispatch.task_id,
                        )
                    )
            return await self._project_accepted_start(dispatch.task_id)
        except _ProjectionRolledBack as rolled_back:
            raise rolled_back.error from None
        except asyncio.CancelledError:
            async with self._lock:
                self._mark_uncertain_locked()
            self._settle_pending_cancel_result(
                pending,
                self._unavailable_cancel(),
            )
            raise
        except BaseException:
            async with self._lock:
                self._mark_uncertain_locked()
            self._settle_pending_cancel_result(
                pending,
                self._unavailable_cancel(),
            )
            raise
        finally:
            async with self._lock:
                self._remove_pending_start_locked(pending)

    async def _project_accepted_start(self, task_id: str) -> WorkStartResult:
        result = WorkStartResult(
            accepted=True,
            state="active",
            task_id=task_id,
        )
        try:
            return await self._project_start(result)
        except BaseException as projection_error:
            try:
                rollback = await self._request_cancel(
                    task_id,
                    reason="task state projection failed",
                )
                if not rollback.accepted:
                    raise RuntimeError("task projection rollback was rejected")
            except ActiveTaskIdentityError:
                pass
            except BaseException as rollback_error:
                async with self._lock:
                    self._mark_uncertain_locked()
                raise BaseExceptionGroup(
                    "task state projection and rollback failed",
                    [projection_error, rollback_error],
                ) from None
            raise _ProjectionRolledBack(projection_error) from None

    async def _settle_cancel(self, *, task_id: str, reason: str) -> WorkCancelResult:
        try:
            outcome = await self._request_cancel(task_id, reason=reason)
        except ActiveTaskIdentityError:
            return await self._project_cancel(
                WorkCancelResult(
                    accepted=False,
                    state="rejected",
                    task_id=task_id,
                    reason="task is not active",
                )
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            async with self._lock:
                self._mark_uncertain_locked()
            raise
        result = self._cancel_result(outcome)
        try:
            return await self._project_cancel(result)
        except BaseException:
            if result.accepted:
                async with self._lock:
                    self._mark_uncertain_locked()
            raise

    async def _request_cancel(self, task_id: str, *, reason: str) -> TaskCancelOutcome:
        outcome = await self._controller.request_cancel(task_id, reason=reason)
        if type(outcome) is not TaskCancelOutcome:
            raise TypeError("task controller returned the wrong cancel outcome")
        return outcome

    async def _await_pending_cancel(
        self,
        pending: _PendingStart,
    ) -> WorkCancelResult:
        return await asyncio.shield(pending.cancel_result)

    async def _project_start(self, result: WorkStartResult) -> WorkStartResult:
        copied = self._copy_start(result)
        gate = self._start_projection_gate
        if gate is not None:
            await gate(self._copy_start(copied))
        if copied.accepted and copied.state == "active":
            active_tasks = self._context.snapshot().active_tasks
            if not active_tasks:
                raise ActiveTaskIdentityError(
                    "task reached terminal state before active projection"
                )
            if copied.task_id not in {task.task_id for task in active_tasks}:
                raise RuntimeError("active task identity changed before projection")
        data: dict[str, _PublicValue] = {
            "status": copied.state,
            "taskId": copied.task_id,
        }
        if copied.reason is not None:
            data["reason"] = copied.reason
        self._observer("task_state", data)
        return self._copy_start(copied)

    async def _project_cancel(self, result: WorkCancelResult) -> WorkCancelResult:
        copied = self._copy_cancel(result)
        data: dict[str, _PublicValue] = {
            "status": copied.state,
            "taskId": copied.task_id,
        }
        if copied.reason is not None:
            data["reason"] = copied.reason
        self._observer("task_state", data)
        return self._copy_cancel(copied)

    def _new_cancel_rejection_locked(
        self,
        invocation_id: str,
        *,
        task_id: str | None,
        reason: str,
    ) -> asyncio.Task[WorkCancelResult]:
        self._reserve()
        return cast(
            asyncio.Task[WorkCancelResult],
            self._create_operation_locked(
                self._project_cancel(
                    WorkCancelResult(
                        accepted=False,
                        state="rejected",
                        task_id=task_id,
                        reason=reason,
                    )
                ),
                name=f"conversation-work-cancel-rejected:{invocation_id}",
            ),
        )

    def _create_operation_locked(
        self,
        coroutine: Coroutine[Any, Any, _ResultT],
        *,
        name: str,
    ) -> asyncio.Task[_ResultT]:
        operation = asyncio.create_task(
            coroutine,
            name=name,
        )
        self._owned_operations.add(operation)
        operation.add_done_callback(self._consume_operation)
        return operation

    def _consume_operation(self, operation: asyncio.Task[Any]) -> None:
        self._owned_operations.discard(operation)
        if not operation.cancelled():
            operation.exception()

    async def _close_owned(self) -> None:
        while True:
            async with self._lock:
                operations = tuple(self._owned_operations)
            if not operations:
                return
            _done, pending = await asyncio.wait(
                operations,
                timeout=self._close_drain_timeout_seconds,
            )
            if pending:
                raise TimeoutError("work-control close drain timed out")

    def _remove_pending_start_locked(self, pending: _PendingStart) -> None:
        current = self._pending_starts.get(pending.invocation_id)
        if current is pending:
            del self._pending_starts[pending.invocation_id]

    def _new_utterance_id(self) -> str:
        token = self._utterance_id_factory()
        _validate_identifier(token, "utterance identifier token")
        utterance_id = f"utterance_{token}"
        _validate_identifier(utterance_id, "utterance_id")
        return utterance_id

    def _validate_objective(self, objective: str) -> None:
        _validate_exact_text(
            objective,
            "objective",
            maximum=self.max_objective_chars,
        )

    def _admit_binding_locked(self) -> bool:
        if len(self._bindings) >= self._max_invocations:
            for invocation_id, binding in tuple(self._bindings.items()):
                if binding.operation.done():
                    del self._bindings[invocation_id]
                    break
        return len(self._bindings) < self._max_invocations

    def _unavailable_start_locked(self) -> WorkStartResult | None:
        if self._health is WorkControlHealth.OPEN:
            return None
        return self._unavailable_start()

    def _unavailable_cancel_locked(self) -> WorkCancelResult | None:
        if self._health is WorkControlHealth.OPEN:
            return None
        return self._unavailable_cancel()

    def _mark_uncertain_locked(self) -> None:
        if self._health is WorkControlHealth.OPEN:
            self._health = WorkControlHealth.UNCERTAIN

    @staticmethod
    def _settle_pending_cancel_result(
        pending: _PendingStart,
        result: WorkCancelResult,
    ) -> None:
        if not pending.cancel_result.done():
            pending.cancel_result.set_result(ConversationWorkControlSurface._copy_cancel(result))

    @staticmethod
    def _cancel_result(outcome: TaskCancelOutcome) -> WorkCancelResult:
        return WorkCancelResult(
            accepted=outcome.accepted,
            state="cancelling" if outcome.accepted else "rejected",
            task_id=outcome.task_id,
            reason=(
                None if outcome.accepted else outcome.reason or "Hermes rejected task cancellation"
            ),
        )

    @staticmethod
    def _copy_start(result: WorkStartResult) -> WorkStartResult:
        if type(result) is not WorkStartResult:
            raise TypeError("work start result must be exact")
        return WorkStartResult(
            accepted=result.accepted,
            state=result.state,
            task_id=result.task_id,
            reason=result.reason,
        )

    @staticmethod
    def _copy_cancel(result: WorkCancelResult) -> WorkCancelResult:
        if type(result) is not WorkCancelResult:
            raise TypeError("work cancel result must be exact")
        return WorkCancelResult(
            accepted=result.accepted,
            state=result.state,
            task_id=result.task_id,
            reason=result.reason,
        )

    @staticmethod
    def _unavailable_start() -> WorkStartResult:
        return WorkStartResult(
            accepted=False,
            state="rejected",
            reason=_UNAVAILABLE_REASON,
        )

    @staticmethod
    def _unavailable_cancel() -> WorkCancelResult:
        return WorkCancelResult(
            accepted=False,
            state="rejected",
            reason=_UNAVAILABLE_REASON,
        )


__all__ = [
    "ConversationWorkControlSurface",
    "WorkCancelResult",
    "WorkControlHealth",
    "WorkStartResult",
]
