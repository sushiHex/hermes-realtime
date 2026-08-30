"""Truthful dispatch boundary between realtime events and Hermes work."""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    Durability,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
)

from .sequencing import EventSequencer
from .session import SessionBinding, SessionBindings


@dataclass(frozen=True, slots=True)
class HermesDispatchCommand:
    """Transport-neutral command submitted to a real Hermes dispatcher."""

    session_id: str
    task_id: str
    objective: str
    durability: Durability

    def __post_init__(self) -> None:
        for name, value in (("session_id", self.session_id), ("task_id", self.task_id)):
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
                raise ValueError(f"{name} contains unsupported identifier characters")


class HermesWorkDispatcher(Protocol):
    """Minimal execution primitive supplied by the Hermes plugin runtime."""

    async def dispatch(self, command: HermesDispatchCommand) -> str:
        """Accept work and return Hermes's authoritative run identifier."""


class HermesWorkCanceller(Protocol):
    """Exact-handle interruption primitive supplied by Hermes."""

    async def cancel(self, run_id: str) -> bool:
        """Return whether an interruption signal reached a running delegation."""


class HermesDispatchRejected(Exception):
    """Hermes explicitly declined to accept a dispatch command."""

    def __init__(self, reason: str) -> None:
        reason = reason.strip()
        if not reason:
            raise ValueError("dispatch rejection reason must not be blank")
        self.reason = reason
        super().__init__(reason)


class HermesIntegrationService:
    """Authorize and translate realtime dispatch requests into Hermes work."""

    def __init__(
        self,
        *,
        bindings: SessionBindings,
        dispatcher: HermesWorkDispatcher,
        canceller: HermesWorkCanceller | None = None,
        event_id_factory: Callable[[], str],
        clock: Callable[[], datetime],
        max_retained_dispatches: int = 256,
        max_active_dispatches: int = 32,
        sequencer: EventSequencer | None = None,
    ) -> None:
        if max_retained_dispatches < 1:
            raise ValueError("max_retained_dispatches must be positive")
        if max_active_dispatches < 1:
            raise ValueError("max_active_dispatches must be positive")
        self._bindings = bindings
        self._dispatcher = dispatcher
        self._canceller = canceller
        self._event_id_factory = event_id_factory
        self._clock = clock
        self._max_retained_dispatches = max_retained_dispatches
        self._max_active_dispatches = max_active_dispatches
        self._sequencer = sequencer or EventSequencer()
        self._dispatch_tasks: OrderedDict[
            tuple[str, int, str], asyncio.Task[WorkDispatchAcknowledgedEvent]
        ] = OrderedDict()
        self._dispatch_fingerprints: dict[tuple[str, int, str], tuple[object, ...]] = {}
        self._active_runs: dict[tuple[str, int, str], str] = {}
        self._active_run_ids_snapshot: frozenset[str] = frozenset()
        self._run_bindings: dict[str, SessionBinding] = {}
        self._run_keys: dict[str, tuple[str, int, str]] = {}
        self._expired_dispatches: OrderedDict[
            tuple[str, int, str], tuple[object, ...]
        ] = OrderedDict()
        self._cancellation_tasks: OrderedDict[
            tuple[str, int, str], asyncio.Task[ControlCancelAcknowledgedEvent]
        ] = OrderedDict()
        self._cancellation_fingerprints: dict[
            tuple[str, int, str], tuple[object, ...]
        ] = {}
        self._dispatch_lock = asyncio.Lock()


    async def dispatch(
        self,
        participant_id: str,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        binding = self._bindings.binding_for(participant_id)
        if binding is None or binding.session_id != request.session_id:
            raise PermissionError("participant is not bound to the requested Hermes session")

        key = (request.session_id, binding.generation, request.task_id)
        fingerprint = (
            request.payload.objective,
            request.payload.durability,
            request.payload.progress_reporting,
            request.payload.completion_reporting,
        )
        capacity_rejected = False
        replay_expired = False
        async with self._dispatch_lock:
            with self._bindings.admission(binding) as binding_active:
                if not binding_active:
                    raise PermissionError(
                        "participant binding was released before dispatch admission"
                    )
                dispatch_task = self._dispatch_tasks.get(key)
                expired_fingerprint = self._expired_dispatches.get(key)
                if expired_fingerprint is not None:
                    if expired_fingerprint != fingerprint:
                        raise ValueError(
                            "task_id was reused with conflicting dispatch parameters"
                        )
                    replay_expired = True
                if (
                    dispatch_task is not None
                    and self._dispatch_fingerprints[key] != fingerprint
                ):
                    raise ValueError(
                        "task_id was reused with conflicting dispatch parameters"
                    )
                if dispatch_task is None and not replay_expired:
                    self._prune_completed_dispatches()
                    active_count = len(self._active_runs) + sum(
                        not task.done() and key not in self._active_runs
                        for key, task in self._dispatch_tasks.items()
                    )
                    if active_count >= self._max_active_dispatches:
                        capacity_rejected = True
                    else:
                        dispatch_task = asyncio.create_task(
                            self._dispatch_once(request, key, binding)
                        )
                        dispatch_task.add_done_callback(self._schedule_completed_prune)
                        self._dispatch_tasks[key] = dispatch_task
                        self._dispatch_fingerprints[key] = fingerprint
        if capacity_rejected:
            return await self._acknowledgment(
                request,
                reason="Hermes dispatch capacity exhausted",
            )
        if replay_expired:
            return await self._acknowledgment(
                request,
                reason="Terminal replay retention expired; submit a new task_id",
            )
        assert dispatch_task is not None
        acknowledgment = await asyncio.shield(dispatch_task)
        await self._prune_after_completion()
        return acknowledgment

    async def cancel(
        self,
        participant_id: str,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        """Authorize and idempotently signal cancellation by exact run handle."""

        binding = self._bindings.binding_for(participant_id)
        if binding is None or binding.session_id != request.session_id:
            raise PermissionError("participant is not bound to the requested Hermes session")
        key = (request.session_id, binding.generation, request.event_id)
        fingerprint = (
            request.payload.scope,
            request.task_id,
            request.payload.reason,
        )
        async with self._dispatch_lock:
            with self._bindings.admission(binding) as binding_active:
                if not binding_active:
                    raise PermissionError(
                        "participant binding was released before cancellation admission"
                    )
                cancellation_task = self._cancellation_tasks.get(key)
                if (
                    cancellation_task is not None
                    and self._cancellation_fingerprints[key] != fingerprint
                ):
                    raise ValueError(
                        "cancellation event_id was reused with conflicting parameters"
                    )
                if cancellation_task is None:
                    self._prune_completed_cancellations()
                    prefix = (request.session_id, binding.generation)
                    targets: tuple[asyncio.Task[WorkDispatchAcknowledgedEvent], ...]
                    known_run_ids: tuple[str, ...]
                    if request.payload.scope is CancelScope.TASK:
                        assert request.task_id is not None
                        dispatch_key = (*prefix, request.task_id)
                        active_run = self._active_runs.get(dispatch_key)
                        known_run_ids = () if active_run is None else (active_run,)
                        dispatch_task = self._dispatch_tasks.get(dispatch_key)
                        targets = (
                            ()
                            if active_run is not None or dispatch_task is None
                            else (dispatch_task,)
                        )
                    elif request.payload.scope is CancelScope.SESSION:
                        known_run_ids = tuple(
                            run_id
                            for (session_id, generation, _), run_id in self._active_runs.items()
                            if (session_id, generation) == prefix
                        )
                        targets = tuple(
                            task
                            for dispatch_key, task in self._dispatch_tasks.items()
                            for session_id, generation, _ in (dispatch_key,)
                            if (session_id, generation) == prefix
                            and dispatch_key not in self._active_runs
                        )
                    else:
                        targets = ()
                        known_run_ids = ()
                    cancellation_task = asyncio.create_task(
                        self._cancel_once(request, targets, known_run_ids)
                    )
                    cancellation_task.add_done_callback(
                        self._schedule_completed_cancellation_prune
                    )
                    self._cancellation_tasks[key] = cancellation_task
                    self._cancellation_fingerprints[key] = fingerprint

        acknowledgment = await asyncio.shield(cancellation_task)
        await self._prune_after_completion()
        return acknowledgment

    async def _cancel_once(
        self,
        request: ControlCancelEvent,
        targets: tuple[asyncio.Task[WorkDispatchAcknowledgedEvent], ...],
        known_run_ids: tuple[str, ...],
    ) -> ControlCancelAcknowledgedEvent:
        if request.payload.scope not in {CancelScope.TASK, CancelScope.SESSION}:
            return await self._cancel_acknowledgment(
                request,
                reason="Hermes bridge supports only task or session cancellation",
            )
        if not targets and not known_run_ids:
            return await self._cancel_acknowledgment(
                request,
                reason="No work in the active binding matched the cancellation request",
            )
        if self._canceller is None:
            return await self._cancel_acknowledgment(
                request,
                reason="Hermes runtime does not provide exact-handle cancellation",
            )

        outcomes = await asyncio.gather(
            *(asyncio.shield(task) for task in targets),
            return_exceptions=True,
        )
        run_ids = list(
            dict.fromkeys(
                (
                    *known_run_ids,
                    *(
                        outcome.payload.run_id
                        for outcome in outcomes
                        if isinstance(outcome, WorkDispatchAcknowledgedEvent)
                        and outcome.payload.accepted
                        and outcome.payload.run_id is not None
                    ),
                )
            )
        )
        if not run_ids:
            return await self._cancel_acknowledgment(
                request,
                reason="No acknowledged Hermes run matched the cancellation request",
            )

        signal_results = await asyncio.gather(
            *(self._canceller.cancel(run_id) for run_id in run_ids),
            return_exceptions=True,
        )
        signaled = [
            run_id
            for run_id, result in zip(run_ids, signal_results, strict=True)
            if result is True
        ]
        if not signaled:
            return await self._cancel_acknowledgment(
                request,
                reason="Matched Hermes runs were no longer interruptible",
            )
        return await self._cancel_acknowledgment(request, signaled_run_ids=signaled)

    def _prune_completed_dispatches(self) -> None:
        while (
            sum(task.done() for task in self._dispatch_tasks.values())
            > self._max_retained_dispatches
        ):
            completed_key = next(
                (key for key, task in self._dispatch_tasks.items() if task.done()),
                None,
            )
            if completed_key is None:
                return
            completed_task = self._dispatch_tasks[completed_key]
            if not completed_task.cancelled() and completed_task.exception() is None:
                run_id = completed_task.result().payload.run_id
                if run_id is not None and run_id not in self._active_runs.values():
                    self._run_bindings.pop(run_id, None)
                    self._run_keys.pop(run_id, None)
            del self._dispatch_tasks[completed_key]
            del self._dispatch_fingerprints[completed_key]

    def _prune_completed_cancellations(self) -> None:
        while (
            sum(task.done() for task in self._cancellation_tasks.values())
            > self._max_retained_dispatches
        ):
            completed_key = next(
                (
                    key
                    for key, task in self._cancellation_tasks.items()
                    if task.done()
                ),
                None,
            )
            if completed_key is None:
                return
            del self._cancellation_tasks[completed_key]
            del self._cancellation_fingerprints[completed_key]

    def _schedule_completed_prune(
        self,
        completed: asyncio.Task[WorkDispatchAcknowledgedEvent],
    ) -> None:
        del completed
        asyncio.create_task(self._prune_after_completion())

    def _schedule_completed_cancellation_prune(
        self,
        completed: asyncio.Task[ControlCancelAcknowledgedEvent],
    ) -> None:
        del completed
        asyncio.create_task(self._prune_after_completion())

    async def _prune_after_completion(self) -> None:
        async with self._dispatch_lock:
            self._prune_completed_dispatches()
            self._prune_completed_cancellations()

    async def _dispatch_once(
        self,
        request: WorkDispatchRequestedEvent,
        key: tuple[str, int, str],
        binding: SessionBinding,
    ) -> WorkDispatchAcknowledgedEvent:
        command = HermesDispatchCommand(
            session_id=request.session_id,
            task_id=request.task_id,
            objective=request.payload.objective,
            durability=request.payload.durability,
        )
        try:
            run_id = (await self._dispatcher.dispatch(command)).strip()
        except HermesDispatchRejected as exc:
            return await self._acknowledgment(request, reason=exc.reason)
        if not run_id:
            raise RuntimeError("Hermes accepted dispatch without a run identifier")

        acknowledgment = await self._acknowledgment(request, run_id=run_id)
        async with self._dispatch_lock:
            if run_id in self._run_bindings:
                raise RuntimeError("Hermes reused an active delegation identifier")
            self._active_runs[key] = run_id
            self._run_bindings[run_id] = binding
            self._run_keys[run_id] = key
            self._refresh_active_run_ids_snapshot()
            publisher = getattr(self._dispatcher, "publish", None)
            if callable(publisher):
                publisher(run_id)
        return acknowledgment

    async def binding_for_run(self, run_id: str) -> SessionBinding | None:
        """Return the immutable admitted binding that owns an active run."""

        async with self._dispatch_lock:
            return self._run_bindings.get(run_id)

    async def active_run_ids(self) -> frozenset[str]:
        """Snapshot exact active run IDs for authoritative evidence retention."""

        async with self._dispatch_lock:
            return frozenset(self._active_runs.values())

    def active_run_ids_now(self) -> frozenset[str]:
        """Return the immutable cross-thread snapshot used at hook ingress."""

        return self._active_run_ids_snapshot

    def _refresh_active_run_ids_snapshot(self) -> None:
        self._active_run_ids_snapshot = frozenset(self._active_runs.values())

    def current_binding(self, participant_id: str) -> SessionBinding | None:
        """Snapshot the participant's immutable binding for bounded output."""

        return self._bindings.binding_for(participant_id)

    def emit_if_binding_current(
        self,
        binding: SessionBinding,
        emit: Callable[[], None],
    ) -> bool:
        """Atomically validate an immutable generation and emit without awaiting."""

        with self._bindings.admission(binding) as active:
            if active:
                emit()
            return active

    async def mark_terminal(self, run_id: str) -> None:
        """Release active ownership only after exact terminal evidence."""

        async with self._dispatch_lock:
            completed_keys = [
                key for key, active_run_id in self._active_runs.items() if active_run_id == run_id
            ]
            for key in completed_keys:
                del self._active_runs[key]
                if key not in self._dispatch_tasks:
                    self._run_bindings.pop(run_id, None)
            self._refresh_active_run_ids_snapshot()

    async def expire_terminal_replay(self, run_id: str) -> None:
        """Expire the matching dispatch retry entry with its terminal evidence."""

        async with self._dispatch_lock:
            key = self._run_keys.pop(run_id, None)
            self._run_bindings.pop(run_id, None)
            if key is None:
                return
            if self._active_runs.get(key) == run_id:
                del self._active_runs[key]
                self._refresh_active_run_ids_snapshot()
            self._dispatch_tasks.pop(key, None)
            fingerprint = self._dispatch_fingerprints.pop(key, None)
            if fingerprint is None:
                return
            self._expired_dispatches[key] = fingerprint
            self._expired_dispatches.move_to_end(key)
            while len(self._expired_dispatches) > self._max_retained_dispatches:
                self._expired_dispatches.popitem(last=False)

    async def _acknowledgment(
        self,
        request: WorkDispatchRequestedEvent,
        *,
        run_id: str | None = None,
        reason: str | None = None,
    ) -> WorkDispatchAcknowledgedEvent:
        sequence = await self._sequencer.next(request.sequence)
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id=self._event_id_factory(),
            session_id=request.session_id,
            sequence=sequence,
            timestamp=self._clock(),
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=run_id is not None,
                run_id=run_id,
                reason=reason,
            ),
        )

    async def _cancel_acknowledgment(
        self,
        request: ControlCancelEvent,
        *,
        signaled_run_ids: list[str] | None = None,
        reason: str | None = None,
    ) -> ControlCancelAcknowledgedEvent:
        sequence = await self._sequencer.next(request.sequence)
        run_ids = signaled_run_ids or []
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id=self._event_id_factory(),
            session_id=request.session_id,
            sequence=sequence,
            timestamp=self._clock(),
            request_event_id=request.event_id,
            scope=request.payload.scope,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=bool(run_ids),
                signaled_run_ids=run_ids,
                reason=reason,
            ),
        )
