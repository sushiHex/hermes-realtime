"""Correlate Hermes terminal callbacks with acknowledged realtime work."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import datetime

from hermes_realtime.protocol import (
    ProtocolEvent,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkTerminalStatus,
)

from .sequencing import EventSequencer


class HermesCompletionRouter:
    """Emit terminal events only for authoritative, acknowledged Hermes runs."""

    def __init__(
        self,
        *,
        sequencer: EventSequencer,
        event_id_factory: Callable[[], str],
        clock: Callable[[], datetime],
        max_retained_completions: int = 256,
        max_tracked_acknowledgments: int = 1024,
    ) -> None:
        if max_retained_completions < 1:
            raise ValueError("max_retained_completions must be positive")
        if max_tracked_acknowledgments < max_retained_completions:
            raise ValueError(
                "max_tracked_acknowledgments must not be below max_retained_completions"
            )
        self._sequencer = sequencer
        self._event_id_factory = event_id_factory
        self._clock = clock
        self._max_retained_completions = max_retained_completions
        self._max_tracked_acknowledgments = max_tracked_acknowledgments
        self._acknowledgments: OrderedDict[str, WorkDispatchAcknowledgedEvent] = OrderedDict()
        self._completed: OrderedDict[str, WorkCompletedEvent] = OrderedDict()
        self._expiration_callback: Callable[[str], Awaitable[None]] | None = None
        self._lock = asyncio.Lock()

    def set_expiration_callback(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Couple terminal eviction to the dispatch retry-cache owner."""

        self._expiration_callback = callback

    async def track(self, acknowledgment: WorkDispatchAcknowledgedEvent) -> None:
        """Track one accepted acknowledgment by its authoritative run ID."""

        run_id = acknowledgment.payload.run_id
        if not acknowledgment.payload.accepted or run_id is None:
            raise ValueError("only accepted acknowledgments can be tracked")
        async with self._lock:
            existing = self._acknowledgments.get(run_id)
            if existing is not None and existing != acknowledgment:
                raise ValueError("run_id was reused for a different acknowledged task")
            # Acknowledgments are released only when their completion is evicted,
            # so a run that never reports a terminal callback would otherwise be
            # retained for the process lifetime. Fail closed on the leak instead.
            if existing is None and (
                len(self._acknowledgments) >= self._max_tracked_acknowledgments
            ):
                raise RuntimeError("acknowledgment retention capacity was exceeded")
            self._acknowledgments[run_id] = acknowledgment
            await self._sequencer.observe(acknowledgment.sequence)

    async def complete(
        self,
        run_id: str,
        *,
        status: WorkTerminalStatus | str,
        summary: str | None = None,
        reason: str | None = None,
    ) -> WorkCompletedEvent:
        """Return a stable terminal event for a known run, deduplicating retries."""

        async with self._lock:
            completed = self._completed.get(run_id)
            if completed is not None:
                return completed
            acknowledgment = self._acknowledgments.get(run_id)
            if acknowledgment is None:
                raise KeyError(f"unknown Hermes run: {run_id}")
            sequence = await self._sequencer.next(acknowledgment.sequence)
            event = WorkCompletedEvent(
                type="work.completed",
                event_id=self._event_id_factory(),
                session_id=acknowledgment.session_id,
                sequence=sequence,
                timestamp=self._clock(),
                task_id=acknowledgment.task_id,
                run_id=run_id,
                payload=WorkCompletedPayload(
                    status=WorkTerminalStatus(status),
                    summary=summary,
                    reason=reason,
                ),
            )
            self._completed[run_id] = event
            while len(self._completed) > self._max_retained_completions:
                expired_run_id, _ = self._completed.popitem(last=False)
                self._acknowledgments.pop(expired_run_id, None)
                if self._expiration_callback is not None:
                    await self._expiration_callback(expired_run_id)
            return event

    async def completed(self, run_id: str) -> WorkCompletedEvent | None:
        """Return retained terminal evidence for reconnect replay."""

        async with self._lock:
            return self._completed.get(run_id)

    async def is_tracked(self, run_id: str) -> bool:
        """Return whether an acknowledgment established this exact run."""

        async with self._lock:
            return run_id in self._acknowledgments

    async def resequence(
        self,
        event: ProtocolEvent,
        *,
        after: int,
    ) -> ProtocolEvent:
        """Give retained semantic evidence a fresh transport-order sequence."""

        sequence = await self._sequencer.next(after)
        return event.model_copy(update={"sequence": sequence})
