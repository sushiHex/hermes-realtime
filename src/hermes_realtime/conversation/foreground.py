"""Exclusive ownership for the foreground conversation turn."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TurnRunner = Callable[["ForegroundTurnLease"], Awaitable[None]]
_OWNER_COORDINATOR: ContextVar[object | None] = ContextVar(
    "foreground_owner_coordinator",
    default=None,
)


class ForegroundTurnDrainTimeout(RuntimeError):
    """Raised when an invalidated owner does not settle within its bound."""

    def __init__(self, turn_id: str, timeout: float) -> None:
        self.turn_id = turn_id
        self.timeout = timeout
        super().__init__(
            f"foreground turn {turn_id!r} did not drain within {timeout:g} seconds"
        )


class ForegroundOutputBackpressure(RuntimeError):
    """Raised when the bounded foreground publication queue is full."""


class ForegroundTurnClosed(RuntimeError):
    """Raised when no more foreground publications can arrive."""


@dataclass(frozen=True, slots=True)
class ForegroundPublication:
    """One generation-labelled output admitted by the foreground owner."""

    turn_id: str
    generation: int
    text: str
    sequence: int = 1
    _invalidated: asyncio.Event = field(
        default_factory=asyncio.Event,
        compare=False,
        repr=False,
    )

    @property
    def is_valid(self) -> bool:
        return not self._invalidated.is_set()

    async def wait_invalidated(self) -> None:
        """Wait until cancellation invalidates this claimed publication."""

        await self._invalidated.wait()

    @property
    def segment_id(self) -> str:
        return f"segment_{self.sequence}"


class ForegroundTurnLease:
    """Capability held by one admitted foreground-turn task.

    A lease becomes stale before cancellation is delivered. Publication is
    admitted atomically by the coordinator, so a runner that suppresses
    cancellation cannot emit speech or other foreground output afterward.
    """

    __slots__ = (
        "_coordinator",
        "_invalidated",
        "_publication_sequence",
        "generation",
        "turn_id",
    )

    def __init__(
        self,
        coordinator: ForegroundTurnCoordinator,
        *,
        generation: int,
        turn_id: str,
    ) -> None:
        self._coordinator = coordinator
        self._invalidated = asyncio.Event()
        self._publication_sequence = 0
        self.generation = generation
        self.turn_id = turn_id

    @property
    def is_current(self) -> bool:
        """Return an observational snapshot; use ``publish`` for admission."""

        return self._coordinator._is_current(self)

    async def publish(
        self,
        text: str,
        *,
        on_admit: Callable[[ForegroundPublication], None] | None = None,
    ) -> bool:
        """Atomically enqueue one bounded, generation-labelled publication."""

        return await self._coordinator._publish(self, text, on_admit=on_admit)

    def invalidate_publications(self) -> None:
        """Fail closed before an owning producer begins asynchronous cleanup."""

        if _OWNER_COORDINATOR.get() is not self._coordinator:
            raise RuntimeError("only the foreground owner may invalidate its lease")
        self._invalidated.set()


class ForegroundTurnCoordinator:
    """Own exactly one foreground runner and invalidate stale generations."""

    def __init__(
        self,
        *,
        drain_timeout: float = 1.0,
        output_capacity: int = 32,
    ) -> None:
        if not math.isfinite(drain_timeout) or drain_timeout <= 0:
            raise ValueError("drain_timeout must be finite and greater than zero")
        if (
            not isinstance(output_capacity, int)
            or isinstance(output_capacity, bool)
            or output_capacity <= 0
        ):
            raise ValueError("output_capacity must be greater than zero")
        self._lifecycle_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._publication_available = asyncio.Condition(self._state_lock)
        self._generation = 0
        self._active_lease: ForegroundTurnLease | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._owned_tasks: set[asyncio.Task[None]] = set()
        self._task_turn_ids: dict[asyncio.Task[None], str] = {}
        self._publications: deque[ForegroundPublication] = deque()
        self._publication_invalidations: dict[int, asyncio.Event] = {}
        self._output_capacity = output_capacity
        self._closed = False
        self._drain_timeout = drain_timeout

    @property
    def active_turn_id(self) -> str | None:
        lease = self._active_lease
        return None if lease is None else lease.turn_id

    @property
    def active_task_count(self) -> int:
        """Count active and draining tasks still owned by this coordinator."""

        return len(self._owned_tasks)

    @property
    def pending_publication_count(self) -> int:
        return len(self._publications)

    async def next_publication(self) -> ForegroundPublication:
        """Wait for the next atomically admitted foreground output."""

        async with self._publication_available:
            while True:
                if self._publications:
                    return self._publications.popleft()
                if self._closed:
                    raise ForegroundTurnClosed(
                        "foreground turn coordinator is closed"
                    )
                await self._publication_available.wait()

    async def start(self, turn_id: str, runner: TurnRunner) -> ForegroundTurnLease:
        """Cancel and drain the previous owner, then admit one new turn task."""

        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id.strip():
            raise ValueError("turn_id must not be blank")
        if not callable(runner):
            raise TypeError("runner must be callable")
        self._raise_if_owner_lifecycle("replace")

        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("foreground turn coordinator is closed")
            await self._cancel_active()
            async with self._state_lock:
                if self._closed:
                    raise RuntimeError("foreground turn coordinator is closed")
                self._invalidate_publications_locked()
                self._generation += 1
                lease = ForegroundTurnLease(
                    self,
                    generation=self._generation,
                    turn_id=turn_id,
                )
                task = asyncio.create_task(
                    self._run(lease, runner),
                    name=f"foreground-turn:{turn_id}:{lease.generation}",
                )
                task.add_done_callback(self._consume_task_exception)
                self._active_lease = lease
                self._active_task = task
                self._publication_invalidations[lease.generation] = lease._invalidated
                self._owned_tasks.add(task)
                self._task_turn_ids[task] = turn_id
                return lease

    async def cancel(self) -> None:
        """Invalidate, cancel, and drain the current foreground owner."""

        self._raise_if_owner_lifecycle("cancel")
        async with self._lifecycle_lock:
            await self._cancel_active()

    def revoke_if_current(
        self,
        lease: ForegroundTurnLease | None = None,
    ) -> tuple[asyncio.Task[None], str] | None:
        """Synchronously remove an owner from publication authority.

        This deliberately contains no await.  The streaming controller calls it
        while recording its own authority transition, then drains the returned
        task outside that controller's lock.  asyncio cannot interleave another
        task during this straight-line state mutation.
        """

        self._raise_if_owner_lifecycle("cancel")
        if lease is not None and self._active_lease is not lease:
            lease._invalidated.set()
            return None
        self._invalidate_publications_locked()
        task = self._active_task
        if task is None:
            self._active_lease = None
            return None
        turn_id = self._task_turn_ids[task]
        active_lease = self._active_lease
        if active_lease is not None:
            active_lease._invalidated.set()
        self._active_lease = None
        self._generation += 1
        task.cancel()
        return task, turn_id

    async def settle_revoked(self, task: asyncio.Task[None], *, turn_id: str) -> None:
        """Boundedly drain one task already removed from active authority."""

        try:
            await self._drain_task(task, turn_id=turn_id)
        finally:
            if task.done():
                async with self._state_lock:
                    if self._active_task is task:
                        self._active_task = None
                    self._owned_tasks.discard(task)
                    self._task_turn_ids.pop(task, None)

    async def cancel_if_current(self, lease: ForegroundTurnLease) -> bool:
        """Cancel only the exact still-current lease, never a replacement."""

        if type(lease) is not ForegroundTurnLease:
            raise TypeError("lease must be an exact ForegroundTurnLease")
        self._raise_if_owner_lifecycle("cancel")
        async with self._lifecycle_lock:
            return await self._cancel_active(expected_lease=lease)

    async def close(self) -> None:
        """Permanently reject new turns after draining the current owner."""

        self._raise_if_owner_lifecycle("close")
        async with self._lifecycle_lock:
            async with self._state_lock:
                self._closed = True
                self._invalidate_publications_locked()
                self._publication_available.notify_all()
            await self._cancel_active()

    def _is_current(self, lease: ForegroundTurnLease) -> bool:
        return self._active_lease is lease

    async def _publish(
        self,
        lease: ForegroundTurnLease,
        text: str,
        *,
        on_admit: Callable[[ForegroundPublication], None] | None = None,
    ) -> bool:
        if type(text) is not str:
            raise TypeError("foreground publication must be text")
        if not text:
            raise ValueError("foreground publication must not be empty")
        if on_admit is not None and not callable(on_admit):
            raise TypeError("foreground publication observer must be callable")
        current_task = asyncio.current_task()
        async with self._state_lock:
            if self._active_lease is not lease or self._active_task is not current_task:
                return False
            if len(self._publications) >= self._output_capacity:
                raise ForegroundOutputBackpressure(
                    "foreground publication queue is full"
                )
            sequence = lease._publication_sequence + 1
            publication = ForegroundPublication(
                turn_id=lease.turn_id,
                generation=lease.generation,
                text=text,
                sequence=sequence,
                _invalidated=lease._invalidated,
            )
            if on_admit is not None:
                on_admit(publication)
            lease._publication_sequence = sequence
            self._publications.append(publication)
            self._publication_available.notify(1)
            return True

    def _raise_if_owner_lifecycle(self, action: str) -> None:
        if _OWNER_COORDINATOR.get() is self:
            raise RuntimeError(
                f"foreground owner task cannot {action} itself; return from the runner"
            )

    async def _cancel_active(
        self,
        *,
        expected_lease: ForegroundTurnLease | None = None,
    ) -> bool:
        revoked = self.revoke_if_current(expected_lease)
        if revoked is None:
            return False
        task, turn_id = revoked
        await self.settle_revoked(task, turn_id=turn_id)
        return True

    def _invalidate_publications_locked(self) -> None:
        """Invalidate every claimed or queued generation while state is locked."""

        for invalidated in self._publication_invalidations.values():
            invalidated.set()
        self._publication_invalidations.clear()
        self._publications.clear()

    async def _drain_task(self, task: asyncio.Task[None], *, turn_id: str) -> None:
        deadline = asyncio.get_running_loop().time() + self._drain_timeout
        caller_cancellation: asyncio.CancelledError | None = None

        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                if caller_cancellation is not None:
                    logger.error(
                        "foreground owner remained active after caller cancellation",
                        extra={"turn_id": turn_id, "timeout": self._drain_timeout},
                    )
                    raise caller_cancellation
                raise ForegroundTurnDrainTimeout(turn_id, self._drain_timeout)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except TimeoutError:
                if caller_cancellation is not None:
                    logger.error(
                        "foreground owner drain timed out after caller cancellation",
                        extra={"turn_id": turn_id, "timeout": self._drain_timeout},
                    )
                    raise caller_cancellation from None
                raise ForegroundTurnDrainTimeout(turn_id, self._drain_timeout) from None
            except asyncio.CancelledError as error:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    caller_cancellation = error
                    if not task.done():
                        task.cancel()
                    continue
                if task.done():
                    break
                task.cancel()
            except Exception:
                break

        if caller_cancellation is not None:
            raise caller_cancellation
        caller = asyncio.current_task()
        if caller is not None and caller.cancelling():
            raise asyncio.CancelledError

    async def _run(self, lease: ForegroundTurnLease, runner: TurnRunner) -> None:
        current_task = asyncio.current_task()
        assert current_task is not None
        owner_token = _OWNER_COORDINATOR.set(self)
        try:
            await runner(lease)
        finally:
            _OWNER_COORDINATOR.reset(owner_token)
            async with self._state_lock:
                self._owned_tasks.discard(current_task)
                self._task_turn_ids.pop(current_task, None)
                if self._active_task is current_task:
                    self._active_task = None
                    self._active_lease = None

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "foreground turn task failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )
