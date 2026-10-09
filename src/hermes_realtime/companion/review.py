"""Owned Hermes memory and skills review over verified voice archive windows.

The archive is the authority for the rows. This coordinator owns only review admission,
thread lifetime, and a content-free durable ledger. A completed review is never described
as learned: Hermes may choose to make no memory or skill change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from hermes_realtime.companion.archive import VoiceArchive
from hermes_realtime.companion.integrity import (
    MAX_ARCHIVE_ROWS,
    MAX_IDENTITY,
    ArchiveRefusal,
    validate_conversation_id,
)
from hermes_realtime.companion.store import (
    QUARANTINE_CATEGORIES,
    CompanionStore,
    ConversationRecord,
)

_MARKER = "[voice-review] "
MAX_REVIEW_ROWS = 24
MAX_REVIEW_BYTES = 65_536
MAX_REVIEW_TOKENS = 16_384
MAX_REVIEW_INTERVAL = 1000


def review_snapshot_admitted(snapshot: list[dict[str, str]]) -> bool:
    """Whether a review snapshot is within the rows and bytes this companion admits.

    The snapshot itself has a conservative token bound (one token per UTF-8 byte).
    Hermes separately accounts for its prompt and schema. Realtime chooses its review
    windows to fit this same rule.
    """
    payload_bytes = len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))
    return (
        1 <= len(snapshot) <= MAX_REVIEW_ROWS
        and payload_bytes <= MAX_REVIEW_BYTES
        and payload_bytes <= MAX_REVIEW_TOKENS
    )


MAX_JOIN_SECONDS = 300.0
_CLOSE_JOIN_SECONDS = 10.0
_REVIEW_WALL_SECONDS = 300.0


class ReviewQuiescenceError(RuntimeError):
    """The owned review or parent still runs; the profile lease must remain held."""


def _marker(evidence: dict[str, str | int]) -> None:
    print(_MARKER + json.dumps(evidence, sort_keys=True, separators=(",", ":")), flush=True)


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    conversation_id: str
    generation: int
    seq_from: int
    seq_through: int
    memory: bool
    skills: bool
    closing: bool

    def __post_init__(self) -> None:
        validate_conversation_id(self.conversation_id)
        for value in (self.generation, self.seq_from, self.seq_through):
            if type(value) is not int or not 0 <= value <= MAX_IDENTITY:
                raise ValueError("review identity is out of range")
        if self.seq_from > self.seq_through:
            raise ValueError("review range is reversed")
        if any(type(value) is not bool for value in (self.memory, self.skills, self.closing)):
            raise TypeError("review flags must be exact booleans")
        if not self.memory or not self.skills:
            raise ValueError("voice review requires both Hermes memory and skills")


@dataclass(frozen=True, slots=True)
class ReviewAdmission:
    review_id: str
    status: str
    conversation_id: str
    generation: int
    seq_from: int
    seq_through: int
    closing: bool


class ReviewPort(Protocol):
    def settings(self) -> tuple[int, dict[str, object]]: ...

    def verify_parent_binding(self, parent: Any) -> None: ...

    def bind_parent_callbacks(self, parent: Any, failed: Callable[[], None]) -> None: ...

    def close_parent(self, parent: Any) -> None: ...

    def drain_failed_parents(self, timeout: float) -> bool: ...

    def admit(
        self, parent: Any, record: ConversationRecord, request: ReviewRequest,
        cap: int, lease_ttl_seconds: float,
    ) -> tuple[list[dict[str, str]], Any] | None: ...

    def spawn(
        self, parent: Any, snapshot: list[dict[str, str]], token: Any,
        task_cfg: dict[str, object],
    ) -> Callable[[], None]: ...

    def finish(self, parent: Any, token: Any) -> None: ...

    def cancel(self, parent: Any, token: Any) -> None: ...


@dataclass(slots=True)
class _Running:
    review_id: str
    parent: Any
    token: Any
    thread: threading.Thread
    failure: bool = False
    cancelled: bool = False
    watchdog: asyncio.TimerHandle | None = None


class VoiceReviewCoordinator:
    """One authoritative review parent per conversation and generation."""

    def __init__(
        self,
        archive: VoiceArchive,
        store: CompanionStore,
        port: ReviewPort,
        parent_factory: Callable[[str], Any],
    ) -> None:
        if type(archive) is not VoiceArchive or type(store) is not CompanionStore:
            raise TypeError("review needs the exact archive and store")
        if not callable(parent_factory):
            raise TypeError("review parent factory must be callable")
        self._archive = archive
        self._store = store
        self._port = port
        self._parent_factory = parent_factory
        self._parents: dict[tuple[str, int], Any] = {}
        self._parent_builds: dict[tuple[str, int], asyncio.Task[Any]] = {}
        self._requests: set[asyncio.Task[Any]] = set()
        self._active: dict[str, _Running] = {}
        self._interval: int | None = None
        self._closed = False
        self._on_finished: Callable[[], None] = lambda: None
        self._on_ended: Callable[[str], None] = lambda _conversation: None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._parent_closes: dict[int, asyncio.Task[None]] = {}
        self._cancel_tasks: dict[str, asyncio.Task[None]] = {}
        self._owned_releases: dict[int, tuple[Any, Any]] = {}
        self._release_tasks: dict[int, asyncio.Task[None]] = {}
        quarantined = self._store.recover_reviews()
        if quarantined:
            _marker({"count": quarantined, "refusal": "quarantined", "version": 1})

    @property
    def review_interval(self) -> int:
        if self._interval is None:
            raise RuntimeError("review settings have not been verified")
        return self._interval

    async def start(self) -> None:
        interval, _ = await self._settings(require_enabled=False)
        self._interval = interval
        self._loop = asyncio.get_running_loop()

    async def _settings(
        self, *, require_enabled: bool = True
    ) -> tuple[int, dict[str, object]]:
        interval, task_cfg = await self._archive._call(self._port.settings)
        if type(interval) is not int or not 1 <= interval <= MAX_REVIEW_INTERVAL:
            raise ArchiveRefusal("configuration")
        if self._interval is not None and interval != self._interval:
            raise ArchiveRefusal("configuration")
        if type(task_cfg) is not dict:
            raise ArchiveRefusal("configuration")
        if require_enabled and task_cfg.get("enabled") is not True:
            raise ArchiveRefusal("disabled")
        extras = task_cfg.get("extra_tools", [])
        if require_enabled and (type(extras) is not list or extras):
            raise ArchiveRefusal("configuration")
        return interval, task_cfg

    def outcome(self, conversation_id: str, review_id: str) -> str | None:
        validate_conversation_id(conversation_id)
        if type(review_id) is not str:
            raise TypeError("review ID must be a string")
        return self._store.review_outcome(conversation_id, review_id)

    def active(self, conversation_id: str) -> str | None:
        running = self._active.get(conversation_id)
        return None if running is None else running.review_id

    def admitted(self, conversation_id: str) -> bool:
        """Whether a local review still owns admission or may still write."""
        running = self._active.get(conversation_id)
        return running is not None and (
            running.thread.is_alive()
            or self._store.review_outcome(conversation_id, running.review_id)
            in {"reserved", "accepted"}
        )

    async def review(self, request: ReviewRequest) -> ReviewAdmission:
        evidence: dict[str, str | int] | None = None
        caller = asyncio.current_task()
        if caller is not None:
            self._requests.add(caller)
        try:
            if type(request) is not ReviewRequest:
                raise TypeError("review request must be exact")
            if self._closed:
                raise ArchiveRefusal("not_ready")
            async with self._archive._guard(request.conversation_id):
                return await self._review(request)
        except ArchiveRefusal as refusal:
            evidence = {"refusal": refusal.category, "version": 1}
            raise
        except BaseException as error:
            evidence = {"failure": type(error).__name__, "version": 1}
            raise
        finally:
            if caller is not None:
                self._requests.discard(caller)
            if evidence is not None:
                _marker(evidence)

    async def _review(self, request: ReviewRequest) -> ReviewAdmission:
        if not self._archive.ready(request.conversation_id):
            raise ArchiveRefusal("not_ready")
        record = self._store.read(request.conversation_id)
        if record is None or record.committed is None:
            raise ArchiveRefusal("unbound")
        if record.quarantine is not None:
            raise ArchiveRefusal("quarantined")
        if record.tombstone is not None:
            raise ArchiveRefusal("tombstoned")
        if record.pending is not None:
            raise ArchiveRefusal("busy")
        cursor = record.committed.cursor
        if cursor is None or request.generation != cursor.generation:
            raise ArchiveRefusal("invalid")
        if request.seq_through > cursor.seq:
            raise ArchiveRefusal("not_ready")
        previous = self._store.find_review(request)
        if previous is not None:
            previous_id, previous_outcome = previous
            if previous_outcome in {"reserved", "unknown"}:
                raise ArchiveRefusal("unknown")
            if previous_outcome == "failed":
                raise ArchiveRefusal("failed")
            return ReviewAdmission(
                previous_id, "accepted", request.conversation_id, request.generation,
                request.seq_from, request.seq_through, request.closing,
            )
        if request.conversation_id in self._active and not await self.join(
            request.conversation_id, 0.0
        ):
            raise ArchiveRefusal("busy")
        _, task_cfg = await self._settings()
        if self._closed:
            raise ArchiveRefusal("not_ready")
        key = request.conversation_id, request.generation
        parent = self._parents.get(key)
        if parent is None:
            parent = await self._build_parent_owned(key, record.session_id)
        if self._closed:
            raise ArchiveRefusal("not_ready")
        await self._archive._call(self._port.verify_parent_binding, parent)
        if not await self._drain_owned_releases(parent, _CLOSE_JOIN_SECONDS):
            raise ArchiveRefusal("busy")
        try:
            admitted = await self._admit_owned(parent, record, request)
        except ArchiveRefusal as refusal:
            if refusal.category in QUARANTINE_CATEGORIES:
                self._archive._quarantine(request.conversation_id, refusal.category)
            raise
        if admitted is None:
            raise ArchiveRefusal("busy")
        snapshot, token = admitted
        if type(snapshot) is not list or not 1 <= len(snapshot) <= MAX_REVIEW_ROWS:
            self._finish_owned(parent, token)
            raise ArchiveRefusal("window")
        review_id = f"vr_{uuid.uuid4().hex}"
        try:
            target = self._port.spawn(parent, snapshot, token, task_cfg)
            self._store.reserve_review(request, review_id)
            running = _Running(
                review_id, parent, token,
                threading.Thread(
                    target=lambda: self._run_target(request.conversation_id, review_id, target),
                    name="voice-review", daemon=True,
                ),
            )
            self._active[request.conversation_id] = running
            running.thread.start()
            self._store.accept_review(request, review_id)
            if self._loop is not None:
                running.watchdog = self._loop.call_later(
                    _REVIEW_WALL_SECONDS,
                    lambda: asyncio.create_task(
                        self._expire(request.conversation_id, review_id)
                    ),
                )
        except BaseException:
            active_run = self._active.get(request.conversation_id)
            if active_run is not None and active_run.thread.is_alive():
                await self.cancel_and_join(request.conversation_id, _CLOSE_JOIN_SECONDS)
            else:
                self._active.pop(request.conversation_id, None)
                self._finish_owned(parent, token)
            if self._store.review_outcome(request.conversation_id, review_id) == "reserved":
                self._store.finish_review(request.conversation_id, review_id, "failed")
            raise
        return ReviewAdmission(
            review_id, "accepted", request.conversation_id, request.generation,
            request.seq_from, request.seq_through, request.closing,
        )

    def _run_target(self, conversation_id: str, review_id: str, target: Callable[[], None]) -> None:
        running = self._active[conversation_id]
        parent = running.parent

        class _OwnReviewLogFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return threading.current_thread() is not running.thread

        review_logger = logging.getLogger("agent.background_review")
        own_filter = _OwnReviewLogFilter()
        review_logger.addFilter(own_filter)
        try:
            self._port.bind_parent_callbacks(parent, lambda: setattr(running, "failure", True))
            target()
        except BaseException:
            running.failure = True
        finally:
            review_logger.removeFilter(own_filter)
            try:
                self._finish_owned(parent, running.token)
            except BaseException:
                running.failure = True
            loop = self._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._finished, conversation_id, review_id)

    def _finished(self, conversation_id: str, review_id: str) -> None:
        running = self._active.get(conversation_id)
        if running is None or running.review_id != review_id:
            return
        if self._store.review_outcome(conversation_id, review_id) != "accepted":
            return
        outcome = "cancelled" if running.cancelled else "failed" if running.failure else "finished"
        self._store.finish_review(conversation_id, review_id, outcome)
        if outcome == "finished":
            self._on_finished()
        self._on_ended(conversation_id)
        _marker({"outcome": outcome, "version": 1})

    async def _build_parent_owned(self, key: tuple[str, int], session_id: str) -> Any:
        """Register a constructed parent before propagating caller cancellation."""

        async def construct() -> Any:
            parent = await asyncio.to_thread(self._parent_factory, session_id)
            self._parents[key] = parent
            return parent

        work = asyncio.create_task(construct())
        self._parent_builds[key] = work
        cancelled = False
        try:
            while not work.done():
                try:
                    await asyncio.wait({work})
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                work.exception()
                raise asyncio.CancelledError
            return work.result()
        finally:
            if work.done() and self._parent_builds.get(key) is work:
                del self._parent_builds[key]

    async def _admit_owned(
        self, parent: Any, record: ConversationRecord, request: ReviewRequest
    ) -> tuple[list[dict[str, str]], Any] | None:
        """A cancelled caller still observes and releases any token Hermes took."""

        work = asyncio.create_task(
            asyncio.to_thread(
                self._port.admit, parent, record, request,
                MAX_ARCHIVE_ROWS, self._archive._ttl,
            )
        )
        cancelled = False
        while not work.done():
            try:
                await asyncio.wait({work})
            except asyncio.CancelledError:
                cancelled = True
        admitted = work.result()
        if cancelled:
            if admitted is not None:
                self._finish_owned(parent, admitted[1])
            raise asyncio.CancelledError
        return admitted

    def _finish_owned(self, parent: Any, token: Any) -> None:
        """Keep a strong token owner until Hermes confirms its release."""

        key = id(token)
        self._owned_releases[key] = parent, token
        self._port.finish(parent, token)
        self._owned_releases.pop(key, None)

    async def _drain_owned_releases(self, parent: Any | None, timeout: float) -> bool:
        for key, (owner, token) in tuple(self._owned_releases.items()):
            if parent is not None and owner is not parent:
                continue
            work = self._release_tasks.get(key)
            if work is None or (work.done() and (work.cancelled() or work.exception() is not None)):
                work = asyncio.create_task(asyncio.to_thread(self._port.finish, owner, token))
                self._release_tasks[key] = work
            try:
                await asyncio.wait_for(asyncio.shield(work), timeout)
            except asyncio.CancelledError:
                raise
            except BaseException:
                return False
            self._owned_releases.pop(key, None)
            self._release_tasks.pop(key, None)
        return True

    async def _expire(self, conversation_id: str, review_id: str) -> None:
        active = self._active.get(conversation_id)
        if active is not None and active.review_id == review_id:
            _marker({"refusal": "wall_timeout", "version": 1})
            await self.cancel_and_join(conversation_id, _CLOSE_JOIN_SECONDS)

    async def join(self, conversation_id: str, timeout: float) -> bool:
        if type(timeout) not in (float, int):
            raise TypeError("join timeout must be numeric")
        if not 0 <= timeout <= MAX_JOIN_SECONDS:
            raise ValueError("join timeout is out of range")
        running = self._active.get(conversation_id)
        if running is None:
            return True
        await asyncio.to_thread(running.thread.join, timeout)
        if running.thread.is_alive():
            _marker({"refusal": "join_timeout", "version": 1})
            return False
        await asyncio.sleep(0)
        if self._store.review_outcome(conversation_id, running.review_id) == "accepted":
            self._finished(conversation_id, running.review_id)
        cancel_task = self._cancel_tasks.get(running.review_id)
        if cancel_task is not None and not cancel_task.done():
            # The native helper still owns the parent after the worker exits.
            return False
        if running.watchdog is not None:
            running.watchdog.cancel()
        if self._active.get(conversation_id) is running:
            del self._active[conversation_id]
        return True

    async def cancel_and_join(self, conversation_id: str, timeout: float) -> bool:
        if type(timeout) not in (float, int) or not 0 <= timeout <= MAX_JOIN_SECONDS:
            raise ValueError("cancel timeout is out of range")
        running = self._active.get(conversation_id)
        if running is None:
            return True
        running.cancelled = True
        deadline = asyncio.get_running_loop().time() + timeout
        work = self._cancel_tasks.get(running.review_id)
        if work is None or (work.done() and (work.cancelled() or work.exception() is not None)):
            work = asyncio.create_task(
                asyncio.to_thread(self._port.cancel, running.parent, running.token)
            )
            self._cancel_tasks[running.review_id] = work
        cancel_error: BaseException | None = None
        try:
            await asyncio.wait_for(
                asyncio.shield(work),
                min(timeout, 3.0),
            )
        except BaseException as error:
            cancel_error = error
            running.failure = True
            _marker({
                "refusal": "cancel_timeout" if type(error) is TimeoutError else "cancel_failed",
                "version": 1,
            })
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        try:
            joined = await self.join(conversation_id, remaining)
        except BaseException:
            joined = False
        if not joined:
            return False
        if not work.done():
            # The native cancellation helper still owns the parent. Retain the
            # profile even when the review thread has already exited.
            return False
        self._cancel_tasks.pop(running.review_id, None)
        if isinstance(cancel_error, asyncio.CancelledError):
            raise cancel_error
        return True

    async def close(self) -> None:
        self._closed = True
        if asyncio.current_task() in self._requests:
            raise ReviewQuiescenceError("review request has not quiesced")
        requests = tuple(self._requests)
        for caller in requests:
            caller.cancel()
        if requests:
            try:
                _, pending = await asyncio.wait(requests, timeout=_CLOSE_JOIN_SECONDS)
            except asyncio.CancelledError:
                raise ReviewQuiescenceError("review request has not quiesced") from None
            if pending:
                raise ReviewQuiescenceError("review request has not quiesced")
        for conversation_id in tuple(self._active):
            if not await self.cancel_and_join(conversation_id, _CLOSE_JOIN_SECONDS):
                raise ReviewQuiescenceError("review thread has not quiesced")
        for build_work in tuple(self._parent_builds.values()):
            try:
                await asyncio.wait_for(asyncio.shield(build_work), _CLOSE_JOIN_SECONDS)
            except TimeoutError:
                raise ReviewQuiescenceError("review parent has not been constructed") from None
            except asyncio.CancelledError:
                raise ReviewQuiescenceError("review parent has not been constructed") from None
            except Exception:
                # The factory owns cleanup of a failed construction.
                pass
        try:
            released = await self._drain_owned_releases(None, _CLOSE_JOIN_SECONDS)
        except asyncio.CancelledError:
            raise ReviewQuiescenceError("review token has not been released") from None
        if not released:
            raise ReviewQuiescenceError("review token has not been released")
        if not await asyncio.to_thread(self._port.drain_failed_parents, _CLOSE_JOIN_SECONDS):
            raise ReviewQuiescenceError("failed review parent has not closed")
        for parent in self._parents.values():
            key = id(parent)
            work = self._parent_closes.get(key)
            if work is None or (work.done() and (work.cancelled() or work.exception() is not None)):
                work = asyncio.create_task(
                    asyncio.to_thread(self._port.close_parent, parent)
                )
                self._parent_closes[key] = work
            try:
                await asyncio.wait_for(asyncio.shield(work), _CLOSE_JOIN_SECONDS)
            except BaseException:
                raise ReviewQuiescenceError("review parent has not closed") from None
        self._parents.clear()
        self._parent_closes.clear()
