"""Hosting for the voice companion inside the Hermes process.

Plugin registration builds a ``VoiceCompanionHost``; that is not readiness. Its owned start
runs on one dedicated event-loop thread, so it neither needs nor blocks Hermes's own loop:

1. its companion thread waits interruptibly for the profile's exclusive OS lock beside
   the plugin store; another process (Hermes discovers plugins in the gateway, the CLI
   and cron alike) can hand off ownership without a restart;
2. it binds exactly one profile's database (``open_port``) and the plugin store, and
   checks compatibility and durability for the profile;
3. only then does it build and start the bridge, which advertises ``voice_archive``.

Unload closes the bridge, releases every lease, closes the store and the database, and
releases the lock. A second owned start in the same process is refused (no multiplexing).

Archive events are served through M0's ``VoiceArchive``. A batch is validated into M0's
types (a malformed one is refused whole). A conversation that is not ready is opened on
contact, which is M0's open order (fences, compatibility and durability, the lease,
verification); M0's archive bounds how many are live at once. The answer is an
acknowledgment of the exact range, a category refusal, or nothing when the outcome is
unknown.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from hermes_realtime.companion.archive import (
    DEFAULT_LEASE_TTL_SECONDS,
    DURABLE_SYNCHRONOUS,
    ArchivePort,
    VoiceArchive,
)
from hermes_realtime.companion.forget import DeletePort, VoiceForgetReconciler
from hermes_realtime.companion.integrity import (
    ArchiveRefusal,
    Identity,
    VoiceBatch,
    VoiceRow,
)
from hermes_realtime.companion.review import (
    ReviewQuiescenceError,
    ReviewRequest,
    VoiceReviewCoordinator,
)
from hermes_realtime.companion.store import CompanionStore
from hermes_realtime.integration.run_record import (
    lock_run_record,
    unlock_run_record,
    wait_for_run_record_lock,
)
from hermes_realtime.memory import BuiltinMemorySnapshot
from hermes_realtime.protocol import (
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceForgetAckEvent,
    VoiceForgetEvent,
    VoiceForgetRefusedEvent,
    VoiceMemoryEvent,
    VoiceMemoryRefusedEvent,
    VoiceMemorySnapshotEvent,
    VoiceReviewAckEvent,
    VoiceReviewEvent,
    VoiceReviewRefusedEvent,
)

PORT_VARIABLE = "HERMES_REALTIME_COMPANION_PORT"
TOKEN_VARIABLE = "HERMES_REALTIME_COMPANION_TOKEN"
_MIN_TOKEN_CHARS = 24
_MAX_TOKEN_CHARS = 512
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 15.0
_SUBMIT_TIMEOUT_SECONDS = 60.0
_MEMORY_TIMEOUT_SECONDS = 10.0
_MARKER = "[voice-companion] "
_T = TypeVar("_T")
_owner_lock = threading.Lock()
_owner: VoiceCompanionHost | None = None


def _marker(evidence: dict[str, str | int]) -> None:
    print(_MARKER + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


@dataclass(frozen=True, slots=True)
class CompanionEndpoint:
    """Where the companion listens on loopback, and the token realtime proves."""

    port: int
    token: str

    def __repr__(self) -> str:
        return f"CompanionEndpoint(port={self.port}, token=<redacted>)"


def companion_endpoint(environ: Mapping[str, str]) -> CompanionEndpoint | None:
    """The configured endpoint, None when unconfigured; a partial one fails closed."""

    port, token = environ.get(PORT_VARIABLE), environ.get(TOKEN_VARIABLE)
    if port is None and token is None:
        return None
    if port is None or token is None:
        raise ValueError("the companion needs both its port and its token")
    if not (port.isascii() and port.isdigit()) or not 1 <= int(port) <= 65535:
        raise ValueError("the companion port must be a decimal from 1 to 65535")
    if not _MIN_TOKEN_CHARS <= len(token) <= _MAX_TOKEN_CHARS:
        raise ValueError("the companion token must hold 24 to 512 characters")
    return CompanionEndpoint(port=int(port), token=token)


def _refused(event: VoiceArchiveEvent, category: str) -> VoiceArchiveRefusedEvent:
    """A refusal of exactly this batch's range; the category set is the protocol's."""
    return VoiceArchiveRefusedEvent.model_validate(
        {
            "protocol_version": "0.3",
            "type": "voice_archive_refused",
            "conversation_id": event.conversation_id,
            "generation": event.generation,
            "seq_from": event.seq_from,
            "seq_through": event.seq_through,
            "category": category,
        }
    )


class VoiceCompanionService:
    """Serve ``voice_archive`` through M0's archive, on the companion's loop only."""

    def __init__(
        self,
        archive: VoiceArchive,
        store: CompanionStore,
        port: ArchivePort,
        review: VoiceReviewCoordinator | None = None,
    ) -> None:
        if type(archive) is not VoiceArchive or type(store) is not CompanionStore:
            raise TypeError("the service needs an exact VoiceArchive and CompanionStore")
        self._archive = archive
        self._store = store
        self._port = port
        self._review = review
        self._forget = (
            VoiceForgetReconciler(
                store, cast(DeletePort, port),
                lambda conversation: review.admitted(conversation) if review else False,
                archive._guard,
                archive._tombstone_owned,
                archive._retire_owned,
            )
            if all(callable(getattr(port, name, None)) for name in
                   ("capture_delete_targets", "delete_target", "absent", "capture_copies",
                    "copies_absent"))
            else None
        )
        self._forget_tasks: set[asyncio.Task[str]] = set()
        self._open_lock = asyncio.Lock()
        self._memory_ready = False
        self._memory_closed = False
        self._memory_revision = 0
        self._memory_events: set[asyncio.Event] = set()
        self._memory_reads: set[asyncio.Task[BuiltinMemorySnapshot]] = set()
        archive._on_state_change = self._memory_changed
        if review is not None:
            review._on_finished = self._memory_finished
            review._on_ended = self._review_ended

    def _review_ended(self, conversation_id: str) -> None:
        if self._forget is not None and self._store.deletion(conversation_id) is not None:
            task = asyncio.create_task(self._reconcile_after_review_end(conversation_id))
            self._forget_tasks.add(task)
            task.add_done_callback(self._forget_done)

    async def _reconcile_after_review_end(self, conversation_id: str) -> str:
        # The callback is queued from the review thread's finally before that
        # thread actually exits. Observe its natural end without cancelling it.
        while self._review is not None and self._review.admitted(conversation_id):
            await asyncio.sleep(0.01)
        assert self._forget is not None
        return await self._forget.reconcile(conversation_id)

    def _forget_done(self, task: asyncio.Task[str]) -> None:
        self._forget_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                _marker({"failure": type(error).__name__, "version": 1})

    async def start(self) -> None:
        """Compatibility and durability for the profile; no conversation opens here.

        A conversation opens on first contact, so no count of stored conversations can
        refuse start; the archive's own bound caps the live ones.
        """

        if await asyncio.to_thread(self._port.check_compatibility):
            raise ArchiveRefusal("incompatible")
        if await asyncio.to_thread(self._port.durability_level) < DURABLE_SYNCHRONOUS:
            raise ArchiveRefusal("durability")
        if self._review is not None:
            try:
                await self._review.start()
            except ArchiveRefusal as refusal:
                _marker({"refusal": refusal.category, "version": 1})
        if self._forget is not None:
            await self._forget.reconcile_all()
        self._store.drop_completed_bindings()
        self._memory_ready = True

    @property
    def memory_available(self) -> bool:
        return self._memory_ready and callable(getattr(self._port, "read_builtin_memory", None))

    @property
    def forget_available(self) -> bool:
        return self._memory_ready and self._forget is not None

    async def forget(
        self, event: VoiceForgetEvent
    ) -> VoiceForgetAckEvent | VoiceForgetRefusedEvent | None:
        if type(event) is not VoiceForgetEvent:
            raise TypeError("event must be an exact VoiceForgetEvent")
        try:
            if self._forget is None or not self._memory_ready:
                raise ArchiveRefusal("not_ready")
            state = await self._forget.forget(event.conversation_id, event.generation)
        except ArchiveRefusal as refusal:
            _marker({"refusal": refusal.category, "version": 1})
            return VoiceForgetRefusedEvent(
                protocol_version="0.3", type="voice_forget_refused",
                conversation_id=event.conversation_id, generation=event.generation,
                category=refusal.category,
            )
        except Exception as error:
            _marker({"failure": type(error).__name__, "version": 1})
            return None
        self._memory_changed()
        return VoiceForgetAckEvent(
            protocol_version="0.3", type="voice_forget_ack",
            conversation_id=event.conversation_id, generation=event.generation,
            state=state,
        )

    def _memory_changed(self) -> None:
        for event in self._memory_events:
            event.set()

    def _memory_finished(self) -> None:
        if self._memory_revision == 2**53 - 1:
            self._memory_ready = False
        else:
            self._memory_revision += 1
        self._memory_changed()

    def _memory_status(self, event: VoiceMemoryEvent, *, recovering: bool = False) -> None:
        if self._memory_closed or not self.memory_available or self._archive._fenced:
            raise ArchiveRefusal("not_ready")
        if self._store.deletion(event.conversation_id) is not None:
            raise ArchiveRefusal("tombstoned")
        record = self._store.read(event.conversation_id)
        # Authentication binds the profile. A new conversation may recall before its
        # first archived row; a memory read never allocates a Hermes session or binding.
        if record is None:
            return
        if record.quarantine is not None:
            raise ArchiveRefusal("quarantined")
        if record.tombstone is not None:
            raise ArchiveRefusal("tombstoned")
        if record.committed is None:
            raise ArchiveRefusal("pending")
        cursor = record.committed.cursor
        if cursor is not None and event.generation != cursor.generation:
            raise ArchiveRefusal("stale")
        if record.pending is not None and not recovering:
            raise ArchiveRefusal("pending")
        if not recovering and not self._archive.ready(event.conversation_id):
            raise ArchiveRefusal("not_ready")

    async def _read_memory(
        self, event: VoiceMemoryEvent, *, recovering: bool,
    ) -> BuiltinMemorySnapshot:
        self._memory_status(event, recovering=recovering)
        if recovering and self._store.read(event.conversation_id) is not None:
            await self._ensure_open(event.conversation_id)
        async with self._archive._guard(event.conversation_id):
            self._memory_status(event)
            reader = getattr(self._port, "read_builtin_memory", None)
            if not callable(reader):
                raise ArchiveRefusal("not_ready")
            value = await self._archive._call(reader)
            if type(value) is not BuiltinMemorySnapshot:
                raise ArchiveRefusal("invalid")
            return value

    def _memory_read_done(self, task: asyncio.Task[BuiltinMemorySnapshot]) -> None:
        self._memory_reads.discard(task)
        if not task.cancelled():
            task.exception()

    async def memory(
        self, event: VoiceMemoryEvent,
    ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]:
        """One open read and finished-review refreshes; no timer or turn-path I/O."""
        if type(event) is not VoiceMemoryEvent:
            raise TypeError("memory request must be exact")
        if len(self._memory_events) >= 32:
            try:
                yield VoiceMemoryRefusedEvent(
                    protocol_version="0.3", type="voice_memory_refused",
                    conversation_id=event.conversation_id, generation=event.generation,
                    category="capacity",
                )
            finally:
                print('[voice-memory] {"refusal": "capacity", "version": 1}', flush=True)
            return
        changed = asyncio.Event()
        self._memory_events.add(changed)
        last_revision = -1
        first = True
        try:
            while True:
                changed.clear()
                evidence: dict[str, str | int] | None = None
                try:
                    self._memory_status(event, recovering=first)
                    revision = self._memory_revision
                    if revision != last_revision:
                        if len(self._memory_reads) >= 32:
                            raise ArchiveRefusal("capacity")
                        work = asyncio.create_task(self._read_memory(event, recovering=first))
                        self._memory_reads.add(work)
                        work.add_done_callback(self._memory_read_done)
                        first = False
                        value = await asyncio.wait_for(
                            asyncio.shield(work), _MEMORY_TIMEOUT_SECONDS,
                        )
                        self._memory_status(event)
                        yield VoiceMemorySnapshotEvent(
                            protocol_version="0.3", type="voice_memory_snapshot",
                            conversation_id=event.conversation_id, generation=event.generation,
                            revision=revision, memory=value.memory, user=value.user,
                            truncated=value.truncated,
                        )
                        last_revision = revision
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    category = error.category if type(error) is ArchiveRefusal else "not_ready"
                    evidence = {"refusal": category, "version": 1}
                    last_revision = -1
                    yield VoiceMemoryRefusedEvent(
                        protocol_version="0.3", type="voice_memory_refused",
                        conversation_id=event.conversation_id, generation=event.generation,
                        category=category,
                    )
                finally:
                    if evidence is not None:
                        print("[voice-memory] " + json.dumps(evidence, sort_keys=True), flush=True)
                if self._memory_closed:
                    return
                await changed.wait()
        finally:
            self._memory_events.discard(changed)

    @property
    def review_interval(self) -> int | None:
        if self._review is None:
            return None
        try:
            return self._review.review_interval
        except RuntimeError:
            return None

    async def review(
        self, event: VoiceReviewEvent
    ) -> VoiceReviewAckEvent | VoiceReviewRefusedEvent | None:
        if type(event) is not VoiceReviewEvent:
            raise TypeError("event must be an exact VoiceReviewEvent")
        evidence: dict[str, str | int] | None = None
        try:
            if self._review is None:
                raise ArchiveRefusal("not_ready")
            # The fence outlives the binding a completed delete drops.
            if self._store.deletion(event.conversation_id) is not None:
                raise ArchiveRefusal("tombstoned")
            if self._store.read(event.conversation_id) is None:
                evidence = {"refusal": "unbound", "version": 1}
                raise ArchiveRefusal("unbound")
            await self._ensure_open(event.conversation_id)
            result = await self._review.review(
                ReviewRequest(
                    event.conversation_id, event.generation, event.seq_from,
                    event.seq_through, event.memory, event.skills, event.closing,
                )
            )
        except ArchiveRefusal as refusal:
            return VoiceReviewRefusedEvent(
                protocol_version="0.3", type="voice_review_refused",
                conversation_id=event.conversation_id, generation=event.generation,
                seq_from=event.seq_from, seq_through=event.seq_through,
                closing=event.closing, category=refusal.category,
            )
        except Exception:
            return None
        finally:
            if evidence is not None:
                _marker(evidence)
        return VoiceReviewAckEvent(
            protocol_version="0.3", type="voice_review_ack",
            conversation_id=event.conversation_id, generation=event.generation,
            seq_from=event.seq_from, seq_through=event.seq_through,
            closing=event.closing, review_id=result.review_id, status="accepted",
        )

    async def archive(
        self, event: VoiceArchiveEvent
    ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent | None:
        if type(event) is not VoiceArchiveEvent:
            raise TypeError("event must be an exact VoiceArchiveEvent")
        batch = self._batch(event)
        if batch is None:
            return _refused(event, "invalid")
        try:
            await self._ensure_open(event.conversation_id)
            result = await self._archive.archive(event.conversation_id, batch)
        except ArchiveRefusal as refusal:
            return _refused(event, refusal.category)
        except Exception:
            # The commit may or may not have landed; M0 left it for recovery to settle.
            return None
        ack = result.ack
        return VoiceArchiveAckEvent(
            protocol_version="0.3",
            type="voice_archive_ack",
            conversation_id=ack.conversation_id,
            generation=ack.generation,
            seq_from=ack.seq_from,
            seq_through=ack.seq_through,
        )

    @staticmethod
    def _batch(event: VoiceArchiveEvent) -> VoiceBatch | None:
        """M0's types for the batch, or None (with one marker) when a row breaks their rules."""

        evidence: dict[str, str | int] | None = None
        try:
            return VoiceBatch(
                generation=event.generation,
                seq_from=event.seq_from,
                seq_through=event.seq_through,
                rows=tuple(
                    VoiceRow(
                        identity=Identity(event.generation, row.seq),
                        role=row.role,
                        text=row.text,
                        interrupted=row.interrupted,
                        timestamp=row.ts,
                        gap_before=row.gap_before,
                    )
                    for row in event.rows
                ),
            )
        except (TypeError, ValueError):
            evidence = {"refusal": "invalid", "rows": len(event.rows), "version": 1}
            return None
        finally:
            if evidence is not None:
                _marker(evidence)

    async def _ensure_open(self, conversation_id: str) -> None:
        """Open a conversation that is not ready, whatever made it so.

        M0's open re-verifies the archive and settles any pending state, so an unknown
        outcome or a lost lease recovers here; it refuses a quarantined, tombstoned or
        fenced conversation, so nothing under investigation is reopened.
        """

        async with self._open_lock:
            if not self._archive.ready(conversation_id):
                await self._archive.open(conversation_id)

    async def close(self) -> None:
        self._memory_closed = True
        self._memory_changed()
        reads = tuple(self._memory_reads)
        for work in reads:
            work.cancel()
        if reads:
            try:
                _, pending = await asyncio.wait(reads, timeout=_MEMORY_TIMEOUT_SECONDS)
            except asyncio.CancelledError:
                raise ReviewQuiescenceError("memory read has not quiesced") from None
            if pending:
                raise ReviewQuiescenceError("memory read has not quiesced")
        if self._review is not None:
            await self._review.close()
        if self._forget_tasks:
            _, forget_pending = await asyncio.wait(
                self._forget_tasks, timeout=_DEFAULT_CLOSE_TIMEOUT_SECONDS
            )
            if forget_pending:
                raise ReviewQuiescenceError("forget reconciliation has not quiesced")
        await self._archive.close()


class CompanionBridge(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...


class VoiceCompanionHost:
    """Own one profile's companion on a dedicated event-loop thread."""

    def __init__(
        self,
        *,
        store_path: Path,
        open_port: Callable[[], ArchivePort],
        bridge_factory: Callable[[VoiceCompanionService], CompanionBridge],
        lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(store_path, Path):
            raise TypeError("store_path must be a pathlib Path")
        if not callable(open_port) or not callable(bridge_factory):
            raise TypeError("open_port and bridge_factory must be callable")
        self._store_path = store_path
        self._open_port = open_port
        self._bridge_factory = bridge_factory
        self._ttl = lease_ttl_seconds
        self._close_timeout = close_timeout_seconds
        self._state = threading.Lock()
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._service: VoiceCompanionService | None = None
        self._profile_lock: int | None = None

    @property
    def service(self) -> VoiceCompanionService:
        service = self._service
        if service is None or not self._ready.is_set():
            raise RuntimeError("the companion is not ready")
        return service

    def start(self) -> bool:
        """Begin the owned start on the companion thread; readiness comes later.

        Start never waits for another process's profile lock on the plugin thread.
        The owned companion thread waits for it before opening any database.
        """

        global _owner
        with _owner_lock:
            if _owner is not None:
                _marker({"refusal": "multiplexed", "version": 1})
                raise RuntimeError("one companion serves one profile; multiplexing is refused")
            if self._thread is not None:
                raise RuntimeError("a companion host starts at most once")
            _owner = self
            self._thread = threading.Thread(target=self._run, name="voice-companion", daemon=True)
        self._thread.start()
        return True

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def submit(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine on the companion's loop and wait for it (tests and tooling)."""

        loop = self._loop
        if loop is None or not self._ready.is_set():
            raise RuntimeError("the companion is not ready")
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(_SUBMIT_TIMEOUT_SECONDS)

    def close(self) -> None:
        """Stop the bridge, release every lease, close the store; bounded."""

        global _owner
        thread = self._thread
        if thread is None:
            return
        with self._state:
            self._stop_requested.set()
            loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            with contextlib.suppress(RuntimeError):  # The loop already ended.
                loop.call_soon_threadsafe(stop.set)
        thread.join(self._close_timeout)
        if thread.is_alive():
            _marker({"failure": "close_timeout", "version": 1})
            raise RuntimeError("the companion did not stop before the close timeout")
        with _owner_lock:
            profile_lock, self._profile_lock = self._profile_lock, None
            if profile_lock is not None:
                unlock_run_record(profile_lock)  # Last: every lease is released by now.
            if _owner is self:
                _owner = None

    def _run(self) -> None:
        # The marker already carries the evidence; a thread has nobody to raise to.
        with contextlib.suppress(BaseException):
            asyncio.run(self._main())

    async def _main(self) -> None:
        stop = asyncio.Event()
        with self._state:
            self._loop, self._stop = asyncio.get_running_loop(), stop
            if self._stop_requested.is_set():
                stop.set()
        evidence: dict[str, str | int] | None = None
        try:
            profile_lock = await self._wait_for_profile_lock(stop)
            if profile_lock is None:
                return
            self._profile_lock = profile_lock
            port = await asyncio.to_thread(self._open_port)
            try:
                store = CompanionStore(self._store_path)
                try:
                    archive = VoiceArchive(store, port, lease_ttl_seconds=self._ttl)
                    from hermes_realtime.companion.hermes_compat import HermesArchivePort

                    review = (
                        VoiceReviewCoordinator(archive, store, port, port.make_review_parent)
                        if type(port) is HermesArchivePort else None
                    )
                    service = VoiceCompanionService(archive, store, port, review)
                    try:
                        await service.start()
                        if not stop.is_set():
                            await self._serve(service, stop)
                    finally:
                        while True:
                            try:
                                await service.close()
                                break
                            except ReviewQuiescenceError:
                                # An owned review or parent cleanup has not quiesced.
                                # Keep the profile lock, Hermes DB, store and lease alive
                                # until it has; host.close() reports its own timeout.
                                _marker({"refusal": "review_close_pending", "version": 1})
                                await asyncio.sleep(1)
                finally:
                    store.close()
            finally:
                # Last: every lease was released through this database above.
                await asyncio.to_thread(port.close)
        except ArchiveRefusal as refusal:
            evidence = {"refusal": refusal.category, "version": 1}
            raise
        except BaseException as error:
            evidence = {"failure": type(error).__name__, "version": 1}
            raise
        finally:
            self._ready.clear()
            self._service = None
            if evidence is not None:
                _marker(evidence)

    async def _wait_for_profile_lock(self, stop: asyncio.Event) -> int | None:
        """The profile lock, or None on stop; a held lock is awaited, never polled.

        The blocking wait runs on a daemon thread that a stop abandons: if that thread
        acquires the lock afterwards, it releases it at once, so a stop never keeps it.
        """

        held = lock_run_record(self._store_path)
        if held is not None:
            return held
        loop = asyncio.get_running_loop()
        acquired = asyncio.Event()
        handoff = threading.Lock()
        outcome: dict[str, int | bool] = {"abandoned": False}

        def wait() -> None:
            try:
                descriptor = wait_for_run_record_lock(self._store_path)
            except BaseException:
                descriptor = None
            with handoff:
                if outcome["abandoned"]:
                    if descriptor is not None:
                        unlock_run_record(descriptor)
                    return
                if descriptor is not None:
                    outcome["descriptor"] = descriptor
            with contextlib.suppress(RuntimeError):  # The loop already ended.
                loop.call_soon_threadsafe(acquired.set)

        threading.Thread(target=wait, name="voice-companion-lock", daemon=True).start()
        waits = {asyncio.create_task(acquired.wait()), asyncio.create_task(stop.wait())}
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waits:
                task.cancel()
            await asyncio.gather(*waits, return_exceptions=True)
            with handoff:
                descriptor = outcome.get("descriptor")
                keep = descriptor is not None and not stop.is_set()
                outcome["abandoned"] = not keep
        if keep:
            assert type(descriptor) is int
            return descriptor
        if descriptor is not None:
            assert type(descriptor) is int
            unlock_run_record(descriptor)
        if not stop.is_set():
            raise RuntimeError("the profile lock could not be awaited")
        return None

    async def _serve(self, service: VoiceCompanionService, stop: asyncio.Event) -> None:
        bridge = self._bridge_factory(service)
        await bridge.start()
        try:
            self._service = service
            self._ready.set()  # The bridge now advertises voice_archive.
            await stop.wait()
        finally:
            self._ready.clear()
            await bridge.close()


__all__ = [
    "PORT_VARIABLE",
    "TOKEN_VARIABLE",
    "CompanionEndpoint",
    "VoiceCompanionHost",
    "VoiceCompanionService",
    "companion_endpoint",
]
