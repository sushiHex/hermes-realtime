"""Hosting for the voice companion inside the Hermes process.

Plugin registration builds a ``VoiceCompanionHost``; that is not readiness. Its owned start
runs on one dedicated event-loop thread, so it neither needs nor blocks Hermes's own loop:

1. it binds exactly one profile's database (``open_port``) and the plugin store;
2. it runs M0's open order: the fences first, then compatibility and durability for the
   whole profile, then, for every conversation the store already binds, M0's own open
   (fences, compatibility and durability, the lease, verification);
3. only then does it build and start the bridge, which advertises ``voice_archive``.

Unload closes the bridge, then releases every lease, then the store. One companion serves
one profile: a second owned start in the same process is refused (no multiplexing).

Archive events are served through M0's ``VoiceArchive``: a batch is validated into M0's
types (a malformed one is refused whole), a conversation the store does not yet bind is
opened on first contact (M0 binds it, creates its session, leases and verifies it), and
the answer is an acknowledgment of the exact range, a category refusal, or nothing when
the outcome is unknown.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from hermes_realtime.companion.archive import (
    DEFAULT_LEASE_TTL_SECONDS,
    DURABLE_SYNCHRONOUS,
    ArchivePort,
    VoiceArchive,
)
from hermes_realtime.companion.integrity import (
    ArchiveRefusal,
    Identity,
    VoiceBatch,
    VoiceRow,
)
from hermes_realtime.companion.store import CompanionStore
from hermes_realtime.protocol import (
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
)

PORT_VARIABLE = "HERMES_REALTIME_COMPANION_PORT"
TOKEN_VARIABLE = "HERMES_REALTIME_COMPANION_TOKEN"
_MIN_TOKEN_CHARS = 24
_MAX_TOKEN_CHARS = 512
_MAX_CONVERSATIONS = 16
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 15.0
_SUBMIT_TIMEOUT_SECONDS = 60.0
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
        *,
        max_conversations: int = _MAX_CONVERSATIONS,
    ) -> None:
        if type(archive) is not VoiceArchive or type(store) is not CompanionStore:
            raise TypeError("the service needs an exact VoiceArchive and CompanionStore")
        self._archive = archive
        self._store = store
        self._port = port
        self._max_conversations = max_conversations
        self._opened: set[str] = set()
        self._open_lock = asyncio.Lock()

    async def start(self) -> None:
        """Fences, then compatibility and durability, then each known conversation's open."""

        known = self._store.conversation_ids(self._max_conversations)
        if await asyncio.to_thread(self._port.check_compatibility):
            raise ArchiveRefusal("incompatible")
        if await asyncio.to_thread(self._port.durability_level) < DURABLE_SYNCHRONOUS:
            raise ArchiveRefusal("durability")
        for conversation_id in known:
            # A quarantined or otherwise refused conversation stays closed and is refused
            # on contact; it never keeps the others from their archive.
            try:
                await self._archive.open(conversation_id)
            except ArchiveRefusal:
                continue
            self._opened.add(conversation_id)

    async def archive(
        self, event: VoiceArchiveEvent
    ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent | None:
        if type(event) is not VoiceArchiveEvent:
            raise TypeError("event must be an exact VoiceArchiveEvent")
        try:
            batch = VoiceBatch(
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
            type="voice_archive_ack",
            conversation_id=ack.conversation_id,
            generation=ack.generation,
            seq_from=ack.seq_from,
            seq_through=ack.seq_through,
        )

    async def _ensure_open(self, conversation_id: str) -> None:
        """Open a conversation this process has not owned yet; never re-open a fenced one."""

        async with self._open_lock:
            if conversation_id in self._opened:
                return
            await self._archive.open(conversation_id)
            self._opened.add(conversation_id)

    async def close(self) -> None:
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

    @property
    def service(self) -> VoiceCompanionService:
        service = self._service
        if service is None or not self._ready.is_set():
            raise RuntimeError("the companion is not ready")
        return service

    def start(self) -> None:
        """Begin the owned start on the companion thread; readiness comes later."""

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
            port = await asyncio.to_thread(self._open_port)
            try:
                store = CompanionStore(self._store_path)
                try:
                    archive = VoiceArchive(store, port, lease_ttl_seconds=self._ttl)
                    service = VoiceCompanionService(archive, store, port)
                    try:
                        await service.start()
                        if not stop.is_set():
                            await self._serve(service, stop)
                    finally:
                        await service.close()
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
