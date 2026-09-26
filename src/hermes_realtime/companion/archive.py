"""The voice archive's commit protocol: integrity, the fence store and Hermes, tied together.

An archive commits write-ahead, and never in a transaction spanning two files:

1. a store transaction asserts no pending state, quarantine or tombstone, then records the
   fingerprint the archive will have next (``pending``);
2. one Hermes transaction checks the lease, verifies the whole archive against
   ``committed`` (or ``pending``: already applied), and appends exactly the extension
   that reaches ``pending``;
3. a store transaction promotes ``pending`` to ``committed``;
4. only then is the batch acknowledged.

A crash anywhere leaves at most one pending state, which startup settles exactly: the archive
still at ``committed`` clears it, the archive at ``pending`` promotes it, and anything else is
quarantined. A quarantine is durable before any refusal it causes is returned; if it cannot
be persisted the whole companion stays fenced.

Startup runs in a fixed order: fences, then compatibility and durability, then the lease
(a fresh holder for every open, after releasing the previous one), then verification, then
readiness. The lease is refreshed at once and then every third of its TTL; a refresh that
fails, raises or breaks fences the conversation. A quarantined or otherwise fenced
conversation keeps its lease, and keeps refreshing it, until it is closed, so ordinary
Hermes turns stay locked out of an archive under investigation.

Hermes runs on worker threads, which asyncio cannot stop. Every Hermes call is therefore
waited for to its end, even when the caller is cancelled, inside the conversation lock the
caller holds: no later step can overlap a Hermes write still in flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from hermes_realtime.companion.integrity import (
    EXPECTED_HEADER,
    MAX_ARCHIVE_ROWS,
    ArchivePlan,
    ArchiveRefusal,
    Fingerprint,
    Projection,
    VoiceBatch,
    expected_after,
    genesis,
    split_batch,
    validate_conversation_id,
)
from hermes_realtime.companion.store import (
    QUARANTINE_CATEGORIES,
    CompanionStore,
    ConversationRecord,
    Progress,
)

DEFAULT_LEASE_TTL_SECONDS = 300.0
# SQLite's PRAGMA synchronous level FULL: every commit is synced before it returns.
DURABLE_SYNCHRONOUS = 2
_MAX_LEASE_TTL_SECONDS = 3600.0
_MAX_CONVERSATIONS = 64
_ARCHIVE_MARKER = "[voice-archive] "
_OPEN_MARKER = "[voice-archive-open] "
_LEASE_MARKER = "[voice-archive-lease] "
# Refusals raised from inside the Hermes transaction, which therefore rolled back: unless
# the refusal says the archive already held pending, it provably still holds ``committed``,
# so the pending state is cleared.
_ROLLED_BACK = frozenset({"conflict", "capacity", "identity", "lease_lost", "drift"})
# Of those, the ones that also end this companion's ownership until it opens again.
_FENCING = frozenset({"lease_lost", "drift"})
_T = TypeVar("_T")


class ArchivePort(Protocol):
    """The Hermes operations the protocol drives; ``hermes_compat`` implements it."""

    def check_compatibility(self) -> tuple[str, ...]: ...

    def durability_level(self) -> int: ...

    def create_session(self, session_id: str) -> None: ...

    def acquire_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool: ...

    def refresh_lease(self, session_id: str, holder: str, ttl_seconds: float) -> bool: ...

    def release_lease(self, session_id: str, holder: str) -> None: ...

    def read_projection(self, session_id: str, cap: int) -> Projection | None: ...

    def archive_rows(
        self,
        session_id: str,
        holder: str,
        conversation_id: str,
        batch: VoiceBatch,
        expected_committed: Fingerprint,
        expected_pending: Fingerprint,
        cap: int,
        lease_ttl_seconds: float,
    ) -> ArchivePlan: ...


@dataclass(frozen=True, slots=True)
class ArchiveAck:
    """The range the archive provably holds: sent only after the commit is promoted."""

    conversation_id: str
    generation: int
    seq_from: int
    seq_through: int


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    """The acknowledgment, and how the commit reached it (content-free evidence)."""

    ack: ArchiveAck
    inserted: int
    already_applied: bool


@dataclass(frozen=True, slots=True)
class OpenReport:
    """How startup settled the conversation: created, none, cleared or promoted."""

    recovery: str


@dataclass(slots=True)
class _Live:
    """One open conversation. ``ready`` admits work; ``leased`` keeps the lease refreshed."""

    session_id: str
    holder: str
    ready: bool = False
    leased: bool = True
    refresh: asyncio.Task[None] | None = None


def _marker(prefix: str, evidence: dict[str, str | int]) -> None:
    print(prefix + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


def _checked_id(conversation_id: str) -> str:
    try:
        return validate_conversation_id(conversation_id)
    except (TypeError, ValueError):
        raise ArchiveRefusal("invalid") from None


class VoiceArchive:
    """Own the archives of a bounded set of voice conversations for one Hermes profile."""

    def __init__(
        self,
        store: CompanionStore,
        port: ArchivePort,
        *,
        cap: int = MAX_ARCHIVE_ROWS,
        lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        max_conversations: int = 16,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if type(store) is not CompanionStore:
            raise TypeError("store must be an exact CompanionStore")
        if type(cap) is not int or type(max_conversations) is not int:
            raise TypeError("cap and max_conversations must be exact ints")
        if type(lease_ttl_seconds) is not float:
            raise TypeError("lease TTL must be an exact float")
        if not 1 <= cap <= MAX_ARCHIVE_ROWS:
            raise ValueError("cap must be between 1 and the provisional archive bound")
        if not (math.isfinite(lease_ttl_seconds) and lease_ttl_seconds > 0):
            raise ValueError("lease TTL must be finite and positive")
        if lease_ttl_seconds > _MAX_LEASE_TTL_SECONDS:
            raise ValueError("lease TTL is out of range")
        if not 1 <= max_conversations <= _MAX_CONVERSATIONS:
            raise ValueError("max_conversations is out of range")
        self._store = store
        self._port = port
        self._cap = cap
        self._ttl = lease_ttl_seconds
        self._max_conversations = max_conversations
        self._sleep = sleep
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}
        self._live: dict[str, _Live] = {}
        self._fenced = False

    def holder(self, conversation_id: str) -> str:
        """The lease holder of this open conversation: unique to this process and this open."""

        live = self._live.get(conversation_id)
        if live is None:
            raise ArchiveRefusal("not_ready")
        return live.holder

    def ready(self, conversation_id: str) -> bool:
        live = self._live.get(conversation_id)
        return not self._fenced and live is not None and live.ready

    @contextlib.asynccontextmanager
    async def _guard(self, conversation_id: str) -> AsyncIterator[None]:
        """Hold the conversation's lock. Its slot is freed once nobody holds, awaits or
        needs it, so the lock a waiter holds is always the conversation's only lock."""

        lock = self._locks.get(conversation_id)
        if lock is None:
            if len(self._locks) >= self._max_conversations:
                raise ArchiveRefusal("conversations")
            lock = self._locks[conversation_id] = asyncio.Lock()
        self._users[conversation_id] = self._users.get(conversation_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._users[conversation_id] -= 1
            if self._users[conversation_id] == 0 and conversation_id not in self._live:
                del self._users[conversation_id]
                del self._locks[conversation_id]

    async def _call(self, function: Callable[..., _T], *arguments: object) -> _T:
        """Run one Hermes call on a worker thread and wait for it to end, even if cancelled.

        The thread cannot be stopped, so a cancelled caller keeps waiting (and keeps the
        conversation lock) until Hermes returns; the cancellation is re-raised after.
        """

        work = asyncio.ensure_future(asyncio.to_thread(function, *arguments))
        cancelled = False
        while not work.done():
            try:
                await asyncio.wait({work})
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            if not work.cancelled():
                work.exception()  # Retrieved: the outcome is left for recovery to settle.
            raise asyncio.CancelledError
        return work.result()

    # --- startup ---------------------------------------------------------------------------

    async def open(self, conversation_id: str) -> OpenReport:
        """Fences, then compatibility, then the lease, then verification, then readiness."""

        evidence: dict[str, str | int] | None = None
        try:
            conversation_id = _checked_id(conversation_id)
            async with self._guard(conversation_id):
                return await self._open(conversation_id)
        except ArchiveRefusal as refusal:
            evidence = {"refusal": refusal.category, "version": 1}
            raise
        except BaseException as error:
            evidence = {"failure": type(error).__name__, "version": 1}
            raise
        finally:
            if evidence is not None:
                _marker(_OPEN_MARKER, evidence)

    async def _open(self, conversation_id: str) -> OpenReport:
        if self._fenced:
            raise ArchiveRefusal("fenced")
        current = self._live.get(conversation_id)
        if current is not None and current.ready:
            raise ArchiveRefusal("bound")
        record = self._store.read(conversation_id)
        if record is None:
            record = self._store.bind(
                conversation_id,
                f"voice_{uuid.uuid4().hex}",
                Progress(genesis(EXPECTED_HEADER), None),
            )
        if record.quarantine is not None:
            raise ArchiveRefusal("quarantined")
        if record.tombstone is not None:
            raise ArchiveRefusal("tombstoned")
        if await self._call(self._port.check_compatibility):
            raise ArchiveRefusal("incompatible")
        if await self._call(self._port.durability_level) < DURABLE_SYNCHRONOUS:
            raise ArchiveRefusal("durability")
        # The previous open's holder, in this process or a dead one, gives the lease up first,
        # so nothing it may still have in flight can pass the lease guard again.
        if current is not None:
            del self._live[conversation_id]
            await self._release(current)
        elif record.holder is not None:
            await self._call(self._port.release_lease, record.session_id, record.holder)
        holder = f"pid={os.getpid()}:voice={conversation_id}:boot={uuid.uuid4().hex}"
        self._store.set_holder(conversation_id, holder)
        acquired = await self._call(
            self._port.acquire_lease, record.session_id, holder, self._ttl
        )
        if acquired is not True:
            raise ArchiveRefusal("lease_held")
        live = _Live(record.session_id, holder)
        self._live[conversation_id] = live
        try:
            recovery = await self._verify(conversation_id, record)
        except BaseException:
            del self._live[conversation_id]
            # The refusal already carries the evidence; an unreleased lease expires by TTL.
            with contextlib.suppress(Exception):
                await self._release(live)
            raise
        live.ready = True
        live.refresh = asyncio.create_task(self._refresh(live), name="voice-archive-lease")
        return OpenReport(recovery)

    async def _read(self, session_id: str) -> Projection | None:
        return await self._call(self._port.read_projection, session_id, self._cap)

    async def _verify(self, conversation_id: str, record: ConversationRecord) -> str:
        try:
            return await self._settle(conversation_id, record)
        except ArchiveRefusal as refusal:
            if refusal.category in QUARANTINE_CATEGORIES:
                self._quarantine(conversation_id, refusal.category)
            raise

    async def _settle(self, conversation_id: str, record: ConversationRecord) -> str:
        """Settle any pending state against a fresh projection, or refuse to quarantine."""

        committed, pending = record.committed, record.pending
        projection = await self._read(record.session_id)
        created = False
        if committed is None and projection is None:
            # The recorded creation never happened; it is the only archive ever created.
            await self._call(self._port.create_session, record.session_id)
            projection = await self._read(record.session_id)
            created = True
        current = None if projection is None else projection.fingerprint()
        if pending is not None:
            if committed is not None and current == committed.fingerprint:
                self._store.clear_pending(conversation_id, pending)
                return "cleared"
            if current == pending.fingerprint:
                self._store.promote(conversation_id, pending)
                return "created" if created else "promoted"
            raise ArchiveRefusal("missing" if projection is None else "recovery")
        if committed is None:
            # The store's constraints forbid a record with no progress at all.
            raise ArchiveRefusal("recovery")
        if current != committed.fingerprint:
            raise ArchiveRefusal("missing" if projection is None else "mismatch")
        return "none"

    def _quarantine(self, conversation_id: str, category: str) -> None:
        """Persist the fence first; a refusal is only returned once the fence is durable.

        The conversation stops admitting work but keeps its lease until it is closed.
        """

        live = self._live.get(conversation_id)
        if live is not None:
            live.ready = False
        try:
            self._store.quarantine(conversation_id, category)
        except BaseException:
            self._fenced = True
            raise RuntimeError("the quarantine could not be persisted; companion fenced") from None

    # --- archiving -------------------------------------------------------------------------

    async def archive(self, conversation_id: str, batch: VoiceBatch) -> ArchiveResult:
        """Commit one batch, or refuse it whole; see the module docstring."""

        rows = batch.rows if type(batch) is VoiceBatch else ()
        count = len(rows) if type(rows) is tuple else 0
        evidence: dict[str, str | int] | None = None
        try:
            conversation_id = _checked_id(conversation_id)
            async with self._guard(conversation_id):
                return await self._archive(conversation_id, batch)
        except ArchiveRefusal as refusal:
            evidence = {"refusal": refusal.category, "rows": count, "version": 1}
            raise
        except BaseException as error:
            evidence = {"failure": type(error).__name__, "rows": count, "version": 1}
            raise
        finally:
            if evidence is not None:
                _marker(_ARCHIVE_MARKER, evidence)

    async def _archive(self, conversation_id: str, batch: VoiceBatch) -> ArchiveResult:
        if self._fenced:
            raise ArchiveRefusal("fenced")
        live = self._live.get(conversation_id)
        if live is None or not live.ready:
            raise ArchiveRefusal("not_ready")
        record = self._store.read(conversation_id)
        if record is None or record.committed is None:
            raise ArchiveRefusal("unbound")
        committed = record.committed
        split = split_batch(committed.cursor, batch)
        pending = Progress(
            expected_after(committed.fingerprint, conversation_id, split.new),
            split.new[-1].identity if split.new else committed.cursor,
        )
        self._store.begin_pending(conversation_id, committed, pending)
        try:
            plan = await self._call(
                self._port.archive_rows,
                live.session_id,
                live.holder,
                conversation_id,
                batch,
                committed.fingerprint,
                pending.fingerprint,
                self._cap,
                self._ttl,
            )
        except ArchiveRefusal as refusal:
            if refusal.category in QUARANTINE_CATEGORIES:
                self._quarantine(conversation_id, refusal.category)
            elif refusal.at_pending:
                # The archive already holds pending: keep it for recovery, admit nothing more.
                live.ready = False
            elif refusal.category in _ROLLED_BACK:
                try:
                    self._store.clear_pending(conversation_id, pending)
                except BaseException:
                    live.ready = False
                    raise
                if refusal.category in _FENCING:
                    live.ready = False
            else:
                live.ready = False
            raise
        except BaseException:
            # The outcome is unknown: pending stays for startup recovery to settle.
            live.ready = False
            raise
        try:
            self._store.promote(conversation_id, pending)
        except BaseException:
            live.ready = False
            raise
        return ArchiveResult(
            ack=ArchiveAck(
                conversation_id=conversation_id,
                generation=batch.generation,
                seq_from=batch.seq_from,
                seq_through=batch.seq_through,
            ),
            inserted=len(plan.inserts),
            already_applied=plan.already_applied,
        )

    # --- lease -----------------------------------------------------------------------------

    async def _refresh(self, live: _Live) -> None:
        """Refresh now, then every TTL/3, for as long as the lease is held."""

        evidence: dict[str, str | int] | None = None
        try:
            while live.leased:
                try:
                    kept = await self._call(
                        self._port.refresh_lease, live.session_id, live.holder, self._ttl
                    )
                except Exception:
                    kept, cause = False, "raised"
                else:
                    cause = "lost"
                if kept is not True:
                    live.ready = live.leased = False
                    evidence = {"fence": cause, "version": 1}
                    return
                await self._sleep(self._ttl / 3)
        except Exception:
            live.ready = live.leased = False
            evidence = {"fence": "error", "version": 1}
        finally:
            if evidence is not None:
                _marker(_LEASE_MARKER, evidence)

    async def _release(self, live: _Live) -> None:
        live.ready = live.leased = False
        task, live.refresh = live.refresh, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._call(self._port.release_lease, live.session_id, live.holder)

    async def close(self) -> None:
        """Stop every conversation's work and release its lease."""

        failures: list[BaseException] = []
        for conversation_id in list(self._live):
            async with self._guard(conversation_id):
                live = self._live.pop(conversation_id, None)
                if live is None:
                    continue
                try:
                    await self._release(live)
                except Exception as error:
                    failures.append(error)
        if failures:
            raise failures[0]
