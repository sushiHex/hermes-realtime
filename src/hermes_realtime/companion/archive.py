"""The voice archive's commit protocol: integrity, the fence store and Hermes, tied together.

An archive commits write-ahead, and never in a transaction spanning two files:

1. a store transaction asserts no pending state, quarantine or tombstone, then records the
   fingerprint the archive will have next (``pending``);
2. one Hermes transaction checks the lease, verifies the whole archive against
   ``committed`` (or ``pending``: already applied), and appends only the missing rows;
3. a store transaction promotes ``pending`` to ``committed``;
4. only then is the batch acknowledged.

A crash anywhere leaves at most one pending state, which startup settles exactly: the archive
still at ``committed`` clears it, the archive at ``pending`` promotes it, and anything else is
quarantined. A quarantine is durable before any refusal it causes is returned; if it cannot
be persisted the whole companion stays fenced.

Startup runs in a fixed order: fences, then the lease (a fresh holder per process), then
verification, then readiness. The lease is refreshed every third of its TTL, and a refresh
that fails or raises fences the conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

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
_MAX_LEASE_TTL_SECONDS = 3600.0
_MAX_CONVERSATIONS = 64
_ARCHIVE_MARKER = "[voice-archive] "
_OPEN_MARKER = "[voice-archive-open] "
_LEASE_MARKER = "[voice-archive-lease] "
# Refusals raised from inside the Hermes transaction, which therefore rolled back: the
# archive provably still holds ``committed``, so the pending state is cleared.
_ROLLED_BACK = frozenset({"conflict", "capacity", "identity", "lease_lost", "drift"})
# Of those, the ones that also end this companion's ownership until it opens again.
_FENCING = frozenset({"lease_lost", "drift"})


class ArchivePort(Protocol):
    """The Hermes operations the protocol drives; ``hermes_compat`` implements it."""

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
    session_id: str
    holder: str
    ready: bool = False
    refresh: asyncio.Task[None] | None = None


def _marker(prefix: str, evidence: dict[str, str | int]) -> None:
    print(prefix + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


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
        # One per process instance: a restarted companion never presents a stale holder.
        self._boot = uuid.uuid4().hex
        self._locks: dict[str, asyncio.Lock] = {}
        self._live: dict[str, _Live] = {}
        self._fenced = False

    def holder(self, conversation_id: str) -> str:
        validate_conversation_id(conversation_id)
        return f"pid={os.getpid()}:voice={conversation_id}:boot={self._boot}"

    def ready(self, conversation_id: str) -> bool:
        live = self._live.get(conversation_id)
        return not self._fenced and live is not None and live.ready

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            if len(self._locks) >= self._max_conversations:
                raise ArchiveRefusal("conversations")
            lock = self._locks[conversation_id] = asyncio.Lock()
        return lock

    # --- startup ---------------------------------------------------------------------------

    async def open(self, conversation_id: str) -> OpenReport:
        """Fences, then the lease, then verification, then readiness."""

        validate_conversation_id(conversation_id)
        evidence: dict[str, str | int] | None = None
        try:
            async with self._lock(conversation_id):
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
        holder = self.holder(conversation_id)
        self._store.set_holder(conversation_id, holder)
        acquired = await asyncio.to_thread(
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
        return await asyncio.to_thread(self._port.read_projection, session_id, self._cap)

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
            await asyncio.to_thread(self._port.create_session, record.session_id)
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
        # The store's CHECK constraint: a record without committed state always has pending.
        assert committed is not None
        if current != committed.fingerprint:
            raise ArchiveRefusal("missing" if projection is None else "mismatch")
        return "none"

    def _quarantine(self, conversation_id: str, category: str) -> None:
        """Persist the fence first; a refusal is only returned once the fence is durable."""

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

        validate_conversation_id(conversation_id)
        rows = batch.rows if type(batch) is VoiceBatch else ()
        count = len(rows) if type(rows) is tuple else 0
        evidence: dict[str, str | int] | None = None
        try:
            async with self._lock(conversation_id):
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
            plan = await asyncio.to_thread(
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
            elif refusal.category in _ROLLED_BACK:
                self._store.clear_pending(conversation_id, pending)
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
        evidence: dict[str, str | int] | None = None
        try:
            while live.ready:
                await self._sleep(self._ttl / 3)
                if not live.ready:
                    return
                try:
                    kept = await asyncio.to_thread(
                        self._port.refresh_lease, live.session_id, live.holder, self._ttl
                    )
                except Exception:
                    kept, cause = False, "raised"
                else:
                    cause = "lost"
                if kept is not True:
                    live.ready = False
                    evidence = {"fence": cause, "version": 1}
                    return
        finally:
            if evidence is not None:
                _marker(_LEASE_MARKER, evidence)

    async def _release(self, live: _Live) -> None:
        live.ready = False
        task, live.refresh = live.refresh, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(self._port.release_lease, live.session_id, live.holder)

    async def close(self) -> None:
        """Stop every conversation's work and release its lease."""

        failures: list[BaseException] = []
        for conversation_id in list(self._live):
            async with self._locks[conversation_id]:
                live = self._live.pop(conversation_id, None)
                if live is None:
                    continue
                try:
                    await self._release(live)
                except Exception as error:
                    failures.append(error)
        if failures:
            raise failures[0]
