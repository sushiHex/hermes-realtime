"""Realtime's archive sender: frozen tail batches to the Hermes companion, off the voice path.

One batch is in flight per conversation. The sender takes the tail's frozen batch, sends it
on a connection whose hello advertised ``voice_archive``, and then:

- **acknowledged** with exactly its range: the tail drops it and the cursor advances;
- **unknown** (no connection, no answer in time, a dropped connection, or an answer that
  names another range): the connection is dropped, so no late answer can be misread, and
  the same frozen batch is resent after a bounded backoff;
- **refused, transient** (``VOICE_TRANSIENT_REFUSALS``: the companion cannot take it now,
  and the batch is not wrong): the same frozen batch is retried unchanged after the bounded
  backoff, with one marker per episode, which the next acknowledgment ends;
- **refused, integrity** (every other category, including one neither set lists): archiving
  for this conversation is fenced, with one content-free marker, until this process
  restarts; the outbox keeps its rows under its own bound.

Realtime never reads the voice session, so no resend is ever keyed on a negative read:
the only answer that removes a batch is the companion's acknowledgment of it. Nothing here
is on the voice path: the store's callback only records rows, and this task only waits.
Review and forget are not wired here (milestones M2 and M3).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from collections.abc import Awaitable, Callable
from typing import Protocol

from hermes_realtime.integration.bridge import LocalHermesBridgeClient
from hermes_realtime.integration.voice_tail import ArchiveBatch, VoiceTailWriter
from hermes_realtime.protocol import (
    VOICE_ARCHIVE_CAPABILITY,
    VOICE_TRANSIENT_REFUSALS,
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceArchiveRow,
)

_MARKER = "[voice-archive-send] "
_PARTICIPANT = "voice-archive"
_MAX_TIMEOUT_SECONDS = 3600.0
_MAX_BACKOFF_LIMIT_SECONDS = 300.0


class VoiceArchiveLink(Protocol):
    """One authenticated companion connection."""

    @property
    def capabilities(self) -> frozenset[str]: ...

    async def archive(
        self, event: VoiceArchiveEvent
    ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent: ...

    async def close(self) -> None: ...


def _marker(evidence: dict[str, str | int]) -> None:
    print(_MARKER + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


def archive_event(batch: ArchiveBatch) -> VoiceArchiveEvent:
    """The wire form of a frozen batch: the same rows, byte for byte, every time."""

    return VoiceArchiveEvent(
        type="voice_archive",
        conversation_id=batch.conversation_id,
        generation=batch.generation,
        seq_from=batch.seq_from,
        seq_through=batch.seq_through,
        rows=[
            VoiceArchiveRow(
                seq=row.seq,
                role=row.role,  # type: ignore[arg-type]
                text=row.text,
                interrupted=row.interrupted,
                ts=row.ts,
                gap_before=row.gap_before,
            )
            for row in batch.rows
        ],
    )


def bridge_connector(
    *, host: str, port: int, token: str
) -> Callable[[], Awaitable[VoiceArchiveLink]]:
    """Connect to the companion's bridge, asking for ``voice_archive`` only."""

    async def connect() -> VoiceArchiveLink:
        return await LocalHermesBridgeClient.connect(
            host=host,
            port=port,
            token=token,
            participant_id=_PARTICIPANT,
            capabilities=(VOICE_ARCHIVE_CAPABILITY,),
        )

    return connect


class VoiceArchiveSender:
    """Drain one tail's outbox to the companion; never on the voice path."""

    def __init__(
        self,
        writer: VoiceTailWriter,
        connect: Callable[[], Awaitable[VoiceArchiveLink]],
        *,
        reply_timeout: float = 30.0,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if type(writer) is not VoiceTailWriter:
            raise TypeError("writer must be an exact VoiceTailWriter")
        if not callable(connect) or not callable(sleep):
            raise TypeError("connect and sleep must be callable")
        if not (math.isfinite(reply_timeout) and 0 < reply_timeout <= _MAX_TIMEOUT_SECONDS):
            raise ValueError("reply_timeout must be finite, positive and bounded")
        if not 0 < initial_backoff_seconds <= max_backoff_seconds <= _MAX_BACKOFF_LIMIT_SECONDS:
            raise ValueError("backoff must be positive and bounded")
        self._writer = writer
        self._connect = connect
        self._timeout = float(reply_timeout)
        self._initial_backoff = float(initial_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._link: VoiceArchiveLink | None = None
        self._fence: str | None = None
        self._transient = False
        self.sent = 0
        self.resent = 0

    @property
    def fence(self) -> str | None:
        """The refusal category that fenced archiving, or None."""
        return self._fence

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("the archive sender starts at most once")
        self._task = asyncio.create_task(self._run(), name="voice-archive-sender")

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            # A cancellation can be swallowed by a wait that completed at the same moment;
            # cancel again until the task has ended.
            while not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=1.0)
            await asyncio.gather(task, return_exceptions=True)
        await self._drop()

    async def _drop(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            with contextlib.suppress(Exception):
                await link.close()

    async def _run(self) -> None:
        backoff = self._initial_backoff
        previous: VoiceArchiveEvent | None = None
        while True:
            event = archive_event(await self._writer.next_batch())
            outcome = await self._exchange(event, resend=event == previous)
            previous = event
            if outcome == "acknowledged":
                backoff = self._initial_backoff
                continue
            if outcome == "refused":
                return
            await self._sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    async def _exchange(self, event: VoiceArchiveEvent, *, resend: bool) -> str:
        """Send one batch; return acknowledged, transient, refused, or unknown.

        Transient and unknown both keep the frozen batch for an unchanged retry after the
        bounded backoff; only refused (an integrity category) stops archiving.
        """

        evidence: dict[str, str | int] | None = None
        try:
            if self._link is None:
                try:
                    async with asyncio.timeout(self._timeout):
                        link = await self._connect()
                except Exception as error:  # Any failure leaves the outcome unknown.
                    evidence = {"outcome": "unavailable", "cause": type(error).__name__}
                    return "unknown"
                if VOICE_ARCHIVE_CAPABILITY not in link.capabilities:
                    with contextlib.suppress(Exception):
                        await link.close()
                    evidence = {"refusal": "capability"}
                    return "unknown"
                self._link = link
            self.sent += 1
            self.resent += resend
            try:
                async with asyncio.timeout(self._timeout):
                    reply = await self._link.archive(event)
            except Exception as error:  # Any failure leaves the outcome unknown.
                await self._drop()
                evidence = {"outcome": "unknown", "cause": type(error).__name__}
                return "unknown"
            if type(reply) is VoiceArchiveRefusedEvent and _names(reply, event):
                if reply.category in VOICE_TRANSIENT_REFUSALS:
                    # The companion cannot take it now; the batch is not wrong. One marker
                    # per episode, which the next acknowledgment ends.
                    if not self._transient:
                        evidence = {"transient": reply.category}
                    self._transient = True
                    return "transient"
                # Integrity, or a category neither set lists: fail closed to a fence.
                self._fence = reply.category
                evidence = {"fence": reply.category}
                return "refused"
            if type(reply) is VoiceArchiveAckEvent and self._writer.acknowledge(
                reply.conversation_id, reply.generation, reply.seq_from, reply.seq_through
            ):
                self._transient = False
                return "acknowledged"
            # An answer about another range settles nothing about this one.
            await self._drop()
            evidence = {"outcome": "unknown", "cause": "mismatch"}
            return "unknown"
        finally:
            if evidence is not None:
                _marker(evidence | {"version": 1})


def _names(reply: VoiceArchiveRefusedEvent, event: VoiceArchiveEvent) -> bool:
    return (reply.conversation_id, reply.generation, reply.seq_from, reply.seq_through) == (
        event.conversation_id,
        event.generation,
        event.seq_from,
        event.seq_through,
    )


__all__ = ["VoiceArchiveLink", "VoiceArchiveSender", "archive_event", "bridge_connector"]
