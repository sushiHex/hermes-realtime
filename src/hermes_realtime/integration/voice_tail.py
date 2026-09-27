"""Durable, bounded snapshot of the heard voice conversation, and its archive outbox.

The context store owns the truth and does no I/O. This module mirrors its durable view,
off the voice path, into one file that the next start restores before its first turn.
The file is plaintext user data; it holds only rows the store would accept.

Version 2 adds the archive outbox to the same file, under the same lock and atomic writes:

- every closed row gets an immutable identity, ``(generation, seq)``, and the wall-clock
  time it closed; a row still in progress never gets one;
- a batch is frozen into the file before it is sent, and stays exactly as frozen until an
  acknowledgment of exactly its range removes it; a batch is returned for sending only
  once a write taken after the freeze has completed, so that one write proves both the
  frozen batch and every row in it durable (a row is eligible only once a completed write
  holds it);
- on overflow only rows never sent are discarded, from the oldest up to the next user row,
  and the discarded interval travels on that user row as ``gap_before`` (or waits in the
  file as a trailing gap until a user row arrives);
- the cursor is the last seq the acknowledged rows and their gaps cover.

A version-1 tail migrates on the first write: its rows restore, then close, and so get
identities like any other row, with the restart's time as their close time (version 1 never
recorded one). Nothing is refused for being version 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from hermes_realtime.companion.integrity import validate_voice_row
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.integration.run_record import (
    lock_run_record,
    read_run_record,
    remove_orphaned_temporaries,
    strict_object,
    unlock_run_record,
    write_run_record,
)

_VERSION = 2
_LEGACY_VERSION = 1
_MARKER_PREFIX = "[voice-tail] "
_LOCK_MARKER_PREFIX = "[voice-tail-lock] "
_OUTBOX_MARKER_PREFIX = "[voice-tail-outbox] "
# ASCII-escaped JSON spends at most twelve bytes on one str character (an astral
# character becomes a surrogate-pair escape), plus a fixed per-row and envelope cost.
_MAX_BYTES_PER_CHAR = 12
_ROW_OVERHEAD_BYTES = 64
_OUTBOX_ROW_OVERHEAD_BYTES = 192
_ENVELOPE_OVERHEAD_BYTES = 64
_ARCHIVE_OVERHEAD_BYTES = 320
_LEGACY_FIELDS = frozenset({"messages", "prior_work", "version"})
_DOCUMENT_FIELDS = frozenset({"archive", "messages", "prior_work", "version"})
_ROW_FIELDS = frozenset({"interrupted", "role", "text"})
_ARCHIVE_FIELDS = frozenset(
    {"conversation_id", "cursor", "frozen", "gap", "generation", "next_seq", "outbox", "settled"}
)
_OUTBOX_FIELDS = frozenset({"gap_before", "interrupted", "role", "seq", "text", "ts"})
_CONVERSATION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_IDENTITY = 2**53 - 1
DEFAULT_MAX_OUTBOX_ROWS = 128
DEFAULT_MAX_BATCH_ROWS = 32
# A batch travels as one bridge line (64 KiB); a row costs at most six wire bytes per
# character (a control character's escape) plus its fields.
MAX_BATCH_WIRE_BYTES = 48 * 1024
_MAX_WIRE_BYTES_PER_CHAR = 6
_WIRE_ROW_OVERHEAD_BYTES = 160
_MAX_BATCH_ROWS_LIMIT = 256
_MAX_OUTBOX_ROWS_LIMIT = 4096
_MAX_BACKOFF_LIMIT_SECONDS = 60.0
# A to_thread write cannot be cancelled, so close waits this long for the writer to finish.
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 10.0
_CONCRETE_PATH = type(Path())
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OutboxRow:
    """One closed row awaiting the archive, exactly as it closed."""

    seq: int
    role: str
    text: str
    interrupted: bool
    ts: float
    gap_before: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ArchiveBatch:
    """One frozen batch: its rows and their gaps cover ``[seq_from, seq_through]``."""

    conversation_id: str
    generation: int
    seq_from: int
    seq_through: int
    rows: tuple[OutboxRow, ...]


@dataclass(frozen=True, slots=True)
class ArchiveOutbox:
    """The archive section of a version-2 tail.

    ``settled`` leading rows of the tail's messages already hold identities, the last of
    them ``next_seq - 1``. ``frozen`` leading outbox rows are the batch in flight.
    """

    conversation_id: str
    generation: int
    next_seq: int
    settled: int
    cursor: int | None
    rows: tuple[OutboxRow, ...]
    frozen: int
    gap: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class VoiceTail:
    """A parsed tail. ``archive`` is None only for a version-1 tail, which migrates."""

    conversation: DurableConversation
    archive: ArchiveOutbox | None


def max_voice_tail_bytes(
    max_messages: int,
    max_item_chars: int,
    max_outbox_rows: int = DEFAULT_MAX_OUTBOX_ROWS,
) -> int:
    """The largest tail a store and an outbox with these bounds can produce."""
    return (
        max_messages * (_ROW_OVERHEAD_BYTES + _MAX_BYTES_PER_CHAR * max_item_chars)
        + max_outbox_rows * (_OUTBOX_ROW_OVERHEAD_BYTES + _MAX_BYTES_PER_CHAR * max_item_chars)
        + _ENVELOPE_OVERHEAD_BYTES
        + _ARCHIVE_OVERHEAD_BYTES
    )


def row_wire_bound(text: str) -> int:
    """An upper bound on one row's bytes in a ``voice_archive`` line."""
    return _WIRE_ROW_OVERHEAD_BYTES + _MAX_WIRE_BYTES_PER_CHAR * len(text)


def voice_tail_bytes(view: DurableConversation, archive: ArchiveOutbox) -> bytes:
    document = {
        "archive": {
            "conversation_id": archive.conversation_id,
            "cursor": archive.cursor,
            "frozen": archive.frozen,
            "gap": None if archive.gap is None else list(archive.gap),
            "generation": archive.generation,
            "next_seq": archive.next_seq,
            "outbox": [
                {
                    "gap_before": None if row.gap_before is None else list(row.gap_before),
                    "interrupted": row.interrupted,
                    "role": row.role,
                    "seq": row.seq,
                    "text": row.text,
                    "ts": row.ts,
                }
                for row in archive.rows
            ],
            "settled": archive.settled,
        },
        "messages": [
            {"interrupted": message.interrupted, "role": message.role, "text": message.text}
            for message in view.messages
        ],
        "prior_work": view.prior_work,
        "version": _VERSION,
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _identity(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_IDENTITY


def _text(role: object, text: object, interrupted: object, max_item_chars: int) -> bool:
    if type(role) is not str or type(text) is not str or type(interrupted) is not bool:
        return False
    if len(text) > max_item_chars:
        return False
    try:
        # A lone surrogate decodes from JSON but is not text the store can hold.
        text.encode("utf-8")
        ConversationMessage(role=role, text=text, interrupted=interrupted)
    except (ValueError, RuntimeError):
        # Unknown roles, flagged user rows, blank text, and private run tokens.
        return False
    return True


class _Malformed(ValueError):
    """One field of an archive section is malformed."""


def _interval(value: object) -> tuple[int, int] | None:
    """A gap as ``(first, last)``, or None when absent; raises when malformed."""
    if value is None:
        return None
    if type(value) is not list or len(value) != 2 or not all(_identity(end) for end in value):
        raise _Malformed("a gap must be two identities")
    first, last = value
    if first > last:
        raise _Malformed("a gap must not be reversed")
    return (first, last)


def _parse_outbox_row(row: object, max_item_chars: int) -> OutboxRow | None:
    if type(row) is not dict or set(row) != _OUTBOX_FIELDS:
        return None
    seq, ts = row["seq"], row["ts"]
    try:
        gap = _interval(row["gap_before"])
    except _Malformed:
        return None
    if not _identity(seq):
        return None
    if not _text(row["role"], row["text"], row["interrupted"], max_item_chars):
        return None
    try:
        # The companion's own row rules: a row it would refuse never sits in the outbox.
        validate_voice_row(row["role"], row["text"], row["interrupted"], ts, gap)
    except (TypeError, ValueError):
        return None
    return OutboxRow(seq, row["role"], row["text"], row["interrupted"], ts, gap)


def _parse_archive(
    archive: object, messages: int, max_item_chars: int, max_outbox_rows: int
) -> ArchiveOutbox | None:
    if type(archive) is not dict or set(archive) != _ARCHIVE_FIELDS:
        return None
    conversation_id = archive["conversation_id"]
    generation, next_seq, settled = archive["generation"], archive["next_seq"], archive["settled"]
    cursor, frozen = archive["cursor"], archive["frozen"]
    try:
        gap = _interval(archive["gap"])
    except _Malformed:
        return None
    if type(conversation_id) is not str or _CONVERSATION_ID.fullmatch(conversation_id) is None:
        return None
    if not (_identity(generation) and _identity(next_seq) and _identity(settled)):
        return None
    if settled > messages or settled > next_seq:
        return None
    if cursor is not None and not _identity(cursor):
        return None
    raw_rows = archive["outbox"]
    if type(raw_rows) is not list or len(raw_rows) > max_outbox_rows:
        return None
    rows: list[OutboxRow] = []
    for raw_row in raw_rows:
        row = _parse_outbox_row(raw_row, max_item_chars)
        if row is None:
            return None
        rows.append(row)
    if type(frozen) is not int or not 0 <= frozen <= min(len(rows), _MAX_BATCH_ROWS_LIMIT):
        return None
    # Acknowledged rows, outbox rows, carried gaps and the trailing gap partition every seq
    # ever assigned, with no hole and no overlap.
    expected = 0 if cursor is None else cursor + 1
    for row in rows:
        start = row.seq if row.gap_before is None else row.gap_before[0]
        if start != expected or (row.gap_before is not None and row.gap_before[1] + 1 != row.seq):
            return None
        expected = row.seq + 1
    if gap is not None:
        if gap[0] != expected:
            return None
        expected = gap[1] + 1
    if expected != next_seq:
        return None
    return ArchiveOutbox(
        conversation_id, generation, next_seq, settled, cursor, tuple(rows), frozen, gap
    )


def parse_voice_tail(
    raw: bytes,
    *,
    max_messages: int,
    max_item_chars: int,
    max_outbox_rows: int = DEFAULT_MAX_OUTBOX_ROWS,
) -> VoiceTail | None:
    """Return the tail, or None when it is malformed, oversized, or an unknown version."""
    if len(raw) > max_voice_tail_bytes(max_messages, max_item_chars, max_outbox_rows):
        return None
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if type(document) is not dict:
        return None
    version = document.get("version")
    if type(version) is not int or version not in (_LEGACY_VERSION, _VERSION):
        return None
    if set(document) != (_LEGACY_FIELDS if version == _LEGACY_VERSION else _DOCUMENT_FIELDS):
        return None
    rows, prior_work = document["messages"], document["prior_work"]
    if type(prior_work) is not bool:
        return None
    if type(rows) is not list or len(rows) > max_messages:
        return None
    messages: list[ConversationMessage] = []
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_FIELDS:
            return None
        role, text, interrupted = row["role"], row["text"], row["interrupted"]
        # The store checks its own per-item bound on restore; a refusal there is its own.
        if not _text(role, text, interrupted, len(text) if type(text) is str else 0):
            return None
        messages.append(ConversationMessage(role=role, text=text, interrupted=interrupted))
    archive = None
    if version == _VERSION:
        archive = _parse_archive(
            document["archive"], len(messages), max_item_chars, max_outbox_rows
        )
        if archive is None:
            return None
    return VoiceTail(DurableConversation(messages=tuple(messages), prior_work=prior_work), archive)


def _marker(prefix: str, evidence: dict[str, str | int]) -> None:
    print(prefix + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


def _new_conversation_id() -> str:
    return uuid.uuid4().hex


class VoiceTailWriter:
    """Own one tail file: restore it once, then mirror every change with the latest winning.

    ``update`` is the store's ``on_change``: synchronous, and linear only in the rows that
    closed since the last call. One task owns every write, the final one included, so a
    stale snapshot can never land after a newer one. A failed write backs off within a
    bound and retries with whatever is latest by then; closing never cuts that short, it
    only lets the task finish once the tail is clean.

    The outbox is driven by one archive sender: ``next_batch`` freezes (or returns the
    already frozen) batch once the file holds it, and ``acknowledge`` removes exactly it.
    """

    def __init__(
        self,
        path: Path,
        *,
        initial_backoff_seconds: float = 0.05,
        max_backoff_seconds: float = 2.0,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
        max_outbox_rows: int = DEFAULT_MAX_OUTBOX_ROWS,
        max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS,
        clock: Callable[[], float] = time.time,
        conversation_ids: Callable[[], str] = _new_conversation_id,
    ) -> None:
        if type(path) is not _CONCRETE_PATH:
            raise TypeError("voice tail path must be an exact pathlib Path")
        if not 0 < initial_backoff_seconds <= max_backoff_seconds <= _MAX_BACKOFF_LIMIT_SECONDS:
            raise ValueError("voice tail backoff must be positive and bounded")
        if not (math.isfinite(close_timeout_seconds) and close_timeout_seconds > 0):
            raise ValueError("voice tail close timeout must be finite and positive")
        if type(max_outbox_rows) is not int or type(max_batch_rows) is not int:
            raise TypeError("outbox bounds must be exact integers")
        # A batch always leaves room for a new row, so overflow can always discard one.
        if not 1 <= max_batch_rows < max_outbox_rows <= _MAX_OUTBOX_ROWS_LIMIT:
            raise ValueError("outbox bounds must satisfy 1 <= batch < outbox <= 4096")
        if max_batch_rows > _MAX_BATCH_ROWS_LIMIT:
            raise ValueError("a batch holds at most 256 rows")
        if not callable(clock) or not callable(conversation_ids):
            raise TypeError("clock and conversation_ids must be callable")
        self._path = path
        self._initial_backoff = float(initial_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._close_timeout = float(close_timeout_seconds)
        self._max_outbox = max_outbox_rows
        self._max_batch = max_batch_rows
        self._clock = clock
        self._conversation_ids = conversation_ids
        self._latest = DurableConversation(messages=(), prior_work=False)
        self._dirty = False
        self._wake = asyncio.Event()
        self._close_requested = asyncio.Event()
        self._opened = False
        self._owner: int | None = None
        self._task: asyncio.Task[None] | None = None
        # The archive outbox; identities are assigned only once the tail is open.
        self._archiving = False
        self._conversation_id = ""
        self._generation = 0
        self._seq_base = 0
        self._next_seq = 0
        self._cursor: int | None = None
        self._outbox: list[OutboxRow] = []
        self._frozen = 0
        self._gap: tuple[int, int] | None = None
        self._version = 0
        self._frozen_version = 0
        self._written_version = 0
        self._tick = asyncio.Event()
        self._discard_episode = False

    # --- the store's callback ----------------------------------------------------------------

    def update(self, view: DurableConversation) -> None:
        if type(view) is not DurableConversation:
            raise TypeError("voice tail updates must be an exact DurableConversation")
        self._latest = view
        if self._archiving:
            self._assign(view)
        self._changed()

    def _assign(self, view: DurableConversation) -> None:
        """Give every row that closed since the last view its identity and close time."""

        closed_end = view.first + len(view.messages) - view.unsettled
        start = self._next_seq - self._seq_base
        if closed_end <= start:
            return
        now = self._clock()
        for ordinal in range(start, closed_end):
            seq = self._seq_base + ordinal
            if ordinal < view.first:
                # Evicted before this writer saw it close: it cannot be archived.
                self._discard(seq, "evicted")
                continue
            message = view.messages[ordinal - view.first]
            try:
                # The companion's own row rules, before the row can become eligible.
                validate_voice_row(message.role, message.text, message.interrupted, now)
            except (TypeError, ValueError):
                self._discard(seq, "invalid")
                continue
            self._admit(OutboxRow(seq, message.role, message.text, message.interrupted, now))
        self._next_seq = self._seq_base + closed_end

    def _admit(self, row: OutboxRow) -> None:
        if len(self._outbox) >= self._max_outbox and not self._overflow():
            self._discard(row.seq, "overflow")
            return
        if self._gap is not None:
            if row.role != "user":
                # Archiving resumes only at a user row: never a reply without its question.
                self._gap = (self._gap[0], row.seq)
                return
            row = replace(row, gap_before=self._gap)
            self._gap = None
        self._outbox.append(row)

    def _discard(self, seq: int, cause: str) -> None:
        self._gap = (seq if self._gap is None else self._gap[0], seq)
        self._discarding(cause)

    def _discarding(self, cause: str) -> None:
        """One marker per discard episode, which the next acknowledgment ends.

        It runs inside the store's callback, so it never raises: speech outranks evidence.
        """

        if self._discard_episode:
            return
        self._discard_episode = True
        with contextlib.suppress(Exception):
            _marker(_OUTBOX_MARKER_PREFIX, {"cause": cause, "version": 1})

    def _overflow(self) -> bool:
        """Discard the oldest unsent rows up to the next user row; False if none is unsent."""

        unsent = self._outbox[self._frozen :]
        if not unsent:
            return False
        count = 1
        while count < len(unsent) and unsent[count].role != "user":
            count += 1
        dropped, kept = unsent[:count], unsent[count:]
        head = dropped[0]
        start = head.seq if head.gap_before is None else head.gap_before[0]
        if kept:
            kept[0] = replace(kept[0], gap_before=(start, kept[0].seq - 1))
        elif self._gap is None:
            self._gap = (start, dropped[-1].seq)
        else:
            self._gap = (start, self._gap[1])
        self._outbox[self._frozen :] = kept
        self._discarding("overflow")
        return True

    def _changed(self) -> None:
        self._version += 1
        self._dirty = True
        self._wake.set()
        self._notify()

    def _notify(self) -> None:
        tick, self._tick = self._tick, asyncio.Event()
        tick.set()

    def _archive_state(self) -> ArchiveOutbox:
        view = self._latest
        return ArchiveOutbox(
            conversation_id=self._conversation_id,
            generation=self._generation,
            next_seq=self._next_seq,
            settled=max(0, min(len(view.messages) - view.unsettled, self._next_seq)),
            cursor=self._cursor,
            rows=tuple(self._outbox),
            frozen=self._frozen,
            gap=self._gap,
        )

    # --- the archive sender's interface ------------------------------------------------------

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def _batch(self) -> ArchiveBatch:
        rows = tuple(self._outbox[: self._frozen])
        head = rows[0]
        return ArchiveBatch(
            conversation_id=self._conversation_id,
            generation=self._generation,
            seq_from=head.seq if head.gap_before is None else head.gap_before[0],
            seq_through=rows[-1].seq,
            rows=rows,
        )

    def _freeze(self) -> None:
        size = 0
        count = 0
        for row in self._outbox[: self._max_batch]:
            size += row_wire_bound(row.text)
            if count and size > MAX_BATCH_WIRE_BYTES:
                break
            count += 1
        if count:
            self._frozen = count
            self._changed()
            self._frozen_version = self._version

    async def next_batch(self) -> ArchiveBatch:
        """The batch to send: frozen and written to the tail before it is returned.

        While unacknowledged, the same batch is returned every time, unchanged.
        """

        while True:
            if self._owner is not None:
                if not self._frozen:
                    self._freeze()
                if self._frozen and self._written_version >= self._frozen_version:
                    return self._batch()
            await self._tick.wait()

    def acknowledge(
        self, conversation_id: str, generation: int, seq_from: int, seq_through: int
    ) -> bool:
        """Remove the frozen batch if this names exactly its range; the cursor advances."""

        if not self._frozen:
            return False
        batch = self._batch()
        if (conversation_id, generation, seq_from, seq_through) != (
            batch.conversation_id,
            batch.generation,
            batch.seq_from,
            batch.seq_through,
        ) or any(type(value) is not int for value in (generation, seq_from, seq_through)):
            return False
        del self._outbox[: self._frozen]
        self._frozen = 0
        self._cursor = seq_through
        self._discard_episode = False
        self._changed()
        return True

    # --- lifecycle ---------------------------------------------------------------------------

    async def open(self, store: ConversationContextStore) -> None:
        """Lock the tail, clear crashed temporaries, restore it, then start mirroring.

        A second live host fails closed. A tail that cannot be parsed or restored
        starts a fresh conversation with one content-free marker, and the next
        write replaces it.
        """
        if type(store) is not ConversationContextStore:
            raise TypeError("voice tail store must be an exact ConversationContextStore")
        if self._opened:
            raise RuntimeError("a voice tail opens at most once")
        if row_wire_bound("x" * store.max_item_chars) > MAX_BATCH_WIRE_BYTES:
            raise ValueError("the store's per-item bound cannot fit one archive batch")
        self._opened = True
        owner = lock_run_record(self._path)
        if owner is None:
            _marker(_LOCK_MARKER_PREFIX, {"cause": "held", "version": 1})
            raise RuntimeError("another host holds the voice tail")
        try:
            remove_orphaned_temporaries(self._path)
            raw = await asyncio.to_thread(
                read_run_record,
                self._path,
                max_voice_tail_bytes(store.max_messages, store.max_item_chars, self._max_outbox),
            )
            self._archiving = True
            if raw is None:
                self._fresh()
            else:
                self._restore(store, raw)
        except BaseException:
            self._archiving = False
            unlock_run_record(owner)
            raise
        self._owner = owner
        self._task = asyncio.create_task(self._mirror(), name="voice-tail-writer")
        self._notify()

    def _fresh(self) -> None:
        self._conversation_id = self._conversation_ids()
        if type(self._conversation_id) is not str or not _CONVERSATION_ID.fullmatch(
            self._conversation_id
        ):
            raise ValueError("a conversation id must be 1-64 characters of [A-Za-z0-9_-]")
        self._generation = self._seq_base = self._next_seq = 0
        self._cursor = self._gap = None
        self._outbox = []
        self._frozen = 0

    def _load(self, archive: ArchiveOutbox) -> None:
        self._conversation_id = archive.conversation_id
        self._generation = archive.generation
        self._next_seq = archive.next_seq
        self._seq_base = archive.next_seq - archive.settled
        self._cursor = archive.cursor
        self._outbox = list(archive.rows)
        self._frozen = archive.frozen
        self._gap = archive.gap
        self._frozen_version = self._written_version

    def _restore(self, store: ConversationContextStore, raw: bytes) -> None:
        # Counts or a refusal category only; never row text.
        evidence: dict[str, str | int] | None = None
        try:
            tail = parse_voice_tail(
                raw,
                max_messages=store.max_messages,
                max_item_chars=store.max_item_chars,
                max_outbox_rows=self._max_outbox,
            )
            try:
                if tail is None:
                    raise ValueError("voice tail is malformed")
                if tail.archive is None:
                    self._fresh()
                else:
                    self._load(tail.archive)
                store.restore(tail.conversation)
            except ValueError:
                # A refused restore mutated nothing; only a loaded archive must be dropped.
                if tail is None or tail.archive is not None:
                    self._fresh()
                evidence = {"refusal": "malformed", "version": 1}
            else:
                evidence = {
                    "outbox": len(self._outbox),
                    "restored": len(tail.conversation.messages),
                    "tail_version": _LEGACY_VERSION if tail.archive is None else _VERSION,
                    "version": 1,
                }
        finally:
            if evidence is not None:
                _marker(_MARKER_PREFIX, evidence)

    async def close(self) -> None:
        """Let the writer finish the final snapshot, then release the tail.

        On timeout the tail stays owned and RuntimeError is raised, so a later
        close waits again instead of silently giving up the final write.
        """
        self._close_requested.set()
        self._wake.set()
        owner = self._owner
        if owner is None:
            return
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._close_timeout)
            except TimeoutError:
                raise RuntimeError(
                    "voice tail writer did not finish before the close timeout"
                ) from None
        self._task = None
        self._owner = None
        self._archiving = False
        unlock_run_record(owner)

    async def _mirror(self) -> None:
        backoff = self._initial_backoff
        while True:
            if not self._dirty:
                if self._close_requested.is_set():
                    return
                await self._wake.wait()
                self._wake.clear()
                continue
            version = self._version
            data = voice_tail_bytes(self._latest, self._archive_state())
            self._dirty = False
            try:
                await self._write(data)
            except Exception as error:
                # The latest snapshot, whichever it is by now, is still unwritten.
                self._dirty = True
                _LOGGER.warning(
                    "voice tail write failed (%s); retrying in %.2f s",
                    type(error).__name__,
                    backoff,
                )
                await self._wait_backoff(backoff)
                backoff = min(backoff * 2, self._max_backoff)
            else:
                backoff = self._initial_backoff
                self._written_version = version
                self._notify()

    async def _write(self, data: bytes) -> None:
        await asyncio.to_thread(write_run_record, self._path, data)

    async def _wait_backoff(self, delay: float) -> None:
        await asyncio.sleep(delay)


__all__ = [
    "DEFAULT_MAX_BATCH_ROWS",
    "DEFAULT_MAX_OUTBOX_ROWS",
    "MAX_BATCH_WIRE_BYTES",
    "ArchiveBatch",
    "ArchiveOutbox",
    "OutboxRow",
    "VoiceTail",
    "VoiceTailWriter",
    "max_voice_tail_bytes",
    "parse_voice_tail",
    "row_wire_bound",
    "voice_tail_bytes",
]
