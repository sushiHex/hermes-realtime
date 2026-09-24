"""Durable, bounded snapshot of the heard voice conversation, so a restart does not lose it.

The context store owns the truth and does no I/O. This module mirrors its durable view,
off the voice path, into one file that the next start restores before its first turn.
The file is plaintext user data; it holds only rows the store would accept.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path

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

_VERSION = 1
_MARKER_PREFIX = "[voice-tail] "
_LOCK_MARKER_PREFIX = "[voice-tail-lock] "
# ASCII-escaped JSON spends at most twelve bytes on one str character (an astral
# character becomes a surrogate-pair escape), plus a fixed per-row and envelope cost.
_MAX_BYTES_PER_CHAR = 12
_ROW_OVERHEAD_BYTES = 64
_ENVELOPE_OVERHEAD_BYTES = 64
_DOCUMENT_FIELDS = frozenset({"messages", "prior_work", "version"})
_ROW_FIELDS = frozenset({"interrupted", "role", "text"})
_MAX_BACKOFF_LIMIT_SECONDS = 60.0
# A to_thread write cannot be cancelled, so close waits this long for the writer to finish.
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 10.0
_CONCRETE_PATH = type(Path())
_LOGGER = logging.getLogger(__name__)


def max_voice_tail_bytes(max_messages: int, max_item_chars: int) -> int:
    """The largest tail a store with these bounds can produce."""
    return (
        max_messages * (_ROW_OVERHEAD_BYTES + _MAX_BYTES_PER_CHAR * max_item_chars)
        + _ENVELOPE_OVERHEAD_BYTES
    )


def voice_tail_bytes(view: DurableConversation) -> bytes:
    document = {
        "messages": [
            {"interrupted": message.interrupted, "role": message.role, "text": message.text}
            for message in view.messages
        ],
        "prior_work": view.prior_work,
        "version": _VERSION,
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_voice_tail(
    raw: bytes,
    *,
    max_messages: int,
    max_item_chars: int,
) -> DurableConversation | None:
    """Return the tail's view, or None when it is malformed, oversized, or another version."""
    if len(raw) > max_voice_tail_bytes(max_messages, max_item_chars):
        return None
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, ValueError):
        return None
    if type(document) is not dict or set(document) != _DOCUMENT_FIELDS:
        return None
    version, rows, prior_work = document["version"], document["messages"], document["prior_work"]
    if type(version) is not int or version != _VERSION:
        return None
    if type(prior_work) is not bool:
        return None
    if type(rows) is not list or len(rows) > max_messages:
        return None
    messages: list[ConversationMessage] = []
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_FIELDS:
            return None
        role, text, interrupted = row["role"], row["text"], row["interrupted"]
        if type(role) is not str or type(text) is not str or type(interrupted) is not bool:
            return None
        try:
            # A lone surrogate decodes from JSON but is not text the store can hold.
            text.encode("utf-8")
            messages.append(ConversationMessage(role=role, text=text, interrupted=interrupted))
        except (ValueError, RuntimeError):
            # Unknown roles, flagged user rows, blank text, and private run tokens.
            return None
    return DurableConversation(messages=tuple(messages), prior_work=prior_work)


def _marker(prefix: str, evidence: dict[str, str | int]) -> None:
    print(prefix + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


class VoiceTailWriter:
    """Own one tail file: restore it once, then mirror every change with the latest winning.

    ``update`` is the store's ``on_change``: synchronous and O(1). One task owns every
    write, the final one included, so a stale snapshot can never land after a newer one.
    A failed write backs off within a bound and retries with whatever is latest by then;
    closing never cuts that short, it only lets the task finish once the tail is clean.
    """

    def __init__(
        self,
        path: Path,
        *,
        initial_backoff_seconds: float = 0.05,
        max_backoff_seconds: float = 2.0,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        if type(path) is not _CONCRETE_PATH:
            raise TypeError("voice tail path must be an exact pathlib Path")
        if not 0 < initial_backoff_seconds <= max_backoff_seconds <= _MAX_BACKOFF_LIMIT_SECONDS:
            raise ValueError("voice tail backoff must be positive and bounded")
        if not (math.isfinite(close_timeout_seconds) and close_timeout_seconds > 0):
            raise ValueError("voice tail close timeout must be finite and positive")
        self._path = path
        self._initial_backoff = float(initial_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._close_timeout = float(close_timeout_seconds)
        self._latest = DurableConversation(messages=(), prior_work=False)
        self._dirty = False
        self._wake = asyncio.Event()
        self._close_requested = asyncio.Event()
        self._opened = False
        self._owner: int | None = None
        self._task: asyncio.Task[None] | None = None

    def update(self, view: DurableConversation) -> None:
        if type(view) is not DurableConversation:
            raise TypeError("voice tail updates must be an exact DurableConversation")
        self._latest = view
        self._dirty = True
        self._wake.set()

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
                max_voice_tail_bytes(store.max_messages, store.max_item_chars),
            )
            if raw is not None:
                self._restore(store, raw)
        except BaseException:
            unlock_run_record(owner)
            raise
        self._owner = owner
        self._task = asyncio.create_task(self._mirror(), name="voice-tail-writer")

    @staticmethod
    def _restore(store: ConversationContextStore, raw: bytes) -> None:
        # Counts or a refusal category only; never row text.
        evidence: dict[str, str | int] | None = None
        try:
            view = parse_voice_tail(
                raw,
                max_messages=store.max_messages,
                max_item_chars=store.max_item_chars,
            )
            try:
                if view is None:
                    raise ValueError("voice tail is malformed")
                store.restore(view)
            except ValueError:
                evidence = {"refusal": "malformed", "version": 1}
            else:
                evidence = {"restored": len(view.messages), "version": 1}
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
            view = self._latest
            self._dirty = False
            try:
                await self._write(view)
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

    async def _write(self, view: DurableConversation) -> None:
        await asyncio.to_thread(write_run_record, self._path, voice_tail_bytes(view))

    async def _wait_backoff(self, delay: float) -> None:
        await asyncio.sleep(delay)


__all__ = [
    "VoiceTailWriter",
    "max_voice_tail_bytes",
    "parse_voice_tail",
    "voice_tail_bytes",
]
