"""The durable record of voice deletes, kept beside the voice tail and never reset with it.

A delete is an intent until the companion verifies that nothing of its generation remains.
The intent outlives anything that resets the tail: a corrupt tail, a tail a rolled-back
build cannot read, or a restore the store refuses. It is written before the tail is
cleared (write-ahead), so a tail still bound to a recorded intent, or to the completed
outcome, is retired when it opens.

The record holds every pending delete, so a delete that stays pending never blocks a later
one, the outcome (the newest binding verified complete), and whether an earlier record
could not be read. That loss is sticky: no later completion clears it, since the lost
intents may still be unfinished in Hermes.

It also names the live binding, written ahead of every tail that carries it. Once the
record holds any delete, a tail is restored only when it is that live binding: a positive
proof that covers every deleted binding, not only the ones the record still lists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hermes_realtime.integration.run_record import strict_object

_VERSION = 1
# Deletes a request may leave to verify; a bound, never a lifetime limit, since each settles.
MAX_PENDING_DELETES = 64
# The record holds one more: a version-4 tail's single intent moves in even when it is full.
_MAX_RECORDED = MAX_PENDING_DELETES + 1
_CONVERSATION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_MAX_IDENTITY = 2**53 - 1
# A binding is at most 64 id characters and 16 digits, with JSON punctuation.
_MAX_BINDING_BYTES = 96
# Every pending binding, the outcome and the live binding.
MAX_VOICE_DELETES_BYTES = (_MAX_RECORDED + 2) * _MAX_BINDING_BYTES + 128

Binding = tuple[str, int]


@dataclass(frozen=True, slots=True)
class VoiceDeletes:
    pending: tuple[Binding, ...] = ()
    outcome: Binding | None = None
    live: Binding | None = None
    lost: bool = False

    @property
    def holds_deletes(self) -> bool:
        """Whether any delete, settled, pending or lost, was ever recorded here."""
        return bool(self.pending) or self.outcome is not None or self.lost


def _binding(value: object) -> Binding | None:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not str
        or _CONVERSATION_ID.fullmatch(value[0]) is None
        or type(value[1]) is not int
        or not 0 <= value[1] <= _MAX_IDENTITY
    ):
        return None
    return value[0], value[1]


def parse_voice_deletes(raw: bytes) -> VoiceDeletes | None:
    """The record, or None when it is oversized, malformed or of an unknown version."""

    if len(raw) > MAX_VOICE_DELETES_BYTES:
        return None
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if (
        type(document) is not dict
        or set(document) != {"live", "lost", "outcome", "pending", "version"}
        or type(document["lost"]) is not bool
        or type(document["version"]) is not int
        or document["version"] != _VERSION
        or type(document["pending"]) is not list
        or len(document["pending"]) > _MAX_RECORDED
    ):
        return None
    pending = tuple(_binding(item) for item in document["pending"])
    if any(item is None for item in pending) or len(set(pending)) != len(pending):
        return None
    raw_outcome = document["outcome"]
    outcome = None if raw_outcome is None else _binding(raw_outcome)
    if raw_outcome is not None and outcome is None:
        return None
    if outcome in pending:
        return None
    raw_live = document["live"]
    live = None if raw_live is None else _binding(raw_live)
    if raw_live is not None and live is None:
        return None
    # The live binding is never one the record deletes.
    if live is not None and (live in pending or live == outcome):
        return None
    return VoiceDeletes(
        tuple(item for item in pending if item is not None), outcome, live,
        document["lost"],
    )


def voice_deletes_bytes(deletes: VoiceDeletes) -> bytes:
    if type(deletes) is not VoiceDeletes:
        raise TypeError("deletes must be an exact VoiceDeletes")
    if len(deletes.pending) > _MAX_RECORDED:
        raise ValueError("too many pending voice deletes")
    outcome = deletes.outcome
    document = {
        "live": None if deletes.live is None else list(deletes.live),
        "lost": deletes.lost,
        "outcome": None if outcome is None else list(outcome),
        "pending": [list(binding) for binding in deletes.pending],
        "version": _VERSION,
    }
    data = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if parse_voice_deletes(data) != deletes:
        raise ValueError("voice deletes record is invalid")
    return data


__all__ = [
    "MAX_PENDING_DELETES",
    "MAX_VOICE_DELETES_BYTES",
    "VoiceDeletes",
    "parse_voice_deletes",
    "voice_deletes_bytes",
]
