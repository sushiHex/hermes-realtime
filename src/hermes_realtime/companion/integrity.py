"""Durable integrity evidence for a voice archive, computed without I/O.

A voice archive is one Hermes session that only the companion writes. Its evidence is a hash
chain over the canonical serialization of every row Hermes holds for that session, in
ascending ``messages.id`` order, and inactive and compacted rows included:

    C_0 = SHA256(header)        C_i = SHA256(C_{i-1} || SHA256(row_i))

The fingerprint is ``(count, C_n)``. Numeric row ids stay out of the hash, so the fingerprint
an archive will have after an append is known before the append happens. The companion never
compacts or rewrites, so a legitimate archive only ever grows at its end; any other change to
any row, or to the session header, changes the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

VOICE_SOURCE = "hermes-realtime-voice"
# Provisional per-conversation bound: verification is O(rows), so the bound keeps it cheap.
MAX_ARCHIVE_ROWS = 4096
MAX_BATCH_ROWS = 256
MAX_TEXT_CHARS = 65_536
MAX_IDENTITY = 2**53 - 1
PROJECTION_VERSION = 1
ROLES = ("user", "assistant")

# Every column of Hermes's ``messages`` table except ``id`` (kept out so the fingerprint can
# be precomputed) and ``session_id`` (constant within one archive).
MESSAGE_COLUMNS = (
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "effect_disposition",
    "timestamp",
    "token_count",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "platform_message_id",
    "observed",
    "_compressed_summary",
    "active",
    "compacted",
    "api_content",
    "display_kind",
    "display_metadata",
)
HEADER_COLUMNS = ("parent_session_id", "source", "ended_at", "end_reason")

REFUSAL_CATEGORIES = frozenset(
    {
        # The batch itself.
        "invalid",
        "identity",
        "conflict",
        "capacity",
        # The archive no longer matches its durable evidence: quarantined.
        "mismatch",
        "missing",
        "over_cap",
        "rotated",
        "recovery",
        "session_exists",
        # Hermes did something other than what the compatibility surface qualified.
        "drift",
        # Ownership and lifecycle.
        "lease_lost",
        "lease_held",
        "not_ready",
        "fenced",
        "quarantined",
        "tombstoned",
        "pending",
        "stale",
        "unbound",
        "bound",
        "conversations",
    }
)

_CONVERSATION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CHAIN = re.compile(r"[0-9a-f]{64}")


class ArchiveRefusal(Exception):
    """A definitive, content-free refusal. ``category`` is one of ``REFUSAL_CATEGORIES``."""

    def __init__(self, category: str) -> None:
        if type(category) is not str or category not in REFUSAL_CATEGORIES:
            raise ValueError("archive refusal category is not recognized")
        super().__init__(category)
        self.category = category


def validate_conversation_id(conversation_id: str) -> str:
    if type(conversation_id) is not str:
        raise TypeError("conversation id must be an exact str")
    if _CONVERSATION_ID.fullmatch(conversation_id) is None:
        raise ValueError("conversation id must be 1-64 characters of [A-Za-z0-9_-]")
    return conversation_id


def _bounded_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not 0 <= value <= MAX_IDENTITY:
        raise ValueError(f"{name} is out of range")
    return value


@dataclass(frozen=True, slots=True, order=True)
class Identity:
    """One voice row's immutable identity: its generation, then its sequence within it."""

    generation: int
    seq: int

    def __post_init__(self) -> None:
        _bounded_int(self.generation, "generation")
        _bounded_int(self.seq, "seq")


@dataclass(frozen=True, slots=True)
class VoiceRow:
    """One closed conversation row, exactly as it was delivered."""

    identity: Identity
    role: str
    text: str
    interrupted: bool
    timestamp: float

    def __post_init__(self) -> None:
        if type(self.identity) is not Identity:
            raise TypeError("voice row identity must be an exact Identity")
        if type(self.role) is not str or type(self.text) is not str:
            raise TypeError("voice row role and text must be exact str")
        if type(self.interrupted) is not bool:
            raise TypeError("voice row interruption must be an exact bool")
        if type(self.timestamp) is not float:
            raise TypeError("voice row timestamp must be an exact float")
        if self.role not in ROLES:
            raise ValueError("voice row role is not supported")
        if self.interrupted and self.role != "assistant":
            raise ValueError("only assistant rows may be interrupted")
        if not self.text.strip() or len(self.text) > MAX_TEXT_CHARS:
            raise ValueError("voice row text must be non-blank and bounded")
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("voice row text must be encodable text") from None
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("voice row timestamp must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """``(count, C_n)``: how many rows the chain covers, and its last link in lowercase hex."""

    count: int
    chain: str

    def __post_init__(self) -> None:
        if type(self.count) is not int or type(self.chain) is not str:
            raise TypeError("fingerprint fields must be an exact int and str")
        # A sanity bound, not the archive cap: the pending state of a batch that would pass
        # the cap is still computed, so the in-transaction capacity check can refuse it.
        if not 0 <= self.count <= MAX_IDENTITY:
            raise ValueError("fingerprint count is out of range")
        if _CHAIN.fullmatch(self.chain) is None:
            raise ValueError("fingerprint chain must be 64 lowercase hex digits")


@dataclass(frozen=True, slots=True)
class Header:
    """The session facts an archive's genesis link binds."""

    parent_session_id: object
    source: object
    ended_at_is_null: bool
    end_reason: object

    @classmethod
    def from_values(cls, values: Mapping[str, object]) -> Header:
        if set(values) != set(HEADER_COLUMNS):
            raise ValueError("session header must carry exactly the header columns")
        return cls(
            parent_session_id=values["parent_session_id"],
            source=values["source"],
            ended_at_is_null=values["ended_at"] is None,
            end_reason=values["end_reason"],
        )


EXPECTED_HEADER = Header(
    parent_session_id=None, source=VOICE_SOURCE, ended_at_is_null=True, end_reason=None
)


def _typed(value: object) -> list[object]:
    """One stored SQLite value, tagged with its storage class so no two classes collide."""

    kind = type(value)
    if value is None:
        return ["n"]
    if kind is int:
        return ["i", value]
    if kind is float:
        return ["f", cast(float, value).hex()]
    if kind is str:
        return ["s", value]
    if kind is bytes:
        return ["b", cast(bytes, value).hex()]
    raise TypeError("a stored column value must be None, int, float, str or bytes")


def _canonical(document: dict[str, object]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def header_bytes(header: Header) -> bytes:
    if type(header) is not Header or type(header.ended_at_is_null) is not bool:
        raise TypeError("header must be an exact Header")
    return _canonical(
        {
            "end_reason": _typed(header.end_reason),
            "ended_at_is_null": header.ended_at_is_null,
            "kind": "voice-archive-header",
            "parent_session_id": _typed(header.parent_session_id),
            "source": _typed(header.source),
            "version": PROJECTION_VERSION,
        }
    )


def canonical_row(values: Mapping[str, object]) -> bytes:
    """The canonical serialization of one stored row: every message column, typed."""

    if set(values) != set(MESSAGE_COLUMNS):
        raise ValueError("a projected row must carry exactly the message columns")
    return _canonical({column: _typed(values[column]) for column in MESSAGE_COLUMNS})


def genesis(header: Header) -> Fingerprint:
    return Fingerprint(0, hashlib.sha256(header_bytes(header)).hexdigest())


def extend(fingerprint: Fingerprint, rows: Sequence[bytes]) -> Fingerprint:
    """The fingerprint after appending ``rows``, each already canonically serialized."""

    if type(fingerprint) is not Fingerprint:
        raise TypeError("fingerprint must be an exact Fingerprint")
    link = bytes.fromhex(fingerprint.chain)
    for row in rows:
        if type(row) is not bytes:
            raise TypeError("a canonical row must be exact bytes")
        link = hashlib.sha256(link + hashlib.sha256(row).digest()).digest()
    return Fingerprint(fingerprint.count + len(rows), link.hex())


def platform_message_id(conversation_id: str, identity: Identity) -> str:
    validate_conversation_id(conversation_id)
    if type(identity) is not Identity:
        raise TypeError("identity must be an exact Identity")
    return f"voice:{conversation_id}:{identity.generation}:{identity.seq}"


def voice_metadata(row: VoiceRow) -> dict[str, dict[str, object]]:
    """The ``display_metadata`` object a voice row carries, keys in sorted order."""

    return {
        "voice": {
            "gen": row.identity.generation,
            "interrupted": row.interrupted,
            "seq": row.identity.seq,
        }
    }


def expected_row_values(conversation_id: str, row: VoiceRow) -> dict[str, object]:
    """Every column Hermes stores for ``row``, as the companion's append writes it.

    ``display_metadata`` is the text Hermes's encoder produces (``json.dumps`` with default
    separators); the qualification proves this equals what Hermes actually stores.
    """

    if type(row) is not VoiceRow:
        raise TypeError("row must be an exact VoiceRow")
    values: dict[str, object] = dict.fromkeys(MESSAGE_COLUMNS)
    values |= {
        "role": row.role,
        "content": row.text,
        "timestamp": row.timestamp,
        "platform_message_id": platform_message_id(conversation_id, row.identity),
        "observed": 0,
        "_compressed_summary": 0,
        "active": 1,
        "compacted": 0,
        "display_metadata": json.dumps(voice_metadata(row)),
    }
    return values


def expected_after(
    committed: Fingerprint, conversation_id: str, new_rows: Sequence[VoiceRow]
) -> Fingerprint:
    """The fingerprint the archive will have once ``new_rows`` are appended."""

    return extend(
        committed, [canonical_row(expected_row_values(conversation_id, row)) for row in new_rows]
    )


@dataclass(frozen=True, slots=True)
class ProjectedRow:
    platform_message_id: object
    canonical: bytes


@dataclass(frozen=True, slots=True)
class Projection:
    """Everything the fingerprint covers, read from one session."""

    header: Header
    rows: tuple[ProjectedRow, ...]

    def fingerprint(self) -> Fingerprint:
        return extend(genesis(self.header), [row.canonical for row in self.rows])


def project(
    header_values: Mapping[str, object],
    row_values: Sequence[Mapping[str, object]],
    cap: int,
) -> Projection:
    """Build a projection from a bounded read of at most ``cap + 1`` rows.

    More than ``cap`` rows can never be a legitimate archive, and a truncated chain would
    verify nothing, so an over-cap read is refused rather than cut short.
    """

    if len(row_values) > cap:
        raise ArchiveRefusal("over_cap")
    return Projection(
        header=Header.from_values(header_values),
        rows=tuple(
            ProjectedRow(values["platform_message_id"], canonical_row(values))
            for values in row_values
        ),
    )


@dataclass(frozen=True, slots=True)
class BatchSplit:
    duplicates: tuple[VoiceRow, ...]
    new: tuple[VoiceRow, ...]


def split_batch(
    cursor: Identity | None, generation: int, rows: tuple[VoiceRow, ...]
) -> BatchSplit:
    """Split one generation's batch at the committed cursor, refusing any gap or reorder.

    A batch is contiguous within its generation. Rows at or before the cursor are retries to
    verify; the rest must continue the cursor exactly, and a new generation starts at seq 0.
    """

    _bounded_int(generation, "generation")
    if (
        type(rows) is not tuple
        or not 1 <= len(rows) <= MAX_BATCH_ROWS
        or any(type(row) is not VoiceRow for row in rows)
    ):
        raise ArchiveRefusal("invalid")
    for index, row in enumerate(rows):
        if row.identity != Identity(generation, rows[0].identity.seq + index):
            raise ArchiveRefusal("identity")
    duplicates = tuple(row for row in rows if cursor is not None and row.identity <= cursor)
    new = rows[len(duplicates) :]
    if new:
        first = new[0].identity
        if cursor is None or first.generation != cursor.generation:
            continues = first.seq == 0 and (cursor is None or first > cursor)
        else:
            continues = first.seq == cursor.seq + 1
        if not continues:
            raise ArchiveRefusal("identity")
    return BatchSplit(duplicates=duplicates, new=new)


@dataclass(frozen=True, slots=True)
class ArchivePlan:
    inserts: tuple[VoiceRow, ...]
    already_applied: bool


def plan_archive(
    projection: Projection | None,
    conversation_id: str,
    rows: Sequence[VoiceRow],
    expected_committed: Fingerprint,
    expected_pending: Fingerprint,
    cap: int,
) -> ArchivePlan:
    """Decide, against a fresh projection, which batch rows to insert, or refuse.

    The archive must match the committed fingerprint (nothing applied yet) or the pending
    one (already applied). Every batch row already stored must be byte-identical to it, and
    the rows to insert must all follow every stored row.
    """

    if projection is None:
        raise ArchiveRefusal("missing")
    current = projection.fingerprint()
    if current == expected_committed:
        applied = False
    elif current == expected_pending:
        applied = True
    else:
        raise ArchiveRefusal("mismatch")
    stored = {row.platform_message_id: row.canonical for row in projection.rows}
    inserts: list[VoiceRow] = []
    for row in rows:
        existing = stored.get(platform_message_id(conversation_id, row.identity))
        if existing is None:
            inserts.append(row)
            continue
        if inserts:
            raise ArchiveRefusal("identity")
        if existing != canonical_row(expected_row_values(conversation_id, row)):
            raise ArchiveRefusal("conflict")
    if applied and inserts:
        raise ArchiveRefusal("identity")
    if len(projection.rows) + len(inserts) > cap:
        raise ArchiveRefusal("capacity")
    return ArchivePlan(inserts=tuple(inserts), already_applied=not inserts)
