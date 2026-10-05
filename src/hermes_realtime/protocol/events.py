"""Versioned events shared by the realtime worker and Hermes plugin."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8192),
]
# Identifiers are never normalized. Trimming would let " evt_1 " and "evt_1"
# alias to one identity across the bridge, so a value that would need
# normalization is rejected instead; the pattern already forbids whitespace.
IdentifierString = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and coerced primitives."""

    # strict=True keeps Pydantic's lax mode from converting wire values such as
    # {"sequence": "1"} into the declared type. The bridge is the trust
    # boundary, so a malformed event must fail closed rather than be repaired.
    model_config = ConfigDict(extra="forbid", strict=True)


class Durability(StrEnum):
    """Lifetime requested for background work."""

    EPHEMERAL = "ephemeral"
    DURABLE = "durable"


class CancelScope(StrEnum):
    """Independent cancellation boundaries."""

    SPEECH = "speech"
    TURN = "turn"
    TASK = "task"
    SESSION = "session"


class WorkTerminalStatus(StrEnum):
    """Terminal outcomes reported by an authoritative Hermes run."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


# The wire representation of these enums is their string value. Accept exact
# strings and exact enum instances, but reject values such as bytes that
# Pydantic's lax enum parser would otherwise coerce.
def _require_exact_wire_enum(value: object, enum_type: type[StrEnum]) -> object:
    if type(value) is str or type(value) is enum_type:
        return value
    raise ValueError(f"{enum_type.__name__} must be an exact string or enum value")


def _parse_wire_durability(value: object) -> object:
    return _require_exact_wire_enum(value, Durability)


def _parse_wire_cancel_scope(value: object) -> object:
    return _require_exact_wire_enum(value, CancelScope)


def _parse_wire_terminal_status(value: object) -> object:
    return _require_exact_wire_enum(value, WorkTerminalStatus)


# Pydantic strict Python mode rejects an ISO datetime string even though the
# same exact string is valid in strict JSON mode. parse_event() advertises both
# JSON bytes/text and JSON-style dictionaries, so accept only the two canonical
# representations rather than reintroducing broad coercion.
_AWARE_DATETIME_ADAPTER = TypeAdapter(AwareDatetime)


def _parse_wire_datetime(value: object) -> object:
    if type(value) is datetime:
        return value
    if type(value) is str:
        return _AWARE_DATETIME_ADAPTER.validate_python(value)
    raise ValueError("timestamp must be an exact ISO string or datetime value")


WireDurability = Annotated[
    Durability,
    Field(strict=False),
    BeforeValidator(_parse_wire_durability),
]
WireCancelScope = Annotated[
    CancelScope,
    Field(strict=False),
    BeforeValidator(_parse_wire_cancel_scope),
]
WireTerminalStatus = Annotated[
    WorkTerminalStatus,
    Field(strict=False),
    BeforeValidator(_parse_wire_terminal_status),
]
WireAwareDatetime = Annotated[AwareDatetime, BeforeValidator(_parse_wire_datetime)]


class BaseEvent(StrictModel):
    """Fields carried by every protocol event."""

    protocol_version: Literal["0.1"] = "0.1"
    event_id: IdentifierString
    session_id: IdentifierString
    sequence: int = Field(ge=0)
    timestamp: WireAwareDatetime


class WorkDispatchRequestedPayload(StrictModel):
    """Description of background work awaiting real Hermes dispatch."""

    objective: NonEmptyString
    durability: WireDurability = Durability.EPHEMERAL
    progress_reporting: Literal["none", "material_only", "all"] = "material_only"
    completion_reporting: Literal["proactive", "on_request"] = "proactive"


class WorkDispatchRequestedEvent(BaseEvent):
    """Request to start work; this event alone does not prove work started."""

    type: Literal["work.dispatch.requested"]
    task_id: IdentifierString
    utterance_id: IdentifierString
    payload: WorkDispatchRequestedPayload


class WorkDispatchAcknowledgedPayload(StrictModel):
    """Authoritative result of attempting to dispatch work to Hermes."""

    accepted: bool
    run_id: IdentifierString | None = None
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_outcome_evidence(self) -> WorkDispatchAcknowledgedPayload:
        if self.accepted and self.run_id is None:
            raise ValueError("run_id is required when dispatch is accepted")
        if self.accepted and self.reason is not None:
            raise ValueError("reason must not be present when dispatch is accepted")
        if not self.accepted and self.reason is None:
            raise ValueError("reason is required when dispatch is rejected")
        if not self.accepted and self.run_id is not None:
            raise ValueError("run_id must not be present when dispatch is rejected")
        return self


class WorkDispatchAcknowledgedEvent(BaseEvent):
    """Hermes dispatch result used to ground foreground task-state claims."""

    type: Literal["work.dispatch.acknowledged"]
    task_id: IdentifierString
    payload: WorkDispatchAcknowledgedPayload


class WorkCompletedPayload(StrictModel):
    """Terminal evidence emitted after an acknowledged Hermes run ends."""

    status: WireTerminalStatus
    summary: NonEmptyString | None = None
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_terminal_evidence(self) -> WorkCompletedPayload:
        if self.status is WorkTerminalStatus.COMPLETED:
            if self.summary is None:
                raise ValueError("summary is required when work completes")
            if self.reason is not None:
                raise ValueError("reason must not be present when work completes")
        elif self.reason is None:
            raise ValueError("reason is required when work does not complete")
        return self


class WorkCompletedEvent(BaseEvent):
    """Ordered terminal result for one previously acknowledged Hermes run."""

    type: Literal["work.completed"]
    task_id: IdentifierString
    run_id: IdentifierString
    payload: WorkCompletedPayload


class ControlCancelPayload(StrictModel):
    """Requested cancellation boundary."""

    scope: WireCancelScope
    reason: NonEmptyString | None = None


class ControlCancelEvent(BaseEvent):
    """Cancellation request that cannot ambiguously target unrelated work."""

    type: Literal["control.cancel"]
    task_id: IdentifierString | None = None
    payload: ControlCancelPayload

    @model_validator(mode="after")
    def validate_target(self) -> ControlCancelEvent:
        if self.payload.scope is CancelScope.TASK and self.task_id is None:
            raise ValueError("task_id is required for task cancellation")
        if self.payload.scope is not CancelScope.TASK and self.task_id is not None:
            raise ValueError("task_id is only valid for task cancellation")
        return self


class ControlCancelAcknowledgedPayload(StrictModel):
    """Evidence that exact Hermes runs did or did not receive interruption."""

    accepted: bool
    signaled_run_ids: list[IdentifierString] = Field(default_factory=list, max_length=256)
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_signal_evidence(self) -> ControlCancelAcknowledgedPayload:
        if len(set(self.signaled_run_ids)) != len(self.signaled_run_ids):
            raise ValueError("signaled_run_ids must not contain duplicates")
        if self.accepted:
            if not self.signaled_run_ids:
                raise ValueError("signaled_run_ids are required when cancellation is accepted")
            if self.reason is not None:
                raise ValueError("reason must not be present when cancellation is accepted")
        else:
            if self.signaled_run_ids:
                raise ValueError("signaled_run_ids must be empty when cancellation is rejected")
            if self.reason is None:
                raise ValueError("reason is required when cancellation is rejected")
        return self


class ControlCancelAcknowledgedEvent(BaseEvent):
    """Ordered result of attempting to signal one cancellation boundary."""

    type: Literal["control.cancel.acknowledged"]
    request_event_id: IdentifierString
    scope: WireCancelScope
    task_id: IdentifierString | None = None
    payload: ControlCancelAcknowledgedPayload

    @model_validator(mode="after")
    def validate_target(self) -> ControlCancelAcknowledgedEvent:
        if self.scope is CancelScope.TASK and self.task_id is None:
            raise ValueError("task_id is required for task cancellation acknowledgment")
        if self.scope is not CancelScope.TASK and self.task_id is not None:
            raise ValueError("task_id is only valid for task cancellation acknowledgment")
        return self


ProtocolEvent: TypeAlias = Annotated[
    WorkDispatchRequestedEvent
    | WorkDispatchAcknowledgedEvent
    | WorkCompletedEvent
    | ControlCancelEvent
    | ControlCancelAcknowledgedEvent,
    Field(discriminator="type"),
]
_EVENT_ADAPTER: TypeAdapter[ProtocolEvent] = TypeAdapter(ProtocolEvent)


def parse_event(data: str | bytes | dict[str, Any]) -> ProtocolEvent:
    """Parse an event and fail closed on unknown versions, types, or fields."""

    if isinstance(data, (str, bytes)):
        return _EVENT_ADAPTER.validate_json(data)
    return _EVENT_ADAPTER.validate_python(data)


# --- bridge protocol 0.3: capability negotiation, voice archive and review ------------------
#
# The bridge hello names protocol "0.2" and the capabilities each side offers; realtime sends
# a voice event only on a connection whose hello advertised ``voice_archive``. Voice events
# carry no Hermes session, event id or sequence: one batch is in flight per conversation, so
# its exact range correlates the reply. Semantic batch rules (the partition of the range by
# rows and gaps, which role may carry a gap or a flag) are the companion's, which answers
# them with a category refusal rather than a dropped connection.

BRIDGE_PROTOCOL_VERSION = "0.3"
VOICE_ARCHIVE_CAPABILITY = "voice_archive"
VOICE_REVIEW_CAPABILITY = "voice_review"
# A welcome that negotiates it carries ``runtime``: what the serving process loaded.
RUNTIME_ATTESTATION_CAPABILITY = "runtime_attestation"
BRIDGE_CAPABILITIES = frozenset(
    {VOICE_ARCHIVE_CAPABILITY, VOICE_REVIEW_CAPABILITY, RUNTIME_ATTESTATION_CAPABILITY}
)
VOICE_MAX_BATCH_ROWS = 256
VOICE_MAX_TEXT_CHARS = 65_536
_VOICE_MAX_IDENTITY = 2**53 - 1
# Every category the companion may refuse with; a test binds this to the companion's set.
VOICE_REFUSAL_CATEGORIES = frozenset(
    {
        "invalid",
        "partition",
        "identity",
        "conflict",
        "capacity",
        "mismatch",
        "missing",
        "over_cap",
        "rotated",
        "lineage",
        "count",
        "recovery",
        "drift",
        "incompatible",
        "durability",
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
        "busy",
        "unknown",
        "failed",
        "disabled",
        "configuration",
        "window",
    }
)

# The categories split into two closed sets whose union is exactly the set above. A
# transient refusal says the companion cannot take the batch now (not ready, a lease, the
# bound on live or stored conversations, its configuration, an unsettled commit) and never
# that the archive or the batch is wrong: realtime keeps the frozen batch and retries it
# unchanged. Every other category says the batch, the archive or its fences are wrong, or
# the generation is retired: archiving is fenced. On the wire the category is a bounded
# string, so a category a newer companion adds still parses; one in neither set fences.
VOICE_TRANSIENT_REFUSALS = frozenset(
    {
        "not_ready",
        "lease_held",
        "lease_lost",
        "conversations",
        "pending",
        "stale",
        "fenced",
        "incompatible",
        "durability",
        "unbound",
        "bound",
    }
)
VOICE_INTEGRITY_REFUSALS = frozenset(
    {
        "invalid",
        "partition",
        "identity",
        "conflict",
        "capacity",
        "mismatch",
        "missing",
        "over_cap",
        "rotated",
        "lineage",
        "count",
        "recovery",
        "drift",
        "quarantined",
        "tombstoned",
        # Review-only categories must never make an archive refusal transient.
        "busy",
        "unknown",
        "failed",
        "disabled",
        "configuration",
        "window",
    }
)

VoiceConversationId = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]
VoiceIdentity = Annotated[int, Field(ge=0, le=_VOICE_MAX_IDENTITY)]
VoiceRefusalCategory = Annotated[
    str, StringConstraints(min_length=1, max_length=32, pattern=r"^[a-z_]+$")
]


def _parse_wire_timestamp(value: object) -> object:
    # Hermes stores a float; an integer would be a different stored value.
    if type(value) is not float:
        raise ValueError("ts must be an exact float")
    return value


def _parse_wire_gap(value: object) -> object:
    # A JSON-style dict carries the interval as a list; strict mode wants the tuple.
    return tuple(value) if type(value) is list else value


WireVoiceGap = Annotated[
    tuple[VoiceIdentity, VoiceIdentity] | None, BeforeValidator(_parse_wire_gap)
]
WireVoiceTimestamp = Annotated[
    float, BeforeValidator(_parse_wire_timestamp), Field(ge=0, allow_inf_nan=False)
]


class VoiceArchiveRow(StrictModel):
    """One closed voice row: its seq, role, exact heard text, flag, close time and gap."""

    seq: VoiceIdentity
    role: Literal["user", "assistant"]
    text: Annotated[str, StringConstraints(min_length=1, max_length=VOICE_MAX_TEXT_CHARS)]
    interrupted: bool
    ts: WireVoiceTimestamp
    gap_before: WireVoiceGap


class _VoiceRange(StrictModel):
    # Explicit on every voice event: the work events stay at "0.1".
    protocol_version: Literal["0.3"]
    conversation_id: VoiceConversationId
    generation: VoiceIdentity
    seq_from: VoiceIdentity
    seq_through: VoiceIdentity

    @model_validator(mode="after")
    def require_ordered_range(self) -> _VoiceRange:
        if self.seq_from > self.seq_through:
            raise ValueError("seq_from must not follow seq_through")
        return self


class VoiceArchiveEvent(_VoiceRange):
    """Archive these rows, which with their gaps cover exactly ``[seq_from, seq_through]``."""

    type: Literal["voice_archive"]
    rows: list[VoiceArchiveRow] = Field(min_length=1, max_length=VOICE_MAX_BATCH_ROWS)


class VoiceArchiveAckEvent(_VoiceRange):
    """The archive provably holds exactly this range: sent only after the commit."""

    type: Literal["voice_archive_ack"]


class VoiceArchiveRefusedEvent(_VoiceRange):
    """A definitive refusal of exactly this batch, by category; never a negative read."""

    type: Literal["voice_archive_refused"]
    category: VoiceRefusalCategory


def _require_true(value: object) -> object:
    if type(value) is not bool or value is not True:
        raise ValueError("review flags must be exact true booleans")
    return value


class VoiceReviewEvent(_VoiceRange):
    """Request native review of one acknowledged archive range."""

    type: Literal["voice_review"]
    memory: Annotated[Literal[True], BeforeValidator(_require_true)]
    skills: Annotated[Literal[True], BeforeValidator(_require_true)]
    closing: bool


class VoiceReviewAckEvent(_VoiceRange):
    """An owned Hermes review thread was admitted for this exact request."""

    type: Literal["voice_review_ack"]
    closing: bool
    review_id: VoiceConversationId
    status: Literal["accepted"]


class VoiceReviewRefusedEvent(_VoiceRange):
    """The review was not admitted; coverage stays pending."""

    type: Literal["voice_review_refused"]
    closing: bool
    category: VoiceRefusalCategory


_AttestedVersion = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[0-9A-Za-z.+_-]+$")
]


class RuntimeAttestation(StrictModel):
    """What the process serving the companion loaded, captured once when it loaded the plugin.

    ``hermes_commit`` is the checkout's detached commit, or ``unknown``. ``realtime_install``
    says where the imported ``hermes_realtime`` came from: the Hermes install's environment
    (``wheel``), an editable install in it (``editable``), or anywhere else (``elsewhere``).
    """

    pid: Annotated[int, Field(ge=1, le=_VOICE_MAX_IDENTITY)]
    hermes_version: _AttestedVersion
    hermes_commit: Annotated[str, StringConstraints(pattern=r"^(?:[0-9a-f]{40}|unknown)$")]
    realtime_version: _AttestedVersion
    realtime_install: Literal["wheel", "editable", "elsewhere"]


VoiceEvent: TypeAlias = Annotated[
    VoiceArchiveEvent
    | VoiceArchiveAckEvent
    | VoiceArchiveRefusedEvent
    | VoiceReviewEvent
    | VoiceReviewAckEvent
    | VoiceReviewRefusedEvent,
    Field(discriminator="type"),
]
_VOICE_ADAPTER: TypeAdapter[VoiceEvent] = TypeAdapter(VoiceEvent)
VOICE_EVENT_TYPES = frozenset(
    {
        "voice_archive",
        "voice_archive_ack",
        "voice_archive_refused",
        "voice_review",
        "voice_review_ack",
        "voice_review_refused",
    }
)


def parse_voice_event(data: str | bytes | dict[str, Any]) -> VoiceEvent:
    """Parse a voice event and fail closed on unknown versions, types, or fields."""

    if isinstance(data, (str, bytes)):
        return _VOICE_ADAPTER.validate_json(data)
    return _VOICE_ADAPTER.validate_python(data)
