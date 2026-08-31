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
