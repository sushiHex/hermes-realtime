"""Nonblocking, exactly bounded V1 evidence admission primitives.

This module owns process-local reservations and queue admission only.  It does
not own persistence, browser consent routing, or durable lifecycle recovery.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from queue import Empty, Full
from threading import Condition, Event, Lock, RLock
from time import monotonic
from typing import Literal, NoReturn, Protocol, SupportsIndex, TypeAlias
from uuid import uuid4
from weakref import ReferenceType, WeakKeyDictionary, ref

from .models import (
    AppendDisposition,
    AssistantChunkTransportConfirmedFullPayloadV1,
    AssistantSegmentGeneratedPayloadV1,
    BindingCloseAuthorityV1,
    BindingClosedPayloadV1,
    BindingCloseV1,
    CaptureState,
    CauseDisposition,
    CommandAdmissionAuthorityV1,
    CommandDisposition,
    CommandRoutedPayloadV1,
    ConsentCreateAuthorityV1,
    ConsentDisposition,
    ConsentRevokeAuthorityV1,
    ConversationOperationKind,
    CreateEpochReservationV1,
    CreateEpochV1,
    DrainAndStopV1,
    DrainDisposition,
    EventKind,
    EvidenceDiagnosticsV1,
    EvidenceSnapshotV1,
    ExpireSessionV1,
    ExpiryDisposition,
    ExpiryMode,
    LifecycleDrainAuthorityV1,
    LifecycleSealAuthorityV1,
    OwnerState,
    ProactiveTurnAuthorityV1,
    QueuedEvidenceRecordV1,
    QueueReservationClass,
    ReplayTurnAuthorityV1,
    RevokeDisposition,
    RevokeFinalizeV1,
    RevokeRequestV1,
    RolloverAuthorityV1,
    RolloverDisposition,
    RolloverSessionV1,
    RuntimeSessionPhase,
    SealDisposition,
    SealEpochV1,
    SessionExpiryAuthorityV1,
    SettledTerminalOutcomeV1,
    StoreDisposition,
    TaintCode,
    TerminalCauseCapabilityV1,
    TerminalDisposition,
    TerminalReason,
    TerminalResolutionV1,
    TurnKind,
    TurnOpenedPayloadV1,
    TurnSettledPayloadV1,
    TurnSnapshotPayloadV1,
    UserFinalAcceptedPayloadV1,
    UserTurnAuthorityV1,
    WriterFault,
    evidence_snapshot_to_primitive,
    resolve_terminal_causes,
    validate_authority_composition,
)

_MAX_UNSIGNED_63 = 2**63 - 1
_MAX_CONVERSATION_OPERATIONS = 64
MAX_QUEUE_RECORDS = 64
MAX_QUEUE_CANONICAL_BYTES = 2 * 1024 * 1024
MAX_QUEUE_PHYSICAL_ITEMS = 64
MAX_CANONICAL_RECORD_BYTES = 32 * 1024
CREATE_EPOCH_RECORD_CREDITS = 5
CREATE_EPOCH_CANONICAL_BYTE_CREDITS = 163_840
CREATE_OPENING_RECORD_CREDITS = 2
CREATE_OPENING_CANONICAL_BYTE_CREDITS = 65_536
SESSION_CONTROL_RECORD_CREDITS = 3
SESSION_CONTROL_CANONICAL_BYTE_CREDITS = 98_304
TERMINAL_RECORD_CREDITS = 2
TERMINAL_CANONICAL_BYTE_CREDITS = 65_536
ROLLOVER_RECORD_CREDITS = 4
ROLLOVER_CANONICAL_BYTE_CREDITS = 131_072
MAX_TURN_EVENTS = 4_608
MAX_TURN_CANONICAL_BYTES = 16 * 1024 * 1024
TURN_SESSION_EVENT_RESERVATION = 4_356
TURN_SESSION_CANONICAL_BYTE_RESERVATION = 16 * 1024 * 1024
MAX_SESSION_EVENTS = 9_216
MAX_SESSION_CANONICAL_BYTES = 40 * 1024 * 1024
_ROLLOVER_RUNTIME_OWNER_TOKEN = object()
_TICKET_SIGNAL_TIMES: WeakKeyDictionary[Event, float] = WeakKeyDictionary()
_TICKET_SIGNAL_TIMES_LOCK = Lock()


def _set_ticket_signal(event: Event) -> None:
    """Record the exact production milestone time before publishing its event."""

    if type(event) is not Event:
        raise TypeError("ticket signal must be an exact threading.Event")
    with _TICKET_SIGNAL_TIMES_LOCK:
        _TICKET_SIGNAL_TIMES.setdefault(event, monotonic())
        event.set()


def _ticket_signal_status(event: Event, *, deadline: float | None) -> bool | None:
    """Classify an exact ticket event coherently with production publication."""

    if type(event) is not Event:
        raise TypeError("ticket signal must be an exact threading.Event")
    with _TICKET_SIGNAL_TIMES_LOCK:
        signal_time = _TICKET_SIGNAL_TIMES.get(event)
        if signal_time is not None:
            return deadline is None or signal_time <= deadline
        if event.is_set():
            return deadline is None or monotonic() < deadline
        return None


class ReservationError(RuntimeError):
    """A process-local reservation was stale, forged, or used incorrectly."""


class _RuntimeCapability:
    """Identity-only base for nonserializable process-local capabilities."""

    __slots__ = ()

    def __init_subclass__(cls) -> None:
        if any(
            base is not _RuntimeCapability and issubclass(base, _RuntimeCapability)
            for base in cls.__bases__
        ):
            raise TypeError("runtime capability classes cannot be subclassed")
        super().__init_subclass__()

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("runtime capabilities can be minted only by their owner")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} opaque>"

    def __copy__(self) -> NoReturn:
        raise TypeError("runtime capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("runtime capabilities cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("runtime capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("runtime capabilities cannot be serialized")

    def __getstate__(self) -> NoReturn:
        raise TypeError("runtime capabilities do not disclose state")

    def __setstate__(self, state: object) -> NoReturn:
        del state
        raise TypeError("runtime capabilities cannot be retargeted")


class _AdmissionRolloverPreparationV1(_RuntimeCapability):
    __slots__ = (
        "reservation",
        "admission_ordinal",
        "item",
        "terminal",
        "successor",
        "ticket",
        "predecessor_final_event_sequence",
    )
    reservation: SessionControlReservation
    admission_ordinal: int
    item: EvidenceWriterQueueItemV1 | None
    terminal: bool
    successor: SessionControlReservation | None
    ticket: _RolloverCompletionTicketV1
    predecessor_final_event_sequence: int


class _RolloverCompletionTicketV1:
    __slots__ = ("event", "result")

    def __init__(self) -> None:
        self.event = Event()
        self.result: StoreDisposition | None = None


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class ConversationOperationReservation(_RuntimeCapability):
    """One executor-owned slot tagged for an exact conversation operation."""

    owner_generation: int
    operation_serial: int
    kind: ConversationOperationKind


class _OperationState(Enum):
    RESERVED = auto()
    CONSUMED = auto()


def _exact_positive_int(value: object, field_name: str, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact built-in int")
    if not 1 <= value <= maximum:
        raise ValueError(f"{field_name} is outside its closed range")
    return value


def _exact_operation_kind(value: object) -> ConversationOperationKind:
    if type(value) is not ConversationOperationKind:
        raise TypeError("kind must be an exact ConversationOperationKind")
    return value


class ConversationOperationScheduler:
    """Atomically own the shared response/proactive/replay operation limit."""

    def __init__(self, *, owner_generation: int, max_operations: int = 16) -> None:
        self._owner_generation = _exact_positive_int(
            owner_generation,
            "owner_generation",
            _MAX_UNSIGNED_63,
        )
        self._max_operations = _exact_positive_int(
            max_operations,
            "max_operations",
            _MAX_CONVERSATION_OPERATIONS,
        )
        self._lock = Lock()
        self._next_serial = 1
        self._states: dict[
            ConversationOperationReservation,
            tuple[_OperationState, int, ConversationOperationKind],
        ] = {}
        self._closed = False

    @property
    def owner_generation(self) -> int:
        return self._owner_generation

    @property
    def max_operations(self) -> int:
        return self._max_operations

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._states)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def try_reserve(
        self,
        kind: ConversationOperationKind,
    ) -> ConversationOperationReservation | None:
        exact_kind = _exact_operation_kind(kind)
        with self._lock:
            if self._closed or len(self._states) >= self._max_operations:
                return None
            reservation = self._mint_locked(exact_kind)
            self._states[reservation] = (
                _OperationState.RESERVED,
                reservation.operation_serial,
                exact_kind,
            )
            return reservation

    def validate(
        self,
        reservation: ConversationOperationReservation,
        kind: ConversationOperationKind,
        *,
        require_unconsumed: bool = False,
    ) -> None:
        exact_kind = _exact_operation_kind(kind)
        if type(require_unconsumed) is not bool:
            raise TypeError("require_unconsumed must be an exact built-in bool")
        with self._lock:
            state = self._state_locked(reservation, exact_kind)
            if require_unconsumed and state is not _OperationState.RESERVED:
                raise ReservationError("conversation operation reservation was consumed")

    def consume(
        self,
        reservation: ConversationOperationReservation,
        kind: ConversationOperationKind,
    ) -> None:
        exact_kind = _exact_operation_kind(kind)
        with self._lock:
            state = self._state_locked(reservation, exact_kind)
            if state is not _OperationState.RESERVED:
                raise ReservationError("conversation operation reservation was consumed")
            self._states[reservation] = (
                _OperationState.CONSUMED,
                reservation.operation_serial,
                exact_kind,
            )

    def release(self, reservation: ConversationOperationReservation) -> None:
        with self._lock:
            self._state_locked(reservation, None)
            del self._states[reservation]

    def transfer_idle_proactive(
        self,
        reservation: ConversationOperationReservation,
    ) -> ConversationOperationReservation:
        """Atomically replace one idle proactive bearer without freeing its slot."""

        with self._lock:
            state = self._state_locked(
                reservation,
                ConversationOperationKind.PROACTIVE,
            )
            if state is not _OperationState.RESERVED:
                raise ReservationError("conversation operation reservation was consumed")
            replacement = self._mint_locked(ConversationOperationKind.PROACTIVE)
            del self._states[reservation]
            self._states[replacement] = (
                _OperationState.RESERVED,
                replacement.operation_serial,
                ConversationOperationKind.PROACTIVE,
            )
            return replacement

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._states.clear()

    def _state_locked(
        self,
        reservation: ConversationOperationReservation,
        expected_kind: ConversationOperationKind | None,
    ) -> _OperationState:
        if type(reservation) is not ConversationOperationReservation:
            raise ReservationError("conversation operation reservation has the wrong type")
        if self._closed:
            raise ReservationError("conversation operation owner is closed")
        if (
            type(reservation.owner_generation) is not int
            or reservation.owner_generation != self._owner_generation
        ):
            raise ReservationError("conversation operation reservation has the wrong owner")
        entry = self._states.get(reservation)
        if entry is None:
            raise ReservationError("conversation operation reservation is stale")
        state, operation_serial, kind = entry
        if (
            type(reservation.operation_serial) is not int
            or reservation.operation_serial != operation_serial
            or type(reservation.kind) is not ConversationOperationKind
            or reservation.kind is not kind
        ):
            raise ReservationError("conversation operation reservation was tampered")
        if expected_kind is not None and kind is not expected_kind:
            raise ReservationError("conversation operation reservation has the wrong kind")
        return state

    def _mint_locked(
        self,
        kind: ConversationOperationKind,
    ) -> ConversationOperationReservation:
        serial = self._next_serial
        if serial > _MAX_UNSIGNED_63:
            self._closed = True
            self._states.clear()
            raise ReservationError("conversation operation serial overflowed")
        self._next_serial += 1
        reservation = object.__new__(ConversationOperationReservation)
        object.__setattr__(reservation, "owner_generation", self._owner_generation)
        object.__setattr__(reservation, "operation_serial", serial)
        object.__setattr__(reservation, "kind", kind)
        return reservation


ConversationOperationReservations = ConversationOperationScheduler


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class EvidenceTerminalReservation(_RuntimeCapability):
    """Two retained queue records for one turn's snapshot and settlement."""

    owner_generation: int
    reservation_serial: int


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class SessionControlReservation(_RuntimeCapability):
    """Three retained credits bound to one exact created session lineage."""

    owner_generation: int
    reservation_serial: int
    consent_epoch_id: str
    logical_session_id: str
    cleanup_only: bool


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class EvidenceTurnLease(_RuntimeCapability):
    """One live turn bound to exact operation, lineage, and terminal credits."""

    protocol_version: int
    owner_generation: int
    lease_serial: int
    operation_serial: int
    turn_kind: TurnKind
    evidence_turn_id: str
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    terminal_reservation: EvidenceTerminalReservation
    terminal_cause: TerminalCauseCapabilityV1
    user_authority: UserTurnAuthorityV1 | None


@dataclass(frozen=True, slots=True)
class EvidenceLeaseResultV1:
    """Closed result of attempting to open evidence for a conversation turn."""

    lease: EvidenceTurnLease | None
    disposition: AppendDisposition

    def __post_init__(self) -> None:
        if self.lease is not None and type(self.lease) is not EvidenceTurnLease:
            raise TypeError("lease must be an exact EvidenceTurnLease or None")
        if type(self.disposition) is not AppendDisposition:
            raise TypeError("disposition must be an exact AppendDisposition")
        if (self.lease is None) is (self.disposition is AppendDisposition.ADMITTED):
            raise ValueError("only admitted evidence lease results may carry a lease")


@dataclass(frozen=True, slots=True)
class CreateEpochTicketV1:
    control_sequence: int
    disposition: ConsentDisposition
    durability_event: Event

    def __post_init__(self) -> None:
        _exact_positive_int(self.control_sequence, "control_sequence", 2**53 - 1)
        if type(self.disposition) is not ConsentDisposition:
            raise TypeError("disposition must be an exact ConsentDisposition")
        if type(self.durability_event) is not Event:
            raise TypeError("durability_event must be an exact threading.Event")


@dataclass(frozen=True, slots=True)
class RevokeTicketV1:
    control_sequence: int
    erasure_request_id: str
    disposition: RevokeDisposition
    durability_event: Event
    terminal_event: Event

    def __post_init__(self) -> None:
        _exact_positive_int(self.control_sequence, "control_sequence", 2**53 - 1)
        if type(self.erasure_request_id) is not str:
            raise TypeError("erasure_request_id must be an exact built-in string")
        if type(self.disposition) is not RevokeDisposition:
            raise TypeError("disposition must be an exact RevokeDisposition")
        if type(self.durability_event) is not Event or type(self.terminal_event) is not Event:
            raise TypeError("revoke milestones must be exact threading.Event values")
        if self.durability_event is self.terminal_event:
            raise ValueError("revoke durability and terminal milestones must be distinct")


@dataclass(frozen=True, slots=True)
class DrainTicketV1:
    owner_generation: int
    disposition: DrainDisposition
    terminal_event: Event

    def __post_init__(self) -> None:
        _exact_positive_int(
            self.owner_generation,
            "owner_generation",
            _MAX_UNSIGNED_63,
        )
        if type(self.disposition) is not DrainDisposition:
            raise TypeError("disposition must be an exact DrainDisposition")
        if type(self.terminal_event) is not Event:
            raise TypeError("terminal_event must be an exact threading.Event")


class EvidenceAdmissionV1(Protocol):
    protocol_version: Literal[1]

    @property
    def operation_scheduler(self) -> ConversationOperationScheduler: ...

    def try_admit_command(
        self,
        authority: CommandAdmissionAuthorityV1,
        snapshot: EvidenceSnapshotV1,
    ) -> CommandDisposition: ...

    def try_reserve_user_turn(
        self,
        authority: UserTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1: ...

    def try_admit_user_final(
        self,
        lease: EvidenceTurnLease,
        authority: UserTurnAuthorityV1,
        text: str,
    ) -> AppendDisposition: ...

    def try_admit_generated(
        self,
        lease: EvidenceTurnLease,
        text: str,
    ) -> AppendDisposition: ...

    def try_open_non_user_turn(
        self,
        lease: EvidenceTurnLease,
    ) -> AppendDisposition: ...

    def try_admit_transport_confirmed_full(
        self,
        lease: EvidenceTurnLease,
        *,
        segment_ordinal: int | None,
        synthesis_attempt_id: str,
        transport_attempt_id: str,
        text: str | None,
    ) -> AppendDisposition: ...

    def settle_spawn_failed(
        self,
        lease: EvidenceTurnLease,
    ) -> AppendDisposition: ...

    def settle_completed(
        self,
        lease: EvidenceTurnLease,
        *,
        queued_chunk_count: int,
        started_chunk_count: int,
        transport_confirmed_full_count: int,
        assistant_delivery_context_recorded: bool,
    ) -> AppendDisposition: ...

    def settle_terminal(
        self,
        lease: EvidenceTurnLease,
        *,
        queued_chunk_count: int,
        started_chunk_count: int,
        assistant_delivery_context_recorded: bool,
    ) -> AppendDisposition: ...

    def try_reserve_proactive_turn(
        self,
        authority: ProactiveTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1: ...

    def try_reserve_replay_turn(
        self,
        authority: ReplayTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1: ...

    def discard_unopened_user_turn(
        self,
        lease: EvidenceTurnLease,
        authority: UserTurnAuthorityV1,
    ) -> bool: ...

    def discard_unopened_replay_turn(
        self,
        lease: EvidenceTurnLease,
        authority: ReplayTurnAuthorityV1,
    ) -> bool: ...

    def discard_unopened_proactive_turn(
        self,
        lease: EvidenceTurnLease,
        authority: ProactiveTurnAuthorityV1,
    ) -> bool: ...

    def try_append_turn(
        self,
        lease: EvidenceTurnLease,
        snapshot: EvidenceSnapshotV1,
    ) -> AppendDisposition: ...

    def record_terminal_cause(
        self,
        capability: TerminalCauseCapabilityV1,
        cause: TerminalReason,
    ) -> CauseDisposition: ...

    def settled_terminal_outcome(
        self,
        lease: EvidenceTurnLease,
    ) -> SettledTerminalOutcomeV1 | None: ...

    def diagnostics(self) -> EvidenceDiagnosticsV1: ...


class EvidenceLifecycleControlV1(Protocol):
    protocol_version: Literal[1]

    def begin_consent(self, authority: ConsentCreateAuthorityV1) -> CreateEpochTicketV1: ...

    def try_close_binding(
        self,
        authority: BindingCloseAuthorityV1,
        snapshot: EvidenceSnapshotV1,
    ) -> AppendDisposition: ...

    def invalidate_binding(
        self,
        authority: BindingCloseAuthorityV1,
    ) -> AppendDisposition: ...

    def try_rollover(
        self,
        authority: RolloverAuthorityV1,
        command: RolloverSessionV1,
    ) -> RolloverDisposition: ...

    def begin_expiry(self, authority: SessionExpiryAuthorityV1) -> ExpiryDisposition: ...

    def request_seal(
        self,
        authority: LifecycleSealAuthorityV1,
        command: SealEpochV1,
    ) -> SealDisposition: ...

    def begin_revoke(self, authority: ConsentRevokeAuthorityV1) -> RevokeTicketV1: ...

    def request_drain(self, authority: LifecycleDrainAuthorityV1) -> DrainTicketV1: ...


class WriterQueueLane(StrEnum):
    ORDERED = "ordered"
    REVOKE = "revoke"
    DRAIN = "drain"


WriterQueuePayloadV1: TypeAlias = (
    QueuedEvidenceRecordV1
    | CreateEpochV1
    | BindingCloseV1
    | RolloverSessionV1
    | ExpireSessionV1
    | RevokeRequestV1
    | RevokeFinalizeV1
    | SealEpochV1
    | DrainAndStopV1
)

_WRITER_PAYLOAD_TYPES = (
    QueuedEvidenceRecordV1,
    CreateEpochV1,
    BindingCloseV1,
    RolloverSessionV1,
    ExpireSessionV1,
    RevokeRequestV1,
    RevokeFinalizeV1,
    SealEpochV1,
    DrainAndStopV1,
)


@dataclass(frozen=True, slots=True, eq=False)
class EvidenceWriterQueueItemV1:
    """Process-local writer dispatch plus nonpersistent scheduler metadata."""

    protocol_version: int
    lane: WriterQueueLane
    payload: WriterQueuePayloadV1
    admission_ordinal: int

    def __post_init__(self) -> None:
        _exact_positive_int(self.protocol_version, "protocol_version", 1)
        if type(self.lane) is not WriterQueueLane:
            raise TypeError("lane must be an exact WriterQueueLane")
        if type(self.payload) not in _WRITER_PAYLOAD_TYPES:
            raise TypeError("payload must be an exact V1 writer DTO")
        _exact_positive_int(
            self.admission_ordinal,
            "admission_ordinal",
            _MAX_UNSIGNED_63,
        )
        if self.lane is WriterQueueLane.REVOKE and type(self.payload) not in (
            RevokeRequestV1,
            RevokeFinalizeV1,
        ):
            raise ValueError("revoke lane accepts only revoke DTOs")
        if self.lane is WriterQueueLane.DRAIN and type(self.payload) is not DrainAndStopV1:
            raise ValueError("drain lane accepts only DrainAndStopV1")
        if self.lane is WriterQueueLane.ORDERED and type(self.payload) in (
            RevokeRequestV1,
            RevokeFinalizeV1,
            DrainAndStopV1,
        ):
            raise ValueError("ordered lane cannot accept independent-lane DTOs")


class EvidenceWriterSinkV1(Protocol):
    """The bounded nonawaiting Task 3 writer-ingress seam."""

    protocol_version: Literal[1]

    def put_nowait(self, item: EvidenceWriterQueueItemV1) -> None: ...


class BoundedEvidenceWriterQueueV1:
    """Three-lane in-memory ingress used before Task 4 supplies a writer."""

    protocol_version: Literal[1] = 1

    def __init__(self) -> None:
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._ordinary_scheduler_binding: _OrdinarySchedulerBindingV1 | None = None
        self._ordered: deque[EvidenceWriterQueueItemV1] = deque()
        self._revoke: EvidenceWriterQueueItemV1 | None = None
        self._drain: EvidenceWriterQueueItemV1 | None = None
        self._consumer_stopped = False

    @property
    def ordered_count(self) -> int:
        with self._lock:
            return len(self._ordered)

    @property
    def pending_revoke(self) -> bool:
        with self._lock:
            return self._revoke is not None

    @property
    def pending_drain(self) -> bool:
        with self._lock:
            return self._drain is not None

    def put_nowait(self, item: EvidenceWriterQueueItemV1) -> None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise TypeError("writer item must be an exact EvidenceWriterQueueItemV1")
        with self._condition:
            if self._consumer_stopped:
                raise ReservationError("writer queue consumer is stopped")
            if item.lane is WriterQueueLane.ORDERED:
                if len(self._ordered) >= MAX_QUEUE_PHYSICAL_ITEMS:
                    raise Full
                self._ordered.append(item)
                self._condition.notify()
                return
            if item.lane is WriterQueueLane.REVOKE:
                if self._revoke is not None:
                    raise Full
                self._revoke = item
                self._condition.notify()
                return
            if item.lane is WriterQueueLane.DRAIN:
                if self._drain is not None:
                    raise Full
                self._drain = item
                self._condition.notify()
                return
            raise ValueError("writer item has no V1 lane")

    def put_ordered_batch_nowait(
        self,
        items: tuple[EvidenceWriterQueueItemV1, ...],
    ) -> None:
        """Expose one prevalidated ordered batch atomically."""

        if type(items) is not tuple or not items:
            raise TypeError("ordered batch must be a nonempty exact tuple")
        if any(
            type(item) is not EvidenceWriterQueueItemV1 or item.lane is not WriterQueueLane.ORDERED
            for item in items
        ):
            raise TypeError("ordered batch contains an invalid writer item")
        with self._condition:
            if self._consumer_stopped:
                raise ReservationError("writer queue consumer is stopped")
            if len(self._ordered) + len(items) > MAX_QUEUE_PHYSICAL_ITEMS:
                raise Full
            self._ordered.extend(items)
            self._condition.notify()

    def get_nowait(self) -> EvidenceWriterQueueItemV1:
        """Select revoke first, then ordered FIFO, and drain only after both."""

        with self._lock:
            if self._revoke is not None:
                item = self._revoke
                self._revoke = None
                return item
            if self._ordered:
                return self._ordered.popleft()
            if self._drain is not None:
                item = self._drain
                self._drain = None
                return item
            raise Empty

    def get_blocking(self) -> EvidenceWriterQueueItemV1 | None:
        """Wait for the next priority item or permanent consumer stop."""

        with self._condition:
            while True:
                if self._revoke is not None:
                    item = self._revoke
                    self._revoke = None
                    return item
                if self._ordered:
                    return self._ordered.popleft()
                if self._drain is not None:
                    item = self._drain
                    self._drain = None
                    return item
                if self._consumer_stopped:
                    return None
                self._condition.wait()

    def stop_consumer(self) -> None:
        """Permanently close publication and wake the blocking consumer."""

        with self._condition:
            self._consumer_stopped = True
            self._condition.notify_all()


class EvidenceAdmissionError(ValueError):
    """A prevalidated-looking record violates the bounded admission contract."""


def _walk_text_values(value: object) -> tuple[str, ...]:
    if type(value) is dict:
        values = value
        found: list[str] = []
        for key, item in values.items():
            if key == "text":
                if type(item) is not str:
                    raise EvidenceAdmissionError("text must be an exact built-in string")
                found.append(item)
            else:
                found.extend(_walk_text_values(item))
        return tuple(found)
    if type(value) is list:
        found = []
        for item in value:
            found.extend(_walk_text_values(item))
        return tuple(found)
    return ()


def _copy_snapshot(snapshot: EvidenceSnapshotV1) -> EvidenceSnapshotV1:
    if type(snapshot) is not EvidenceSnapshotV1:
        raise TypeError("snapshot must be an exact EvidenceSnapshotV1")
    return EvidenceSnapshotV1(
        schema_version=snapshot.schema_version,
        installation_id=snapshot.installation_id,
        producer_instance_id=snapshot.producer_instance_id,
        logical_session_id=snapshot.logical_session_id,
        event_id=snapshot.event_id,
        event_sequence=snapshot.event_sequence,
        event_kind=snapshot.event_kind,
        payload=snapshot.payload,
    )


def _canonical_record_representation(
    snapshot: EvidenceSnapshotV1,
) -> tuple[EvidenceSnapshotV1, bytes]:
    exact_snapshot = _copy_snapshot(snapshot)
    primitive = evidence_snapshot_to_primitive(exact_snapshot)
    for text in _walk_text_values(primitive):
        code_points = len(text)
        if (
            code_points > 4_096
            or any(0xD800 <= ord(character) <= 0xDFFF for character in text)
            or 6 * code_points + 8_192 > MAX_CANONICAL_RECORD_BYTES
        ):
            raise EvidenceAdmissionError("evidence text exceeds its exact admission bound")
    encoded = json.dumps(
        primitive,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
        raise EvidenceAdmissionError("canonical evidence record exceeds 32768 bytes")
    return exact_snapshot, encoded


def canonical_record_bytes(snapshot: EvidenceSnapshotV1) -> int:
    """Return the exact canonical queue charge after the cheap text bound."""

    _, encoded = _canonical_record_representation(snapshot)
    return len(encoded)


def _raw_snapshot_text_is_oversize(snapshot: object) -> bool:
    if type(snapshot) is not EvidenceSnapshotV1:
        raise TypeError("snapshot must be an exact EvidenceSnapshotV1")
    exact_snapshot = snapshot
    if exact_snapshot.event_kind not in (
        EventKind.USER_FINAL_ACCEPTED,
        EventKind.ASSISTANT_SEGMENT_GENERATED,
        EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
    ):
        return False
    text = getattr(exact_snapshot.payload, "text", None)
    if text is None:
        return False
    if type(text) is not str:
        raise TypeError("snapshot text must be an exact built-in string")
    code_points = len(text)
    return (
        code_points > 4_096
        or any(0xD800 <= ord(character) <= 0xDFFF for character in text)
        or 6 * code_points + 8_192 > MAX_CANONICAL_RECORD_BYTES
    )


class _CreateReservationPhase(Enum):
    RESERVED = auto()
    QUEUED = auto()


@dataclass(slots=True)
class _CreateReservationState:
    command: CreateEpochV1
    phase: _CreateReservationPhase
    item: EvidenceWriterQueueItemV1 | None = None


@dataclass(slots=True)
class _TerminalReservationState:
    remaining_records: int = TERMINAL_RECORD_CREDITS
    remaining_bytes: int = TERMINAL_CANONICAL_BYTE_CREDITS
    queued_records: int = 0


@dataclass(slots=True)
class _SessionControlState:
    remaining_records: int = SESSION_CONTROL_RECORD_CREDITS
    remaining_bytes: int = SESSION_CONTROL_CANONICAL_BYTE_CREDITS
    queued_records: int = 0


@dataclass(slots=True)
class _LeaseState:
    authority: UserTurnAuthorityV1 | ProactiveTurnAuthorityV1 | ReplayTurnAuthorityV1
    operation: ConversationOperationReservation
    terminal_reservation: EvidenceTerminalReservation
    turn_kind: TurnKind
    lease_serial: int
    evidence_turn_id: str
    operation_serial: int
    authority_signature: tuple[object, ...]
    event_count: int = 0
    canonical_bytes: int = 0
    lease_open_ordinal: int | None = None
    opened: bool = False
    user_final: bool = False
    snapshot: bool = False
    settlement_attempted: bool = False
    generated_segment_count: int = 0
    transport_confirmed_full_count: int = 0
    evidence_segment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _CauseState:
    lease: EvidenceTurnLease
    lock: Lock
    causes: set[TerminalReason]
    sink_token: bytes
    frozen: bool = False
    resolution: TerminalResolutionV1 | None = None
    context_committed: bool | None = None
    settled_outcome: SettledTerminalOutcomeV1 | None = None


@dataclass(slots=True)
class _QueuedCharge:
    records_on_completion: int
    bytes_on_completion: int
    release_physical_item: bool
    category: str
    reservation: object | None


class _IdentityQueuedChargesV1:
    """Identity-keyed outstanding charges; structural DTO forgery cannot complete work."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[EvidenceWriterQueueItemV1, _QueuedCharge]] = {}

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __setitem__(self, item: EvidenceWriterQueueItemV1, charge: _QueuedCharge) -> None:
        if type(item) is not EvidenceWriterQueueItemV1 or type(charge) is not _QueuedCharge:
            raise TypeError("queued charge requires exact production types")
        key = id(item)
        if key in self._entries:
            raise ReservationError("writer item is already charged")
        self._entries[key] = (item, charge)

    def get(
        self,
        item: EvidenceWriterQueueItemV1,
        default: _QueuedCharge | None = None,
    ) -> _QueuedCharge | None:
        entry = self._entries.get(id(item))
        if entry is None or entry[0] is not item:
            return default
        return entry[1]

    def pop(
        self,
        item: EvidenceWriterQueueItemV1,
        default: _QueuedCharge | None = None,
    ) -> _QueuedCharge | None:
        entry = self._entries.get(id(item))
        if entry is None or entry[0] is not item:
            return default
        del self._entries[id(item)]
        return entry[1]

    def __delitem__(self, item: EvidenceWriterQueueItemV1) -> None:
        entry = self._entries.get(id(item))
        if entry is None or entry[0] is not item:
            raise KeyError(item)
        del self._entries[id(item)]


class _AdmissionCapacityV1:
    """Single lock, credit ledger, charge ownership, and ordinal source of truth."""

    def __init__(
        self,
        *,
        max_records: int,
        max_canonical_bytes: int,
        max_physical_items: int,
    ) -> None:
        self.lock = Lock()
        self.max_records = _exact_positive_int(max_records, "max_records", _MAX_UNSIGNED_63)
        self.max_canonical_bytes = _exact_positive_int(
            max_canonical_bytes,
            "max_canonical_bytes",
            _MAX_UNSIGNED_63,
        )
        self.max_physical_items = _exact_positive_int(
            max_physical_items,
            "max_physical_items",
            _MAX_UNSIGNED_63,
        )
        self.charged_records = 0
        self.charged_bytes = 0
        self.ordered_physical_items = 0
        self.next_admission_ordinal = 1
        self.queued_charges = _IdentityQueuedChargesV1()
        self._ordinary_scheduler_binding: _OrdinarySchedulerBindingV1 | None = None

    def _try_charge_observed(
        self,
        records: int,
        canonical_bytes: int,
        *,
        physical_items: int,
    ) -> Literal[
        "record_capacity", "canonical_byte_capacity", "physical_capacity"
    ] | None:
        if self.charged_records + records > self.max_records:
            return "record_capacity"
        if self.charged_bytes + canonical_bytes > self.max_canonical_bytes:
            return "canonical_byte_capacity"
        if self.ordered_physical_items + physical_items > self.max_physical_items:
            return "physical_capacity"
        self.charged_records += records
        self.charged_bytes += canonical_bytes
        self.ordered_physical_items += physical_items
        return None

    def try_charge(self, records: int, canonical_bytes: int, *, physical_items: int) -> bool:
        return (
            self._try_charge_observed(
                records,
                canonical_bytes,
                physical_items=physical_items,
            )
            is None
        )

    def release(self, records: int, canonical_bytes: int, *, physical_items: int) -> None:
        if (
            records < 0
            or canonical_bytes < 0
            or physical_items < 0
            or records > self.charged_records
            or canonical_bytes > self.charged_bytes
            or physical_items > self.ordered_physical_items
        ):
            raise ReservationError("queue credit release underflowed")
        self.charged_records -= records
        self.charged_bytes -= canonical_bytes
        self.ordered_physical_items -= physical_items

    def ordinals_available(self, count: int) -> bool:
        if type(count) is not int or count < 1:
            raise TypeError("admission ordinal count must be a positive exact integer")
        return self.next_admission_ordinal <= _MAX_UNSIGNED_63 - count + 1

    def allocate_ordinal(self) -> int:
        ordinal = self.next_admission_ordinal
        if ordinal > _MAX_UNSIGNED_63:
            raise ReservationError("admission ordinal overflowed")
        self.next_admission_ordinal += 1
        return ordinal

    def rollback_last_ordinal(self, ordinal: int) -> None:
        if self.next_admission_ordinal != ordinal + 1:
            raise ReservationError("admission ordinal rollback is no longer contiguous")
        self.next_admission_ordinal = ordinal


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _PreparedEvidenceRecordV1:
    """Detached caller-visible handle; its fields are never accounting authority."""

    snapshot: EvidenceSnapshotV1
    canonical_bytes: bytes
    canonical_byte_charge: int
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class _IssuedPreparedEvidenceV1:
    """Scheduler-retained immutable preparation authority."""

    snapshot: EvidenceSnapshotV1
    canonical_bytes: bytes
    canonical_byte_charge: int
    canonical_sha256: str


_ordinary_scheduler_binding_lock = RLock()


class _OrdinarySchedulerBindingV1:
    """One lifetime binding between an ordinary scheduler and its shared domains."""

    __slots__ = ("capacity", "writer_sink", "_claimed")

    def __init__(
        self,
        *,
        capacity: _AdmissionCapacityV1,
        writer_sink: BoundedEvidenceWriterQueueV1,
    ) -> None:
        self.capacity = capacity
        self.writer_sink = writer_sink
        self._claimed = False

    def claim(self) -> None:
        with _ordinary_scheduler_binding_lock:
            if (
                self._claimed
                or self.capacity._ordinary_scheduler_binding is not self
                or self.writer_sink._ordinary_scheduler_binding is not self
            ):
                raise ReservationError("ordinary scheduler binding is already claimed")
            self._claimed = True


def _bind_ordinary_scheduler_v1(
    *,
    capacity: _AdmissionCapacityV1,
    writer_sink: BoundedEvidenceWriterQueueV1,
) -> _OrdinarySchedulerBindingV1:
    if type(capacity) is not _AdmissionCapacityV1:
        raise TypeError("capacity must be an exact _AdmissionCapacityV1")
    if type(writer_sink) is not BoundedEvidenceWriterQueueV1:
        raise TypeError("writer_sink must be an exact BoundedEvidenceWriterQueueV1")
    with _ordinary_scheduler_binding_lock:
        if (
            capacity._ordinary_scheduler_binding is not None
            or writer_sink._ordinary_scheduler_binding is not None
        ):
            raise ReservationError("ordinary scheduler capacity or writer sink is already bound")
        binding = _OrdinarySchedulerBindingV1(capacity=capacity, writer_sink=writer_sink)
        capacity._ordinary_scheduler_binding = binding
        writer_sink._ordinary_scheduler_binding = binding
        return binding


class _SchedulerPublicationFull(ReservationError):
    pass


class _SchedulerPublicationFault(ReservationError):
    pass


class _ProductionEvidenceSchedulerV1:
    """Production ordinary scheduler shared by live admission and isolated characterization."""

    def __init__(
        self,
        *,
        binding: _OrdinarySchedulerBindingV1,
    ) -> None:
        if type(binding) is not _OrdinarySchedulerBindingV1:
            raise TypeError("binding must be an exact ordinary scheduler binding")
        binding.claim()
        self._capacity = binding.capacity
        self._writer_sink = binding.writer_sink
        self._prepared_lock = RLock()
        self._issued_prepared: dict[
            int,
            tuple[ReferenceType[_PreparedEvidenceRecordV1], _IssuedPreparedEvidenceV1],
        ] = {}

    def prepare(self, snapshot: EvidenceSnapshotV1) -> _PreparedEvidenceRecordV1:
        handle_snapshot, canonical = _canonical_record_representation(snapshot)
        issued = _IssuedPreparedEvidenceV1(
            snapshot=_copy_snapshot(handle_snapshot),
            canonical_bytes=canonical,
            canonical_byte_charge=len(canonical),
            canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        )
        handle = _PreparedEvidenceRecordV1(
            snapshot=handle_snapshot,
            canonical_bytes=issued.canonical_bytes,
            canonical_byte_charge=issued.canonical_byte_charge,
            canonical_sha256=issued.canonical_sha256,
        )
        key = id(handle)

        def forget(dead_ref: ReferenceType[_PreparedEvidenceRecordV1]) -> None:
            with self._prepared_lock:
                current = self._issued_prepared.get(key)
                if current is not None and current[0] is dead_ref:
                    del self._issued_prepared[key]

        handle_ref = ref(handle, forget)
        with self._prepared_lock:
            self._issued_prepared[key] = (handle_ref, issued)
        return handle

    def _require_issued_prepared(
        self,
        prepared: _PreparedEvidenceRecordV1,
    ) -> _IssuedPreparedEvidenceV1:
        if type(prepared) is not _PreparedEvidenceRecordV1:
            raise ReservationError("prepared record is not scheduler-issued")
        with self._prepared_lock:
            entry = self._issued_prepared.get(id(prepared))
            if entry is None or entry[0]() is not prepared:
                raise ReservationError("prepared record is forged, stale, or cross-owner")
            issued = entry[1]
            if (
                prepared.snapshot != issued.snapshot
                or prepared.canonical_bytes != issued.canonical_bytes
                or prepared.canonical_byte_charge != issued.canonical_byte_charge
                or prepared.canonical_sha256 != issued.canonical_sha256
            ):
                raise ReservationError("prepared record no longer matches scheduler authority")
            return issued

    @property
    def credits(self) -> tuple[int, int, int]:
        with self._capacity.lock:
            return (
                self._capacity.charged_records,
                self._capacity.charged_bytes,
                self._capacity.ordered_physical_items,
            )

    @property
    def next_admission_ordinal(self) -> int:
        with self._capacity.lock:
            return self._capacity.next_admission_ordinal

    def try_admit(
        self,
        prepared: _PreparedEvidenceRecordV1,
    ) -> EvidenceWriterQueueItemV1 | None:
        item, _rejection_source = self._try_admit_observed(prepared)
        return item

    def _try_admit_observed(
        self,
        prepared: _PreparedEvidenceRecordV1,
    ) -> tuple[
        EvidenceWriterQueueItemV1 | None,
        Literal["record_capacity", "canonical_byte_capacity", "physical_capacity"] | None,
    ]:
        issued = self._require_issued_prepared(prepared)
        canonical_byte_charge = issued.canonical_byte_charge
        with self._capacity.lock:
            if not self._capacity.ordinals_available(1):
                raise ReservationError("admission ordinal overflowed")
            rejection_source = self._capacity._try_charge_observed(
                1,
                canonical_byte_charge,
                physical_items=1,
            )
            if rejection_source is not None:
                return None, rejection_source
            ordinal = self._capacity.allocate_ordinal()
            record = QueuedEvidenceRecordV1(
                protocol_version=1,
                snapshot=issued.snapshot,
                admission_ordinal=ordinal,
                reservation_class=QueueReservationClass.ORDINARY,
                lease_open_ordinal=None,
            )
            item = EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=WriterQueueLane.ORDERED,
                payload=record,
                admission_ordinal=ordinal,
            )
            self._capacity.queued_charges[item] = _QueuedCharge(
                records_on_completion=1,
                bytes_on_completion=canonical_byte_charge,
                release_physical_item=True,
                category="ordinary",
                reservation=None,
            )
            try:
                self._writer_sink.put_nowait(item)
            except Full as error:
                self._rollback_single_publish_locked(item, ordinal, canonical_byte_charge)
                raise _SchedulerPublicationFull(
                    "bounded writer queue rejected admission"
                ) from error
            except Exception as error:
                self._rollback_single_publish_locked(item, ordinal, canonical_byte_charge)
                raise _SchedulerPublicationFault("writer queue publication failed") from error
            return item, None

    def _rollback_single_publish_locked(
        self,
        item: EvidenceWriterQueueItemV1,
        ordinal: int,
        canonical_byte_charge: int,
    ) -> None:
        charge = self._capacity.queued_charges.pop(item, None)
        if charge is None:
            raise ReservationError("failed publication lost its charge")
        self._capacity.release(1, canonical_byte_charge, physical_items=1)
        self._capacity.rollback_last_ordinal(ordinal)

    def try_admit_batch(
        self,
        prepared_records: tuple[_PreparedEvidenceRecordV1, ...],
    ) -> tuple[EvidenceWriterQueueItemV1, ...] | None:
        items, _rejection_source = self._try_admit_batch_observed(prepared_records)
        return items

    def _try_admit_batch_observed(
        self,
        prepared_records: tuple[_PreparedEvidenceRecordV1, ...],
    ) -> tuple[
        tuple[EvidenceWriterQueueItemV1, ...] | None,
        Literal["record_capacity", "canonical_byte_capacity", "physical_capacity"] | None,
    ]:
        if type(prepared_records) is not tuple or not prepared_records:
            raise TypeError("prepared records must be an exact non-empty tuple")
        issued_records = tuple(
            self._require_issued_prepared(prepared) for prepared in prepared_records
        )
        count = len(issued_records)
        canonical_byte_charge = sum(issued.canonical_byte_charge for issued in issued_records)
        with self._capacity.lock:
            if not self._capacity.ordinals_available(count):
                raise ReservationError("admission ordinal overflowed")
            rejection_source = self._capacity._try_charge_observed(
                count,
                canonical_byte_charge,
                physical_items=count,
            )
            if rejection_source is not None:
                return None, rejection_source
            ordinals = tuple(self._capacity.allocate_ordinal() for _ in issued_records)
            records = tuple(
                QueuedEvidenceRecordV1(
                    protocol_version=1,
                    snapshot=issued.snapshot,
                    admission_ordinal=ordinal,
                    reservation_class=QueueReservationClass.ORDINARY,
                    lease_open_ordinal=None,
                )
                for issued, ordinal in zip(issued_records, ordinals, strict=True)
            )
            items = tuple(
                EvidenceWriterQueueItemV1(
                    protocol_version=1,
                    lane=WriterQueueLane.ORDERED,
                    payload=record,
                    admission_ordinal=record.admission_ordinal,
                )
                for record in records
            )
            for item, issued in zip(items, issued_records, strict=True):
                self._capacity.queued_charges[item] = _QueuedCharge(
                    records_on_completion=1,
                    bytes_on_completion=issued.canonical_byte_charge,
                    release_physical_item=True,
                    category="ordinary",
                    reservation=None,
                )
            try:
                if count == 1:
                    self._writer_sink.put_nowait(items[0])
                else:
                    self._writer_sink.put_ordered_batch_nowait(items)
            except Full as error:
                self._rollback_batch_publish_locked(items, ordinals, canonical_byte_charge)
                raise _SchedulerPublicationFull(
                    "bounded writer queue rejected admission"
                ) from error
            except Exception as error:
                self._rollback_batch_publish_locked(items, ordinals, canonical_byte_charge)
                raise _SchedulerPublicationFault("writer queue publication failed") from error
            return items, None

    def _rollback_batch_publish_locked(
        self,
        items: tuple[EvidenceWriterQueueItemV1, ...],
        ordinals: tuple[int, ...],
        canonical_byte_charge: int,
    ) -> None:
        for item in items:
            charge = self._capacity.queued_charges.pop(item, None)
            if charge is None:
                raise ReservationError("failed publication lost its charge")
        self._capacity.release(
            len(items),
            canonical_byte_charge,
            physical_items=len(items),
        )
        for ordinal in reversed(ordinals):
            self._capacity.rollback_last_ordinal(ordinal)

    def dequeue_nowait(self) -> EvidenceWriterQueueItemV1:
        return self._writer_sink.get_nowait()

    def complete(self, item: EvidenceWriterQueueItemV1) -> None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("writer item has the wrong type")
        with self._capacity.lock:
            self._complete_locked(item)

    def _complete_locked(self, item: EvidenceWriterQueueItemV1) -> None:
        charge = self._capacity.queued_charges.get(item)
        if charge is None or charge.category != "ordinary":
            raise ReservationError("ordinary writer item is stale or has the wrong owner")
        del self._capacity.queued_charges[item]
        self._capacity.release(
            charge.records_on_completion,
            charge.bytes_on_completion,
            physical_items=1 if charge.release_physical_item else 0,
        )


def _new_controller_ordinary_scheduler_v1(
    *,
    capacity: _AdmissionCapacityV1,
    writer_sink: BoundedEvidenceWriterQueueV1,
) -> _ProductionEvidenceSchedulerV1:
    binding = _bind_ordinary_scheduler_v1(capacity=capacity, writer_sink=writer_sink)
    return _ProductionEvidenceSchedulerV1(binding=binding)


def _new_production_evidence_scheduler_v1() -> _ProductionEvidenceSchedulerV1:
    capacity = _AdmissionCapacityV1(
        max_records=MAX_QUEUE_RECORDS,
        max_canonical_bytes=MAX_QUEUE_CANONICAL_BYTES,
        max_physical_items=MAX_QUEUE_PHYSICAL_ITEMS,
    )
    return _new_controller_ordinary_scheduler_v1(
        capacity=capacity,
        writer_sink=BoundedEvidenceWriterQueueV1(),
    )


class EvidenceAdmissionControllerV1:
    """Fail-closed admission owner with exact queue-credit accounting."""

    protocol_version: Literal[1] = 1

    def __init__(
        self,
        *,
        enabled: bool,
        owner_generation: int,
        writer_sink: BoundedEvidenceWriterQueueV1,
        operation_scheduler: ConversationOperationScheduler | None = None,
        conversation_authority_is_current: Callable[[object], bool] | None = None,
        deny_filter: Callable[[EvidenceSnapshotV1], bool] | None = None,
        quota_check: Callable[[EvidenceSnapshotV1], bool] | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be an exact built-in bool")
        self._owner_generation = _exact_positive_int(
            owner_generation,
            "owner_generation",
            _MAX_UNSIGNED_63,
        )
        if type(writer_sink) is not BoundedEvidenceWriterQueueV1:
            raise TypeError("writer_sink must be an exact BoundedEvidenceWriterQueueV1")
        if (
            operation_scheduler is not None
            and type(operation_scheduler) is not ConversationOperationScheduler
        ):
            raise TypeError("operation_scheduler must be exact or None")
        if (
            operation_scheduler is not None
            and operation_scheduler.owner_generation != owner_generation
        ):
            raise ValueError("operation_scheduler owner generation must match admission owner")
        if conversation_authority_is_current is not None and not callable(
            conversation_authority_is_current
        ):
            raise TypeError("conversation authority guard must be callable or None")
        if deny_filter is not None and not callable(deny_filter):
            raise TypeError("deny_filter must be callable or None")
        if quota_check is not None and not callable(quota_check):
            raise TypeError("quota_check must be callable or None")
        self._enabled = enabled
        self._writer_sink = writer_sink
        self._operation_scheduler = (
            operation_scheduler
            if operation_scheduler is not None
            else ConversationOperationScheduler(
                owner_generation=owner_generation,
                max_operations=MAX_QUEUE_RECORDS,
            )
        )
        self._conversation_authority_is_current = conversation_authority_is_current
        self._deny_filter = deny_filter
        self._quota_check = quota_check
        self._rollover_claimed: (
            Callable[[_AdmissionRolloverPreparationV1], None] | None
        ) = None
        self._admission_lock = Lock()
        self._capacity = _AdmissionCapacityV1(
            max_records=MAX_QUEUE_RECORDS,
            max_canonical_bytes=MAX_QUEUE_CANONICAL_BYTES,
            max_physical_items=MAX_QUEUE_PHYSICAL_ITEMS,
        )
        self._credit_lock = self._capacity.lock
        self._ordinary_scheduler = _new_controller_ordinary_scheduler_v1(
            capacity=self._capacity,
            writer_sink=writer_sink,
        )
        from hermes_realtime._qualification import _current_qualification_capacity_probe

        self._qualification_capacity_probe = _current_qualification_capacity_probe()
        self._revoke_finalize_lock = Lock()
        self._next_reservation_serial = 1
        self._create_reservations: dict[
            CreateEpochReservationV1,
            _CreateReservationState,
        ] = {}
        self._terminal_reservations: dict[
            EvidenceTerminalReservation,
            _TerminalReservationState,
        ] = {}
        self._session_controls: dict[
            SessionControlReservation,
            _SessionControlState,
        ] = {}
        self._create_tickets: dict[CreateEpochReservationV1, CreateEpochTicketV1] = {}
        self._leases: dict[EvidenceTurnLease, _LeaseState] = {}
        self._cause_states: dict[TerminalCauseCapabilityV1, _CauseState] = {}
        self._used_authorities: set[object] = set()
        self._evidence_turn_ids: set[str] = set()
        self._owner_state = OwnerState.ABSENT
        self._capture_state = CaptureState.IDLE if enabled else CaptureState.UNAVAILABLE
        self._sticky_fault: WriterFault | None = None
        self._pending_revoke = False
        self._purge_required = False
        self._current_session_control: SessionControlReservation | None = None
        self._installation_id: str | None = None
        self._producer_instance_id: str | None = None
        self._consent_epoch_id: str | None = None
        self._logical_session_id: str | None = None
        self._binding_id: str | None = None
        self._binding_generation: int | None = None
        self._next_event_sequence: int | None = None
        self._phase: RuntimeSessionPhase | None = None
        self._session_tainted = False
        self._taint_code: TaintCode | None = None
        self._session_reserved_events = 0
        self._session_reserved_bytes = 0
        self._binding_close_admitted = False
        self._seal_ticket_disposition: SealDisposition | None = None
        self._expiry_authority: SessionExpiryAuthorityV1 | None = None
        self._replayable_turn_count = 0
        self._revoke_authority: ConsentRevokeAuthorityV1 | None = None
        self._revoke_ticket: RevokeTicketV1 | None = None
        self._revoke_item: EvidenceWriterQueueItemV1 | None = None
        self._revoke_request_completed = False
        self._drain_authority: LifecycleDrainAuthorityV1 | None = None
        self._drain_ticket: DrainTicketV1 | None = None
        self._drain_item: EvidenceWriterQueueItemV1 | None = None
        self._rollover_preparation: _AdmissionRolloverPreparationV1 | None = None
        self._rollover_ticket: _RolloverCompletionTicketV1 | None = None

    @property
    def _queued_charges(self) -> _IdentityQueuedChargesV1:
        return self._capacity.queued_charges

    def _install_rollover_runtime_owner(
        self,
        callback: Callable[[_AdmissionRolloverPreparationV1], None],
        token: object,
    ) -> None:
        if token is not _ROLLOVER_RUNTIME_OWNER_TOKEN or not callable(callback):
            raise TypeError("capacity rollover owner is runtime-installed")
        with self._admission_lock:
            if self._rollover_claimed is not None or self._owner_state is not OwnerState.ABSENT:
                raise ReservationError("capacity rollover owner installation is closed")
            self._rollover_claimed = callback

    @property
    def _charged_records(self) -> int:
        return self._capacity.charged_records

    @_charged_records.setter
    def _charged_records(self, value: int) -> None:
        self._capacity.charged_records = value

    @property
    def _charged_bytes(self) -> int:
        return self._capacity.charged_bytes

    @_charged_bytes.setter
    def _charged_bytes(self, value: int) -> None:
        self._capacity.charged_bytes = value

    @property
    def _ordered_physical_items(self) -> int:
        return self._capacity.ordered_physical_items

    @_ordered_physical_items.setter
    def _ordered_physical_items(self, value: int) -> None:
        self._capacity.ordered_physical_items = value

    @property
    def _next_admission_ordinal(self) -> int:
        return self._capacity.next_admission_ordinal

    @_next_admission_ordinal.setter
    def _next_admission_ordinal(self, value: int) -> None:
        self._capacity.next_admission_ordinal = value

    def diagnostics(self) -> EvidenceDiagnosticsV1:
        with self._admission_lock, self._credit_lock:
            return EvidenceDiagnosticsV1(
                protocol_version=1,
                owner_state=self._owner_state,
                capture_state=self._capture_state,
                sticky_fault=self._sticky_fault,
                queue_record_count=self._charged_records,
                queue_canonical_bytes=self._charged_bytes,
                active_lease_count=len(self._leases),
                pending_revoke=self._pending_revoke,
                purge_required=self._purge_required,
            )

    @property
    def operation_scheduler(self) -> ConversationOperationScheduler:
        return self._operation_scheduler

    @property
    def final_admission_ordinal(self) -> int:
        with self._admission_lock, self._credit_lock:
            return self._next_admission_ordinal - 1

    def begin_consent(
        self,
        authority: ConsentCreateAuthorityV1,
    ) -> CreateEpochTicketV1:
        if type(authority) is not ConsentCreateAuthorityV1:
            raise ReservationError("consent authority has the wrong type")
        try:
            authority._validate()
            reservation = authority.create_epoch_reservation
            with self._admission_lock, self._credit_lock:
                existing = self._create_tickets.get(reservation)
                if existing is not None:
                    return existing
                state = self._create_reservations.get(reservation)
                if state is None or state.phase is not _CreateReservationPhase.RESERVED:
                    raise ReservationError("consent create reservation is stale")
                validate_authority_composition(authority, state.command)
                ticket = CreateEpochTicketV1(
                    control_sequence=authority.control_sequence,
                    disposition=ConsentDisposition.CREATE_PENDING,
                    durability_event=Event(),
                )
                self._create_tickets[reservation] = ticket
            item = self.try_enqueue_create_epoch(reservation)
            if item is None:
                with self._credit_lock:
                    self._create_tickets.pop(reservation, None)
                    object.__setattr__(
                        ticket,
                        "disposition",
                        ConsentDisposition.CREATE_FAILED,
                    )
                    _set_ticket_signal(ticket.durability_event)
            return ticket
        except (KeyboardInterrupt, SystemExit):
            raise
        except ReservationError:
            raise
        except Exception as error:
            raise ReservationError("consent admission failed closed") from error

    def _observe_capacity(self, *, kind: str, rejection_source: str = "none") -> None:
        probe = self._qualification_capacity_probe
        record = getattr(probe, "record", None)
        if callable(record):
            record(
                kind=kind,
                rejection_source=rejection_source,
                records=self._capacity.charged_records,
                canonical_bytes=self._capacity.charged_bytes,
                physical_items=self._capacity.ordered_physical_items,
            )

    def begin_revoke(self, authority: ConsentRevokeAuthorityV1) -> RevokeTicketV1:
        if type(authority) is not ConsentRevokeAuthorityV1:
            raise ReservationError("revoke authority has the wrong type")
        try:
            authority._validate()
            with self._admission_lock:
                if self._revoke_ticket is not None:
                    if self._revoke_authority is authority:
                        self._observe_capacity(kind="revoke_coalesced")
                        return self._revoke_ticket
                    raise ReservationError("a different revoke is already coalesced")
                if (
                    authority.consent_epoch_id != self._consent_epoch_id
                    or authority.binding_id != self._binding_id
                    or authority.binding_generation != self._binding_generation
                    or self._phase is None
                    or self._phase is RuntimeSessionPhase.STOPPED
                ):
                    raise ReservationError("revoke authority is stale")
                with self._credit_lock:
                    last_ordinal = self._next_admission_ordinal - 1
                    command = RevokeRequestV1(
                        protocol_version=1,
                        erasure_request_id=str(uuid4()),
                        consent_epoch_id=authority.consent_epoch_id,
                        control_sequence=authority.control_sequence,
                        control_fingerprint_hash=authority.control_fingerprint_hash,
                        last_admission_ordinal=last_ordinal,
                    )
                    ordinal = self._allocate_admission_ordinal_locked()
                    item = EvidenceWriterQueueItemV1(
                        protocol_version=1,
                        lane=WriterQueueLane.REVOKE,
                        payload=command,
                        admission_ordinal=ordinal,
                    )
                ticket = RevokeTicketV1(
                    control_sequence=authority.control_sequence,
                    erasure_request_id=command.erasure_request_id,
                    disposition=RevokeDisposition.CLOSED_NOT_DURABLE,
                    durability_event=Event(),
                    terminal_event=Event(),
                )
                self._revoke_authority = authority
                self._revoke_ticket = ticket
                self._revoke_item = item
                self._pending_revoke = True
                self._phase = RuntimeSessionPhase.REVOKING
                self._capture_state = CaptureState.REVOKED_PURGING
                self._observe_capacity(kind="revoke_accepted")
                try:
                    self._writer_sink.put_nowait(item)
                except Exception:
                    self._revoke_item = None
                    object.__setattr__(
                        ticket,
                        "disposition",
                        RevokeDisposition.WRITER_FAULT,
                    )
                    self._pending_revoke = False
                    self._purge_required = True
                    self._phase = RuntimeSessionPhase.STOPPED
                    _set_ticket_signal(ticket.durability_event)
                    _set_ticket_signal(ticket.terminal_event)
                    with self._credit_lock:
                        self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                return ticket
        except (KeyboardInterrupt, SystemExit):
            raise
        except ReservationError:
            raise
        except Exception as error:
            raise ReservationError("revoke admission failed closed") from error

    def complete_revoke_request(
        self,
        item: EvidenceWriterQueueItemV1,
        disposition: RevokeDisposition,
    ) -> None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("revoke item has the wrong type")
        if type(item.payload) is not RevokeRequestV1:
            raise ReservationError("revoke request item has the wrong payload")
        if type(disposition) is not RevokeDisposition:
            raise TypeError("disposition must be an exact RevokeDisposition")
        with self._credit_lock:
            ticket = self._revoke_ticket
            if ticket is None or self._revoke_item is not item:
                raise ReservationError("revoke request item is stale")
            if disposition in (
                RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
                RevokeDisposition.ALREADY_SCHEDULED,
            ):
                self._revoke_item = None
                object.__setattr__(
                    ticket,
                    "disposition",
                    RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
                )
                self._revoke_request_completed = True
                _set_ticket_signal(ticket.durability_event)
            else:
                # A request commit can establish durable erase work, never claim its
                # terminal purge result.  Treat every other exact enum member as a
                # transport boundary fault while the exact dequeued item is still
                # owned, so no live lease or revoke ticket can be stranded.
                self._complete_revoke_writer_fault_locked(ticket, item)
        if disposition in (
            RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
            RevokeDisposition.ALREADY_SCHEDULED,
        ):
            self._advance_revoke_finalization_if_ready()

    def try_enqueue_revoke_finalize(
        self,
        ticket: RevokeTicketV1,
    ) -> EvidenceWriterQueueItemV1 | None:
        if type(ticket) is not RevokeTicketV1 or ticket is not self._revoke_ticket:
            raise ReservationError("revoke ticket is stale")
        with self._revoke_finalize_lock:
            return self._try_enqueue_revoke_finalize_locked(ticket)

    def _advance_revoke_finalization_if_ready(self) -> None:
        """Publish ready revoke cleanup without blocking foreground admission."""

        ticket = self._revoke_ticket
        if ticket is not None:
            with self._revoke_finalize_lock:
                self._try_enqueue_revoke_finalize_locked(ticket)

    def _try_enqueue_revoke_finalize_locked(
        self,
        ticket: RevokeTicketV1,
    ) -> EvidenceWriterQueueItemV1 | None:
        if ticket.disposition is not RevokeDisposition.REVOKE_DURABLY_SCHEDULED:
            return None
        if self._revoke_item is not None:
            if type(self._revoke_item.payload) is RevokeFinalizeV1:
                return self._revoke_item
            return None
        if not self._revoke_request_completed or self._leases:
            return None
        authority = self._revoke_authority
        if authority is None:
            raise ReservationError("revoke authority was lost")
        with self._credit_lock:
            if self._queued_charges:
                return None
            command = RevokeFinalizeV1(
                protocol_version=1,
                erasure_request_id=ticket.erasure_request_id,
                consent_epoch_id=authority.consent_epoch_id,
                control_sequence=authority.control_sequence,
                control_fingerprint_hash=authority.control_fingerprint_hash,
                final_admission_ordinal=self._next_admission_ordinal - 1,
            )
            ordinal = self._allocate_admission_ordinal_locked()
            item = EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=WriterQueueLane.REVOKE,
                payload=command,
                admission_ordinal=ordinal,
            )
        self._revoke_item = item
        try:
            self._writer_sink.put_nowait(item)
        except Exception:
            with self._credit_lock:
                self._complete_revoke_writer_fault_locked(ticket, item)
            return None
        return item

    def complete_revoke_finalize(
        self,
        item: EvidenceWriterQueueItemV1,
        disposition: RevokeDisposition,
    ) -> None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("revoke item has the wrong type")
        if type(item.payload) is not RevokeFinalizeV1:
            raise ReservationError("revoke finalization item has the wrong payload")
        if type(disposition) is not RevokeDisposition:
            raise TypeError("disposition must be an exact RevokeDisposition")
        if disposition not in (
            RevokeDisposition.PURGE_COMPLETED,
            RevokeDisposition.PURGE_FAILED,
        ):
            raise ValueError("revoke finalization requires a terminal disposition")
        with self._credit_lock:
            ticket = self._revoke_ticket
            if ticket is None or self._revoke_item is not item:
                raise ReservationError("revoke finalization item is stale")
            # The finalizer is only admitted after lease and ordered-control
            # drain.  A conflicting terminal transport result is fault-completed
            # rather than clearing its item and raising after authority is lost.
            if self._leases or any(
                state.queued_records for state in self._session_controls.values()
            ):
                self._complete_revoke_writer_fault_locked(ticket, item)
                return
            self._release_epoch_session_controls_locked()
            self._revoke_item = None
            object.__setattr__(ticket, "disposition", disposition)
            self._pending_revoke = False
            self._phase = RuntimeSessionPhase.STOPPED
            self._owner_state = OwnerState.STOPPED
            if disposition is RevokeDisposition.PURGE_COMPLETED:
                self._purge_required = False
                self._capture_state = CaptureState.IDLE
            else:
                self._purge_required = True
                self._capture_state = CaptureState.PURGE_FAILED
            _set_ticket_signal(ticket.durability_event)
            _set_ticket_signal(ticket.terminal_event)
            self._observe_capacity(kind="revoke_terminal")

    def request_drain(self, authority: LifecycleDrainAuthorityV1) -> DrainTicketV1:
        if type(authority) is not LifecycleDrainAuthorityV1:
            raise ReservationError("drain authority has the wrong type")
        authority._validate()
        with self._admission_lock:
            if self._drain_ticket is not None:
                if self._drain_authority is not authority:
                    raise ReservationError("drain is owner-only and already queued")
                if (
                    self._drain_ticket.disposition
                    in (
                        DrainDisposition.DRAIN_QUEUED,
                        DrainDisposition.STOPPED,
                        DrainDisposition.WRITER_FAULT,
                    )
                    or not self._drain_ticket.terminal_event.is_set()
                ):
                    self._observe_capacity(kind="drain_coalesced")
                    return self._drain_ticket
                # A completed non-terminal drain was refused or timed out.  Its
                # exact owner capability may retry without minting a new watermark.
                self._drain_ticket = None
                self._drain_item = None
            with self._credit_lock:
                if authority.owner_generation != self._owner_generation:
                    raise ReservationError("drain authority is stale or wrong-owner")
                if self._drain_authority is None and (
                    authority.final_admission_ordinal != self._next_admission_ordinal - 1
                ):
                    raise ReservationError("drain authority is stale or wrong-owner")
                if self._drain_authority is not None and self._drain_authority is not authority:
                    raise ReservationError("drain is owner-only and already queued")
                command = DrainAndStopV1(
                    protocol_version=1,
                    owner_generation=authority.owner_generation,
                    final_admission_ordinal=authority.final_admission_ordinal,
                )
                ordinal = self._allocate_admission_ordinal_locked()
                item = EvidenceWriterQueueItemV1(
                    protocol_version=1,
                    lane=WriterQueueLane.DRAIN,
                    payload=command,
                    admission_ordinal=ordinal,
                )
            ticket = DrainTicketV1(
                owner_generation=authority.owner_generation,
                disposition=DrainDisposition.DRAIN_QUEUED,
                terminal_event=Event(),
            )
            if self._drain_authority is None:
                self._drain_authority = authority
            self._drain_ticket = ticket
            self._drain_item = item
            self._owner_state = OwnerState.DRAINING
            self._observe_capacity(kind="drain_accepted")
            if self._phase is not RuntimeSessionPhase.REVOKING:
                self._phase = RuntimeSessionPhase.CLOSING
            try:
                self._writer_sink.put_nowait(item)
            except Exception:
                self._drain_item = None
                object.__setattr__(ticket, "disposition", DrainDisposition.WRITER_FAULT)
                _set_ticket_signal(ticket.terminal_event)
                with self._credit_lock:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return ticket

    def complete_drain(
        self,
        item: EvidenceWriterQueueItemV1,
        disposition: DrainDisposition,
    ) -> None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("drain item is stale")
        if type(disposition) is not DrainDisposition:
            raise TypeError("disposition must be an exact DrainDisposition")
        with self._admission_lock, self._credit_lock:
            if self._drain_item is not item:
                raise ReservationError("drain item is stale")
            ticket = self._drain_ticket
            if ticket is None:
                raise ReservationError("drain ticket was lost")
            revoke_incomplete = (
                self._revoke_ticket is not None and not self._revoke_ticket.terminal_event.is_set()
            )
            if self._queued_charges or self._leases or revoke_incomplete:
                raise ReservationError("drain prerequisites are not complete")
            self._drain_item = None
            object.__setattr__(ticket, "disposition", disposition)
            if disposition is DrainDisposition.STOPPED:
                self._release_epoch_session_controls_locked()
                self._owner_state = OwnerState.STOPPED
                self._phase = RuntimeSessionPhase.STOPPED
            _set_ticket_signal(ticket.terminal_event)
            self._observe_capacity(kind="drain_terminal")

    def try_close_binding(
        self,
        authority: BindingCloseAuthorityV1,
        snapshot: EvidenceSnapshotV1,
    ) -> AppendDisposition:
        try:
            with self._admission_lock:
                if type(authority) is not BindingCloseAuthorityV1:
                    return AppendDisposition.INVALID_AUTHORITY
                authority._validate()
                exact_snapshot = _copy_snapshot(snapshot)
                payload = exact_snapshot.payload
                if (
                    not self._authority_lineage_matches_locked(authority)
                    or exact_snapshot.event_kind is not EventKind.BINDING_CLOSED
                    or type(payload) is not BindingClosedPayloadV1
                    or authority in self._used_authorities
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if (
                    payload.binding_id != authority.binding_id
                    or payload.close_reason is not authority.close_reason
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if self._sticky_fault is not None:
                    return AppendDisposition.WRITER_FAULT
                if self._session_tainted:
                    return AppendDisposition.SESSION_TAINTED
                if self._phase in (
                    RuntimeSessionPhase.REVOKING,
                    RuntimeSessionPhase.SEAL_QUEUED,
                    RuntimeSessionPhase.STOPPED,
                ):
                    return AppendDisposition.SESSION_CLOSING
                self._phase = RuntimeSessionPhase.CLOSING
                if self._leases:
                    return AppendDisposition.SESSION_CLOSING
                predicted_ordinal = self._next_admission_ordinal
                command = BindingCloseV1(
                    protocol_version=1,
                    binding_id=authority.binding_id,
                    consent_epoch_id=authority.consent_epoch_id,
                    logical_session_id=authority.logical_session_id,
                    admission_ordinal=predicted_ordinal,
                    snapshot=exact_snapshot,
                )
                validate_authority_composition(authority, command)
                disposition = self._try_enqueue_session_control_locked(
                    command,
                    event_sequence=exact_snapshot.event_sequence,
                    category="binding_close",
                )
                if disposition is AppendDisposition.ADMITTED:
                    self._used_authorities.add(authority)
                    self._binding_close_admitted = True
                return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def settled_terminal_outcome(
        self,
        lease: EvidenceTurnLease,
    ) -> SettledTerminalOutcomeV1 | None:
        """Return an outcome only after its terminal batch was admitted.

        This read-only seam deliberately exposes no turn identity, content, or
        evidence snapshots.  Frozen cause resolution alone is insufficient: a
        failed queue admission returns no outcome and must not be observed.
        """

        if type(lease) is not EvidenceTurnLease:
            raise TypeError("lease must be an exact EvidenceTurnLease")
        state = self._cause_states.get(lease.terminal_cause)
        if state is None or state.lease is not lease:
            return None
        with state.lock:
            return state.settled_outcome

    def invalidate_binding(
        self,
        authority: BindingCloseAuthorityV1,
    ) -> AppendDisposition:
        """Retire a binding before replacement media can receive evidence authority.

        This is intentionally stricter than ordinary close: active leases are tainted
        and retired rather than permitted to survive into a new media incarnation.
        The next browser binding must obtain fresh consent.
        """

        try:
            with self._admission_lock:
                if type(authority) is not BindingCloseAuthorityV1:
                    return AppendDisposition.INVALID_AUTHORITY
                authority._validate()
                if (
                    not self._authority_lineage_matches_locked(authority)
                    or authority in self._used_authorities
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if self._sticky_fault is not None:
                    return AppendDisposition.WRITER_FAULT
                if self._session_tainted:
                    return AppendDisposition.SESSION_TAINTED
                if self._phase in (
                    RuntimeSessionPhase.REVOKING,
                    RuntimeSessionPhase.SEAL_QUEUED,
                    RuntimeSessionPhase.STOPPED,
                ):
                    return AppendDisposition.SESSION_CLOSING
                self._phase = RuntimeSessionPhase.CLOSING
                for lease, state in tuple(self._leases.items()):
                    self._cause_states.pop(lease.terminal_cause, None)
                    self._leases.pop(lease, None)
                    with self._credit_lock:
                        self._release_unused_terminal_credits_locked(state.terminal_reservation)
                self._taint_locked(TaintCode.TERMINAL_CONFLICT)
                return AppendDisposition.SESSION_TAINTED
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def request_seal(
        self,
        authority: LifecycleSealAuthorityV1,
        command: SealEpochV1,
    ) -> SealDisposition:
        try:
            with self._admission_lock:
                if (
                    type(authority) is not LifecycleSealAuthorityV1
                    or type(command) is not SealEpochV1
                ):
                    raise ReservationError("seal requires exact owner authority and DTO")
                authority._validate()
                validate_authority_composition(authority, command)
                if self._seal_ticket_disposition is not None:
                    return SealDisposition.ALREADY_QUEUED
                if self._phase is RuntimeSessionPhase.REVOKING:
                    return SealDisposition.REVOKED
                if self._session_tainted:
                    return SealDisposition.SESSION_TAINTED
                if self._sticky_fault is not None:
                    return SealDisposition.WRITER_FAULT
                if self._leases:
                    return SealDisposition.ACTIVE_LEASES
                if (
                    self._phase is not RuntimeSessionPhase.CLOSING
                    or not self._binding_close_admitted
                    or not self._authority_lineage_matches_locked(authority)
                ):
                    return SealDisposition.WRITER_FAULT
                disposition = self._try_enqueue_session_control_locked(
                    command,
                    event_sequence=command.final_event_sequence,
                    category="seal",
                )
                if disposition is AppendDisposition.ADMITTED:
                    self._phase = RuntimeSessionPhase.SEAL_QUEUED
                    self._seal_ticket_disposition = SealDisposition.SEAL_QUEUED
                    return SealDisposition.SEAL_QUEUED
                if disposition is AppendDisposition.SESSION_TAINTED:
                    return SealDisposition.SESSION_TAINTED
                return SealDisposition.WRITER_FAULT
        except (KeyboardInterrupt, SystemExit):
            raise
        except ReservationError:
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return SealDisposition.WRITER_FAULT

    def close_for_retention_expiry(self) -> None:
        """Close new evidence admission before owned terminal cancellation settles."""

        with self._admission_lock:
            if self._phase is RuntimeSessionPhase.OPEN:
                self._phase = RuntimeSessionPhase.EXPIRING
                return
            if self._phase is RuntimeSessionPhase.EXPIRING:
                return
            raise ReservationError("retention expiry cannot close the current phase")

    def begin_expiry(self, authority: SessionExpiryAuthorityV1) -> ExpiryDisposition:
        try:
            with self._admission_lock:
                if type(authority) is not SessionExpiryAuthorityV1:
                    return ExpiryDisposition.WRITER_FAULT
                authority._validate()
                if (
                    authority.owner_generation != self._owner_generation
                    or authority.consent_epoch_id != self._consent_epoch_id
                    or authority.logical_session_id != self._logical_session_id
                ):
                    return ExpiryDisposition.WRITER_FAULT
                if self._phase is RuntimeSessionPhase.EXPIRING:
                    if self._expiry_authority is not None:
                        if self._expiry_authority is authority:
                            return ExpiryDisposition.ALREADY_EXPIRING
                        return ExpiryDisposition.WRITER_FAULT
                elif self._phase is not RuntimeSessionPhase.OPEN:
                    return ExpiryDisposition.WRITER_FAULT
                if authority.deadline_admission_ordinal != self._next_admission_ordinal - 1:
                    return ExpiryDisposition.WRITER_FAULT
                self._phase = RuntimeSessionPhase.EXPIRING
                if authority.mode is ExpiryMode.ROLLOVER:
                    self._expiry_authority = authority
                    return ExpiryDisposition.ROLLOVER_QUEUED
                command = ExpireSessionV1(
                    protocol_version=1,
                    owner_generation=authority.owner_generation,
                    logical_session_id=authority.logical_session_id,
                    consent_epoch_id=authority.consent_epoch_id,
                    expires_at_utc=authority.expires_at_utc,
                    last_admission_ordinal=authority.deadline_admission_ordinal,
                    erasure_request_id=str(uuid4()),
                    mode=authority.mode,
                )
                if not self._try_enqueue_zero_record_control_locked(command, "expiry"):
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    return ExpiryDisposition.WRITER_FAULT
                self._expiry_authority = authority
                return ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return ExpiryDisposition.WRITER_FAULT

    def try_rollover(
        self,
        authority: RolloverAuthorityV1,
        command: RolloverSessionV1,
    ) -> RolloverDisposition:
        try:
            with self._admission_lock:
                if (
                    type(authority) is not RolloverAuthorityV1
                    or type(command) is not RolloverSessionV1
                ):
                    return RolloverDisposition.INVALID_AUTHORITY
                authority._validate()
                validate_authority_composition(authority, command)
                if (
                    authority.owner_generation != self._owner_generation
                    or authority.binding_id != self._binding_id
                    or authority.binding_generation != self._binding_generation
                    or authority.consent_epoch_id != self._consent_epoch_id
                    or authority.predecessor_logical_session_id != self._logical_session_id
                ):
                    return RolloverDisposition.INVALID_AUTHORITY
                if self._leases:
                    return RolloverDisposition.ACTIVE_LEASES
                if self._replayable_turn_count:
                    return RolloverDisposition.REPLAYABLE_TURNS
                if self._session_tainted:
                    return RolloverDisposition.SESSION_TAINTED
                if self._sticky_fault is not None:
                    return RolloverDisposition.WRITER_FAULT
                if self._phase not in (
                    RuntimeSessionPhase.OPEN,
                    RuntimeSessionPhase.EXPIRING,
                ):
                    return RolloverDisposition.INVALID_AUTHORITY
                current = self._current_session_control
                if current is None:
                    return RolloverDisposition.INVALID_AUTHORITY
                with self._credit_lock:
                    control_state = self._session_controls.get(current)
                    if (
                        control_state is None
                        or control_state.remaining_records != SESSION_CONTROL_RECORD_CREDITS
                        or control_state.remaining_bytes != SESSION_CONTROL_CANONICAL_BYTE_CREDITS
                        or control_state.queued_records
                    ):
                        return RolloverDisposition.INVALID_AUTHORITY
                    if command.admission_ordinal != self._next_admission_ordinal:
                        return RolloverDisposition.INVALID_AUTHORITY
                    if not self._try_charge_locked(
                        ROLLOVER_RECORD_CREDITS,
                        ROLLOVER_CANONICAL_BYTE_CREDITS,
                        physical_items=1,
                    ):
                        self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                        return RolloverDisposition.INSUFFICIENT_CAPACITY
                    ordinal = self._allocate_admission_ordinal_locked()
                    item = EvidenceWriterQueueItemV1(
                        protocol_version=1,
                        lane=WriterQueueLane.ORDERED,
                        payload=command,
                        admission_ordinal=ordinal,
                    )
                    self._queued_charges[item] = _QueuedCharge(
                        records_on_completion=ROLLOVER_RECORD_CREDITS,
                        bytes_on_completion=ROLLOVER_CANONICAL_BYTE_CREDITS,
                        release_physical_item=True,
                        category="rollover",
                        reservation=current,
                    )
                try:
                    self._writer_sink.put_nowait(item)
                except Exception:
                    with self._credit_lock:
                        self._queued_charges.pop(item, None)
                        self._release_charge_locked(
                            ROLLOVER_RECORD_CREDITS,
                            ROLLOVER_CANONICAL_BYTE_CREDITS,
                            physical_items=1,
                        )
                        self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    return RolloverDisposition.WRITER_FAULT
                self._phase = RuntimeSessionPhase.EXPIRING
                return RolloverDisposition.ROLLOVER_QUEUED
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return RolloverDisposition.WRITER_FAULT

    def _try_claim_capacity_rollover_locked(
        self,
    ) -> _AdmissionRolloverPreparationV1 | None:
        """Linearize the capacity predicate, phase transition, and one charge."""

        if (
            not self._enabled
            or self._phase is not RuntimeSessionPhase.OPEN
            or self._session_tainted
            or self._sticky_fault is not None
            or self._leases
            or self._replayable_turn_count
            or self._rollover_preparation is not None
            or (
                self._session_reserved_events + TURN_SESSION_EVENT_RESERVATION <= MAX_SESSION_EVENTS
                and self._session_reserved_bytes + TURN_SESSION_CANONICAL_BYTE_RESERVATION
                <= MAX_SESSION_CANONICAL_BYTES
            )
        ):
            return None
        current = self._current_session_control
        if current is None:
            return None
        with self._credit_lock:
            state = self._session_controls.get(current)
            if (
                state is None
                or state.remaining_records != SESSION_CONTROL_RECORD_CREDITS
                or state.remaining_bytes != SESSION_CONTROL_CANONICAL_BYTE_CREDITS
                or state.queued_records
                or not self._try_charge_locked(
                    ROLLOVER_RECORD_CREDITS,
                    ROLLOVER_CANONICAL_BYTE_CREDITS,
                    physical_items=1,
                )
            ):
                return None
            preparation = object.__new__(_AdmissionRolloverPreparationV1)
            preparation.reservation = current
            preparation.admission_ordinal = self._next_admission_ordinal
            preparation.item = None
            preparation.terminal = False
            preparation.successor = None
            preparation.ticket = _RolloverCompletionTicketV1()
            assert self._next_event_sequence is not None
            preparation.predecessor_final_event_sequence = self._next_event_sequence + 1
            self._phase = RuntimeSessionPhase.EXPIRING
            self._rollover_preparation = preparation
            self._rollover_ticket = preparation.ticket
            return preparation

    def prepared_rollover_coordinates(
        self, preparation: _AdmissionRolloverPreparationV1
    ) -> tuple[int, int]:
        """Return owner-fixed composition coordinates to the installed coordinator."""

        with self._admission_lock:
            if self._rollover_preparation is not preparation or preparation.terminal:
                raise ReservationError("admission rollover preparation is stale")
            return (
                preparation.admission_ordinal,
                preparation.predecessor_final_event_sequence,
            )

    def enqueue_prepared_rollover(
        self,
        preparation: _AdmissionRolloverPreparationV1,
        authority: RolloverAuthorityV1,
        command: RolloverSessionV1,
    ) -> RolloverDisposition:
        with self._admission_lock:
            if (
                type(preparation) is not _AdmissionRolloverPreparationV1
                or self._rollover_preparation is not preparation
                or preparation.terminal
                or preparation.item is not None
            ):
                return RolloverDisposition.INVALID_AUTHORITY
            try:
                authority._validate()
                validate_authority_composition(authority, command)
            except Exception:
                self._reject_prepared_rollover_locked(
                    preparation,
                    StoreDisposition.REJECTED_STATE,
                )
                return RolloverDisposition.INVALID_AUTHORITY
            if (
                command.admission_ordinal != preparation.admission_ordinal
                or authority.predecessor_logical_session_id != self._logical_session_id
            ):
                self._reject_prepared_rollover_locked(
                    preparation,
                    StoreDisposition.REJECTED_STATE,
                )
                return RolloverDisposition.INVALID_AUTHORITY
            with self._credit_lock:
                if self._next_admission_ordinal != preparation.admission_ordinal:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    ordinal = None
                else:
                    ordinal = self._allocate_admission_ordinal_locked()
                    item = EvidenceWriterQueueItemV1(
                        protocol_version=1,
                        lane=WriterQueueLane.ORDERED,
                        payload=command,
                        admission_ordinal=ordinal,
                    )
                    self._queued_charges[item] = _QueuedCharge(
                        records_on_completion=ROLLOVER_RECORD_CREDITS,
                        bytes_on_completion=ROLLOVER_CANONICAL_BYTE_CREDITS,
                        release_physical_item=True,
                        category="rollover_prepared",
                        reservation=preparation.reservation,
                    )
                    preparation.item = item
            if ordinal is None:
                self._reject_prepared_rollover_locked(
                    preparation,
                    StoreDisposition.FAULTED,
                )
                return RolloverDisposition.WRITER_FAULT
            try:
                probe = self._qualification_capacity_probe
                if (
                    probe is not None
                    and probe.consume_rollover_publication_fault()
                ):
                    raise RuntimeError("evidence writer queue publication failed")
                self._writer_sink.put_nowait(item)
            except Exception:
                with self._credit_lock:
                    self._queued_charges.pop(item, None)
                self._reject_prepared_rollover_locked(
                    preparation,
                    StoreDisposition.FAULTED,
                )
                return RolloverDisposition.WRITER_FAULT
            return RolloverDisposition.ROLLOVER_QUEUED

    def _reject_prepared_rollover_locked(
        self,
        preparation: _AdmissionRolloverPreparationV1,
        disposition: StoreDisposition,
    ) -> None:
        """Settle a post-lifecycle admission rejection without signaling its ticket."""

        with self._credit_lock:
            self._release_charge_locked(
                ROLLOVER_RECORD_CREDITS,
                ROLLOVER_CANONICAL_BYTE_CREDITS,
                physical_items=1,
            )
            if self._next_admission_ordinal == preparation.admission_ordinal + 1:
                self._next_admission_ordinal = preparation.admission_ordinal
        preparation.item = None
        preparation.terminal = True
        preparation.ticket.result = disposition
        self._rollover_preparation = None
        if self._phase is RuntimeSessionPhase.EXPIRING:
            self._phase = RuntimeSessionPhase.OPEN

    def abort_prepared_rollover(self, preparation: _AdmissionRolloverPreparationV1) -> None:
        with self._admission_lock:
            self._abort_prepared_rollover_locked(preparation)

    def _abort_prepared_rollover_locked(
        self, preparation: _AdmissionRolloverPreparationV1
    ) -> None:
        if self._rollover_preparation is not preparation or preparation.item is not None:
            raise ReservationError("admission rollover preparation is stale or queued")
        with self._credit_lock:
            self._release_charge_locked(
                ROLLOVER_RECORD_CREDITS,
                ROLLOVER_CANONICAL_BYTE_CREDITS,
                physical_items=1,
            )
        preparation.terminal = True
        preparation.ticket.result = StoreDisposition.REJECTED_STATE
        _set_ticket_signal(preparation.ticket.event)
        self._rollover_preparation = None
        if self._phase is RuntimeSessionPhase.EXPIRING:
            self._phase = RuntimeSessionPhase.OPEN

    def _dispatch_claimed_rollover(
        self,
        claimed: _AdmissionRolloverPreparationV1 | None,
        callback: Callable[[_AdmissionRolloverPreparationV1], None] | None,
    ) -> None:
        """Invoke runtime-owned rollover work only after admission is unlocked."""

        if claimed is None:
            return
        if callback is None:
            self.abort_prepared_rollover(claimed)
            return
        try:
            callback(claimed)
        except (KeyboardInterrupt, SystemExit):
            self.abort_prepared_rollover(claimed)
            raise
        except Exception:
            self.abort_prepared_rollover(claimed)

    def prepare_rollover_completion(
        self,
        preparation: _AdmissionRolloverPreparationV1,
        item: EvidenceWriterQueueItemV1,
        disposition: StoreDisposition,
    ) -> bool:
        """Settle the writer charge, retaining EXPIRING until lifecycle commits."""

        if type(disposition) is not StoreDisposition:
            raise TypeError("disposition must be an exact StoreDisposition")
        with self._admission_lock, self._credit_lock:
            if preparation.ticket.result is not None:
                return preparation.ticket.result in (
                    StoreDisposition.COMMITTED,
                    StoreDisposition.IDEMPOTENT,
                )
            if self._rollover_preparation is not preparation or preparation.item is not item:
                raise ReservationError("rollover completion is stale")
            charge = self._queued_charges.pop(item, None)
            if charge is None or charge.category != "rollover_prepared":
                raise ReservationError("rollover charge is stale")
            self._release_charge_locked(
                ROLLOVER_RECORD_CREDITS,
                ROLLOVER_CANONICAL_BYTE_CREDITS,
                physical_items=1,
            )
            preparation.ticket.result = disposition
            if disposition not in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT):
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                preparation.terminal = True
                self._rollover_preparation = None
                return False
            old_state = self._session_controls.get(preparation.reservation)
            payload = item.payload
            if (
                old_state is None
                or old_state.queued_records
                or type(payload) is not RolloverSessionV1
            ):
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                preparation.terminal = True
                self._rollover_preparation = None
                return False
            preparation.successor = self._mint_session_control_locked(
                consent_epoch_id=payload.consent_epoch_id,
                logical_session_id=payload.successor_logical_session_id,
                cleanup_only=False,
            )
            return True

    def finalize_failed_rollover_completion(
        self,
        preparation: _AdmissionRolloverPreparationV1,
        disposition: StoreDisposition,
    ) -> None:
        """Signal a failed rollover only after runtime aborts lifecycle preparation."""

        if type(preparation) is not _AdmissionRolloverPreparationV1:
            raise TypeError("preparation must be an exact admission rollover capability")
        if type(disposition) is not StoreDisposition:
            raise TypeError("disposition must be an exact StoreDisposition")
        if disposition in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT):
            raise ReservationError("durable rollover completion cannot be finalized as failed")
        with self._admission_lock:
            if (
                self._rollover_ticket is not preparation.ticket
                or preparation.ticket.result is not disposition
                or not preparation.terminal
                or preparation.successor is not None
                or self._rollover_preparation is not None
            ):
                raise ReservationError("failed rollover completion is stale")
            _set_ticket_signal(preparation.ticket.event)

    def publish_prepared_rollover(self, preparation: _AdmissionRolloverPreparationV1) -> None:
        """No-fail second-owner publication after durable lifecycle commit."""

        with self._admission_lock, self._credit_lock:
            if self._rollover_preparation is not preparation or preparation.successor is None:
                raise ReservationError("rollover publication is stale")
            item = preparation.item
            assert item is not None and type(item.payload) is RolloverSessionV1
            command = item.payload
            del self._session_controls[preparation.reservation]
            self._current_session_control = preparation.successor
            successor_open = command.snapshots[2]
            self._installation_id = successor_open.installation_id
            self._producer_instance_id = successor_open.producer_instance_id
            self._logical_session_id = command.successor_logical_session_id
            self._next_event_sequence = 3
            self._phase = RuntimeSessionPhase.OPEN
            self._binding_close_admitted = False
            self._seal_ticket_disposition = None
            self._expiry_authority = None
            self._session_reserved_events = 2
            self._session_reserved_bytes = 0
            preparation.terminal = True
            self._rollover_preparation = None
            _set_ticket_signal(preparation.ticket.event)

    def complete_rollover(
        self,
        item: EvidenceWriterQueueItemV1,
        disposition: StoreDisposition,
    ) -> SessionControlReservation | None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("rollover item has the wrong type")
        if type(disposition) is not StoreDisposition:
            raise TypeError("disposition must be an exact StoreDisposition")
        with self._credit_lock:
            charge = self._queued_charges.get(item)
            if charge is None or charge.category != "rollover":
                raise ReservationError("rollover item is stale")
            old = charge.reservation
            if type(old) is not SessionControlReservation:
                raise ReservationError("rollover lost predecessor control credits")
            old_state = self._session_controls.get(old)
            if old_state is None:
                raise ReservationError("rollover predecessor reservation is stale")
            del self._queued_charges[item]
            self._release_charge_locked(
                ROLLOVER_RECORD_CREDITS,
                ROLLOVER_CANONICAL_BYTE_CREDITS,
                physical_items=1,
            )
            if disposition not in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT):
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                return None
            payload = item.payload
            if type(payload) is not RolloverSessionV1:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                raise ReservationError("rollover item payload changed kind")
            command = payload
            if old_state.queued_records:
                raise ReservationError("rollover predecessor control credits are in use")
            del self._session_controls[old]
            successor = self._mint_session_control_locked(
                consent_epoch_id=command.consent_epoch_id,
                logical_session_id=command.successor_logical_session_id,
                cleanup_only=False,
            )
            self._current_session_control = successor
            successor_open = command.snapshots[2]
            self._installation_id = successor_open.installation_id
            self._producer_instance_id = successor_open.producer_instance_id
            self._logical_session_id = command.successor_logical_session_id
            self._next_event_sequence = 3
            self._phase = RuntimeSessionPhase.OPEN
            self._binding_close_admitted = False
            self._seal_ticket_disposition = None
            self._expiry_authority = None
            self._session_reserved_events = 2
            self._session_reserved_bytes = 0
            return successor

    def try_admit_command(
        self,
        authority: CommandAdmissionAuthorityV1,
        snapshot: EvidenceSnapshotV1,
    ) -> CommandDisposition:
        """Consume one accepted-command bearer and enqueue no command content."""

        try:
            with self._admission_lock:
                preliminary = self._preliminary_append_disposition_locked()
                if preliminary is not None:
                    return _command_disposition(preliminary)
                if type(authority) is not CommandAdmissionAuthorityV1:
                    return CommandDisposition.INVALID_AUTHORITY
                authority._validate()
                if not self._conversation_authority_current_locked(authority):
                    return CommandDisposition.INVALID_AUTHORITY
                exact_snapshot = _copy_snapshot(snapshot)
                payload = exact_snapshot.payload
                if (
                    authority in self._used_authorities
                    or not self._authority_lineage_matches_locked(authority)
                    or exact_snapshot.event_kind is not EventKind.COMMAND_ROUTED
                    or exact_snapshot.logical_session_id != authority.logical_session_id
                    or type(payload) is not CommandRoutedPayloadV1
                ):
                    return CommandDisposition.INVALID_AUTHORITY
                if (
                    payload.utterance_id != authority.utterance_id
                    or payload.source is not authority.source
                ):
                    return CommandDisposition.INVALID_AUTHORITY
                disposition = self._try_enqueue_ordinary_locked(exact_snapshot)
                if disposition is AppendDisposition.ADMITTED:
                    self._used_authorities.add(authority)
                return _command_disposition(disposition)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return CommandDisposition.WRITER_FAULT

    def try_reserve_user_turn(
        self,
        authority: UserTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1:
        return self._try_reserve_turn(
            authority,
            operation,
            authority_type=UserTurnAuthorityV1,
            operation_kind=ConversationOperationKind.RESPONSE,
            turn_kind=TurnKind.USER_RESPONSE,
        )

    def try_admit_user_final(
        self,
        lease: EvidenceTurnLease,
        authority: UserTurnAuthorityV1,
        text: str,
    ) -> AppendDisposition:
        """Atomically enqueue turn-open plus the exact routed final user text."""

        try:
            with self._admission_lock:
                preliminary = self._preliminary_append_disposition_locked()
                if preliminary is not None:
                    return preliminary
                if (
                    type(lease) is not EvidenceTurnLease
                    or type(authority) is not UserTurnAuthorityV1
                    or type(text) is not str
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or state.authority is not authority
                    or state.turn_kind is not TurnKind.USER_RESPONSE
                    or state.opened
                    or not self._lease_lineage_matches_locked(lease)
                    or _turn_authority_signature(authority) != state.authority_signature
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if len(text) > 4_096 or any(
                    0xD800 <= ord(character) <= 0xDFFF for character in text
                ):
                    self._taint_locked(TaintCode.OVERSIZE)
                    return AppendDisposition.REJECTED_OVERSIZE
                if self._next_event_sequence is None:
                    return AppendDisposition.CONSENT_MISSING
                installation_id = self._installation_id
                producer_instance_id = self._producer_instance_id
                if installation_id is None or producer_instance_id is None:
                    return AppendDisposition.CONSENT_MISSING
                opened = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=installation_id,
                    producer_instance_id=producer_instance_id,
                    logical_session_id=authority.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.TURN_OPENED,
                    payload=TurnOpenedPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        turn_kind=TurnKind.USER_RESPONSE,
                        utterance_id=authority.utterance_id,
                        replay_of_evidence_turn_id=None,
                    ),
                )
                accepted = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=installation_id,
                    producer_instance_id=producer_instance_id,
                    logical_session_id=authority.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence + 1,
                    event_kind=EventKind.USER_FINAL_ACCEPTED,
                    payload=UserFinalAcceptedPayloadV1(
                        utterance_id=authority.utterance_id,
                        evidence_turn_id=lease.evidence_turn_id,
                        source=authority.source,
                        routing_disposition="response",
                        text=text,
                    ),
                )
                return self._try_enqueue_user_final_batch_locked(
                    lease,
                    state,
                    opened,
                    accepted,
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def discard_unopened_user_turn(
        self,
        lease: EvidenceTurnLease,
        authority: UserTurnAuthorityV1,
    ) -> bool:
        """Release an exact user lease only before its opening batch is admitted."""

        claimed: _AdmissionRolloverPreparationV1 | None = None
        callback: Callable[[_AdmissionRolloverPreparationV1], None] | None = None
        with self._admission_lock:
            if type(lease) is not EvidenceTurnLease or type(authority) is not UserTurnAuthorityV1:
                return False
            state = self._leases.get(lease)
            if (
                state is None
                or state.authority is not authority
                or state.turn_kind is not TurnKind.USER_RESPONSE
                or state.opened
                or not self._lease_lineage_matches_locked(lease)
                or _turn_authority_signature(authority) != state.authority_signature
            ):
                return False
            self._discard_unopened_lease_locked(lease)
            claimed = self._try_claim_capacity_rollover_locked()
            callback = self._rollover_claimed
        self._dispatch_claimed_rollover(claimed, callback)
        self._advance_revoke_finalization_if_ready()
        return True

    def discard_unopened_replay_turn(
        self,
        lease: EvidenceTurnLease,
        authority: ReplayTurnAuthorityV1,
    ) -> bool:
        """Release one exact replay lease that never crossed speech admission."""

        claimed: _AdmissionRolloverPreparationV1 | None = None
        callback: Callable[[_AdmissionRolloverPreparationV1], None] | None = None
        with self._admission_lock:
            if type(lease) is not EvidenceTurnLease or type(authority) is not ReplayTurnAuthorityV1:
                return False
            state = self._leases.get(lease)
            if (
                state is None
                or state.authority is not authority
                or state.turn_kind is not TurnKind.REPLAY
                or state.opened
                or not self._lease_lineage_matches_locked(lease)
                or _turn_authority_signature(authority) != state.authority_signature
            ):
                return False
            self._discard_unopened_lease_locked(lease)
            claimed = self._try_claim_capacity_rollover_locked()
            callback = self._rollover_claimed
        self._dispatch_claimed_rollover(claimed, callback)
        self._advance_revoke_finalization_if_ready()
        return True

    def discard_unopened_proactive_turn(
        self,
        lease: EvidenceTurnLease,
        authority: ProactiveTurnAuthorityV1,
    ) -> bool:
        """Release one exact proactive lease that never crossed speech admission."""

        claimed: _AdmissionRolloverPreparationV1 | None = None
        callback: Callable[[_AdmissionRolloverPreparationV1], None] | None = None
        with self._admission_lock:
            if (
                type(lease) is not EvidenceTurnLease
                or type(authority) is not ProactiveTurnAuthorityV1
            ):
                return False
            state = self._leases.get(lease)
            if (
                state is None
                or state.authority is not authority
                or state.turn_kind is not TurnKind.PROACTIVE_UPDATE
                or state.opened
                or not self._lease_lineage_matches_locked(lease)
                or _turn_authority_signature(authority) != state.authority_signature
            ):
                return False
            self._discard_unopened_lease_locked(lease)
            claimed = self._try_claim_capacity_rollover_locked()
            callback = self._rollover_claimed
        self._dispatch_claimed_rollover(claimed, callback)
        self._advance_revoke_finalization_if_ready()
        return True

    def settle_spawn_failed(self, lease: EvidenceTurnLease) -> AppendDisposition:
        """Atomically retire an admitted turn whose response task could not spawn."""

        try:
            disposition: AppendDisposition
            with self._admission_lock:
                if type(lease) is not EvidenceTurnLease:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or not state.opened
                    or state.snapshot
                    or state.settlement_attempted
                    or not self._lease_lineage_matches_locked(lease)
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                cause_disposition = self.record_terminal_cause(
                    lease.terminal_cause,
                    TerminalReason.TASK_SPAWN_FAILED,
                )
                if cause_disposition not in (
                    CauseDisposition.RECORDED,
                    CauseDisposition.ALREADY_RECORDED,
                ):
                    self._taint_locked(TaintCode.TERMINAL_CONFLICT)
                    return AppendDisposition.SESSION_TAINTED
                resolution = self.freeze_and_resolve_terminal_causes(
                    lease,
                    context_committed=False,
                )
                if (
                    resolution.terminal_disposition is not TerminalDisposition.FAILED
                    or resolution.terminal_reason is not TerminalReason.TASK_SPAWN_FAILED
                ):
                    self._taint_locked(TaintCode.TERMINAL_CONFLICT)
                    return AppendDisposition.SESSION_TAINTED
                if self._next_event_sequence is None:
                    return AppendDisposition.CONSENT_MISSING
                installation_id = self._installation_id
                producer_instance_id = self._producer_instance_id
                if installation_id is None or producer_instance_id is None:
                    return AppendDisposition.CONSENT_MISSING
                snapshot = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=installation_id,
                    producer_instance_id=producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.TURN_SNAPSHOT,
                    payload=TurnSnapshotPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        turn_kind=state.turn_kind,
                        generated_segment_count=state.generated_segment_count,
                        queued_chunk_count=0,
                        started_chunk_count=0,
                        transport_confirmed_full_count=state.transport_confirmed_full_count,
                        model_context_admitted=False,
                        assistant_delivery_context_recorded=False,
                    ),
                )
                settled = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=installation_id,
                    producer_instance_id=producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence + 1,
                    event_kind=EventKind.TURN_SETTLED,
                    payload=TurnSettledPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        terminal_disposition=resolution.terminal_disposition,
                        terminal_reason=resolution.terminal_reason,
                        context_committed=False,
                        generated_segment_count=state.generated_segment_count,
                        transport_confirmed_full_count=state.transport_confirmed_full_count,
                    ),
                )
                disposition, _, _ = self._try_enqueue_terminal_batch_locked(
                    lease,
                    state,
                    snapshot,
                    settled,
                    claim_rollover=False,
                )
                self._taint_locked(TaintCode.SPAWN_FAILED)
            return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def settle_completed(
        self,
        lease: EvidenceTurnLease,
        *,
        queued_chunk_count: int,
        started_chunk_count: int,
        transport_confirmed_full_count: int,
        assistant_delivery_context_recorded: bool,
    ) -> AppendDisposition:
        """Settle completed conversation or retire its incomplete capture lease."""

        retired_tainted = False
        try:
            claimed: _AdmissionRolloverPreparationV1 | None = None
            callback: Callable[[_AdmissionRolloverPreparationV1], None] | None = None
            with self._admission_lock:
                if type(lease) is not EvidenceTurnLease:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or not state.opened
                    or state.snapshot
                    or state.settlement_attempted
                    or not self._lease_lineage_matches_locked(lease)
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if self._retire_tainted_turn_locked(lease, state):
                    retired_tainted = True
                    return AppendDisposition.SESSION_TAINTED
                if transport_confirmed_full_count != state.transport_confirmed_full_count:
                    return AppendDisposition.INVALID_AUTHORITY
                cause = self.record_terminal_cause(
                    lease.terminal_cause,
                    TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
                )
                if cause not in (
                    CauseDisposition.RECORDED,
                    CauseDisposition.ALREADY_RECORDED,
                ):
                    self._taint_locked(TaintCode.TERMINAL_CONFLICT)
                    return AppendDisposition.SESSION_TAINTED
                resolution = self.freeze_and_resolve_terminal_causes(
                    lease,
                    context_committed=True,
                )
                if (
                    resolution.terminal_disposition is not TerminalDisposition.COMPLETED
                    or resolution.terminal_reason
                    is not TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED
                ):
                    self._taint_locked(TaintCode.TERMINAL_CONFLICT)
                    return AppendDisposition.SESSION_TAINTED
                if (
                    self._next_event_sequence is None
                    or self._installation_id is None
                    or self._producer_instance_id is None
                ):
                    return AppendDisposition.CONSENT_MISSING
                snapshot = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=self._installation_id,
                    producer_instance_id=self._producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.TURN_SNAPSHOT,
                    payload=TurnSnapshotPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        turn_kind=state.turn_kind,
                        generated_segment_count=state.generated_segment_count,
                        queued_chunk_count=queued_chunk_count,
                        started_chunk_count=started_chunk_count,
                        transport_confirmed_full_count=transport_confirmed_full_count,
                        model_context_admitted=state.turn_kind is TurnKind.USER_RESPONSE,
                        assistant_delivery_context_recorded=(assistant_delivery_context_recorded),
                    ),
                )
                settled = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=self._installation_id,
                    producer_instance_id=self._producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence + 1,
                    event_kind=EventKind.TURN_SETTLED,
                    payload=TurnSettledPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        terminal_disposition=resolution.terminal_disposition,
                        terminal_reason=resolution.terminal_reason,
                        context_committed=True,
                        generated_segment_count=state.generated_segment_count,
                        transport_confirmed_full_count=(state.transport_confirmed_full_count),
                    ),
                )
                disposition, claimed, callback = self._try_enqueue_terminal_batch_locked(
                    lease,
                    state,
                    snapshot,
                    settled,
                )
            self._dispatch_claimed_rollover(claimed, callback)
            return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT
        finally:
            if retired_tainted:
                # A tainted retirement queues no terminal records whose writer
                # completion could otherwise advance pending revocation.
                self._advance_revoke_finalization_if_ready()

    def settle_terminal(
        self,
        lease: EvidenceTurnLease,
        *,
        queued_chunk_count: int,
        started_chunk_count: int,
        assistant_delivery_context_recorded: bool,
    ) -> AppendDisposition:
        """Settle recorded terminal causes or retire incomplete capture."""

        retired_tainted = False
        try:
            claimed: _AdmissionRolloverPreparationV1 | None = None
            callback: Callable[[_AdmissionRolloverPreparationV1], None] | None = None
            with self._admission_lock:
                if type(lease) is not EvidenceTurnLease:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or not state.opened
                    or state.snapshot
                    or state.settlement_attempted
                    or not self._lease_lineage_matches_locked(lease)
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if self._retire_tainted_turn_locked(lease, state):
                    retired_tainted = True
                    return AppendDisposition.SESSION_TAINTED
                context_committed = assistant_delivery_context_recorded
                resolution = self.freeze_and_resolve_terminal_causes(
                    lease,
                    context_committed=context_committed,
                )
                if (
                    self._next_event_sequence is None
                    or self._installation_id is None
                    or self._producer_instance_id is None
                ):
                    return AppendDisposition.CONSENT_MISSING
                snapshot = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=self._installation_id,
                    producer_instance_id=self._producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.TURN_SNAPSHOT,
                    payload=TurnSnapshotPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        turn_kind=state.turn_kind,
                        generated_segment_count=state.generated_segment_count,
                        queued_chunk_count=queued_chunk_count,
                        started_chunk_count=started_chunk_count,
                        transport_confirmed_full_count=(state.transport_confirmed_full_count),
                        model_context_admitted=context_committed,
                        assistant_delivery_context_recorded=(assistant_delivery_context_recorded),
                    ),
                )
                settled = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=self._installation_id,
                    producer_instance_id=self._producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence + 1,
                    event_kind=EventKind.TURN_SETTLED,
                    payload=TurnSettledPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        terminal_disposition=resolution.terminal_disposition,
                        terminal_reason=resolution.terminal_reason,
                        context_committed=context_committed,
                        generated_segment_count=state.generated_segment_count,
                        transport_confirmed_full_count=(state.transport_confirmed_full_count),
                    ),
                )
                disposition, claimed, callback = self._try_enqueue_terminal_batch_locked(
                    lease,
                    state,
                    snapshot,
                    settled,
                )
            self._dispatch_claimed_rollover(claimed, callback)
            return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT
        finally:
            if retired_tainted:
                self._advance_revoke_finalization_if_ready()

    def _retire_tainted_turn_locked(self, lease: EvidenceTurnLease, state: _LeaseState) -> bool:
        """Release an already-validated live lease without publishing incomplete evidence."""

        if not self._session_tainted:
            return False
        cause_state = self._cause_states.pop(lease.terminal_cause, None)
        if cause_state is not None:
            # Reports that already retained this state must also fail closed.
            with cause_state.lock:
                cause_state.frozen = True
        state.settlement_attempted = True
        self._leases.pop(lease, None)
        with self._credit_lock:
            self._release_unused_terminal_credits_locked(state.terminal_reservation)
        return True

    def try_admit_generated(
        self,
        lease: EvidenceTurnLease,
        text: str,
    ) -> AppendDisposition:
        """Capture one publication-authorized user-response segment."""

        try:
            with self._admission_lock:
                preliminary = self._preliminary_append_disposition_locked()
                if preliminary is not None:
                    return preliminary
                if type(lease) is not EvidenceTurnLease or type(text) is not str:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or state.turn_kind is not TurnKind.USER_RESPONSE
                    or not state.opened
                    or not state.user_final
                    or state.snapshot
                    or not self._lease_lineage_matches_locked(lease)
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if len(text) > 4_096 or any(
                    0xD800 <= ord(character) <= 0xDFFF for character in text
                ):
                    self._taint_locked(TaintCode.OVERSIZE)
                    return AppendDisposition.REJECTED_OVERSIZE
                if self._next_event_sequence is None:
                    return AppendDisposition.CONSENT_MISSING
                installation_id = self._installation_id
                producer_instance_id = self._producer_instance_id
                if installation_id is None or producer_instance_id is None:
                    return AppendDisposition.CONSENT_MISSING
                generated = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=installation_id,
                    producer_instance_id=producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.ASSISTANT_SEGMENT_GENERATED,
                    payload=AssistantSegmentGeneratedPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        evidence_segment_id=str(uuid4()),
                        segment_ordinal=state.generated_segment_count + 1,
                        text=text,
                    ),
                )
                if not self._turn_event_is_valid_locked(lease, state, generated):
                    return AppendDisposition.INVALID_AUTHORITY
                size = canonical_record_bytes(generated)
                if (
                    state.event_count + 1 > MAX_TURN_EVENTS
                    or state.canonical_bytes + size > MAX_TURN_CANONICAL_BYTES
                ):
                    self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                    return AppendDisposition.QUOTA_EXCEEDED
                disposition = self._try_enqueue_ordinary_locked(generated)
                if disposition is AppendDisposition.ADMITTED:
                    state.event_count += 1
                    state.canonical_bytes += size
                    self._apply_turn_event_locked(state, generated)
                elif disposition is AppendDisposition.DROPPED_CAPACITY:
                    self._taint_locked(TaintCode.ADMISSION_GAP)
                return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def try_open_non_user_turn(
        self,
        lease: EvidenceTurnLease,
    ) -> AppendDisposition:
        """Atomically open one proactive or replay lease before speech delivery."""

        try:
            with self._admission_lock:
                preliminary = self._preliminary_append_disposition_locked()
                if preliminary is not None:
                    return preliminary
                if type(lease) is not EvidenceTurnLease:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or state.turn_kind not in (TurnKind.PROACTIVE_UPDATE, TurnKind.REPLAY)
                    or state.opened
                    or state.snapshot
                    or not self._lease_lineage_matches_locked(lease)
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if (
                    self._next_event_sequence is None
                    or self._installation_id is None
                    or self._producer_instance_id is None
                ):
                    return AppendDisposition.CONSENT_MISSING
                authority = state.authority
                replay_of_evidence_turn_id = (
                    authority.replay_of_evidence_turn_id
                    if type(authority) is ReplayTurnAuthorityV1
                    else None
                )
                opened = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=self._installation_id,
                    producer_instance_id=self._producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.TURN_OPENED,
                    payload=TurnOpenedPayloadV1(
                        evidence_turn_id=lease.evidence_turn_id,
                        turn_kind=state.turn_kind,
                        utterance_id=None,
                        replay_of_evidence_turn_id=replay_of_evidence_turn_id,
                    ),
                )
                if not self._turn_event_is_valid_locked(lease, state, opened):
                    return AppendDisposition.INVALID_AUTHORITY
                disposition = self._try_enqueue_ordinary_locked(opened)
                if disposition is AppendDisposition.ADMITTED:
                    state.event_count += 1
                    state.canonical_bytes += canonical_record_bytes(opened)
                    self._apply_turn_event_locked(state, opened)
                return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def try_admit_transport_confirmed_full(
        self,
        lease: EvidenceTurnLease,
        *,
        segment_ordinal: int | None,
        synthesis_attempt_id: str,
        transport_attempt_id: str,
        text: str | None,
    ) -> AppendDisposition:
        """Capture one fully confirmed transport transition."""

        try:
            with self._admission_lock:
                preliminary = self._preliminary_append_disposition_locked()
                if preliminary is not None:
                    return preliminary
                if type(lease) is not EvidenceTurnLease:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if (
                    state is None
                    or not state.opened
                    or state.snapshot
                    or not self._lease_lineage_matches_locked(lease)
                ):
                    return AppendDisposition.INVALID_AUTHORITY
                if state.turn_kind is TurnKind.USER_RESPONSE:
                    if (
                        type(segment_ordinal) is not int
                        or not 1 <= segment_ordinal <= len(state.evidence_segment_ids)
                        or type(text) is not str
                    ):
                        return AppendDisposition.INVALID_AUTHORITY
                    evidence_segment_id = state.evidence_segment_ids[segment_ordinal - 1]
                else:
                    if segment_ordinal is not None or text is not None:
                        return AppendDisposition.INVALID_AUTHORITY
                    evidence_segment_id = None
                payload = AssistantChunkTransportConfirmedFullPayloadV1(
                    evidence_turn_id=lease.evidence_turn_id,
                    evidence_segment_id=evidence_segment_id,
                    synthesis_attempt_id=synthesis_attempt_id,
                    transport_attempt_id=transport_attempt_id,
                    evidence_chunk_id=str(uuid4()),
                    chunk_ordinal=state.transport_confirmed_full_count + 1,
                    text=text,
                )
                if (
                    self._next_event_sequence is None
                    or self._installation_id is None
                    or self._producer_instance_id is None
                ):
                    return AppendDisposition.CONSENT_MISSING
                snapshot = EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=self._installation_id,
                    producer_instance_id=self._producer_instance_id,
                    logical_session_id=lease.logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=self._next_event_sequence,
                    event_kind=EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
                    payload=payload,
                )
                if _raw_snapshot_text_is_oversize(snapshot):
                    self._taint_locked(TaintCode.OVERSIZE)
                    return AppendDisposition.REJECTED_OVERSIZE
                if not self._turn_event_is_valid_locked(lease, state, snapshot):
                    return AppendDisposition.INVALID_AUTHORITY
                size = canonical_record_bytes(snapshot)
                if (
                    state.event_count + 1 > MAX_TURN_EVENTS
                    or state.canonical_bytes + size > MAX_TURN_CANONICAL_BYTES
                ):
                    self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                    return AppendDisposition.QUOTA_EXCEEDED
                disposition = self._try_enqueue_ordinary_locked(snapshot)
                if disposition is AppendDisposition.ADMITTED:
                    state.event_count += 1
                    state.canonical_bytes += size
                    self._apply_turn_event_locked(state, snapshot)
                elif disposition is AppendDisposition.DROPPED_CAPACITY:
                    self._taint_locked(TaintCode.ADMISSION_GAP)
                return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def try_reserve_proactive_turn(
        self,
        authority: ProactiveTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1:
        return self._try_reserve_turn(
            authority,
            operation,
            authority_type=ProactiveTurnAuthorityV1,
            operation_kind=ConversationOperationKind.PROACTIVE,
            turn_kind=TurnKind.PROACTIVE_UPDATE,
        )

    def try_reserve_replay_turn(
        self,
        authority: ReplayTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1:
        return self._try_reserve_turn(
            authority,
            operation,
            authority_type=ReplayTurnAuthorityV1,
            operation_kind=ConversationOperationKind.REPLAY,
            turn_kind=TurnKind.REPLAY,
        )

    def try_append_turn(
        self,
        lease: EvidenceTurnLease,
        snapshot: EvidenceSnapshotV1,
    ) -> AppendDisposition:
        """Append one exact turn transition without waiting for its writer."""

        try:
            with self._admission_lock:
                if not self._enabled:
                    return AppendDisposition.DISABLED
                if self._sticky_fault is not None:
                    return AppendDisposition.WRITER_FAULT
                if type(lease) is not EvidenceTurnLease:
                    return AppendDisposition.INVALID_AUTHORITY
                state = self._leases.get(lease)
                if state is None or not self._lease_lineage_matches_locked(lease):
                    return AppendDisposition.INVALID_AUTHORITY
                if _raw_snapshot_text_is_oversize(snapshot):
                    self._taint_locked(TaintCode.OVERSIZE)
                    return AppendDisposition.REJECTED_OVERSIZE
                exact_snapshot = _copy_snapshot(snapshot)
                if not self._turn_event_is_valid_locked(lease, state, exact_snapshot):
                    return AppendDisposition.INVALID_AUTHORITY
                if exact_snapshot.event_kind is EventKind.TURN_SETTLED:
                    terminal_taint = self._terminal_resolution_taint_locked(
                        lease,
                        exact_snapshot,
                    )
                    if terminal_taint is not None:
                        self._taint_locked(terminal_taint)
                        state.settlement_attempted = True
                        self._leases.pop(lease, None)
                        with self._credit_lock:
                            self._release_unused_terminal_credits_locked(state.terminal_reservation)
                        return AppendDisposition.SESSION_TAINTED
                try:
                    size = canonical_record_bytes(exact_snapshot)
                except EvidenceAdmissionError:
                    self._taint_locked(TaintCode.OVERSIZE)
                    return AppendDisposition.REJECTED_OVERSIZE
                terminal = exact_snapshot.event_kind in (
                    EventKind.TURN_SNAPSHOT,
                    EventKind.TURN_SETTLED,
                )
                if not terminal:
                    preliminary = self._preliminary_append_disposition_locked()
                    if preliminary is not None:
                        return preliminary
                elif self._phase in (
                    RuntimeSessionPhase.SEAL_QUEUED,
                    RuntimeSessionPhase.STOPPED,
                ):
                    return AppendDisposition.SESSION_CLOSING
                if not terminal and (
                    state.event_count + 1 > MAX_TURN_EVENTS
                    or state.canonical_bytes + size > MAX_TURN_CANONICAL_BYTES
                ):
                    self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                    return AppendDisposition.QUOTA_EXCEEDED
                if terminal:
                    disposition = self._try_enqueue_terminal_locked(
                        lease,
                        state,
                        exact_snapshot,
                        size,
                    )
                else:
                    disposition = self._try_enqueue_ordinary_locked(exact_snapshot)
                if disposition is AppendDisposition.ADMITTED:
                    state.event_count += 1
                    state.canonical_bytes += size
                    self._apply_turn_event_locked(state, exact_snapshot)
                if (
                    exact_snapshot.event_kind is EventKind.TURN_SETTLED
                    and disposition is not AppendDisposition.DROPPED_CAPACITY
                ):
                    state.settlement_attempted = True
                    self._leases.pop(lease, None)
                    with self._credit_lock:
                        self._release_unused_terminal_credits_locked(state.terminal_reservation)
                return disposition
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT

    def _try_reserve_turn(
        self,
        authority: UserTurnAuthorityV1 | ProactiveTurnAuthorityV1 | ReplayTurnAuthorityV1,
        operation: ConversationOperationReservation,
        *,
        authority_type: type[
            UserTurnAuthorityV1 | ProactiveTurnAuthorityV1 | ReplayTurnAuthorityV1
        ],
        operation_kind: ConversationOperationKind,
        turn_kind: TurnKind,
    ) -> EvidenceLeaseResultV1:
        try:
            with self._admission_lock:
                preliminary = self._preliminary_append_disposition_locked()
                if preliminary is not None:
                    return EvidenceLeaseResultV1(lease=None, disposition=preliminary)
                if type(authority) is not authority_type:
                    return _invalid_lease_result()
                authority._validate()
                if not self._conversation_authority_current_locked(authority):
                    return _invalid_lease_result()
                if (
                    authority in self._used_authorities
                    or not self._authority_lineage_matches_locked(authority)
                ):
                    return _invalid_lease_result()
                try:
                    self._operation_scheduler.validate(
                        operation,
                        operation_kind,
                        require_unconsumed=True,
                    )
                except ReservationError:
                    return _invalid_lease_result()
                self._used_authorities.add(authority)
                with self._credit_lock:
                    terminal = self._try_reserve_terminal_locked()
                    if terminal is None:
                        self._taint_locked(TaintCode.LEASE_CAPACITY_EXHAUSTED)
                        return EvidenceLeaseResultV1(
                            lease=None,
                            disposition=AppendDisposition.DROPPED_CAPACITY,
                        )
                    if (
                        self._session_reserved_events + TURN_SESSION_EVENT_RESERVATION
                        > MAX_SESSION_EVENTS
                        or self._session_reserved_bytes + TURN_SESSION_CANONICAL_BYTE_RESERVATION
                        > MAX_SESSION_CANONICAL_BYTES
                    ):
                        self._release_terminal_locked(terminal)
                        self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                        return EvidenceLeaseResultV1(
                            lease=None,
                            disposition=AppendDisposition.QUOTA_EXCEEDED,
                        )
                    evidence_turn_id = str(uuid4())
                    if evidence_turn_id in self._evidence_turn_ids:
                        self._release_terminal_locked(terminal)
                        self._taint_locked(TaintCode.LINEAGE_INVALID)
                        return EvidenceLeaseResultV1(
                            lease=None,
                            disposition=AppendDisposition.SESSION_TAINTED,
                        )
                    lease = self._mint_lease_locked(
                        authority=authority,
                        operation=operation,
                        terminal=terminal,
                        turn_kind=turn_kind,
                        evidence_turn_id=evidence_turn_id,
                    )
                try:
                    self._operation_scheduler.consume(operation, operation_kind)
                except ReservationError:
                    self._discard_unopened_lease_locked(lease)
                    return _invalid_lease_result()
                with self._credit_lock:
                    self._session_reserved_events += TURN_SESSION_EVENT_RESERVATION
                    self._session_reserved_bytes += TURN_SESSION_CANONICAL_BYTE_RESERVATION
                    self._evidence_turn_ids.add(evidence_turn_id)
                return EvidenceLeaseResultV1(
                    lease=lease,
                    disposition=AppendDisposition.ADMITTED,
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._admission_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return EvidenceLeaseResultV1(
                lease=None,
                disposition=AppendDisposition.WRITER_FAULT,
            )

    def record_terminal_cause(
        self,
        capability: TerminalCauseCapabilityV1,
        cause: TerminalReason,
    ) -> CauseDisposition:
        if type(capability) is not TerminalCauseCapabilityV1:
            return CauseDisposition.INVALID_AUTHORITY
        if type(cause) is not TerminalReason:
            return CauseDisposition.INVALID_AUTHORITY
        state = self._cause_states.get(capability)
        if state is None or state.lease.terminal_cause is not capability:
            return CauseDisposition.INVALID_AUTHORITY
        try:
            capability._validate()
        except Exception:
            return CauseDisposition.INVALID_AUTHORITY
        lease = state.lease
        if (
            capability.owner_generation != self._owner_generation
            or capability.logical_session_id != lease.logical_session_id
            or capability.evidence_turn_id != lease.evidence_turn_id
            or capability.lease_serial != lease.lease_serial
            or capability.sink_token != state.sink_token
        ):
            return CauseDisposition.INVALID_AUTHORITY
        with state.lock:
            if state.frozen:
                return CauseDisposition.CAUSE_SET_FROZEN
            if cause in state.causes:
                return CauseDisposition.ALREADY_RECORDED
            state.causes.add(cause)
            return CauseDisposition.RECORDED

    def freeze_and_resolve_terminal_causes(
        self,
        lease: EvidenceTurnLease,
        *,
        context_committed: bool,
        zero_output_no_op: bool = False,
    ) -> TerminalResolutionV1:
        if type(context_committed) is not bool:
            raise TypeError("context_committed must be an exact built-in bool")
        if type(zero_output_no_op) is not bool:
            raise TypeError("zero_output_no_op must be an exact built-in bool")
        if type(lease) is not EvidenceTurnLease:
            raise ReservationError("lease has the wrong type")
        state = self._cause_states.get(lease.terminal_cause)
        if state is None or state.lease is not lease:
            raise ReservationError("lease cause capability is stale")
        with state.lock:
            if state.frozen:
                raise ReservationError("lease cause set is already frozen")
            resolution = resolve_terminal_causes(
                tuple(state.causes),
                context_committed=context_committed,
                zero_output_no_op=zero_output_no_op,
            )
            state.frozen = True
            state.resolution = resolution
            state.context_committed = context_committed
            return resolution

    def complete_ordered_item(
        self,
        item: EvidenceWriterQueueItemV1,
        *,
        writer_succeeded: bool = True,
    ) -> None:
        """Release one dequeued ordinary item without taking the admission lock."""

        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("writer item has the wrong type")
        if type(writer_succeeded) is not bool:
            raise TypeError("writer_succeeded must be an exact built-in bool")
        with self._credit_lock:
            charge = self._queued_charges.get(item)
            if charge is None or charge.category not in (
                "ordinary",
                "terminal",
                "binding_close",
                "seal",
                "expiry",
            ):
                raise ReservationError("ordered writer item is stale or unsupported")
            if charge.category == "ordinary":
                self._ordinary_scheduler._complete_locked(item)
                self._observe_capacity(kind="ordinary_completed")
            else:
                del self._queued_charges[item]
                self._release_charge_locked(
                    charge.records_on_completion,
                    charge.bytes_on_completion,
                    physical_items=1 if charge.release_physical_item else 0,
                )
            if not writer_succeeded:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            if charge.category == "terminal":
                reservation = charge.reservation
                if type(reservation) is not EvidenceTerminalReservation:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("terminal item lost its reservation")
                terminal_state = self._terminal_reservations.get(reservation)
                if terminal_state is None or terminal_state.queued_records < 1:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("terminal reservation accounting is inconsistent")
                terminal_state.queued_records -= 1
                if terminal_state.remaining_records == 0 and terminal_state.queued_records == 0:
                    del self._terminal_reservations[reservation]
            elif charge.category in ("binding_close", "seal"):
                reservation = charge.reservation
                if type(reservation) is not SessionControlReservation:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("session-control item lost its reservation")
                control = self._session_controls.get(reservation)
                if control is None or control.queued_records < 1:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("session-control accounting is inconsistent")
                control.queued_records -= 1
                if charge.category == "seal" and writer_succeeded:
                    self._release_charge_locked(
                        control.remaining_records,
                        control.remaining_bytes,
                        physical_items=0,
                    )
                    del self._session_controls[reservation]
                    if self._current_session_control is reservation:
                        self._current_session_control = None
                    self._phase = RuntimeSessionPhase.STOPPED
                    self._capture_state = CaptureState.IDLE
                    self._owner_state = OwnerState.STOPPED
            elif charge.category == "expiry" and writer_succeeded:
                reservation = self._current_session_control
                if type(reservation) is not SessionControlReservation:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("expiry lost current session-control credits")
                control = self._session_controls.get(reservation)
                if control is None or control.queued_records:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("expiry session-control accounting is inconsistent")
                self._release_charge_locked(
                    control.remaining_records,
                    control.remaining_bytes,
                    physical_items=0,
                )
                del self._session_controls[reservation]
                self._current_session_control = None
                self._phase = RuntimeSessionPhase.STOPPED
                self._capture_state = CaptureState.IDLE
                self._owner_state = OwnerState.STOPPED
        self._advance_revoke_finalization_if_ready()

    def try_reserve_create_epoch(
        self,
        command: CreateEpochV1,
    ) -> CreateEpochReservationV1 | None:
        if type(command) is not CreateEpochV1:
            raise TypeError("command must be an exact CreateEpochV1")
        with self._admission_lock:
            if (
                not self._enabled
                or self._sticky_fault is not None
                or self._create_reservations
                or self._session_controls
                or self._pending_revoke
                or self._purge_required
                or self._drain_authority is not None
            ):
                return None
            with self._credit_lock:
                if self._queued_charges or self._leases:
                    return None
                if not self._try_charge_locked(
                    CREATE_EPOCH_RECORD_CREDITS,
                    CREATE_EPOCH_CANONICAL_BYTE_CREDITS,
                    physical_items=0,
                ):
                    return None
                reservation = object.__new__(CreateEpochReservationV1)
                reservation._validate()
                self._create_reservations[reservation] = _CreateReservationState(
                    command=command,
                    phase=_CreateReservationPhase.RESERVED,
                )
                return reservation

    def release_create_epoch_reservation(
        self,
        reservation: CreateEpochReservationV1,
    ) -> None:
        with self._admission_lock:
            if type(reservation) is not CreateEpochReservationV1:
                raise ReservationError("create reservation has the wrong type")
            with self._credit_lock:
                state = self._create_reservations.get(reservation)
                if state is None or state.phase is not _CreateReservationPhase.RESERVED:
                    raise ReservationError("create reservation is stale or consumed")
                del self._create_reservations[reservation]
                self._release_charge_locked(
                    CREATE_EPOCH_RECORD_CREDITS,
                    CREATE_EPOCH_CANONICAL_BYTE_CREDITS,
                    physical_items=0,
                )

    def try_enqueue_create_epoch(
        self,
        reservation: CreateEpochReservationV1,
    ) -> EvidenceWriterQueueItemV1 | None:
        with self._admission_lock:
            if type(reservation) is not CreateEpochReservationV1:
                raise ReservationError("create reservation has the wrong type")
            with self._credit_lock:
                state = self._create_reservations.get(reservation)
                if state is None or state.phase is not _CreateReservationPhase.RESERVED:
                    raise ReservationError("create reservation is stale or consumed")
                if not self._try_charge_locked(0, 0, physical_items=1):
                    return None
                # CreateEpochV1 atomically commits both opening records. Its single
                # physical queue item therefore advances the durable admission
                # watermark by two record ordinals.
                self._allocate_admission_ordinal_locked()
                ordinal = self._allocate_admission_ordinal_locked()
                item = EvidenceWriterQueueItemV1(
                    protocol_version=1,
                    lane=WriterQueueLane.ORDERED,
                    payload=state.command,
                    admission_ordinal=ordinal,
                )
                state.phase = _CreateReservationPhase.QUEUED
                state.item = item
                self._queued_charges[item] = _QueuedCharge(
                    records_on_completion=0,
                    bytes_on_completion=0,
                    release_physical_item=True,
                    category="create",
                    reservation=reservation,
                )
            try:
                self._writer_sink.put_nowait(item)
            except Full:
                with self._credit_lock:
                    self._rollback_failed_create_enqueue_locked(item, reservation)
                return None
            except Exception:
                with self._credit_lock:
                    self._rollback_failed_create_enqueue_locked(item, reservation)
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                return None
            return item

    def complete_create_epoch(
        self,
        item: EvidenceWriterQueueItemV1,
        *,
        disposition: StoreDisposition,
        binding_current: bool,
    ) -> SessionControlReservation | None:
        if type(item) is not EvidenceWriterQueueItemV1:
            raise ReservationError("writer item has the wrong type")
        if type(disposition) is not StoreDisposition:
            raise TypeError("disposition must be an exact StoreDisposition")
        if type(binding_current) is not bool:
            raise TypeError("binding_current must be an exact built-in bool")
        with self._admission_lock, self._credit_lock:
            charge = self._queued_charges.get(item)
            if charge is None or charge.category != "create":
                raise ReservationError("create writer item is stale or already completed")
            reservation = charge.reservation
            if type(reservation) is not CreateEpochReservationV1:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                raise ReservationError("create writer item lost its reservation")
            state = self._create_reservations.get(reservation)
            if (
                state is None
                or state.phase is not _CreateReservationPhase.QUEUED
                or state.item is not item
            ):
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                raise ReservationError("create writer item has inconsistent ownership")
            del self._queued_charges[item]
            del self._create_reservations[reservation]
            self._release_charge_locked(0, 0, physical_items=1)
            if disposition not in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT):
                self._release_charge_locked(
                    CREATE_EPOCH_RECORD_CREDITS,
                    CREATE_EPOCH_CANONICAL_BYTE_CREDITS,
                    physical_items=0,
                )
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                ticket = self._create_tickets.pop(reservation, None)
                if ticket is not None:
                    object.__setattr__(
                        ticket,
                        "disposition",
                        ConsentDisposition.CREATE_FAILED,
                    )
                    _set_ticket_signal(ticket.durability_event)
                return None
            self._release_charge_locked(
                CREATE_OPENING_RECORD_CREDITS,
                CREATE_OPENING_CANONICAL_BYTE_CREDITS,
                physical_items=0,
            )
            command = state.command
            if binding_current:
                self._reset_epoch_state_for_activation_locked()
            retained = self._mint_session_control_locked(
                consent_epoch_id=command.consent_epoch_id,
                logical_session_id=command.logical_session_id,
                cleanup_only=not binding_current,
            )
            if binding_current:
                if self._current_session_control is not None:
                    self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
                    raise ReservationError("a current session control reservation already exists")
                self._current_session_control = retained
                self._installation_id = command.installation_id
                self._producer_instance_id = command.producer_instance_id
                self._consent_epoch_id = command.consent_epoch_id
                self._logical_session_id = command.logical_session_id
                self._binding_id = command.binding_id
                self._binding_generation = command.binding_generation
                self._next_event_sequence = 3
                self._phase = RuntimeSessionPhase.OPEN
                self._session_reserved_events = 2
                self._session_reserved_bytes = 0
                self._owner_state = OwnerState.RUNNING
                self._capture_state = CaptureState.ACTIVE
            ticket = self._create_tickets.pop(reservation, None)
            if ticket is not None:
                object.__setattr__(
                    ticket,
                    "disposition",
                    ConsentDisposition.CONSENT_ACTIVATED,
                )
                _set_ticket_signal(ticket.durability_event)
            return retained

    def _reset_epoch_state_for_activation_locked(self) -> None:
        """Discard only completed epoch state before making a new epoch ACTIVE.

        The owner generation, admission/reservation serials, queue accounting,
        scheduler, and sticky writer fault are owner-lifetime.  Every lineage,
        consent terminal latch, used bearer, lease outcome, and session budget is
        epoch-lifetime.  The caller holds ``_admission_lock`` then
        ``_credit_lock`` so no fresh ACTIVE state is observable partway through
        the handoff.
        """

        if (
            self._leases
            or self._terminal_reservations
            or self._queued_charges
            or self._session_controls
            or self._create_reservations
            or self._pending_revoke
            or self._purge_required
            or self._current_session_control is not None
        ):
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("new epoch activation has live prior-epoch state")
        self._used_authorities.clear()
        self._evidence_turn_ids.clear()
        self._leases.clear()
        self._cause_states.clear()
        self._owner_state = OwnerState.ABSENT
        self._capture_state = CaptureState.IDLE
        self._installation_id = None
        self._producer_instance_id = None
        self._consent_epoch_id = None
        self._logical_session_id = None
        self._binding_id = None
        self._binding_generation = None
        self._next_event_sequence = None
        self._phase = None
        self._session_tainted = False
        self._taint_code = None
        self._session_reserved_events = 0
        self._session_reserved_bytes = 0
        self._binding_close_admitted = False
        self._seal_ticket_disposition = None
        self._expiry_authority = None
        self._replayable_turn_count = 0
        self._revoke_authority = None
        self._revoke_ticket = None
        self._revoke_item = None
        self._revoke_request_completed = False

    def release_session_control(self, reservation: SessionControlReservation) -> None:
        with self._admission_lock:
            if type(reservation) is not SessionControlReservation:
                raise ReservationError("session control reservation has the wrong type")
            with self._credit_lock:
                state = self._session_controls.get(reservation)
                if state is None:
                    raise ReservationError("session control reservation is stale")
                if self._current_session_control is reservation:
                    raise ReservationError("current session control cannot be released directly")
                del self._session_controls[reservation]
                self._release_charge_locked(
                    state.remaining_records,
                    state.remaining_bytes,
                    physical_items=0,
                )

    def try_reserve_terminal(self) -> EvidenceTerminalReservation | None:
        with self._admission_lock:
            if not self._enabled or self._sticky_fault is not None:
                return None
            with self._credit_lock:
                if not self._try_charge_locked(
                    TERMINAL_RECORD_CREDITS,
                    TERMINAL_CANONICAL_BYTE_CREDITS,
                    physical_items=0,
                ):
                    return None
                reservation = object.__new__(EvidenceTerminalReservation)
                object.__setattr__(reservation, "owner_generation", self._owner_generation)
                object.__setattr__(
                    reservation,
                    "reservation_serial",
                    self._allocate_reservation_serial_locked(),
                )
                self._terminal_reservations[reservation] = _TerminalReservationState()
                return reservation

    def release_terminal_reservation(
        self,
        reservation: EvidenceTerminalReservation,
    ) -> None:
        with self._admission_lock:
            if type(reservation) is not EvidenceTerminalReservation:
                raise ReservationError("terminal reservation has the wrong type")
            with self._credit_lock:
                state = self._terminal_reservations.get(reservation)
                if state is None:
                    raise ReservationError("terminal reservation is stale")
                if state.queued_records:
                    raise ReservationError("queued terminal credits cannot be released early")
                del self._terminal_reservations[reservation]
                self._release_charge_locked(
                    state.remaining_records,
                    state.remaining_bytes,
                    physical_items=0,
                )

    def _preliminary_append_disposition_locked(self) -> AppendDisposition | None:
        if not self._enabled:
            return AppendDisposition.DISABLED
        if self._sticky_fault is not None:
            return AppendDisposition.WRITER_FAULT
        if self._session_tainted:
            return AppendDisposition.SESSION_TAINTED
        if self._current_session_control is None or self._capture_state is not CaptureState.ACTIVE:
            return AppendDisposition.CONSENT_MISSING
        if self._phase is not RuntimeSessionPhase.OPEN:
            return AppendDisposition.SESSION_CLOSING
        return None

    def _conversation_authority_current_locked(self, authority: object) -> bool:
        """Fail closed once the host lifecycle owner has retired this binding."""

        guard = self._conversation_authority_is_current
        if guard is None:
            return True
        current = guard(authority)
        if type(current) is not bool:
            raise TypeError("conversation authority guard must return an exact bool")
        return current

    def _authority_lineage_matches_locked(
        self,
        authority: CommandAdmissionAuthorityV1
        | UserTurnAuthorityV1
        | ProactiveTurnAuthorityV1
        | ReplayTurnAuthorityV1
        | BindingCloseAuthorityV1
        | LifecycleSealAuthorityV1,
    ) -> bool:
        return (
            authority.owner_generation == self._owner_generation
            and authority.binding_id == self._binding_id
            and authority.binding_generation == self._binding_generation
            and authority.consent_epoch_id == self._consent_epoch_id
            and authority.logical_session_id == self._logical_session_id
        )

    def _lease_lineage_matches_locked(self, lease: EvidenceTurnLease) -> bool:
        state = self._leases.get(lease)
        if state is None:
            return False
        cause = lease.terminal_cause
        if type(cause) is not TerminalCauseCapabilityV1:
            return False
        cause_state = self._cause_states.get(cause)
        return (
            type(lease.protocol_version) is int
            and lease.protocol_version == 1
            and type(lease.owner_generation) is int
            and lease.owner_generation == self._owner_generation
            and type(lease.lease_serial) is int
            and lease.lease_serial == state.lease_serial
            and type(lease.operation_serial) is int
            and lease.operation_serial == state.operation_serial
            and lease.turn_kind is state.turn_kind
            and type(lease.evidence_turn_id) is str
            and lease.evidence_turn_id == state.evidence_turn_id
            and lease.binding_id == self._binding_id
            and lease.binding_generation == self._binding_generation
            and lease.consent_epoch_id == self._consent_epoch_id
            and lease.logical_session_id == self._logical_session_id
            and lease.terminal_reservation is state.terminal_reservation
            and cause_state is not None
            and cause_state.lease is lease
            and type(cause.protocol_version) is int
            and cause.protocol_version == 1
            and type(cause.owner_generation) is int
            and cause.owner_generation == self._owner_generation
            and cause.logical_session_id == self._logical_session_id
            and cause.evidence_turn_id == state.evidence_turn_id
            and type(cause.lease_serial) is int
            and cause.lease_serial == state.lease_serial
            and type(cause.sink_token) is bytes
            and cause.sink_token == cause_state.sink_token
            and (
                lease.user_authority is state.authority
                if state.turn_kind is TurnKind.USER_RESPONSE
                else lease.user_authority is None
            )
            and _turn_authority_signature(state.authority) == state.authority_signature
        )

    def _turn_event_is_valid_locked(
        self,
        lease: EvidenceTurnLease,
        state: _LeaseState,
        snapshot: EvidenceSnapshotV1,
    ) -> bool:
        if snapshot.logical_session_id != lease.logical_session_id:
            return False
        kind = snapshot.event_kind
        payload = snapshot.payload
        if kind is EventKind.TURN_OPENED:
            if state.opened:
                return False
            if (
                getattr(payload, "evidence_turn_id", None) != lease.evidence_turn_id
                or getattr(payload, "turn_kind", None) is not state.turn_kind
            ):
                return False
            if state.turn_kind is TurnKind.USER_RESPONSE:
                authority = state.authority
                return (
                    type(authority) is UserTurnAuthorityV1
                    and getattr(payload, "utterance_id", None) == authority.utterance_id
                    and getattr(payload, "replay_of_evidence_turn_id", None) is None
                )
            if state.turn_kind is TurnKind.REPLAY:
                authority = state.authority
                return (
                    type(authority) is ReplayTurnAuthorityV1
                    and getattr(payload, "utterance_id", None) is None
                    and getattr(payload, "replay_of_evidence_turn_id", None)
                    == authority.replay_of_evidence_turn_id
                )
            return (
                getattr(payload, "utterance_id", None) is None
                and getattr(payload, "replay_of_evidence_turn_id", None) is None
            )
        if kind is EventKind.USER_FINAL_ACCEPTED:
            authority = state.authority
            return (
                state.opened
                and not state.user_final
                and state.turn_kind is TurnKind.USER_RESPONSE
                and type(authority) is UserTurnAuthorityV1
                and getattr(payload, "evidence_turn_id", None) == lease.evidence_turn_id
                and getattr(payload, "utterance_id", None) == authority.utterance_id
                and getattr(payload, "source", None) is authority.source
            )
        if kind is EventKind.ASSISTANT_SEGMENT_GENERATED:
            return (
                state.opened
                and state.user_final
                and not state.snapshot
                and state.turn_kind is TurnKind.USER_RESPONSE
                and getattr(payload, "evidence_turn_id", None) == lease.evidence_turn_id
                and getattr(payload, "segment_ordinal", None) == state.generated_segment_count + 1
            )
        if kind is EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL:
            ready = state.opened and not state.snapshot
            if state.turn_kind is TurnKind.USER_RESPONSE:
                ready = ready and state.user_final
            return (
                ready
                and getattr(payload, "evidence_turn_id", None) == lease.evidence_turn_id
                and getattr(payload, "chunk_ordinal", None)
                == state.transport_confirmed_full_count + 1
            )
        if kind is EventKind.TURN_SNAPSHOT:
            ready = state.opened and not state.snapshot
            if state.turn_kind is TurnKind.USER_RESPONSE:
                ready = ready and state.user_final
            return (
                ready
                and getattr(payload, "evidence_turn_id", None) == lease.evidence_turn_id
                and getattr(payload, "turn_kind", None) is state.turn_kind
                and getattr(payload, "generated_segment_count", None)
                == state.generated_segment_count
                and getattr(payload, "transport_confirmed_full_count", None)
                == state.transport_confirmed_full_count
            )
        if kind is EventKind.TURN_SETTLED:
            return (
                state.snapshot
                and not state.settlement_attempted
                and getattr(payload, "evidence_turn_id", None) == lease.evidence_turn_id
                and getattr(payload, "generated_segment_count", None)
                == state.generated_segment_count
                and getattr(payload, "transport_confirmed_full_count", None)
                == state.transport_confirmed_full_count
            )
        return False

    def _terminal_resolution_taint_locked(
        self,
        lease: EvidenceTurnLease,
        snapshot: EvidenceSnapshotV1,
    ) -> TaintCode | None:
        payload = snapshot.payload
        if type(payload) is not TurnSettledPayloadV1:
            return TaintCode.TERMINAL_CONFLICT
        cause_state = self._cause_states.get(lease.terminal_cause)
        if cause_state is None or cause_state.lease is not lease:
            return TaintCode.TERMINAL_MISSING
        with cause_state.lock:
            resolution = cause_state.resolution
            context_committed = cause_state.context_committed
            if not cause_state.frozen or resolution is None or context_committed is None:
                return TaintCode.TERMINAL_MISSING
            if (
                payload.terminal_disposition is not resolution.terminal_disposition
                or payload.terminal_reason is not resolution.terminal_reason
                or payload.context_committed is not context_committed
            ):
                return TaintCode.TERMINAL_CONFLICT
        return None

    def _apply_turn_event_locked(
        self,
        state: _LeaseState,
        snapshot: EvidenceSnapshotV1,
    ) -> None:
        kind = snapshot.event_kind
        if kind is EventKind.TURN_OPENED:
            state.opened = True
            state.lease_open_ordinal = self._next_admission_ordinal - 1
        elif kind is EventKind.USER_FINAL_ACCEPTED:
            state.user_final = True
        elif kind is EventKind.ASSISTANT_SEGMENT_GENERATED:
            state.generated_segment_count += 1
            payload = snapshot.payload
            if type(payload) is AssistantSegmentGeneratedPayloadV1:
                state.evidence_segment_ids.append(payload.evidence_segment_id)
        elif kind is EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL:
            state.transport_confirmed_full_count += 1
        elif kind is EventKind.TURN_SNAPSHOT:
            state.snapshot = True

    def _try_enqueue_terminal_locked(
        self,
        lease: EvidenceTurnLease,
        state: _LeaseState,
        snapshot: EvidenceSnapshotV1,
        canonical_bytes: int,
    ) -> AppendDisposition:
        if (
            snapshot.installation_id != self._installation_id
            or snapshot.producer_instance_id != self._producer_instance_id
            or snapshot.logical_session_id != self._logical_session_id
        ):
            return AppendDisposition.INVALID_AUTHORITY
        if self._next_event_sequence is None:
            return AppendDisposition.CONSENT_MISSING
        if snapshot.event_sequence != self._next_event_sequence:
            self._taint_locked(TaintCode.SEQUENCE_CONFLICT)
            return AppendDisposition.SESSION_TAINTED
        if state.lease_open_ordinal is None:
            return AppendDisposition.INVALID_AUTHORITY
        if not 1 <= canonical_bytes <= MAX_CANONICAL_RECORD_BYTES:
            self._taint_locked(TaintCode.OVERSIZE)
            return AppendDisposition.REJECTED_OVERSIZE
        with self._credit_lock:
            if self._sticky_fault is not None:
                return AppendDisposition.WRITER_FAULT
            if not self._enabled:
                return AppendDisposition.CONSENT_MISSING
            if self._phase is RuntimeSessionPhase.STOPPED:
                return AppendDisposition.SESSION_CLOSING
            terminal = self._terminal_reservations.get(state.terminal_reservation)
            if (
                terminal is None
                or terminal.remaining_records < 1
                or terminal.remaining_bytes < MAX_CANONICAL_RECORD_BYTES
            ):
                self._taint_locked(TaintCode.TERMINAL_MISSING)
                return AppendDisposition.SESSION_TAINTED
            if not self._try_charge_locked(0, 0, physical_items=1):
                return AppendDisposition.DROPPED_CAPACITY
            ordinal = self._allocate_admission_ordinal_locked()
            self._next_event_sequence += 1
            terminal.remaining_records -= 1
            terminal.remaining_bytes -= MAX_CANONICAL_RECORD_BYTES
            terminal.queued_records += 1
            self._release_charge_locked(
                0,
                MAX_CANONICAL_RECORD_BYTES - canonical_bytes,
                physical_items=0,
            )
            queued = QueuedEvidenceRecordV1(
                protocol_version=1,
                snapshot=snapshot,
                admission_ordinal=ordinal,
                reservation_class=QueueReservationClass.TERMINAL,
                lease_open_ordinal=state.lease_open_ordinal,
            )
            item = EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=WriterQueueLane.ORDERED,
                payload=queued,
                admission_ordinal=ordinal,
            )
            self._queued_charges[item] = _QueuedCharge(
                records_on_completion=1,
                bytes_on_completion=canonical_bytes,
                release_physical_item=True,
                category="terminal",
                reservation=state.terminal_reservation,
            )
        try:
            self._writer_sink.put_nowait(item)
        except Full:
            with self._credit_lock:
                self._rollback_failed_terminal_enqueue_locked(
                    item,
                    state.terminal_reservation,
                )
            return AppendDisposition.DROPPED_CAPACITY
        except Exception:
            with self._credit_lock:
                self._rollback_failed_terminal_enqueue_locked(
                    item,
                    state.terminal_reservation,
                )
                self._taint_locked(TaintCode.ADMISSION_GAP)
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.SESSION_TAINTED
        return AppendDisposition.ADMITTED

    def _rollback_failed_terminal_enqueue_locked(
        self,
        item: EvidenceWriterQueueItemV1,
        reservation: EvidenceTerminalReservation,
    ) -> None:
        charge = self._queued_charges.pop(item, None)
        if (
            charge is None
            or charge.category != "terminal"
            or charge.reservation is not reservation
            or charge.records_on_completion != 1
            or not charge.release_physical_item
        ):
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("terminal enqueue rollback lost its queue charge")
        terminal = self._terminal_reservations.get(reservation)
        if terminal is None or terminal.queued_records < 1:
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("terminal enqueue rollback lost its reservation")
        terminal.queued_records -= 1
        terminal.remaining_records += 1
        terminal.remaining_bytes += MAX_CANONICAL_RECORD_BYTES
        self._release_charge_locked(
            charge.records_on_completion,
            charge.bytes_on_completion,
            physical_items=1,
        )
        if not self._try_charge_locked(
            1,
            MAX_CANONICAL_RECORD_BYTES,
            physical_items=0,
        ):
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("terminal enqueue rollback could not restore credits")
        if (
            self._next_event_sequence is None
            or self._next_event_sequence < 1
            or self._next_admission_ordinal != item.admission_ordinal + 1
        ):
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("terminal enqueue rollback lost its sequence")
        self._next_event_sequence -= 1
        self._next_admission_ordinal = item.admission_ordinal

    def _release_unused_terminal_credits_locked(
        self,
        reservation: EvidenceTerminalReservation,
    ) -> None:
        terminal = self._terminal_reservations.get(reservation)
        if terminal is None:
            return
        self._release_charge_locked(
            terminal.remaining_records,
            terminal.remaining_bytes,
            physical_items=0,
        )
        terminal.remaining_records = 0
        terminal.remaining_bytes = 0
        if terminal.queued_records == 0:
            del self._terminal_reservations[reservation]

    def _try_enqueue_session_control_locked(
        self,
        command: BindingCloseV1 | SealEpochV1,
        *,
        event_sequence: int,
        category: str,
    ) -> AppendDisposition:
        reservation = self._current_session_control
        if reservation is None:
            return AppendDisposition.INVALID_AUTHORITY
        if command.admission_ordinal != self._next_admission_ordinal:
            return AppendDisposition.INVALID_AUTHORITY
        if self._next_event_sequence is None:
            return AppendDisposition.CONSENT_MISSING
        if event_sequence != self._next_event_sequence:
            self._taint_locked(TaintCode.SEQUENCE_CONFLICT)
            return AppendDisposition.SESSION_TAINTED
        with self._credit_lock:
            control = self._session_controls.get(reservation)
            if (
                control is None
                or control.remaining_records < 1
                or control.remaining_bytes < MAX_CANONICAL_RECORD_BYTES
            ):
                self._taint_locked(TaintCode.TERMINAL_MISSING)
                return AppendDisposition.SESSION_TAINTED
            if not self._try_charge_locked(0, 0, physical_items=1):
                return AppendDisposition.DROPPED_CAPACITY
            ordinal = self._allocate_admission_ordinal_locked()
            self._next_event_sequence += 1
            control.remaining_records -= 1
            control.remaining_bytes -= MAX_CANONICAL_RECORD_BYTES
            control.queued_records += 1
            item = EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=WriterQueueLane.ORDERED,
                payload=command,
                admission_ordinal=ordinal,
            )
            self._queued_charges[item] = _QueuedCharge(
                records_on_completion=1,
                bytes_on_completion=MAX_CANONICAL_RECORD_BYTES,
                release_physical_item=True,
                category=category,
                reservation=reservation,
            )
        try:
            self._writer_sink.put_nowait(item)
        except Exception:
            with self._credit_lock:
                self._queued_charges.pop(item, None)
                control = self._session_controls.get(reservation)
                if control is not None and control.queued_records:
                    control.queued_records -= 1
                self._release_charge_locked(
                    1,
                    MAX_CANONICAL_RECORD_BYTES,
                    physical_items=1,
                )
                self._taint_locked(TaintCode.ADMISSION_GAP)
            return AppendDisposition.SESSION_TAINTED
        return AppendDisposition.ADMITTED

    def _try_enqueue_zero_record_control_locked(
        self,
        command: ExpireSessionV1,
        category: str,
    ) -> bool:
        with self._credit_lock:
            if not self._try_charge_locked(0, 0, physical_items=1):
                return False
            ordinal = self._allocate_admission_ordinal_locked()
            item = EvidenceWriterQueueItemV1(
                protocol_version=1,
                lane=WriterQueueLane.ORDERED,
                payload=command,
                admission_ordinal=ordinal,
            )
            self._queued_charges[item] = _QueuedCharge(
                records_on_completion=0,
                bytes_on_completion=0,
                release_physical_item=True,
                category=category,
                reservation=None,
            )
        try:
            self._writer_sink.put_nowait(item)
        except Exception:
            with self._credit_lock:
                self._queued_charges.pop(item, None)
                self._release_charge_locked(0, 0, physical_items=1)
            return False
        return True

    def _try_reserve_terminal_locked(self) -> EvidenceTerminalReservation | None:
        if not self._try_charge_locked(
            TERMINAL_RECORD_CREDITS,
            TERMINAL_CANONICAL_BYTE_CREDITS,
            physical_items=0,
        ):
            return None
        reservation = object.__new__(EvidenceTerminalReservation)
        object.__setattr__(reservation, "owner_generation", self._owner_generation)
        object.__setattr__(
            reservation,
            "reservation_serial",
            self._allocate_reservation_serial_locked(),
        )
        self._terminal_reservations[reservation] = _TerminalReservationState()
        return reservation

    def _release_terminal_locked(
        self,
        reservation: EvidenceTerminalReservation,
    ) -> None:
        state = self._terminal_reservations.get(reservation)
        if state is None or state.queued_records:
            raise ReservationError("terminal reservation cannot be released")
        del self._terminal_reservations[reservation]
        self._release_charge_locked(
            state.remaining_records,
            state.remaining_bytes,
            physical_items=0,
        )

    def _mint_lease_locked(
        self,
        *,
        authority: UserTurnAuthorityV1 | ProactiveTurnAuthorityV1 | ReplayTurnAuthorityV1,
        operation: ConversationOperationReservation,
        terminal: EvidenceTerminalReservation,
        turn_kind: TurnKind,
        evidence_turn_id: str,
    ) -> EvidenceTurnLease:
        if (
            self._binding_id is None
            or self._binding_generation is None
            or self._consent_epoch_id is None
            or self._logical_session_id is None
        ):
            raise ReservationError("active lease lineage is incomplete")
        lease_serial = self._allocate_reservation_serial_locked()
        capability = object.__new__(TerminalCauseCapabilityV1)
        object.__setattr__(capability, "protocol_version", 1)
        object.__setattr__(capability, "owner_generation", self._owner_generation)
        object.__setattr__(capability, "logical_session_id", self._logical_session_id)
        object.__setattr__(capability, "evidence_turn_id", evidence_turn_id)
        object.__setattr__(capability, "lease_serial", lease_serial)
        object.__setattr__(capability, "sink_token", secrets.token_bytes(16))
        capability._validate()
        lease = object.__new__(EvidenceTurnLease)
        object.__setattr__(lease, "protocol_version", 1)
        object.__setattr__(lease, "owner_generation", self._owner_generation)
        object.__setattr__(lease, "lease_serial", lease_serial)
        object.__setattr__(lease, "operation_serial", operation.operation_serial)
        object.__setattr__(lease, "turn_kind", turn_kind)
        object.__setattr__(lease, "evidence_turn_id", evidence_turn_id)
        object.__setattr__(lease, "binding_id", self._binding_id)
        object.__setattr__(lease, "binding_generation", self._binding_generation)
        object.__setattr__(lease, "consent_epoch_id", self._consent_epoch_id)
        object.__setattr__(lease, "logical_session_id", self._logical_session_id)
        object.__setattr__(lease, "terminal_reservation", terminal)
        object.__setattr__(lease, "terminal_cause", capability)
        object.__setattr__(
            lease,
            "user_authority",
            authority if type(authority) is UserTurnAuthorityV1 else None,
        )
        self._leases[lease] = _LeaseState(
            authority=authority,
            operation=operation,
            terminal_reservation=terminal,
            turn_kind=turn_kind,
            lease_serial=lease_serial,
            evidence_turn_id=evidence_turn_id,
            operation_serial=operation.operation_serial,
            authority_signature=_turn_authority_signature(authority),
        )
        self._cause_states[capability] = _CauseState(
            lease=lease,
            lock=Lock(),
            causes=set(),
            sink_token=capability.sink_token,
        )
        return lease

    def _discard_unopened_lease_locked(self, lease: EvidenceTurnLease) -> None:
        with self._credit_lock:
            state = self._leases.pop(lease, None)
            if state is None:
                return
            self._cause_states.pop(lease.terminal_cause, None)
            self._release_terminal_locked(state.terminal_reservation)

    def _try_enqueue_ordinary_locked(
        self,
        snapshot: EvidenceSnapshotV1,
    ) -> AppendDisposition:
        if (
            snapshot.installation_id != self._installation_id
            or snapshot.producer_instance_id != self._producer_instance_id
            or snapshot.logical_session_id != self._logical_session_id
        ):
            return AppendDisposition.INVALID_AUTHORITY
        try:
            prepared = self._ordinary_scheduler.prepare(snapshot)
        except EvidenceAdmissionError:
            self._taint_locked(TaintCode.OVERSIZE)
            return AppendDisposition.REJECTED_OVERSIZE
        if self._deny_filter is not None:
            denied = self._deny_filter(snapshot)
            if type(denied) is not bool:
                raise TypeError("deny_filter must return an exact built-in bool")
            if denied:
                self._taint_locked(TaintCode.DENY_FILTER)
                return AppendDisposition.REJECTED_DENIED
        if self._quota_check is not None:
            allowed = self._quota_check(snapshot)
            if type(allowed) is not bool:
                raise TypeError("quota_check must return an exact built-in bool")
            if not allowed:
                self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                return AppendDisposition.QUOTA_EXCEEDED
        if self._next_event_sequence is None:
            return AppendDisposition.CONSENT_MISSING
        if snapshot.event_sequence != self._next_event_sequence:
            self._taint_locked(TaintCode.SEQUENCE_CONFLICT)
            return AppendDisposition.SESSION_TAINTED
        with self._credit_lock:
            if self._sticky_fault is not None:
                return AppendDisposition.WRITER_FAULT
            if not self._enabled or self._pending_revoke:
                return AppendDisposition.CONSENT_MISSING
            if self._phase is not RuntimeSessionPhase.OPEN:
                return AppendDisposition.SESSION_CLOSING
            if self._purge_required or self._capture_state is CaptureState.FAULTED:
                return AppendDisposition.SESSION_TAINTED
        try:
            item, rejection_source = self._ordinary_scheduler._try_admit_observed(prepared)
        except _SchedulerPublicationFull:
            with self._credit_lock:
                self._taint_locked(TaintCode.ADMISSION_GAP)
            return AppendDisposition.SESSION_TAINTED
        except _SchedulerPublicationFault:
            with self._credit_lock:
                self._taint_locked(TaintCode.ADMISSION_GAP)
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.SESSION_TAINTED
        except ReservationError:
            with self._credit_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT
        if item is None:
            if rejection_source is None:
                raise ReservationError("rejected ordinary admission lost its rejection source")
            with self._credit_lock:
                self._observe_capacity(
                    kind="ordinary_rejected",
                    rejection_source=rejection_source,
                )
            return AppendDisposition.DROPPED_CAPACITY
        if rejection_source is not None:
            raise ReservationError("admitted ordinary admission has a rejection source")
        self._next_event_sequence += 1
        with self._credit_lock:
            self._observe_capacity(kind="ordinary_admitted")
        return AppendDisposition.ADMITTED

    def _try_enqueue_user_final_batch_locked(
        self,
        lease: EvidenceTurnLease,
        state: _LeaseState,
        opened: EvidenceSnapshotV1,
        accepted: EvidenceSnapshotV1,
    ) -> AppendDisposition:
        snapshots = (opened, accepted)
        if any(
            snapshot.installation_id != self._installation_id
            or snapshot.producer_instance_id != self._producer_instance_id
            or snapshot.logical_session_id != self._logical_session_id
            for snapshot in snapshots
        ):
            return AppendDisposition.INVALID_AUTHORITY
        if self._next_event_sequence is None:
            return AppendDisposition.CONSENT_MISSING
        if tuple(snapshot.event_sequence for snapshot in snapshots) != (
            self._next_event_sequence,
            self._next_event_sequence + 1,
        ):
            self._taint_locked(TaintCode.SEQUENCE_CONFLICT)
            return AppendDisposition.SESSION_TAINTED
        if not self._turn_event_is_valid_locked(lease, state, opened):
            return AppendDisposition.INVALID_AUTHORITY
        prepared_records: list[_PreparedEvidenceRecordV1] = []
        for snapshot in snapshots:
            try:
                prepared_records.append(self._ordinary_scheduler.prepare(snapshot))
            except EvidenceAdmissionError:
                self._taint_locked(TaintCode.OVERSIZE)
                return AppendDisposition.REJECTED_OVERSIZE
            if self._deny_filter is not None:
                denied = self._deny_filter(snapshot)
                if type(denied) is not bool:
                    raise TypeError("deny_filter must return an exact built-in bool")
                if denied:
                    self._taint_locked(TaintCode.DENY_FILTER)
                    return AppendDisposition.REJECTED_DENIED
            if self._quota_check is not None:
                allowed = self._quota_check(snapshot)
                if type(allowed) is not bool:
                    raise TypeError("quota_check must return an exact built-in bool")
                if not allowed:
                    self._taint_locked(TaintCode.QUOTA_EXCEEDED)
                    return AppendDisposition.QUOTA_EXCEEDED
        with self._credit_lock:
            if self._sticky_fault is not None:
                return AppendDisposition.WRITER_FAULT
            if not self._enabled or self._pending_revoke:
                return AppendDisposition.CONSENT_MISSING
            if self._phase is not RuntimeSessionPhase.OPEN:
                return AppendDisposition.SESSION_CLOSING
            if self._purge_required or self._capture_state is CaptureState.FAULTED:
                return AppendDisposition.SESSION_TAINTED
        prepared = tuple(prepared_records)
        try:
            items, rejection_source = self._ordinary_scheduler._try_admit_batch_observed(prepared)
        except _SchedulerPublicationFull:
            with self._credit_lock:
                self._taint_locked(TaintCode.ADMISSION_GAP)
            return AppendDisposition.SESSION_TAINTED
        except _SchedulerPublicationFault:
            with self._credit_lock:
                self._taint_locked(TaintCode.ADMISSION_GAP)
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.SESSION_TAINTED
        except ReservationError:
            with self._credit_lock:
                self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            return AppendDisposition.WRITER_FAULT
        if items is None:
            if rejection_source is None:
                raise ReservationError("rejected ordinary batch lost its rejection source")
            with self._credit_lock:
                self._observe_capacity(
                    kind="ordinary_rejected",
                    rejection_source=rejection_source,
                )
            return AppendDisposition.DROPPED_CAPACITY
        if rejection_source is not None:
            raise ReservationError("admitted ordinary batch has a rejection source")
        total_bytes = sum(record.canonical_byte_charge for record in prepared)
        self._next_event_sequence += 2
        state.opened = True
        state.user_final = True
        state.lease_open_ordinal = items[0].admission_ordinal
        state.event_count += 2
        state.canonical_bytes += total_bytes
        with self._credit_lock:
            self._observe_capacity(kind="ordinary_admitted")
        return AppendDisposition.ADMITTED

    def _try_enqueue_terminal_batch_locked(
        self,
        lease: EvidenceTurnLease,
        state: _LeaseState,
        snapshot: EvidenceSnapshotV1,
        settled: EvidenceSnapshotV1,
        *,
        claim_rollover: bool = True,
    ) -> tuple[
        AppendDisposition,
        _AdmissionRolloverPreparationV1 | None,
        Callable[[_AdmissionRolloverPreparationV1], None] | None,
    ]:
        def result(
            disposition: AppendDisposition,
        ) -> tuple[
            AppendDisposition,
            _AdmissionRolloverPreparationV1 | None,
            Callable[[_AdmissionRolloverPreparationV1], None] | None,
        ]:
            return disposition, None, None

        snapshots = (snapshot, settled)
        if any(
            item.installation_id != self._installation_id
            or item.producer_instance_id != self._producer_instance_id
            or item.logical_session_id != self._logical_session_id
            for item in snapshots
        ):
            return result(AppendDisposition.INVALID_AUTHORITY)
        if self._next_event_sequence is None:
            return result(AppendDisposition.CONSENT_MISSING)
        if tuple(item.event_sequence for item in snapshots) != (
            self._next_event_sequence,
            self._next_event_sequence + 1,
        ):
            self._taint_locked(TaintCode.SEQUENCE_CONFLICT)
            return result(AppendDisposition.SESSION_TAINTED)
        if not self._turn_event_is_valid_locked(lease, state, snapshot):
            return result(AppendDisposition.INVALID_AUTHORITY)
        snapshot_payload = snapshot.payload
        settled_payload = settled.payload
        if (
            type(snapshot_payload) is not TurnSnapshotPayloadV1
            or type(settled_payload) is not TurnSettledPayloadV1
            or settled_payload.evidence_turn_id != snapshot_payload.evidence_turn_id
            or settled_payload.generated_segment_count != snapshot_payload.generated_segment_count
            or settled_payload.transport_confirmed_full_count
            != snapshot_payload.transport_confirmed_full_count
        ):
            return result(AppendDisposition.INVALID_AUTHORITY)
        terminal_taint = self._terminal_resolution_taint_locked(lease, settled)
        if terminal_taint is not None:
            self._taint_locked(terminal_taint)
            state.settlement_attempted = True
            self._leases.pop(lease, None)
            with self._credit_lock:
                self._release_unused_terminal_credits_locked(state.terminal_reservation)
            return result(AppendDisposition.SESSION_TAINTED)
        if state.lease_open_ordinal is None:
            return result(AppendDisposition.INVALID_AUTHORITY)
        canonical_sizes: list[int] = []
        for terminal_snapshot in snapshots:
            try:
                canonical_sizes.append(canonical_record_bytes(terminal_snapshot))
            except EvidenceAdmissionError:
                self._taint_locked(TaintCode.OVERSIZE)
                return result(AppendDisposition.REJECTED_OVERSIZE)
        with self._credit_lock:
            terminal = self._terminal_reservations.get(state.terminal_reservation)
            if (
                terminal is None
                or terminal.remaining_records != 2
                or terminal.remaining_bytes != TERMINAL_CANONICAL_BYTE_CREDITS
                or terminal.queued_records != 0
            ):
                self._taint_locked(TaintCode.TERMINAL_MISSING)
                return result(AppendDisposition.SESSION_TAINTED)
            if not self._admission_ordinals_available_locked(2):
                return result(AppendDisposition.WRITER_FAULT)
            if not self._try_charge_locked(0, 0, physical_items=2):
                return result(AppendDisposition.DROPPED_CAPACITY)
            first_ordinal = self._allocate_admission_ordinal_locked()
            second_ordinal = self._allocate_admission_ordinal_locked()
            self._next_event_sequence += 2
            terminal.remaining_records = 0
            terminal.remaining_bytes = 0
            terminal.queued_records = 2
            self._release_charge_locked(
                0,
                TERMINAL_CANONICAL_BYTE_CREDITS - sum(canonical_sizes),
                physical_items=0,
            )
            records = tuple(
                QueuedEvidenceRecordV1(
                    protocol_version=1,
                    snapshot=item,
                    admission_ordinal=ordinal,
                    reservation_class=QueueReservationClass.TERMINAL,
                    lease_open_ordinal=state.lease_open_ordinal,
                )
                for item, ordinal in zip(
                    snapshots,
                    (first_ordinal, second_ordinal),
                    strict=True,
                )
            )
            items = tuple(
                EvidenceWriterQueueItemV1(
                    protocol_version=1,
                    lane=WriterQueueLane.ORDERED,
                    payload=record,
                    admission_ordinal=record.admission_ordinal,
                )
                for record in records
            )
            for writer_item, canonical_bytes in zip(items, canonical_sizes, strict=True):
                self._queued_charges[writer_item] = _QueuedCharge(
                    records_on_completion=1,
                    bytes_on_completion=canonical_bytes,
                    release_physical_item=True,
                    category="terminal",
                    reservation=state.terminal_reservation,
                )
        try:
            self._writer_sink.put_ordered_batch_nowait(items)
        except Exception:
            with self._credit_lock:
                for writer_item in items:
                    self._queued_charges.pop(writer_item, None)
                self._release_charge_locked(
                    TERMINAL_RECORD_CREDITS,
                    sum(canonical_sizes),
                    physical_items=2,
                )
                self._terminal_reservations.pop(state.terminal_reservation, None)
                self._taint_locked(TaintCode.ADMISSION_GAP)
            state.settlement_attempted = True
            self._leases.pop(lease, None)
            return result(AppendDisposition.SESSION_TAINTED)
        state.snapshot = True
        state.settlement_attempted = True
        state.event_count += 2
        state.canonical_bytes += sum(canonical_sizes)
        cause_state = self._cause_states.get(lease.terminal_cause)
        if cause_state is None or cause_state.lease is not lease:
            self._taint_locked(TaintCode.TERMINAL_MISSING)
            return result(AppendDisposition.SESSION_TAINTED)
        with cause_state.lock:
            resolution = cause_state.resolution
            context_committed = cause_state.context_committed
            if (
                not cause_state.frozen
                or resolution is None
                or context_committed is None
                or cause_state.settled_outcome is not None
            ):
                self._taint_locked(TaintCode.TERMINAL_CONFLICT)
                return result(AppendDisposition.SESSION_TAINTED)
            cause_state.settled_outcome = SettledTerminalOutcomeV1(
                terminal_disposition=resolution.terminal_disposition,
                terminal_reason=resolution.terminal_reason,
                context_committed=context_committed,
            )
        self._leases.pop(lease, None)
        claimed = self._try_claim_capacity_rollover_locked() if claim_rollover else None
        callback = self._rollover_claimed
        return AppendDisposition.ADMITTED, claimed, callback

    def _taint_locked(self, code: TaintCode) -> None:
        if self._taint_code is None:
            self._taint_code = code
        self._session_tainted = True
        self._capture_state = CaptureState.FAULTED

    def _rollback_failed_create_enqueue_locked(
        self,
        item: EvidenceWriterQueueItemV1,
        reservation: CreateEpochReservationV1,
    ) -> None:
        self._queued_charges.pop(item, None)
        self._create_reservations.pop(reservation, None)
        self._release_charge_locked(
            CREATE_EPOCH_RECORD_CREDITS,
            CREATE_EPOCH_CANONICAL_BYTE_CREDITS,
            physical_items=1,
        )

    def _mint_session_control_locked(
        self,
        *,
        consent_epoch_id: str,
        logical_session_id: str,
        cleanup_only: bool,
    ) -> SessionControlReservation:
        reservation = object.__new__(SessionControlReservation)
        object.__setattr__(reservation, "owner_generation", self._owner_generation)
        object.__setattr__(
            reservation,
            "reservation_serial",
            self._allocate_reservation_serial_locked(),
        )
        object.__setattr__(reservation, "consent_epoch_id", consent_epoch_id)
        object.__setattr__(reservation, "logical_session_id", logical_session_id)
        object.__setattr__(reservation, "cleanup_only", cleanup_only)
        self._session_controls[reservation] = _SessionControlState()
        return reservation

    def _release_epoch_session_controls_locked(self) -> None:
        controls = tuple(self._session_controls.items())
        if any(state.queued_records for _, state in controls):
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("revoke terminal outcome preceded ordered control drain")
        records = sum(state.remaining_records for _, state in controls)
        canonical_bytes = sum(state.remaining_bytes for _, state in controls)
        self._release_charge_locked(records, canonical_bytes, physical_items=0)
        self._session_controls.clear()
        self._current_session_control = None

    def _complete_revoke_writer_fault_locked(
        self,
        ticket: RevokeTicketV1,
        item: EvidenceWriterQueueItemV1,
    ) -> None:
        """Terminally fault one owned revoke item without touching live leases."""

        if self._revoke_ticket is not ticket or self._revoke_item is not item:
            raise ReservationError("revoke fault completion item is stale")
        self._revoke_item = None
        object.__setattr__(ticket, "disposition", RevokeDisposition.WRITER_FAULT)
        self._pending_revoke = False
        self._purge_required = True
        self._phase = RuntimeSessionPhase.STOPPED
        self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
        _set_ticket_signal(ticket.durability_event)
        _set_ticket_signal(ticket.terminal_event)

    def _try_charge_locked(
        self,
        records: int,
        canonical_bytes: int,
        *,
        physical_items: int,
    ) -> bool:
        return self._capacity.try_charge(
            records,
            canonical_bytes,
            physical_items=physical_items,
        )

    def _release_charge_locked(
        self,
        records: int,
        canonical_bytes: int,
        *,
        physical_items: int,
    ) -> None:
        try:
            self._capacity.release(
                records,
                canonical_bytes,
                physical_items=physical_items,
            )
        except ReservationError:
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise

    def _admission_ordinals_available_locked(self, count: int) -> bool:
        available = self._capacity.ordinals_available(count)
        if not available:
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
        return available

    def _allocate_admission_ordinal_locked(self) -> int:
        try:
            return self._capacity.allocate_ordinal()
        except ReservationError:
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise

    def _allocate_reservation_serial_locked(self) -> int:
        serial = self._next_reservation_serial
        if serial > _MAX_UNSIGNED_63:
            self._latch_writer_fault_locked(WriterFault.SQLITE_FAULT)
            raise ReservationError("reservation serial overflowed")
        self._next_reservation_serial += 1
        return serial

    def _latch_writer_fault_locked(self, fault: WriterFault) -> None:
        if self._sticky_fault is None:
            self._sticky_fault = fault
        self._owner_state = OwnerState.FAULTED
        self._capture_state = CaptureState.FAULTED


class EvidenceAdmissionViewV1:
    """The narrow, conversation-safe view of one admission owner.

    Lifecycle control remains on the host-owned controller.  This object does
    not use ``__getattr__`` deliberately: its attribute surface is the
    capability boundary, not merely a structural typing convention.
    """

    __slots__ = ("__controller",)

    protocol_version: Literal[1] = 1

    def __init__(self, controller: EvidenceAdmissionControllerV1) -> None:
        if type(controller) is not EvidenceAdmissionControllerV1:
            raise TypeError("admission view requires an exact admission controller")
        self.__controller = controller

    @property
    def operation_scheduler(self) -> ConversationOperationScheduler:
        return self.__controller.operation_scheduler

    def try_admit_command(
        self,
        authority: CommandAdmissionAuthorityV1,
        snapshot: EvidenceSnapshotV1,
    ) -> CommandDisposition:
        return self.__controller.try_admit_command(authority, snapshot)

    def try_reserve_user_turn(
        self,
        authority: UserTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1:
        return self.__controller.try_reserve_user_turn(authority, operation)

    def try_admit_user_final(
        self,
        lease: EvidenceTurnLease,
        authority: UserTurnAuthorityV1,
        text: str,
    ) -> AppendDisposition:
        return self.__controller.try_admit_user_final(lease, authority, text)

    def discard_unopened_user_turn(
        self,
        lease: EvidenceTurnLease,
        authority: UserTurnAuthorityV1,
    ) -> bool:
        return self.__controller.discard_unopened_user_turn(lease, authority)

    def try_admit_generated(
        self,
        lease: EvidenceTurnLease,
        text: str,
    ) -> AppendDisposition:
        return self.__controller.try_admit_generated(lease, text)

    def try_open_non_user_turn(self, lease: EvidenceTurnLease) -> AppendDisposition:
        return self.__controller.try_open_non_user_turn(lease)

    def try_admit_transport_confirmed_full(
        self,
        lease: EvidenceTurnLease,
        *,
        segment_ordinal: int | None,
        synthesis_attempt_id: str,
        transport_attempt_id: str,
        text: str | None,
    ) -> AppendDisposition:
        return self.__controller.try_admit_transport_confirmed_full(
            lease,
            segment_ordinal=segment_ordinal,
            synthesis_attempt_id=synthesis_attempt_id,
            transport_attempt_id=transport_attempt_id,
            text=text,
        )

    def settle_spawn_failed(self, lease: EvidenceTurnLease) -> AppendDisposition:
        return self.__controller.settle_spawn_failed(lease)

    def settle_completed(
        self,
        lease: EvidenceTurnLease,
        *,
        queued_chunk_count: int,
        started_chunk_count: int,
        transport_confirmed_full_count: int,
        assistant_delivery_context_recorded: bool,
    ) -> AppendDisposition:
        return self.__controller.settle_completed(
            lease,
            queued_chunk_count=queued_chunk_count,
            started_chunk_count=started_chunk_count,
            transport_confirmed_full_count=transport_confirmed_full_count,
            assistant_delivery_context_recorded=assistant_delivery_context_recorded,
        )

    def settle_terminal(
        self,
        lease: EvidenceTurnLease,
        *,
        queued_chunk_count: int,
        started_chunk_count: int,
        assistant_delivery_context_recorded: bool,
    ) -> AppendDisposition:
        return self.__controller.settle_terminal(
            lease,
            queued_chunk_count=queued_chunk_count,
            started_chunk_count=started_chunk_count,
            assistant_delivery_context_recorded=assistant_delivery_context_recorded,
        )

    def try_reserve_proactive_turn(
        self,
        authority: ProactiveTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1:
        return self.__controller.try_reserve_proactive_turn(authority, operation)

    def try_reserve_replay_turn(
        self,
        authority: ReplayTurnAuthorityV1,
        operation: ConversationOperationReservation,
    ) -> EvidenceLeaseResultV1:
        return self.__controller.try_reserve_replay_turn(authority, operation)

    def discard_unopened_replay_turn(
        self,
        lease: EvidenceTurnLease,
        authority: ReplayTurnAuthorityV1,
    ) -> bool:
        return self.__controller.discard_unopened_replay_turn(lease, authority)

    def discard_unopened_proactive_turn(
        self,
        lease: EvidenceTurnLease,
        authority: ProactiveTurnAuthorityV1,
    ) -> bool:
        return self.__controller.discard_unopened_proactive_turn(lease, authority)

    def try_append_turn(
        self,
        lease: EvidenceTurnLease,
        snapshot: EvidenceSnapshotV1,
    ) -> AppendDisposition:
        return self.__controller.try_append_turn(lease, snapshot)

    def record_terminal_cause(
        self,
        capability: TerminalCauseCapabilityV1,
        cause: TerminalReason,
    ) -> CauseDisposition:
        return self.__controller.record_terminal_cause(capability, cause)

    def settled_terminal_outcome(
        self,
        lease: EvidenceTurnLease,
    ) -> SettledTerminalOutcomeV1 | None:
        return self.__controller.settled_terminal_outcome(lease)

    def diagnostics(self) -> EvidenceDiagnosticsV1:
        return self.__controller.diagnostics()


def _turn_authority_signature(
    authority: UserTurnAuthorityV1 | ProactiveTurnAuthorityV1 | ReplayTurnAuthorityV1,
) -> tuple[object, ...]:
    common: tuple[object, ...] = (
        type(authority),
        authority.protocol_version,
        authority.owner_generation,
        authority.binding_id,
        authority.binding_generation,
        authority.consent_epoch_id,
        authority.logical_session_id,
    )
    if type(authority) is UserTurnAuthorityV1:
        return common + (
            authority.utterance_id,
            authority.source,
            authority.input_incarnation,
            authority.media_incarnation,
            authority.typed_sequence,
            authority.routing_serial,
            authority.routing_disposition,
        )
    if type(authority) is ProactiveTurnAuthorityV1:
        return common + (authority.proactive_invocation_serial,)
    if type(authority) is ReplayTurnAuthorityV1:
        return common + (
            authority.replay_of_evidence_turn_id,
            authority.replay_generation,
        )
    raise TypeError("turn authority must be exact")


def _invalid_lease_result() -> EvidenceLeaseResultV1:
    return EvidenceLeaseResultV1(
        lease=None,
        disposition=AppendDisposition.INVALID_AUTHORITY,
    )


def _command_disposition(disposition: AppendDisposition) -> CommandDisposition:
    mapping = {
        AppendDisposition.ADMITTED: CommandDisposition.ADMITTED,
        AppendDisposition.DISABLED: CommandDisposition.DISABLED,
        AppendDisposition.CONSENT_MISSING: CommandDisposition.CONSENT_MISSING,
        AppendDisposition.INVALID_AUTHORITY: CommandDisposition.INVALID_AUTHORITY,
        AppendDisposition.SESSION_CLOSING: CommandDisposition.SESSION_CLOSING,
        AppendDisposition.SESSION_TAINTED: CommandDisposition.SESSION_TAINTED,
        AppendDisposition.DROPPED_CAPACITY: CommandDisposition.DROPPED_CAPACITY,
        AppendDisposition.REJECTED_OVERSIZE: CommandDisposition.SESSION_TAINTED,
        AppendDisposition.REJECTED_DENIED: CommandDisposition.SESSION_TAINTED,
        AppendDisposition.QUOTA_EXCEEDED: CommandDisposition.SESSION_TAINTED,
        AppendDisposition.WRITER_FAULT: CommandDisposition.WRITER_FAULT,
    }
    return mapping[disposition]


__all__ = [
    "BoundedEvidenceWriterQueueV1",
    "CREATE_EPOCH_CANONICAL_BYTE_CREDITS",
    "CREATE_EPOCH_RECORD_CREDITS",
    "ConversationOperationReservation",
    "ConversationOperationReservations",
    "ConversationOperationScheduler",
    "EvidenceAdmissionControllerV1",
    "EvidenceAdmissionError",
    "EvidenceAdmissionViewV1",
    "EvidenceLeaseResultV1",
    "EvidenceTerminalReservation",
    "EvidenceTurnLease",
    "EvidenceWriterQueueItemV1",
    "EvidenceWriterSinkV1",
    "MAX_CANONICAL_RECORD_BYTES",
    "MAX_QUEUE_CANONICAL_BYTES",
    "MAX_QUEUE_PHYSICAL_ITEMS",
    "MAX_QUEUE_RECORDS",
    "ReservationError",
    "ROLLOVER_CANONICAL_BYTE_CREDITS",
    "ROLLOVER_RECORD_CREDITS",
    "SESSION_CONTROL_CANONICAL_BYTE_CREDITS",
    "SESSION_CONTROL_RECORD_CREDITS",
    "SessionControlReservation",
    "TERMINAL_CANONICAL_BYTE_CREDITS",
    "TERMINAL_RECORD_CREDITS",
    "WriterQueueLane",
    "canonical_record_bytes",
]
