"""Bounded in-process production observations and qualification trace facts.

The public view deliberately contains metadata only.  Raw model-visible text is
kept in a separate owner-created, in-process qualification trace; neither
surface has a browser, persistence, logging, or configuration path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Event, Lock
from typing import TypeAlias

from hermes_realtime.evidence.models import TerminalDisposition, TerminalReason

_MAX_RECORDS = 256
_CHANNEL_TOKEN = object()


class ObservationKindV1(StrEnum):
    CONTEXT_COMMITTED = "context_committed"
    TERMINAL_SETTLED = "terminal_settled"
    CLOSE_STAGE = "close_stage"
    ROLLOVER = "rollover"


class RolloverStageV1(StrEnum):
    CLAIMED = "claimed"
    QUEUED = "queued"
    DURABLE = "durable"
    PUBLISHED = "published"
    TERMINAL = "terminal"


class RolloverResultV1(StrEnum):
    ACCEPTED = "accepted"
    COMMITTED = "committed"
    IDEMPOTENT = "idempotent"
    REJECTED = "rejected"
    FAULTED = "faulted"
    CLOSED = "closed"


class CloseStageV1(StrEnum):
    BINDING_CLEANUP = "binding_cleanup"
    BROWSER_CLIENT = "browser_client"
    CONSENT_SETTLEMENT = "consent_settlement"
    EPOCH_RETIREMENT = "epoch_retirement"
    EVIDENCE_RUNTIME = "evidence_runtime"
    FOREGROUND_CLOSE = "foreground_close"
    HOST_WORK = "host_work"
    LAUNCHER = "launcher"
    LIVEKIT_WORKER = "livekit_worker"
    RETENTION_CANCELLATION = "retention_cancellation"
    REVOKE_OBSERVER_CANCELLATION = "revoke_observer_cancellation"
    SESSION_WORKER = "session_worker"
    SPEECH_LOOP = "speech_loop"
    TRANSPORT_CLOSE = "transport_close"
    UPDATE_EXECUTOR = "update_executor"
    WRITER_DRAIN = "writer_drain"
    WRITER_STOP = "writer_stop"


class CloseResultV1(StrEnum):
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class ContextCommittedObservationV1:
    """Content-free record that authoritative conversation context committed."""

    context_committed: bool

    def __post_init__(self) -> None:
        if type(self.context_committed) is not bool:
            raise TypeError("context_committed must be an exact built-in bool")

    @property
    def kind(self) -> ObservationKindV1:
        return ObservationKindV1.CONTEXT_COMMITTED


@dataclass(frozen=True, slots=True)
class TerminalSettledObservationV1:
    """One terminal outcome admitted to the writer's ordered terminal batch."""

    terminal_disposition: TerminalDisposition
    terminal_reason: TerminalReason
    context_committed: bool

    def __post_init__(self) -> None:
        if type(self.terminal_disposition) is not TerminalDisposition:
            raise TypeError("terminal disposition must be an exact TerminalDisposition")
        if type(self.terminal_reason) is not TerminalReason:
            raise TypeError("terminal reason must be an exact TerminalReason")
        if type(self.context_committed) is not bool:
            raise TypeError("context_committed must be an exact built-in bool")

    @property
    def kind(self) -> ObservationKindV1:
        return ObservationKindV1.TERMINAL_SETTLED


@dataclass(frozen=True, slots=True)
class CloseStageObservationV1:
    """One immutable ordered attempt for an owner-controlled close stage."""

    stage: CloseStageV1
    result: CloseResultV1

    def __post_init__(self) -> None:
        if type(self.stage) is not CloseStageV1:
            raise TypeError("close stage must be an exact CloseStageV1")
        if type(self.result) is not CloseResultV1:
            raise TypeError("close result must be an exact CloseResultV1")

    @property
    def kind(self) -> ObservationKindV1:
        return ObservationKindV1.CLOSE_STAGE


@dataclass(frozen=True, slots=True)
class RolloverObservationV1:
    stage: RolloverStageV1
    result: RolloverResultV1

    def __post_init__(self) -> None:
        if type(self.stage) is not RolloverStageV1:
            raise TypeError("rollover stage must be exact")
        if type(self.result) is not RolloverResultV1:
            raise TypeError("rollover result must be exact")

    @property
    def kind(self) -> ObservationKindV1:
        return ObservationKindV1.ROLLOVER


@dataclass(frozen=True, slots=True)
class ProductionObservationStatusV1:
    """Content-free integrity state for one bounded observation trace."""

    trace_complete: bool

    def __post_init__(self) -> None:
        if type(self.trace_complete) is not bool:
            raise TypeError("trace_complete must be an exact built-in bool")


ProductionObservationV1: TypeAlias = (
    ContextCommittedObservationV1
    | TerminalSettledObservationV1
    | CloseStageObservationV1
    | RolloverObservationV1
)


@dataclass(slots=True)
class _TraceState:
    records: tuple[object, ...]
    lock: Lock
    incomplete: Event
    max_records: int


class ProductionObservationViewV1:
    """Public metadata-only observation view backed only by opaque callables."""

    __slots__ = ("_records", "_status", "_matches_recorder")

    def __init__(
        self,
        records: Callable[[], tuple[ProductionObservationV1, ...]],
        status: Callable[[], ProductionObservationStatusV1],
        matches_recorder: Callable[[object], bool],
        token: object,
    ) -> None:
        if (
            token is not _CHANNEL_TOKEN
            or not callable(records)
            or not callable(status)
            or not callable(matches_recorder)
        ):
            raise TypeError("production observation views are owner-created")
        self._records = records
        self._status = status
        self._matches_recorder = matches_recorder

    def records(self) -> tuple[ProductionObservationV1, ...]:
        return self._records()

    def status(self) -> ProductionObservationStatusV1:
        return self._status()


class _ProductionObservationRecorderV1:
    """Owner-only recorder capability, never returned from browser composition."""

    __slots__ = ("_append", "_append_close", "_matches")

    def __init__(
        self,
        append: Callable[[ProductionObservationV1], None],
        append_close: Callable[[CloseStageObservationV1], None],
        matches: Callable[[object], bool],
        token: object,
    ) -> None:
        if (
            token is not _CHANNEL_TOKEN
            or not callable(append)
            or not callable(append_close)
            or not callable(matches)
        ):
            raise TypeError("production observation recorders are owner-created")
        self._append = append
        self._append_close = append_close
        self._matches = matches

    def record_context_committed(self) -> None:
        self._append(ContextCommittedObservationV1(context_committed=True))

    def record_terminal_settled(
        self,
        *,
        terminal_disposition: TerminalDisposition,
        terminal_reason: TerminalReason,
        context_committed: bool,
    ) -> None:
        self._append(
            TerminalSettledObservationV1(
                terminal_disposition=terminal_disposition,
                terminal_reason=terminal_reason,
                context_committed=context_committed,
            )
        )

    def record_close_stage(self, *, stage: CloseStageV1, result: CloseResultV1) -> None:
        self._append_close(CloseStageObservationV1(stage=stage, result=result))

    def record_rollover(self, *, stage: RolloverStageV1, result: RolloverResultV1) -> None:
        self._append(RolloverObservationV1(stage=stage, result=result))


def _new_trace_state(*, max_records: int = _MAX_RECORDS) -> _TraceState:
    if type(max_records) is not int or max_records < 1:
        raise ValueError("production observation capacity must be a positive exact integer")
    return _TraceState(
        records=(),
        lock=Lock(),
        incomplete=Event(),
        max_records=max_records,
    )


def _new_bounded_callbacks(
    state: _TraceState,
) -> tuple[
    Callable[[object], None],
    Callable[[], tuple[object, ...]],
    Callable[[], ProductionObservationStatusV1],
]:
    """Return nonblocking append and immutable snapshot callables for one trace."""

    def append(record: object) -> None:
        # Foreground producers must never wait for a diagnostic reader or writer.
        # A contended lock or capacity exhaustion is an explicit incomplete trace,
        # never an eviction of earlier facts.
        if not state.lock.acquire(blocking=False):
            # Keep loss independent of the record tuple so a lock holder which
            # observed the old state can never publish it as complete again.
            state.incomplete.set()
            return
        try:
            if len(state.records) >= state.max_records:
                state.incomplete.set()
                return
            if type(record) is RolloverObservationV1:
                rollover = record
                prior = tuple(
                    item for item in state.records if type(item) is RolloverObservationV1
                )
                stages = list(RolloverStageV1)
                prior_index = -1
                if prior and prior[-1].stage is not RolloverStageV1.TERMINAL:
                    prior_index = stages.index(prior[-1].stage)
                valid = (
                    rollover.stage is RolloverStageV1.CLAIMED
                    if prior_index == -1
                    else stages.index(rollover.stage) > prior_index
                )
                if not valid:
                    state.incomplete.set()
                    return
            state.records = (*state.records, record)
        finally:
            state.lock.release()

    def records() -> tuple[object, ...]:
        with state.lock:
            return state.records

    def status() -> ProductionObservationStatusV1:
        return ProductionObservationStatusV1(trace_complete=not state.incomplete.is_set())

    return append, records, status


def _new_observation_channel(*, max_records: int = _MAX_RECORDS) -> tuple[
    ProductionObservationViewV1,
    _ProductionObservationRecorderV1,
]:
    state = _new_trace_state(max_records=max_records)
    append, records, status = _new_bounded_callbacks(state)
    identity = object()

    def append_close(record: CloseStageObservationV1) -> None:
        if not state.lock.acquire(blocking=False):
            state.incomplete.set()
            return
        try:
            if len(state.records) >= state.max_records:
                state.incomplete.set()
                return
            # Close results are immutable attempt facts.  A later retry may
            # derive a current final state, but must never rewrite the failed
            # attempt that preceded it; Task 11 compares this exact ordering.
            state.records = (*state.records, record)
        finally:
            state.lock.release()

    recorder = _ProductionObservationRecorderV1(
        append=append,
        append_close=append_close,
        matches=lambda candidate: candidate is identity,
        token=_CHANNEL_TOKEN,
    )
    return (
        ProductionObservationViewV1(
            records=records,  # type: ignore[arg-type]
            status=status,
            matches_recorder=lambda candidate: (
                type(candidate) is _ProductionObservationRecorderV1 and candidate._matches(identity)
            ),
            token=_CHANNEL_TOKEN,
        ),
        recorder,
    )


__all__ = [
    "CloseResultV1",
    "CloseStageObservationV1",
    "CloseStageV1",
    "ContextCommittedObservationV1",
    "ObservationKindV1",
    "ProductionObservationStatusV1",
    "ProductionObservationV1",
    "ProductionObservationViewV1",
    "TerminalSettledObservationV1",
]
