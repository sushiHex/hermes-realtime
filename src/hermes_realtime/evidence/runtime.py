"""Owned runtime dispatch from bounded evidence admission to persistent transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ThreadFuture
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Protocol, cast
from uuid import uuid4

from hermes_realtime._qualification import (
    _current_qualification_capacity_probe,
    _current_qualification_checkpoint_channel,
    _qualification_capacity_probe_scope,
)
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    ProductionObservationViewV1,
    RolloverResultV1,
    RolloverStageV1,
    _new_observation_channel,
    _ProductionObservationRecorderV1,
)

from .admission import (
    _ROLLOVER_RUNTIME_OWNER_TOKEN,
    BoundedEvidenceWriterQueueV1,
    ConversationOperationScheduler,
    CreateEpochTicketV1,
    DrainTicketV1,
    EvidenceAdmissionControllerV1,
    EvidenceAdmissionViewV1,
    EvidenceWriterQueueItemV1,
    ReservationError,
    RevokeTicketV1,
    _AdmissionRolloverPreparationV1,
    _ticket_signal_status,
)
from .lifecycle import (
    EvidenceConversationAuthorityV1,
    EvidenceLifecycleOwner,
    _LifecycleRolloverPreparationV1,
)
from .models import (
    CONSENT_VERSION,
    AppendDisposition,
    BindingCloseAuthorityV1,
    BindingClosedPayloadV1,
    BindingCloseReason,
    BindingCloseV1,
    BindingOpenedPayloadV1,
    CaptureState,
    CaptureStatusV1,
    ConsentCreateAuthorityV1,
    ConsentDisposition,
    ConsentRevokeAuthorityV1,
    CreateEpochReservationV1,
    CreateEpochV1,
    DrainAndStopV1,
    DrainDisposition,
    EventKind,
    EvidenceConsentRequestV1,
    EvidenceRevokeRequestV1,
    EvidenceSnapshotV1,
    ExpireSessionV1,
    ExpiryDisposition,
    ExpiryMode,
    LifecycleDrainAuthorityV1,
    ProjectionReservation,
    PurgeDisposition,
    QueuedEvidenceRecordV1,
    RevokeDisposition,
    RevokeFinalizeV1,
    RevokeRequestV1,
    RolloverDisposition,
    RolloverSessionV1,
    SealEpochV1,
    SessionExpiryAuthorityV1,
    SessionOpenedPayloadV1,
    SessionSealRequestedPayloadV1,
    StoreDisposition,
    validate_authority_composition,
)

_CLOSE_DRAIN_TIMEOUT_SECONDS = 2.0
_CLOSE_WRITER_TIMEOUT_SECONDS = 2.0
_THREAD_SIGNAL_INITIAL_POLL_SECONDS = 0.001
_THREAD_SIGNAL_MAX_POLL_SECONDS = 0.05


async def _wait_for_thread_signal(event: Event, *, timeout_seconds: float | None) -> bool:
    """Observe a writer-thread signal without allocating bridge workers.

    ``Event.is_set`` is thread-safe and non-blocking.  Cooperative polling keeps
    timeout and cancellation ownership entirely on this event loop, without an
    executor queue or a background thread that can outlive the awaiting task.
    Admission records each production milestone's monotonic signal time, so a
    pre-deadline signal survives event-loop delay while a late signal is refused.
    Short initial polls preserve writer-milestone latency; bounded exponential
    backoff prevents retained observers from continuously churning the loop.
    """

    if type(event) is not Event:
        raise TypeError("event must be an exact threading Event")
    if timeout_seconds is not None:
        if type(timeout_seconds) not in (int, float) or float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be a positive number or None")
        deadline = monotonic() + float(timeout_seconds)
    else:
        deadline = None
    poll_seconds = _THREAD_SIGNAL_INITIAL_POLL_SECONDS
    while True:
        signal_status = _ticket_signal_status(event, deadline=deadline)
        if signal_status is not None:
            return signal_status
        if deadline is None:
            wait_seconds = poll_seconds
        else:
            remaining = deadline - monotonic()
            if remaining <= 0:
                signal_status = _ticket_signal_status(event, deadline=deadline)
                return False if signal_status is None else signal_status
            wait_seconds = min(poll_seconds, remaining)
        await asyncio.sleep(wait_seconds)
        poll_seconds = min(_THREAD_SIGNAL_MAX_POLL_SECONDS, poll_seconds * 2)


class _WriterTransportV1(Protocol):
    def create_epoch(self, command: CreateEpochV1) -> StoreDisposition: ...

    def append_record(self, item: QueuedEvidenceRecordV1) -> StoreDisposition: ...

    def append_binding_close(self, command: BindingCloseV1) -> StoreDisposition: ...

    def rollover_session(self, command: RolloverSessionV1) -> StoreDisposition: ...

    def expire_session(self, command: ExpireSessionV1) -> PurgeDisposition: ...

    def seal_epoch(self, command: SealEpochV1) -> StoreDisposition: ...

    def commit_revoke_request(self, command: RevokeRequestV1) -> RevokeDisposition: ...

    def finalize_revoke(self, command: RevokeFinalizeV1) -> RevokeDisposition: ...

    def drain_and_close(self, command: DrainAndStopV1) -> DrainDisposition: ...


_WRITER_TRANSPORT_METHODS = (
    "create_epoch",
    "append_record",
    "append_binding_close",
    "rollover_session",
    "expire_session",
    "seal_epoch",
    "commit_revoke_request",
    "finalize_revoke",
    "drain_and_close",
)

_STORE_DISPOSITIONS = frozenset(StoreDisposition)
_PURGE_DISPOSITIONS = frozenset(PurgeDisposition)
_REVOKE_REQUEST_DISPOSITIONS = frozenset(
    (
        RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
        RevokeDisposition.ALREADY_SCHEDULED,
    )
)
_REVOKE_FINALIZE_DISPOSITIONS = frozenset(
    (
        RevokeDisposition.PURGE_COMPLETED,
        RevokeDisposition.PURGE_FAILED,
    )
)
_DRAIN_DISPOSITIONS = frozenset(DrainDisposition)
_CREATE_DURABLE_DISPOSITIONS = frozenset(
    (
        StoreDisposition.COMMITTED,
        StoreDisposition.IDEMPOTENT,
    )
)


def _validate_writer_transport_v1(transport: object) -> None:
    """Reject malformed injected transports before they can acquire authority."""

    for method in _WRITER_TRANSPORT_METHODS:
        if not callable(getattr(transport, method, None)):
            raise TypeError(f"transport must provide {method}()")


def _checked_binding_is_current(
    binding_is_current: Callable[[CreateEpochV1], bool],
    command: CreateEpochV1,
) -> bool:
    """Call one binding boundary and require its exact boolean decision."""

    current = binding_is_current(command)
    if type(current) is not bool:
        raise TypeError("binding_is_current must return an exact built-in bool")
    return current


class EvidenceWriterDispatcherV1:
    """Dispatch exact bounded-queue items and complete admission accounting."""

    def __init__(
        self,
        *,
        source: BoundedEvidenceWriterQueueV1,
        admission: EvidenceAdmissionControllerV1,
        transport: _WriterTransportV1,
        binding_is_current: Callable[[CreateEpochV1], bool],
        rollover_completion: (
            Callable[[EvidenceWriterQueueItemV1, StoreDisposition], None] | None
        ) = None,
    ) -> None:
        if type(source) is not BoundedEvidenceWriterQueueV1:
            raise TypeError("source must be an exact bounded evidence writer queue")
        if type(admission) is not EvidenceAdmissionControllerV1:
            raise TypeError("admission must be an exact V1 controller")
        _validate_writer_transport_v1(transport)
        if not callable(binding_is_current):
            raise TypeError("binding_is_current must be callable")
        self._source = source
        self._admission = admission
        self._transport = transport
        self._binding_is_current = binding_is_current
        self._rollover_completion = rollover_completion

    def dispatch_one(self) -> bool:
        try:
            item = self._source.get_nowait()
        except Empty:
            return False
        self.dispatch_item(item)
        return True

    def dispatch_item(self, item: EvidenceWriterQueueItemV1) -> None:
        """Dispatch one exact item already dequeued by the runtime owner."""

        if type(item) is not EvidenceWriterQueueItemV1:
            raise TypeError("item must be an exact evidence writer queue item")
        payload = item.payload
        if type(payload) is CreateEpochV1:
            try:
                disposition = self._transport.create_epoch(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                disposition,
                StoreDisposition,
                _STORE_DISPOSITIONS,
            ):
                return
            if disposition in _CREATE_DURABLE_DISPOSITIONS:
                try:
                    binding_current = _checked_binding_is_current(
                        self._binding_is_current,
                        payload,
                    )
                except BaseException as error:
                    self._complete_transport_failure(item)
                    self._reraise_system_transport_exception(error)
                    return
            else:
                binding_current = False
            self._admission.complete_create_epoch(
                item,
                disposition=disposition,
                binding_current=binding_current,
            )
            return
        if type(payload) is BindingCloseV1:
            try:
                disposition = self._transport.append_binding_close(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                disposition,
                StoreDisposition,
                _STORE_DISPOSITIONS,
            ):
                return
            self._admission.complete_ordered_item(
                item,
                writer_succeeded=disposition
                in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT),
            )
            return
        if type(payload) is RolloverSessionV1:
            try:
                disposition = self._transport.rollover_session(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                disposition,
                StoreDisposition,
                _STORE_DISPOSITIONS,
            ):
                return
            completion = getattr(self, "_rollover_completion", None)
            if completion is None:
                self._admission.complete_rollover(item, disposition)
            else:
                completion(item, disposition)
            return
        if type(payload) is ExpireSessionV1:
            try:
                purge_disposition = self._transport.expire_session(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                purge_disposition,
                PurgeDisposition,
                _PURGE_DISPOSITIONS,
            ):
                return
            self._admission.complete_ordered_item(
                item,
                writer_succeeded=purge_disposition
                in (PurgeDisposition.PURGE_COMPLETED, PurgeDisposition.ALREADY_ABSENT),
            )
            return
        if type(payload) is SealEpochV1:
            try:
                disposition = self._transport.seal_epoch(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                disposition,
                StoreDisposition,
                _STORE_DISPOSITIONS,
            ):
                return
            self._admission.complete_ordered_item(
                item,
                writer_succeeded=disposition
                in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT),
            )
            return
        if type(payload) is RevokeRequestV1:
            try:
                revoke_disposition = self._transport.commit_revoke_request(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                revoke_disposition,
                RevokeDisposition,
                _REVOKE_REQUEST_DISPOSITIONS,
            ):
                return
            self._admission.complete_revoke_request(item, revoke_disposition)
            return
        if type(payload) is RevokeFinalizeV1:
            try:
                revoke_disposition = self._transport.finalize_revoke(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                revoke_disposition,
                RevokeDisposition,
                _REVOKE_FINALIZE_DISPOSITIONS,
            ):
                return
            self._admission.complete_revoke_finalize(item, revoke_disposition)
            return
        if type(payload) is DrainAndStopV1:
            try:
                drain_disposition = self._transport.drain_and_close(payload)
            except BaseException as error:
                self._complete_transport_failure(item)
                self._reraise_system_transport_exception(error)
                return
            if self._complete_disallowed_transport_disposition(
                item,
                drain_disposition,
                DrainDisposition,
                _DRAIN_DISPOSITIONS,
            ):
                return
            self._admission.complete_drain(item, drain_disposition)
            return
        if type(payload) is not QueuedEvidenceRecordV1:
            raise TypeError("writer dispatcher does not support this payload type")
        try:
            disposition = self._transport.append_record(payload)
        except BaseException as error:
            self._complete_transport_failure(item)
            self._reraise_system_transport_exception(error)
            return
        if self._complete_disallowed_transport_disposition(
            item,
            disposition,
            StoreDisposition,
            _STORE_DISPOSITIONS,
        ):
            return
        self._admission.complete_ordered_item(
            item,
            writer_succeeded=disposition
            in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT),
        )

    def _complete_disallowed_transport_disposition(
        self,
        item: EvidenceWriterQueueItemV1,
        result: object,
        expected_type: type[Any],
        allowed_members: frozenset[Any],
    ) -> bool:
        """Fault-complete a dequeued item whose disposition breaks its branch domain.

        Transport implementations are injected runtime boundaries.  A return of
        the wrong exact type or an exact but disallowed enum member has the same
        terminal accounting and continuation behavior as an ordinary transport
        exception: admission, not the transport, chooses the matching fault
        disposition and releases the item's owned credits/ticket exactly once.
        """

        if type(result) is expected_type and result in allowed_members:
            return False
        self._complete_transport_failure(item)
        return True

    def _complete_transport_failure(self, item: EvidenceWriterQueueItemV1) -> None:
        """Complete the exact dequeued item after a transport boundary fault.

        A transport implementation is an injected blocking boundary, not an
        authority owner.  Its exception must therefore become the matching
        admission-owned terminal/fault disposition before the writer can move
        on or a runtime can attempt its one drain.  In particular, a create
        completion releases both its physical item and retained create credits
        and signals its exact consent ticket.
        """

        payload = item.payload
        if type(payload) is CreateEpochV1:
            self._admission.complete_create_epoch(
                item,
                disposition=StoreDisposition.FAULTED,
                binding_current=False,
            )
            return
        if type(payload) in (
            BindingCloseV1,
            ExpireSessionV1,
            QueuedEvidenceRecordV1,
            SealEpochV1,
        ):
            self._admission.complete_ordered_item(item, writer_succeeded=False)
            return
        if type(payload) is RolloverSessionV1:
            completion = getattr(self, "_rollover_completion", None)
            if completion is None:
                self._admission.complete_rollover(item, StoreDisposition.FAULTED)
            else:
                completion(item, StoreDisposition.FAULTED)
            return
        if type(payload) is RevokeRequestV1:
            self._admission.complete_revoke_request(item, RevokeDisposition.WRITER_FAULT)
            return
        if type(payload) is RevokeFinalizeV1:
            self._admission.complete_revoke_finalize(item, RevokeDisposition.PURGE_FAILED)
            return
        if type(payload) is DrainAndStopV1:
            self._admission.complete_drain(item, DrainDisposition.WRITER_FAULT)
            return
        raise TypeError("writer dispatcher does not support this payload type")

    @staticmethod
    def _reraise_system_transport_exception(error: BaseException) -> None:
        """Preserve process-level termination after exact accounting.

        ``CancelledError`` from a transport cannot cancel the dedicated writer
        thread, so it is treated like every other transport fault.  Process
        control exceptions retain their normal propagation after the exact
        item has been completed.
        """

        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error


class EvidenceWriterRuntimeOwnerV1:
    """Single blocking consumer for the bounded evidence writer ingress."""

    THREAD_NAME = "hermes-evidence-dispatcher"

    def __init__(
        self,
        *,
        source: BoundedEvidenceWriterQueueV1,
        dispatch: Callable[[EvidenceWriterQueueItemV1], None],
    ) -> None:
        if type(source) is not BoundedEvidenceWriterQueueV1:
            raise TypeError("source must be an exact bounded evidence writer queue")
        if not callable(dispatch):
            raise TypeError("dispatch must be callable")
        self._source = source
        self._dispatch = dispatch
        self._close_lock = Lock()
        self._closed = False
        self._failure: BaseException | None = None
        self._thread = Thread(target=self._run, name=self.THREAD_NAME, daemon=True)
        self._thread.start()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def _run(self) -> None:
        try:
            while True:
                item = self._source.get_blocking()
                if item is None:
                    return
                self._dispatch(item)
        except BaseException as error:
            self._failure = error
            self._source.stop_consumer()

    def close(self, timeout: float | None = None) -> bool:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self._source.stop_consumer()
        self._thread.join(timeout)
        return not self._thread.is_alive()


class HostEvidenceRuntimeV1:
    """Host-owned dormant evidence runtime before durable consent activation."""

    def __init__(
        self,
        *,
        database: Path,
        owner_generation: int,
        retention_hours: int,
        production_observations: ProductionObservationViewV1 | None = None,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
    ) -> None:
        if not isinstance(database, Path):
            raise TypeError("database must be a pathlib Path")
        if type(owner_generation) is not int or not 1 <= owner_generation <= 2**63 - 1:
            raise ValueError("owner_generation must be an exact positive 63-bit integer")
        if type(retention_hours) is not int or not 1 <= retention_hours <= 168:
            raise ValueError("retention_hours must be an exact integer from 1 through 168")
        if (
            production_observations is not None
            and type(production_observations) is not ProductionObservationViewV1
        ):
            raise TypeError("production observations must be an exact view or None")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        if (production_observations is None) != (production_observation_recorder is None):
            raise ValueError("production observation view and recorder must be paired")
        if production_observations is None:
            production_observations, production_observation_recorder = _new_observation_channel()
        else:
            assert production_observation_recorder is not None
            if not production_observations._matches_recorder(production_observation_recorder):
                raise ValueError("production observation view and recorder must share one owner")
        assert production_observation_recorder is not None
        self._database = database
        self._qualification_checkpoint_channel = _current_qualification_checkpoint_channel()
        self._qualification_drain_checkpoint_emitted = False
        self._qualification_capacity_probe = _current_qualification_capacity_probe()
        self._owner_generation = owner_generation
        self._retention_hours = retention_hours
        self._production_observations = production_observations
        self._production_observation_recorder = production_observation_recorder
        self._capture_state = CaptureState.IDLE
        self._operation_scheduler = ConversationOperationScheduler(
            owner_generation=owner_generation,
            max_operations=16,
        )
        self._lifecycle_owner = EvidenceLifecycleOwner(
            owner_generation=owner_generation,
            operation_scheduler=self._operation_scheduler,
        )
        self._queue: BoundedEvidenceWriterQueueV1 | None = None
        self._admission: EvidenceAdmissionControllerV1 | None = None
        self._admission_view: EvidenceAdmissionViewV1 | None = None
        self._writer: EvidenceWriterRuntimeOwnerV1 | None = None
        self._pending_create: CreateEpochV1 | None = None
        self._pending_consent_authority: ConsentCreateAuthorityV1 | None = None
        self._pending_consent_ticket: CreateEpochTicketV1 | None = None
        self._pending_revoke_authority: ConsentRevokeAuthorityV1 | None = None
        self._pending_revoke_ticket: RevokeTicketV1 | None = None
        self._revoke_reservation_lock = Lock()
        self._revoke_observers: set[asyncio.Task[None]] = set()
        self._revoke_observer_failures: list[BaseException] = []
        self._retention_cancel: Callable[[], Awaitable[None]] | None = None
        self._retention_wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC)
        self._retention_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._retention_task: asyncio.Task[ExpiryDisposition] | None = None
        self._transport: _WriterTransportV1 | None = None
        self._close_drain_authority: LifecycleDrainAuthorityV1 | None = None
        self._close_drain_ticket: DrainTicketV1 | None = None
        self._lifecycle_status_reservations: list[ProjectionReservation] = []
        self._release_lifecycle_status_reservations: (
            Callable[[tuple[ProjectionReservation, ...]], None] | None
        ) = None
        self._pending_consent_projection_authority: ConsentCreateAuthorityV1 | None = None
        self._pending_consent_projection_reservation: ProjectionReservation | None = None
        self._release_pending_consent_projection: (
            Callable[[tuple[ProjectionReservation, ...]], None] | None
        ) = None
        self._consent_settlement_authority: ConsentCreateAuthorityV1 | None = None
        self._consent_settlement_operation: asyncio.Task[ConsentDisposition] | None = None
        self._consent_settlement_claimed = False
        self._consent_activation_authority: ConsentCreateAuthorityV1 | None = None
        self._consent_activation_operation: asyncio.Task[Any] | None = None
        self._published = False
        self._closed = False
        self._closing = False
        self._epoch_retiring = False
        self._state_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        self._invalidation_authority: BindingCloseAuthorityV1 | None = None
        self._invalidation_operation: asyncio.Task[None] | None = None
        self._rollover_tasks: set[asyncio.Task[None]] = set()
        self._rollover_lock = Lock()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._rollover_handoffs: dict[
            ThreadFuture[None], _AdmissionRolloverPreparationV1
        ] = {}
        self._rollover_handoffs_closed = False
        self._active_rollover: (
            tuple[
                _AdmissionRolloverPreparationV1,
                _LifecycleRolloverPreparationV1,
            ]
            | None
        ) = None
        self._completed_rollovers: dict[EvidenceWriterQueueItemV1, StoreDisposition] = {}
        self._rollover_completion_claimed: set[EvidenceWriterQueueItemV1] = set()

    @property
    def production_observations(self) -> ProductionObservationViewV1:
        return self._production_observations

    @property
    def operation_scheduler(self) -> ConversationOperationScheduler:
        return self._operation_scheduler

    @property
    def evidence_lifecycle(self) -> EvidenceConversationAuthorityV1 | None:
        return self._lifecycle_owner.conversation_authority if self._published else None

    @property
    def conversation_authority(self) -> EvidenceConversationAuthorityV1:
        """Return only the immutable conversation-scoped lifecycle facade."""

        return self._lifecycle_owner.conversation_authority

    @property
    def evidence_admission(self) -> EvidenceAdmissionViewV1 | None:
        return self._admission_view if self._published else None

    @property
    def writer_running(self) -> bool:
        writer = self._writer
        return writer is not None and writer.is_running

    def resolve_evidence_pair(
        self,
    ) -> tuple[EvidenceConversationAuthorityV1 | None, EvidenceAdmissionViewV1 | None]:
        return self.evidence_lifecycle, self.evidence_admission

    def capture_status(self, *, disclosure_digest: str) -> CaptureStatusV1:
        """Return the host-owned capture status for browser projection."""

        closed = self._closed
        return CaptureStatusV1(
            available=not closed,
            capture_state=CaptureState.UNAVAILABLE if closed else self._capture_state,
            retention_hours=self._retention_hours,
            consent_version=CONSENT_VERSION,
            disclosure_digest=disclosure_digest,
        )

    async def invalidate_active_binding(self, reason: BindingCloseReason) -> None:
        """Fail closed before a browser binding or media incarnation is replaced.

        The conversation worker remains available, but no activated evidence capability
        survives the old browser/media authority.  A later binding therefore needs a
        fresh consent operation before it can receive any capture authority.
        """

        if type(reason) is not BindingCloseReason:
            raise TypeError("binding invalidation reason must be exact")
        await self._retire_current_epoch(reason=reason)

    async def _retire_current_epoch(
        self,
        *,
        reason: BindingCloseReason | None,
    ) -> None:
        """Coalesce retirement of one epoch without awaiting under the state gate."""

        async with self._state_lock:
            if self._closed or self._closing:
                return
            if (
                self._writer is None
                and self._pending_consent_ticket is None
                and not self._published
            ):
                return
            self._epoch_retiring = True
            if reason is not None and self._published and self._invalidation_authority is None:
                admission = self._admission
                if admission is None:
                    raise RuntimeError("published evidence binding has no admission authority")
                authority = self._lifecycle_owner.close_binding(reason)
                disposition = admission.invalidate_binding(authority)
                if disposition not in (
                    AppendDisposition.SESSION_TAINTED,
                    AppendDisposition.SESSION_CLOSING,
                ):
                    raise RuntimeError("binding invalidation did not fail closed")
                self._invalidation_authority = authority
                self._published = False
                self._capture_state = CaptureState.IDLE
            operation = self._invalidation_operation
            if operation is None or (
                operation.done() and (operation.cancelled() or operation.exception() is not None)
            ):
                operation = asyncio.create_task(
                    self._retire_invalidated_epoch(),
                    name="host-evidence-runtime-epoch-retirement",
                )
                self._invalidation_operation = operation
        await asyncio.shield(operation)

    async def _retire_invalidated_epoch(self) -> None:
        """Stop the tainted owner before a later consent creates a fresh epoch."""

        retention = self._retention_task
        self._retention_task = None
        if retention is not None and not retention.done():
            retention.cancel()
            await asyncio.gather(retention, return_exceptions=True)
        writer = self._writer
        if writer is not None:
            await self._drain_before_writer_stop()
            stopped = await asyncio.to_thread(writer.close, 2.0)
            if not stopped:
                raise RuntimeError("evidence writer did not stop before close timeout")
            self._writer = None
        transport = self._transport
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            closed = await asyncio.to_thread(close_transport)
            if closed is False:
                raise RuntimeError("evidence transport close reported failure")
        self._transport = None
        self._lifecycle_owner.retire_drain_lineage()
        async with self._state_lock:
            self._queue = None
            self._admission = None
            self._admission_view = None
            self._pending_create = None
            self._close_drain_authority = None
            self._close_drain_ticket = None
            self._release_retained_lifecycle_status_reservations()
            self._published = False
            self._capture_state = CaptureState.IDLE
            self._epoch_retiring = False
            self._invalidation_authority = None
            self._invalidation_operation = None
            # A timeout owner may have finished rollback before its gateway has a
            # chance to claim the exact retained task.  Do not erase that handoff
            # capability until it has been claimed; a later reservation must fail
            # closed rather than replace an unobserved settlement.
            if self._consent_settlement_claimed:
                self._consent_settlement_authority = None
                self._consent_settlement_operation = None
                self._consent_settlement_claimed = False
            self._pending_evidence_consent_key_reset()

    def _pending_evidence_consent_key_reset(self) -> None:
        """Discard only in-memory activation authority after an external close."""

        self._pending_consent_authority = None
        self._pending_consent_ticket = None
        self._pending_revoke_authority = None
        self._pending_revoke_ticket = None

    def create_sqlite_transport(self) -> _WriterTransportV1:
        """Create the real lazy single-owner SQLite transport after accepted consent."""

        from .sqlite_spool import SQLiteEvidenceSpool, SQLiteEvidenceWriterDaemonV1

        return SQLiteEvidenceWriterDaemonV1(
            lambda: SQLiteEvidenceSpool(
                self._database,
                clock=self._retention_wall_clock,
                owner_generation=self._owner_generation,
            )
        )

    def _rollover_claimed(self, preparation: _AdmissionRolloverPreparationV1) -> None:
        """Attach the admission winner to this runtime's owned asyncio topology."""

        handoff: ThreadFuture[None] = ThreadFuture()
        with self._rollover_lock:
            loop = self._owner_loop
            if (
                loop is None
                or loop.is_closed()
                or self._rollover_handoffs_closed
                or self._closing
                or self._closed
            ):
                self._production_observation_recorder.record_rollover(
                    stage=RolloverStageV1.CLAIMED,
                    result=RolloverResultV1.ACCEPTED,
                )
                self._record_rollover_terminal(RolloverResultV1.REJECTED)
                raise RuntimeError("capacity rollover owner loop is unavailable")
            self._rollover_handoffs[handoff] = preparation
            try:
                loop.call_soon_threadsafe(
                    self._accept_rollover_handoff,
                    preparation,
                    handoff,
                )
            except BaseException:
                self._rollover_handoffs.pop(handoff, None)
                self._production_observation_recorder.record_rollover(
                    stage=RolloverStageV1.CLAIMED,
                    result=RolloverResultV1.ACCEPTED,
                )
                self._record_rollover_terminal(RolloverResultV1.REJECTED)
                raise

    def _accept_rollover_handoff(
        self,
        preparation: _AdmissionRolloverPreparationV1,
        handoff: ThreadFuture[None],
    ) -> None:
        """Create the retained task only on the loop captured by activation."""

        with self._rollover_lock:
            if self._rollover_handoffs.get(handoff) is not preparation:
                return
            try:
                task = asyncio.create_task(
                    self._queue_capacity_rollover(preparation),
                    name="host-evidence-capacity-rollover",
                )
                self._rollover_tasks.add(task)
                self._rollover_handoffs.pop(handoff, None)
                task.add_done_callback(self._discard_rollover_task)
                self._production_observation_recorder.record_rollover(
                    stage=RolloverStageV1.CLAIMED,
                    result=RolloverResultV1.ACCEPTED,
                )
            except BaseException as error:
                self._rollover_handoffs.pop(handoff, None)
                admission = self._admission
                if (
                    admission is not None
                    and preparation.item is None
                    and not preparation.terminal
                ):
                    admission.abort_prepared_rollover(preparation)
                self._production_observation_recorder.record_rollover(
                    stage=RolloverStageV1.CLAIMED,
                    result=RolloverResultV1.ACCEPTED,
                )
                self._record_rollover_terminal(RolloverResultV1.REJECTED)
                if not handoff.done():
                    handoff.set_exception(error)
                return
            if not handoff.done():
                handoff.set_result(None)

    def _discard_rollover_task(self, task: asyncio.Task[None]) -> None:
        with self._rollover_lock:
            self._rollover_tasks.discard(task)

    async def _queue_capacity_rollover(self, preparation: _AdmissionRolloverPreparationV1) -> None:
        admission = self._admission
        create = self._pending_create
        if admission is None or create is None or self._closing or self._closed:
            if admission is not None:
                admission.abort_prepared_rollover(preparation)
            self._record_rollover_terminal(RolloverResultV1.CLOSED)
            return
        successor = str(uuid4())
        expires = (
            self._retention_wall_clock().astimezone(UTC) + timedelta(hours=self._retention_hours)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        lifecycle: _LifecycleRolloverPreparationV1 | None = None
        try:
            lifecycle = self._lifecycle_owner.prepare_rollover(
                successor_logical_session_id=successor,
                successor_expires_at_utc=expires,
                reason=BindingCloseReason.CAPACITY_ROLLOVER,
            )
            authority = self._lifecycle_owner.rollover_authority(lifecycle)
            ordinal, final_sequence = admission.prepared_rollover_coordinates(preparation)
            close_sequence = final_sequence - 1
            snapshots = (
                EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=create.installation_id,
                    producer_instance_id=create.producer_instance_id,
                    logical_session_id=authority.predecessor_logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=close_sequence,
                    event_kind=EventKind.BINDING_CLOSED,
                    payload=BindingClosedPayloadV1(
                        binding_id=create.binding_id,
                        close_reason=BindingCloseReason.CAPACITY_ROLLOVER,
                    ),
                ),
                EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=create.installation_id,
                    producer_instance_id=create.producer_instance_id,
                    logical_session_id=authority.predecessor_logical_session_id,
                    event_id=str(uuid4()),
                    event_sequence=final_sequence,
                    event_kind=EventKind.SESSION_SEAL_REQUESTED,
                    payload=SessionSealRequestedPayloadV1(
                        final_event_sequence=final_sequence,
                        consent_epoch_id=create.consent_epoch_id,
                        consent_version=create.consent_version,
                        disclosure_digest=create.disclosure_digest,
                    ),
                ),
                EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=create.installation_id,
                    producer_instance_id=create.producer_instance_id,
                    logical_session_id=successor,
                    event_id=str(uuid4()),
                    event_sequence=1,
                    event_kind=EventKind.SESSION_OPENED,
                    payload=SessionOpenedPayloadV1(
                        consent_epoch_id=create.consent_epoch_id,
                        binding_id=create.binding_id,
                        consent_version=create.consent_version,
                        disclosure_digest=create.disclosure_digest,
                        retention_hours=create.retention_hours,
                        microphone_accepted=create.microphone_accepted,
                        typed_accepted=create.typed_accepted,
                        predecessor_session_id=authority.predecessor_logical_session_id,
                    ),
                ),
                EvidenceSnapshotV1(
                    schema_version=1,
                    installation_id=create.installation_id,
                    producer_instance_id=create.producer_instance_id,
                    logical_session_id=successor,
                    event_id=str(uuid4()),
                    event_sequence=2,
                    event_kind=EventKind.BINDING_OPENED,
                    payload=BindingOpenedPayloadV1(
                        binding_id=create.binding_id,
                        binding_generation=create.binding_generation,
                        microphone_available=cast(
                            BindingOpenedPayloadV1, create.binding_opened.payload
                        ).microphone_available,
                        typed_available=cast(
                            BindingOpenedPayloadV1, create.binding_opened.payload
                        ).typed_available,
                    ),
                ),
            )
            command = RolloverSessionV1(
                protocol_version=1,
                consent_epoch_id=create.consent_epoch_id,
                binding_id=create.binding_id,
                binding_generation=create.binding_generation,
                predecessor_logical_session_id=authority.predecessor_logical_session_id,
                successor_logical_session_id=successor,
                predecessor_final_event_sequence=final_sequence,
                successor_expires_at_utc=expires,
                admission_ordinal=ordinal,
                snapshots=snapshots,
            )
            with self._rollover_lock:
                self._active_rollover = (preparation, lifecycle)
                result = admission.enqueue_prepared_rollover(preparation, authority, command)
                if result is RolloverDisposition.ROLLOVER_QUEUED:
                    self._production_observation_recorder.record_rollover(
                        stage=RolloverStageV1.QUEUED,
                        result=RolloverResultV1.ACCEPTED,
                    )
            if result is not RolloverDisposition.ROLLOVER_QUEUED:
                probe = self._qualification_capacity_probe
                if probe is not None:
                    await probe.pause_after_rollover_rejection()
                self._lifecycle_owner.abort_prepared_rollover(lifecycle)
                failed_disposition = (
                    StoreDisposition.FAULTED
                    if result is RolloverDisposition.WRITER_FAULT
                    else StoreDisposition.REJECTED_STATE
                )
                admission.finalize_failed_rollover_completion(
                    preparation,
                    failed_disposition,
                )
                with self._rollover_lock:
                    self._active_rollover = None
                self._record_rollover_terminal(RolloverResultV1.REJECTED)
                return
        except ReservationError:
            with self._rollover_lock:
                self._active_rollover = None
            if lifecycle is not None:
                self._lifecycle_owner.abort_prepared_rollover(lifecycle)
            if preparation.item is None and not preparation.terminal:
                admission.abort_prepared_rollover(preparation)
            self._record_rollover_terminal(RolloverResultV1.REJECTED)
        except BaseException:
            with self._rollover_lock:
                self._active_rollover = None
            if lifecycle is not None:
                self._lifecycle_owner.abort_prepared_rollover(lifecycle)
            if preparation.item is None and not preparation.terminal:
                admission.abort_prepared_rollover(preparation)
            self._record_rollover_terminal(RolloverResultV1.FAULTED)

    def _complete_capacity_rollover(
        self, item: EvidenceWriterQueueItemV1, disposition: StoreDisposition
    ) -> None:
        with self._rollover_lock:
            if (
                item in self._completed_rollovers
                or item in self._rollover_completion_claimed
            ):
                return
            active = self._active_rollover
            self._rollover_completion_claimed.add(item)
        if active is None:
            raise RuntimeError("rollover completion has no runtime owner")
        admission_preparation, lifecycle_preparation = active
        admission = self._admission
        if admission is None:
            raise RuntimeError("rollover completion lost admission owner")
        durable = admission.prepare_rollover_completion(admission_preparation, item, disposition)
        if not durable:
            probe = self._qualification_capacity_probe
            if probe is not None:
                loop = self._owner_loop
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        probe.pause_after_rollover_rejection(), loop
                    ).result()
            self._lifecycle_owner.abort_prepared_rollover(lifecycle_preparation)
            admission.finalize_failed_rollover_completion(
                admission_preparation,
                disposition,
            )
            with self._rollover_lock:
                self._active_rollover = None
                self._completed_rollovers[item] = disposition
                self._rollover_completion_claimed.discard(item)
            self._record_rollover_terminal(RolloverResultV1.FAULTED)
            return
        durable_result = (
            RolloverResultV1.COMMITTED
            if disposition is StoreDisposition.COMMITTED
            else RolloverResultV1.IDEMPOTENT
        )
        self._production_observation_recorder.record_rollover(
            stage=RolloverStageV1.DURABLE,
            result=durable_result,
        )
        self._lifecycle_owner.commit_prepared_rollover(lifecycle_preparation)
        admission.publish_prepared_rollover(admission_preparation)
        assert self._pending_create is not None and type(item.payload) is RolloverSessionV1
        self._production_observation_recorder.record_rollover(
            stage=RolloverStageV1.PUBLISHED,
            result=durable_result,
        )
        with self._rollover_lock:
            self._active_rollover = None
            self._completed_rollovers[item] = disposition
            self._rollover_completion_claimed.discard(item)
        self._record_rollover_terminal(durable_result)

    def _record_rollover_terminal(self, result: RolloverResultV1) -> None:
        self._production_observation_recorder.record_rollover(
            stage=RolloverStageV1.TERMINAL,
            result=result,
        )

    def configure_retention_owner(
        self,
        *,
        cancel: Callable[[], Awaitable[None]],
        wall_clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Bind the one host cancellation owner used at persisted TTL expiry."""

        if not callable(cancel):
            raise TypeError("retention cancellation must be callable")
        if wall_clock is not None and not callable(wall_clock):
            raise TypeError("retention wall clock must be callable or None")
        if sleep is not None and not callable(sleep):
            raise TypeError("retention sleep must be callable or None")
        if self._retention_cancel is not None:
            raise RuntimeError("retention owner is already configured")
        self._retention_cancel = cancel
        if wall_clock is not None:
            self._retention_wall_clock = wall_clock
        if sleep is not None:
            self._retention_sleep = sleep

    def reserve_consent(self, command: CreateEpochV1) -> CreateEpochReservationV1 | None:
        if self._closed or self._closing:
            raise RuntimeError("host evidence runtime is closed")
        if self._epoch_retiring:
            raise RuntimeError("host evidence runtime is retiring the active epoch")
        if type(command) is not CreateEpochV1:
            raise TypeError("command must be an exact CreateEpochV1")
        if command.retention_hours != self._retention_hours:
            raise ValueError("consent retention does not match host configuration")
        if self._admission is None:
            queue = BoundedEvidenceWriterQueueV1()
            with _qualification_capacity_probe_scope(self._qualification_capacity_probe):
                admission = EvidenceAdmissionControllerV1(
                    enabled=True,
                    owner_generation=self._lifecycle_owner.owner_generation,
                    writer_sink=queue,
                    operation_scheduler=self._operation_scheduler,
                    conversation_authority_is_current=(
                        self._lifecycle_owner.conversation_authority_is_current
                    ),
                )
                admission._install_rollover_runtime_owner(
                    self._rollover_claimed,
                    _ROLLOVER_RUNTIME_OWNER_TOKEN,
                )
            self._queue = queue
            self._admission = admission
            self._admission_view = EvidenceAdmissionViewV1(admission)
        reservation = self._admission.try_reserve_create_epoch(command)
        if reservation is not None:
            try:
                self._lifecycle_owner.stage_drain_lineage(
                    binding_id=command.binding_id,
                    binding_generation=command.binding_generation,
                    consent_epoch_id=command.consent_epoch_id,
                    logical_session_id=command.logical_session_id,
                )
            except BaseException:
                self._admission.release_create_epoch_reservation(reservation)
                raise
            self._pending_create = command
        return reservation

    def reserve_consent_authority(
        self,
        command: CreateEpochV1,
        projection_reservation: ProjectionReservation,
        validate_projection_reservation: Callable[[ProjectionReservation], None],
    ) -> ConsentCreateAuthorityV1:
        """Atomically reserve create capacity and mint its process-local authority."""

        if type(projection_reservation) is not ProjectionReservation:
            raise TypeError("projection_reservation must be an exact ProjectionReservation")
        if not callable(validate_projection_reservation):
            raise TypeError("projection reservation validator must be callable")
        validate_projection_reservation(projection_reservation)
        if self._consent_settlement_operation is not None and not self._consent_settlement_claimed:
            raise RuntimeError("the prior timed-out consent settlement has not been claimed")
        if self._pending_create is not None:
            raise RuntimeError("a consent create is already pending or reserved")
        reservation = self.reserve_consent(command)
        if reservation is None:
            raise RuntimeError("consent create capacity is unavailable")
        authority = object.__new__(ConsentCreateAuthorityV1)
        values: dict[str, object] = {
            "protocol_version": command.protocol_version,
            "binding_id": command.binding_id,
            "binding_generation": command.binding_generation,
            "control_sequence": command.control_sequence,
            "control_fingerprint_hash": command.control_fingerprint_hash,
            "consent_version": command.consent_version,
            "disclosure_digest": command.disclosure_digest,
            "retention_hours": command.retention_hours,
            "microphone_accepted": command.microphone_accepted,
            "typed_accepted": command.typed_accepted,
            "projection_reservation": projection_reservation,
            "create_epoch_reservation": reservation,
        }
        for name, value in values.items():
            object.__setattr__(authority, name, value)
        authority._validate()
        validate_authority_composition(authority, command)
        self._pending_consent_authority = authority
        return authority

    def reserve_browser_consent(
        self,
        *,
        binding_generation: int,
        request: EvidenceConsentRequestV1,
        projection_reservation: ProjectionReservation,
        validate_projection_reservation: Callable[[ProjectionReservation], None],
        microphone_available: bool,
        typed_available: bool,
    ) -> ConsentCreateAuthorityV1:
        """Derive all evidence lineage server-side after strict browser parsing."""

        if type(request) is not EvidenceConsentRequestV1:
            raise TypeError("request must be an exact EvidenceConsentRequestV1")
        if request.retention_hours != self._retention_hours:
            raise ValueError("consent retention does not match host configuration")
        if request.sources.microphone and not microphone_available:
            raise RuntimeError("accepted microphone source is unavailable")
        if request.sources.typed and not typed_available:
            raise RuntimeError("accepted typed source is unavailable")
        ids = [str(uuid4()) for _ in range(7)]
        (
            installation_id,
            producer_instance_id,
            consent_epoch_id,
            logical_session_id,
            binding_id,
            session_event_id,
            binding_event_id,
        ) = ids
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "accepted": request.accepted,
                    "consentVersion": request.consent_version,
                    "disclosureDigest": request.disclosure_digest,
                    "retentionHours": request.retention_hours,
                    "sequence": request.sequence,
                    "sources": {
                        "microphone": request.sources.microphone,
                        "typed": request.sources.typed,
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        command = CreateEpochV1(
            protocol_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            consent_epoch_id=consent_epoch_id,
            logical_session_id=logical_session_id,
            binding_id=binding_id,
            binding_generation=binding_generation,
            consent_version=CONSENT_VERSION,
            disclosure_digest=request.disclosure_digest,
            retention_hours=request.retention_hours,
            microphone_accepted=request.sources.microphone,
            typed_accepted=request.sources.typed,
            control_sequence=request.sequence,
            control_fingerprint_hash=fingerprint,
            session_opened=EvidenceSnapshotV1(
                schema_version=1,
                installation_id=installation_id,
                producer_instance_id=producer_instance_id,
                logical_session_id=logical_session_id,
                event_id=session_event_id,
                event_sequence=1,
                event_kind=EventKind.SESSION_OPENED,
                payload=SessionOpenedPayloadV1(
                    consent_epoch_id=consent_epoch_id,
                    binding_id=binding_id,
                    consent_version=CONSENT_VERSION,
                    disclosure_digest=request.disclosure_digest,
                    retention_hours=request.retention_hours,
                    microphone_accepted=request.sources.microphone,
                    typed_accepted=request.sources.typed,
                    predecessor_session_id=None,
                ),
            ),
            binding_opened=EvidenceSnapshotV1(
                schema_version=1,
                installation_id=installation_id,
                producer_instance_id=producer_instance_id,
                logical_session_id=logical_session_id,
                event_id=binding_event_id,
                event_sequence=2,
                event_kind=EventKind.BINDING_OPENED,
                payload=BindingOpenedPayloadV1(
                    binding_id=binding_id,
                    binding_generation=binding_generation,
                    microphone_available=microphone_available,
                    typed_available=typed_available,
                ),
            ),
        )
        return self.reserve_consent_authority(
            command,
            projection_reservation,
            validate_projection_reservation,
        )

    def retain_lifecycle_status_reservations(
        self,
        reservations: tuple[ProjectionReservation, ...],
        release: Callable[[tuple[ProjectionReservation, ...]], None],
    ) -> None:
        """Retain consent-time status capacity for later revoke/terminal publication."""

        if not 1 <= len(reservations) <= 3 or any(
            type(reservation) is not ProjectionReservation for reservation in reservations
        ):
            raise TypeError("lifecycle status reservations must contain exact capabilities")
        if self._lifecycle_status_reservations:
            raise RuntimeError("lifecycle status reservations are already retained")
        if self._release_lifecycle_status_reservations is not None:
            raise RuntimeError("lifecycle status reservation release is already retained")
        if not callable(release):
            raise TypeError("lifecycle status reservation release must be callable")
        self._lifecycle_status_reservations.extend(reservations)
        self._release_lifecycle_status_reservations = release

    def retain_pending_consent_projection_reservation(
        self,
        authority: ConsentCreateAuthorityV1,
        reservation: ProjectionReservation,
        release: Callable[[tuple[ProjectionReservation, ...]], None],
    ) -> None:
        """Transfer the unpublished consent-status slot to runtime close ownership."""

        if type(authority) is not ConsentCreateAuthorityV1:
            raise TypeError("authority must be an exact ConsentCreateAuthorityV1")
        if authority is not self._pending_consent_authority:
            raise RuntimeError("only the exact pending consent owns its projection slot")
        if type(reservation) is not ProjectionReservation:
            raise TypeError("reservation must be an exact ProjectionReservation")
        if reservation is not authority.projection_reservation:
            raise RuntimeError("consent projection reservation does not match its authority")
        if not callable(release):
            raise TypeError("projection reservation release must be callable")
        if (
            self._pending_consent_projection_authority is not None
            or self._pending_consent_projection_reservation is not None
            or self._release_pending_consent_projection is not None
        ):
            raise RuntimeError("a pending consent projection reservation is already retained")
        self._pending_consent_projection_authority = authority
        self._pending_consent_projection_reservation = reservation
        self._release_pending_consent_projection = release

    def mark_pending_consent_projection_published(
        self,
        authority: ConsentCreateAuthorityV1,
        reservation: ProjectionReservation,
    ) -> None:
        """Release runtime close ownership after the exact status slot is published."""

        if type(authority) is not ConsentCreateAuthorityV1:
            raise TypeError("authority must be an exact ConsentCreateAuthorityV1")
        if type(reservation) is not ProjectionReservation:
            raise TypeError("reservation must be an exact ProjectionReservation")
        if (
            authority is not self._pending_consent_projection_authority
            or reservation is not self._pending_consent_projection_reservation
            or self._release_pending_consent_projection is None
        ):
            raise RuntimeError("pending consent projection reservation ownership conflicts")
        self._pending_consent_projection_authority = None
        self._pending_consent_projection_reservation = None
        self._release_pending_consent_projection = None
        if (
            self._consent_activation_authority is authority
            and self._consent_activation_operation is asyncio.current_task()
        ):
            self._consent_activation_authority = None
            self._consent_activation_operation = None

    def take_lifecycle_status_reservation(self) -> ProjectionReservation:
        """Transfer the next consent-retained lifecycle status capability."""

        if not self._lifecycle_status_reservations:
            raise RuntimeError("no lifecycle status reservation remains")
        reservation = self._lifecycle_status_reservations.pop(0)
        if not self._lifecycle_status_reservations:
            self._release_lifecycle_status_reservations = None
        return reservation

    def release_lifecycle_status_reservations(
        self,
        reservations: tuple[ProjectionReservation, ...],
    ) -> None:
        """Abandon one exact retained lifecycle set after activation rolls back.

        Retirement may have already cleared the owner-side list before the
        browser composition gets control back.  Treat that exact completed
        retirement as an idempotent release, but reject any foreign or partial
        capability set.
        """

        if (
            type(reservations) is not tuple
            or not reservations
            or len(reservations) > 3
            or any(type(reservation) is not ProjectionReservation for reservation in reservations)
            or len(set(reservations)) != len(reservations)
        ):
            raise TypeError("lifecycle status reservations must be exact and unique")
        retained = tuple(self._lifecycle_status_reservations)
        if retained != reservations:
            if not retained:
                return
            raise RuntimeError("lifecycle status reservation ownership conflicts")
        self._release_retained_lifecycle_status_reservations()

    def _release_retained_lifecycle_status_reservations(self) -> None:
        """Return all still-owned projection slots through their exact owner callback."""

        reservations = tuple(self._lifecycle_status_reservations)
        if not reservations:
            self._release_lifecycle_status_reservations = None
            return
        release = self._release_lifecycle_status_reservations
        if release is None:
            raise RuntimeError("retained lifecycle status reservations lost their release owner")
        release(reservations)
        self._lifecycle_status_reservations.clear()
        self._release_lifecycle_status_reservations = None

    def reserve_browser_revoke(
        self,
        *,
        binding_generation: int,
        request: EvidenceRevokeRequestV1,
        projection_reservation: ProjectionReservation,
        validate_projection_reservation: Callable[[ProjectionReservation], None],
    ) -> ConsentRevokeAuthorityV1:
        """Mint one runtime-owned revoke capability for the active epoch."""

        if type(request) is not EvidenceRevokeRequestV1:
            raise TypeError("request must be an exact EvidenceRevokeRequestV1")
        if type(projection_reservation) is not ProjectionReservation:
            raise TypeError("projection_reservation must be exact")
        if not callable(validate_projection_reservation):
            raise TypeError("projection reservation validator must be callable")
        validate_projection_reservation(projection_reservation)
        with self._revoke_reservation_lock:
            if self._pending_revoke_authority is not None:
                raise RuntimeError("a revoke is already pending or reserved")
            command = self._pending_create
            admission = self._admission
            if (
                command is None
                or admission is None
                or not self._published
                or binding_generation != command.binding_generation
            ):
                raise RuntimeError("revoke binding is stale or capture is inactive")
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"sequence": request.sequence},
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            authority = object.__new__(ConsentRevokeAuthorityV1)
            values = {
                "protocol_version": 1,
                "binding_id": command.binding_id,
                "binding_generation": command.binding_generation,
                "consent_epoch_id": command.consent_epoch_id,
                "revoke_gate_generation": self._owner_generation,
                "control_sequence": request.sequence,
                "control_fingerprint_hash": fingerprint,
                "projection_reservation": projection_reservation,
            }
            for name, value in values.items():
                object.__setattr__(authority, name, value)
            authority._validate()
            # This is the one synchronous revoke linearization point.  It closes
            # foreground capability issuance before the writer receives its
            # request, while admission retains pre-existing lease settlement.
            self._lifecycle_owner.close_binding(BindingCloseReason.CONSENT_REVOKED)
            self._published = False
            ticket = admission.begin_revoke(authority)
            self._pending_revoke_authority = authority
            self._pending_revoke_ticket = ticket
            if ticket.disposition in {
                RevokeDisposition.PURGE_COMPLETED,
                RevokeDisposition.PURGE_FAILED,
                RevokeDisposition.WRITER_FAULT,
            }:
                self._settle_revoke_terminal_state(ticket.disposition)
            else:
                self._capture_state = CaptureState.REVOKED_PURGING
            return authority

    async def activate_revoke(
        self,
        authority: ConsentRevokeAuthorityV1,
        *,
        timeout_seconds: float,
    ) -> RevokeDisposition:
        """Dispatch the exact trusted revoke and wait only for durability."""

        if authority is not self._pending_revoke_authority:
            raise RuntimeError("revoke requires the exact pending authority")
        ticket = self._pending_revoke_ticket
        if ticket is None:
            raise RuntimeError("revoke reservation did not publish its exact ticket")
        completed = await _wait_for_thread_signal(
            ticket.durability_event,
            timeout_seconds=float(timeout_seconds),
        )
        if not completed:
            return RevokeDisposition.CLOSED_NOT_DURABLE
        if ticket.disposition is RevokeDisposition.REVOKE_DURABLY_SCHEDULED:
            admission = self._admission
            if admission is None:
                raise RuntimeError("evidence admission is unavailable")
            admission.try_enqueue_revoke_finalize(ticket)
        elif ticket.disposition in {
            RevokeDisposition.PURGE_COMPLETED,
            RevokeDisposition.PURGE_FAILED,
            RevokeDisposition.WRITER_FAULT,
        }:
            self._settle_revoke_terminal_state(ticket.disposition)
        return ticket.disposition

    async def wait_revoke_terminal(self, *, timeout_seconds: float) -> RevokeDisposition:
        """Observe the terminal purge milestone without delaying revoke acknowledgement."""

        if type(timeout_seconds) not in (int, float) or not 0 < float(timeout_seconds) <= 2:
            raise ValueError("timeout_seconds must be positive and at most two")
        ticket = self._pending_revoke_ticket
        if ticket is None:
            raise RuntimeError("no revoke is pending")
        # The durable revoke can be acknowledged while already-admitted ordered
        # records still drain.  Re-attempt exact finalization from this owner-only
        # observation point; the admission authority refuses duplicates until the
        # preceding credits are gone.
        if ticket.disposition is RevokeDisposition.REVOKE_DURABLY_SCHEDULED:
            admission = self._admission
            if admission is None:
                raise RuntimeError("evidence admission is unavailable")
            admission.try_enqueue_revoke_finalize(ticket)
        completed = await _wait_for_thread_signal(
            ticket.terminal_event,
            timeout_seconds=float(timeout_seconds),
        )
        if not completed:
            return RevokeDisposition.CONTROL_TIMED_OUT
        self._settle_revoke_terminal_state(ticket.disposition)
        return ticket.disposition

    def _settle_revoke_terminal_state(self, disposition: RevokeDisposition) -> None:
        """Publish the terminal state that the admission owner has already settled."""

        if disposition is RevokeDisposition.PURGE_COMPLETED:
            self._capture_state = CaptureState.IDLE
        elif disposition is RevokeDisposition.PURGE_FAILED:
            self._capture_state = CaptureState.PURGE_FAILED
        elif disposition is RevokeDisposition.WRITER_FAULT:
            self._capture_state = CaptureState.FAULTED
        else:
            raise RuntimeError("revoke terminal disposition is not terminal")
        self._published = False

    def observe_revoke_terminal(
        self,
        callback: Callable[[RevokeDisposition], bool],
        release_unpublished: Callable[[], None],
    ) -> None:
        """Own one terminal observer and its transferred projection reservation."""

        if not callable(callback):
            raise TypeError("revoke terminal callback must be callable")
        if not callable(release_unpublished):
            raise TypeError("revoke terminal release must be callable")
        if self._pending_revoke_ticket is None:
            raise RuntimeError("no revoke is pending")

        async def observe() -> None:
            published = False
            try:
                while True:
                    disposition = await self.wait_revoke_terminal(timeout_seconds=2.0)
                    if disposition is RevokeDisposition.CONTROL_TIMED_OUT:
                        continue
                    published = callback(disposition)
                    if type(published) is not bool:
                        raise TypeError("revoke terminal callback must return an exact bool")
                    if not published:
                        raise RuntimeError("revoke terminal projection was not published")
                    return
            finally:
                if not published:
                    release_unpublished()

        task = asyncio.create_task(observe(), name="evidence-revoke-terminal")
        self._revoke_observers.add(task)

        def complete_observer(completed: asyncio.Task[None]) -> None:
            self._revoke_observers.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                self._revoke_observer_failures.append(error)

        task.add_done_callback(complete_observer)

    def validate_pending_consent_authority(
        self,
        authority: ConsentCreateAuthorityV1,
    ) -> None:
        """Reject structurally equivalent capabilities not minted by this owner."""

        if type(authority) is not ConsentCreateAuthorityV1:
            raise TypeError("authority must be an exact ConsentCreateAuthorityV1")
        if authority is not self._pending_consent_authority:
            raise RuntimeError("consent activation requires the exact pending authority")

    def abandon_reserved_consent(self, authority: ConsentCreateAuthorityV1) -> None:
        """Atomically abandon a pre-dispatch consent reservation after composition fails.

        The writer factory runs after the browser projection and admission
        reservations exist but before any writer or consent dispatch is published.
        A factory failure must therefore return every reservation and leave the
        runtime equivalent to a clean retry.
        """

        if type(authority) is not ConsentCreateAuthorityV1:
            raise TypeError("authority must be an exact ConsentCreateAuthorityV1")
        if authority is not self._pending_consent_authority:
            raise RuntimeError("only the exact pending consent may be abandoned")
        if self._pending_consent_ticket is not None:
            raise RuntimeError("dispatched consent cannot be abandoned")
        admission = self._admission
        if admission is None or self._pending_create is None:
            raise RuntimeError("pending consent is missing its admission reservation")
        admission.release_create_epoch_reservation(authority.create_epoch_reservation)
        self._pending_create = None
        self._pending_evidence_consent_key_reset()
        self._queue = None
        self._admission = None
        self._admission_view = None
        self._lifecycle_owner.retire_drain_lineage()

    async def activate_consent(
        self,
        authority: ConsentCreateAuthorityV1,
        *,
        transport: _WriterTransportV1,
        binding_is_current: Callable[[CreateEpochV1], bool],
        timeout_seconds: float,
    ) -> ConsentDisposition:
        """Dispatch one consent and wait at most the control durability bound.

        A create that misses that bound remains owned by the runtime.  Its writer
        dispatch cannot be abandoned or retried by a caller while it is still in
        flight, so this method starts one owner background settlement instead.
        """

        if type(authority) is not ConsentCreateAuthorityV1:
            raise TypeError("authority must be an exact ConsentCreateAuthorityV1")
        authority._validate()
        if type(timeout_seconds) not in (int, float) or not 0 < float(timeout_seconds) <= 2:
            raise ValueError("timeout_seconds must be positive and at most two")
        _validate_writer_transport_v1(transport)
        operation = asyncio.current_task()
        if operation is None:
            raise RuntimeError("consent activation must run in an asyncio task")
        activation_loop = asyncio.get_running_loop()
        rollback_before_dispatch = False
        async with self._state_lock:
            with self._rollover_lock:
                owner_loop = self._owner_loop
                if owner_loop is None:
                    self._owner_loop = activation_loop
                elif owner_loop is not activation_loop:
                    raise RuntimeError("consent activation must use the runtime owner loop")
            # Only the gateway retains the primary capture-status reservation.
            # Direct runtime callers have no public completion slot, so retaining
            # their ambient task would let a later close await its own caller.
            if self._pending_consent_projection_authority is authority:
                active = self._consent_activation_operation
                if active is None or active.done():
                    self._consent_activation_authority = authority
                    self._consent_activation_operation = operation
                elif active is not operation or self._consent_activation_authority is not authority:
                    raise RuntimeError("a different consent activation is already in flight")
            if (
                self._published
                and self._pending_consent_authority is authority
                and self._pending_consent_ticket is not None
                and self._pending_consent_ticket.disposition is ConsentDisposition.CONSENT_ACTIVATED
            ):
                return ConsentDisposition.CONSENT_ACTIVATED
            settlement = self._consent_settlement_operation
            if (
                self._consent_settlement_authority is authority
                and settlement is not None
                and not settlement.done()
            ):
                return ConsentDisposition.CONTROL_TIMED_OUT
            if self._pending_consent_authority is not authority:
                # Close owns rollback of an undispatched authority.  Its late
                # completion is a terminal non-active result, not a stale
                # authority error that would strand browser reservations.
                if self._closed or self._closing:
                    rollback_before_dispatch = True
                else:
                    self.validate_pending_consent_authority(authority)
            if self._closed or self._closing or self._epoch_retiring:
                rollback_before_dispatch = True
            if rollback_before_dispatch:
                ticket = None
                command = None
            else:
                queue = self._queue
                admission = self._admission
                command = self._pending_create
                if queue is None or admission is None or command is None:
                    raise RuntimeError("consent must be reserved before activation")
                validate_authority_composition(authority, command)
                ticket = self._pending_consent_ticket
                if ticket is None:
                    # Admission owns the decision before a named consumer is created.
                    # Rejected creates therefore cannot leak a dormant writer thread.
                    ticket = admission.begin_consent(authority)
                    self._pending_consent_ticket = ticket
                    if ticket.disposition is ConsentDisposition.CREATE_PENDING:
                        dispatcher = EvidenceWriterDispatcherV1(
                            source=queue,
                            admission=admission,
                            transport=transport,
                            binding_is_current=binding_is_current,
                            rollover_completion=self._complete_capacity_rollover,
                        )
                        self._writer = EvidenceWriterRuntimeOwnerV1(
                            source=queue,
                            dispatch=dispatcher.dispatch_item,
                        )
                        self._transport = transport
                elif authority is not self._pending_consent_authority:
                    raise RuntimeError("consent retry must use the exact pending authority")
        if rollback_before_dispatch:
            await self._settle_nonactive_consent_rollback()
            return ConsentDisposition.CREATE_FAILED
        assert ticket is not None
        assert command is not None
        completed = await _wait_for_thread_signal(
            ticket.durability_event,
            timeout_seconds=float(timeout_seconds),
        )
        if not completed:
            self._begin_timed_out_consent_settlement(
                authority,
                ticket,
                command,
                binding_is_current,
            )
            await self._clear_timed_out_consent_activation(authority, operation)
            return ConsentDisposition.CONTROL_TIMED_OUT
        return await self._finalize_consent_activation(
            authority,
            ticket,
            command,
            binding_is_current,
        )

    async def _clear_timed_out_consent_activation(
        self,
        authority: ConsentCreateAuthorityV1,
        operation: asyncio.Task[Any],
    ) -> None:
        """The retained timeout settlement, not its finished request, owns close."""

        async with self._state_lock:
            if (
                self._consent_activation_authority is authority
                and self._consent_activation_operation is operation
            ):
                self._consent_activation_authority = None
                self._consent_activation_operation = None

    def claim_consent_settlement_task(
        self,
        authority: ConsentCreateAuthorityV1,
    ) -> asyncio.Task[ConsentDisposition]:
        """Atomically claim observation of the exact retained timeout settlement."""

        if type(authority) is not ConsentCreateAuthorityV1:
            raise TypeError("authority must be an exact ConsentCreateAuthorityV1")
        if authority is not self._consent_settlement_authority:
            raise RuntimeError("consent authority has no timed-out settlement")
        operation = self._consent_settlement_operation
        if operation is None:
            raise RuntimeError("timed-out consent settlement is unavailable")
        self._consent_settlement_claimed = True
        return operation

    def _begin_timed_out_consent_settlement(
        self,
        authority: ConsentCreateAuthorityV1,
        ticket: CreateEpochTicketV1,
        command: CreateEpochV1,
        binding_is_current: Callable[[CreateEpochV1], bool],
    ) -> None:
        """Own final activation after the foreground durability wait expires."""

        current = self._consent_settlement_operation
        if (
            self._consent_settlement_authority is authority
            and current is not None
            and not current.done()
        ):
            return

        async def settle() -> ConsentDisposition:
            try:
                await _wait_for_thread_signal(ticket.durability_event, timeout_seconds=None)
                return await self._finalize_consent_activation(
                    authority,
                    ticket,
                    command,
                    binding_is_current,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                # A dispatched activation has exactly one terminal rollback path.
                # Preserve that path even if its observation/finalization fails.
                await self._settle_nonactive_consent_rollback(
                    reason=BindingCloseReason.BINDING_REPLACED
                )
                return ConsentDisposition.CREATE_FAILED

        operation = asyncio.create_task(
            settle(),
            name="host-evidence-consent-timeout-settlement",
        )
        self._consent_settlement_authority = authority
        self._consent_settlement_operation = operation
        self._consent_settlement_claimed = False

    async def _finalize_consent_activation(
        self,
        authority: ConsentCreateAuthorityV1,
        ticket: CreateEpochTicketV1,
        command: CreateEpochV1,
        binding_is_current: Callable[[CreateEpochV1], bool],
    ) -> ConsentDisposition:
        """Publish the one durable create or retire its exact owned activation."""

        if ticket.disposition is not ConsentDisposition.CONSENT_ACTIVATED:
            await self._settle_nonactive_consent_rollback()
            return ticket.disposition

        # The dispatch-time binding check establishes durable provenance, but close
        # or replacement can win while this caller observes that durability event.
        # Recheck outside the asyncio gate, then atomically bind and publish.
        try:
            binding_current = _checked_binding_is_current(binding_is_current, command)
        except BaseException as error:
            await self._settle_nonactive_consent_rollback(
                reason=BindingCloseReason.BINDING_REPLACED
            )
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return ConsentDisposition.CREATE_FAILED
        async with self._state_lock:
            if (
                self._published
                and self._pending_consent_authority is authority
                and self._pending_consent_ticket is ticket
                and self._pending_create is command
            ):
                return ticket.disposition
            can_publish = (
                not self._closed
                and not self._closing
                and not self._epoch_retiring
                and self._pending_consent_authority is authority
                and self._pending_consent_ticket is ticket
                and self._pending_create is command
                and binding_current
            )
            if can_publish:
                self._lifecycle_owner.activate_binding(
                    binding_id=authority.binding_id,
                    binding_generation=authority.binding_generation,
                    consent_epoch_id=command.consent_epoch_id,
                    logical_session_id=command.logical_session_id,
                )
                self._published = True
                self._capture_state = CaptureState.ACTIVE
                if self._retention_cancel is not None and self._retention_task is None:
                    self._retention_task = asyncio.create_task(
                        self._run_retention_owner(),
                        name="evidence-retention-owner",
                    )
                if self._qualification_checkpoint_channel is not None:
                    await self._qualification_checkpoint_channel.emit("host_consent_active")
                return ticket.disposition

        await self._settle_nonactive_consent_rollback(reason=BindingCloseReason.BINDING_REPLACED)
        return ConsentDisposition.CREATE_FAILED

    async def _settle_nonactive_consent_rollback(
        self,
        *,
        reason: BindingCloseReason | None = None,
    ) -> None:
        """Finish one terminal non-active activation before reporting failure."""

        async with self._state_lock:
            close = self._close_operation if self._closing else None
            closed = self._closed
            active = self._consent_activation_operation
        if close is not None and close is not asyncio.current_task():
            if active is asyncio.current_task():
                # The close owner must continue draining the real create.  This
                # request returns its terminal result so the gateway can consume
                # the still-owned primary status reservation before close cleans
                # the remaining lifecycle state.
                return
            await asyncio.shield(close)
            return
        if closed:
            return
        await self._retire_current_epoch(reason=reason)

    async def _run_retention_owner(self) -> ExpiryDisposition:
        admission = self._admission
        command = self._pending_create
        transport = self._transport
        cancel = self._retention_cancel
        if admission is None or command is None or transport is None or cancel is None:
            return ExpiryDisposition.WRITER_FAULT
        expiry_observer = getattr(transport, "active_session_expiry", None)
        if not callable(expiry_observer):
            return ExpiryDisposition.WRITER_FAULT
        expires_at = await asyncio.to_thread(expiry_observer, command.logical_session_id)
        if type(expires_at) is not str:
            return ExpiryDisposition.WRITER_FAULT
        try:
            deadline = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        except ValueError:
            return ExpiryDisposition.WRITER_FAULT
        now = self._retention_wall_clock()
        if type(now) is not datetime or now.tzinfo is None:
            return ExpiryDisposition.WRITER_FAULT
        await self._retention_sleep(max(0.0, (deadline - now.astimezone(UTC)).total_seconds()))
        admission.close_for_retention_expiry()
        while True:
            await cancel()
            if not admission.diagnostics().active_lease_count:
                break
            await self._retention_sleep(0.01)
        authority = object.__new__(SessionExpiryAuthorityV1)
        for name, value in {
            "protocol_version": 1,
            "owner_generation": self._lifecycle_owner.owner_generation,
            "consent_epoch_id": command.consent_epoch_id,
            "logical_session_id": command.logical_session_id,
            "expires_at_utc": expires_at,
            "deadline_admission_ordinal": admission.final_admission_ordinal,
            "mode": ExpiryMode.ERASE_STUCK,
        }.items():
            object.__setattr__(authority, name, value)
        authority._validate()
        disposition = admission.begin_expiry(authority)
        if disposition is ExpiryDisposition.ERASURE_DURABLY_SCHEDULED:
            for _ in range(200):
                if admission.diagnostics().owner_state.value == "stopped":
                    self._capture_state = CaptureState.IDLE
                    self._published = False
                    break
                await asyncio.sleep(0.01)
        return disposition

    async def wait_retention_terminal(self, *, timeout_seconds: float) -> ExpiryDisposition:
        if type(timeout_seconds) not in (int, float) or not 0 < float(timeout_seconds) <= 2:
            raise ValueError("timeout_seconds must be positive and at most two")
        task = self._retention_task
        if task is None:
            raise RuntimeError("retention owner is not running")
        try:
            return await asyncio.wait_for(asyncio.shield(task), float(timeout_seconds))
        except TimeoutError:
            return ExpiryDisposition.WRITER_FAULT

    async def close(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            self._closing = True
            with self._rollover_lock:
                self._rollover_handoffs_closed = True
                rollover_handoffs = tuple(self._rollover_handoffs)
        if rollover_handoffs:
            try:
                async with asyncio.timeout(_CLOSE_DRAIN_TIMEOUT_SECONDS):
                    await asyncio.shield(
                        asyncio.gather(
                            *(asyncio.wrap_future(handoff) for handoff in rollover_handoffs),
                            return_exceptions=True,
                        )
                    )
            except TimeoutError:
                with self._rollover_lock:
                    admission = self._admission
                    for handoff in rollover_handoffs:
                        preparation = self._rollover_handoffs.pop(handoff, None)
                        if preparation is None:
                            continue
                        if (
                            admission is not None
                            and preparation.item is None
                            and not preparation.terminal
                        ):
                            admission.abort_prepared_rollover(preparation)
                        self._production_observation_recorder.record_rollover(
                            stage=RolloverStageV1.CLAIMED,
                            result=RolloverResultV1.ACCEPTED,
                        )
                        self._record_rollover_terminal(RolloverResultV1.CLOSED)
                        if not handoff.done():
                            handoff.set_result(None)
        with self._rollover_lock:
            rollover_tasks = tuple(self._rollover_tasks)
        if rollover_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in rollover_tasks),
                return_exceptions=True,
            )
        with self._rollover_lock:
            active_rollover = self._active_rollover
        if active_rollover is not None:
            settled = await _wait_for_thread_signal(
                active_rollover[0].ticket.event,
                timeout_seconds=_CLOSE_DRAIN_TIMEOUT_SECONDS,
            )
            if not settled:
                raise RuntimeError("capacity rollover did not settle before close timeout")
        async with self._state_lock:
            operation = self._close_operation
            if operation is None and self._published:
                # Ordinary host close has no browser replacement authority to
                # append, but it must still revoke the active lifecycle bearer
                # before the first close owner drains. Joins and retries retain
                # that revocation instead of attempting to consume it again.
                self._lifecycle_owner.close_binding(BindingCloseReason.CLIENT_CLOSED)
            if operation is None or (
                operation.done() and (operation.cancelled() or operation.exception() is not None)
            ):
                operation = asyncio.create_task(
                    self._close_owned(),
                    name="host-evidence-runtime-close",
                )
                self._close_operation = operation
        await asyncio.shield(operation)

    async def _close_owned(self) -> None:
        recorder = self._production_observation_recorder
        try:
            await self._close_owned_stages()
        except asyncio.CancelledError:
            recorder.record_close_stage(
                stage=CloseStageV1.EVIDENCE_RUNTIME,
                result=CloseResultV1.CANCELLED,
            )
            raise
        except BaseException:
            recorder.record_close_stage(
                stage=CloseStageV1.EVIDENCE_RUNTIME,
                result=CloseResultV1.FAILED,
            )
            raise
        recorder.record_close_stage(
            stage=CloseStageV1.EVIDENCE_RUNTIME,
            result=CloseResultV1.SUCCEEDED,
        )

    async def _close_owned_stages(self) -> None:
        errors: list[BaseException] = []
        settlement_result, settlement_error = await self._cancel_and_observe_consent_settlement()
        if settlement_result is not None:
            self._observe_close_stage(CloseStageV1.CONSENT_SETTLEMENT, settlement_result)
        if settlement_error is not None:
            errors.append(settlement_error)
        retirement = self._invalidation_operation
        if retirement is not None and retirement is not asyncio.current_task():
            try:
                await self._await_or_recover_epoch_retirement_for_close(retirement)
            except BaseException as error:
                self._observe_close_stage(
                    CloseStageV1.EPOCH_RETIREMENT,
                    (
                        CloseResultV1.CANCELLED
                        if isinstance(error, asyncio.CancelledError)
                        else CloseResultV1.FAILED
                    ),
                )
                errors.append(error)
            else:
                self._observe_close_stage(
                    CloseStageV1.EPOCH_RETIREMENT,
                    CloseResultV1.SUCCEEDED,
                )
        retention = self._retention_task
        self._retention_task = None
        if retention is not None and not retention.done():
            retention.cancel()
            try:
                retention_results = await asyncio.gather(retention, return_exceptions=True)
            except BaseException as error:
                errors.append(error)
                self._observe_close_stage(
                    CloseStageV1.RETENTION_CANCELLATION,
                    (
                        CloseResultV1.CANCELLED
                        if isinstance(error, asyncio.CancelledError)
                        else CloseResultV1.FAILED
                    ),
                )
            else:
                retention_failed = any(
                    isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                    for result in retention_results
                )
                if retention_failed:
                    errors.extend(
                        result
                        for result in retention_results
                        if isinstance(result, BaseException)
                        and not isinstance(result, asyncio.CancelledError)
                    )
                self._observe_close_stage(
                    CloseStageV1.RETENTION_CANCELLATION,
                    CloseResultV1.FAILED if retention_failed else CloseResultV1.SUCCEEDED,
                )
        # The browser projection reservation for terminal revoke status is owned
        # by this observer.  Cancel and observe it before writer shutdown so a
        # close that wins before purge releases the exact unpublished slot.
        observers = tuple(self._revoke_observers)
        if observers:
            for observer in observers:
                observer.cancel()
            try:
                observer_results = await asyncio.gather(*observers, return_exceptions=True)
            except BaseException as error:
                errors.append(error)
                self._observe_close_stage(
                    CloseStageV1.REVOKE_OBSERVER_CANCELLATION,
                    (
                        CloseResultV1.CANCELLED
                        if isinstance(error, asyncio.CancelledError)
                        else CloseResultV1.FAILED
                    ),
                )
            else:
                observers_failed = any(
                    isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                    for result in observer_results
                )
                if observers_failed:
                    errors.extend(
                        result
                        for result in observer_results
                        if isinstance(result, BaseException)
                        and not isinstance(result, asyncio.CancelledError)
                    )
                self._observe_close_stage(
                    CloseStageV1.REVOKE_OBSERVER_CANCELLATION,
                    CloseResultV1.FAILED if observers_failed else CloseResultV1.SUCCEEDED,
                )
        if self._revoke_observer_failures:
            errors.extend(self._revoke_observer_failures)
            self._revoke_observer_failures.clear()
        writer = self._writer
        if writer is not None:
            try:
                await self._drain_before_writer_stop()
            except BaseException as error:
                errors.append(error)
                self._observe_close_stage(
                    CloseStageV1.WRITER_DRAIN,
                    (
                        CloseResultV1.CANCELLED
                        if isinstance(error, asyncio.CancelledError)
                        else CloseResultV1.FAILED
                    ),
                )
            else:
                self._observe_close_stage(CloseStageV1.WRITER_DRAIN, CloseResultV1.SUCCEEDED)
                try:
                    stopped = await asyncio.to_thread(writer.close, _CLOSE_WRITER_TIMEOUT_SECONDS)
                except BaseException as error:
                    errors.append(error)
                    self._observe_close_stage(
                        CloseStageV1.WRITER_STOP,
                        (
                            CloseResultV1.CANCELLED
                            if isinstance(error, asyncio.CancelledError)
                            else CloseResultV1.FAILED
                        ),
                    )
                else:
                    if stopped:
                        self._writer = None
                        self._observe_close_stage(
                            CloseStageV1.WRITER_STOP,
                            CloseResultV1.SUCCEEDED,
                        )
                    else:
                        errors.append(
                            RuntimeError("evidence writer did not stop before close timeout")
                        )
                        self._observe_close_stage(CloseStageV1.WRITER_STOP, CloseResultV1.TIMED_OUT)
        # The transport remains owned by a live writer.  Closing it after a
        # drain/stop failure would make a retry incapable of safely completing
        # the exact admitted terminal work.
        if self._writer is None:
            transport = self._transport
            close_transport = getattr(transport, "close", None)
            if callable(close_transport):
                try:
                    closed = await asyncio.to_thread(close_transport)
                except BaseException as error:
                    errors.append(error)
                    self._observe_close_stage(
                        CloseStageV1.TRANSPORT_CLOSE,
                        (
                            CloseResultV1.CANCELLED
                            if isinstance(error, asyncio.CancelledError)
                            else CloseResultV1.FAILED
                        ),
                    )
                else:
                    if closed is False:
                        errors.append(RuntimeError("evidence transport close reported failure"))
                        self._observe_close_stage(
                            CloseStageV1.TRANSPORT_CLOSE,
                            CloseResultV1.FAILED,
                        )
                    else:
                        self._transport = None
                        self._observe_close_stage(
                            CloseStageV1.TRANSPORT_CLOSE,
                            CloseResultV1.SUCCEEDED,
                        )
        activation_result, activation_error = await self._observe_active_consent_activation()
        if activation_result is not None:
            self._observe_close_stage(CloseStageV1.CONSENT_SETTLEMENT, activation_result)
        if activation_error is not None:
            errors.append(activation_error)
        async with self._state_lock:
            was_published = self._published
            self._published = False
            if not errors:
                if not was_published:
                    self._release_unpublished_consent_for_close_locked()
                self._release_retained_lifecycle_status_reservations()
                self._queue = None
                self._admission = None
                self._admission_view = None
                self._transport = None
                self._pending_create = None
                self._close_drain_authority = None
                self._close_drain_ticket = None
                self._epoch_retiring = False
                self._invalidation_authority = None
                self._invalidation_operation = None
                self._consent_settlement_authority = None
                self._consent_settlement_operation = None
                self._consent_settlement_claimed = False
                self._consent_activation_authority = None
                self._consent_activation_operation = None
                self._pending_evidence_consent_key_reset()
                with self._rollover_lock:
                    self._active_rollover = None
                    self._completed_rollovers.clear()
                    self._rollover_completion_claimed.clear()
                self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("host evidence runtime close failed", errors)

    async def _cancel_and_observe_consent_settlement(
        self,
    ) -> tuple[CloseResultV1 | None, BaseException | None]:
        """Settle or cancel the one retained activation task before owner cleanup.

        The runtime, rather than a browser request, owns this task.  Close is
        therefore responsible for observing its terminal outcome before it can
        release the create reservation, writer, or projection lineage.
        """

        async with self._state_lock:
            operation = self._consent_settlement_operation
        if operation is None or operation is asyncio.current_task():
            return None, None
        result: ConsentDisposition | BaseException
        if operation.done():
            # A terminal task can outlive its loop. Reading its outcome needs
            # no new future or callback scheduled on that stopped loop.
            try:
                result = operation.result()
            except BaseException as error:
                result = error
        else:
            operation.cancel()
            result = (await asyncio.gather(operation, return_exceptions=True))[0]
        if isinstance(result, asyncio.CancelledError):
            return CloseResultV1.CANCELLED, None
        if isinstance(result, BaseException):
            return CloseResultV1.FAILED, result
        return CloseResultV1.SUCCEEDED, None

    async def _observe_active_consent_activation(
        self,
    ) -> tuple[CloseResultV1 | None, BaseException | None]:
        """Wait for the gateway task that owns the primary consent-status slot."""

        async with self._state_lock:
            operation = self._consent_activation_operation
        if operation is None or operation is asyncio.current_task():
            return None, None
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            return CloseResultV1.CANCELLED, None
        except BaseException as error:
            return CloseResultV1.FAILED, error
        return CloseResultV1.SUCCEEDED, None

    async def _await_or_recover_epoch_retirement_for_close(
        self,
        retirement: asyncio.Task[None],
    ) -> None:
        """Retry one failed retained epoch once under the close owner's authority."""

        try:
            await asyncio.shield(retirement)
            return
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass
        async with self._state_lock:
            current = self._invalidation_operation
            if current is not retirement:
                if current is None:
                    if not self._epoch_retiring:
                        return
                    raise RuntimeError("retiring evidence epoch lost its recovery authority")
                recovery = current
            else:
                if not (
                    retirement.done()
                    and (retirement.cancelled() or retirement.exception() is not None)
                ):
                    recovery = retirement
                else:
                    recovery = asyncio.create_task(
                        self._retire_invalidated_epoch(),
                        name="host-evidence-runtime-close-epoch-recovery",
                    )
                    self._invalidation_operation = recovery
        await asyncio.shield(recovery)

    def _observe_close_stage(self, stage: CloseStageV1, result: CloseResultV1) -> None:
        self._production_observation_recorder.record_close_stage(stage=stage, result=result)

    def _release_unpublished_consent_for_close_locked(self) -> None:
        """Release one never-dispatched create reservation during owned close.

        A dispatched create is settled by the writer/drain stages before this
        point.  Only the exact reserved phase remains directly releasable here.
        This method is called under ``_state_lock`` and is idempotent after the
        pending authority has been cleared.
        """

        authority = self._pending_consent_authority
        ticket = self._pending_consent_ticket
        admission = self._admission
        if authority is not None and ticket is None and admission is not None:
            admission.release_create_epoch_reservation(authority.create_epoch_reservation)
        projection_authority = self._pending_consent_projection_authority
        projection_reservation = self._pending_consent_projection_reservation
        release_projection = self._release_pending_consent_projection
        if projection_authority is not None or projection_reservation is not None:
            if (
                projection_authority is None
                or (authority is not None and projection_authority is not authority)
                or projection_reservation is None
                or release_projection is None
            ):
                raise RuntimeError("pending consent projection reservation ownership conflicts")
            release_projection((projection_reservation,))
            self._pending_consent_projection_authority = None
            self._pending_consent_projection_reservation = None
            self._release_pending_consent_projection = None

    async def _drain_before_writer_stop(self) -> None:
        """Use the admission owner's one terminal drain before stopping its consumer."""

        async with self._state_lock:
            admission = self._admission
            if admission is None:
                raise RuntimeError("active evidence writer has no admission owner")
            authority = self._close_drain_authority
            if authority is None:
                authority = self._lifecycle_owner.mint_drain_authority(
                    final_admission_ordinal=admission.final_admission_ordinal,
                )
                self._close_drain_authority = authority
            ticket = self._close_drain_ticket
            if (
                ticket is not None
                and ticket.terminal_event.is_set()
                and ticket.disposition is not DrainDisposition.STOPPED
                and ticket.disposition is not DrainDisposition.WRITER_FAULT
            ):
                # A timed-out/refused drain has no terminal ownership result.  Keep
                # the exact authority but ask admission to enqueue its bounded retry.
                ticket = None
                self._close_drain_ticket = None
            if ticket is None:
                if (
                    self._qualification_checkpoint_channel is not None
                    and not self._qualification_drain_checkpoint_emitted
                ):
                    await self._qualification_checkpoint_channel.emit("host_drain_started")
                    self._qualification_drain_checkpoint_emitted = True
                ticket = admission.request_drain(authority)
                self._close_drain_ticket = ticket
        terminal = await _wait_for_thread_signal(
            ticket.terminal_event,
            timeout_seconds=_CLOSE_DRAIN_TIMEOUT_SECONDS,
        )
        if not terminal:
            raise RuntimeError("evidence drain did not stop before close timeout")
        if ticket.disposition is not DrainDisposition.STOPPED:
            raise RuntimeError("evidence drain did not reach terminal stopped state")
