"""One-shot execution of authoritative proactive conversation updates."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from hermes_realtime.evidence import (
    AppendDisposition,
    ConversationOperationKind,
    ConversationOperationReservation,
    ConversationOperationScheduler,
    EvidenceAdmissionControllerV1,
    EvidenceAdmissionViewV1,
    EvidenceTurnLease,
    ProactiveTurnAuthorityV1,
    ReplayTurnAuthorityV1,
    ReservationError,
    UserTurnAuthorityV1,
)
from hermes_realtime.evidence.lifecycle import EvidenceConversationAuthorityV1
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    _ProductionObservationRecorderV1,
)
from hermes_realtime.speech import Transcript

from .context import ConversationMessage, ConversationRole
from .streaming import StreamingSpeechLoop, _UpdateSpeechAuthority
from .updates import (
    ConversationUpdateDirector,
    UpdateDecision,
    UpdateDecisionKind,
    _UpdateActionClaim,
)

_ACTION_DISPOSITIONS = frozenset(
    ("delivered", "failed", "superseded", "requeued")
)
_MAX_MENTIONS_PER_TURN = 16
_MAX_OWNER_GENERATION = 2**63 - 1
_OWNER_GENERATION_LOCK = Lock()
_NEXT_OWNER_GENERATION = 1


def _allocate_owner_generation() -> int:
    global _NEXT_OWNER_GENERATION
    with _OWNER_GENERATION_LOCK:
        generation = _NEXT_OWNER_GENERATION
        if generation > _MAX_OWNER_GENERATION:
            raise RuntimeError("conversation operation owner generation overflowed")
        _NEXT_OWNER_GENERATION += 1
        return generation


@dataclass(frozen=True, slots=True)
class UpdateActionRecord:
    """Bounded inspectable disposition of one claimed proactive action."""

    decision: UpdateDecision
    disposition: str

    def __post_init__(self) -> None:
        if type(self.decision) is not UpdateDecision:
            raise TypeError("decision must be an exact UpdateDecision")
        decision = UpdateDecision(
            sequence=self.decision.sequence,
            completion=self.decision.completion,
            kind=self.decision.kind,
            text=self.decision.text,
        )
        if type(self.disposition) is not str:
            raise TypeError("action disposition must be an exact built-in string")
        if self.disposition not in _ACTION_DISPOSITIONS:
            raise ValueError("action disposition is unsupported")
        object.__setattr__(self, "decision", decision)


class ConversationUpdateExecutor:
    """Route director claims only through the authoritative streaming speech loop."""

    def __init__(
        self,
        *,
        director: ConversationUpdateDirector,
        speech: StreamingSpeechLoop,
        max_action_records: int = 256,
        max_owned_operations: int = 16,
        cleanup_timeout_seconds: float = 1.0,
        operation_scheduler: ConversationOperationScheduler | None = None,
        evidence_lifecycle: EvidenceConversationAuthorityV1 | None = None,
        evidence_admission: EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1 | None = None,
        evidence_resolver: Callable[
            [],
            tuple[
                EvidenceConversationAuthorityV1 | None,
                EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1 | None,
            ],
        ]
        | None = None,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
    ) -> None:
        if type(director) is not ConversationUpdateDirector:
            raise TypeError("director must be an exact ConversationUpdateDirector")
        if type(speech) is not StreamingSpeechLoop:
            raise TypeError("speech must be an exact StreamingSpeechLoop")
        if type(max_action_records) is not int or not 1 <= max_action_records <= 4096:
            raise ValueError("max_action_records must be between 1 and 4096")
        if type(max_owned_operations) is not int:
            raise TypeError("max_owned_operations must be an exact integer")
        if not 1 <= max_owned_operations <= 64:
            raise ValueError("max_owned_operations must be between 1 and 64")
        if type(cleanup_timeout_seconds) not in (int, float):
            raise TypeError("cleanup_timeout_seconds must be an exact number")
        if not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be finite and positive")
        if (
            operation_scheduler is not None
            and type(operation_scheduler) is not ConversationOperationScheduler
        ):
            raise TypeError("operation_scheduler must be exact or None")
        if (
            operation_scheduler is not None
            and operation_scheduler.max_operations != max_owned_operations
        ):
            raise ValueError("operation_scheduler limit must match max_owned_operations")
        if (
            evidence_lifecycle is not None
            and type(evidence_lifecycle) is not EvidenceConversationAuthorityV1
        ):
            raise TypeError("evidence_lifecycle must be exact or None")
        if (
            evidence_admission is not None
            and type(evidence_admission)
            not in (EvidenceAdmissionControllerV1, EvidenceAdmissionViewV1)
        ):
            raise TypeError("evidence_admission must be an exact admission surface or None")
        if evidence_resolver is not None and not callable(evidence_resolver):
            raise TypeError("evidence_resolver must be callable or None")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        if evidence_resolver is not None and (
            evidence_lifecycle is not None or evidence_admission is not None
        ):
            raise ValueError("dynamic and static evidence configuration are mutually exclusive")
        self._director = director
        self._speech = speech
        director._bind_action_executor(self)
        try:
            self._speech_authority: _UpdateSpeechAuthority = (
                speech._bind_update_executor(self)
            )
        except BaseException:
            director._unbind_action_executor(self)
            raise
        self._records: deque[UpdateActionRecord] = deque(maxlen=max_action_records)
        self._action_recorded = asyncio.Event()
        self._dropped_record_count = 0
        self._cleanup_timeout = float(cleanup_timeout_seconds)
        self._operation_scheduler = (
            operation_scheduler
            if operation_scheduler is not None
            else ConversationOperationScheduler(
                owner_generation=_allocate_owner_generation(),
                max_operations=max_owned_operations,
            )
        )
        self._evidence_lifecycle = evidence_lifecycle
        self._evidence_admission = evidence_admission
        self._evidence_resolver = evidence_resolver
        self._production_observation_recorder = production_observation_recorder
        if (
            evidence_lifecycle is not None
            and evidence_lifecycle.owner_generation
            != self._operation_scheduler.owner_generation
        ):
            raise ValueError("evidence lifecycle owner must match operation scheduler")
        if (
            evidence_admission is not None
            and evidence_admission.operation_scheduler is not self._operation_scheduler
        ):
            raise ValueError("evidence admission must share operation scheduler")
        self._decision_consumer: asyncio.Task[None] | None = None
        self._owned_operations: set[asyncio.Task[None]] = set()
        self._operation_reservations: dict[
            asyncio.Task[None],
            ConversationOperationReservation,
        ] = {}
        self._proactive_authorities: dict[
            ConversationOperationReservation,
            ProactiveTurnAuthorityV1,
        ] = {}
        self._idle_release_operations: set[asyncio.Task[None]] = set()
        self._foreground_cancel_operation: asyncio.Task[None] | None = None
        self._interrupt_claims: dict[
            asyncio.Task[None],
            tuple[_UpdateActionClaim, UpdateDecision],
        ] = {}
        self._interrupt_transfer_origins: dict[
            asyncio.Task[None],
            asyncio.Task[None],
        ] = {}
        self._mention_drain_operation: asyncio.Task[None] | None = None
        self._close_operation: asyncio.Task[None] | None = None
        self._terminal_error: BaseException | None = None
        self._started = False
        self._closed = False

    def _resolve_evidence_pair(
        self,
    ) -> tuple[
        EvidenceConversationAuthorityV1 | None,
        EvidenceAdmissionControllerV1 | EvidenceAdmissionViewV1 | None,
    ]:
        resolver = self._evidence_resolver
        if resolver is None:
            lifecycle = self._evidence_lifecycle
            admission = self._evidence_admission
        else:
            resolved = resolver()
            if type(resolved) is not tuple or len(resolved) != 2:
                raise TypeError("evidence resolver must return an exact pair")
            lifecycle, admission = resolved
        if lifecycle is not None and type(lifecycle) is not EvidenceConversationAuthorityV1:
            raise TypeError("resolved evidence authority must be exact or None")
        if admission is not None and type(admission) not in (
            EvidenceAdmissionControllerV1,
            EvidenceAdmissionViewV1,
        ):
            raise TypeError("resolved evidence admission must be an exact surface or None")
        if (lifecycle is None) is not (admission is None):
            raise RuntimeError("evidence lifecycle and admission must publish atomically")
        if lifecycle is not None:
            if lifecycle.owner_generation != self._operation_scheduler.owner_generation:
                raise RuntimeError("resolved lifecycle owner does not match scheduler")
            assert admission is not None
            if admission.operation_scheduler is not self._operation_scheduler:
                raise RuntimeError("resolved admission does not share scheduler")
        return lifecycle, admission

    @property
    def foreground_active(self) -> bool:
        """Whether the authoritative speech loop owns a live foreground turn."""

        return self._speech.foreground_active

    @property
    def dropped_action_record_count(self) -> int:
        return self._dropped_record_count

    @property
    def active_operation_count(self) -> int:
        return len(self._owned_operations)

    @property
    def reserved_operation_count(self) -> int:
        return self._operation_scheduler.active_count

    @property
    def operation_scheduler(self) -> ConversationOperationScheduler:
        return self._operation_scheduler

    def action_records(self) -> tuple[UpdateActionRecord, ...]:
        return tuple(
            UpdateActionRecord(
                decision=record.decision,
                disposition=record.disposition,
            )
            for record in self._records
        )

    async def wait_for_action_record(self, sequence: int) -> UpdateActionRecord:
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("action sequence must be a positive integer")
        while True:
            for record in self._records:
                if record.decision.sequence == sequence:
                    return UpdateActionRecord(
                        decision=record.decision,
                        disposition=record.disposition,
                    )
            if self._closed:
                raise RuntimeError("conversation update executor is closed")
            self._action_recorded.clear()
            for record in self._records:
                if record.decision.sequence == sequence:
                    return UpdateActionRecord(
                        decision=record.decision,
                        disposition=record.disposition,
                    )
            await self._action_recorded.wait()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("conversation update executor is closed")
        self._raise_terminal_error()
        if self._started:
            return
        self._director.start()
        self._decision_consumer = asyncio.create_task(
            self._consume_decisions(),
            name="conversation-update-actions",
        )
        self._started = True

    def reserve_response(self) -> ConversationOperationReservation:
        """Own response capacity before any caller-side turn mutation."""

        if not self._started:
            raise RuntimeError("conversation update executor is not started")
        if self._closed:
            raise RuntimeError("conversation update executor is closed")
        self._raise_terminal_error()
        reservation = self._operation_scheduler.try_reserve(
            ConversationOperationKind.RESPONSE
        )
        if reservation is None:
            raise RuntimeError("conversation update operation capacity exhausted")
        return reservation

    def release_response(
        self,
        reservation: ConversationOperationReservation,
    ) -> None:
        """Release a response bearer not transferred into ``respond``."""

        self._operation_scheduler.validate(
            reservation,
            ConversationOperationKind.RESPONSE,
            require_unconsumed=True,
        )
        self._operation_scheduler.release(reservation)

    async def respond(
        self,
        turn_id: str,
        transcript: Transcript,
        authority: UserTurnAuthorityV1 | None = None,
        *,
        reservation: ConversationOperationReservation | None = None,
    ) -> None:
        if not self._started:
            raise RuntimeError("conversation update executor is not started")
        if self._closed:
            raise RuntimeError("conversation update executor is closed")
        self._raise_terminal_error()
        supplied_reservation = reservation is not None
        if supplied_reservation:
            assert reservation is not None
            self._operation_scheduler.validate(
                reservation,
                ConversationOperationKind.RESPONSE,
                require_unconsumed=True,
            )
        try:
            if type(turn_id) is not str:
                raise TypeError("turn_id must be an exact built-in string")
            if not turn_id.strip():
                raise ValueError("turn_id must not be blank")
            if type(transcript) is not Transcript:
                raise TypeError("transcript must be an exact Transcript value")
            if authority is not None and type(authority) is not UserTurnAuthorityV1:
                raise TypeError("authority must be exact or None")
            transcript = Transcript(text=transcript.text, final=transcript.final)
            if not transcript.final:
                raise ValueError("only final transcripts can start a response")
            ConversationMessage(
                role=ConversationRole.USER.value,
                text=transcript.text,
            )
        except BaseException:
            if supplied_reservation:
                assert reservation is not None
                self._operation_scheduler.release(reservation)
            raise
        if reservation is None:
            reservation = self.reserve_response()
        evidence_lease: EvidenceTurnLease | None = None
        _, admission = self._resolve_evidence_pair()
        if admission is not None:
            if authority is None:
                self._operation_scheduler.release(reservation)
                raise RuntimeError("evidence admission requires user turn authority")
            result = admission.try_reserve_user_turn(authority, reservation)
            if result.disposition is AppendDisposition.ADMITTED:
                evidence_lease = result.lease
                if type(evidence_lease) is not EvidenceTurnLease:
                    self._operation_scheduler.release(reservation)
                    raise RuntimeError("admitted evidence result requires an exact lease")
                final_disposition = admission.try_admit_user_final(
                    evidence_lease,
                    authority,
                    transcript.text,
                )
                if final_disposition is not AppendDisposition.ADMITTED:
                    if not admission.discard_unopened_user_turn(evidence_lease, authority):
                        self._operation_scheduler.release(reservation)
                        raise RuntimeError("failed to retire rejected evidence user turn")
                    self._operation_scheduler.release(reservation)
                    raise RuntimeError(
                        "evidence final-input admission rejected: "
                        f"{final_disposition.value}"
                    )
        coroutine = self._execute_response(turn_id, transcript, evidence_lease)
        try:
            operation = asyncio.create_task(
                coroutine,
                name=f"conversation-update-response:{turn_id}",
            )
        except BaseException as spawn_error:
            coroutine.close()
            settlement_error: BaseException | None = None
            if evidence_lease is not None:
                assert admission is not None
                disposition = admission.settle_spawn_failed(evidence_lease)
                recorder = self._production_observation_recorder
                if recorder is not None and disposition is AppendDisposition.ADMITTED:
                    outcome = admission.settled_terminal_outcome(evidence_lease)
                    if outcome is not None:
                        recorder.record_terminal_settled(
                            terminal_disposition=outcome.terminal_disposition,
                            terminal_reason=outcome.terminal_reason,
                            context_committed=outcome.context_committed,
                        )
                if disposition is not AppendDisposition.ADMITTED:
                    settlement_error = RuntimeError(
                        "evidence spawn-failure settlement rejected: "
                        f"{disposition.value}"
                    )
            self._operation_scheduler.release(reservation)
            if settlement_error is not None:
                raise BaseExceptionGroup(
                    "response spawn and evidence settlement failed",
                    [spawn_error, settlement_error],
                ) from None
            raise
        self._owned_operations.add(operation)
        self._operation_reservations[operation] = reservation
        self._idle_release_operations.add(operation)
        operation.add_done_callback(self._operation_done)
        await asyncio.shield(operation)

    async def cancel_foreground(self) -> None:
        """Cancel only current foreground speech, preserving background work."""

        if not self._started:
            raise RuntimeError("conversation update executor is not started")
        if self._closed:
            raise RuntimeError("conversation update executor is closed")
        self._raise_terminal_error()
        await self._cancel_foreground_for_cleanup()

    @property
    def foreground_turn_id(self) -> str | None:
        """Return the exact current speech owner, if one exists."""

        return self._speech.active_turn_id

    async def cancel_foreground_if(self, turn_id: str) -> bool:
        """Cancel foreground only if ``turn_id`` still owns it."""

        if not self._started:
            raise RuntimeError("conversation update executor is not started")
        if self._closed:
            raise RuntimeError("conversation update executor is closed")
        self._raise_terminal_error()
        cancelled = await self._speech.cancel_if_active(turn_id)
        if type(cancelled) is not bool:
            raise TypeError("speech conditional cancellation result must be an exact boolean")
        return cancelled

    async def resume_foreground(self) -> bool:
        """Replay a retained unconfirmed suffix without starting inference."""

        if not self._started:
            raise RuntimeError("conversation update executor is not started")
        if self._closed:
            raise RuntimeError("conversation update executor is closed")
        self._raise_terminal_error()
        replay_identity = self._speech._eligible_replay_identity()
        lifecycle, admission = self._resolve_evidence_pair()
        if lifecycle is not None and replay_identity is None:
            return False
        reservation = self._operation_scheduler.try_reserve(
            ConversationOperationKind.REPLAY
        )
        if reservation is None:
            raise RuntimeError("conversation update operation capacity exhausted")
        evidence_lease: EvidenceTurnLease | None = None
        replay_authority: ReplayTurnAuthorityV1 | None = None
        if lifecycle is not None:
            assert admission is not None
            assert replay_identity is not None
            replay_authority = lifecycle.mint_replay_turn(
                reservation,
                replay_of_evidence_turn_id=replay_identity.replay_of_evidence_turn_id,
                replay_generation=replay_identity.replay_generation,
                source_binding_id=replay_identity.source_binding_id,
                source_binding_generation=replay_identity.source_binding_generation,
                source_logical_session_id=replay_identity.source_logical_session_id,
            )
            result = admission.try_reserve_replay_turn(
                replay_authority,
                reservation,
            )
            if result.disposition is not AppendDisposition.ADMITTED:
                self._operation_scheduler.release(reservation)
                return False
            evidence_lease = result.lease
            if type(evidence_lease) is not EvidenceTurnLease:
                self._operation_scheduler.release(reservation)
                raise RuntimeError("admitted replay result requires an exact lease")
        unopened_retirement_attempted = False

        def discard_unopened_replay_lease(*, required: bool) -> None:
            nonlocal unopened_retirement_attempted
            if evidence_lease is None:
                return
            if unopened_retirement_attempted:
                return
            unopened_retirement_attempted = True
            assert admission is not None
            assert replay_authority is not None
            discarded = admission.discard_unopened_replay_turn(
                evidence_lease,
                replay_authority,
            )
            if required and not discarded:
                raise RuntimeError("unopened replay evidence lease could not be retired")

        resumed_results: list[bool] = []

        async def resume() -> None:
            try:
                if evidence_lease is None:
                    result = await self._speech.resume_interrupted()
                else:
                    result = await self._speech.resume_interrupted(evidence_lease)
                if type(result) is not bool:
                    raise TypeError("speech resume result must be an exact boolean")
                if not result:
                    discard_unopened_replay_lease(required=True)
                resumed_results.append(result)
            except BaseException:
                # An early close or startup failure leaves the lease unopened.
                # Once speech opened it, _run_turn owns terminal settlement and
                # this exact retirement returns False.
                discard_unopened_replay_lease(required=False)
                raise

        coroutine = resume()
        try:
            operation = asyncio.create_task(
                coroutine,
                name="conversation-update-foreground-resume",
            )
        except BaseException:
            coroutine.close()
            discard_unopened_replay_lease(required=True)
            self._operation_scheduler.release(reservation)
            raise
        self._owned_operations.add(operation)
        self._operation_reservations[operation] = reservation
        self._idle_release_operations.add(operation)
        operation.add_done_callback(self._operation_done)
        await asyncio.shield(operation)
        if len(resumed_results) != 1:
            raise RuntimeError("speech resume did not publish one result")
        return resumed_results[0]

    async def _cancel_foreground_for_cleanup(
        self,
        *,
        host_shutdown: bool = False,
        binding_closed: bool = False,
    ) -> None:
        """Settle foreground speech even after new action admission is closed."""

        if type(host_shutdown) is not bool or type(binding_closed) is not bool:
            raise TypeError("cleanup reason flags must be exact booleans")
        if host_shutdown and binding_closed:
            raise ValueError("cleanup reason flags are mutually exclusive")

        operation = self._foreground_cancel_operation
        if operation is None or operation.done():
            if host_shutdown:
                cancellation = self._speech.cancel_for_host_shutdown()
            elif binding_closed:
                cancellation = self._speech.cancel_for_binding_close()
            else:
                cancellation = self._speech.cancel()
            operation = asyncio.create_task(
                cancellation,
                name=(
                    "conversation-update-foreground-host-shutdown"
                    if host_shutdown
                    else (
                        "conversation-update-foreground-binding-close"
                        if binding_closed
                        else "conversation-update-foreground-cancel"
                    )
                ),
            )
            self._foreground_cancel_operation = operation
            self._owned_operations.add(operation)
            operation.add_done_callback(self._operation_done)
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            raise
        except BaseException:
            raise

    async def close(self) -> None:
        operation = self._close_operation
        if operation is None or self._close_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="conversation-update-executor-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    @staticmethod
    def _close_failed(operation: asyncio.Task[None]) -> bool:
        if not operation.done():
            return False
        if operation.cancelled():
            return True
        return operation.exception() is not None

    async def _execute_response(
        self,
        turn_id: str,
        transcript: Transcript,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> None:
        claims = self._director._claim_pending_mentions(
            self,
            max_count=_MAX_MENTIONS_PER_TURN,
        )
        decisions = tuple(
            self._director._decision_for_action_claim(self, claim)
            for claim in claims
        )
        operation = None
        disposition = "delivered"
        try:
            operation = self._speech._begin_update_operation(
                self._speech_authority,
                turn_id,
            )
            if evidence_lease is None:
                await self._speech._respond_with_updates(
                    self._speech_authority,
                    operation,
                    turn_id,
                    transcript,
                    decisions,
                )
            else:
                await self._speech._respond_with_updates(
                    self._speech_authority,
                    operation,
                    turn_id,
                    transcript,
                    decisions,
                    evidence_lease,
                )
        except asyncio.CancelledError:
            disposition = "superseded"
            raise
        except BaseException:
            disposition = "failed"
            raise
        finally:
            delivered = False
            if operation is not None:
                delivered = self._speech._finish_update_operation(
                    self._speech_authority,
                    operation,
                )
            self._finish_mentions(
                claims,
                decisions,
                delivered=delivered,
                delivered_disposition=disposition,
            )

    def _finish_mentions(
        self,
        claims: tuple[_UpdateActionClaim, ...],
        decisions: tuple[UpdateDecision, ...],
        *,
        delivered: bool,
        delivered_disposition: str,
    ) -> None:
        if not claims:
            return
        if not delivered:
            self._director._release_mention_action_claims(self, claims)
            for decision in decisions:
                self._record(decision, "requeued")
            return
        for claim, decision in zip(claims, decisions, strict=True):
            self._director._settle_action_claim(self, claim)
            self._record(decision, delivered_disposition)

    async def _consume_decisions(self) -> None:
        active: asyncio.Task[None] | None = None
        try:
            while True:
                decision, claim = await self._director._next_execution(self)
                if decision.kind == UpdateDecisionKind.MENTION_NEXT.value:
                    if claim is not None:
                        raise RuntimeError("mention decision unexpectedly received an action claim")
                    self._start_idle_mention_drain()
                    continue
                if decision.kind != UpdateDecisionKind.INTERRUPT.value:
                    if claim is not None:
                        raise RuntimeError("non-interrupt decision received an action claim")
                    continue
                if claim is None:
                    raise RuntimeError("interrupt decision is missing its action claim")
                replacement = self._start_interrupt_operation(claim, decision)
                if replacement is None:
                    self._director._settle_action_claim(self, claim)
                    self._record(decision, "failed")
                    continue
                if active is not None:
                    done, _ = await asyncio.wait(
                        {active},
                        timeout=self._cleanup_timeout,
                    )
                    if not done:
                        raise TimeoutError(
                            "replaced proactive announcement did not settle"
                        )
                active = replacement
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not self._closed:
                self._terminal_error = error

    def _start_idle_mention_drain(self) -> None:
        if (
            self._closed
            or self._speech.foreground_active
            or not self._director.pending_mentions()
        ):
            return
        active = self._mention_drain_operation
        if active is not None and not active.done():
            return
        reservation = self._operation_scheduler.try_reserve(
            ConversationOperationKind.PROACTIVE
        )
        if reservation is None:
            return
        coroutine = self._drain_idle_mentions()
        try:
            operation = asyncio.create_task(
                coroutine,
                name="conversation-update-idle-mentions",
            )
        except BaseException:
            coroutine.close()
            self._operation_scheduler.release(reservation)
            raise
        self._mention_drain_operation = operation
        self._owned_operations.add(operation)
        self._operation_reservations[operation] = reservation
        operation.add_done_callback(self._mention_drain_done)

    async def _drain_idle_mentions(self) -> None:
        while not self._closed and not self._speech.foreground_active:
            claims = self._director._claim_pending_mentions(self, max_count=1)
            if not claims:
                return
            delivered = await self._execute_idle_mention(claims[0])
            if not delivered:
                return

    def _admit_proactive_evidence(
        self,
        reservation: ConversationOperationReservation,
    ) -> tuple[ProactiveTurnAuthorityV1 | None, EvidenceTurnLease | None]:
        """Mint and reserve evidence only from the pair published at use time."""

        lifecycle, admission = self._resolve_evidence_pair()
        if lifecycle is None:
            return None, None
        assert admission is not None
        authority = lifecycle.mint_proactive_turn(reservation)
        result = admission.try_reserve_proactive_turn(authority, reservation)
        if result.disposition is not AppendDisposition.ADMITTED:
            return authority, None
        lease = result.lease
        if type(lease) is not EvidenceTurnLease:
            raise RuntimeError("admitted proactive result requires an exact lease")
        return authority, lease

    async def _execute_idle_mention(self, claim: _UpdateActionClaim) -> bool:
        decision = self._director._decision_for_action_claim(self, claim)
        turn_id = f"mention_{decision.sequence}"
        task = asyncio.current_task()
        reservation = self._operation_reservations.get(task) if task is not None else None
        operation = None
        disposition = "delivered"
        delivered = False
        evidence_authority: ProactiveTurnAuthorityV1 | None = None
        evidence_lease: EvidenceTurnLease | None = None
        unopened_retirement_attempted = False

        def discard_unopened_proactive_lease(*, required: bool) -> None:
            nonlocal unopened_retirement_attempted
            if evidence_lease is None or unopened_retirement_attempted:
                return
            unopened_retirement_attempted = True
            _, admission = self._resolve_evidence_pair()
            if admission is None or evidence_authority is None:
                if required:
                    raise RuntimeError(
                        "proactive evidence owner disappeared before retirement"
                    )
                return
            discarded = admission.discard_unopened_proactive_turn(
                evidence_lease,
                evidence_authority,
            )
            if required and not discarded:
                raise RuntimeError("unopened proactive evidence lease could not be retired")

        try:
            if reservation is None:
                raise RuntimeError("proactive operation lost its reservation")
            evidence_authority, evidence_lease = self._admit_proactive_evidence(
                reservation
            )
            if evidence_authority is not None:
                self._proactive_authorities[reservation] = evidence_authority
            if evidence_authority is None:
                operation = self._speech._begin_update_operation(
                    self._speech_authority,
                    turn_id,
                )
            else:
                operation = self._speech._begin_update_operation(
                    self._speech_authority,
                    turn_id,
                    evidence_authority,
                )
            if evidence_lease is None:
                started = await self._speech._announce_idle_update(
                    self._speech_authority,
                    operation,
                    turn_id,
                    decision,
                )
            else:
                started = await self._speech._announce_idle_update(
                    self._speech_authority,
                    operation,
                    turn_id,
                    decision,
                    evidence_lease,
                )
            if not started:
                discard_unopened_proactive_lease(required=True)
                return False
        except asyncio.CancelledError:
            discard_unopened_proactive_lease(required=False)
            disposition = "superseded"
            raise
        except BaseException:
            discard_unopened_proactive_lease(required=False)
            disposition = "failed"
            raise
        finally:
            if operation is not None:
                delivered = self._speech._finish_update_operation(
                    self._speech_authority,
                    operation,
                )
            if delivered:
                self._director._settle_action_claim(self, claim)
                self._record(decision, disposition)
            elif self._director._action_claim_is_active(self, claim):
                self._director._release_mention_action_claims(self, (claim,))
                self._record(decision, "requeued")
        return delivered

    def _mention_drain_done(self, task: asyncio.Task[None]) -> None:
        self._operation_done(task)
        if self._mention_drain_operation is task:
            self._mention_drain_operation = None

    def _start_interrupt_operation(
        self,
        claim: _UpdateActionClaim,
        decision: UpdateDecision,
    ) -> asyncio.Task[None] | None:
        if type(claim) is not _UpdateActionClaim:
            raise TypeError("interrupt claim must be an exact action claim")
        if type(decision) is not UpdateDecision:
            raise TypeError("interrupt decision must be exact")
        idle = self._mention_drain_operation
        idle_reservation = (
            self._operation_reservations.get(idle) if idle is not None else None
        )
        transferred_idle: asyncio.Task[None] | None = None
        if idle is not None and not idle.done() and idle_reservation is not None:
            reservation = self._operation_scheduler.transfer_idle_proactive(
                idle_reservation
            )
            self._proactive_authorities.pop(idle_reservation, None)
            del self._operation_reservations[idle]
            transferred_idle = idle
        else:
            new_reservation = self._operation_scheduler.try_reserve(
                ConversationOperationKind.PROACTIVE
            )
            if new_reservation is None:
                return None
            reservation = new_reservation
        coroutine = self._execute_interrupt(claim)
        try:
            operation = asyncio.create_task(
                coroutine,
                name=f"conversation-update-interrupt:{decision.sequence}",
            )
        except BaseException:
            coroutine.close()
            if transferred_idle is None:
                self._operation_scheduler.release(reservation)
            else:
                restored = self._operation_scheduler.transfer_idle_proactive(reservation)
                self._operation_reservations[transferred_idle] = restored
            self._director._settle_action_claim(self, claim)
            self._record(decision, "failed")
            raise
        self._owned_operations.add(operation)
        self._operation_reservations[operation] = reservation
        self._idle_release_operations.add(operation)
        self._interrupt_claims[operation] = (claim, decision)
        if transferred_idle is not None:
            self._interrupt_transfer_origins[operation] = transferred_idle
        operation.add_done_callback(self._operation_done)
        return operation

    async def _execute_interrupt(self, claim: _UpdateActionClaim) -> None:
        decision = self._director._decision_for_action_claim(self, claim)
        turn_id = f"update_{decision.sequence}"
        task = asyncio.current_task()
        reservation = self._operation_reservations.get(task) if task is not None else None
        if reservation is None:
            raise RuntimeError("proactive operation lost its reservation")
        evidence_authority, evidence_lease = self._admit_proactive_evidence(reservation)
        if evidence_authority is not None:
            self._proactive_authorities[reservation] = evidence_authority
        if evidence_authority is None:
            operation = self._speech._begin_update_operation(
                self._speech_authority,
                turn_id,
            )
        else:
            operation = self._speech._begin_update_operation(
                self._speech_authority,
                turn_id,
                evidence_authority,
            )
        disposition = "delivered"
        try:
            if evidence_lease is None:
                await self._speech._announce_update(
                    self._speech_authority,
                    operation,
                    turn_id,
                    decision,
                )
            else:
                await self._speech._announce_update(
                    self._speech_authority,
                    operation,
                    turn_id,
                    decision,
                    evidence_lease,
                )
        except asyncio.CancelledError:
            disposition = "superseded"
            raise
        except BaseException:
            disposition = "failed"
            raise
        finally:
            delivered = self._speech._finish_update_operation(
                self._speech_authority,
                operation,
            )
            if disposition == "delivered" and not delivered:
                disposition = "failed"
            self._director._settle_action_claim(self, claim)
            self._record(decision, disposition)

    def _operation_done(self, task: asyncio.Task[None]) -> None:
        self._owned_operations.discard(task)
        succeeded = not task.cancelled() and task.exception() is None
        reservation = self._operation_reservations.pop(task, None)
        transfer_origin = self._interrupt_transfer_origins.pop(task, None)
        if reservation is not None:
            self._proactive_authorities.pop(reservation, None)
            restored = False
            if transfer_origin is not None and not succeeded and not transfer_origin.done():
                try:
                    replacement = self._operation_scheduler.transfer_idle_proactive(
                        reservation
                    )
                except ReservationError:
                    pass
                else:
                    self._operation_reservations[transfer_origin] = replacement
                    restored = True
            if not restored:
                with contextlib.suppress(ReservationError):
                    self._operation_scheduler.release(reservation)
        releases_idle_floor = task in self._idle_release_operations
        self._idle_release_operations.discard(task)
        interrupt = self._interrupt_claims.pop(task, None)
        if interrupt is not None:
            claim, decision = interrupt
            if self._director._action_claim_is_active(self, claim):
                self._director._settle_action_claim(self, claim)
                self._record(
                    decision,
                    "superseded" if task.cancelled() else "failed",
                )
        with contextlib.suppress(BaseException):
            task.result()
        if succeeded and releases_idle_floor:
            self._start_idle_mention_drain()

    async def _close_owned(self) -> None:
        self._closed = True
        self._operation_scheduler.close()
        self._action_recorded.set()
        recorder = self._production_observation_recorder
        errors: list[BaseException] = []
        try:
            await self._cancel_foreground_for_cleanup(host_shutdown=True)
        except BaseException as error:
            errors.append(error)
        consumer = self._decision_consumer
        if consumer is not None and not consumer.done():
            consumer.cancel()
        if consumer is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(consumer),
                    timeout=self._cleanup_timeout,
                )
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                errors.append(error)
        owned = tuple(self._owned_operations)
        for task in owned:
            if not task.done():
                task.cancel()
        if owned:
            done, pending = await asyncio.wait(
                owned,
                timeout=self._cleanup_timeout,
            )
            for task in done:
                self._operation_done(task)
            if pending:
                errors.append(
                    TimeoutError("update action cleanup exceeded finite timeout")
                )
        try:
            await self._director.close()
        except BaseException as error:
            errors.append(error)
        try:
            await self._speech.close()
        except BaseException as error:
            errors.append(error)
        if self._terminal_error is not None:
            errors.insert(0, self._terminal_error)
            self._terminal_error = None
        if len(errors) == 1:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.UPDATE_EXECUTOR,
                    result=CloseResultV1.FAILED,
                )
            raise errors[0]
        if errors:
            if recorder is not None:
                recorder.record_close_stage(
                    stage=CloseStageV1.UPDATE_EXECUTOR,
                    result=CloseResultV1.FAILED,
                )
            raise BaseExceptionGroup("conversation update executor close failed", errors)
        if recorder is not None:
            recorder.record_close_stage(
                stage=CloseStageV1.UPDATE_EXECUTOR,
                result=CloseResultV1.SUCCEEDED,
            )

    def _record(self, decision: UpdateDecision, disposition: str) -> None:
        if len(self._records) == self._records.maxlen:
            self._records.popleft()
            self._dropped_record_count += 1
        self._records.append(
            UpdateActionRecord(decision=decision, disposition=disposition)
        )
        self._action_recorded.set()

    def _raise_terminal_error(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error
