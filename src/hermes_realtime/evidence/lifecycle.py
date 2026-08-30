"""Process-local ownership for realtime evidence lifecycle authorities."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from threading import Lock
from typing import TypeVar, cast
from uuid import UUID, uuid4

from .admission import (
    ConversationOperationReservation,
    ConversationOperationScheduler,
    ReservationError,
)
from .models import (
    BindingCloseAuthorityV1,
    BindingCloseReason,
    CommandAdmissionAuthorityV1,
    ConversationOperationKind,
    FinalInputAuthorityV1,
    InputSource,
    LifecycleDrainAuthorityV1,
    LifecycleSealAuthorityV1,
    ProactiveTurnAuthorityV1,
    ReplayTurnAuthorityV1,
    RolloverAuthorityV1,
    UserTurnAuthorityV1,
)

_MAX_UNSIGNED_63 = 2**63 - 1
_Capability = TypeVar(
    "_Capability",
    BindingCloseAuthorityV1,
    CommandAdmissionAuthorityV1,
    FinalInputAuthorityV1,
    LifecycleDrainAuthorityV1,
    LifecycleSealAuthorityV1,
    ProactiveTurnAuthorityV1,
    ReplayTurnAuthorityV1,
    RolloverAuthorityV1,
    UserTurnAuthorityV1,
)

_MAX_TRACKED_IDENTITIES = 4096


class _LifecycleRolloverPreparationV1:
    """Opaque owner-bound lifecycle transaction retained only in-process."""

    __slots__ = ("_owner", "_predecessor", "_authority", "_terminal")

    def __init__(
        self,
        owner: EvidenceLifecycleOwner,
        predecessor: tuple[str, int, str, str],
        authority: RolloverAuthorityV1,
    ) -> None:
        self._owner = owner
        self._predecessor = predecessor
        self._authority = authority
        self._terminal = False

    def __repr__(self) -> str:
        return "<_LifecycleRolloverPreparationV1 opaque>"


class EvidenceLifecycleOwner:
    """Mint and retire binding-scoped evidence authority capabilities."""

    def __init__(
        self,
        *,
        owner_generation: int,
        uuid_factory: Callable[[], str] | None = None,
        operation_scheduler: ConversationOperationScheduler | None = None,
    ) -> None:
        if type(owner_generation) is not int or not 1 <= owner_generation <= _MAX_UNSIGNED_63:
            raise ValueError("owner_generation must be an unsigned 63-bit positive integer")
        if uuid_factory is not None and not callable(uuid_factory):
            raise TypeError("uuid_factory must be callable or None")
        if (
            operation_scheduler is not None
            and type(operation_scheduler) is not ConversationOperationScheduler
        ):
            raise TypeError("operation_scheduler must be exact or None")
        if (
            operation_scheduler is not None
            and operation_scheduler.owner_generation != owner_generation
        ):
            raise ValueError("operation scheduler owner generation must match lifecycle owner")
        self._owner_generation = owner_generation
        self._uuid_factory = uuid_factory or (lambda: str(uuid4()))
        self._operation_scheduler = operation_scheduler
        self._lock = Lock()
        self._binding: tuple[str, int, str, str] | None = None
        self._drain_lineage: tuple[str, int, str, str] | None = None
        self._drain_authority: LifecycleDrainAuthorityV1 | None = None
        self._live_final_inputs: set[FinalInputAuthorityV1] = set()
        self._used_ids: set[str] = set()
        self._used_id_order: deque[str] = deque()
        self._next_routing_serial = 1
        self._next_proactive_invocation_serial = 1
        self._faulted = False
        self._conversation_authority: EvidenceConversationAuthorityV1 | None = None
        self._rollover_preparation: _LifecycleRolloverPreparationV1 | None = None

    @property
    def owner_generation(self) -> int:
        return self._owner_generation

    @property
    def conversation_authority(self) -> EvidenceConversationAuthorityV1:
        """Return the narrow authority surface allowed into conversation code."""

        with self._lock:
            authority = self._conversation_authority
            if authority is None:
                authority = EvidenceConversationAuthorityV1(self)
                self._conversation_authority = authority
            return authority

    @property
    def operation_scheduler(self) -> ConversationOperationScheduler | None:
        return self._operation_scheduler

    @property
    def faulted(self) -> bool:
        with self._lock:
            return self._faulted

    def activate_binding(
        self,
        *,
        binding_id: str,
        binding_generation: int,
        consent_epoch_id: str,
        logical_session_id: str,
    ) -> None:
        self._require_uuid4(binding_id, "binding_id")
        self._require_uuid4(consent_epoch_id, "consent_epoch_id")
        self._require_uuid4(logical_session_id, "logical_session_id")
        if type(binding_generation) is not int or not 1 <= binding_generation <= _MAX_UNSIGNED_63:
            raise ValueError("binding_generation must be an unsigned 63-bit positive integer")
        with self._lock:
            if self._rollover_preparation is not None:
                raise ReservationError("a lifecycle rollover is prepared")
            self._binding = (
                binding_id,
                binding_generation,
                consent_epoch_id,
                logical_session_id,
            )
            self._drain_lineage = None
            self._drain_authority = None
            self._live_final_inputs.clear()

    def stage_drain_lineage(
        self,
        *,
        binding_id: str,
        binding_generation: int,
        consent_epoch_id: str,
        logical_session_id: str,
    ) -> None:
        """Retain the host-reserved create lineage for pre-publication teardown."""

        self._require_uuid4(binding_id, "binding_id")
        self._require_uuid4(consent_epoch_id, "consent_epoch_id")
        self._require_uuid4(logical_session_id, "logical_session_id")
        if type(binding_generation) is not int or not 1 <= binding_generation <= _MAX_UNSIGNED_63:
            raise ValueError("binding_generation must be an unsigned 63-bit positive integer")
        lineage = (
            binding_id,
            binding_generation,
            consent_epoch_id,
            logical_session_id,
        )
        with self._lock:
            if self._binding is not None:
                raise ReservationError("an active evidence binding cannot stage a drain lineage")
            if self._drain_authority is not None:
                raise ReservationError("a prior owner drain authority is still retained")
            self._drain_lineage = lineage

    def retire_drain_lineage(self) -> None:
        """Release completed host-close drain state before a replacement epoch."""

        with self._lock:
            if self._binding is not None:
                raise ReservationError("an active evidence binding cannot retire drain lineage")
            self._drain_lineage = None
            self._drain_authority = None

    def mint_drain_authority(
        self,
        *,
        final_admission_ordinal: int,
    ) -> LifecycleDrainAuthorityV1:
        """Mint the one exact host-close bearer for this binding lineage.

        A lifecycle owner, not the admission surface given to conversation,
        decides when close may bind its final admitted watermark.  Retrying a
        refused drain returns the same capability and rejects any watermark
        change.
        """

        if (
            type(final_admission_ordinal) is not int
            or not 1 <= final_admission_ordinal <= _MAX_UNSIGNED_63
        ):
            raise ValueError("final admission ordinal must be an unsigned 63-bit positive integer")
        with self._lock:
            lineage = self._binding or self._drain_lineage
            if lineage is None:
                raise ReservationError("no active or retiring evidence binding")
            authority = self._drain_authority
            if authority is not None:
                if authority.final_admission_ordinal != final_admission_ordinal:
                    raise ReservationError("drain watermark changed during owner close")
                return authority
            authority = self._mint(
                LifecycleDrainAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                final_admission_ordinal=final_admission_ordinal,
            )
            self._drain_lineage = lineage
            self._drain_authority = authority
            return authority

    def mint_final_input(
        self,
        *,
        source: InputSource,
        input_incarnation: int,
        media_incarnation: int | None,
        typed_sequence: int | None,
    ) -> FinalInputAuthorityV1:
        with self._lock:
            binding = self._binding
            if binding is None:
                raise ReservationError("no active evidence binding")
            utterance_id = self._allocate_uuid_locked("utterance_id")
            authority = self._mint(
                FinalInputAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                logical_session_id=binding[3],
                utterance_id=utterance_id,
                source=source,
                input_incarnation=input_incarnation,
                media_incarnation=media_incarnation,
                typed_sequence=typed_sequence,
            )
            self._live_final_inputs.add(authority)
            return authority

    def decline_to_user(self, authority: FinalInputAuthorityV1) -> UserTurnAuthorityV1:
        with self._lock:
            if (
                type(authority) is not FinalInputAuthorityV1
                or authority not in self._live_final_inputs
            ):
                raise ReservationError("final input authority is stale or consumed")
            self._live_final_inputs.remove(authority)
            serial = self._next_routing_serial
            if serial > _MAX_UNSIGNED_63:
                raise ReservationError("routing serial exhausted")
            self._next_routing_serial += 1
            return self._mint(
                UserTurnAuthorityV1,
                protocol_version=authority.protocol_version,
                owner_generation=authority.owner_generation,
                binding_id=authority.binding_id,
                binding_generation=authority.binding_generation,
                consent_epoch_id=authority.consent_epoch_id,
                logical_session_id=authority.logical_session_id,
                utterance_id=authority.utterance_id,
                source=authority.source,
                input_incarnation=authority.input_incarnation,
                media_incarnation=authority.media_incarnation,
                typed_sequence=authority.typed_sequence,
                routing_serial=serial,
                routing_disposition="response",
            )

    def accept_command(
        self,
        authority: FinalInputAuthorityV1,
    ) -> CommandAdmissionAuthorityV1:
        with self._lock:
            if (
                type(authority) is not FinalInputAuthorityV1
                or authority not in self._live_final_inputs
            ):
                raise ReservationError("final input authority is stale or consumed")
            self._live_final_inputs.remove(authority)
            serial = self._next_routing_serial
            if serial > _MAX_UNSIGNED_63:
                raise ReservationError("routing serial exhausted")
            self._next_routing_serial += 1
            return self._mint(
                CommandAdmissionAuthorityV1,
                protocol_version=authority.protocol_version,
                owner_generation=authority.owner_generation,
                binding_id=authority.binding_id,
                binding_generation=authority.binding_generation,
                consent_epoch_id=authority.consent_epoch_id,
                logical_session_id=authority.logical_session_id,
                utterance_id=authority.utterance_id,
                source=authority.source,
                input_incarnation=authority.input_incarnation,
                media_incarnation=authority.media_incarnation,
                typed_sequence=authority.typed_sequence,
                routing_serial=serial,
                routing_disposition="command",
            )

    def retire_final_input(self, authority: FinalInputAuthorityV1) -> None:
        with self._lock:
            if (
                type(authority) is not FinalInputAuthorityV1
                or authority not in self._live_final_inputs
            ):
                raise ReservationError("final input authority is stale or consumed")
            self._live_final_inputs.remove(authority)

    def conversation_authority_is_current(self, authority: object) -> bool:
        """Whether an issued conversation bearer still belongs to the open binding."""

        if type(authority) not in (
            CommandAdmissionAuthorityV1,
            ProactiveTurnAuthorityV1,
            ReplayTurnAuthorityV1,
            UserTurnAuthorityV1,
        ):
            return False
        current_authority = cast(
            CommandAdmissionAuthorityV1
            | ProactiveTurnAuthorityV1
            | ReplayTurnAuthorityV1
            | UserTurnAuthorityV1,
            authority,
        )
        with self._lock:
            binding = self._binding
            return (
                binding is not None
                and current_authority.owner_generation == self._owner_generation
                and current_authority.binding_id == binding[0]
                and current_authority.binding_generation == binding[1]
                and current_authority.consent_epoch_id == binding[2]
                and current_authority.logical_session_id == binding[3]
            )

    def mint_proactive_turn(
        self,
        operation: ConversationOperationReservation,
    ) -> ProactiveTurnAuthorityV1:
        with self._lock:
            scheduler = self._operation_scheduler
            binding = self._binding
            if scheduler is None:
                raise ReservationError("lifecycle owner has no operation scheduler")
            if binding is None:
                raise ReservationError("no active evidence binding")
            scheduler.validate(
                operation,
                ConversationOperationKind.PROACTIVE,
                require_unconsumed=True,
            )
            serial = self._next_proactive_invocation_serial
            if serial > _MAX_UNSIGNED_63:
                raise ReservationError("proactive invocation serial exhausted")
            self._next_proactive_invocation_serial += 1
            return self._mint(
                ProactiveTurnAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                logical_session_id=binding[3],
                proactive_invocation_serial=serial,
            )

    def mint_replay_turn(
        self,
        operation: ConversationOperationReservation,
        *,
        replay_of_evidence_turn_id: str,
        replay_generation: int,
        source_binding_id: str,
        source_binding_generation: int,
        source_logical_session_id: str,
    ) -> ReplayTurnAuthorityV1:
        self._require_uuid4(replay_of_evidence_turn_id, "replay_of_evidence_turn_id")
        self._require_uuid4(source_binding_id, "source_binding_id")
        self._require_uuid4(source_logical_session_id, "source_logical_session_id")
        if (
            type(source_binding_generation) is not int
            or not 1 <= source_binding_generation <= _MAX_UNSIGNED_63
        ):
            raise ValueError(
                "source_binding_generation must be an unsigned 63-bit positive integer"
            )
        if type(replay_generation) is not int or not 1 <= replay_generation <= _MAX_UNSIGNED_63:
            raise ValueError("replay_generation must be an unsigned 63-bit positive integer")
        with self._lock:
            scheduler = self._operation_scheduler
            binding = self._binding
            if scheduler is None:
                raise ReservationError("lifecycle owner has no operation scheduler")
            if binding is None:
                raise ReservationError("no active evidence binding")
            if (
                source_binding_id != binding[0]
                or source_binding_generation != binding[1]
                or source_logical_session_id != binding[3]
            ):
                raise ReservationError("replay source binding is stale")
            scheduler.validate(
                operation,
                ConversationOperationKind.REPLAY,
                require_unconsumed=True,
            )
            return self._mint(
                ReplayTurnAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                logical_session_id=binding[3],
                replay_of_evidence_turn_id=replay_of_evidence_turn_id,
                replay_generation=replay_generation,
            )

    def close_binding(
        self,
        reason: BindingCloseReason,
    ) -> BindingCloseAuthorityV1:
        with self._lock:
            if self._rollover_preparation is not None:
                raise ReservationError("a lifecycle rollover is prepared")
            binding = self._binding
            if binding is None:
                raise ReservationError("no active evidence binding")
            authority = self._mint(
                BindingCloseAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                logical_session_id=binding[3],
                close_reason=reason,
            )
            self._drain_lineage = binding
            self._binding = None
            self._live_final_inputs.clear()
            return authority

    def rollover_binding(
        self,
        *,
        successor_logical_session_id: str,
        successor_expires_at_utc: str,
        reason: BindingCloseReason,
    ) -> RolloverAuthorityV1:
        with self._lock:
            if self._rollover_preparation is not None:
                raise ReservationError("a lifecycle rollover is prepared")
            binding = self._binding
            if binding is None:
                raise ReservationError("no active evidence binding")
            authority = self._mint(
                RolloverAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                predecessor_logical_session_id=binding[3],
                successor_logical_session_id=successor_logical_session_id,
                successor_expires_at_utc=successor_expires_at_utc,
                reason=reason,
            )
            self._binding = (
                binding[0],
                binding[1],
                binding[2],
                authority.successor_logical_session_id,
            )
            self._live_final_inputs.clear()
            return authority

    def prepare_rollover(
        self,
        *,
        successor_logical_session_id: str,
        successor_expires_at_utc: str,
        reason: BindingCloseReason,
    ) -> _LifecycleRolloverPreparationV1:
        """Prepare, but do not publish, one exact successor lineage."""

        with self._lock:
            if self._rollover_preparation is not None:
                raise ReservationError("a lifecycle rollover is already prepared")
            binding = self._binding
            if binding is None:
                raise ReservationError("no active evidence binding")
            authority = self._mint(
                RolloverAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                predecessor_logical_session_id=binding[3],
                successor_logical_session_id=successor_logical_session_id,
                successor_expires_at_utc=successor_expires_at_utc,
                reason=reason,
            )
            preparation = _LifecycleRolloverPreparationV1(self, binding, authority)
            self._rollover_preparation = preparation
            return preparation

    def rollover_authority(
        self, preparation: _LifecycleRolloverPreparationV1
    ) -> RolloverAuthorityV1:
        with self._lock:
            if self._rollover_preparation is not preparation:
                raise ReservationError("lifecycle rollover preparation is stale")
            return preparation._authority

    def commit_prepared_rollover(self, preparation: _LifecycleRolloverPreparationV1) -> None:
        with self._lock:
            if self._rollover_preparation is not preparation:
                raise ReservationError("lifecycle rollover preparation is stale")
            if self._binding != preparation._predecessor:
                raise ReservationError("lifecycle rollover predecessor changed")
            authority = preparation._authority
            self._binding = (
                authority.binding_id,
                authority.binding_generation,
                authority.consent_epoch_id,
                authority.successor_logical_session_id,
            )
            self._live_final_inputs.clear()
            self._rollover_preparation = None
            preparation._terminal = True

    def abort_prepared_rollover(self, preparation: _LifecycleRolloverPreparationV1) -> None:
        with self._lock:
            if self._rollover_preparation is not preparation:
                raise ReservationError("lifecycle rollover preparation is stale")
            self._rollover_preparation = None
            preparation._terminal = True

    def seal_lifecycle(
        self,
        *,
        reason: BindingCloseReason,
        final_event_sequence: int,
    ) -> LifecycleSealAuthorityV1:
        with self._lock:
            if self._rollover_preparation is not None:
                raise ReservationError("a lifecycle rollover is prepared")
            binding = self._binding
            if binding is None:
                raise ReservationError("no active evidence binding")
            authority = self._mint(
                LifecycleSealAuthorityV1,
                protocol_version=1,
                owner_generation=self._owner_generation,
                binding_id=binding[0],
                binding_generation=binding[1],
                consent_epoch_id=binding[2],
                logical_session_id=binding[3],
                close_reason=reason,
                final_event_sequence=final_event_sequence,
                close_epoch=True,
            )
            self._drain_lineage = binding
            self._binding = None
            self._live_final_inputs.clear()
            return authority

    def _allocate_uuid_locked(self, field_name: str) -> str:
        if self._faulted:
            raise ReservationError("evidence lifecycle owner is faulted")
        value = self._uuid_factory()
        self._require_uuid4(value, field_name)
        if value in self._used_ids:
            self._faulted = True
            raise ReservationError("evidence identity collision")
        self._used_ids.add(value)
        self._used_id_order.append(value)
        # A long-lived host mints one authority per turn, so unbounded retention
        # would grow for the session lifetime. A bounded recent window still
        # catches a broken uuid_factory immediately, which is what this set is
        # for; a genuine UUIDv4 collision beyond the window is not credible.
        while len(self._used_id_order) > _MAX_TRACKED_IDENTITIES:
            self._used_ids.discard(self._used_id_order.popleft())
        return value

    @staticmethod
    def _require_uuid4(value: object, field_name: str) -> None:
        if type(value) is not str:
            raise TypeError(f"{field_name} must be an exact built-in string")
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as error:
            raise ValueError(f"{field_name} must be a canonical UUIDv4") from error
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError(f"{field_name} must be a canonical UUIDv4")

    @staticmethod
    def _mint(capability_type: type[_Capability], **fields: object) -> _Capability:
        capability = object.__new__(capability_type)
        for name, value in fields.items():
            object.__setattr__(capability, name, value)
        capability._validate()
        return capability


class EvidenceConversationAuthorityV1:
    """Conversation-only authority issuance; lifecycle transitions stay host-owned."""

    __slots__ = ("__owner",)

    def __init__(self, owner: EvidenceLifecycleOwner) -> None:
        if type(owner) is not EvidenceLifecycleOwner:
            raise TypeError("owner must be an exact EvidenceLifecycleOwner")
        self.__owner = owner

    @property
    def owner_generation(self) -> int:
        return self.__owner.owner_generation

    def mint_final_input(
        self,
        *,
        source: InputSource,
        input_incarnation: int,
        media_incarnation: int | None,
        typed_sequence: int | None,
    ) -> FinalInputAuthorityV1:
        return self.__owner.mint_final_input(
            source=source,
            input_incarnation=input_incarnation,
            media_incarnation=media_incarnation,
            typed_sequence=typed_sequence,
        )

    def decline_to_user(self, authority: FinalInputAuthorityV1) -> UserTurnAuthorityV1:
        return self.__owner.decline_to_user(authority)

    def accept_command(
        self,
        authority: FinalInputAuthorityV1,
    ) -> CommandAdmissionAuthorityV1:
        return self.__owner.accept_command(authority)

    def retire_final_input(self, authority: FinalInputAuthorityV1) -> None:
        self.__owner.retire_final_input(authority)

    def mint_proactive_turn(
        self,
        operation: ConversationOperationReservation,
    ) -> ProactiveTurnAuthorityV1:
        return self.__owner.mint_proactive_turn(operation)

    def mint_replay_turn(
        self,
        operation: ConversationOperationReservation,
        *,
        replay_of_evidence_turn_id: str,
        replay_generation: int,
        source_binding_id: str,
        source_binding_generation: int,
        source_logical_session_id: str,
    ) -> ReplayTurnAuthorityV1:
        return self.__owner.mint_replay_turn(
            operation,
            replay_of_evidence_turn_id=replay_of_evidence_turn_id,
            replay_generation=replay_generation,
            source_binding_id=source_binding_id,
            source_binding_generation=source_binding_generation,
            source_logical_session_id=source_logical_session_id,
        )


__all__ = ["EvidenceConversationAuthorityV1", "EvidenceLifecycleOwner"]
