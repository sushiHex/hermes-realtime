from __future__ import annotations

from uuid import UUID

import pytest


def _uuid(value: int) -> str:
    return str(UUID(int=value, version=4))


def test_owner_mints_one_typed_user_authority_with_exact_lineage() -> None:
    try:
        from hermes_realtime.evidence import InputSource
        from hermes_realtime.evidence.admission import ReservationError
        from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner
    except ImportError as error:
        pytest.fail(f"RED bootstrap: expected evidence lifecycle owner: {error}")

    identities = iter((_uuid(101),))
    owner = EvidenceLifecycleOwner(
        owner_generation=7,
        uuid_factory=lambda: next(identities),
    )
    owner.activate_binding(
        binding_id=_uuid(1),
        binding_generation=2,
        consent_epoch_id=_uuid(3),
        logical_session_id=_uuid(4),
    )

    final_input = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=5,
        media_incarnation=None,
        typed_sequence=6,
    )
    user_turn = owner.decline_to_user(final_input)

    assert user_turn.owner_generation == 7
    assert user_turn.binding_id == _uuid(1)
    assert user_turn.binding_generation == 2
    assert user_turn.consent_epoch_id == _uuid(3)
    assert user_turn.logical_session_id == _uuid(4)
    assert user_turn.utterance_id == _uuid(101)
    assert user_turn.source is InputSource.TYPED
    assert user_turn.input_incarnation == 5
    assert user_turn.media_incarnation is None
    assert user_turn.typed_sequence == 6
    assert user_turn.routing_serial == 1
    assert user_turn.routing_disposition == "response"

    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(final_input)


def test_uuid_collision_faults_owner_without_retry_or_replacing_existing_authority() -> None:
    from hermes_realtime.evidence import InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    draws = 0

    def colliding_uuid() -> str:
        nonlocal draws
        draws += 1
        return _uuid(201)

    owner = EvidenceLifecycleOwner(owner_generation=9, uuid_factory=colliding_uuid)
    owner.activate_binding(
        binding_id=_uuid(11),
        binding_generation=12,
        consent_epoch_id=_uuid(13),
        logical_session_id=_uuid(14),
    )
    first = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=15,
        media_incarnation=None,
        typed_sequence=16,
    )

    with pytest.raises(ReservationError, match="collision"):
        owner.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=17,
            media_incarnation=None,
            typed_sequence=18,
        )

    assert draws == 2
    assert owner.faulted is True
    assert owner.decline_to_user(first).utterance_id == _uuid(201)
    with pytest.raises(ReservationError, match="faulted"):
        owner.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=19,
            media_incarnation=None,
            typed_sequence=20,
        )
    assert draws == 2


def test_binding_replacement_invalidates_old_final_input_authority() -> None:
    from hermes_realtime.evidence import InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(
        owner_generation=21,
        uuid_factory=iter((_uuid(301), _uuid(302))).__next__,
    )
    owner.activate_binding(
        binding_id=_uuid(31),
        binding_generation=1,
        consent_epoch_id=_uuid(32),
        logical_session_id=_uuid(33),
    )
    stale = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )

    owner.activate_binding(
        binding_id=_uuid(34),
        binding_generation=2,
        consent_epoch_id=_uuid(35),
        logical_session_id=_uuid(36),
    )

    with pytest.raises(ReservationError, match="stale"):
        owner.decline_to_user(stale)
    current = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=2,
        media_incarnation=None,
        typed_sequence=2,
    )
    assert owner.decline_to_user(current).binding_generation == 2


def test_fresh_owner_rejects_capability_from_previous_owner_with_reused_values() -> None:
    from hermes_realtime.evidence import InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    lineage = {
        "binding_id": _uuid(41),
        "binding_generation": 1,
        "consent_epoch_id": _uuid(42),
        "logical_session_id": _uuid(43),
    }
    old_owner = EvidenceLifecycleOwner(
        owner_generation=1,
        uuid_factory=lambda: _uuid(401),
    )
    new_owner = EvidenceLifecycleOwner(
        owner_generation=1,
        uuid_factory=lambda: _uuid(401),
    )
    old_owner.activate_binding(**lineage)
    new_owner.activate_binding(**lineage)
    old_authority = old_owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )
    new_authority = new_owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )

    with pytest.raises(ReservationError, match="stale"):
        new_owner.decline_to_user(old_authority)
    assert new_owner.decline_to_user(new_authority).utterance_id == _uuid(401)


def test_proactive_authority_requires_owned_unconsumed_proactive_reservation() -> None:
    from hermes_realtime.evidence import (
        ConversationOperationKind,
        ConversationOperationScheduler,
    )
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    scheduler = ConversationOperationScheduler(owner_generation=51, max_operations=2)
    owner = EvidenceLifecycleOwner(
        owner_generation=51,
        uuid_factory=lambda: _uuid(501),
        operation_scheduler=scheduler,
    )
    owner.activate_binding(
        binding_id=_uuid(52),
        binding_generation=1,
        consent_epoch_id=_uuid(53),
        logical_session_id=_uuid(54),
    )
    response = scheduler.try_reserve(ConversationOperationKind.RESPONSE)
    proactive = scheduler.try_reserve(ConversationOperationKind.PROACTIVE)
    assert response is not None
    assert proactive is not None

    with pytest.raises(ReservationError, match="kind"):
        owner.mint_proactive_turn(response)
    authority = owner.mint_proactive_turn(proactive)

    assert authority.owner_generation == 51
    assert authority.binding_id == _uuid(52)
    assert authority.proactive_invocation_serial == 1
    scheduler.consume(proactive, ConversationOperationKind.PROACTIVE)
    with pytest.raises(ReservationError, match="consumed"):
        owner.mint_proactive_turn(proactive)


def test_replay_authority_requires_owned_unconsumed_replay_reservation() -> None:
    from hermes_realtime.evidence import (
        ConversationOperationKind,
        ConversationOperationScheduler,
    )
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    scheduler = ConversationOperationScheduler(owner_generation=61, max_operations=2)
    owner = EvidenceLifecycleOwner(
        owner_generation=61,
        operation_scheduler=scheduler,
    )
    owner.activate_binding(
        binding_id=_uuid(62),
        binding_generation=1,
        consent_epoch_id=_uuid(63),
        logical_session_id=_uuid(64),
    )
    proactive = scheduler.try_reserve(ConversationOperationKind.PROACTIVE)
    replay = scheduler.try_reserve(ConversationOperationKind.REPLAY)
    assert proactive is not None
    assert replay is not None

    with pytest.raises(ReservationError, match="kind"):
        owner.mint_replay_turn(
            proactive,
            replay_of_evidence_turn_id=_uuid(601),
            replay_generation=1,
            source_binding_id=_uuid(62),
            source_binding_generation=1,
            source_logical_session_id=_uuid(64),
        )
    authority = owner.mint_replay_turn(
        replay,
        replay_of_evidence_turn_id=_uuid(601),
        replay_generation=1,
        source_binding_id=_uuid(62),
        source_binding_generation=1,
        source_logical_session_id=_uuid(64),
    )

    assert authority.replay_of_evidence_turn_id == _uuid(601)
    assert authority.replay_generation == 1
    scheduler.consume(replay, ConversationOperationKind.REPLAY)
    with pytest.raises(ReservationError, match="consumed"):
        owner.mint_replay_turn(
            replay,
            replay_of_evidence_turn_id=_uuid(601),
            replay_generation=2,
            source_binding_id=_uuid(62),
            source_binding_generation=1,
            source_logical_session_id=_uuid(64),
        )


def test_replay_authority_rejects_source_lineage_from_predecessor_binding() -> None:
    from hermes_realtime.evidence import (
        BindingCloseReason,
        ConversationOperationKind,
        ConversationOperationScheduler,
    )
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    scheduler = ConversationOperationScheduler(owner_generation=66, max_operations=1)
    owner = EvidenceLifecycleOwner(
        owner_generation=66,
        operation_scheduler=scheduler,
    )
    owner.activate_binding(
        binding_id=_uuid(67),
        binding_generation=3,
        consent_epoch_id=_uuid(68),
        logical_session_id=_uuid(69),
    )
    owner.rollover_binding(
        successor_logical_session_id=_uuid(70),
        successor_expires_at_utc="2026-08-12T00:00:00.000000Z",
        reason=BindingCloseReason.RETENTION_ROLLOVER,
    )
    replay = scheduler.try_reserve(ConversationOperationKind.REPLAY)
    assert replay is not None

    with pytest.raises(ReservationError, match="source binding"):
        owner.mint_replay_turn(
            replay,
            replay_of_evidence_turn_id=_uuid(660),
            replay_generation=1,
            source_binding_id=_uuid(67),
            source_binding_generation=3,
            source_logical_session_id=_uuid(69),
        )


def test_binding_close_authority_atomically_retires_binding_and_live_inputs() -> None:
    from hermes_realtime.evidence import BindingCloseReason, InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(owner_generation=71)
    owner.activate_binding(
        binding_id=_uuid(72),
        binding_generation=3,
        consent_epoch_id=_uuid(73),
        logical_session_id=_uuid(74),
    )
    final_input = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )

    authority = owner.close_binding(BindingCloseReason.CLIENT_CLOSED)

    assert authority.binding_id == _uuid(72)
    assert authority.binding_generation == 3
    assert authority.logical_session_id == _uuid(74)
    assert authority.close_reason is BindingCloseReason.CLIENT_CLOSED
    with pytest.raises(ReservationError, match="stale|consumed"):
        owner.decline_to_user(final_input)
    with pytest.raises(ReservationError, match="active evidence binding"):
        owner.close_binding(BindingCloseReason.CLIENT_CLOSED)


def test_rollover_atomically_replaces_session_and_invalidates_predecessor_input() -> None:
    from hermes_realtime.evidence import BindingCloseReason, InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    identities = iter((_uuid(801), _uuid(802)))
    owner = EvidenceLifecycleOwner(
        owner_generation=81,
        uuid_factory=identities.__next__,
    )
    owner.activate_binding(
        binding_id=_uuid(82),
        binding_generation=5,
        consent_epoch_id=_uuid(83),
        logical_session_id=_uuid(84),
    )
    predecessor_input = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )

    rollover = owner.rollover_binding(
        successor_logical_session_id=_uuid(85),
        successor_expires_at_utc="2026-08-12T00:00:00.000000Z",
        reason=BindingCloseReason.CAPACITY_ROLLOVER,
    )

    assert rollover.predecessor_logical_session_id == _uuid(84)
    assert rollover.successor_logical_session_id == _uuid(85)
    assert rollover.binding_generation == 5
    with pytest.raises(ReservationError, match="stale|consumed"):
        owner.decline_to_user(predecessor_input)
    successor_input = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=2,
        media_incarnation=None,
        typed_sequence=2,
    )
    assert successor_input.logical_session_id == _uuid(85)


def test_lifecycle_seal_is_single_use_and_retires_active_binding() -> None:
    from hermes_realtime.evidence import BindingCloseReason, InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(owner_generation=91)
    owner.activate_binding(
        binding_id=_uuid(92),
        binding_generation=1,
        consent_epoch_id=_uuid(93),
        logical_session_id=_uuid(94),
    )
    final_input = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )

    seal = owner.seal_lifecycle(
        reason=BindingCloseReason.HOST_SHUTDOWN,
        final_event_sequence=7,
    )

    assert seal.binding_id == _uuid(92)
    assert seal.final_event_sequence == 7
    assert seal.close_epoch is True
    with pytest.raises(ReservationError, match="stale|consumed"):
        owner.decline_to_user(final_input)
    with pytest.raises(ReservationError, match="active evidence binding"):
        owner.seal_lifecycle(
            reason=BindingCloseReason.HOST_SHUTDOWN,
            final_event_sequence=7,
        )


def test_prepared_rollover_keeps_predecessor_visible_until_exact_commit() -> None:
    from hermes_realtime.evidence import BindingCloseReason, InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(owner_generation=101)
    owner.activate_binding(
        binding_id=_uuid(102),
        binding_generation=3,
        consent_epoch_id=_uuid(103),
        logical_session_id=_uuid(104),
    )
    preparation = owner.prepare_rollover(
        successor_logical_session_id=_uuid(105),
        successor_expires_at_utc="2026-08-16T00:00:00.000000Z",
        reason=BindingCloseReason.CAPACITY_ROLLOVER,
    )
    predecessor = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )
    assert predecessor.logical_session_id == _uuid(104)
    with pytest.raises(ReservationError, match="prepared"):
        owner.close_binding(BindingCloseReason.CLIENT_CLOSED)

    owner.commit_prepared_rollover(preparation)
    successor = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=2,
        media_incarnation=None,
        typed_sequence=2,
    )
    assert successor.logical_session_id == _uuid(105)
    with pytest.raises(ReservationError, match="stale"):
        owner.commit_prepared_rollover(preparation)
    with pytest.raises(ReservationError, match="stale"):
        owner.abort_prepared_rollover(preparation)


def test_prepared_rollover_abort_preserves_predecessor_and_is_single_use() -> None:
    from hermes_realtime.evidence import BindingCloseReason, InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(owner_generation=111)
    owner.activate_binding(
        binding_id=_uuid(112),
        binding_generation=1,
        consent_epoch_id=_uuid(113),
        logical_session_id=_uuid(114),
    )
    preparation = owner.prepare_rollover(
        successor_logical_session_id=_uuid(115),
        successor_expires_at_utc="2026-08-16T00:00:00.000000Z",
        reason=BindingCloseReason.CAPACITY_ROLLOVER,
    )
    owner.abort_prepared_rollover(preparation)
    authority = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )
    assert authority.logical_session_id == _uuid(114)
    with pytest.raises(ReservationError, match="stale"):
        owner.abort_prepared_rollover(preparation)


def test_owner_bounds_retained_identity_history() -> None:
    # Regression: the collision-detection set grew one UUID per minted authority
    # with no eviction path, so a long-lived host accumulated them for the whole
    # session. A bounded recent window still catches a broken uuid_factory.
    import itertools

    from hermes_realtime.evidence import InputSource
    from hermes_realtime.evidence.lifecycle import (
        _MAX_TRACKED_IDENTITIES,
        EvidenceLifecycleOwner,
    )

    identities = itertools.count(1000)
    owner = EvidenceLifecycleOwner(
        owner_generation=7,
        uuid_factory=lambda: _uuid(next(identities)),
    )
    owner.activate_binding(
        binding_id=_uuid(1),
        binding_generation=2,
        consent_epoch_id=_uuid(3),
        logical_session_id=_uuid(4),
    )

    for index in range(1, _MAX_TRACKED_IDENTITIES + 51):
        final_input = owner.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=index,
            media_incarnation=None,
            typed_sequence=index,
        )
        owner.decline_to_user(final_input)

    # Private state is the whole point of this assertion: the bound is a memory
    # guarantee with no public projection.
    assert len(owner._used_ids) <= _MAX_TRACKED_IDENTITIES
    assert len(owner._used_id_order) <= _MAX_TRACKED_IDENTITIES


def test_owner_still_faults_on_a_repeated_identity() -> None:
    from hermes_realtime.evidence import InputSource
    from hermes_realtime.evidence.admission import ReservationError
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(
        owner_generation=7,
        uuid_factory=lambda: _uuid(101),
    )
    owner.activate_binding(
        binding_id=_uuid(1),
        binding_generation=2,
        consent_epoch_id=_uuid(3),
        logical_session_id=_uuid(4),
    )

    owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )
    with pytest.raises(ReservationError, match="identity collision"):
        owner.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=2,
            media_incarnation=None,
            typed_sequence=2,
        )
