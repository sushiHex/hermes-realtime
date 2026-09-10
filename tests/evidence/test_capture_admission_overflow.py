"""Lost one-shot capture publications cannot become complete evidence."""

import pytest
from test_admission import (
    _active_admission,
    _owner_drain_authority,
    _revoke_authority,
    _turn_authority,
    _turn_snapshot,
    _uuid,
)

from hermes_realtime.evidence import admission as a
from hermes_realtime.evidence import models as m

SETTLEMENT_REASONS = [
    None,
    m.TerminalReason.PROVIDER_FAILED,
    m.TerminalReason.CALLER_CANCELLED,
    m.TerminalReason.TRANSPORT_FAILED,
    m.TerminalReason.TASK_SPAWN_FAILED,
]


@pytest.mark.parametrize("reason", SETTLEMENT_REASONS)
@pytest.mark.parametrize("revoked", [False, True])
def test_capture_overflow_retires_only_the_owned_evidence_turn(revoked, reason):
    admission, operations, writer, create = _active_admission(a, m, owner_generation=721)
    authority = _turn_authority(m, "UserTurnAuthorityV1", create, owner_generation=721)
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    lease = admission.try_reserve_user_turn(authority, operation).lease
    assert (
        admission.try_admit_user_final(lease, authority, "Synthetic accepted input.")
        is m.AppendDisposition.ADMITTED
    )
    dispositions = []
    for n in range(80):
        text = f"Synthetic segment {n}."
        dispositions.append(admission.try_admit_generated(lease, text))
        dispositions.append(
            admission.try_admit_transport_confirmed_full(
                lease,
                segment_ordinal=n + 1,
                synthesis_attempt_id=_uuid(10000 + n * 2),
                transport_attempt_id=_uuid(10001 + n * 2),
                text=text,
            )
        )
    assert m.AppendDisposition.DROPPED_CAPACITY in dispositions
    before = admission.diagnostics()
    assert before.active_lease_count == 1
    if revoked:
        while writer.ordered_count:
            admission.complete_ordered_item(writer.get_nowait())
        revoke = admission.begin_revoke(_revoke_authority(m, create))
        admission.complete_revoke_request(
            writer.get_nowait(), m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        )
        assert not revoke.terminal_event.is_set()
    queued_before = writer.ordered_count
    result = _settle(admission, lease, reason=reason)
    after = admission.diagnostics()
    assert result is m.AppendDisposition.SESSION_TAINTED
    assert after.active_lease_count == 0
    assert admission.settled_terminal_outcome(lease) is None

    if revoked:
        finalizer = writer.get_nowait()
        assert type(finalizer.payload) is m.RevokeFinalizeV1
        admission.complete_revoke_finalize(finalizer, m.RevokeDisposition.PURGE_COMPLETED)
        assert revoke.terminal_event.is_set()
    else:
        assert writer.ordered_count == queued_before
        assert after.queue_record_count == before.queue_record_count - 2
        assert after.capture_state is m.CaptureState.FAULTED
    while writer.ordered_count:
        admission.complete_ordered_item(writer.get_nowait())
    drain = admission.request_drain(
        _owner_drain_authority(m, create, admission, owner_generation=721)
    )
    admission.complete_drain(writer.get_nowait(), m.DrainDisposition.STOPPED)
    assert drain.terminal_event.is_set()
    final = admission.diagnostics()
    assert final.owner_state is m.OwnerState.STOPPED
    assert final.queue_record_count == final.queue_canonical_bytes == final.active_lease_count == 0


def _opened(owner=721):
    admission, operations, writer, create = _active_admission(a, m, owner_generation=owner)
    authority = _turn_authority(m, "UserTurnAuthorityV1", create, owner_generation=owner)
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    lease = admission.try_reserve_user_turn(authority, operation).lease
    assert (
        admission.try_admit_user_final(lease, authority, "Synthetic accepted input.")
        is m.AppendDisposition.ADMITTED
    )
    return admission, writer, lease


def _overflow(admission, lease, source):
    if source == "transport":
        assert (
            admission.try_admit_generated(lease, "Synthetic segment.")
            is m.AppendDisposition.ADMITTED
        )
    for ordinal in range(80):
        if source == "generated":
            disposition = admission.try_admit_generated(lease, f"Synthetic segment {ordinal}.")
        else:
            disposition = admission.try_admit_transport_confirmed_full(
                lease,
                segment_ordinal=1,
                synthesis_attempt_id=_uuid(20000 + ordinal * 2),
                transport_attempt_id=_uuid(20001 + ordinal * 2),
                text="Synthetic segment.",
            )
        if disposition is m.AppendDisposition.DROPPED_CAPACITY:
            return
        assert disposition is m.AppendDisposition.ADMITTED
    raise AssertionError("The real ordinary queue did not reach its bound")


def _settle(admission, lease, count=80, reason=None):
    if reason is m.TerminalReason.TASK_SPAWN_FAILED:
        return admission.settle_spawn_failed(lease)
    if reason is not None:
        admission.record_terminal_cause(lease.terminal_cause, reason)
        return admission.settle_terminal(
            lease,
            queued_chunk_count=count,
            started_chunk_count=count,
            assistant_delivery_context_recorded=False,
        )
    return admission.settle_completed(
        lease,
        queued_chunk_count=count,
        started_chunk_count=count,
        transport_confirmed_full_count=count,
        assistant_delivery_context_recorded=True,
    )


@pytest.mark.parametrize("reason", SETTLEMENT_REASONS)
@pytest.mark.parametrize("source", ["generated", "transport"])
def test_one_shot_source_refusal_latches_capture_gap(source, reason):
    admission, writer, lease = _opened()
    _overflow(admission, lease, source)
    assert admission.diagnostics().capture_state is m.CaptureState.FAULTED
    while writer.ordered_count:
        admission.complete_ordered_item(writer.get_nowait())
    # Free queue space cannot turn a lost publication into a complete capture.
    assert (
        admission.try_admit_generated(lease, "Synthetic later segment.")
        is m.AppendDisposition.SESSION_TAINTED
    )
    assert _settle(admission, lease, reason=reason) is m.AppendDisposition.SESSION_TAINTED
    assert admission.settled_terminal_outcome(lease) is None
    assert admission.diagnostics().active_lease_count == 0


@pytest.mark.parametrize("reason", SETTLEMENT_REASONS)
def test_tainted_retirement_requires_the_exact_live_owner_lease(reason):
    admission, writer, lease = _opened()
    other, _, foreign = _opened(722)
    _overflow(admission, lease, "generated")
    before = admission.diagnostics()
    assert _settle(admission, foreign, reason=reason) is m.AppendDisposition.INVALID_AUTHORITY
    assert admission.diagnostics() == before
    assert other.diagnostics().active_lease_count == 1
    assert _settle(admission, lease, reason=reason) is m.AppendDisposition.SESSION_TAINTED
    retired = admission.diagnostics()
    assert _settle(admission, lease, reason=reason) is m.AppendDisposition.INVALID_AUTHORITY
    assert admission.diagnostics() == retired
    assert other.diagnostics().active_lease_count == 1
    assert _settle(other, foreign, 0, reason=reason) is m.AppendDisposition.ADMITTED


def test_healthy_capture_still_rejects_a_mismatched_delivery_count():
    admission, _, lease = _opened()
    before = admission.diagnostics()
    assert _settle(admission, lease, 1) is m.AppendDisposition.INVALID_AUTHORITY
    assert admission.diagnostics() == before
    assert _settle(admission, lease, 0) is m.AppendDisposition.ADMITTED


@pytest.mark.parametrize("reason", SETTLEMENT_REASONS)
def test_tainted_retirement_preserves_another_live_operation_in_the_same_owner(reason):
    admission, operations, writer, create = _active_admission(a, m, owner_generation=721)
    response = _turn_authority(m, "UserTurnAuthorityV1", create, owner_generation=721)
    lease = admission.try_reserve_user_turn(
        response, operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    ).lease
    assert (
        admission.try_admit_user_final(lease, response, "Synthetic accepted input.")
        is m.AppendDisposition.ADMITTED
    )
    proactive = _turn_authority(m, "ProactiveTurnAuthorityV1", create, owner_generation=721)
    other = admission.try_reserve_proactive_turn(
        proactive, operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    ).lease
    assert admission.try_open_non_user_turn(other) is m.AppendDisposition.ADMITTED
    _overflow(admission, lease, "generated")
    assert admission.diagnostics().active_lease_count == 2
    while writer.ordered_count:
        admission.complete_ordered_item(writer.get_nowait())
    assert _settle(admission, lease) is m.AppendDisposition.SESSION_TAINTED
    assert admission.diagnostics().active_lease_count == 1
    assert writer.ordered_count == 0
    assert _settle(admission, other, 0, reason=reason) is m.AppendDisposition.SESSION_TAINTED
    assert admission.diagnostics().active_lease_count == 0
    assert writer.ordered_count == 0
    assert (
        admission.settled_terminal_outcome(lease)
        is admission.settled_terminal_outcome(other)
        is None
    )


@pytest.mark.parametrize("reason", SETTLEMENT_REASONS)
def test_retired_capture_refuses_late_terminal_causes(reason):
    admission, _, lease = _opened()
    _overflow(admission, lease, "generated")
    assert _settle(admission, lease, reason=reason) is m.AppendDisposition.SESSION_TAINTED
    for late in SETTLEMENT_REASONS[1:]:
        assert (
            admission.record_terminal_cause(lease.terminal_cause, late)
            is m.CauseDisposition.INVALID_AUTHORITY
        )
    assert admission.settled_terminal_outcome(lease) is None


def test_retirement_freezes_a_cause_report_that_already_retained_its_state(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    admission, _, lease = _opened()
    _overflow(admission, lease, "generated")
    validated, release = Event(), Event()
    owner = current_thread()
    original = a.TerminalCauseCapabilityV1._validate

    def pause_after_validation(capability):
        original(capability)
        if capability is lease.terminal_cause and current_thread() is not owner:
            validated.set()
            assert release.wait(2.0)

    monkeypatch.setattr(a.TerminalCauseCapabilityV1, "_validate", pause_after_validation)
    with ThreadPoolExecutor(max_workers=1) as worker:
        report = worker.submit(
            admission.record_terminal_cause, lease.terminal_cause, m.TerminalReason.PROVIDER_FAILED
        )
        try:
            assert validated.wait(2.0)
            assert _settle(admission, lease) is m.AppendDisposition.SESSION_TAINTED
        finally:
            release.set()
        assert report.result(timeout=2.0) is m.CauseDisposition.CAUSE_SET_FROZEN
    assert admission.settled_terminal_outcome(lease) is None


@pytest.mark.parametrize("snapshot_already_queued", [False, True])
@pytest.mark.parametrize("revoked", [False, True])
def test_low_level_terminal_append_cannot_complete_a_tainted_session(
    snapshot_already_queued, revoked
):
    admission, operations, writer, create = _active_admission(a, m, owner_generation=721)
    response = _turn_authority(m, "UserTurnAuthorityV1", create, owner_generation=721)
    overflowing = admission.try_reserve_user_turn(
        response, operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    ).lease
    assert (
        admission.try_admit_user_final(overflowing, response, "Synthetic accepted input.")
        is m.AppendDisposition.ADMITTED
    )
    proactive = _turn_authority(m, "ProactiveTurnAuthorityV1", create, owner_generation=721)
    lease = admission.try_reserve_proactive_turn(
        proactive, operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    ).lease
    assert admission.try_open_non_user_turn(lease) is m.AppendDisposition.ADMITTED

    def snapshot():
        return _turn_snapshot(
            m,
            create,
            lease,
            admission._next_event_sequence,
            m.EventKind.TURN_SNAPSHOT,
            m.TurnSnapshotPayloadV1(
                evidence_turn_id=lease.evidence_turn_id,
                turn_kind=m.TurnKind.PROACTIVE_UPDATE,
                generated_segment_count=0,
                queued_chunk_count=0,
                started_chunk_count=0,
                transport_confirmed_full_count=0,
                model_context_admitted=False,
                assistant_delivery_context_recorded=False,
            ),
        )

    if snapshot_already_queued:
        assert admission.try_append_turn(lease, snapshot()) is m.AppendDisposition.ADMITTED
    _overflow(admission, overflowing, "generated")
    while writer.ordered_count:
        admission.complete_ordered_item(writer.get_nowait())
    assert _settle(admission, overflowing) is m.AppendDisposition.SESSION_TAINTED
    if revoked:
        revoke = admission.begin_revoke(_revoke_authority(m, create))
        admission.complete_revoke_request(
            writer.get_nowait(), m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        )
        assert not revoke.terminal_event.is_set()
    if snapshot_already_queued:
        assert (
            admission.record_terminal_cause(lease.terminal_cause, m.TerminalReason.PROVIDER_FAILED)
            is m.CauseDisposition.RECORDED
        )
        admission.freeze_and_resolve_terminal_causes(lease, context_committed=False)
        terminal = _turn_snapshot(
            m,
            create,
            lease,
            admission._next_event_sequence,
            m.EventKind.TURN_SETTLED,
            m.TurnSettledPayloadV1(
                evidence_turn_id=lease.evidence_turn_id,
                terminal_disposition=m.TerminalDisposition.FAILED,
                terminal_reason=m.TerminalReason.PROVIDER_FAILED,
                context_committed=False,
                generated_segment_count=0,
                transport_confirmed_full_count=0,
            ),
        )
    else:
        terminal = snapshot()
    assert admission.try_append_turn(lease, terminal) is m.AppendDisposition.SESSION_TAINTED
    assert admission.diagnostics().active_lease_count == 0
    assert admission.settled_terminal_outcome(lease) is None
    assert (
        admission.record_terminal_cause(lease.terminal_cause, m.TerminalReason.CALLER_CANCELLED)
        is m.CauseDisposition.INVALID_AUTHORITY
    )
    if revoked:
        finalizer = writer.get_nowait()
        assert type(finalizer.payload) is m.RevokeFinalizeV1
        admission.complete_revoke_finalize(finalizer, m.RevokeDisposition.PURGE_COMPLETED)
        assert revoke.terminal_event.is_set()
    assert writer.ordered_count == 0
    drain = admission.request_drain(
        _owner_drain_authority(m, create, admission, owner_generation=721)
    )
    admission.complete_drain(writer.get_nowait(), m.DrainDisposition.STOPPED)
    assert drain.terminal_event.is_set()
    assert admission.diagnostics().queue_record_count == 0
    assert admission.diagnostics().queue_canonical_bytes == 0
