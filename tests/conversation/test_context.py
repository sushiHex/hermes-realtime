from typing import cast

import pytest

from hermes_realtime.conversation import (
    ActiveTaskCapacityError,
    ActiveTaskIdentityError,
    ActiveTaskSummary,
    AssistantTextCapacityError,
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
    ConversationRole,
)
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechConfirmation,
    DeliveredSpeechLedger,
    SpeechChunk,
    Transcript,
)


def _confirm_assistant_text(
    context: ConversationContextStore,
    text: str,
    *,
    chunk_id: str,
) -> None:
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_context",
        chunk_id=chunk_id,
        text=text,
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = context.prepare_assistant_text(chunk.text)
    ledger.queue(chunk, admission=admission)
    receipt = ledger.mark_started(chunk.turn_id, chunk.chunk_id)
    confirmation = ledger.mark_delivered_confirmed(receipt)
    context.record_assistant_delivery(
        admission=admission,
        ledger=ledger,
        confirmation=confirmation,
    )


def test_snapshot_is_immutable_and_message_retention_is_bounded() -> None:
    context = ConversationContextStore(
        max_messages=2,
        max_item_chars=64,
    )

    context.record_user_transcript(Transcript(text="first question", final=True))
    first = context.snapshot()
    exposed = context.snapshot()
    assert type(exposed.messages[0].role) is str
    object.__setattr__(exposed.messages[0], "text", "untrusted rewrite")

    assert context.snapshot().messages == (
        ConversationMessage(ConversationRole.USER.value, "first question"),
    )
    assert context.snapshot().revision == 1

    _confirm_assistant_text(context, "first answer", chunk_id="chunk_001")
    context.record_user_transcript(Transcript(text="second question", final=True))

    assert first.revision == 1
    assert first.messages == (
        ConversationMessage(ConversationRole.USER.value, "first question"),
    )
    assert context.snapshot().revision == 3
    assert context.snapshot().messages == (
        ConversationMessage(ConversationRole.ASSISTANT.value, "first answer"),
        ConversationMessage(ConversationRole.USER.value, "second question"),
    )


def test_projected_value_constructors_require_exact_persistent_types() -> None:
    touched = False

    class SideEffectString(str):
        def strip(self, chars: str | None = None) -> str:
            nonlocal touched
            touched = True
            return super().strip(chars)

    with pytest.raises(TypeError):
        ConversationMessage("user", SideEffectString("untrusted"))
    with pytest.raises(TypeError):
        ActiveTaskSummary(
            task_id=SideEffectString("task_001"),
            objective=SideEffectString("untrusted"),
        )
    with pytest.raises(TypeError, match="exact integer"):
        ConversationContextSnapshot(cast(int, True), (), ())

    assert touched is False


def test_projected_value_constructors_enforce_private_and_size_bounds() -> None:
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        ConversationMessage("user", "leak deleg_future")
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        ActiveTaskSummary("task_deleg_future", "ordinary")
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        ActiveTaskSummary("task_public", "leak deleg_future")
    with pytest.raises(ValueError, match="supported maximum"):
        ConversationMessage("assistant", "x" * 65_537)
    with pytest.raises(ValueError, match="supported maximum"):
        ActiveTaskSummary("task_public", "x" * 65_537)

    message = ConversationMessage("user", "bounded")
    task = ActiveTaskSummary("task_public", "bounded")
    with pytest.raises(ValueError, match="message capacity"):
        ConversationContextSnapshot(0, (message,) * 1_025, ())
    with pytest.raises(ValueError, match="task capacity"):
        ConversationContextSnapshot(0, (), (task,) * 257)
    with pytest.raises(ValueError, match="supported maximum"):
        ConversationContextSnapshot(2**63, (), ())


def test_snapshot_constructor_revalidates_and_detaches_nested_values() -> None:
    mutated_message = ConversationMessage("user", "safe")
    object.__setattr__(mutated_message, "text", "leak deleg_mutated")
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        ConversationContextSnapshot(0, (mutated_message,), ())

    mutated_task = ActiveTaskSummary("task_public", "safe")
    object.__setattr__(mutated_task, "objective", "x" * 65_537)
    with pytest.raises(ValueError, match="supported maximum"):
        ConversationContextSnapshot(0, (), (mutated_task,))

    original_message = ConversationMessage("user", "detached")
    original_task = ActiveTaskSummary("task_public", "detached")
    snapshot = ConversationContextSnapshot(0, (original_message,), (original_task,))
    object.__setattr__(original_message, "text", "leak deleg_after")
    object.__setattr__(original_task, "objective", "leak deleg_after")

    assert snapshot.messages == (ConversationMessage("user", "detached"),)
    assert snapshot.active_tasks == (ActiveTaskSummary("task_public", "detached"),)


def test_task_reservation_is_invisible_until_authoritative_acceptance() -> None:
    context = ConversationContextStore(max_active_tasks=1)

    admission = context.prepare_task("task_weather", "check the forecast")

    assert context.snapshot().active_tasks == ()

    context.record_reserved_task_accepted(admission=admission, run_id="deleg_weather")

    assert context.snapshot().active_tasks == (
        ActiveTaskSummary("task_weather", "check the forecast"),
    )


def test_discarded_task_reservation_releases_capacity_but_retires_identity() -> None:
    context = ConversationContextStore(max_active_tasks=1)
    admission = context.prepare_task("task_first", "first")

    context.discard_task(admission)

    context.prepare_task("task_second", "second")
    with pytest.raises(ActiveTaskIdentityError, match="active or retired"):
        context.prepare_task("task_first", "retry")


def test_active_task_snapshot_is_immutable_and_completion_removes_it() -> None:
    context = ConversationContextStore(
        max_messages=2,
        max_active_tasks=1,
        max_item_chars=64,
    )
    context.record_task_accepted(
        task_id="task_001",
        run_id="deleg_001",
        objective="Compare the two deployment options.",
    )

    active = context.snapshot()
    context.record_task_completed(task_id="task_001", run_id="deleg_001")

    assert active.active_tasks == (
        ActiveTaskSummary(
            task_id="task_001",
            objective="Compare the two deployment options.",
        ),
    )
    assert not hasattr(active.active_tasks[0], "run_id")
    assert active.terminal_task_count == 0
    completed = context.snapshot()
    assert completed.active_tasks == ()
    assert completed.terminal_task_count == 1
    assert completed.revision == 2


def test_active_task_capacity_fails_closed_without_evicting_live_work() -> None:
    context = ConversationContextStore(
        max_messages=1,
        max_active_tasks=1,
        max_item_chars=64,
    )
    context.record_task_accepted(
        task_id="task_001",
        run_id="deleg_001",
        objective="First task",
    )

    with pytest.raises(ActiveTaskCapacityError):
        context.record_task_accepted(
            task_id="task_002",
            run_id="deleg_002",
            objective="Second task",
        )

    assert tuple(task.task_id for task in context.snapshot().active_tasks) == (
        "task_001",
    )
    assert context.snapshot().revision == 1


def test_active_task_identity_conflicts_fail_closed() -> None:
    context = ConversationContextStore(
        max_messages=1,
        max_active_tasks=2,
        max_item_chars=64,
    )
    context.record_task_accepted(
        task_id="task_001",
        run_id="deleg_001",
        objective="Original",
    )

    with pytest.raises(ActiveTaskIdentityError):
        context.record_task_accepted(
            task_id="task_001",
            run_id="deleg_002",
            objective="Conflicting task identity",
        )
    with pytest.raises(ActiveTaskIdentityError):
        context.record_task_accepted(
            task_id="task_002",
            run_id="deleg_001",
            objective="Conflicting run identity",
        )

    assert context.snapshot().active_tasks[0].objective == "Original"
    assert context.snapshot().revision == 1


def test_task_completion_requires_exact_active_task_and_run_pair() -> None:
    context = ConversationContextStore(
        max_messages=1,
        max_active_tasks=1,
        max_item_chars=64,
    )
    context.record_task_accepted(
        task_id="task_001",
        run_id="deleg_001",
        objective="Active",
    )
    before = context.snapshot()

    with pytest.raises(ActiveTaskIdentityError):
        context.record_task_completed(task_id="task_001", run_id="deleg_wrong")
    with pytest.raises(ActiveTaskIdentityError):
        context.record_task_completed(task_id="task_unknown", run_id="deleg_001")

    assert context.snapshot() == before


def test_partial_user_transcript_is_rejected() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)

    with pytest.raises(ValueError, match="final"):
        context.record_user_transcript(Transcript(text="still speaking", final=False))

    assert context.snapshot().revision == 0
    assert context.snapshot().messages == ()


def test_user_transcript_requires_exact_value_type_before_field_access() -> None:
    touched = False

    class TranscriptLike:
        @property
        def final(self) -> bool:
            nonlocal touched
            touched = True
            return True

        text = "untrusted"

    context = ConversationContextStore(max_messages=2, max_item_chars=64)

    with pytest.raises(TypeError):
        context.record_user_transcript(cast(Transcript, TranscriptLike()))

    assert touched is False
    assert context.snapshot().revision == 0


def test_transcript_final_marker_requires_exact_boolean() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)

    with pytest.raises(TypeError):
        context.record_user_transcript(
            Transcript(text="not a trustworthy final marker", final=cast(bool, 1))
        )

    assert context.snapshot().revision == 0


def test_assistant_text_enters_context_only_after_delivery_confirmation() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="heard answer",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = context.prepare_assistant_text(chunk.text)
    ledger.queue(chunk, admission=admission)
    receipt = ledger.mark_started(chunk.turn_id, chunk.chunk_id)

    assert context.snapshot().messages == ()
    confirmation = ledger.mark_delivered_confirmed(receipt)
    delivered = context.record_assistant_delivery(
        admission=admission,
        ledger=ledger,
        confirmation=confirmation,
    )

    assert delivered == chunk.text
    assert context.snapshot().messages == (
        ConversationMessage(ConversationRole.ASSISTANT.value, "heard answer"),
    )
    assert ledger.delivered_text("turn_001") == "heard answer"


def test_assistant_text_bound_is_checked_before_playback_admission() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)
    ledger = DeliveredSpeechLedger()

    with pytest.raises(ValueError, match="max_item_chars"):
        context.prepare_assistant_text("x" * 65)

    assert ledger.retained_chunk_count == 0
    assert context.snapshot().revision == 0


def test_task_summary_inputs_are_strict_and_bounded_before_mutation() -> None:
    context = ConversationContextStore(
        max_messages=1,
        max_active_tasks=2,
        max_item_chars=16,
    )

    with pytest.raises(TypeError):
        context.record_task_accepted(
            task_id=cast(str, 123),
            run_id="deleg_001",
            objective="valid",
        )
    with pytest.raises(ValueError):
        context.record_task_accepted(
            task_id="task with spaces",
            run_id="deleg_001",
            objective="valid",
        )
    with pytest.raises(ValueError, match="max_item_chars"):
        context.record_task_accepted(
            task_id="task_001",
            run_id="deleg_001",
            objective="x" * 17,
        )

    assert context.snapshot().active_tasks == ()
    assert context.snapshot().revision == 0


@pytest.mark.parametrize(
    "name",
    [
        "max_messages",
        "max_active_tasks",
        "max_item_chars",
        "max_task_incarnations",
        "max_pending_assistant_admissions",
    ],
)
def test_context_rejects_extreme_constructor_bounds(name: str) -> None:
    with pytest.raises(ValueError, match="supported maximum"):
        ConversationContextStore(**{name: 10**100})


def test_task_and_private_run_identifier_namespaces_cannot_overlap() -> None:
    context = ConversationContextStore(max_messages=1, max_active_tasks=2)

    with pytest.raises(ActiveTaskIdentityError, match="namespace"):
        context.record_task_accepted(
            task_id="deleg_private_run_9f3d",
            run_id="deleg_private_run_9f3d",
            objective="Must remain private",
        )
    with pytest.raises(ActiveTaskIdentityError, match="namespace"):
        context.record_task_accepted(
            task_id="task_public_001",
            run_id="task_private_001",
            objective="Reserved namespace",
        )

    assert context.snapshot().revision == 0
    assert context.snapshot().active_tasks == ()


def test_forged_delivery_confirmation_cannot_enter_context() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)
    admission = context.prepare_assistant_text("forged model context")
    forged = object.__new__(DeliveredSpeechConfirmation)

    with pytest.raises(KeyError, match="delivery confirmation"):
        context.record_assistant_delivery(
            admission=admission,
            ledger=DeliveredSpeechLedger(),
            confirmation=forged,
        )

    assert context.snapshot().revision == 0
    assert context.snapshot().messages == ()
    assert not hasattr(context, "record_delivered_assistant_text")


def test_mismatched_delivery_preserves_both_capabilities_without_projection() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)
    admission = context.prepare_assistant_text("expected")
    ledger = DeliveredSpeechLedger()
    item = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="different",
        audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=16_000, channels=1),
    )
    ledger.queue(item, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(item.turn_id, item.chunk_id)
    )

    with pytest.raises(ValueError, match="does not match"):
        context.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=confirmation,
        )

    assert ledger.confirmed_text(confirmation, admission) == item.text
    context.discard_assistant_text(admission)
    assert context.snapshot().revision == 0


def test_completed_task_identity_cannot_be_reused() -> None:
    context = ConversationContextStore(max_messages=1, max_active_tasks=1)
    context.record_task_accepted(
        task_id="task_public_001",
        run_id="deleg_private_001",
        objective="First incarnation",
    )
    context.record_task_completed(
        task_id="task_public_001",
        run_id="deleg_private_001",
    )

    with pytest.raises(ActiveTaskIdentityError, match="retired"):
        context.record_task_accepted(
            task_id="task_public_001",
            run_id="deleg_private_001",
            objective="Confusable second incarnation",
        )

    assert context.snapshot().revision == 2
    assert context.snapshot().active_tasks == ()


def test_private_run_handle_cannot_be_embedded_in_model_visible_task_fields() -> None:
    context = ConversationContextStore(max_messages=1, max_active_tasks=2)

    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.record_task_accepted(
            task_id="task_deleg_private_001",
            run_id="deleg_private_001",
            objective="normal objective",
        )
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.record_task_accepted(
            task_id="task_public_001",
            run_id="deleg_private_001",
            objective="Inspect deleg_private_001",
        )

    context.record_task_accepted(
        task_id="task_001",
        run_id="deleg_001",
        objective="First task",
    )
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.record_task_accepted(
            task_id="task_002",
            run_id="deleg_002",
            objective="Disclose deleg_001",
        )
    context.record_task_completed(task_id="task_001", run_id="deleg_001")
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.record_task_accepted(
            task_id="task_deleg_001",
            run_id="deleg_002",
            objective="Second task",
        )
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.record_task_accepted(
            task_id="task_future",
            run_id="deleg_002",
            objective="Future handle deleg_future",
        )

    assert context.snapshot().active_tasks == ()


def test_private_run_namespace_is_reserved_before_future_identity_exists() -> None:
    context = ConversationContextStore(max_messages=2, max_active_tasks=1)

    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.record_user_transcript(Transcript("heard deleg_future", final=True))
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.prepare_assistant_text("say deleg_future")
    with pytest.raises(ActiveTaskIdentityError, match="deleg_ namespace"):
        context.record_task_accepted(
            task_id="task_001",
            run_id="run_future",
            objective="ordinary task",
        )

    assert context.snapshot().revision == 0


def test_assistant_admission_and_confirmation_are_identity_bound() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=64)
    ledger = DeliveredSpeechLedger()
    admission = context.prepare_assistant_text("trusted delivery")
    item = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="trusted delivery",
        audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=16_000, channels=1),
    )
    ledger.queue(item, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(item.turn_id, item.chunk_id)
    )
    with pytest.raises(AttributeError):
        object.__setattr__(admission, "text", "mutated admission")
    with pytest.raises(AttributeError):
        object.__setattr__(confirmation, "confirmation_id", 999)

    delivered = context.record_assistant_delivery(
        admission=admission,
        ledger=ledger,
        confirmation=confirmation,
    )

    assert delivered == "trusted delivery"
    assert context.snapshot().messages == (
        ConversationMessage(ConversationRole.ASSISTANT.value, "trusted delivery"),
    )
    with pytest.raises(KeyError, match="admission"):
        context.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=confirmation,
        )


def test_assistant_admissions_are_bounded_and_releasable() -> None:
    context = ConversationContextStore(
        max_messages=1,
        max_pending_assistant_admissions=1,
    )
    first = context.prepare_assistant_text("first")
    with pytest.raises(AssistantTextCapacityError):
        context.prepare_assistant_text("second")

    context.discard_assistant_text(first)
    second = context.prepare_assistant_text("second")
    assert second is not first


def test_task_tombstone_capacity_is_bounded() -> None:
    context = ConversationContextStore(
        max_messages=1,
        max_active_tasks=1,
        max_task_incarnations=1,
    )
    context.record_task_accepted(
        task_id="task_001",
        run_id="deleg_001",
        objective="one",
    )
    context.record_task_completed(task_id="task_001", run_id="deleg_001")

    with pytest.raises(ActiveTaskCapacityError, match="incarnation"):
        context.record_task_accepted(
            task_id="task_002",
            run_id="deleg_002",
            objective="two",
        )

    assert context.snapshot().revision == 2
