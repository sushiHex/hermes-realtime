from typing import Any, cast

import pytest

from hermes_realtime.conversation import (
    ActiveTaskCapacityError,
    ActiveTaskIdentityError,
    ActiveTaskSummary,
    AssistantSegmentKey,
    AssistantTextAdmission,
    AssistantTextCapacityError,
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
    ConversationRole,
    DurableConversation,
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
    segment: AssistantSegmentKey | None = None,
    heard_text: str | None = None,
    ledger: DeliveredSpeechLedger | None = None,
) -> str:
    ledger = DeliveredSpeechLedger() if ledger is None else ledger
    chunk = SpeechChunk(
        turn_id="turn_context",
        chunk_id=chunk_id,
        text=text,
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = context.prepare_assistant_text(
        chunk.text,
        segment=AssistantSegmentKey() if segment is None else segment,
        heard_text=text if heard_text is None else heard_text,
    )
    ledger.queue(chunk, admission=admission)
    receipt = ledger.mark_started(chunk.turn_id, chunk.chunk_id)
    confirmation = ledger.mark_delivered_confirmed(receipt)
    return context.record_assistant_delivery(
        admission=admission,
        ledger=ledger,
        confirmation=confirmation,
    )


def _prepare(context: ConversationContextStore, text: str) -> AssistantTextAdmission:
    return context.prepare_assistant_text(
        text,
        segment=AssistantSegmentKey(),
        heard_text=text,
    )


def _texts(context: ConversationContextStore) -> list[str]:
    return [message.text for message in context.snapshot().messages]


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
    admission = _prepare(context, chunk.text)
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
        _prepare(context, "x" * 65)

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
    admission = _prepare(context, "forged model context")
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
    admission = _prepare(context, "expected")
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
        _prepare(context, "say deleg_future")
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
    admission = _prepare(context, "trusted delivery")
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
    first = _prepare(context, "first")
    with pytest.raises(AssistantTextCapacityError):
        _prepare(context, "second")

    context.discard_assistant_text(first)
    second = _prepare(context, "second")
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


def test_generation_time_context_write_does_not_exist() -> None:
    context = ConversationContextStore()

    assert not hasattr(context, "record_assistant_generation")
    context.validate_assistant_generation("generated but not heard")
    assert context.snapshot().messages == ()
    assert context.snapshot().revision == 0


def test_later_chunks_of_one_segment_upsert_its_single_heard_row() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    ledger = DeliveredSpeechLedger()
    segment = AssistantSegmentKey()

    _confirm_assistant_text(
        context,
        "Hello there.",
        chunk_id="chunk_001",
        segment=segment,
        heard_text="Hello there.",
        ledger=ledger,
    )
    assert _texts(context) == ["Hello there."]
    assert context.snapshot().revision == 1

    delivered = _confirm_assistant_text(
        context,
        "How are you?",
        chunk_id="chunk_002",
        segment=segment,
        heard_text="Hello there.  How are you?",
        ledger=ledger,
    )

    assert delivered == "How are you?"
    assert _texts(context) == ["Hello there.  How are you?"]
    assert context.snapshot().revision == 2

    _confirm_assistant_text(
        context,
        "Next segment.",
        chunk_id="chunk_003",
        ledger=ledger,
    )
    assert _texts(context) == ["Hello there.  How are you?", "Next segment."]


def test_heard_text_must_extend_its_open_segment_row_before_consumption() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    ledger = DeliveredSpeechLedger()
    segment = AssistantSegmentKey()
    _confirm_assistant_text(
        context,
        "Hello there.",
        chunk_id="chunk_001",
        segment=segment,
        ledger=ledger,
    )
    chunk = SpeechChunk(
        turn_id="turn_context",
        chunk_id="chunk_002",
        text="How are you?",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = context.prepare_assistant_text(
        chunk.text,
        segment=segment,
        heard_text="Rewritten history. How are you?",
    )
    ledger.queue(chunk, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(chunk.turn_id, chunk.chunk_id)
    )

    with pytest.raises(ValueError, match="extend"):
        context.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=confirmation,
        )

    assert _texts(context) == ["Hello there."]
    assert context.snapshot().revision == 1
    assert ledger.confirmed_text(confirmation, admission) == "How are you?"
    context.discard_assistant_text(admission)


def test_displaced_open_segment_row_fails_closed_without_overwriting() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    ledger = DeliveredSpeechLedger()
    segment = AssistantSegmentKey()
    _confirm_assistant_text(
        context,
        "Hello there.",
        chunk_id="chunk_001",
        segment=segment,
        ledger=ledger,
    )
    context.record_user_transcript(Transcript(text="Interjection", final=True))

    with pytest.raises(RuntimeError, match="displaced"):
        _confirm_assistant_text(
            context,
            "How are you?",
            chunk_id="chunk_002",
            segment=segment,
            heard_text="Hello there. How are you?",
            ledger=ledger,
        )

    assert _texts(context) == ["Hello there.", "Interjection"]
    assert context.snapshot().revision == 2


def test_admission_without_heard_text_consumes_delivery_without_a_row() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_context",
        chunk_id="chunk_001",
        text="unlocated speech",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = context.prepare_assistant_text(
        chunk.text,
        segment=AssistantSegmentKey(),
        heard_text=None,
    )
    ledger.queue(chunk, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(chunk.turn_id, chunk.chunk_id)
    )

    assert (
        context.record_assistant_delivery(
            admission=admission,
            ledger=ledger,
            confirmation=confirmation,
        )
        == "unlocated speech"
    )
    assert context.snapshot().messages == ()
    assert context.snapshot().revision == 0


def test_admission_segment_and_heard_text_are_strict_before_capacity_use() -> None:
    context = ConversationContextStore(
        max_messages=4,
        max_item_chars=64,
        max_pending_assistant_admissions=1,
    )

    with pytest.raises(TypeError, match="segment"):
        context.prepare_assistant_text(
            "text",
            segment=cast(AssistantSegmentKey, object()),
            heard_text="text",
        )
    with pytest.raises(TypeError, match="exact built-in string"):
        context.prepare_assistant_text(
            "text",
            segment=AssistantSegmentKey(),
            heard_text=cast(str, b"text"),
        )
    with pytest.raises(ValueError, match="blank"):
        context.prepare_assistant_text("text", segment=AssistantSegmentKey(), heard_text="  ")
    with pytest.raises(ValueError, match="max_item_chars"):
        context.prepare_assistant_text(
            "text",
            segment=AssistantSegmentKey(),
            heard_text="x" * 65,
        )
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.prepare_assistant_text(
            "text",
            segment=AssistantSegmentKey(),
            heard_text="say deleg_future",
        )

    assert _prepare(context, "capacity was not consumed") is not None


def test_interruption_marker_applies_once_to_the_open_heard_row() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    segment = AssistantSegmentKey()
    _confirm_assistant_text(context, "Partial reply.", chunk_id="chunk_001", segment=segment)

    context.mark_assistant_segment_interrupted(segment)

    interrupted = ConversationMessage("assistant", "Partial reply.", interrupted=True)
    assert context.snapshot().messages == (interrupted,)
    assert context.snapshot().messages[0].interrupted is True
    assert context.snapshot().revision == 2
    with pytest.raises(KeyError, match="open heard"):
        context.mark_assistant_segment_interrupted(segment)
    with pytest.raises(RuntimeError, match="displaced"):
        _confirm_assistant_text(
            context,
            "More.",
            chunk_id="chunk_002",
            segment=segment,
            heard_text="Partial reply. More.",
        )
    assert context.snapshot().messages == (interrupted,)
    assert context.snapshot().revision == 2


def test_interruption_marker_requires_the_exact_open_heard_row() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    segment = AssistantSegmentKey()

    with pytest.raises(TypeError, match="segment"):
        context.mark_assistant_segment_interrupted(cast(AssistantSegmentKey, object()))
    with pytest.raises(KeyError, match="open heard"):
        context.mark_assistant_segment_interrupted(segment)

    _confirm_assistant_text(context, "Heard reply.", chunk_id="chunk_001", segment=segment)
    with pytest.raises(KeyError, match="open heard"):
        context.mark_assistant_segment_interrupted(AssistantSegmentKey())
    context.close_assistant_segment(segment)
    context.record_user_transcript(Transcript(text="Next question", final=True))
    with pytest.raises(KeyError, match="open heard"):
        context.mark_assistant_segment_interrupted(segment)

    assert context.snapshot().messages == (
        ConversationMessage("assistant", "Heard reply."),
        ConversationMessage("user", "Next question"),
    )
    assert not any(message.interrupted for message in context.snapshot().messages)


def test_interrupted_row_keeps_exact_heard_text_at_the_per_item_limit() -> None:
    context = ConversationContextStore(max_item_chars=64)
    segment = AssistantSegmentKey()
    _confirm_assistant_text(context, "x" * 64, chunk_id="chunk_001", segment=segment)

    context.mark_assistant_segment_interrupted(segment)

    [message] = context.snapshot().messages
    assert message.text == "x" * 64
    assert len(message.text) == context.max_item_chars
    assert message.interrupted is True


def test_store_accepts_the_full_supported_item_bound() -> None:
    assert ConversationContextStore(max_item_chars=65_536).max_item_chars == 65_536


def test_interrupted_flag_is_an_exact_boolean_on_assistant_rows_only() -> None:
    with pytest.raises(TypeError, match="exact boolean"):
        ConversationMessage("assistant", "cut", interrupted=cast(bool, 1))
    with pytest.raises(ValueError, match="assistant"):
        ConversationMessage("user", "not speech", interrupted=True)

    mutated = ConversationMessage("user", "question")
    object.__setattr__(mutated, "interrupted", True)
    with pytest.raises(ValueError, match="assistant"):
        ConversationContextSnapshot(0, (mutated,), ())

    snapshot = ConversationContextSnapshot(
        0,
        (ConversationMessage("assistant", "cut", interrupted=True),),
        (),
    )
    assert snapshot.messages[0].interrupted is True


def _durable_rows(context: ConversationContextStore) -> list[tuple[str, str, bool]]:
    return [
        (message.role, message.text, message.interrupted)
        for message in context.durable_view().messages
    ]


def _view(
    *messages: ConversationMessage,
    prior_work: bool = False,
) -> DurableConversation:
    return DurableConversation(messages=messages, prior_work=prior_work)


def test_closing_a_completed_segment_ends_its_open_heard_row() -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    segment = AssistantSegmentKey()
    _confirm_assistant_text(context, "Whole reply.", chunk_id="chunk_001", segment=segment)
    assert _durable_rows(context) == [("assistant", "Whole reply.", True)]

    context.close_assistant_segment(segment)

    assert _durable_rows(context) == [("assistant", "Whole reply.", False)]
    assert context.snapshot().messages == (ConversationMessage("assistant", "Whole reply."),)
    with pytest.raises(KeyError, match="open heard"):
        context.mark_assistant_segment_interrupted(segment)
    with pytest.raises(TypeError, match="segment"):
        context.close_assistant_segment(cast(AssistantSegmentKey, object()))


def test_closing_a_segment_that_is_not_open_is_a_no_op() -> None:
    # A displaced or already-closed row is final, so the success path never raises.
    changes: list[DurableConversation] = []
    context = ConversationContextStore(max_messages=4, max_item_chars=64, on_change=changes.append)
    segment = AssistantSegmentKey()
    _confirm_assistant_text(context, "Open reply.", chunk_id="chunk_001", segment=segment)
    notified = len(changes)

    context.close_assistant_segment(AssistantSegmentKey())

    assert _durable_rows(context) == [("assistant", "Open reply.", True)]
    assert len(changes) == notified
    context.close_assistant_segment(segment)
    context.close_assistant_segment(segment)
    assert _durable_rows(context) == [("assistant", "Open reply.", False)]
    assert len(changes) == notified + 1


def test_durable_view_flags_only_the_still_open_assistant_row() -> None:
    context = ConversationContextStore(max_messages=8, max_item_chars=64)
    context.record_user_transcript(Transcript(text="First question", final=True))
    first = AssistantSegmentKey()
    _confirm_assistant_text(context, "First answer.", chunk_id="chunk_001", segment=first)
    context.close_assistant_segment(first)
    second = AssistantSegmentKey()
    _confirm_assistant_text(context, "Second", chunk_id="chunk_002", segment=second)

    assert _durable_rows(context) == [
        ("user", "First question", False),
        ("assistant", "First answer.", False),
        ("assistant", "Second", True),
    ]
    # Live context stays heard-first: the open row is not interrupted yet.
    assert [message.interrupted for message in context.snapshot().messages] == [
        False,
        False,
        False,
    ]
    view = context.durable_view()
    assert type(view) is DurableConversation
    assert type(view.messages) is tuple
    assert view.prior_work is False


def test_durable_conversation_is_an_exact_frozen_value() -> None:
    view = _view(ConversationMessage("user", "Question"), prior_work=True)

    with pytest.raises(AttributeError):
        view.prior_work = False  # type: ignore[misc]
    with pytest.raises(TypeError, match="prior_work"):
        DurableConversation(messages=(), prior_work=cast(bool, 1))
    with pytest.raises(TypeError, match="messages"):
        DurableConversation(
            messages=cast(tuple[ConversationMessage, ...], [ConversationMessage("user", "x")]),
            prior_work=False,
        )
    with pytest.raises(TypeError, match="messages"):
        DurableConversation(
            messages=cast(tuple[ConversationMessage, ...], (("user", "x", False),)),
            prior_work=False,
        )


def test_restore_replaces_a_pristine_store_with_the_exact_tail() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(
        max_messages=4,
        max_item_chars=64,
        on_change=changes.append,
    )
    tail = _view(
        ConversationMessage("user", "Earlier question"),
        ConversationMessage("assistant", "Earlier answer", interrupted=True),
    )

    context.restore(tail)

    assert context.snapshot().messages == tail.messages
    assert context.snapshot().revision == 1
    assert context.snapshot().terminal_task_count == 0
    assert context.durable_view() == tail
    assert changes == [tail]
    with pytest.raises(RuntimeError, match="pristine"):
        context.restore(tail)


def test_restoring_prior_work_renders_inactive_history_and_never_active_work() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(on_change=changes.append)
    tail = _view(ConversationMessage("user", "Start the report"), prior_work=True)

    context.restore(tail)

    snapshot = context.snapshot()
    assert snapshot.terminal_task_count == 1
    assert snapshot.active_tasks == ()
    assert snapshot.revision == 1
    assert context.durable_view() == tail
    assert changes == [tail]


def test_restore_accepts_a_tail_at_the_exact_bounds() -> None:
    context = ConversationContextStore(max_messages=2, max_item_chars=8)
    tail = _view(ConversationMessage("user", "x" * 8), ConversationMessage("assistant", "y" * 8))

    context.restore(tail)

    assert context.snapshot().messages == tail.messages


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(
            _view(
                ConversationMessage("user", "one"),
                ConversationMessage("assistant", "two"),
                ConversationMessage("user", "three"),
            ),
            id="more-rows-than-max-messages",
        ),
        pytest.param(_view(ConversationMessage("user", "x" * 9)), id="row-over-max-item-chars"),
    ],
)
def test_restore_refuses_a_tail_outside_this_stores_bounds(tail: DurableConversation) -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(max_messages=2, max_item_chars=8, on_change=changes.append)

    with pytest.raises(ValueError):
        context.restore(tail)

    assert context.snapshot().messages == ()
    assert context.snapshot().revision == 0
    assert changes == []


def _forged(role: str, text: str, interrupted: bool = False) -> ConversationMessage:
    message = ConversationMessage("assistant", "placeholder")
    object.__setattr__(message, "role", role)
    object.__setattr__(message, "text", text)
    object.__setattr__(message, "interrupted", interrupted)
    return message


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_forged("system", "hello"), id="unknown-role"),
        pytest.param(_forged("user", "hello", True), id="interrupted-user"),
        pytest.param(_forged("assistant", "   "), id="blank-text"),
        pytest.param(_forged("assistant", "said deleg_private"), id="private-run-token"),
        pytest.param(_forged("assistant", cast(str, b"bytes")), id="non-string-text"),
        pytest.param(_forged("assistant", "lone \ud800 surrogate"), id="not-utf8-encodable"),
    ],
)
def test_restore_refuses_the_whole_tail_on_any_invalid_row(row: ConversationMessage) -> None:
    context = ConversationContextStore(max_messages=4, max_item_chars=64)
    tail = _view(ConversationMessage("user", "valid first row"))
    object.__setattr__(tail, "messages", (*tail.messages, row))

    with pytest.raises(ValueError):
        context.restore(tail)

    assert context.snapshot().messages == ()
    assert context.snapshot().revision == 0


def test_restore_is_refused_once_the_store_has_any_state() -> None:
    tail = _view(ConversationMessage("user", "Earlier question"))
    with_row = ConversationContextStore()
    with_row.record_user_transcript(Transcript(text="live", final=True))
    with_admission = ConversationContextStore()
    _prepare(with_admission, "pending speech")
    with_task_admission = ConversationContextStore()
    with_task_admission.prepare_task("task_pending", "Pending objective")
    with_task = ConversationContextStore()
    with_task.record_task_accepted(task_id="task_live", run_id="deleg_live", objective="Live")

    for context in (with_row, with_admission, with_task_admission, with_task):
        with pytest.raises(RuntimeError, match="pristine"):
            context.restore(tail)
    with pytest.raises(TypeError):
        ConversationContextStore().restore(cast(DurableConversation, tail.messages))


def test_on_change_receives_the_durable_view_after_every_row_mutation() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(max_messages=8, max_item_chars=64, on_change=changes.append)

    context.record_user_transcript(Transcript(text="Question", final=True))
    assert changes[-1] == _view(ConversationMessage("user", "Question"))

    segment = AssistantSegmentKey()
    _confirm_assistant_text(context, "Partial", chunk_id="chunk_001", segment=segment)
    assert changes[-1].messages[-1] == ConversationMessage("assistant", "Partial", interrupted=True)

    context.close_assistant_segment(segment)
    assert changes[-1].messages[-1] == ConversationMessage("assistant", "Partial")

    interrupted = AssistantSegmentKey()
    _confirm_assistant_text(context, "Cut", chunk_id="chunk_002", segment=interrupted)
    context.mark_assistant_segment_interrupted(interrupted)
    assert changes[-1].messages[-1] == ConversationMessage("assistant", "Cut", interrupted=True)
    assert len(changes) == 5
    assert changes[-1] == context.durable_view()


def test_on_change_fires_when_prior_work_flips_and_never_otherwise_for_tasks() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(max_messages=8, max_item_chars=64, on_change=changes.append)

    admission = context.prepare_task("task_b", "Other objective")
    context.discard_task(admission)
    assert changes == []

    # Active work is prior work too: a restart never resumes it.
    context.record_task_accepted(task_id="task_a", run_id="deleg_a", objective="Objective")
    assert changes == [_view(prior_work=True)]
    context.record_task_completed(task_id="task_a", run_id="deleg_a")
    context.mark_prior_work_ended()
    assert changes == [_view(prior_work=True)]

    reserved = ConversationContextStore(on_change=changes.append)
    task = reserved.prepare_task("task_c", "Reserved objective")
    reserved.record_reserved_task_accepted(admission=task, run_id="deleg_c")
    assert changes[-1] == _view(prior_work=True)
    assert len(changes) == 2


def test_on_change_ignores_unrecorded_speech() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(max_messages=8, max_item_chars=64, on_change=changes.append)
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_context",
        chunk_id="chunk_001",
        text="unlocated speech",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = context.prepare_assistant_text(
        chunk.text,
        segment=AssistantSegmentKey(),
        heard_text=None,
    )
    ledger.queue(chunk, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(chunk.turn_id, chunk.chunk_id)
    )
    context.record_assistant_delivery(
        admission=admission,
        ledger=ledger,
        confirmation=confirmation,
    )
    discarded = _prepare(context, "never played")
    context.discard_assistant_text(discarded)

    assert changes == []


def test_on_change_must_be_callable_and_its_failure_propagates() -> None:
    with pytest.raises(TypeError, match="on_change"):
        ConversationContextStore(on_change=cast(Any, "not callable"))

    def failing(_view: DurableConversation) -> None:
        raise RuntimeError("tail mirror bug")

    context = ConversationContextStore(on_change=failing)
    with pytest.raises(RuntimeError, match="tail mirror bug"):
        context.record_user_transcript(Transcript(text="Question", final=True))


def test_marking_prior_work_ended_is_idempotent_inactive_history() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(on_change=changes.append)
    context.restore(_view(ConversationMessage("user", "Earlier question")))

    context.mark_prior_work_ended()
    context.mark_prior_work_ended()

    snapshot = context.snapshot()
    assert snapshot.terminal_task_count == 1
    assert snapshot.active_tasks == ()
    assert snapshot.revision == 2
    assert context.durable_view().prior_work is True
    assert [change.prior_work for change in changes] == [False, True]


def test_marking_prior_work_keeps_real_terminal_history() -> None:
    context = ConversationContextStore()
    for index in range(2):
        context.record_task_accepted(
            task_id=f"task_{index}", run_id=f"deleg_{index}", objective="Objective"
        )
        context.record_task_completed(task_id=f"task_{index}", run_id=f"deleg_{index}")
    revision = context.snapshot().revision

    context.mark_prior_work_ended()

    assert context.snapshot().terminal_task_count == 2
    assert context.snapshot().revision == revision


def test_validating_user_text_is_pure_and_matches_what_the_store_records() -> None:
    changes: list[DurableConversation] = []
    context = ConversationContextStore(max_item_chars=8, on_change=changes.append)

    context.validate_user_text("task: x")
    for text in ("x" * 9, "   ", "bad \ud800"):
        with pytest.raises(ValueError):
            context.validate_user_text(text)
        with pytest.raises(ValueError):
            context.record_user_transcript(Transcript(text=text or "x", final=True))
    with pytest.raises(ActiveTaskIdentityError, match="private run"):
        context.validate_user_text("deleg_ab")
    with pytest.raises(TypeError):
        context.validate_user_text(cast(str, b"bytes"))

    assert context.snapshot().revision == 0
    assert changes == []
