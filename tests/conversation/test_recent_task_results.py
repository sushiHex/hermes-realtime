"""Completed work remains model reference data, independently of heard speech."""

import pytest

from hermes_realtime.conversation import ConversationContextStore, ConversationTaskController
from tests.conversation.test_tasks import QueuedUpdateSession, terminal_event


@pytest.mark.asyncio
async def test_validated_terminal_is_in_snapshot_before_completion_consumption():
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_001",
        id_factory=iter(("weather", "dispatch_weather")).__next__,
    )
    try:
        await controller.dispatch(objective="check weather", utterance_id="utterance_001")
        assert context.snapshot().recent_task_results == ()
        await session.updates.put(terminal_event())
        completion = await controller.next_completion()
        snapshot = context.snapshot()
        assert completion.status == "completed"
        assert len(snapshot.recent_task_results) == 1
        result = snapshot.recent_task_results[0]
        assert (result.task_id, result.objective, result.status, result.text) == (
            "task_weather",
            "check weather",
            "completed",
            "Finished",
        )
        assert snapshot.messages == ()
        context.clear_voice_history()
        assert context.snapshot().recent_task_results == ()
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_followup_and_both_provider_prompts_receive_reference():
    from hermes_realtime.conversation import ForegroundTurnCoordinator, StreamingSpeechLoop
    from hermes_realtime.conversation.context import RecentTaskResult
    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
    from hermes_realtime.providers.ollama import OllamaStreamingInference
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
    from tests.conversation.test_streaming import (
        RecordingPlayback,
        RequestRecordingInference,
        SegmentSynthesizer,
    )

    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_weather", run_id="deleg_12345678", objective="weather"
    )
    result = RecentTaskResult("task_weather", "weather", "completed", "Sunny, source: example.com")
    context.record_task_completed(
        task_id="task_weather",
        run_id="deleg_12345678",
        result_status=result.status,
        result_text=result.text,
    )
    inference = RequestRecordingInference()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    await loop.respond("turn_followup", Transcript(text="How is it going?", final=True))
    request = inference.requests[0]
    assert request.recent_task_results == (result,)
    assert request.context.recent_task_results == (result,)
    assert "Sunny, source: example.com" in CodexAppServerStreamingInference._prompt(request)
    assert "Sunny, source: example.com" in OllamaStreamingInference._messages(request)[0]["content"]


def test_reference_budget_preserves_latest_heard_and_durable_history():
    from hermes_realtime.speech import Transcript

    context = ConversationContextStore(max_item_chars=20, max_window_chars=40)
    context.record_user_transcript(Transcript(text="older user message", final=True))
    context.record_user_transcript(Transcript(text="latest user message", final=True))
    for n in range(10):
        context.record_task_accepted(task_id=f"task_{n}", run_id=f"deleg_{n}", objective="work")
        context.record_task_completed(
            task_id=f"task_{n}",
            run_id=f"deleg_{n}",
            result_status="completed",
            result_text="result",
        )
    snapshot = context.snapshot()
    assert [row.text for row in snapshot.messages] == ["latest user message"]
    assert [row.task_id for row in snapshot.recent_task_results] == ["task_8", "task_9"]
    assert (
        sum(len(row.text) for row in snapshot.messages)
        + sum(len(row.objective) + len(row.text) for row in snapshot.recent_task_results)
        <= 40
    )
    assert [row.text for row in context.durable_view().messages] == [
        "older user message",
        "latest user message",
    ]


def test_reference_is_source_safe_and_status_preserving():
    from hermes_realtime.conversation.result_projection import completed_result_reference

    text = completed_result_reference(
        "At 4pm [Source](https://user:secret@example.com/path(a)?token=hidden#frag) "
        "and https://user:secret@example.org/path(b)?q=hidden xdeleg_secret"
    )
    assert "Source" in text and "example.org" in text and "4pm" in text
    for secret in ("secret", "hidden", "path(a)", "path(b)", "deleg_"):
        assert secret not in text
    context = ConversationContextStore()
    for n, status in enumerate(("failed", "interrupted")):
        context.record_task_accepted(task_id=f"task_{n}", run_id=f"deleg_{n}", objective="work")
        context.record_task_completed(
            task_id=f"task_{n}", run_id=f"deleg_{n}", result_status=status, result_text=text
        )
    assert [row.status for row in context.snapshot().recent_task_results] == [
        "failed",
        "interrupted",
    ]


@pytest.mark.parametrize("heard", ["none", "partial", "full"])
def test_followup_snapshots_are_point_in_time_and_independent_of_heard_speech(heard):
    from hermes_realtime.conversation import AssistantSegmentKey
    from tests.conversation.test_context import _confirm_assistant_text

    context = ConversationContextStore()
    context.record_task_accepted(task_id="task_weather", run_id="deleg_123", objective="weather")
    before_terminal = context.snapshot()
    context.record_task_completed(
        task_id="task_weather",
        run_id="deleg_123",
        result_status="completed",
        result_text="Sunny at noon, example.com",
    )
    if heard != "none":
        segment = AssistantSegmentKey()
        _confirm_assistant_text(
            context,
            "Sunny" if heard == "partial" else "Sunny at noon.",
            chunk_id="chunk_result",
            segment=segment,
        )
        if heard == "partial":
            context.mark_assistant_segment_interrupted(segment)
    assert before_terminal.recent_task_results == ()
    assert context.snapshot().recent_task_results[0].text == "Sunny at noon, example.com"


def test_only_matching_terminal_records_and_fifo_is_bounded():
    from hermes_realtime.conversation import ActiveTaskIdentityError

    context = ConversationContextStore()
    context.record_task_accepted(task_id="task_first", run_id="deleg_first", objective="first")
    with pytest.raises(ActiveTaskIdentityError):
        context.record_task_completed(
            task_id="task_first",
            run_id="deleg_wrong",
            result_status="failed",
            result_text="No result",
        )
    assert context.snapshot().recent_task_results == ()
    context.record_task_completed(
        task_id="task_first", run_id="deleg_first", result_status="failed", result_text="No result"
    )
    for n in range(9):
        context.record_task_accepted(task_id=f"task_{n}", run_id=f"deleg_{n}", objective="work")
        context.record_task_completed(
            task_id=f"task_{n}", run_id=f"deleg_{n}", result_status="completed", result_text="done"
        )
    assert [row.task_id for row in context.snapshot().recent_task_results] == [
        f"task_{n}" for n in range(1, 9)
    ]
