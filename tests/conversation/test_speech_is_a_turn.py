"""Commands are typed bytes; speech is always a turn.

A microphone final never reaches the task command grammar, whatever its words. Only a
pending proposal confirmation can consume speech; that path is covered by
``tests/conversation/test_proposal_capture.py``.
"""

from __future__ import annotations

import pytest

from hermes_realtime.conversation import ConversationContextStore, ConversationTaskCommandRouter
from hermes_realtime.conversation.work_tools import WorkCommandAdmission
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.speech import AudioFrame, Transcript, VoiceActivity
from tests.conversation.test_worker import (
    CapturingTextFinalTranscriber,
    ScriptedVad,
    action_executor_probe,
)

_TASK = "task_fixture"
_GRAMMAR_SHAPED = (
    "start task Inspect release evidence",
    "Start task.",
    "cancel task",
    "Cancel task.",
    f"cancel task {_TASK}",
    "task: Inspect release evidence",
    f"cancel task: {_TASK}",
)


class RecordingSurface:
    """Records every command admission; it never settles anything itself."""

    max_objective_chars = 1024
    # Present only so a restored spoken grammar could freeze the sole task.
    active_task_ids = (_TASK,)

    def __init__(self) -> None:
        self.starts: list[str] = []
        self.cancels: list[str] = []

    async def submit_start_command(
        self, *, objective: str, invocation_id: str
    ) -> WorkCommandAdmission:
        del invocation_id
        self.starts.append(objective)
        return WorkCommandAdmission.ADMITTED

    async def submit_cancel_command(
        self,
        *,
        task_id: str,
        invocation_id: str,
        reason: str = "user requested task cancellation",
    ) -> WorkCommandAdmission:
        del invocation_id, reason
        self.cancels.append(task_id)
        return WorkCommandAdmission.ADMITTED


def _worker(
    text: str,
) -> tuple[ConversationSessionWorker, RecordingSurface, ConversationContextStore, list[object]]:
    context = ConversationContextStore()
    context.record_task_accepted(task_id=_TASK, run_id="deleg_fixture", objective="Synthetic")
    surface = RecordingSurface()
    events: list[object] = []
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
    )
    actions, _cancelled, responses = action_executor_probe()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=CapturingTextFinalTranscriber(text),
        stt_streams_partials=False,
        actions=actions,
        command_router=router,
    )
    worker.start()
    return worker, surface, context, [responses, events]


def _active_ids(context: ConversationContextStore) -> tuple[str, ...]:
    return tuple(task.task_id for task in context.snapshot().active_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _GRAMMAR_SHAPED)
async def test_a_microphone_final_shaped_like_a_command_is_an_ordinary_turn(text: str) -> None:
    worker, surface, context, (responses, events) = _worker(text)
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)
    try:
        await worker.receive_audio("browser_user", 1, frame)
        await worker.receive_audio("browser_user", 1, frame)
        await worker.wait_for_responses()
    finally:
        await worker.close()
    assert (surface.starts, surface.cancels) == ([], [])
    assert events == []
    assert responses == [("session_1_turn_1", Transcript(text=text, final=True))]
    assert _active_ids(context) == (_TASK,)


async def _typed(text: str) -> tuple[RecordingSurface, ConversationContextStore, list[object]]:
    worker, surface, context, (responses, _events) = _worker(text)
    try:
        await worker.submit_final_transcript(
            participant_identity="browser_user",
            session_generation=1,
            typed_sequence=1,
            text=text,
        )
        await worker.wait_for_responses()
    finally:
        await worker.close()
    return surface, context, responses


@pytest.mark.asyncio
async def test_typed_task_colon_form_still_starts_work() -> None:
    surface, _context, responses = await _typed("task: Inspect release evidence")
    assert (surface.starts, surface.cancels, responses) == (
        ["Inspect release evidence"],
        [],
        [],
    )


@pytest.mark.asyncio
async def test_typed_cancel_task_colon_form_still_cancels_the_named_task() -> None:
    surface, _context, responses = await _typed(f"cancel task: {_TASK}")
    assert (surface.starts, surface.cancels, responses) == ([], [_TASK], [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ("start task Inspect release evidence", "Start task.", "cancel task", f"cancel task {_TASK}"),
)
async def test_removed_spoken_forms_typed_are_ordinary_turns(text: str) -> None:
    surface, context, responses = await _typed(text)
    assert (surface.starts, surface.cancels) == ([], [])
    assert responses == [("session_1_turn_1", Transcript(text=text, final=True))]
    assert _active_ids(context) == (_TASK,)
