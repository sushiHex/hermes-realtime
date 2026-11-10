"""#225: a typed cancel that lands while nothing is playing keeps the cancelled task's note.

The path is the production one: typed input through the session worker, the explicit
``cancel task:`` command through the shared work surface, the real task controller and
its terminal update, the host's proactive update policy and its floor check, the update
executor, and finally the Ollama adapter's assembled prompt for the next user turn.
"""

import asyncio
import itertools
import json
from datetime import UTC, datetime

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationTaskCommandRouter,
    ConversationTaskController,
    ConversationUpdateDirector,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.protocol import WorkCompletedEvent, WorkCompletedPayload, WorkTerminalStatus
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_actions import Playback, RecordingInference, Synthesizer
from tests.conversation.test_tasks import QueuedUpdateSession
from tests.conversation.test_worker import FinalOnPushTranscriber, SilenceVad

_REASON = "user requested task cancellation"


async def _until(predicate, what: str) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
async def test_idle_typed_cancel_keeps_cancelled_task_note_in_next_prompt() -> None:
    from hermes_realtime.providers import OllamaStreamingInference

    context = ConversationContextStore()
    session = QueuedUpdateSession()
    counter = itertools.count(1)
    controller = ConversationTaskController(
        context=context, session=session, session_id="session_001",
        id_factory=lambda: f"{next(counter):04d}",
    )
    states: list[dict[str, object]] = []

    def observe(kind: str, data: dict[str, object]) -> None:
        if kind == "task_state":
            states.append(dict(data))

    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=observe,
        reserve_observer_capacity=lambda: None,
    )
    inference = RecordingInference()
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=Synthesizer(), playback=Playback(), ledger=DeliveredSpeechLedger(),
    )

    async def record_command_input(text: str) -> None:
        await speech.record_user_input(Transcript(text=text, final=True))

    router = ConversationTaskCommandRouter(
        surface=surface, observer=observe, reserve_observer_capacity=lambda: None,
        validate_user_input=context.validate_user_text, record_user_input=record_command_input,
    )
    # The host's own floor check (host_launcher.completion_floor_available); typed input
    # never sets the host's user_speaking flag.
    director = ConversationUpdateDirector(
        context=context, completions=controller,
        policy=ProactiveCompletionUpdatePolicy(
            projection=BrowserEventProjection(),
            floor_available=lambda: not speech.foreground_active,
        ),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    worker = ConversationSessionWorker(
        participant_identity="browser_user", session_generation=7, vad=SilenceVad(),
        stt=FinalOnPushTranscriber(), actions=executor, command_router=router,
    )
    worker.start()
    try:
        await worker.submit_final_transcript(
            participant_identity="browser_user", session_generation=7, typed_sequence=1,
            text="task: Keep working on the synthetic cancellation task.",
        )
        await _until(lambda: bool(context.snapshot().active_tasks), "task_active")
        (task,) = context.snapshot().active_tasks

        # Nothing is playing when the cancel lands: no speech for cancel_foreground to stop.
        assert not executor.foreground_active
        await worker.submit_final_transcript(
            participant_identity="browser_user", session_generation=7, typed_sequence=2,
            text=f"cancel task: {task.task_id}",
        )
        await _until(
            lambda: {"status": "cancelling", "taskId": task.task_id} in states, "cancelling"
        )
        await session.updates.put(WorkCompletedEvent(
            type="work.completed", event_id="evt_terminal_cancel", session_id="session_001",
            sequence=100, timestamp=datetime(2026, 11, 10, tzinfo=UTC), task_id=task.task_id,
            run_id="deleg_private_001",
            payload=WorkCompletedPayload(status=WorkTerminalStatus.INTERRUPTED, reason=_REASON),
        ))
        record = await asyncio.wait_for(executor.wait_for_action_record(1), timeout=2)
        assert inference.requests == []

        await worker.submit_final_transcript(
            participant_identity="browser_user", session_generation=7, typed_sequence=3,
            text="What happened to that task?",
        )
        await worker.wait_for_responses()

        assert len(inference.requests) == 1
        prompt = OllamaStreamingInference._messages(inference.requests[0])
        # Idle floor: the policy spoke the terminal at once rather than deferring it to
        # this turn, so the note reaches the prompt as a reference and a heard row,
        # never as a pending update.
        assert (record.decision.kind, record.disposition) == ("interrupt", "delivered")
        assert OllamaStreamingInference.prompt_metadata(prompt)["reference_kinds"] == [
            "recent_task_results",
        ]
        _, _, references = prompt[0]["content"].partition("\n\nReference data:\n")
        (result,) = json.loads(references)["recent_task_results"]
        assert (result["task_id"], result["status"]) == (task.task_id, "interrupted")
        assert [(row["role"], row["content"]) for row in prompt[1:]] == [
            ("user", "task: Keep working on the synthetic cancellation task."),
            ("user", f"cancel task: {task.task_id}"),
            ("assistant", record.decision.text),
            ("user", "What happened to that task?"),
        ]
    finally:
        await worker.close()
        await surface.close()
        await director.close()
        await controller.close()
