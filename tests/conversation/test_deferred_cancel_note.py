"""#225: the deferred paths of a typed cancel keep the cancelled task's note.

``test_idle_cancel_note`` pins the path where nothing is playing and the terminal update is
spoken at once. These tests pin the two paths where the terminal lands while speech owns the
floor, through the same production wiring (session worker, work surface, task controller,
proactive update policy with the host's floor check, update executor, Ollama prompt):

* the update is deferred as ``mention_next`` and reaches the next prompt as ``updates``;
* an immediate interrupt announcement is superseded before it is heard, is settled without
  being requeued, and the note survives only as ``recent_task_results``.
"""

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
from hermes_realtime.conversation.turn import ForegroundTurn
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.protocol import WorkCompletedEvent, WorkCompletedPayload, WorkTerminalStatus
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_actions import Playback, RecordingInference, Synthesizer
from tests.conversation.test_idle_cancel_note import _REASON, _until
from tests.conversation.test_tasks import QueuedUpdateSession
from tests.conversation.test_worker import FinalOnPushTranscriber, SilenceVad

_TASK_TEXT = "task: Keep working on the synthetic cancellation task."


class _GatedPlayback(Playback):
    """Hold the next played chunk until released or cancelled; later chunks play at once."""

    def __init__(self) -> None:
        self.armed = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def play(self, chunk: Any, *, is_valid: Any) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        if not self.armed:
            return
        self.armed = False
        self.started.set()
        await self.release.wait()
        if not is_valid():
            raise asyncio.CancelledError

    async def cancel(self, turn_id: str) -> None:
        del turn_id
        self.release.set()


@dataclass
class _Stack:
    worker: ConversationSessionWorker
    executor: ConversationUpdateExecutor
    director: ConversationUpdateDirector
    context: ConversationContextStore
    session: QueuedUpdateSession
    inference: RecordingInference
    playback: _GatedPlayback
    states: list[dict[str, object]]
    sequence: itertools.count

    async def say(self, text: str) -> None:
        await self.worker.submit_final_transcript(
            participant_identity="browser_user", session_generation=7,
            typed_sequence=next(self.sequence), text=text,
        )

    async def cancel_started_task(self) -> str:
        await self.say(_TASK_TEXT)
        await _until(lambda: bool(self.context.snapshot().active_tasks), "task_active")
        (task,) = self.context.snapshot().active_tasks
        await self.say(f"cancel task: {task.task_id}")
        await _until(
            lambda: {"status": "cancelling", "taskId": task.task_id} in self.states, "cancelling"
        )
        return task.task_id

    async def terminal_interrupted(self, task_id: str) -> None:
        await self.session.updates.put(WorkCompletedEvent(
            type="work.completed", event_id="evt_terminal_cancel", session_id="session_001",
            sequence=100, timestamp=datetime(2026, 11, 10, tzinfo=UTC), task_id=task_id,
            run_id="deleg_private_001",
            payload=WorkCompletedPayload(status=WorkTerminalStatus.INTERRUPTED, reason=_REASON),
        ))


@asynccontextmanager
async def _stack() -> AsyncIterator[_Stack]:
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
    playback = _GatedPlayback()
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=Synthesizer(), playback=playback, ledger=DeliveredSpeechLedger(),
    )

    async def record_command_input(text: str) -> None:
        await speech.record_user_input(Transcript(text=text, final=True))

    router = ConversationTaskCommandRouter(
        surface=surface, observer=observe, reserve_observer_capacity=lambda: None,
        validate_user_input=context.validate_user_text, record_user_input=record_command_input,
    )
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
        yield _Stack(
            worker, executor, director, context, session, inference, playback, states,
            itertools.count(1),
        )
    finally:
        playback.release.set()
        await worker.close()
        await surface.close()
        await director.close()
        await controller.close()


def _prompt(stack: _Stack, index: int) -> list[dict[str, str]]:
    from hermes_realtime.providers import OllamaStreamingInference

    return OllamaStreamingInference._messages(ForegroundTurn(stack.inference.requests[index]))


def _references(prompt: list[dict[str, str]]) -> dict[str, Any]:
    _, _, references = prompt[0]["content"].partition("\n\nReference data:\n")
    parsed: dict[str, Any] = json.loads(references)
    return parsed


@pytest.mark.asyncio
async def test_cancel_terminal_landing_during_speech_is_deferred_into_the_next_prompt() -> None:
    from hermes_realtime.providers import OllamaStreamingInference

    async with _stack() as stack:
        task_id = await stack.cancel_started_task()

        # Speech owns the floor when the terminal lands: the policy must defer it.
        stack.playback.armed = True
        await stack.say("Tell me something while that finishes.")
        await asyncio.wait_for(stack.playback.started.wait(), timeout=2)
        assert stack.executor.foreground_active
        await stack.terminal_interrupted(task_id)
        await _until(lambda: bool(stack.director.pending_mentions()), "deferred mention")
        (mention,) = stack.director.pending_mentions()
        assert mention.kind == "mention_next"
        assert not any(
            record.decision.kind == "interrupt" for record in stack.executor.action_records()
        )

        stack.playback.release.set()
        await stack.worker.wait_for_responses()
        await stack.say("What happened to that task?")
        await stack.worker.wait_for_responses()

        assert len(stack.inference.requests) == 2
        prompt = _prompt(stack, 1)
        kinds = OllamaStreamingInference.prompt_metadata(prompt)["reference_kinds"]
        assert kinds == ["recent_task_results", "updates"]
        references = _references(prompt)
        (result,) = references["recent_task_results"]
        assert (result["task_id"], result["status"]) == (task_id, "interrupted")
        assert references["updates"]
        assert stack.director.pending_mentions() == ()


@pytest.mark.asyncio
async def test_superseded_cancel_announcement_leaves_the_note_only_as_a_task_result() -> None:
    from hermes_realtime.providers import OllamaStreamingInference

    async with _stack() as stack:
        task_id = await stack.cancel_started_task()

        # Idle floor: the terminal is announced at once, and the announcement is held in
        # playback so it has not been heard when the user speaks over it.
        stack.playback.armed = True
        await stack.terminal_interrupted(task_id)
        await asyncio.wait_for(stack.playback.started.wait(), timeout=2)
        assert stack.inference.requests == []

        await stack.say("What happened to that task?")
        await stack.worker.wait_for_responses()
        await _until(
            lambda: any(
                record.decision.kind == "interrupt" for record in stack.executor.action_records()
            ),
            "settled announcement",
        )

        (record,) = stack.executor.action_records()
        assert (record.decision.kind, record.disposition) == ("interrupt", "superseded")
        # Settled without requeueing: nothing is left to mention on a later turn.
        assert stack.director.pending_mentions() == ()
        assert len(stack.inference.requests) == 1
        prompt = _prompt(stack, 0)
        assert OllamaStreamingInference.prompt_metadata(prompt)["reference_kinds"] == [
            "recent_task_results",
        ]
        (result,) = _references(prompt)["recent_task_results"]
        assert (result["task_id"], result["status"]) == (task_id, "interrupted")
        # The unheard announcement is not a heard assistant row.
        assert [(row["role"], row["content"]) for row in prompt[1:]] == [
            ("user", _TASK_TEXT),
            ("user", f"cancel task: {task_id}"),
            ("user", "What happened to that task?"),
        ]
