"""Characterization, not endorsement: one dispatch exception ends all conversation.

A background dispatch that raises once (here, the message the Hermes session raises when a
resend goes unanswered) latches the task controller. The update director takes that error
from the controller and latches, and the update executor takes it from the director and
latches. The executor is shared by every binding, so afterwards:

* an ordinary turn with no task involved is refused before any inference, and nothing is
  spoken;
* a rebind closes the old binding and then fails to start the new one, leaving no binding.

The only evidence is the command path's own ``[task-command-settlement]`` line and one
``uncertain`` task state. Nothing says that conversation has ended, and nothing ends the
process. This pins today's behaviour through the production wiring (session worker,
reconnect-safe worker, command router, work surface, task controller, proactive update
policy, update director, update executor, speech loop) with in-memory providers. It does not
drive the transport or the browser: that the page then shows the worker as disconnected and
the reconnect as failed is read from source, not asserted here.

Proposed ADR 0008 (docs/adr/0008-failure-domains.md, "a latch ends its holder") changes this
in its migration step 4, where a host-lifetime latch exits the host and a restart recovers.
Step 2 adds an evidence line first. Update this test in those steps; do not delete it.
"""

import asyncio
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
from hermes_realtime.conversation.work_tools import (
    ConversationWorkControlSurface,
    WorkControlHealth,
)
from hermes_realtime.conversation.worker import (
    ConversationSessionWorker,
    ReconnectSafeConversationWorker,
)
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.protocol import WorkDispatchAcknowledgedEvent, WorkDispatchRequestedEvent
from hermes_realtime.speech import DeliveredSpeechLedger, SpeechChunk, Transcript
from tests.conversation.test_actions import Playback, RecordingInference, Synthesizer
from tests.conversation.test_tasks import QueuedUpdateSession
from tests.conversation.test_worker import FinalOnPushTranscriber, SilenceVad

_DISPATCH_ERROR = "Hermes API dispatch outcome is unknown"
_USER = "browser_user"


class _FailFirstDispatchSession(QueuedUpdateSession):
    """Raise on the first dispatch only; every later dispatch would be accepted."""

    def __init__(self) -> None:
        super().__init__()
        self.dispatch_calls = 0

    async def dispatch(
        self, request: WorkDispatchRequestedEvent
    ) -> WorkDispatchAcknowledgedEvent:
        self.dispatch_calls += 1
        if self.dispatch_calls == 1:
            raise RuntimeError(_DISPATCH_ERROR)
        return await super().dispatch(request)


class _HeardPlayback(Playback):
    def __init__(self) -> None:
        self.heard: list[str] = []

    async def play(self, chunk: SpeechChunk, *, is_valid: Any) -> None:
        await super().play(chunk, is_valid=is_valid)
        self.heard.append(chunk.text)


@dataclass
class _Stack:
    worker: ReconnectSafeConversationWorker
    controller: ConversationTaskController
    surface: ConversationWorkControlSurface
    director: ConversationUpdateDirector
    executor: ConversationUpdateExecutor
    session: _FailFirstDispatchSession
    inference: RecordingInference
    playback: _HeardPlayback
    events: list[tuple[str, dict[str, object]]]
    sequence: "itertools.count[int]"

    async def say(self, generation: int, text: str) -> None:
        await self.worker.submit_final_transcript(
            participant_identity=_USER, session_generation=generation,
            typed_sequence=next(self.sequence), text=text,
        )
        binding = self.worker._binding
        assert binding is not None
        await binding.wait_for_responses()


@asynccontextmanager
async def _stack() -> AsyncIterator[_Stack]:
    context = ConversationContextStore()
    session = _FailFirstDispatchSession()
    counter = itertools.count(1)
    controller = ConversationTaskController(
        context=context, session=session, session_id="session_001",
        id_factory=lambda: f"{next(counter):04d}",
    )
    events: list[tuple[str, dict[str, object]]] = []

    def observe(kind: str, data: dict[str, object]) -> None:
        events.append((kind, dict(data)))

    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=observe,
        reserve_observer_capacity=lambda: None,
    )
    inference = RecordingInference()
    playback = _HeardPlayback()
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

    def binding_factory(
        identity: str, generation: int, actions: ConversationUpdateExecutor
    ) -> ConversationSessionWorker:
        # As the host does: a fresh worker and detector around the one shared executor.
        return ConversationSessionWorker(
            participant_identity=identity, session_generation=generation, vad=SilenceVad(),
            stt=FinalOnPushTranscriber(), actions=actions, command_router=router,
        )

    worker = ReconnectSafeConversationWorker(actions=executor, binding_factory=binding_factory)
    try:
        yield _Stack(
            worker, controller, surface, director, executor, session, inference, playback,
            events, itertools.count(1),
        )
    finally:
        # The latched executor re-raises its stored error when closed; that is part of what
        # is characterized, and the remaining owners still get their close.
        await asyncio.gather(
            worker.close(), surface.close(), director.close(), controller.close(),
            return_exceptions=True,
        )


async def _until(predicate: Any, what: str) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


async def _fail_one_dispatch(stack: _Stack) -> int:
    """Bind, hold one healthy turn, then issue the task command whose dispatch raises."""

    generation = await stack.worker.bind(_USER)
    await stack.say(generation, "Hello there.")
    assert stack.playback.heard == ["I have the report."]
    assert len(stack.inference.requests) == 1
    # The command settles off the input path and does not raise to its caller.
    await stack.say(generation, "task: Summarize the synthetic report.")
    await _until(lambda: stack.executor._terminal_error is not None, "the executor latch")
    await _until(lambda: bool(stack.events), "the command's own settlement")
    assert stack.session.dispatch_calls == 1
    return generation


@pytest.mark.asyncio
async def test_one_dispatch_exception_latches_controller_director_and_executor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack() as stack:
        await _fail_one_dispatch(stack)

        latched = stack.controller._terminal_error
        assert type(latched) is RuntimeError and str(latched) == _DISPATCH_ERROR
        # One error object travels the whole chain; nothing wraps or reclassifies it.
        assert stack.director._error is latched
        assert stack.executor._terminal_error is latched
        assert stack.surface._health is WorkControlHealth.UNCERTAIN
        assert stack.controller.task_status().available is False
        # The binding that issued the command is not itself latched yet.
        binding = stack.worker._binding
        assert binding is not None and binding._terminal_error is None

        # Everything the user or an operator is given: one task state, one marker line.
        assert stack.events == [(
            "task_state",
            {
                "status": "uncertain",
                "taskId": None,
                "reason": (
                    "Task command acknowledgment is uncertain. "
                    "Check task status before trying again."
                ),
            },
        )]
        assert capsys.readouterr().out == (
            '[task-command-settlement] {"kind":"start","category":"acknowledgment-uncertain"}\n'
        )


@pytest.mark.asyncio
async def test_after_one_dispatch_exception_an_ordinary_turn_is_refused_unheard(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack() as stack:
        generation = await _fail_one_dispatch(stack)
        capsys.readouterr()
        events_before = list(stack.events)

        # No task is involved in this turn.
        with pytest.raises(RuntimeError) as caught:
            await stack.say(generation, "What is two plus two?")

        assert caught.value is stack.controller._terminal_error
        # No inference ran and nothing was spoken, shown or printed.
        assert len(stack.inference.requests) == 1
        assert stack.playback.heard == ["I have the report."]
        assert stack.events == events_before
        assert capsys.readouterr().out == ""
        # The turn's failure latched the binding too, so every later input is refused.
        binding = stack.worker._binding
        assert binding is not None and binding._terminal_error is caught.value
        with pytest.raises(RuntimeError) as again:
            await stack.say(generation, "Are you still there?")
        assert again.value is caught.value
        # A later dispatch would have been accepted; none is ever attempted.
        assert stack.session.dispatch_calls == 1


@pytest.mark.asyncio
async def test_after_one_dispatch_exception_a_rebind_fails_and_leaves_no_binding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack() as stack:
        generation = await _fail_one_dispatch(stack)
        capsys.readouterr()
        events_before = list(stack.events)
        first = stack.worker._binding
        assert first is not None

        with pytest.raises(RuntimeError) as caught:
            await stack.worker.bind(_USER)

        assert caught.value is stack.controller._terminal_error
        # The old binding was closed before the new one failed to start.
        assert stack.worker.active_generation is None
        with pytest.raises(RuntimeError, match="conversation session worker is closed"):
            await first.submit_final_transcript(
                participant_identity=_USER, session_generation=generation,
                typed_sequence=next(stack.sequence), text="Are you still there?",
            )
        with pytest.raises(RuntimeError, match="stale session generation"):
            await stack.say(generation, "Are you still there?")
        # Every further rebind fails the same way: nothing but a new process recovers.
        with pytest.raises(RuntimeError) as again:
            await stack.worker.bind(_USER)
        assert again.value is caught.value
        assert len(stack.inference.requests) == 1
        assert stack.playback.heard == ["I have the report."]
        assert stack.events == events_before
        assert capsys.readouterr().out == ""
