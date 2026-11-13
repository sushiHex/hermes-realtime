"""One dispatch exception ends the host; an ordinary failure does not (#531).

A background dispatch that raises latches the task controller. The update director takes
that error from the controller and latches, and the update executor takes it from the
director and latches. The executor is shared by every binding, so nothing short of a new
process converses again. Until #531 that left a live host that could not answer: these
tests first pinned that as a characterization (#530).

Now each of the three holders reports its latch to the host's one ``HostFatalExit``, which
stops input, lets the saved conversation reach the disk, prints one line with a fixed cause
and exits with a distinct status (proposed ADR 0008, "a latch ends its holder", step 4).

The second half is the guard against escalating too widely: a refused model reply, a failed
turn, a failed chunk, a rejected dispatch, a failed run and an orderly close are all
recoverable or deliberate, and none of them may end the host.

Production wiring with in-memory providers and a real tail writer; the process exit is a
recorder. The transport and the browser are not driven.
"""

import asyncio
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationTaskCommandRouter,
    ConversationTaskController,
    ConversationUpdateDirector,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.streaming import ModelOutputRefused
from hermes_realtime.conversation.work_tools import (
    ConversationWorkControlSurface,
    WorkControlHealth,
)
from hermes_realtime.conversation.worker import (
    ConversationSessionWorker,
    ReconnectSafeConversationWorker,
)
from hermes_realtime.host_launcher import (
    HOST_FATAL_EXIT_STATUS,
    HostFatalExit,
    ProactiveCompletionUpdatePolicy,
)
from hermes_realtime.integration.voice_tail import VoiceTailWriter, parse_voice_tail
from hermes_realtime.protocol import (
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
)
from hermes_realtime.speech import DeliveredSpeechLedger, SpeechChunk, Transcript
from tests.conversation.test_actions import Playback, RecordingInference, Synthesizer
from tests.conversation.test_tasks import QueuedUpdateSession
from tests.conversation.test_worker import FinalOnPushTranscriber, SilenceVad

_DISPATCH_ERROR = "Hermes API dispatch outcome is unknown"
_USER = "browser_user"
_REPLY = "I have the report."
_TASK_COMMAND = "task: Summarize the synthetic report."


class _FailFirstDispatchSession(QueuedUpdateSession):
    """Raise on the first dispatch only; every later dispatch would be accepted."""

    def __init__(self) -> None:
        super().__init__()
        self.dispatch_calls = 0
        self.close_calls = 0

    async def dispatch(
        self, request: WorkDispatchRequestedEvent
    ) -> WorkDispatchAcknowledgedEvent:
        self.dispatch_calls += 1
        if self.dispatch_calls == 1:
            raise RuntimeError(_DISPATCH_ERROR)
        return await super().dispatch(request)

    async def close(self) -> None:
        self.close_calls += 1


class _RejectingSession(QueuedUpdateSession):
    async def dispatch(
        self, request: WorkDispatchRequestedEvent
    ) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged", event_id="evt_ack_rejected",
            session_id=request.session_id, sequence=request.sequence + 1,
            timestamp=request.timestamp, task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=False, reason="Hermes API rejected dispatch"
            ),
        )


class _ScriptedInference(RecordingInference):
    """Raise the queued errors on successive turns, then answer normally."""

    def __init__(self, *errors: BaseException) -> None:
        super().__init__()
        self.errors = list(errors)

    async def _stream(
        self, snapshot: ConversationContextSnapshot, *, turn_id: str
    ) -> AsyncIterator[str]:
        if self.errors:
            raise self.errors.pop(0)
        async for part in super()._stream(snapshot, turn_id=turn_id):
            yield part


class _HeardPlayback(Playback):
    def __init__(self, *, fail_first: bool = False) -> None:
        self.heard: list[str] = []
        self.fail_next = fail_first

    async def play(self, chunk: SpeechChunk, *, is_valid: Any) -> None:
        await super().play(chunk, is_valid=is_valid)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected transport failure")
        self.heard.append(chunk.text)


@dataclass
class _Exit:
    """Stand in for the process exit, and capture what was true at that instant."""

    capsys: pytest.CaptureFixture[str]
    tail_path: Path
    steps: list[str]
    statuses: list[int] = field(default_factory=list)
    stdout: list[str] = field(default_factory=list)
    tail_rows: list[list[tuple[str, str]]] = field(default_factory=list)

    def __call__(self, status: int) -> None:
        self.steps.append("exit")
        self.statuses.append(status)
        self.stdout.append(self.capsys.readouterr().out)
        self.tail_rows.append(_tail_rows(self.tail_path))


def _tail_rows(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        # Nothing was ever said, so the writer had nothing to save.
        return []
    tail = parse_voice_tail(path.read_bytes(), max_messages=32, max_item_chars=1024)
    assert tail is not None
    return [(message.role, message.text) for message in tail.conversation.messages]


@dataclass
class _Stack:
    worker: ReconnectSafeConversationWorker
    controller: ConversationTaskController
    surface: ConversationWorkControlSurface
    director: ConversationUpdateDirector
    executor: ConversationUpdateExecutor
    session: Any
    inference: RecordingInference
    playback: _HeardPlayback
    events: list[tuple[str, dict[str, object]]]
    sequence: "itertools.count[int]"
    host_fatal: HostFatalExit
    exit: _Exit
    steps: list[str]
    tail_path: Path

    async def say(self, generation: int, text: str) -> None:
        await self.worker.submit_final_transcript(
            participant_identity=_USER, session_generation=generation,
            typed_sequence=next(self.sequence), text=text,
        )
        binding = self.worker._binding
        assert binding is not None
        await binding.wait_for_responses()

    async def exited(self) -> None:
        await _until(
            lambda: self.host_fatal._operation is not None, "a report of a host-wide fault"
        )
        operation = self.host_fatal._operation
        assert operation is not None
        await asyncio.wait_for(operation, timeout=5)

    def assert_host_still_running(self) -> None:
        assert self.host_fatal.cause is None
        assert self.host_fatal._operation is None
        assert self.exit.statuses == []
        assert "[host-fatal-exit]" not in self.exit.capsys.readouterr().out


@asynccontextmanager
async def _stack(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    session: Any = None,
    inference: RecordingInference | None = None,
    playback: _HeardPlayback | None = None,
    session_id: str = "session_001",
) -> AsyncIterator[_Stack]:
    tail_path = tmp_path / "voice-tail-v1.json"
    writer = VoiceTailWriter(tail_path)
    context = ConversationContextStore(on_change=writer.update)
    await writer.open(context)
    session = _FailFirstDispatchSession() if session is None else session
    inference = RecordingInference() if inference is None else inference
    playback = _HeardPlayback() if playback is None else playback
    steps: list[str] = []
    exit_process = _Exit(capsys, tail_path, steps)

    async def stop_input() -> None:
        steps.append("input")
        await worker.close_binding()

    async def flush_conversation() -> None:
        steps.append("flush")
        await writer.close()

    # As the host composes it: one owner, and each host-lifetime holder reports to it.
    host_fatal = HostFatalExit(
        stop_input=stop_input, flush_conversation=flush_conversation, exit_process=exit_process,
    )
    counter = itertools.count(1)
    controller = ConversationTaskController(
        context=context, session=session, session_id=session_id,
        id_factory=lambda: f"{next(counter):04d}",
        on_latched=host_fatal.reporter("task_control"),
    )
    events: list[tuple[str, dict[str, object]]] = []

    def observe(kind: str, data: dict[str, object]) -> None:
        events.append((kind, dict(data)))

    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=observe,
        reserve_observer_capacity=lambda: None,
    )
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
        on_latched=host_fatal.reporter("update_director"),
    )
    executor = ConversationUpdateExecutor(
        director=director, speech=speech, on_latched=host_fatal.reporter("update_executor"),
    )

    def binding_factory(
        identity: str, generation: int, actions: ConversationUpdateExecutor
    ) -> ConversationSessionWorker:
        # As the host does: a fresh worker and detector around the one shared executor.
        return ConversationSessionWorker(
            participant_identity=identity, session_generation=generation, vad=SilenceVad(),
            stt=FinalOnPushTranscriber(), actions=actions, command_router=router,
        )

    worker = ReconnectSafeConversationWorker(actions=executor, binding_factory=binding_factory)
    host_fatal.arm()
    try:
        yield _Stack(
            worker, controller, surface, director, executor, session, inference, playback,
            events, itertools.count(1), host_fatal, exit_process, steps, tail_path,
        )
    finally:
        host_fatal.disarm()
        await asyncio.gather(
            worker.close(), surface.close(), director.close(), controller.close(),
            writer.close(), return_exceptions=True,
        )


async def _until(predicate: Any, what: str) -> None:
    # Loop turns first, for the in-memory stacks; then real time, for a session doing I/O.
    for attempt in range(1500):
        if predicate():
            return
        await asyncio.sleep(0 if attempt < 500 else 0.005)
    raise AssertionError(f"timed out waiting for {what}")


async def _rebind_after_failure(stack: _Stack) -> int:
    """Rebind as the host does once a binding has failed.

    The transport settles its failed receiver by closing the binding, which re-raises the
    binding's own latched error once (the transport logs it). The next bind then starts a
    fresh binding around the same shared objects.
    """

    with pytest.raises((RuntimeError, ModelOutputRefused)):
        await stack.worker.close_binding()
    return await stack.worker.bind(_USER)


async def _healthy_turn(stack: _Stack) -> int:
    generation = await stack.worker.bind(_USER)
    await stack.say(generation, "Hello there.")
    assert stack.playback.heard == [_REPLY]
    return generation


# --- a host-wide latch ends the host -----------------------------------------------------


@pytest.mark.asyncio
async def test_one_dispatch_exception_ends_the_host_once_with_the_conversation_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack(tmp_path, capsys) as stack:
        generation = await _healthy_turn(stack)
        capsys.readouterr()

        # The command settles off the input path and does not raise to its caller.
        await stack.say(generation, _TASK_COMMAND)
        await stack.exited()

        # Once, with the distinct status, in the order: input, saved conversation, exit.
        assert stack.exit.statuses == [HOST_FATAL_EXIT_STATUS]
        assert stack.steps == ["input", "flush", "exit"]
        assert stack.host_fatal.cause == "task_control"
        # The same error object still travels the chain; all three holders reported, and
        # the first report decided the cause.
        latched = stack.controller._terminal_error
        assert type(latched) is RuntimeError and str(latched) == _DISPATCH_ERROR
        await _until(lambda: stack.executor._terminal_error is not None, "the executor latch")
        assert stack.director._error is latched and stack.executor._terminal_error is latched
        assert stack.exit.statuses == [HOST_FATAL_EXIT_STATUS]

        # Exactly one exit line, already printed when the exit came. It carries the fixed
        # cause and outcomes and nothing from the error.
        (printed,) = stack.exit.stdout
        lines = [line for line in printed.splitlines() if line.startswith("[host-fatal-exit]")]
        assert lines == [
            '[host-fatal-exit] {"cause":"task_control","flush":"done","input":"done",'
            '"version":1}'
        ]
        assert _DISPATCH_ERROR not in printed and "RuntimeError" not in printed

        # Everything delivered before the fault was on disk when the exit came.
        assert stack.exit.tail_rows == [[
            ("user", "Hello there."),
            ("assistant", _REPLY),
            ("user", _TASK_COMMAND),
        ]]

        # Nothing closed the work session: its close is what stops runs and empties the
        # run record, and the restart must still find them.
        assert stack.session.close_calls == 0
        assert stack.session.dispatch_calls == 1
        assert stack.surface._health is WorkControlHealth.UNCERTAIN


@pytest.mark.asyncio
async def test_after_the_exit_decision_no_further_input_is_admitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack(tmp_path, capsys) as stack:
        generation = await _healthy_turn(stack)
        await stack.say(generation, _TASK_COMMAND)
        await stack.exited()

        assert stack.worker.active_generation is None
        with pytest.raises(RuntimeError, match="stale session generation"):
            await stack.say(generation, "Are you still there?")
        assert len(stack.inference.requests) == 1
        assert stack.playback.heard == [_REPLY]
        # The saved conversation did not move after it was flushed.
        assert _tail_rows(stack.tail_path) == stack.exit.tail_rows[0]


@pytest.mark.asyncio
async def test_a_director_that_latches_alone_ends_the_host_under_its_own_cause(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _stack(tmp_path, capsys, session=QueuedUpdateSession()) as stack:
        async def failing_completion() -> object:
            raise RuntimeError("synthetic director failure")

        monkeypatch.setattr(stack.controller, "next_completion", failing_completion)
        await stack.worker.bind(_USER)
        await stack.exited()

        assert stack.exit.statuses == [HOST_FATAL_EXIT_STATUS]
        assert stack.host_fatal.cause == "update_director"
        assert stack.controller._terminal_error is None
        assert '"cause":"update_director"' in stack.exit.stdout[0]
        assert "synthetic director failure" not in stack.exit.stdout[0]


@pytest.mark.asyncio
async def test_an_executor_that_latches_alone_ends_the_host_under_its_own_cause(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _stack(tmp_path, capsys, session=QueuedUpdateSession()) as stack:
        async def failing_execution(_owner: object) -> object:
            raise RuntimeError("synthetic executor failure")

        monkeypatch.setattr(stack.director, "_next_execution", failing_execution)
        await stack.worker.bind(_USER)
        await stack.exited()

        assert stack.exit.statuses == [HOST_FATAL_EXIT_STATUS]
        assert stack.host_fatal.cause == "update_executor"
        assert stack.controller._terminal_error is None and stack.director._error is None
        assert '"cause":"update_executor"' in stack.exit.stdout[0]


# --- what must not end the host ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_model_reply_fails_only_its_turn_and_the_host_runs_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    inference = _ScriptedInference(ModelOutputRefused("call_imitation"))
    async with _stack(
        tmp_path, capsys, session=QueuedUpdateSession(), inference=inference
    ) as stack:
        generation = await stack.worker.bind(_USER)
        await stack.say(generation, "Hello there.")
        assert stack.playback.heard == []
        stack.assert_host_still_running()

        await stack.say(generation, "Are you still there?")
        assert stack.playback.heard == [_REPLY]
        stack.assert_host_still_running()


@pytest.mark.asyncio
async def test_three_refusals_running_end_the_binding_and_not_the_host(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    inference = _ScriptedInference(*(ModelOutputRefused("decision_json") for _ in range(3)))
    async with _stack(
        tmp_path, capsys, session=QueuedUpdateSession(), inference=inference
    ) as stack:
        generation = await stack.worker.bind(_USER)
        await stack.say(generation, "One.")
        await stack.say(generation, "Two.")
        with pytest.raises(ModelOutputRefused):
            await stack.say(generation, "Three.")
        binding = stack.worker._binding
        assert binding is not None and type(binding._terminal_error) is ModelOutputRefused
        stack.assert_host_still_running()

        # A rebind recovers, which is what makes this a binding failure and not a host one.
        generation = await _rebind_after_failure(stack)
        await stack.say(generation, "Are you still there?")
        assert stack.playback.heard == [_REPLY]
        stack.assert_host_still_running()


@pytest.mark.asyncio
async def test_one_failed_turn_ends_the_binding_and_not_the_host(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    inference = _ScriptedInference(RuntimeError("synthetic model transport failure"))
    async with _stack(
        tmp_path, capsys, session=QueuedUpdateSession(), inference=inference
    ) as stack:
        generation = await stack.worker.bind(_USER)
        with pytest.raises(RuntimeError, match="synthetic model transport failure"):
            await stack.say(generation, "Hello there.")
        stack.assert_host_still_running()

        generation = await _rebind_after_failure(stack)
        await stack.say(generation, "Are you still there?")
        assert stack.playback.heard == [_REPLY]
        stack.assert_host_still_running()


@pytest.mark.asyncio
async def test_one_failed_speech_chunk_ends_the_binding_and_not_the_host(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    playback = _HeardPlayback(fail_first=True)
    async with _stack(
        tmp_path, capsys, session=QueuedUpdateSession(), playback=playback
    ) as stack:
        generation = await stack.worker.bind(_USER)
        with pytest.raises(RuntimeError, match="injected transport failure"):
            await stack.say(generation, "Hello there.")
        stack.assert_host_still_running()

        generation = await _rebind_after_failure(stack)
        await stack.say(generation, "Are you still there?")
        assert stack.playback.heard == [_REPLY]
        stack.assert_host_still_running()


@pytest.mark.asyncio
async def test_a_rejected_dispatch_fails_only_that_task_and_the_host_runs_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack(tmp_path, capsys, session=_RejectingSession()) as stack:
        generation = await _healthy_turn(stack)
        await stack.say(generation, _TASK_COMMAND)
        await _until(lambda: bool(stack.events), "the rejection")
        assert stack.controller._terminal_error is None
        stack.assert_host_still_running()

        await stack.say(generation, "Are you still there?")
        assert stack.playback.heard == [_REPLY, _REPLY]
        stack.assert_host_still_running()


@pytest.mark.asyncio
async def test_a_failed_run_is_announced_and_the_host_runs_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    session = QueuedUpdateSession()
    async with _stack(tmp_path, capsys, session=session) as stack:
        generation = await _healthy_turn(stack)
        await stack.say(generation, _TASK_COMMAND)
        await _until(
            lambda: bool(stack.controller._active_run_ids), "the accepted task"
        )
        (task_id,) = stack.controller._active_run_ids
        await session.updates.put(WorkCompletedEvent(
            type="work.completed", event_id="evt_terminal_failed", session_id="session_001",
            sequence=100, timestamp=datetime(2026, 11, 13, tzinfo=UTC), task_id=task_id,
            run_id="deleg_private_001",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.FAILED, reason="the synthetic run failed"
            ),
        ))
        await _until(lambda: len(stack.playback.heard) > 1, "the failure announcement")
        assert stack.playback.heard[1].startswith("I ran into a problem with that work.")
        assert stack.controller._terminal_error is None
        stack.assert_host_still_running()


@pytest.mark.asyncio
async def test_an_orderly_close_of_every_holder_does_not_end_the_host(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    async with _stack(tmp_path, capsys, session=QueuedUpdateSession()) as stack:
        await _healthy_turn(stack)
        # Still armed: closing must not be mistaken for a failure by the holders themselves.
        await stack.worker.close()
        await stack.surface.close()
        await stack.director.close()
        await stack.controller.close()
        for _ in range(50):
            await asyncio.sleep(0)

        # Closing latches the controller so later calls are refused; that is not a fault.
        assert stack.controller._terminal_error is not None
        stack.assert_host_still_running()
