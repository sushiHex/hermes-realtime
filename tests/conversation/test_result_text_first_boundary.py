"""Untrusted result text meets the private-handle rule once, where it enters conversation.

A background task's summary or reason is text a model wrote, possibly quoting a web page. The
task controller is the first conversation boundary it crosses. These tests drive the
production wiring (reconnect-safe worker, command router, work surface, task controller,
proactive update policy, update director, update executor, speech loop) with in-memory
providers and a session that delivers terminal events.
"""

import asyncio
import itertools
import json
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
from hermes_realtime.conversation.result_projection import (
    UNUSABLE_RESULT_REASON,
    UNUSABLE_RESULT_SUMMARY,
    admit_result_text,
)
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.conversation.worker import (
    ConversationSessionWorker,
    ReconnectSafeConversationWorker,
)
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelAcknowledgedEvent,
    ControlCancelAcknowledgedPayload,
    ControlCancelEvent,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
)
from hermes_realtime.protocol.identifiers import PRIVATE_HANDLE, PRIVATE_RUN_TOKEN
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_actions import Playback, RecordingInference, Synthesizer
from tests.conversation.test_tasks import TaskSessionStub
from tests.conversation.test_worker import FinalOnPushTranscriber, SilenceVad

MARKER = "[task-result-admission] "
_USER = "browser_user"
_NOW = datetime(2026, 7, 20, 23, 35, tzinfo=UTC)
STATUSES = ("completed", "failed", "interrupted")
HANDLES = {
    "letter before a run token": "xdeleg_a1b2c3",
    "digit before a run token": "1deleg_a1b2c3",
    "bare run token": "deleg_a1b2c3",
    "run handle": "run_0123456789abcdef",
    "subagent handle": "subagent_a1b2c3",
}


class _Session(TaskSessionStub):
    """Accept every dispatch under its own run id and deliver queued terminal events."""

    def __init__(self) -> None:
        self.requests: list[WorkDispatchRequestedEvent] = []
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()
        self.inbound = itertools.count(1000)
        self.dispatch_refusal: str | None = None
        self.cancel_refusal = "cancellation is not available"

    async def dispatch(self, request: WorkDispatchRequestedEvent) -> WorkDispatchAcknowledgedEvent:
        self.requests.append(request)
        refusal = self.dispatch_refusal
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id=f"evt_ack_{len(self.requests)}",
            session_id=request.session_id,
            sequence=next(self.inbound),
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=refusal is None,
                run_id=f"deleg_run{len(self.requests):04d}" if refusal is None else None,
                reason=refusal,
            ),
        )

    async def cancel(self, request: ControlCancelEvent) -> ControlCancelAcknowledgedEvent:
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged",
            event_id=f"evt_cancel_ack_{next(self.inbound)}",
            session_id=request.session_id,
            sequence=next(self.inbound),
            timestamp=request.timestamp,
            request_event_id=request.event_id,
            scope=CancelScope.TASK,
            task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(
                accepted=False, reason=self.cancel_refusal
            ),
        )

    async def next_update(self) -> WorkCompletedEvent:
        return await self.updates.get()


@dataclass
class _Stack:
    worker: ReconnectSafeConversationWorker
    controller: ConversationTaskController
    director: ConversationUpdateDirector
    executor: ConversationUpdateExecutor
    context: ConversationContextStore
    projection: BrowserEventProjection
    session: _Session
    generation: int = 0
    typed: "itertools.count[int]" = field(default_factory=lambda: itertools.count(1))
    terminal: "itertools.count[int]" = field(default_factory=lambda: itertools.count(1))
    seen: int = 0
    published: dict[str, tuple[str, str]] = field(default_factory=dict)
    observed: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    async def say(self, text: str) -> None:
        await self.worker.submit_final_transcript(
            participant_identity=_USER, session_generation=self.generation,
            typed_sequence=next(self.typed), text=text,
        )

    def reasons(self) -> list[str]:
        return [str(data["reason"]) for _, data in self.observed if data.get("reason")]

    @property
    def latched(self) -> list[str]:
        owners = (
            ("controller", self.controller._terminal_error),
            ("director", self.director._error),
            ("executor", self.executor._terminal_error),
        )
        return [f"{name}:{type(error).__name__}" for name, error in owners if error is not None]

    def failure(self) -> str:
        error = (
            self.controller._terminal_error or self.director._error
            or self.executor._terminal_error
        )
        return "" if error is None else f"{type(error).__name__}: {error}"[:200]

    def results(self) -> dict[str, tuple[str, str]]:
        """Read and acknowledge published pages as a browser does; keep every task result."""
        while page := self.projection.events_after(self.seen):
            for event in page:
                if event.kind == "task_result":
                    self.published[str(event.data["taskId"])] = (
                        str(event.data["status"]), str(event.data["text"]),
                    )
            self.seen = page[-1].sequence
            self.projection.acknowledge_through(self.seen)
        return self.published

    async def finish_task(self, status: str, text: str) -> str:
        """Start one task by command and deliver its terminal event; return its task id."""
        started = len(self.session.requests)
        await self.worker.submit_final_transcript(
            participant_identity=_USER, session_generation=self.generation,
            typed_sequence=next(self.typed), text=f"task: Synthetic work {started + 1}.",
        )
        await _until(lambda: len(self.session.requests) > started or bool(self.latched))
        if len(self.session.requests) == started:
            return ""
        request = self.session.requests[-1]
        await _until(lambda: request.task_id in self.controller._active_run_ids)
        completed = status == "completed"
        number = next(self.terminal)
        await self.session.updates.put(
            WorkCompletedEvent(
                type="work.completed", event_id=f"evt_terminal_{number}",
                session_id=request.session_id, sequence=next(self.session.inbound),
                timestamp=_NOW,
                task_id=request.task_id, run_id=f"deleg_run{len(self.session.requests):04d}",
                payload=WorkCompletedPayload(
                    status=WorkTerminalStatus(status),
                    summary=text if completed else None,
                    reason=None if completed else text,
                ),
            )
        )
        await _until(lambda: request.task_id in self.results() or bool(self.latched))
        return request.task_id


async def _until(predicate: object) -> None:
    assert callable(predicate)
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the stack did not settle")


@asynccontextmanager
async def _stack() -> AsyncIterator[_Stack]:
    context = ConversationContextStore()
    session = _Session()
    counter = itertools.count(1)
    controller = ConversationTaskController(
        context=context, session=session, session_id="session_001",
        id_factory=lambda: f"{next(counter):04d}",
    )

    observed: list[tuple[str, dict[str, object]]] = []

    def observe(kind: str, data: dict[str, object]) -> None:
        observed.append((kind, dict(data)))

    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=observe,
        reserve_observer_capacity=lambda: None,
    )
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=RecordingInference(),
        synthesizer=Synthesizer(), playback=Playback(), ledger=DeliveredSpeechLedger(),
    )

    async def record_command_input(text: str) -> None:
        await speech.record_user_input(Transcript(text=text, final=True))

    router = ConversationTaskCommandRouter(
        surface=surface, observer=observe, reserve_observer_capacity=lambda: None,
        validate_user_input=context.validate_user_text, record_user_input=record_command_input,
    )
    projection = BrowserEventProjection()
    director = ConversationUpdateDirector(
        context=context, completions=controller,
        policy=ProactiveCompletionUpdatePolicy(
            projection=projection, floor_available=lambda: not speech.foreground_active,
        ),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)

    def binding_factory(
        identity: str, generation: int, actions: ConversationUpdateExecutor
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=identity, session_generation=generation, vad=SilenceVad(),
            stt=FinalOnPushTranscriber(), actions=actions, command_router=router,
        )

    worker = ReconnectSafeConversationWorker(actions=executor, binding_factory=binding_factory)
    stack = _Stack(worker, controller, director, executor, context, projection, session)
    stack.observed = observed
    try:
        stack.generation = await worker.bind(_USER)
        yield stack
    finally:
        await asyncio.gather(
            worker.close(), surface.close(), director.close(), controller.close(),
            return_exceptions=True,
        )


def _admission_lines(output: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(MARKER))
        for line in output.splitlines()
        if line.startswith(MARKER)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("name", HANDLES)
async def test_a_result_naming_a_private_handle_stays_local_to_its_task(
    capsys: pytest.CaptureFixture[str], status: str, name: str
) -> None:
    handle = HANDLES[name]
    async with _stack() as stack:
        first = await stack.finish_task(status, f"The page said {handle} was done.")
        # The director keeps running: the next completion is announced.
        second = await stack.finish_task(status, "Plain outcome.")

        assert stack.latched == []
        results = stack.results()
        assert results[second] == (status, "Plain outcome.")
        shown_status, shown = results[first]
        assert shown_status == status
        assert shown.startswith("The page said ") and shown.endswith(" was done.")
        assert PRIVATE_RUN_TOKEN.search(shown) is None and PRIVATE_HANDLE.search(shown) is None
        assert "private reference" in shown
        remembered = " ".join(row.text for row in stack.context.snapshot().recent_task_results)
        assert PRIVATE_RUN_TOKEN.search(remembered) is None

    lines = _admission_lines(capsys.readouterr().out)
    expected_field = "summary" if status == "completed" else "reason"
    if "run token" in name:
        assert lines == [
            {"category": "private_token", "field": expected_field, "source": status, "version": 1}
        ]
    else:
        assert lines == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "text", "category", "shown"),
    [
        ("completed", "x" * 4097, "bound", UNUSABLE_RESULT_SUMMARY),
        ("failed", "x" * 1025, "bound", UNUSABLE_RESULT_REASON),
        ("interrupted", "y" * 8192, "bound", UNUSABLE_RESULT_REASON),
        # The replacement is longer than the token, so a text at the bound no longer fits.
        ("failed", "x" * 1016 + " deleg_a", "bound", UNUSABLE_RESULT_REASON),
    ],
    ids=["summary-bound", "reason-bound", "reason-wire-bound", "scrub-overflow"],
)
async def test_unusable_result_text_becomes_a_fixed_notice_and_stays_local(
    capsys: pytest.CaptureFixture[str], status: str, text: str, category: str, shown: str
) -> None:
    async with _stack() as stack:
        first = await stack.finish_task(status, text)
        second = await stack.finish_task(status, "Plain outcome.")

        assert stack.latched == []
        assert stack.results()[first] == (status, shown)
        assert stack.results()[second] == (status, "Plain outcome.")

    output = capsys.readouterr().out
    field_name = "summary" if status == "completed" else "reason"
    assert _admission_lines(output) == [
        {"category": category, "field": field_name, "source": status, "version": 1}
    ]
    assert "xxxx" not in output and "yyyy" not in output and "deleg_a" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("status", STATUSES)
async def test_a_token_assembled_by_rendering_a_link_stays_local(status: str) -> None:
    """The shipped API adapter admits this text: it holds no token until the host name of the
    link is rendered in lower case, directly after a word."""
    text = "Fetch failed for sitehttps://DELEG_a1b2/report today."
    assert PRIVATE_RUN_TOKEN.search(text) is None and "deleg_" not in text
    async with _stack() as stack:
        first = await stack.finish_task(status, text)
        second = await stack.finish_task(status, "Plain outcome.")

        assert stack.latched == []
        assert first in stack.results()
        assert stack.results()[second] == (status, "Plain outcome.")
        remembered = " ".join(row.text for row in stack.context.snapshot().recent_task_results)
        assert PRIVATE_RUN_TOKEN.search(remembered) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", STATUSES)
async def test_no_admitted_result_text_makes_a_later_stage_raise(status: str) -> None:
    """Generated texts, including ones whose rendering would assemble a token downstream."""
    generator = random.Random(531)
    pieces = (
        "deleg", "_", "a", "x", "1", " ", ".", ":", "-", "deleg_", "deleg_a1", "xdeleg_b2",
        "task_", "run_", "run_0123456789abcdef", "subagent_z", "[", "](", ")",
        "https://example.com/a", "[deleg](https://example.com/a)", "\u2022", "`", "*",
        "<voice>", "</voice>", "<detail>", "</detail>", "\n\nSources:\n", "\n",
        "\U0001F600", "\x00", "\u202e", "Summary:", "Details:", "https://DELEG_a1/x", "DELEG_",
    )  # A lone surrogate cannot be generated: the wire type refuses it before the controller.
    texts = [
        # Rendering a link as its host name lowers its case: no token here, one after.
        "See https://DELEG_a1b2/report for details.",
        "Read [the page](https://DELEG_a1b2/report) today.",
        "x[deleg](https://example.com/a)_abc done",
        "[deleg](https://example.com/a)_abc done",
        "<voice>deleg</voice><detail>_abc</detail>",
        "deleg\n\nSources:\n_abc",
        "deleg\u2022_abc and deleg` `_abc",
    ]
    for _ in range(395):
        text = "".join(generator.choice(pieces) for _ in range(generator.randint(1, 40)))
        texts.append(text if text.strip() else "x")

    # Nothing here drains queued announcements, and the director holds at most 32 of them.
    for start in range(0, len(texts), 25):
        async with _stack() as stack:
            for text in texts[start : start + 25]:
                task_id = await stack.finish_task(status, text)
                assert stack.latched == [], (text.encode("unicode_escape"), stack.failure())
                shown = stack.results()[task_id][1]
                assert PRIVATE_RUN_TOKEN.search(shown) is None
            last = await stack.finish_task(status, "Plain outcome.")
            assert stack.latched == []
            assert stack.results()[last] == (status, "Plain outcome.")
            remembered = " ".join(
                row.text for row in stack.context.snapshot().recent_task_results
            )
            assert PRIVATE_RUN_TOKEN.search(remembered) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HANDLES)
async def test_a_rejection_reason_naming_a_private_handle_stays_local(
    capsys: pytest.CaptureFixture[str], name: str
) -> None:
    handle = HANDLES[name]
    async with _stack() as stack:
        stack.session.dispatch_refusal = f"The queue named {handle} is full."
        await stack.say("task: Synthetic refused work.")
        await _until(lambda: bool(stack.reasons()) or bool(stack.latched))

        assert stack.latched == []
        (shown,) = stack.reasons()
        assert shown.startswith("The queue named ") and shown.endswith(" is full.")
        assert PRIVATE_RUN_TOKEN.search(shown) is None
        assert ("private reference" in shown) == ("run token" in name)

        # Dispatch still works, a refused cancellation is local too, and the task finishes.
        stack.session.dispatch_refusal = None
        stack.session.cancel_refusal = f"The run {handle} cannot be stopped."
        before = len(stack.session.requests)
        await stack.say("task: Synthetic accepted work.")
        await _until(lambda: len(stack.session.requests) > before or bool(stack.latched))
        task_id = stack.session.requests[-1].task_id
        await _until(lambda: task_id in stack.controller._active_run_ids)
        await stack.say(f"cancel task: {task_id}")
        await _until(lambda: len(stack.reasons()) > 1 or bool(stack.latched))
        assert stack.latched == []
        refused = stack.reasons()[-1]
        assert refused.startswith("The run ") and refused.endswith(" cannot be stopped.")
        assert PRIVATE_RUN_TOKEN.search(refused) is None
        last = await stack.finish_task("completed", "Plain outcome.")
        assert stack.latched == []
        assert stack.results()[last] == ("completed", "Plain outcome.")

    lines = _admission_lines(capsys.readouterr().out)
    if "run token" in name:
        assert lines == [
            {"category": "private_token", "field": "reason", "source": source, "version": 1}
            for source in ("dispatch_rejected", "cancel_rejected")
        ]
    else:
        assert lines == []


@pytest.mark.parametrize(
    ("text", "limit", "admitted", "category"),
    [
        ("plain", 1024, "plain", None),
        ("see deleg_a now", 1024, "see private reference now", "private_token"),
        ("xdeleg_a 1deleg_b", 1024, "xprivate reference 1private reference", "private_token"),
        ("deleg_ and deleg_-a are not tokens", 1024, "deleg_ and deleg_-a are not tokens", None),
        ("x" * 1024, 1024, "x" * 1024, None),
        ("x" * 1025, 1024, "NOTICE", "bound"),
        ("x" * 1017 + "deleg_a", 1024, "NOTICE", "bound"),
        (" \n", 1024, "NOTICE", "blank"),
    ],
)
def test_admission_decides_once_and_never_raises_for_text(
    text: str, limit: int, admitted: str, category: str | None
) -> None:
    assert admit_result_text(text, limit=limit, notice="NOTICE") == (admitted, category)
