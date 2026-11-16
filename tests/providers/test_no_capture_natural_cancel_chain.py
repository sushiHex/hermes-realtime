"""The natural cancel path, composed the way the host composes it with capture off (#237).

No evidence lifecycle and no evidence runtime: the command router has no lifecycle owner,
the session worker has no lifecycle resolver, and the executor and speech loop have no
evidence scheduler, admission or resolver, as in ``host_launcher`` when
``--evidence-capture`` is not passed. The real task controller, work-control surface,
command router, session worker, update executor, streaming loop and Codex adapter run
over fake transports only. Each boundary of the chain is asserted in order:

1. the target is eligible at final admission, predicate by predicate;
2. the provider request advertises ``cancel_work``;
3. the provider selects cancel (``HostCancelRequest``);
4. the accepted cancel is signalled through ``cancel_work`` to the Hermes session;
5. the task reaches its terminal.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from test_codex_app_server import DynamicStartCodexTransport, FakeCodexTransport

from hermes_realtime import host_launcher
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationTaskCommandRouter,
    ConversationTaskController,
    ConversationUpdateExecutor,
    ConversationWorkControlSurface,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
    WorkCancelResult,
)
from hermes_realtime.conversation.natural_cancel import HostCancelRequest
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.task_status import TaskStatusSnapshot
from hermes_realtime.conversation.turn import ForegroundTurn, turn_routes
from hermes_realtime.conversation.work_tools import WorkControlHealth
from hermes_realtime.conversation.worker import ConversationSessionWorker
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
from hermes_realtime.providers import codex_app_server as codex
from hermes_realtime.speech import DeliveredSpeechLedger
from tests.conversation.test_actions import director_for_test
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
from tests.conversation.test_worker import CapturingTextFinalTranscriber, ScriptedVad

_RUN = "deleg_private_chain"
_TASK = "task_chain"
_TEXT = "Please stop that background work."
_BOUND = 5.0


class _HermesSession:
    """Accepts one dispatch and one cancel; the terminal arrives only after the cancel."""

    def __init__(self) -> None:
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()
        self.cancels: list[ControlCancelEvent] = []

    async def dispatch(self, request: WorkDispatchRequestedEvent) -> WorkDispatchAcknowledgedEvent:
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged", event_id="evt_chain_ack",
            session_id=request.session_id, sequence=request.sequence + 1,
            timestamp=request.timestamp, task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(accepted=True, run_id=_RUN),
        )

    async def cancel(self, request: ControlCancelEvent) -> ControlCancelAcknowledgedEvent:
        self.cancels.append(request)
        return ControlCancelAcknowledgedEvent(
            type="control.cancel.acknowledged", event_id="evt_chain_cancel_ack",
            session_id=request.session_id, sequence=request.sequence + 1,
            timestamp=request.timestamp, request_event_id=request.event_id,
            scope=CancelScope.TASK, task_id=request.task_id,
            payload=ControlCancelAcknowledgedPayload(accepted=True, signaled_run_ids=[_RUN]),
        )

    async def next_update(self) -> WorkCompletedEvent:
        return await self.updates.get()

    async def close(self) -> None:
        return None

    def interrupt(self) -> None:
        self.updates.put_nowait(WorkCompletedEvent(
            type="work.completed", event_id="evt_chain_interrupted", session_id="session_chain",
            sequence=99, timestamp=datetime(2026, 11, 16, tzinfo=UTC), task_id=_TASK,
            run_id=_RUN, payload=WorkCompletedPayload(
                status=WorkTerminalStatus.INTERRUPTED, reason="Cancelled by request.",
            ),
        ))


class _CancelThenTextTransport(DynamicStartCodexTransport):
    """The first turn selects ``cancel_work``; the acknowledgment turn speaks text."""

    def __init__(self) -> None:
        super().__init__(tool="cancel_work", arguments={}, assistant_delta="Okay.")
        self.turn_starts = 0

    async def send(self, message: Mapping[str, object]) -> None:
        if message.get("method") == "turn/start":
            self.turn_starts += 1
            if self.turn_starts > 1:
                await FakeCodexTransport.send(self, message)
                return
        await super().send(message)


class _RecordingInference:
    """Delegates to the real Codex adapter; records each request and each yielded item."""

    def __init__(self, inner: codex.CodexAppServerStreamingInference) -> None:
        self.inner = inner
        self.requests: list[object] = []
        self.items: list[list[object]] = []

    async def stream(self, snapshot: Any, *, turn_id: str) -> Any:
        self.requests.append(snapshot)
        items: list[object] = []
        self.items.append(items)
        async for item in self.inner.stream(snapshot, turn_id=turn_id):
            items.append(item)
            yield item

    async def cancel(self, turn_id: str) -> None:
        await self.inner.cancel(turn_id)


class _RecordingCancellation:
    """Delegates to the real surface; records each predicate when the host freezes."""

    def __init__(self, surface: ConversationWorkControlSurface,
                 context: ConversationContextStore) -> None:
        self.surface, self.context = surface, context
        self.freezes: list[dict[str, object]] = []
        self.results: list[WorkCancelResult] = []

    def predicates(self) -> dict[str, object]:
        surface = self.surface
        return {
            "health_open": surface.health is WorkControlHealth.OPEN,
            "no_pending_starts": not surface._pending_starts,
            "one_active_in_context": len(self.context.snapshot().active_tasks) == 1,
            "status_exactly_one_active": surface.task_status() == TaskStatusSnapshot(active=1),
            "target": surface.natural_cancel_target(),
        }

    def natural_cancel_target(self) -> str | None:
        self.freezes.append(self.predicates())
        return self.surface.natural_cancel_target()

    async def cancel_work(self, **kwargs: Any) -> WorkCancelResult:
        result = await self.surface.cancel_work(**kwargs)
        self.results.append(result)
        return result


async def _chain() -> dict[str, Any]:
    context = ConversationContextStore()
    session = _HermesSession()
    ids = iter(("chain", "dispatch_chain", "cancel_chain", "spare_1", "spare_2"))
    controller = ConversationTaskController(
        context=context, session=session, session_id="session_chain",
        id_factory=lambda: next(ids),
    )
    controller.start()
    surface_events: list[tuple[str, dict[str, object]]] = []
    surface = ConversationWorkControlSurface(
        controller=controller, context=context,
        observer=lambda kind, data: surface_events.append((kind, dict(data))),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "chain",
    )
    transport = _CancelThenTextTransport()
    adapter = codex.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    host_launcher._bind_natural_work_tools(inference=adapter, surface=surface, enabled=True)
    inference = _RecordingInference(adapter)
    cancellation = _RecordingCancellation(surface, context)
    speech_events: list[tuple[str, dict[str, object]]] = []
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=SegmentSynthesizer(), playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(), task_status=surface.task_status,
        task_cancellation=cancellation,
        observer=lambda kind, data: speech_events.append((kind, dict(data))),
    )
    actions = ConversationUpdateExecutor(
        director=director_for_test(context, _SilentSource()), speech=speech,
    )
    actions.start()
    # Capture off: no lifecycle owner, so typed input mints no authority (worker.py).
    router = ConversationTaskCommandRouter(
        surface=surface, observer=lambda kind, data: surface_events.append((kind, dict(data))),
        reserve_observer_capacity=lambda: None,
    )
    worker = ConversationSessionWorker(
        participant_identity="browser_user", session_generation=1, vad=ScriptedVad(),
        stt=CapturingTextFinalTranscriber(_TEXT), stt_streams_partials=False,
        actions=actions, command_router=router,
    )
    worker.start()
    facts: dict[str, Any] = {"session": session, "context": context, "transport": transport,
                             "inference": inference, "cancellation": cancellation,
                             "surface_events": surface_events, "speech_events": speech_events}
    try:
        async with asyncio.timeout(_BOUND):
            started = await surface.start_work(
                objective="Synthetic background work", invocation_id="invocation_chain",
            )
            facts["started"] = started
            facts["before"] = cancellation.predicates()
            # A failure at any boundary is recorded, not raised, so each step's test
            # reports its own boundary and the first failing one is visible.
            try:
                await worker.submit_final_transcript(
                    participant_identity="browser_user", session_generation=1,
                    typed_sequence=1, text=_TEXT,
                )
                await worker.wait_for_responses()
            except Exception as error:
                facts["turn_error"] = type(error).__name__
            if session.cancels:
                session.interrupt()
                facts["completion"] = await controller.next_completion()
            facts["after_target"] = surface.natural_cancel_target()
            facts["after_status"] = surface.task_status()
    finally:
        for close in (worker.close, actions.close, adapter.close, controller.close):
            try:
                await close()
            except Exception as error:
                facts.setdefault("close_errors", []).append(type(error).__name__)
    return facts


class _SilentSource:
    async def next_completion(self) -> Any:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.fixture(scope="module")
def chain() -> dict[str, Any]:
    return asyncio.run(_chain())


def test_step0_dispatch_is_accepted_before_the_cancel(chain: dict[str, Any]) -> None:
    assert chain["started"].accepted is True and chain["started"].state == "active"
    assert chain["started"].task_id == _TASK


def test_step1_target_is_eligible_at_final_admission_predicate_by_predicate(
    chain: dict[str, Any],
) -> None:
    expected = {"health_open": True, "no_pending_starts": True,
                "one_active_in_context": True, "status_exactly_one_active": True,
                "target": _TASK}
    assert chain["before"] == expected
    assert chain["cancellation"].freezes[0] == expected


def test_step2_provider_request_advertises_cancel_work(chain: dict[str, Any]) -> None:
    request = chain["inference"].requests[0]
    assert type(request) is ConversationInferenceRequest and request.cancel_offered is True
    routes = turn_routes(ForegroundTurn(request), work_bound=True, private_material=False,
                         lookup_attempted=False, start_requires_admission=False)
    assert routes.cancel_work is True
    start = next(row for row in chain["transport"].sent if row.get("method") == "thread/start")
    names = [tool["name"] for namespace in start["params"]["dynamicTools"]
             for tool in namespace["tools"]]
    assert "cancel_work" in names


def test_step3_provider_selects_cancel(chain: dict[str, Any]) -> None:
    assert chain["inference"].items[0] == [HostCancelRequest()]


def test_step4_accepted_cancel_is_signalled_through_cancel_work(chain: dict[str, Any]) -> None:
    assert chain["cancellation"].results == [
        WorkCancelResult(accepted=True, state="cancelling", task_id=_TASK, reason=None),
    ]
    assert [request.task_id for request in chain["session"].cancels] == [_TASK]
    assert ("task_state", {"status": "cancelling", "taskId": _TASK}) in chain["surface_events"]


def test_step5_task_reaches_its_terminal(chain: dict[str, Any]) -> None:
    assert "turn_error" not in chain and "close_errors" not in chain
    assert ("assistant_turn_completed" in {kind for kind, _ in chain["speech_events"]})
    assert chain["completion"].task_id == _TASK
    assert chain["completion"].status == "interrupted"
    assert chain["context"].snapshot().active_tasks == ()
    assert chain["after_target"] is None
    assert chain["after_status"] == TaskStatusSnapshot()
