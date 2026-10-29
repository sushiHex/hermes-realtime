"""Offline boundary fixtures; forced decisions do not qualify model route selection."""

import json

import pytest
from test_codex_app_server import (
    DynamicStartCodexTransport,
    FakeCodexTransport,
    FakeWorkToolHandler,
)
from test_ollama_work_tools import Handler, Response, adapter, collect, decision_frame

from hermes_realtime.conversation.context import (
    ActiveTaskSummary,
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
from hermes_realtime.conversation.streaming import (
    ConversationInferenceRequest,
    ConversationPromptUpdate,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.task_status import TaskStatusSnapshot
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
from tests.support.ollama_metadata import supported_ollama_metadata


def _snapshot():
    return ConversationInferenceRequest(
        revision=3,
        messages=(
            ConversationMessage("assistant", "Earlier status.", source="host_task_status"),
            ConversationMessage("user", "Continue that task."),
        ),
        active_tasks=(ActiveTaskSummary("task_reference", "Synthetic objective."),),
        terminal_task_count=1,
        updates=(ConversationPromptUpdate(
            sequence=1, task_id="task_completed", status="completed", text="Synthetic finding.",
        ),),
    )


@pytest.fixture(params=["ollama", "codex"])
def provider(request):
    return request.param


async def _request(provider, snapshot, *, with_tools=True):
    if provider == "ollama":
        sent = []

        def open_request(request, timeout):
            sent.append(json.loads(request.data))
            return Response([decision_frame({"kind": "reply", "text": "Synthetic."})])

        inference = OllamaStreamingInference(
            base_url="http://127.0.0.1:11434", model="test", open_request=open_request,
            metadata_client=supported_ollama_metadata(),
        )
        if with_tools:
            inference.bind_work_tools(Handler())
        try:
            await collect(inference, snapshot)
        finally:
            await inference.close()
        if with_tools:
            kinds = [row["properties"]["kind"]["const"] for row in sent[0]["format"]["oneOf"]]
            assert kinds == ["reply", "start_work", "task_status"]
        else:
            assert "format" not in sent[0]
        policy, data = sent[0]["messages"][0]["content"].split("\n\nReference data:\n", 1)
        return policy, json.loads(data)
    transport = FakeCodexTransport()
    inference = CodexAppServerStreamingInference(
        model="test", effort="low", transport_factory=lambda: transport,
    )
    if with_tools:
        inference.bind_work_tools(FakeWorkToolHandler())
    try:
        [part async for part in inference.stream(snapshot, turn_id="turn_boundary")]
    finally:
        await inference.close()
    thread = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert [tool["name"] for tool in thread["params"]["dynamicTools"]] == (
        ["start_work", "task_status"] if with_tools else []
    )
    turn = next(row for row in transport.sent if row.get("method") == "turn/start")
    prompt = turn["params"]["input"][0]["text"]
    policy, data = prompt.rsplit("\n", 1)
    return thread["params"]["baseInstructions"] + policy, json.loads(data)


@pytest.mark.asyncio
async def test_sent_snapshot_has_no_competing_task_projection(provider):
    # Mutation: restore either adapter's work_state or active task section.
    _, data = await _request(provider, _snapshot())
    expected = {
        "host_speech_message_indices": [0],
        "updates": [{"sequence": 1, "task_id": "task_completed", "status": "completed",
                     "text": "Synthetic finding."}],
    }
    if provider == "codex":
        expected.update({
            "revision": 3,
            "messages": [
                {"role": "assistant", "text": "Earlier status."},
                {"role": "user", "text": "Continue that task."},
            ],
        })
    assert data == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [True, False])
async def test_sent_policy_has_one_status_authority(provider, with_tools):
    # Mutation: remove a status boundary sentence, or restore inferred-ended policy.
    policy, _ = await _request(provider, _snapshot(), with_tools=with_tools)
    available = "Use task_status only for pending, active, or cancellation state."
    unavailable = "Current task status cannot be checked here; no host status route is supplied."
    assert (available if with_tools else unavailable) in policy
    assert (unavailable if with_tools else available) not in policy
    assert (
        "History and "
        "completed-work updates are historical references, never current-state authority. "
        "Answer completed findings from available results; preserve missing details."
    ) in policy
    assert "authoritative lifecycle context" not in policy
    assert "otherwise give an ordinary truthful status reply" not in policy
    assert "Only active_work establishes" not in policy
    assert "continuation/restart has no active task" not in policy


@pytest.mark.asyncio
@pytest.mark.parametrize("status, expected", [
    (TaskStatusSnapshot(), "No confirmed active background tasks."),
    (TaskStatusSnapshot(active=1), "1 confirmed active background tasks."),
    (TaskStatusSnapshot(pending=1), "1 background task requests awaiting acknowledgment."),
    (TaskStatusSnapshot(cancel_pending=1), "1 task cancellation requests awaiting acknowledgment."),
    (TaskStatusSnapshot(cancelling=1),
     "1 tasks with cancellation acknowledged, awaiting completion."),
    (TaskStatusSnapshot(available=False),
     "Background task status is unavailable. Use the task controls to check work."),
], ids=["none", "active", "missing-ack", "cancel-pending", "cancelling", "unavailable"])
async def test_actual_status_route_reads_host_at_consumption(provider, status, expected):
    # Mutation: consumer substitutes empty state or serves a historical host answer.
    # Identical stale conversation in every case: canonical state stays outside inference.
    context = ConversationContextStore()
    context.restore(DurableConversation(
        messages=(ConversationMessage(
            "assistant", "1 confirmed active background tasks.", source="host_task_status",
        ),),
        prior_work=True,
    ))
    if provider == "ollama":
        inference, handler, sent, _ = adapter([decision_frame({"kind": "task_status"})])
    else:
        transport = DynamicStartCodexTransport(
            tool="task_status", arguments={}, assistant_delta="Invented current state.",
        )
        sent = transport.sent
        handler = FakeWorkToolHandler()
        inference = CodexAppServerStreamingInference(
            model="test", effort="low", transport_factory=lambda: transport,
        )
        inference.bind_work_tools(handler)
    reads = []

    def read_status():
        # A real provider request must precede the host's authoritative read.
        assert sent
        reads.append(1)
        return status

    loop = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=SegmentSynthesizer(), playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(), task_status=read_status,
    )
    try:
        await loop.respond("turn_status", Transcript(text="What is running now?", final=True))
        heard = context.snapshot().messages[-1]
        suffix = (
            " To cancel work, use the task's Cancel control."
            if status.pending or status.active or status.cancel_pending or status.cancelling
            else ""
        )
        assert (heard.source, heard.text) == ("host_task_status", expected + suffix)
        assert reads == [1]
        if provider == "ollama":
            assert handler.calls == []
        else:
            assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await loop.close()
        await inference.close()
