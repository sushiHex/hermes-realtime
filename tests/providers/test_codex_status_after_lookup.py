"""Read-only host status remains available after bounded foreground lookup."""

import asyncio
from types import SimpleNamespace

import pytest
from test_codex_app_server import FakeCodexTransport, ManualDynamicCodexTransport

from hermes_realtime import host_launcher as host
from hermes_realtime.conversation.context import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
)
from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
from hermes_realtime.conversation.streaming import ConversationInferenceRequest, StreamingSpeechLoop
from hermes_realtime.conversation.task_status import HostSpeechRequest, TaskStatusSnapshot
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.providers.current_facts import CurrentFactEvidence
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer


@pytest.fixture(autouse=True)
def synthetic_codex_executable(monkeypatch):
    # The real factory resolves its executable before we replace its transport.
    monkeypatch.setattr(
        "hermes_realtime.providers.codex_app_server._resolve_codex_executable",
        lambda _explicit: "synthetic-codex",
    )


class Lookup:
    def __init__(self, *, failed=False):
        self.queries = []
        self.failed = failed

    async def lookup(self, query):
        self.queries.append(query)
        return CurrentFactEvidence(
            query=query, retrieved_date="2026-08-05", sources=(),
            error="No source evidence." if self.failed else None,
        )

    async def close(self):
        pass


class Controller:
    async def dispatch(self, **kwargs):
        raise AssertionError("status cannot dispatch")

    async def request_cancel(self, *args, **kwargs):
        raise AssertionError("status cannot cancel")


def setup(transport, *, lookup=None, coordinator=None, bound=True):
    context = ConversationContextStore()
    surface = ConversationWorkControlSurface(
        controller=Controller(), context=context, observer=lambda *args: None,
        reserve_observer_capacity=lambda: None,
    )
    inference = host._build_streaming_inference(
        inference_provider="codex", ollama_base_url="http://127.0.0.1:1", ollama_model="",
        codex_model="gpt-5.6-terra", codex_effort="low", codex_executable=None,
        current_fact_lookup=lookup, knowledge_coordinator=coordinator,
    )
    inference._transport_factory = lambda: transport
    inference._request_timeout = 0.5
    host._bind_natural_work_tools(inference=inference, surface=surface, enabled=bound)
    return inference, context, surface


def advertised(transport):
    thread = next(m for m in transport.sent if m.get("method") == "thread/start")
    return [t["name"] for t in thread["params"]["dynamicTools"]]


def request(text="What is running now?", *, admitted=True, role="user"):
    cls = ConversationInferenceRequest if admitted else ConversationContextSnapshot
    return cls(revision=1, messages=(ConversationMessage(role, text),), active_tasks=())


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_prefetch_keeps_only_status_and_reads_host_at_consumption(failed):
    # Mutation: lookup completion still removes the read-only status capability.
    transport = ManualDynamicCodexTransport()
    lookup = Lookup(failed=failed)
    inference, context, surface = setup(transport, lookup=lookup)
    reads = []

    def status():
        reads.append(1)
        return TaskStatusSnapshot(pending=1)

    loop = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=SegmentSynthesizer(), playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(), task_status=status,
    )
    turn = asyncio.create_task(loop.respond(
        "turn_status", Transcript(text="What is running now?", final=True),
    ))
    try:
        await asyncio.wait_for(transport.turn_started.get(), 1)
        assert lookup.queries == ["What is running now?"]
        assert advertised(transport) == ["task_status"]
        await transport.start_item(tool="task_status", arguments={})
        await transport.request_tool(request_id=91, tool="task_status", arguments={})
        await asyncio.wait_for(turn, 1)
        assert reads == [1]
        assert context.snapshot().messages[-1].text == TaskStatusSnapshot(pending=1).render()
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await loop.close()
        await inference.close()
        await surface.close()


@pytest.mark.asyncio
async def test_coordinator_without_evidence_keeps_status():
    # Mutation: a reserved/withheld lookup suppresses status along with proposals.
    class Coordinator:
        lookup = Lookup()

        async def consume_turn_result(self, turn_id, query):
            return SimpleNamespace(evidence=None, timing=None)

        async def close(self):
            pass

    transport = FakeCodexTransport()
    inference, _, surface = setup(transport, coordinator=Coordinator())
    try:
        [part async for part in inference.stream(request(), turn_id="turn_reserved")]
        assert advertised(transport) == ["task_status"]
    finally:
        await inference.close()
        await surface.close()


@pytest.mark.asyncio
async def test_dynamic_lookup_can_end_in_status_without_losing_prefix_or_lookup_evidence():
    # Mutation: the lookup budget is incorrectly shared with the host terminal.
    transport = ManualDynamicCodexTransport()
    lookup = Lookup()
    inference, _, surface = setup(transport, lookup=lookup)
    prefix_seen = asyncio.Event()

    async def consume():
        parts = []
        async for part in inference.stream(request("Hello."), turn_id="turn_dynamic_status"):
            parts.append(part)
            if type(part) is str:
                prefix_seen.set()
        return parts

    turn = asyncio.create_task(consume())
    try:
        identity = await asyncio.wait_for(transport.turn_started.get(), 1)
        await transport.start_item(tool="search_knowledge", arguments={"query": "Public sources"})
        await transport.request_tool(
            request_id=91, tool="search_knowledge", arguments={"query": "Public sources"},
        )
        response = await asyncio.wait_for(transport.tool_responses.get(), 1)
        routing = inference._turn_routing[identity]
        await transport.complete_item(
            response, tool="search_knowledge", arguments={"query": "Public sources"},
        )
        await transport.incoming.put({"method": "item/agentMessage/delta", "params": {
            "threadId": identity[0], "turnId": identity[1], "itemId": "prefix",
            "delta": "I searched public sources. ",
        }})
        # Let the actual consumer emit the source prefix before terminal selection.
        await asyncio.wait_for(prefix_seen.wait(), 1)
        await transport.start_item(call_id="status_2", tool="task_status", arguments={})
        await transport.request_tool(
            request_id=92, call_id="status_2", tool="task_status", arguments={},
        )
        response = await asyncio.wait_for(transport.tool_responses.get(), 1)
        assert response["result"]["success"] is True
        parts = await asyncio.wait_for(turn, 1)
        assert parts == ["I searched public sources.", HostSpeechRequest()]
        assert lookup.queries == ["Public sources"]
        assert routing.lookup_attempted is True
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await inference.close()
        await surface.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unbound", "private", "nonuser", "legacy"])
async def test_status_still_needs_bound_admitted_public_user_turn(mode):
    # Mutations: ignore the original binding, private, current-user or typed-input gate.
    transport = FakeCodexTransport()
    inference, _, surface = setup(transport, lookup=Lookup(), bound=mode != "unbound")
    snapshot = request(
        "Read D:\\private\\secret.txt" if mode == "private" else "Hello.",
        admitted=mode != "legacy", role="assistant" if mode == "nonuser" else "user",
    )
    try:
        [part async for part in inference.stream(snapshot, turn_id="turn_excluded")]
        assert "task_status" not in advertised(transport)
    finally:
        await inference.close()
        await surface.close()


def test_status_schema_does_not_require_proposal_bounds():
    # Mutation: restore the early return that hides status when proposals are suppressed.
    tools = CodexAppServerStreamingInference._dynamic_tools(None, include_status=True)
    assert tools == [{"type": "function", "name": "task_status", "description": (
        "Report host-owned background task status and explicit Cancel controls."
    ), "inputSchema": {"type": "object", "additionalProperties": False,
                       "required": [], "properties": {}}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("first,second", [
    ("search_knowledge", "search_knowledge"),
    ("search_knowledge", "start_work"),
    ("start_work", "task_status"),
    ("task_status", "task_status"),
    ("task_status", "start_work"),
    ("task_status", "search_knowledge"),
])
async def test_lookup_and_terminal_budgets_still_reject_other_choices(first, second):
    # Mutations: waive the lookup, prior start, or prior status budget.
    transport = ManualDynamicCodexTransport()
    inference, _, surface = setup(transport, lookup=Lookup())
    thread, turn = "thread_budget", "turn_budget"
    inference._advertised_dynamic_tools[thread] = frozenset({
        "search_knowledge", "start_work", "task_status",
    })
    inference._server_turn_ready[thread] = asyncio.Event()

    def start(tool, call_id):
        arguments = (
            {"query": "Public sources"} if tool == "search_knowledge"
            else {"objective": "Inspect the build"} if tool == "start_work" else {}
        )
        item = {"id": call_id, "type": "dynamicToolCall", "tool": tool,
                "arguments": arguments, "status": "inProgress"}
        inference._handle_dynamic_lifecycle("item/started", {
            "threadId": thread, "turnId": turn, "item": item, "startedAtMs": 1,
        }, item)

    try:
        start(first, "first")
        start(second, "second")
        rejected = inference._dynamic_calls[(thread, turn, "second")]
        assert rejected.response is not None and rejected.response["success"] is False
        assert rejected.operation is None
    finally:
        await inference.close()
        await surface.close()


def test_prefetch_attempt_witness_cannot_be_cleared():
    # Mutation: overwrite the lookup witness when no later prefetch occurs.
    routing = SimpleNamespace(lookup_attempted=False)
    CodexAppServerStreamingInference._merge_prefetch_lookup_attempt(routing, attempted=True)
    CodexAppServerStreamingInference._merge_prefetch_lookup_attempt(routing, attempted=False)
    assert routing.lookup_attempted is True
