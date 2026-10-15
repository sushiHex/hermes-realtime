"""Forced provider decisions; no model process is launched."""

import pytest
from test_ollama_work_tools import adapter, collect, decision_frame

from hermes_realtime.conversation.task_status import HostSpeechRequest


@pytest.mark.asyncio
async def test_ollama_status_is_a_marker_without_work_effects():
    inference, handler, sent, _ = adapter([decision_frame({"kind": "task_status"})])
    try:
        assert await collect(inference) == [HostSpeechRequest()]
        assert handler.calls == []
        assert sent[0]["format"]["oneOf"][-1] == {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind"],
            "properties": {"kind": {"const": "task_status"}},
        }
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_codex_status_bypasses_model_continuation():
    import asyncio

    from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    transport = DynamicStartCodexTransport(
        tool="task_status", arguments={}, assistant_delta="The task has been cancelled."
    )
    handler = FakeWorkToolHandler()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(handler)
    try:
        async with asyncio.timeout(1):
            assert [
                part async for part in inference.stream(_snapshot(), turn_id="turn_status")
            ] == [HostSpeechRequest()]
        assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_provider_prompts_preserve_heard_host_provenance():
    import json

    from test_codex_app_server import FakeCodexTransport, FakeWorkToolHandler

    from hermes_realtime.conversation.context import ConversationMessage
    from hermes_realtime.conversation.streaming import ConversationInferenceRequest
    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    snapshot = ConversationInferenceRequest(
        revision=1,
        messages=(
            ConversationMessage("assistant", "Observed state.", source="host_task_status"),
            ConversationMessage("user", "Now?"),
        ),
        active_tasks=(),
    )
    inference, _, sent, _ = adapter([decision_frame({"kind": "reply", "text": "Synthetic."})])
    try:
        await collect(inference, snapshot)
    finally:
        await inference.close()
    reference = json.loads(sent[0]["messages"][0]["content"].split("Reference data:\n", 1)[1])
    assert reference["host_speech_message_indices"] == [0]
    transport = FakeCodexTransport()
    codex = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    codex.bind_work_tools(FakeWorkToolHandler())
    try:
        [part async for part in codex.stream(snapshot, turn_id="turn_source")]
    finally:
        await codex.close()
    request = next(row for row in transport.sent if row.get("method") == "turn/start")
    payload = json.loads(request["params"]["input"][0]["text"].split("\n")[-1])
    assert payload["host_speech_message_indices"] == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{"text": "Cancelled."}, {"objective": "Work"}, {"task_id": "task_test"}, {"approved": True}],
)
async def test_ollama_status_rejects_every_argument(extra):
    inference, handler, _, _ = adapter([decision_frame({"kind": "task_status", **extra})])
    try:
        with pytest.raises(ValueError, match="decision_schema"):
            await collect(inference)
        assert handler.calls == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_ollama_status_not_offered_to_plain_snapshot():
    from hermes_realtime.conversation.context import (
        ConversationContextSnapshot,
        ConversationMessage,
    )

    inference, handler, sent, _ = adapter([decision_frame({"kind": "task_status"})])
    try:
        with pytest.raises(ValueError, match="unadvertised"):
            await collect(
                inference,
                ConversationContextSnapshot(
                    revision=1, messages=(ConversationMessage("user", "Status?"),), active_tasks=()
                ),
            )
        assert sent[0]["format"]["properties"]["kind"] == {"const": "reply"}
        assert handler.calls == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_codex_status_finishes_without_any_model_continuation():
    import asyncio

    from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    class SilentAfterTool(DynamicStartCodexTransport):
        async def send(self, message):
            if message.get("id") == self.request_id:
                self.sent.append(dict(message))
                self.tool_response.set()
                return
            await super().send(message)

    transport = SilentAfterTool(tool="task_status", arguments={})
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        async with asyncio.timeout(1):
            assert [
                part async for part in inference.stream(_snapshot(), turn_id="turn_silent")
            ] == [HostSpeechRequest()]
        assert any(row.get("method") == "turn/interrupt" for row in transport.sent)
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_free_reply_counterexample_is_preserved():
    text = "The background task has been cancelled."
    inference, handler, _, _ = adapter([decision_frame({"kind": "reply", "text": text})])
    try:
        assert await collect(inference) == [text]
        assert handler.calls == []
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_kind", ["plain", "announcement"])
async def test_codex_status_not_advertised_without_admitted_user(snapshot_kind):
    from test_codex_app_server import FakeCodexTransport, FakeWorkToolHandler

    from hermes_realtime.conversation.context import (
        ConversationContextSnapshot,
        ConversationMessage,
    )
    from hermes_realtime.conversation.streaming import ConversationInferenceRequest
    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    transport = FakeCodexTransport()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    snapshot = (
        ConversationContextSnapshot(
            revision=1, messages=(ConversationMessage("user", "Status?"),), active_tasks=()
        )
        if snapshot_kind == "plain"
        else ConversationInferenceRequest(
            revision=1,
            messages=(ConversationMessage("assistant", "Synthetic update."),),
            active_tasks=(),
        )
    )
    try:
        [part async for part in inference.stream(snapshot, turn_id="turn_no_status")]
    finally:
        await inference.close()
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert {tool["name"] for tool in request["params"]["dynamicTools"]} == (
        {"start_work"} if snapshot_kind == "plain" else set()
    )


@pytest.mark.parametrize("arguments", [{"text": "Cancelled."}, {"task_id": "task_test"}])
def test_codex_status_accepts_no_arguments(arguments):
    from test_codex_app_server import FakeCodexTransport

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=FakeCodexTransport
    )
    with pytest.raises(RuntimeError, match="arguments"):
        inference._dynamic_semantics({"tool": "task_status", "arguments": arguments})


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["task_status", "start_work"])
async def test_codex_first_admitted_choice_refuses_other_choice(first):
    import asyncio

    from test_codex_app_server import FakeWorkToolHandler, ManualDynamicCodexTransport, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    transport = ManualDynamicCodexTransport()
    handler = FakeWorkToolHandler()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(handler)

    async def collect_turn():
        return [part async for part in inference.stream(_snapshot(), turn_id="turn_order")]

    response = asyncio.create_task(collect_turn())
    try:
        async with asyncio.timeout(2):
            await transport.turn_started.get()
            second = "start_work" if first == "task_status" else "task_status"
            for number, tool in enumerate((first, second), 1):
                arguments = {} if tool == "task_status" else {"objective": "Inspect the build"}
                await transport.start_item(call_id=f"call_{number}", tool=tool, arguments=arguments)
                await transport.request_tool(
                    request_id=90 + number, call_id=f"call_{number}", tool=tool, arguments=arguments
                )
            if first == "task_status":
                assert await response == [HostSpeechRequest()]
                assert handler.starts == []
            else:
                responses = [await transport.tool_responses.get() for _ in range(2)]
                assert sum(row["result"]["success"] for row in responses) == 1
                await transport.complete_turn(delta="Synthetic response.")
                assert await response == ["Synthetic response."]
                assert len(handler.starts) == 1
            assert handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()
        await asyncio.gather(response, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_codex_turn_cannot_emit_ready_status():
    import asyncio

    from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    transport = DynamicStartCodexTransport(tool="task_status", arguments={})
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    original = inference._next_event

    async def cancel_at_delivery(*args, **kwargs):
        event = await original(*args, **kwargs)
        if isinstance(event, HostSpeechRequest):
            await inference.cancel("turn_cancel_status")
        return event

    inference._next_event = cancel_at_delivery
    try:
        with pytest.raises(asyncio.CancelledError):
            [part async for part in inference.stream(_snapshot(), turn_id="turn_cancel_status")]
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_codex_status_selection_suppresses_prose_before_tool_request():
    import asyncio

    from test_codex_app_server import FakeWorkToolHandler, ManualDynamicCodexTransport, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    transport = ManualDynamicCodexTransport()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    consumed = asyncio.Event()
    original = inference._next_event

    async def observe(*args, **kwargs):
        event = await original(*args, **kwargs)
        if type(event) is dict and event.get("method") == "item/agentMessage/delta":
            consumed.set()
        return event

    inference._next_event = observe

    async def collect_turn():
        return [part async for part in inference.stream(_snapshot(), turn_id="turn_pre_request")]

    response = asyncio.create_task(collect_turn())
    try:
        async with asyncio.timeout(2):
            await transport.turn_started.get()
            await transport.start_item(tool="task_status", arguments={})
            await transport.incoming.put(
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": transport.thread_id,
                        "turnId": transport.turn_id,
                        "itemId": "synthetic",
                        "delta": "The task has been cancelled.",
                    },
                }
            )
            await consumed.wait()
            await transport.request_tool(request_id=91, tool="task_status", arguments={})
            assert await response == [HostSpeechRequest()]
    finally:
        await inference.close()
        await asyncio.gather(response, return_exceptions=True)


@pytest.mark.asyncio
async def test_codex_completion_during_tool_response_send_still_selects_host_status():
    import asyncio

    from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    class CompletionDuringSend(DynamicStartCodexTransport):
        def __init__(self):
            super().__init__(tool="task_status", arguments={}, assistant_delta="Cancelled.")
            self.send_waiting = asyncio.Event()
            self.release_send = asyncio.Event()

        async def send(self, message):
            await super().send(message)
            if message.get("id") == self.request_id:
                self.send_waiting.set()
                await self.release_send.wait()

    transport = CompletionDuringSend()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    selected = asyncio.Event()
    rows = []

    async def collect_turn():
        async for part in inference.stream(_snapshot(), turn_id="turn_send_race"):
            rows.append(part)
            selected.set()

    response = asyncio.create_task(collect_turn())
    try:
        await asyncio.wait_for(transport.send_waiting.wait(), 1)
        await asyncio.wait_for(selected.wait(), 0.2)
        assert rows == [HostSpeechRequest()]
    finally:
        transport.release_send.set()
        await asyncio.wait_for(response, 1)
        await inference.close()


@pytest.mark.asyncio
async def test_codex_completed_turn_does_not_receive_late_interrupt():
    import asyncio

    from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    class AlreadyCompleted(DynamicStartCodexTransport):
        async def send(self, message):
            if message.get("method") == "turn/interrupt":
                raise RuntimeError("synthetic completed turn cannot be interrupted")
            routing = inference._turn_routing.get(("thread_server_1", "turn_server_1"))
            await super().send(message)
            if message.get("id") == self.request_id:
                # Yield to the reader until it has observed the matching terminal notification.
                while not routing.terminal:
                    await asyncio.sleep(0)

    transport = AlreadyCompleted(tool="task_status", arguments={})
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        async with asyncio.timeout(1):
            assert [
                part async for part in inference.stream(_snapshot(), turn_id="turn_completed")
            ] == [HostSpeechRequest()]
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_status_receipt_send_uses_remaining_deadline_and_drains_terminal_turn():
    import asyncio

    from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    class HeldReceipt(DynamicStartCodexTransport):
        async def send(self, message):
            await super().send(message)
            if message.get("id") == self.request_id:
                await asyncio.Future()

    transport = HeldReceipt(tool="task_status", arguments={})
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
        work_tool_timeout_seconds=0.05,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        with pytest.raises(RuntimeError, match="status receipt"):
            async with asyncio.timeout(0.5):
                [
                    part
                    async for part in inference.stream(_snapshot(), turn_id="turn_deadline_status")
                ]
        assert len([row for row in transport.sent if row.get("id") == transport.request_id]) == 1
        assert not inference._server_request_tasks
        with pytest.raises(RuntimeError, match="closed"):
            [part async for part in inference.stream(_snapshot(), turn_id="turn_after_failure")]
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [True, False, "terminal"])
async def test_expired_status_call_cannot_send_or_select_with_immediate_transport(ready):
    import asyncio

    from test_codex_app_server import FakeWorkToolHandler, ManualDynamicCodexTransport, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    transport = ManualDynamicCodexTransport()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    registered = asyncio.Event()
    original = inference._handle_dynamic_lifecycle

    def expire(method, params, item):
        original(method, params, item)
        if method == "item/started":
            call = inference._dynamic_calls[(params["threadId"], params["turnId"], item["id"])]
            call.deadline = asyncio.get_running_loop().time() - 1
            if ready is False:
                inference._server_turn_ready[params["threadId"]].clear()
            if ready == "terminal":
                inference._turn_routing[(params["threadId"], params["turnId"])].terminal = True
            registered.set()

    inference._handle_dynamic_lifecycle = expire
    rows = []

    async def collect_turn():
        async for part in inference.stream(_snapshot(), turn_id="turn_expired"):
            rows.append(part)

    response = asyncio.create_task(collect_turn())
    try:
        await asyncio.wait_for(transport.turn_started.get(), 1)
        await transport.start_item(tool="task_status", arguments={})
        await asyncio.wait_for(registered.wait(), 1)
        await transport.request_tool(request_id=91, tool="task_status", arguments={})
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(response, 0.3)
        assert rows == []
        assert transport.tool_responses.empty()
    finally:
        await inference.close()
        await asyncio.gather(response, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("matching", [True, False])
async def test_terminal_during_status_interrupt_quiesce_is_rechecked(matching):
    import asyncio

    from test_codex_app_server import FakeWorkToolHandler, ManualDynamicCodexTransport, _snapshot

    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    class ObservedTransport(ManualDynamicCodexTransport):
        def __init__(self):
            super().__init__()
            self.terminal_observed = asyncio.Event()

        async def receive(self):
            message = await super().receive()
            if message.get("method") == "turn/completed":
                self.terminal_observed.set()
            return message

        async def send(self, message):
            if message.get("method") == "turn/interrupt":
                routing = inference._turn_routing[(self.thread_id, self.turn_id)]
                if routing.terminal:
                    raise RuntimeError("synthetic peer turn already ended")
            await super().send(message)

    transport = ObservedTransport()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    quiescing = asyncio.Event()
    release = asyncio.Event()
    original = inference._quiesce_dynamic_turn

    async def hold_interrupt(active):
        if active.cancelled:
            quiescing.set()
            await release.wait()
        await original(active)

    inference._quiesce_dynamic_turn = hold_interrupt

    async def collect_turn():
        return [part async for part in inference.stream(_snapshot(), turn_id="turn_terminal_race")]

    response = asyncio.create_task(collect_turn())
    try:
        async with asyncio.timeout(1):
            await transport.turn_started.get()
            await transport.start_item(tool="task_status", arguments={})
            await transport.request_tool(request_id=91, tool="task_status", arguments={})
            await quiescing.wait()
            await transport.incoming.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": transport.thread_id,
                        "turn": {
                            "id": transport.turn_id if matching else "turn_unrelated",
                            "status": "completed",
                        },
                    },
                }
            )
            await transport.terminal_observed.wait()
            release.set()
            assert await response == [HostSpeechRequest()]
            assert (
                any(row.get("method") == "turn/interrupt" for row in transport.sent) is not matching
            )
    finally:
        release.set()
        await inference.close()
        await asyncio.gather(response, return_exceptions=True)
