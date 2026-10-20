"""Provider selection cannot call task authority, even when a handler exposes it."""

import asyncio

import pytest

from hermes_realtime.conversation.task_proposals import HostTaskProposalRequest
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from tests.providers.test_codex_app_server import (
    DynamicStartCodexTransport,
    FakeWorkToolHandler,
    ManualDynamicCodexTransport,
    _snapshot,
)
from tests.providers.test_ollama_work_tools import adapter, start_frame


@pytest.mark.asyncio
async def test_ollama_start_selection_is_only_a_terminal_proposal():
    inference, handler, _sent, _response = adapter([start_frame()])
    try:
        items = [item async for item in inference.stream(_snapshot(), turn_id="turn_proposal")]
    finally:
        await inference.close()
    assert handler.calls == []
    assert len(items) == 1
    assert type(items[0]) is HostTaskProposalRequest
    assert items[0].objective == "Inspect release"


@pytest.mark.asyncio
async def test_codex_second_parallel_proposal_is_refused_before_it_can_replace_the_first():
    transport = ManualDynamicCodexTransport()
    handler = FakeWorkToolHandler()
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)

    async def collect():
        return [item async for item in inference.stream(_snapshot(), turn_id="turn_parallel")]

    operation = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(transport.turn_started.get(), 1)
        await transport.start_item(call_id="call_first", arguments={"objective": "First objective"})
        await transport.start_item(
            call_id="call_second", arguments={"objective": "Second objective"}
        )
        # Invert response order: the later proposal must already own a refusal.
        await transport.request_tool(
            call_id="call_second",
            request_id="request_second",
            arguments={"objective": "Second objective"},
        )
        second = await asyncio.wait_for(transport.tool_responses.get(), 1)
        assert second["result"]["success"] is False
        await transport.request_tool(
            call_id="call_first",
            request_id="request_first",
            arguments={"objective": "First objective"},
        )
        first = await asyncio.wait_for(transport.tool_responses.get(), 1)
        assert first["result"]["success"] is True
        assert await asyncio.wait_for(operation, 1) == [HostTaskProposalRequest("First objective")]
        assert handler.starts == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_codex_start_selection_is_only_a_terminal_proposal():
    transport = DynamicStartCodexTransport()
    handler = FakeWorkToolHandler()
    inference = CodexAppServerStreamingInference(
        model="synthetic",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)
    try:
        items = [item async for item in inference.stream(_snapshot(), turn_id="turn_proposal")]
    finally:
        await inference.close()
    assert handler.starts == []
    assert len(items) == 1
    assert type(items[0]) is HostTaskProposalRequest
