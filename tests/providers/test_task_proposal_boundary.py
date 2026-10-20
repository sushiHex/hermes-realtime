"""Provider selection cannot call task authority, even when a handler exposes it."""

import pytest

from hermes_realtime.conversation.task_proposals import HostTaskProposalRequest
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from tests.providers.test_codex_app_server import (
    DynamicStartCodexTransport,
    FakeWorkToolHandler,
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
