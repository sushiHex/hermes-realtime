import pytest

from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from tests.providers.test_codex_app_server import (
    FakeCodexTransport,
    FakeWorkToolHandler,
    _snapshot,
)


@pytest.mark.asyncio
async def test_resolved_background_tools_are_described_without_foreground_execution_tools():
    transport = FakeCodexTransport(assistant_delta="I'm here.")
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
        background_tools=lambda: ("mcp__vram__vram_status", "web_search"),
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        _ = [part async for part in inference.stream(_snapshot(), turn_id="turn_catalog")]
        thread = next(row for row in transport.sent if row.get("method") == "thread/start")
        instructions = thread["params"]["baseInstructions"]
        assert "mcp__vram__vram_status" in instructions
        assert "descriptive" in instructions
        assert "use advertised start_work" in instructions
        assert "mcp__vram__vram_status" not in str(thread["params"]["dynamicTools"])
    finally:
        await inference.close()
