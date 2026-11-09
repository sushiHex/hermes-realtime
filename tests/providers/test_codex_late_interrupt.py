"""A late interrupt never ends the session for a turn that already completed (#478).

Upstream app-server rejects interrupting a completed turn with -32600. On the normal
interrupt path the adapter rechecks the routed completion under send ownership, and
tolerates that error once the completion is routed. A real routing failure stays terminal.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from test_codex_app_server import (
    FakeWorkToolHandler,
    ManualDynamicCodexTransport,
    _assert_codex_refusal,
    _collect_stream,
    _snapshot,
)

from hermes_realtime.conversation.streaming import ModelOutputRefused
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from tests.support.automatic_gc import no_automatic_gc

_IMITATION = 'Then I run start_work(objective="x"). Another sentence follows here. And one more.'


@pytest.fixture(autouse=True)
def _no_automatic_gc() -> Iterator[None]:
    # The fakes' sends never suspend; keep a collection out of the bounded waits.
    with no_automatic_gc():
        yield


def _inference(transport: ManualDynamicCodexTransport) -> CodexAppServerStreamingInference:
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
        request_timeout_seconds=1.0,
        work_tool_timeout_seconds=1.0,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    return inference


async def _until_waiting() -> None:
    for _ in range(200):
        if any(task.get_name() == "codex-app-server-next-event" for task in asyncio.all_tasks()):
            return
        await asyncio.sleep(0)
    pytest.fail("the stream never waited for an event")


async def _delta(transport: ManualDynamicCodexTransport, text: str) -> None:
    await transport.incoming.put({
        "method": "item/agentMessage/delta",
        "params": {
            "threadId": transport.thread_id, "turnId": transport.turn_id,
            "itemId": "item_delta", "delta": text,
        },
    })


async def _until_completion_routed(
    inference: CodexAppServerStreamingInference, transport: ManualDynamicCodexTransport,
) -> None:
    assert transport.thread_id is not None and transport.turn_id is not None
    routing = inference._turn_routing[(transport.thread_id, transport.turn_id)]
    while not routing.terminal:
        await asyncio.sleep(0)


def _interrupts(transport: ManualDynamicCodexTransport) -> list[dict[str, object]]:
    return [row for row in transport.sent if row.get("method") == "turn/interrupt"]


@pytest.mark.asyncio
async def test_refusal_queued_before_completion_sends_no_interrupt(capsys: Any) -> None:
    transport = ManualDynamicCodexTransport()
    transport.interrupt_reply = "error"
    inference = _inference(transport)
    stream = asyncio.create_task(_collect_stream(inference, "turn_late_refusal"))
    try:
        async with asyncio.timeout(1):
            await transport.turn_started.get()
            await _until_waiting()
            # The reader routes the delta and the completion before the stream runs.
            await transport.complete_turn(delta=_IMITATION)
            with pytest.raises(ModelOutputRefused) as caught:
                await stream
        _assert_codex_refusal(caught.value, capsys, "call_imitation")
        assert _interrupts(transport) == []
    finally:
        await inference.close()
        await asyncio.gather(stream, return_exceptions=True)


@pytest.mark.asyncio
async def test_completion_racing_the_interrupt_reply_keeps_the_refusal(capsys: Any) -> None:
    transport = ManualDynamicCodexTransport()
    transport.interrupt_reply = "race"
    inference = _inference(transport)
    stream = asyncio.create_task(_collect_stream(inference, "turn_racing_interrupt"))
    try:
        async with asyncio.timeout(1):
            await transport.turn_started.get()
            await _until_waiting()
            await _delta(transport, _IMITATION)
            with pytest.raises(ModelOutputRefused) as caught:
                await stream
        _assert_codex_refusal(caught.value, capsys, "call_imitation")
        assert len(_interrupts(transport)) == 1
    finally:
        await inference.close()
        await asyncio.gather(stream, return_exceptions=True)


@pytest.mark.asyncio
async def test_barge_in_after_completion_is_routed_is_silent() -> None:
    transport = ManualDynamicCodexTransport()
    transport.interrupt_reply = "error"
    inference = _inference(transport)
    stream: AsyncIterator[object] = inference.stream(_snapshot(), turn_id="turn_late_barge_in")
    try:
        async with asyncio.timeout(1):
            first = asyncio.ensure_future(stream.__anext__())
            await transport.turn_started.get()
            await _until_waiting()
            await transport.complete_turn(delta="Hello there. More")
            assert await first == "Hello there."
            await _until_completion_routed(inference, transport)
            assert await inference.cancel("turn_late_barge_in") is None
            assert [part async for part in stream] == ["More"]
        assert _interrupts(transport) == []
    finally:
        await inference.close()
        await stream.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_barge_in_after_a_real_failure_and_completion_still_raises() -> None:
    transport = ManualDynamicCodexTransport()
    transport.interrupt_reply = "error"
    inference = _inference(transport)
    stream: AsyncIterator[object] = inference.stream(_snapshot(), turn_id="turn_failed_barge_in")
    try:
        async with asyncio.timeout(1):
            first = asyncio.ensure_future(stream.__anext__())
            await transport.turn_started.get()
            await _until_waiting()
            await _delta(transport, "Hello there. ")
            # The stream now waits at its yield while the reader marks the failure and
            # then routes the completion.
            assert await first == "Hello there."
            await transport.start_item(item_type="commandExecution")
            await transport.complete_turn()
            await _until_completion_routed(inference, transport)
            with pytest.raises(RuntimeError) as caught:
                await inference.cancel("turn_failed_barge_in")
            assert type(caught.value) is RuntimeError
            assert str(caught.value) == "Codex app-server reader failed"
            with pytest.raises(RuntimeError, match="reader failed"):
                await stream.__anext__()
    finally:
        await inference.close()
        await stream.aclose()  # type: ignore[attr-defined]
