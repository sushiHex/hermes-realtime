"""Every Codex protocol send is bounded, and a failed send closes the adapter (#274).

The fakes model the subprocess transport's one pipe: once a send stalls, every later send
stalls behind it, and only closing the transport wakes them with a connection loss.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping

import pytest
from test_codex_app_server import (
    FakeWorkToolHandler,
    ManualDynamicCodexTransport,
    _collect_stream,
)

from hermes_realtime.conversation.task_proposals import HostTaskProposalRequest
from hermes_realtime.providers import codex_app_server as module
from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    SubprocessCodexJsonLineTransport,
)
from hermes_realtime.providers.current_facts import CurrentFactEvidence

_RESULT_ID = 91
_SEND_MARKER = "[codex-send-bound] "


def _no_transport() -> None:
    raise AssertionError("unit test opened a Codex transport")


def _is_tool_result(message: Mapping[str, object]) -> bool:
    return message.get("id") == _RESULT_ID and "result" in message


class StalledPipeTransport(ManualDynamicCodexTransport):
    """One pipe lock, like the subprocess transport: a stall holds every later sender."""

    def __init__(self, *, stall: Callable[[Mapping[str, object]], bool]) -> None:
        super().__init__()
        self._stall = stall
        self._pipe = asyncio.Lock()
        self.stalled = asyncio.Event()
        self.lost = asyncio.Event()
        self.stalled_messages: list[dict[str, object]] = []

    async def send(self, message: Mapping[str, object]) -> None:
        async with self._pipe:
            if self.closed:
                raise RuntimeError("Codex app-server process is not available")
            if self.stalled.is_set() or self._stall(message):
                self.stalled.set()
                self.stalled_messages.append(dict(message))
                self.sent.append(dict(message))  # Written; delivery is uncertain.
                await self.lost.wait()
                raise RuntimeError("Codex app-server connection lost")
            await super().send(message)

    async def close(self) -> None:
        await super().close()
        self.lost.set()


class BrokenAfterWriteTransport(ManualDynamicCodexTransport):
    async def send(self, message: Mapping[str, object]) -> None:
        if _is_tool_result(message):
            self.sent.append(dict(message))
            raise RuntimeError("Codex app-server pipe broke after the write")
        await super().send(message)


class CompletesWhileReceiptStalls(StalledPipeTransport):
    def __init__(self) -> None:
        super().__init__(stall=_is_tool_result)

    async def send(self, message: Mapping[str, object]) -> None:
        if _is_tool_result(message):
            await self.complete_turn(delta="On it.")
        await super().send(message)


def _inference(
    transport: ManualDynamicCodexTransport,
    *,
    request_timeout: float = 0.05,
) -> CodexAppServerStreamingInference:
    return CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
        request_timeout_seconds=request_timeout,
        work_tool_timeout_seconds=0.05,
    )


async def _start_stalled_receipt(
    transport: StalledPipeTransport,
    inference: CodexAppServerStreamingInference,
    turn_id: str,
) -> tuple[asyncio.Task[list[str]], module._TurnRouting, module._DynamicCall]:
    stream = asyncio.create_task(_collect_stream(inference, turn_id))
    await transport.turn_started.get()
    await transport.start_item()
    await transport.request_tool(request_id=_RESULT_ID)
    await transport.stalled.wait()
    assert transport.thread_id is not None and transport.turn_id is not None
    routing = inference._turn_routing[(transport.thread_id, transport.turn_id)]
    call = inference._dynamic_calls[(transport.thread_id, transport.turn_id, "call_1")]
    return stream, routing, call


async def _close_bounded(inference: CodexAppServerStreamingInference) -> None:
    async with asyncio.timeout(2):
        await inference.close()


async def _release_and_close(
    transport: StalledPipeTransport | ManualDynamicCodexTransport,
    inference: CodexAppServerStreamingInference,
    stream: asyncio.Task[list[str]],
) -> None:
    # Last resort for a failing run: break the modeled pipe so nothing can hang the loop.
    if isinstance(transport, StalledPipeTransport):
        transport.lost.set()
    await _close_bounded(inference)
    async with asyncio.timeout(2):
        await asyncio.gather(stream, return_exceptions=True)


async def _until_closed(inference: CodexAppServerStreamingInference) -> None:
    while not inference._closed:
        await asyncio.sleep(0)


async def _assert_refuses_next_stream(inference: CodexAppServerStreamingInference) -> None:
    with pytest.raises(RuntimeError, match="closed"):
        await _collect_stream(inference, "turn_after_failure")


@pytest.mark.asyncio
async def test_stalled_tool_result_bounds_cancel() -> None:
    transport = StalledPipeTransport(stall=_is_tool_result)
    inference = _inference(transport)
    handler = FakeWorkToolHandler()
    inference.bind_work_tools(handler)
    stream, _routing, _call = await _start_stalled_receipt(transport, inference, "turn_cancel")
    try:
        async with asyncio.timeout(1):
            with pytest.raises(RuntimeError):
                await inference.cancel("turn_cancel")
            with pytest.raises(RuntimeError):
                await stream
        assert inference._closed is True
        assert handler.starts == handler.cancels == handler.exact_cancels == []
        await _assert_refuses_next_stream(inference)
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_stall_during_stream_cleanup_quiescence_exits_bounded() -> None:
    transport = StalledPipeTransport(stall=_is_tool_result)
    inference = _inference(transport)
    inference.bind_work_tools(FakeWorkToolHandler())
    stream, _routing, _call = await _start_stalled_receipt(transport, inference, "turn_quiesce")
    try:
        stream.cancel()
        async with asyncio.timeout(1):
            with pytest.raises((asyncio.CancelledError, RuntimeError)):
                await stream
        assert inference._closed is True
        await _assert_refuses_next_stream(inference)
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_stalled_rejection_of_malformed_request_keeps_close_bounded() -> None:
    transport = StalledPipeTransport(stall=lambda message: message.get("id") == 92)
    inference = _inference(transport)
    inference.bind_work_tools(FakeWorkToolHandler())
    stream = asyncio.create_task(_collect_stream(inference, "turn_malformed"))
    try:
        async with asyncio.timeout(1):
            await transport.turn_started.get()
            await transport.request_tool(request_id=92, call_id="call_unregistered")
            await transport.stalled.wait()
            await inference.close()
            with pytest.raises(RuntimeError):
                await stream
        assert transport.closed
        assert [row["id"] for row in transport.stalled_messages][0] == 92
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_stalled_turn_interrupt_send_keeps_close_bounded() -> None:
    transport = StalledPipeTransport(
        stall=lambda message: message.get("method") == "turn/interrupt",
    )
    inference = _inference(transport, request_timeout=0.2)
    stream = asyncio.create_task(_collect_stream(inference, "turn_interrupt_stall"))
    try:
        async with asyncio.timeout(2):
            await transport.turn_started.get()
            await inference.close()
            with pytest.raises(RuntimeError):
                await stream
        assert transport.closed
        assert transport.stalled_messages[0]["method"] == "turn/interrupt"
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_stalled_initialized_send_keeps_close_bounded() -> None:
    transport = StalledPipeTransport(
        stall=lambda message: message.get("method") == "initialized",
    )
    inference = _inference(transport)
    stream = asyncio.create_task(_collect_stream(inference, "turn_initialized_stall"))
    try:
        async with asyncio.timeout(1):
            await transport.stalled.wait()
            await inference.close()
            with pytest.raises(RuntimeError):
                await stream
        assert transport.closed
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_failed_turn_closes_bounded_while_another_request_holds_the_send_lock() -> None:
    # A request awaiting its response may hold the send lock for its whole request timeout.
    # close() waits for the failed turn's interrupt before closing the transport, so the
    # interrupt must give up on the lock at the cleanup deadline; nothing else can wake it.
    transport = StalledPipeTransport(stall=_is_tool_result)
    inference = _inference(transport, request_timeout=30.0)
    inference.bind_work_tools(FakeWorkToolHandler())
    stream = asyncio.create_task(_collect_stream(inference, "turn_lock_held"))
    held = False
    try:
        async with asyncio.timeout(1):
            await transport.turn_started.get()
            await transport.start_item()
            await inference._send_lock.acquire()
            held = True
            await transport.request_tool(request_id=_RESULT_ID)
        async with asyncio.timeout(3):
            with pytest.raises(RuntimeError):
                await stream
            await inference.close()
        assert inference._closed is True and transport.closed
        assert inference._send_lock.locked()
    finally:
        if held:
            inference._send_lock.release()
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_send_that_returns_after_its_deadline_without_suspending_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    inference = CodexAppServerStreamingInference(
        model="synthetic", effort="low", transport_factory=_no_transport,
    )
    loop = asyncio.get_running_loop()
    now = loop.time()
    clock = {"now": now}
    monkeypatch.setattr(loop, "time", lambda: clock["now"])
    sent: list[dict[str, object]] = []

    async def slow_send(message: Mapping[str, object]) -> None:
        sent.append(dict(message))
        clock["now"] = now + 5  # A synchronous stall: no suspension for a timeout to cancel.

    inference._send = slow_send  # type: ignore[method-assign]
    routing = module._TurnRouting()
    try:
        delivered = await inference._send_bounded(
            {"id": 7, "result": {}}, deadline=now + 1, category="tool_result", routing=routing,
        )
        assert delivered is False and len(sent) == 1
        assert routing.failure.is_set() and inference._closed is True
        evidence = [
            json.loads(line.removeprefix(_SEND_MARKER))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith(_SEND_MARKER)
        ]
        assert evidence == [{"category": "tool_result", "outcome": "deadline", "version": 1}]
    finally:
        monkeypatch.undo()
        await _close_bounded(inference)


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [True, False])
async def test_duplicate_after_failed_receipt_shares_one_cleanup_deadline(
    expired: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Proposal receipts now use the host-selection fail-closed path. Keep the
    # generic tool-result cleanup-deadline proof on the remaining knowledge arm.
    class Lookup:
        async def lookup(self, query: str) -> CurrentFactEvidence:
            return CurrentFactEvidence(query=query, retrieved_date="2026-10-19", sources=())

    inference = CodexAppServerStreamingInference(
        model="synthetic",
        effort="low",
        transport_factory=_no_transport,
        request_timeout_seconds=0.05,
        work_tool_timeout_seconds=0.05,
        current_fact_lookup=Lookup(),
    )
    handler = FakeWorkToolHandler()
    inference.bind_work_tools(handler)
    loop = asyncio.get_running_loop()
    thread_id, turn_id = "thread", "turn"
    inference._server_turn_ready[thread_id] = asyncio.Event()
    inference._server_turn_ready[thread_id].set()
    inference._advertised_dynamic_tools[thread_id] = frozenset({"search_knowledge"})
    inference._active_turns["turn_public"] = module._ActiveTurn("turn_public", thread_id, turn_id)
    routing = module._TurnRouting(lookup_attempted=True)
    inference._turn_routing[(thread_id, turn_id)] = routing
    arguments: dict[str, object] = {"query": "Inspect synthetic documentation"}
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    call = module._DynamicCall(
        (thread_id, turn_id, "call_1"), "search_knowledge", None, dict(arguments), canonical,
        loop.time() + 0.05,
    )
    inference._dynamic_calls[call.identity] = call
    attempts: list[dict[str, object]] = []
    lost = asyncio.Event()

    async def stalled_send(message: Mapping[str, object]) -> None:
        attempts.append(dict(message))
        await lost.wait()
        raise RuntimeError("Codex app-server connection lost")

    inference._send = stalled_send  # type: ignore[method-assign]

    def request(request_id: int) -> dict[str, object]:
        return {
            "id": request_id,
            "method": "item/tool/call",
            "params": {
                "threadId": thread_id,
                "turnId": turn_id,
                "callId": "call_1",
                "tool": "search_knowledge",
                "arguments": arguments,
            },
        }

    def responses() -> list[object]:
        # The close this failure begins may also attempt its own turn/interrupt request.
        return [row["id"] for row in attempts if "method" not in row]

    try:
        async with asyncio.timeout(1):
            await asyncio.create_task(inference._route_server_request(request(_RESULT_ID)))
        assert responses() == [_RESULT_ID]
        deadline = routing.cleanup_deadline
        assert deadline is not None
        assert routing.failure.is_set() and inference._closed is True
        assert handler.starts == handler.cancels == handler.exact_cancels == []
        assert routing.lookup_attempted is True and routing.accepted_start is False
        monkeypatch.setattr(loop, "time", lambda: deadline + (1 if expired else -0.01))
        async with asyncio.timeout(1):
            await asyncio.create_task(inference._route_server_request(request(92)))
        assert responses() == ([_RESULT_ID] if expired else [_RESULT_ID, 92])
        assert routing.cleanup_deadline == deadline
    finally:
        lost.set()
        await _close_bounded(inference)


@pytest.mark.asyncio
async def test_send_error_after_write_answers_each_request_once() -> None:
    transport = BrokenAfterWriteTransport()
    inference = _inference(transport)
    handler = FakeWorkToolHandler()
    inference.bind_work_tools(handler)
    stream = asyncio.create_task(_collect_stream(inference, "turn_broken"))
    try:
        async with asyncio.timeout(1):
            await transport.turn_started.get()
            await transport.start_item()
            await transport.request_tool(request_id=_RESULT_ID)
            with pytest.raises(RuntimeError):
                await stream
            await _until_closed(inference)
            await inference.close()
        assert len([row for row in transport.sent if row.get("id") == _RESULT_ID]) == 1
        assert handler.starts == handler.cancels == handler.exact_cancels == []
        await _assert_refuses_next_stream(inference)
    finally:
        await _close_bounded(inference)
        await asyncio.gather(stream, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_receipt_preserves_exact_proposal_without_start_authority(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = StalledPipeTransport(stall=_is_tool_result)
    inference = _inference(transport)
    handler = FakeWorkToolHandler()
    inference.bind_work_tools(handler)
    stream, routing, call = await _start_stalled_receipt(transport, inference, "turn_accepted")
    try:
        async with asyncio.timeout(1):
            with pytest.raises(RuntimeError):
                await stream
            await _until_closed(inference)
        assert handler.starts == []
        assert handler.cancels == handler.exact_cancels == []
        assert routing.accepted_start is False
        assert call.operation is not None and call.operation.done()
        assert not call.operation.cancelled()
        proposal = call.operation.result()
        assert type(proposal) is HostTaskProposalRequest
        assert proposal.objective == "Inspect the build"
        assert routing.proposal == proposal
        assert call.response is not None
        assert json.loads(call.response["contentItems"][0]["text"]) == {  # type: ignore[index]
            "proposed": True, "host_applies_start_policy": True,
        }
        assert len([row for row in transport.sent if row.get("id") == _RESULT_ID]) == 1
        markers = [
            json.loads(line.removeprefix("[codex-task-status] "))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[codex-task-status] ")
        ]
        assert markers == [{"category": "response_send_deadline"}]
        await _assert_refuses_next_stream(inference)
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_terminal_while_proposal_receipt_stalls_fails_without_a_second_send() -> None:
    transport = CompletesWhileReceiptStalls()
    inference = _inference(transport)
    handler = FakeWorkToolHandler()
    inference.bind_work_tools(handler)
    stream, routing, _call = await _start_stalled_receipt(transport, inference, "turn_terminal")
    try:
        async with asyncio.timeout(1):
            with pytest.raises(RuntimeError, match="Codex status receipt send failed"):
                await stream
            await _until_closed(inference)
        assert routing.terminal is True
        # A failed host-selection receipt closes the adapter. If close races the
        # queued terminal event, it may send one exact-turn interrupt, never retry it.
        interrupts = [row for row in transport.sent if row.get("method") == "turn/interrupt"]
        assert len(interrupts) <= 1
        assert all(row["params"] == {"threadId": transport.thread_id,
                                      "turnId": transport.turn_id} for row in interrupts)
        assert len([row for row in transport.sent if row.get("id") == _RESULT_ID]) == 1
        assert handler.starts == handler.cancels == handler.exact_cancels == []
        await _assert_refuses_next_stream(inference)
    finally:
        await _release_and_close(transport, inference, stream)


@pytest.mark.asyncio
async def test_subprocess_transport_close_wakes_a_send_the_child_never_reads() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        "import time; time.sleep(30)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    transport = SubprocessCodexJsonLineTransport(process, close_timeout_seconds=0.2)
    sender = asyncio.create_task(
        transport.send({"method": "fill", "params": {"pad": "x" * (512 * 1024)}})
    )
    try:
        await asyncio.sleep(0.1)
        assert not sender.done()
        started = asyncio.get_running_loop().time()
        async with asyncio.timeout(2):
            await transport.close()
        assert asyncio.get_running_loop().time() - started < 1.5
        with pytest.raises(Exception):  # noqa: B017 - any connection-loss error wakes the drain
            async with asyncio.timeout(1):
                await sender
        assert process.returncode is not None
    finally:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        if process.returncode is None:
            process.kill()
            await process.wait()
