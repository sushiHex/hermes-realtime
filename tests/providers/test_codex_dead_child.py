"""Characterization, not endorsement: a Codex child that dies is never replaced.

The fake answers as the subprocess transport does once its child has exited: reads end with
the transport-closed error and sends report that the process is not available. Today the
adapter re-derives only the login before a turn, so it keeps the dead transport: the next
turn fails at its first send, that failure closes the adapter for the life of the process,
and every later turn is refused. The transport factory is never called a second time. The
one evidence line is the send-bound marker of the failed turn.

The session worker treats both errors as terminal for the binding, and a rebind reuses this
adapter, so reconnecting does not help; that part is read from source and not driven here.

Proposed ADR 0008 (docs/adr/0008-failure-domains.md, "a latch ends its holder") changes this
in its migration step 6, where a dead child found before a turn ends the host and a restart
recovers. Update this test in that step; do not delete it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping

import pytest
from test_codex_app_server import (
    FakeWorkToolHandler,
    ManualDynamicCodexTransport,
    _collect_stream,
)

from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    _CodexTransportClosed,
)
from tests.support.automatic_gc import no_automatic_gc

_EOF: dict[str, object] = {"eof": True}


@pytest.fixture(autouse=True)
def _no_automatic_gc() -> Iterator[None]:
    # The fakes' sends never suspend; keep a collection out of the bounded waits.
    with no_automatic_gc():
        yield


class _MortalTransport(ManualDynamicCodexTransport):
    """After ``die``, behave as ``SubprocessCodexJsonLineTransport`` does for an exited child."""

    def __init__(self) -> None:
        super().__init__()
        self.dead = False

    async def die(self) -> None:
        self.dead = True
        await self.incoming.put(_EOF)

    async def send(self, message: Mapping[str, object]) -> None:
        if self.dead:
            raise RuntimeError("Codex app-server process is not available")
        await super().send(message)

    async def receive(self) -> Mapping[str, object]:
        message = await super().receive()
        if message is _EOF:
            raise _CodexTransportClosed("Codex app-server stdout closed with exit code 1")
        return message


@pytest.mark.asyncio
async def test_a_dead_codex_child_fails_the_next_turn_and_closes_the_adapter_for_good(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = _MortalTransport()
    spawns = 0

    def factory() -> _MortalTransport:
        nonlocal spawns
        spawns += 1
        return transport

    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=factory,
        request_timeout_seconds=1.0, work_tool_timeout_seconds=1.0,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        async with asyncio.timeout(5):
            # A healthy turn first, so the child dies mid-session and not at start-up.
            healthy = asyncio.create_task(_collect_stream(inference, "turn_healthy"))
            await transport.turn_started.get()
            await transport.complete_turn(delta="Four.")
            assert await healthy == ["Four."]
            assert capsys.readouterr().out == ""

            await transport.die()
            reader = inference._reader_task
            assert reader is not None
            await asyncio.wait({reader})
            assert type(reader.exception()) is _CodexTransportClosed
            # The death alone changes nothing: no close, no respawn, no evidence.
            assert inference._closed is False
            assert inference._transport is transport
            assert capsys.readouterr().out == ""

            sent_before = len(transport.sent)
            with pytest.raises(RuntimeError) as next_turn:
                await _collect_stream(inference, "turn_after_death")
            assert str(next_turn.value) == "Codex app-server thread/start send failed"
            assert type(next_turn.value) is RuntimeError
            assert len(transport.sent) == sent_before
            assert capsys.readouterr().out == (
                '[codex-send-bound] {"category":"thread/start","outcome":"send_failed",'
                '"version":1}\n'
            )
            # That one failed send closed the adapter for the life of the process.
            assert inference._closed is True

            with pytest.raises(RuntimeError) as later_turn:
                await _collect_stream(inference, "turn_later")
            assert str(later_turn.value) == "Codex inference adapter is closed"
            assert capsys.readouterr().out == ""
            assert spawns == 1
    finally:
        await inference.close()
