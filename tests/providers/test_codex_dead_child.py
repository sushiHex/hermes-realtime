"""A Codex adapter that can serve no further turn says so once; a model mistake does not.

The fake answers as the subprocess transport does once its child has exited: reads end with
the transport-closed error and sends report that the process is not available. The adapter
re-derives only the login before a turn, so it keeps the dead transport: the next turn fails
at its first send, and that failure closes the adapter for the life of the process. Until
#531 that was a live host with no foreground model (first pinned as a characterization in
#530). The adapter is never respawned in place; that is unchanged and deliberate.

Now the adapter calls ``on_latched`` exactly once when it closes itself after a failed send
or receipt, or when a turn finds its reader dead. The host binds that to its one
``HostFatalExit`` under the fixed cause ``foreground_adapter`` (proposed ADR 0008, step 6).

The second half guards against reporting too widely: a refused model reply, a rejected tool
call and the adapter's own close are not this latch.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping

import pytest
from test_codex_app_server import (
    FakeWorkToolHandler,
    InvalidDynamicRequestCodexTransport,
    ManualDynamicCodexTransport,
    _collect_stream,
    _snapshot,
)

from hermes_realtime.conversation.streaming import ModelOutputRefused
from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    _CodexTransportClosed,
)
from tests.support.automatic_gc import no_automatic_gc

_EOF: dict[str, object] = {"eof": True}
_IMITATION = 'Then I run start_work(objective="x"). Another sentence follows here. And one more.'


@pytest.fixture(autouse=True)
def _no_automatic_gc() -> Iterator[None]:
    # The fakes' sends never suspend; keep a collection out of the bounded waits.
    with no_automatic_gc():
        yield


class _MortalTransport(ManualDynamicCodexTransport):
    """After ``die``, behave as ``SubprocessCodexJsonLineTransport`` does for an exited child."""

    def __init__(self, *, writes_survive: bool = False) -> None:
        super().__init__()
        self.dead = False
        self.writes_survive = writes_survive

    async def die(self) -> None:
        self.dead = True
        await self.incoming.put(_EOF)

    async def send(self, message: Mapping[str, object]) -> None:
        if self.dead and not self.writes_survive:
            raise RuntimeError("Codex app-server process is not available")
        if self.dead:
            # The pipe still takes the bytes; nothing will ever answer them.
            self.sent.append(dict(message))
            return
        await super().send(message)

    async def receive(self) -> Mapping[str, object]:
        message = await super().receive()
        if message is _EOF:
            raise _CodexTransportClosed("Codex app-server stdout closed with exit code 1")
        return message


class _Adapter:
    def __init__(self, transport: ManualDynamicCodexTransport) -> None:
        self.reports = 0
        self.spawns = 0
        self.transport = transport
        self.inference = CodexAppServerStreamingInference(
            model="gpt-5.6-terra", effort="low", transport_factory=self._spawn,
            request_timeout_seconds=1.0, work_tool_timeout_seconds=1.0,
            on_latched=self._report,
        )
        self.inference.bind_work_tools(FakeWorkToolHandler())

    def _spawn(self) -> ManualDynamicCodexTransport:
        self.spawns += 1
        return self.transport

    def _report(self) -> None:
        self.reports += 1

    async def healthy_turn(self, turn_id: str = "turn_healthy") -> None:
        turn = asyncio.create_task(_collect_stream(self.inference, turn_id))
        await self.transport.turn_started.get()
        await self.transport.complete_turn(delta="Four.")
        assert await turn == ["Four."]


async def _reader_ended(inference: CodexAppServerStreamingInference) -> None:
    reader = inference._reader_task
    assert reader is not None
    await asyncio.wait({reader})
    assert type(reader.exception()) is _CodexTransportClosed


@pytest.mark.asyncio
async def test_a_dead_codex_child_latches_the_adapter_at_the_next_turn_and_reports_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = _MortalTransport()
    adapter = _Adapter(transport)
    inference = adapter.inference
    try:
        async with asyncio.timeout(5):
            # A healthy turn first, so the child dies mid-session and not at start-up.
            await adapter.healthy_turn()
            assert capsys.readouterr().out == ""

            await transport.die()
            await _reader_ended(inference)
            # The death alone changes nothing: it is found at use.
            assert inference._closed is False and adapter.reports == 0
            assert capsys.readouterr().out == ""

            sent_before = len(transport.sent)
            with pytest.raises(RuntimeError) as next_turn:
                await _collect_stream(inference, "turn_after_death")
            assert str(next_turn.value) == "Codex app-server thread/start send failed"
            assert len(transport.sent) == sent_before
            assert capsys.readouterr().out == (
                '[codex-send-bound] {"category":"thread/start","outcome":"send_failed",'
                '"version":1}\n'
            )
            # That one failed send closed the adapter for good, and said so once.
            assert inference._closed is True
            assert adapter.reports == 1

            with pytest.raises(RuntimeError) as later_turn:
                await _collect_stream(inference, "turn_later")
            assert str(later_turn.value) == "Codex inference adapter is closed"
            # Never respawned in place, and never reported twice.
            assert adapter.spawns == 1
            assert adapter.reports == 1
    finally:
        await inference.close()
    assert adapter.reports == 1


@pytest.mark.asyncio
async def test_a_dead_reader_whose_pipe_still_takes_writes_reports_once_at_the_next_turn() -> None:
    transport = _MortalTransport(writes_survive=True)
    adapter = _Adapter(transport)
    inference = adapter.inference
    try:
        async with asyncio.timeout(5):
            await adapter.healthy_turn()
            await transport.die()
            await _reader_ended(inference)
            assert adapter.reports == 0

            with pytest.raises(RuntimeError, match="Codex app-server reader failed"):
                await _collect_stream(inference, "turn_after_death")
            assert adapter.reports == 1

            with pytest.raises(RuntimeError):
                await _collect_stream(inference, "turn_later")
            assert adapter.reports == 1 and adapter.spawns == 1
    finally:
        await inference.close()
    assert adapter.reports == 1


@pytest.mark.asyncio
async def test_a_reader_that_dies_while_a_turn_waits_for_events_reports_once() -> None:
    transport = _MortalTransport(writes_survive=True)
    adapter = _Adapter(transport)
    inference = adapter.inference
    try:
        async with asyncio.timeout(5):
            await adapter.healthy_turn()
            waiting = asyncio.create_task(_collect_stream(inference, "turn_in_flight"))
            await transport.turn_started.get()
            await transport.die()

            with pytest.raises(RuntimeError, match="Codex app-server reader failed"):
                await waiting
            assert adapter.reports == 1
    finally:
        await inference.close()
    assert adapter.reports == 1


@pytest.mark.asyncio
async def test_a_refused_model_reply_does_not_latch_the_adapter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport = ManualDynamicCodexTransport()
    adapter = _Adapter(transport)
    inference = adapter.inference
    try:
        async with asyncio.timeout(5):
            refused = asyncio.create_task(_collect_stream(inference, "turn_refused"))
            await transport.turn_started.get()
            await transport.complete_turn(delta=_IMITATION)
            with pytest.raises(ModelOutputRefused):
                await refused
            assert adapter.reports == 0 and inference._closed is False

            # The same adapter serves the next turn.
            await adapter.healthy_turn("turn_after_refusal")
            assert adapter.reports == 0
    finally:
        await inference.close()
    assert adapter.reports == 0


@pytest.mark.asyncio
async def test_a_rejected_tool_call_does_not_report_a_host_wide_fault() -> None:
    # A model mistake that stays terminal for the binding (#478 item 1). It is not the
    # adapter's own latch, so it must not end the host.
    adapter = _Adapter(InvalidDynamicRequestCodexTransport())
    try:
        with pytest.raises(RuntimeError, match="reader failed"):
            async for _segment in adapter.inference.stream(
                _snapshot(), turn_id="turn_wrong_namespace"
            ):
                pass
        assert adapter.reports == 0
    finally:
        await adapter.inference.close()
    assert adapter.reports == 0


@pytest.mark.asyncio
async def test_the_adapters_own_close_is_not_a_fault_even_with_a_turn_in_flight() -> None:
    transport = ManualDynamicCodexTransport()
    adapter = _Adapter(transport)
    inference = adapter.inference
    async with asyncio.timeout(5):
        await adapter.healthy_turn()
        in_flight = asyncio.create_task(_collect_stream(inference, "turn_in_flight"))
        await transport.turn_started.get()
        await inference.close()
        await asyncio.gather(in_flight, return_exceptions=True)
        with pytest.raises(RuntimeError, match="Codex inference adapter is closed"):
            await _collect_stream(inference, "turn_after_close")
    assert adapter.reports == 0


def test_the_latch_observer_must_be_callable() -> None:
    with pytest.raises(TypeError, match="on_latched must be callable or None"):
        CodexAppServerStreamingInference(
            model="gpt-5.6-terra", effort="low",
            transport_factory=ManualDynamicCodexTransport,
            on_latched="not callable",  # type: ignore[arg-type]
        )
