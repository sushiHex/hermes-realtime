"""The foreground probe consumes the actual Codex adapter without a process or model."""

import asyncio
import io
import json

import pytest
from test_codex_app_server import FakeCodexTransport, ManualDynamicCodexTransport

from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.host_launcher import _bind_natural_work_tools
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from scripts import probe_confirmed_natural_work as probe


def setup(transport):
    context, request = probe.harness.make_context(
        probe.legacy.IntentCase("synthetic", "negative", "Research a synthetic topic", None),
        1, (), active=False,
    )
    recorder = probe.harness.Recorder()
    surface = ConversationWorkControlSurface(
        controller=recorder, context=context, observer=lambda *args: None,
        reserve_observer_capacity=lambda: None,
    )
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
        request_timeout_seconds=1,
    )
    _bind_natural_work_tools(inference=inference, surface=surface, enabled=True)
    return inference, surface, recorder, request


class DelayedInterruptTransport(ManualDynamicCodexTransport):
    def __init__(self):
        super().__init__()
        self.interrupt_started = asyncio.Event()
        self.release_interrupt = asyncio.Event()

    async def send(self, message):
        if message.get("method") == "turn/interrupt":
            self.interrupt_started.set()
            await self.release_interrupt.wait()
        await super().send(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,selection", [
    ("start_work", "proposal"), ("task_status", "task_status"),
])
async def test_typed_codex_decision_precedes_deliberate_interrupt_settlement(tool, selection):
    transport = DelayedInterruptTransport()
    inference, surface, recorder, request = setup(transport)
    clock = [10.0]
    row = {}
    task = asyncio.create_task(probe.collect_decision(
        inference, request, "synthetic", row, clock=lambda: clock[0],
    ))
    try:
        await asyncio.wait_for(transport.turn_started.get(), 1)
        clock[0] += 0.125
        arguments = {"objective": "Research a synthetic topic"} if tool == "start_work" else {}
        await transport.start_item(tool=tool, arguments=arguments)
        await transport.request_tool(request_id=91, tool=tool, arguments=arguments)
        response = await asyncio.wait_for(transport.tool_responses.get(), 1)
        await transport.complete_item(response, tool=tool, arguments=arguments)
        await asyncio.wait_for(transport.interrupt_started.wait(), 1)
        assert row["selection"] == selection
        assert row["first_usable_decision_ms"] == 125
        assert row["generator_elapsed_ms"] is None
        assert row["first_speakable_ms"] is None
        assert not task.done()
        clock[0] += 0.5
        transport.release_interrupt.set()
        await asyncio.wait_for(task, 1)
        assert row["generator_elapsed_ms"] == row["decision_elapsed_ms"] == 625
        assert row["speech"] == []
        assert recorder.objectives == []
        interrupts = [m for m in transport.sent if m.get("method") == "turn/interrupt"]
        assert len(interrupts) == 1
        assert interrupts[0]["params"] == {
            "threadId": transport.thread_id, "turnId": transport.turn_id,
        }
    finally:
        transport.release_interrupt.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await inference.close()
        await surface.close()
    assert transport.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unsolicited_interrupt", "provider_failure"])
async def test_actual_codex_failure_is_not_accepted_as_a_host_decision(failure):
    transport = ManualDynamicCodexTransport()
    inference, surface, recorder, request = setup(transport)
    row = {}
    task = asyncio.create_task(probe.collect_decision(inference, request, "synthetic", row))
    try:
        await asyncio.wait_for(transport.turn_started.get(), 1)
        if failure == "unsolicited_interrupt":
            await transport.complete_turn(status="interrupted")
        else:
            await transport.incoming.put({"method": "error", "params": {
                "threadId": transport.thread_id, "turnId": transport.turn_id,
                "willRetry": False, "error": {
                    "message": "SYNTHETIC_PRIVATE_account_response",
                    "codexErrorInfo": "unauthorized",
                },
            }})
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(task, 1)
        assert row["selection"] is None
        assert row["first_usable_decision_ms"] is None
        assert recorder.objectives == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await inference.close()
        await surface.close()
    assert transport.closed


@pytest.mark.asyncio
async def test_codex_rows_use_real_preflight_adapter_and_separate_owned_cleanup(
    monkeypatch, tmp_path,
):
    transports = []
    clock = [10.0]
    executable = tmp_path / "synthetic-codex.exe"
    executable.write_bytes(b"synthetic executable; never launched")

    def factory(**kwargs):
        assert kwargs["inference_provider"] == "codex"
        assert kwargs["codex_model"] == "gpt-5.6-terra"
        assert kwargs["codex_effort"] == "low"
        assert kwargs["codex_executable"] == str(executable)
        transport = FakeCodexTransport()
        transports.append(transport)
        inference = CodexAppServerStreamingInference(
            model=kwargs["codex_model"], effort=kwargs["codex_effort"],
            transport_factory=lambda: transport,
        )
        close = inference.close

        async def timed_close():
            clock[0] += 0.25
            await close()

        inference.close = timed_close
        return inference

    monkeypatch.setattr(probe, "factory", factory)
    receipt = io.BytesIO()
    rows = await probe.run_cases(
        inference_provider="codex", model="gpt-5.6-terra", codex_effort="low",
        codex_executable=str(executable), codex_executable_sha256=probe.harness.digest(executable),
        private_sink=receipt, clock=lambda: clock[0],
    )
    assert len(rows) == len(transports) == 8
    assert all(row["selection"] == "reply" and row["error"] is None for row in rows)
    assert all(row["cleanup_elapsed_ms"] == 250 for row in rows)
    assert all(row["first_usable_decision_ms"] == 0 for row in rows)
    assert all(transport.closed for transport in transports)
    assert all(sum(m.get("method") == "turn/start" for m in t.sent) == 2 for t in transports)
    report = probe.build_report(rows)
    assert report["dispatches"] == report["errors"] == 0
    assert "Four." not in json.dumps(report)
    assert str(tmp_path) not in json.dumps(report)


@pytest.mark.asyncio
async def test_codex_factory_cannot_substitute_an_unadmitted_provider(monkeypatch, tmp_path):
    executable = tmp_path / "synthetic-codex.exe"
    executable.write_bytes(b"synthetic executable; never launched")
    closed = []

    class WrongProvider:
        async def stream(self, request, turn_id):
            yield "Synthetic reply."

        async def close(self):
            closed.append("inference")

    async def preflight(**kwargs):
        pytest.fail("wrong provider reached readiness")

    monkeypatch.setattr(probe, "factory", lambda **kwargs: WrongProvider())
    monkeypatch.setattr(probe, "preflight", preflight)
    rows = []
    with pytest.raises(ValueError, match="provider_refused"):
        await probe.run_cases(
            inference_provider="codex", model="gpt-5.6-terra", codex_effort="low",
            codex_executable=str(executable),
            codex_executable_sha256=probe.harness.digest(executable),
            private_sink=io.BytesIO(), observations=rows,
        )
    assert closed == ["inference"]
    assert len(rows) == 1 and rows[0]["error"] == "attempt_failed"
    assert rows[0]["readiness_attempted"] is False
