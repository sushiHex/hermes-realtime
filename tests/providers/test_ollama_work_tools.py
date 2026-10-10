from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import deque
from typing import cast
from urllib.request import Request

import pytest

import hermes_realtime.providers.ollama as ollama
from hermes_realtime.conversation.context import ConversationContextSnapshot, ConversationMessage
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.work_tools import WorkStartResult


class Response:
    def __init__(self, lines: list[dict[str, object]]) -> None:
        self.lines = deque(json.dumps(line).encode() + b"\n" for line in lines)
        self.closed = False

    def readline(self, limit: int = -1, /) -> bytes:
        return self.lines.popleft() if self.lines and not self.closed else b""

    def close(self) -> None:
        self.closed = True


class Handler:
    max_objective_chars = 64

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.result = WorkStartResult(accepted=True, state="active", task_id="task_test")

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        self.calls.append((objective, invocation_id))
        self.entered.set()
        await self.release.wait()
        return self.result


def call(**changes: object) -> dict[str, object]:
    return {"function": {"name": "start_work", "arguments": {"objective": "Inspect release"}},
            **changes}


def request() -> ConversationInferenceRequest:
    return ConversationInferenceRequest(revision=1,
        messages=(ConversationMessage("user", "Please inspect the release."),), active_tasks=())


def adapter(lines: list[dict[str, object]], **options: object
            ) -> tuple[ollama.OllamaStreamingInference, Handler, list[dict[str, object]], Response]:
    sent: list[dict[str, object]] = []
    response = Response(lines)

    def open_request(req: Request, timeout: float) -> Response:
        sent.append(json.loads(cast(bytes, req.data)))
        return response

    inference = ollama.OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test",
                                              open_request=open_request, **options)
    handler = Handler()
    inference.bind_work_tools(handler)
    return inference, handler, sent, response


async def collect(inference: ollama.OllamaStreamingInference,
                  snapshot: ConversationContextSnapshot | None = None) -> list[str]:
    return [part async for part in inference.stream(snapshot or request(), turn_id="turn_test")]


@pytest.mark.asyncio
async def test_native_start_advertises_only_exact_start_and_waits_for_acceptance() -> None:
    inference, handler, sent, response = adapter([
        {"message": {"content": "I started already."}, "done": False},
        {"message": {"tool_calls": [call(id="local_call", type="function", function={
            "index": 0, "name": "start_work", "arguments": {"objective": "Inspect release"}})]},
         "done": True}])
    handler.release.clear()
    result = asyncio.create_task(collect(inference))
    await asyncio.wait_for(handler.entered.wait(), 1)
    assert not result.done()
    handler.release.set()
    assert await result == ["I've started that background task."]
    assert handler.calls == [("Inspect release", "ollama_work_" +
                             hashlib.sha256(b"turn_test").hexdigest())]
    assert response.closed
    body = sent[0]
    assert body["options"] == {"num_ctx": 16384, "num_predict": 1024}
    assert body["tools"] == [{"type": "function", "function": {
        "name": "start_work", "description": ollama._WORK_TOOL_DESCRIPTION,
        "parameters": {"type": "object", "additionalProperties": False,
            "required": ["objective"], "properties": {"objective": {
                "type": "string", "minLength": 1, "maxLength": 64}}}}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("result,spoken", [
    (WorkStartResult(accepted=False, state="rejected", reason="capacity exhausted"),
     "The background task was not accepted."),
    (WorkStartResult(accepted=True, state="cancelling", task_id="task_test"),
     "That background task was accepted and is now being cancelled.")])
async def test_rejected_and_pending_cancel_results_do_not_claim_active_work(result, spoken) -> None:
    inference, handler, _, _ = adapter([
        {"message": {"content": "I am working now.", "tool_calls": [call()]}, "done": True}])
    handler.result = result
    assert await collect(inference) == [spoken]


@pytest.mark.asyncio
async def test_normal_prose_retains_segmentation_without_dispatch() -> None:
    inference, handler, _, _ = adapter([
        {"message": {"content": "One sentence. "}, "done": False},
        {"message": {"content": "Another."}, "done": True}])
    assert await collect(inference) == ["One sentence.", "Another."]
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [
    ConversationContextSnapshot(revision=1, messages=request().messages, active_tasks=()),
    ConversationInferenceRequest(revision=1, messages=(), active_tasks=()),
    ConversationInferenceRequest(revision=1,
        messages=(ConversationMessage("assistant", "Update."),), active_tasks=())],
                         ids=["snapshot_only", "no_user", "assistant_last"])
async def test_without_current_user_request_no_work_schema_or_effect(snapshot) -> None:
    inference, handler, sent, _ = adapter([
        {"message": {"tool_calls": [call()]}, "done": True}])
    with pytest.raises(ValueError, match="unadvertised"):
        await collect(inference, snapshot)
    assert "tools" not in sent[0]
    assert handler.calls == []


_INVALID_CALLS = [
    (None, "calls_type"), ({}, "calls_type"), ([call(), call()], "call_count"),
    ([True], "call_schema"), ([call(extra=True)], "call_schema"),
    ([call(type="other")], "call_type"), ([call(id=1)], "call_id"),
    ([call(id="x" * 129)], "call_id"),
    ([call(function=True)], "function_schema"),
    ([call(function={"name": "start_work", "arguments": {"objective": "Inspect"},
                     "extra": True})], "function_schema"),
    ([call(function={"name": "start_work", "arguments": {"objective": "Inspect"},
                     "index": True})], "function_index"),
    ([call(function={"name": "start_work", "arguments": {"objective": "Inspect"},
                     "index": 1})], "function_index"),
    ([call(function={"name": "cancel_active_work", "arguments": {"objective": "Inspect"}})],
     "tool_name"),
    ([call(function={"name": "approve_work", "arguments": {"objective": "Inspect"}})],
     "tool_name"),
    ([call(function={"name": "start_work", "arguments": "{}"})], "arguments_schema"),
    ([call(function={"name": "start_work", "arguments": {"objective": "Inspect", "x": 1}})],
     "arguments_schema"),
    ([call(function={"name": "start_work", "arguments": {"objective": 1}})], "objective_type"),
    ([call(function={"name": "start_work", "arguments": {"objective": " "}})], "objective_blank"),
    ([call(function={"name": "start_work", "arguments": {"objective": "x" * 65}})],
     "objective_length"),
    ([call(function={"name": "start_work", "arguments": {"objective": "deleg_private"}})],
     "private_authority"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("calls,category", _INVALID_CALLS, ids=[x[1] for x in _INVALID_CALLS])
async def test_invalid_native_calls_refuse_before_effect_and_emit_content_free_evidence(
    calls, category, capsys
) -> None:
    inference, handler, _, response = adapter([
        {"message": {"content": "I already started.", "tool_calls": calls}, "done": True}])
    with pytest.raises(ValueError, match=category):
        await collect(inference)
    assert handler.calls == []
    assert response.closed
    line = next(line for line in capsys.readouterr().out.splitlines()
                if line.startswith("[ollama-work] "))
    assert json.loads(line.removeprefix("[ollama-work] "))["category"] == category
    assert "Inspect" not in line and "deleg_private" not in line and "turn_test" not in line


@pytest.mark.asyncio
async def test_second_streamed_call_refuses_whole_completion_before_dispatch() -> None:
    inference, handler, _, _ = adapter([
        {"message": {"tool_calls": [call()]}, "done": False},
        {"message": {"tool_calls": [call()]}, "done": True}])
    with pytest.raises(ValueError, match="call_count"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_missing_done_never_dispatches_partial_completion() -> None:
    inference, handler, _, _ = adapter([{"message": {"tool_calls": [call()]}, "done": False}])
    with pytest.raises(ValueError, match="incomplete"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["bytes", "chunks"])
async def test_decision_buffer_has_independent_byte_and_chunk_bounds(bound, monkeypatch) -> None:
    if bound == "bytes":
        monkeypatch.setattr(ollama, "_MAX_WORK_RESPONSE_BYTES", 20)
        lines = [{"message": {"content": "Long enough response."}, "done": True}]
    else:
        monkeypatch.setattr(ollama, "_MAX_WORK_RESPONSE_CHUNKS", 1)
        lines = [{"message": {"content": "a"}, "done": False},
                 {"message": {"content": "b"}, "done": True}]
    inference, handler, _, _ = adapter(lines)
    with pytest.raises(ValueError, match="response_" + bound):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_ack_timeout_is_unknown_and_never_claims_started(capsys) -> None:
    inference, handler, _, _ = adapter([
        {"message": {"tool_calls": [call()]}, "done": True}], request_timeout_seconds=0.05)
    handler.release.clear()
    started = time.monotonic()
    assert await collect(inference) == ["I couldn't confirm that background work started."]
    assert time.monotonic() - started < 0.12
    assert '[ollama-work] {"category":"ack_timeout","kind":"start","attempts":1}' in (
        capsys.readouterr().out)
    handler.release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_foreground_cancel_during_ack_does_not_cancel_background_work(capsys) -> None:
    inference, handler, _, _ = adapter([
        {"message": {"tool_calls": [call()]}, "done": True}])
    handler.release.clear()
    result = asyncio.create_task(collect(inference))
    await asyncio.wait_for(handler.entered.wait(), 1)
    await inference.cancel("turn_test")
    handler.release.set()
    with pytest.raises(RuntimeError, match="ownership|cancelled"):
        await result
    assert len(handler.calls) == 1
    assert '"category":"ownership_after_admission","kind":"start","attempts":1' in (
        capsys.readouterr().out)


@pytest.mark.asyncio
async def test_preflight_is_tool_less_until_handler_is_bound() -> None:
    response = Response([{"message": {"content": "Ready."}, "done": True}])
    sent = []

    def open_request(req, timeout):
        sent.append(json.loads(req.data))
        return response

    inference = ollama.OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test",
                                              open_request=open_request)
    assert await collect(inference) == ["Ready."]
    assert "tools" not in sent[0]
    inference.bind_work_tools(Handler())


@pytest.mark.parametrize("handler", [object(), type("Bad", (), {
    "start_work": lambda _: None, "max_objective_chars": True})(),
    type("Bad", (), {"start_work": lambda _: None, "max_objective_chars": 0})(),
    type("Bad", (), {"start_work": lambda _: None, "max_objective_chars": 65537})()])
def test_binding_requires_start_method_and_exact_supported_bound(handler) -> None:
    inference = ollama.OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test")
    with pytest.raises((TypeError, ValueError)):
        inference.bind_work_tools(handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["bound", "closed", "active", "opening"])
async def test_binding_requires_idle_open_and_unbound_adapter(state) -> None:
    inference = ollama.OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test")
    if state == "bound":
        inference.bind_work_tools(Handler())
    elif state == "closed":
        await inference.close()
    elif state == "active":
        inference._responses["turn_active"] = Response([])
    else:
        inference._openings["turn_opening"] = asyncio.create_task(
            asyncio.sleep(0, result=Response([])))
    with pytest.raises(RuntimeError, match="only once while idle and open"):
        inference.bind_work_tools(Handler())
    await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [object(), WorkStartResult(
    accepted=True, state="active", task_id="task_test")], ids=["wrong_type", "forged_state"])
async def test_untrusted_handler_result_is_reconstructively_validated(result) -> None:
    inference, handler, _, _ = adapter([
        {"message": {"content": "I started.", "tool_calls": [call()]}, "done": True}])
    if type(result) is WorkStartResult:
        object.__setattr__(result, "state", "rejected")
    handler.result = result
    with pytest.raises(ValueError, match="result_type|result_schema"):
        await collect(inference)


@pytest.mark.asyncio
async def test_handler_failure_is_unknown_and_never_speaks_model_start_claim(capsys) -> None:
    inference, handler, _, _ = adapter([
        {"message": {"content": "I started.", "tool_calls": [call()]}, "done": True}])

    async def fail(**kwargs):
        raise OSError("Synthetic unavailable transport")

    handler.start_work = fail
    assert await collect(inference) == ["I couldn't confirm that background work started."]
    assert '"category":"ack_unavailable","kind":"start","attempts":1' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_whole_decision_deadline_bounds_many_fast_individual_reads() -> None:
    inference, handler, _, response = adapter([
        {"message": {"content": "x"}, "done": False} for _ in range(10)],
        request_timeout_seconds=0.05)
    read = response.readline

    def slow_read(limit=-1):
        time.sleep(0.02)
        return read(limit)

    response.readline = slow_read
    with pytest.raises(ValueError, match="deadline"):
        await asyncio.wait_for(collect(inference), 0.5)
    assert handler.calls == []
    assert response.closed


@pytest.mark.asyncio
async def test_deadline_expired_before_admission_has_no_dispatch_effect() -> None:
    inference, handler, _, _ = adapter([
        {"message": {"tool_calls": [call()]}, "done": True}], request_timeout_seconds=0.05)

    def delay_report(messages, count):
        time.sleep(0.06)

    inference._report_prompt = delay_report
    with pytest.raises(ValueError, match="deadline"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_cancel_before_completed_decision_never_dispatches(capsys) -> None:
    inference, handler, _, response = adapter([
        {"message": {"tool_calls": [call()]}, "done": True}])
    entered = threading.Event()
    release = threading.Event()
    def waiting_read(limit=-1):
        entered.set()
        release.wait(1)
        # Return a valid completed call even after response closure. Authority,
        # rather than the HTTP object's cooperative behavior, must prevent dispatch.
        return json.dumps({"message": {"tool_calls": [call()]}, "done": True}).encode()

    def close():
        response.closed = True
        release.set()

    response.readline = waiting_read
    response.close = close
    result = asyncio.create_task(collect(inference))
    assert await asyncio.to_thread(entered.wait, 1)
    await inference.cancel("turn_test")
    with pytest.raises(RuntimeError, match="ownership|cancelled"):
        await result
    assert handler.calls == []
    marker = next(json.loads(line.removeprefix("[ollama-work] "))
                  for line in capsys.readouterr().out.splitlines()
                  if line.startswith("[ollama-work] "))
    assert marker["category"] in {"ownership_before_admission", "cancelled_before_admission"}
    assert marker == {"category": marker["category"], "kind": "start", "attempts": 0}


@pytest.mark.asyncio
async def test_cancel_after_local_enqueue_before_shared_handler_entry_preserves_owned_start(
    monkeypatch
) -> None:
    inference, handler, _, _ = adapter([
        {"message": {"tool_calls": [call()]}, "done": True}])
    release_entry = asyncio.Event()
    enqueued = asyncio.Event()
    start = handler.start_work

    async def held_start(**kwargs):
        await release_entry.wait()
        return await start(**kwargs)

    handler.start_work = held_start
    create_task = asyncio.create_task

    def observe_enqueue(coroutine, *, name=None, context=None):
        operation = create_task(coroutine, name=name, context=context)
        if name == "ollama-start-work":
            enqueued.set()
        return operation

    monkeypatch.setattr(asyncio, "create_task", observe_enqueue)
    result = asyncio.create_task(collect(inference))
    await asyncio.wait_for(enqueued.wait(), 1)
    assert not handler.calls
    await inference.cancel("turn_test")
    release_entry.set()
    with pytest.raises(RuntimeError, match="ownership|cancelled"):
        await result
    assert len(handler.calls) == 1

