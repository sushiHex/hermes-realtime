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
from hermes_realtime.providers.ollama_models import OllamaModelClient
from tests.support.ollama_metadata import supported_ollama_metadata

pytestmark = pytest.mark.usefixtures("deny_uninjected_ollama_metadata")


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


def decision_frame(value: dict[str, object], **completion: object) -> dict[str, object]:
    return {"message": {"content": json.dumps(value)}, "done": True,
            "done_reason": "stop", **completion}


def start_frame(**completion: object) -> dict[str, object]:
    return decision_frame({"kind": "start_work", "objective": "Inspect release"}, **completion)


def reply_schema() -> dict[str, object]:
    return {"type": "object", "additionalProperties": False, "required": ["kind", "text"],
            "properties": {"kind": {"const": "reply"}, "text": {
                "type": "string", "minLength": 1, "maxLength": 65536}}}


def start_schema() -> dict[str, object]:
    return {"type": "object", "additionalProperties": False, "required": ["kind", "objective"],
            "properties": {"kind": {"const": "start_work"}, "objective": {
                "type": "string", "minLength": 1, "maxLength": 64}}}


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

    options.setdefault("metadata_client", supported_ollama_metadata())
    inference = ollama.OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test",
                                              open_request=open_request, **options)
    handler = Handler()
    inference.bind_work_tools(handler)
    return inference, handler, sent, response


async def collect(inference: ollama.OllamaStreamingInference,
                  snapshot: ConversationContextSnapshot | None = None) -> list[str]:
    return [part async for part in inference.stream(snapshot or request(), turn_id="turn_test")]


@pytest.mark.asyncio
async def test_constrained_start_advertises_exact_union_and_waits_for_acceptance() -> None:
    decision = json.dumps({"kind": "start_work", "objective": "Inspect release"})
    inference, handler, sent, response = adapter([
        {"message": {"content": decision[:18]}, "done": False},
        {"message": {"content": decision[18:]}, "done": True, "done_reason": "stop"},
    ])
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
    assert body["shift"] is False
    assert body["truncate"] is False
    assert body["options"] == {"num_ctx": 16384, "num_predict": 1024}
    assert "tools" not in body
    assert body["format"] == {"oneOf": [reply_schema(), start_schema()]}



@pytest.mark.asyncio
async def test_constrained_start_preserves_selected_context_delivery_and_completion_timing(
    capsys,
) -> None:
    async def read_metadata(endpoint, body):
        if endpoint == "version":
            assert body is None
            return b'{"version":"0.34.4"}'
        if endpoint == "tags":
            assert body is None
            return json.dumps({"models": [{"name": "test", "model": "test",
                                            "digest": "a" * 64}]}).encode()
        assert (endpoint, body) == ("show", {"model": "test"})
        return b'{"capabilities":["completion"],"model_info":{}}'

    inference, handler, sent, _ = adapter([
        {**start_frame(), "prompt_eval_count": 321,
         "load_duration": 11, "prompt_eval_duration": 22, "eval_duration": 33,
         "total_duration": 66},
    ], metadata_client=OllamaModelClient(
        base_url="http://127.0.0.1:11434", read_request=read_metadata,
    ))
    await inference.configure_context(4096)
    snapshot = ConversationInferenceRequest(
        revision=2,
        messages=(ConversationMessage("assistant", "Heard prefix.", interrupted=True),
                  ConversationMessage("user", "Please inspect the release.")),
        active_tasks=(),
    )
    assert await collect(inference, snapshot) == ["I've started that background task."]
    assert len(handler.calls) == 1
    body = sent[0]
    assert body["options"] == {"num_ctx": 4096, "num_predict": 1024}
    assert body["shift"] is False and body["truncate"] is False
    assert "tools" not in body
    assert body["format"] == {"oneOf": [reply_schema(), start_schema()]}
    messages = body["messages"]
    assert messages[1:] == [
        {"role": message.role, "content": message.text} for message in snapshot.messages
    ]
    sections = json.loads(messages[0]["content"].split("\n\nReference data:\n", 1)[1])
    assert sections["speech_delivery"] == {"interrupted_message_indices": [0]}
    assert ollama._WORK_POLICY in messages[0]["content"]
    reports = [json.loads(line.removeprefix("[ollama-prompt] "))
               for line in capsys.readouterr().out.splitlines()
               if line.startswith("[ollama-prompt] ")]
    assert len(reports) == 1
    report = reports[0]
    assert report["num_ctx"] == 4096 and report["prompt_eval_count"] == 321
    assert report["reference_kinds"] == ["speech_delivery"]
    assert [report[field] for field in (
        "load_duration_ns", "prompt_eval_duration_ns", "eval_duration_ns", "total_duration_ns",
    )] == [11, 22, 33, 66]
    assert report["timing_invalid_count"] == 0 and report["timing_invalid_fields"] == []
    await inference.close()


@pytest.mark.asyncio
async def test_constrained_work_refuses_unverified_runtime_before_any_opening() -> None:
    async def read(endpoint, body):
        assert (endpoint, body) == ("version", None)
        return b'{"version":"unsupported"}'

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    inference, handler, sent, response = adapter([
        start_frame(),
    ], metadata_client=client)
    with pytest.raises(ValueError, match="source-reviewed"):
        await collect(inference)
    assert handler.calls == sent == []
    assert response.closed is False
    await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("result,spoken", [
    (WorkStartResult(accepted=False, state="rejected", reason="capacity exhausted"),
     "The background task was not accepted."),
    (WorkStartResult(accepted=True, state="cancelling", task_id="task_test"),
     "That background task was accepted and is now being cancelled.")])
async def test_rejected_and_pending_cancel_results_do_not_claim_active_work(result, spoken) -> None:
    inference, handler, _, _ = adapter([
        start_frame()])
    handler.result = result
    assert await collect(inference) == [spoken]


@pytest.mark.asyncio
async def test_structured_reply_retains_segmentation_without_dispatch() -> None:
    decision = json.dumps({"kind": "reply", "text": "One sentence. Another."})
    inference, handler, _, _ = adapter([
        {"message": {"content": decision[:22]}, "done": False},
        {"message": {"content": decision[22:]}, "done": True, "done_reason": "stop"},
    ])
    assert await collect(inference) == ["One sentence.", "Another."]
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [
    ["I've started that background task."],
    ["An ordinary sentence. I've sta", "rted that background task."],
    ["I\u2019ve started that background task."],
], ids=["single_chunk", "split_claim", "curly_apostrophe"])
async def test_no_call_background_claim_refuses_before_any_speech(chunks, capsys) -> None:
    decision = json.dumps({"kind": "reply", "text": "".join(chunks)}, ensure_ascii=False)
    boundary = decision.index("started") + 3
    inference, handler, _, response = adapter([
        {"message": {"content": decision[:boundary]}, "done": False},
        {"message": {"content": decision[boundary:]}, "done": True, "done_reason": "stop"},
    ])
    spoken = []
    with pytest.raises(ValueError, match="unverified_claim"):
        async for segment in inference.stream(request(), turn_id="turn_test"):
            spoken.append(segment)
    assert spoken == []
    assert handler.calls == []
    assert response.closed
    evidence = [json.loads(line.removeprefix("[ollama-work] "))
                for line in capsys.readouterr().out.splitlines()
                if line.startswith("[ollama-work] ")]
    assert evidence == [{"category": "unverified_claim", "kind": "start", "attempts": 0}]


@pytest.mark.asyncio
async def test_unbound_conversation_retains_existing_prose_behavior() -> None:
    response = Response([
        {"message": {"content": "I've started that background task"}, "done": True}
    ])
    inference = ollama.OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test",
        open_request=lambda request, timeout: response,
        metadata_client=supported_ollama_metadata(),
    )
    assert await collect(inference) == ["I've started that background task"]


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [
    ConversationContextSnapshot(revision=1, messages=request().messages, active_tasks=()),
    ConversationInferenceRequest(revision=1, messages=(), active_tasks=()),
    ConversationInferenceRequest(revision=1,
        messages=(ConversationMessage("assistant", "Update."),), active_tasks=())],
                         ids=["snapshot_only", "no_user", "assistant_last"])
async def test_without_current_user_request_no_work_schema_or_effect(snapshot) -> None:
    inference, handler, sent, _ = adapter([
        start_frame()])
    with pytest.raises(ValueError, match="unadvertised"):
        await collect(inference, snapshot)
    assert "tools" not in sent[0]
    assert sent[0]["format"] == reply_schema()
    assert handler.calls == []


_INVALID_DECISIONS = [
    ("not JSON", "decision_json"),
    ('```json\n{"kind":"start_work","objective":"Inspect"}\n```', "decision_json"),
    ('{"kind":"start_work","objective":"Inspect"} extra', "decision_json"),
    ('{"kind":"start_work","objective":"Inspect"}{"kind":"reply","text":"No"}',
     "decision_json"),
    ('{"kind":"reply","kind":"start_work","objective":"Inspect"}', "duplicate_keys"),
    ('{"kind":"start_work","objective":"Inspect","objective":"Changed"}', "duplicate_keys"),
    ('[]', "decision_schema"), ('true', "decision_schema"), ('null', "decision_schema"),
    ('{}', "decision_schema"), ('{"kind":true,"objective":"Inspect"}', "decision_schema"),
    ('{"kind":"cancel_work","objective":"Inspect"}', "decision_schema"),
    ('{"kind":"approve_work","objective":"Inspect"}', "decision_schema"),
    ('{"kind":"start_work","objective":"Inspect","text":"Started"}', "decision_schema"),
    ('{"kind":"start_work","objective":"Inspect","task_id":"task_other"}', "decision_schema"),
    ('{"kind":"reply","text":"Answer","objective":"Inspect"}', "decision_schema"),
    ('{"kind":"reply"}', "decision_schema"),
    ('{"kind":"start_work"}', "decision_schema"),
    ('{"kind":"reply","text":1}', "reply_type"),
    ('{"kind":"reply","text":true}', "reply_type"),
    ('{"kind":"reply","text":" "}', "reply_blank"),
    ('{"kind":"reply","text":"\\ud800"}', "reply_unicode"),
    ('{"kind":"start_work","objective":1}', "objective_type"),
    ('{"kind":"start_work","objective":true}', "objective_type"),
    ('{"kind":"start_work","objective":" "}', "objective_blank"),
    (json.dumps({"kind": "start_work", "objective": "x" * 65}), "objective_length"),
    ('{"kind":"start_work","objective":"\\udfff"}', "objective_unicode"),
    ('{"kind":"start_work","objective":"deleg_private"}', "private_authority"),
    ('NaN', "decision_json"),
    ('{"kind":"reply","text":NaN}', "decision_json"),
    ('["nested",' * 2000 + 'null' + ']' * 2000, "decision_json"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,category", _INVALID_DECISIONS,
    ids=[f"{index}-{item[1]}" for index, item in enumerate(_INVALID_DECISIONS)],
)
async def test_invalid_decisions_refuse_before_effect_and_emit_content_free_evidence(
    content, category, capsys,
) -> None:
    inference, handler, _, response = adapter([
        {"message": {"content": content}, "done": True, "done_reason": "stop"},
    ])
    spoken = []
    with pytest.raises(ValueError, match=category):
        async for segment in inference.stream(request(), turn_id="turn_test"):
            spoken.append(segment)
    assert handler.calls == spoken == []
    assert response.closed
    lines = [json.loads(line.removeprefix("[ollama-work] "))
             for line in capsys.readouterr().out.splitlines()
             if line.startswith("[ollama-work] ")]
    assert lines == [{"category": category, "kind": "start", "attempts": 0}]


@pytest.mark.asyncio
@pytest.mark.parametrize("calls,category", [
    (None, "calls_type"), ({}, "calls_type"), ([call()], "unexpected_native_calls"),
    ([call(), call()], "unexpected_native_calls"), ([True], "unexpected_native_calls"),
])
async def test_native_calls_never_authorize_constrained_decision(calls, category) -> None:
    frame = start_frame()
    frame["message"]["tool_calls"] = calls
    inference, handler, _, _ = adapter([frame])
    with pytest.raises(ValueError, match=category):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_multiple_streamed_decisions_refuse_before_dispatch() -> None:
    first = start_frame()
    first["done"] = False
    inference, handler, _, _ = adapter([first, start_frame()])
    with pytest.raises(ValueError, match="decision_json"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [
    ConversationContextSnapshot(revision=1, messages=request().messages, active_tasks=()),
    ConversationInferenceRequest(revision=1, messages=(), active_tasks=()),
    ConversationInferenceRequest(revision=1,
        messages=(ConversationMessage("assistant", "Update."),), active_tasks=()),
], ids=["snapshot_only", "no_user", "assistant_last"])
async def test_bound_reference_turns_use_reply_only_schema(snapshot) -> None:
    inference, handler, sent, _ = adapter([
        decision_frame({"kind": "reply", "text": "An ordinary answer."}),
    ])
    assert await collect(inference, snapshot) == ["An ordinary answer."]
    assert handler.calls == []
    assert sent[0]["format"] == reply_schema()
    assert "tools" not in sent[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "length", "future", 1, True])
@pytest.mark.parametrize("kind", ["reply", "start_work"])
async def test_bound_completion_requires_explicit_normal_stop(reason, kind, capsys) -> None:
    value = {"kind": kind, "text" if kind == "reply" else "objective": "Synthetic answer."}
    frame = decision_frame(value, done_reason=reason)
    if reason is None:
        frame.pop("done_reason")
    inference, handler, _, response = adapter([frame])
    spoken = []
    with pytest.raises(ValueError, match="completion_reason"):
        async for segment in inference.stream(request(), turn_id="turn_test"):
            spoken.append(segment)
    assert handler.calls == spoken == []
    assert response.closed
    lines = [json.loads(line.removeprefix("[ollama-work] "))
             for line in capsys.readouterr().out.splitlines()
             if line.startswith("[ollama-work] ")]
    assert lines == [{"category": "completion_reason", "kind": "start", "attempts": 0}]


@pytest.mark.asyncio
async def test_reply_field_bound_is_independent_of_wire_byte_bound(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "_MAX_WORK_RESPONSE_BYTES", 262144)
    inference, handler, _, _ = adapter([
        decision_frame({"kind": "reply", "text": "x" * 65537}),
    ])
    with pytest.raises(ValueError, match="reply_length"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["reply", "start_work"])
async def test_valid_unicode_decisions_preserve_exact_string(kind) -> None:
    text = "Inspect café 漢字 U0001f600."
    key = "text" if kind == "reply" else "objective"
    frame = decision_frame({"kind": kind, key: text})
    frame["message"]["tool_calls"] = []
    inference, handler, _, _ = adapter([frame])
    output = await collect(inference)
    if kind == "reply":
        assert output == [text] and handler.calls == []
    else:
        assert output == ["I've started that background task."]
        assert handler.calls == [(text, "ollama_work_" + hashlib.sha256(b"turn_test").hexdigest())]


@pytest.mark.asyncio
async def test_missing_done_never_dispatches_partial_completion() -> None:
    inference, handler, _, _ = adapter([{**start_frame(), "done": False}])
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
        start_frame()], request_timeout_seconds=0.05)
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
        start_frame()])
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
                                              open_request=open_request,
                                              metadata_client=supported_ollama_metadata())
    assert await collect(inference) == ["Ready."]
    assert "tools" not in sent[0]
    inference.bind_work_tools(Handler())
    with pytest.raises(RuntimeError, match="before the first inference"):
        await inference.configure_context(4096)


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
        start_frame()])
    if type(result) is WorkStartResult:
        object.__setattr__(result, "state", "rejected")
    handler.result = result
    with pytest.raises(ValueError, match="result_type|result_schema"):
        await collect(inference)


@pytest.mark.asyncio
async def test_handler_failure_is_unknown_and_never_speaks_model_start_claim(capsys) -> None:
    inference, handler, _, _ = adapter([
        start_frame()])

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
@pytest.mark.parametrize("kind", ["start_work", "reply"])
async def test_deadline_expired_before_admission_has_no_dispatch_effect(kind) -> None:
    value = {"kind": kind, "text" if kind == "reply" else "objective": "Inspect release"}
    inference, handler, _, _ = adapter([
        decision_frame(value)], request_timeout_seconds=0.05)

    def delay_report(messages, count, completion):
        time.sleep(0.06)

    inference._report_prompt = delay_report
    with pytest.raises(ValueError, match="deadline"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["start_work", "reply"])
async def test_cancel_before_completed_decision_never_dispatches(kind, capsys) -> None:
    frame = decision_frame({
        "kind": kind, "text" if kind == "reply" else "objective": "Inspect release",
    })
    inference, handler, _, response = adapter([
        frame])
    entered = threading.Event()
    release = threading.Event()
    def waiting_read(limit=-1):
        entered.set()
        release.wait(1)
        # Return a valid completed call even after response closure. Authority,
        # rather than the HTTP object's cooperative behavior, must prevent dispatch.
        return json.dumps(frame).encode()

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
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("tail", ["Second sentence.", "Trailing without punctuation"])
async def test_cancel_between_reply_segments_suppresses_resumed_output(bound, tail, capsys) -> None:
    text = "First sentence. " + tail
    frame = decision_frame({"kind": "reply", "text": text})
    if not bound:
        frame["message"]["content"] = text
    inference, handler, _, _ = adapter([frame])
    if not bound:
        inference._work_tool_handler = None
    stream = inference.stream(request(), turn_id="turn_test")
    assert await anext(stream) == "First sentence."
    await inference.cancel("turn_test")
    with pytest.raises(RuntimeError, match="ownership|cancelled"):
        await anext(stream)
    assert handler.calls == []
    await stream.aclose()
    prefix = "[ollama-work] " if bound else "[ollama-stream] "
    evidence = [json.loads(line.removeprefix(prefix))
                for line in capsys.readouterr().out.splitlines() if line.startswith(prefix)]
    assert evidence == [{"category": "ownership_after_admission", "kind": "start", "attempts": 0}
                        if bound else {"category": "ownership_before_yield"}]


@pytest.mark.asyncio
async def test_reply_playback_drain_does_not_reopen_decision_deadline() -> None:
    inference, handler, _, _ = adapter([
        decision_frame({"kind": "reply", "text": "First sentence. Second sentence."}),
    ], request_timeout_seconds=0.05)
    stream = inference.stream(request(), turn_id="turn_test")
    assert await anext(stream) == "First sentence."
    await asyncio.sleep(0.06)
    assert await anext(stream) == "Second sentence."
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_cancel_after_local_enqueue_before_shared_handler_entry_preserves_owned_start(
    monkeypatch
) -> None:
    inference, handler, _, _ = adapter([
        start_frame()])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["closed", "cancelled", "ownership"])
@pytest.mark.parametrize("kind", ["start_work", "reply"])
async def test_each_before_admission_ownership_guard_blocks_dispatch_independently(
    state, kind,
) -> None:
    frame = decision_frame({
        "kind": kind, "text" if kind == "reply" else "objective": "Inspect release",
    })
    inference, handler, _, _ = adapter([
        frame])

    def retire_after_read(messages, count, completion):
        if state == "closed":
            inference._closed = True
        elif state == "cancelled":
            inference._cancelled_turns.add("turn_test")
        else:
            inference._responses.pop("turn_test")

    inference._report_prompt = retire_after_read
    with pytest.raises(RuntimeError, match="before_admission"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["closed", "cancelled"])
async def test_each_after_admission_cancellation_guard_suppresses_speech_independently(
    state
) -> None:
    inference, handler, _, _ = adapter([
        start_frame()])
    start = handler.start_work

    async def retire_after_start(**kwargs):
        result = await start(**kwargs)
        if state == "closed":
            inference._closed = True
        else:
            inference._cancelled_turns.add("turn_test")
        return result

    handler.start_work = retire_after_start
    with pytest.raises(RuntimeError, match="after_admission"):
        await collect(inference)
    assert len(handler.calls) == 1

