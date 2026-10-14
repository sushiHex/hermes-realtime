from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import cast
from urllib.request import Request

import pytest

import hermes_realtime.providers.ollama as ollama
from hermes_realtime.conversation.context import (
    ActiveTaskSummary,
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
)
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.work_tools import WorkStartResult
from hermes_realtime.providers.ollama_models import OllamaModelClient
from hermes_realtime.speech import Transcript
from tests.support.ollama_metadata import supported_ollama_metadata

pytestmark = pytest.mark.usefixtures("deny_uninjected_ollama_metadata")

_OBJECTIVE_CONTEXT_CONTRACT = (
    "Background work receives the objective, not this conversation. Make the objective "
    "self-contained: carry over the target, material baseline and constraints from available "
    "context, or identify a concrete resource Hermes can access to discover them. Do not "
    "assume a workspace, account or baseline merely because the user refers to one. If a "
    "required detail is neither supplied nor discoverable from an identified resource, ask "
    "one clarifying question instead of starting work."
)
_CONTEXT_CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "natural_work_context_contract_cases.json")
    .read_text(encoding="utf-8")
)["cases"]


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
async def test_ollama_request_carries_lifecycle_framing_without_granting_authority() -> None:
    """Witness policy propagation, not semantic obedience by a real model."""
    inference, handler, sent, _response = adapter([
        decision_frame({"kind": "reply", "text": "Use the task's Cancel control."}),
    ])
    snapshot = ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage("user", "Please stop the background research task."),),
        active_tasks=(ActiveTaskSummary("task_unrelated", "Unrelated background research"),),
    )
    try:
        assert await collect(inference, snapshot) == ["Use the task's Cancel control."]
    finally:
        await inference.close()
    instructions = sent[0]["messages"][0]["content"]
    framing = instructions.split("from a style or reference text. ", 1)[1].split(
        "\n\nResponse presentation:\n", 1,
    )[0]
    assert framing == (
        "Task cancellation is controlled only by explicit user commands or trusted task controls. "
        "Never cancel tasks or grant approvals yourself. For a request to stop background work, "
        "direct the user to the task's Cancel control. Do not claim work stopped, was cancelled, "
        "was approved, or completed merely because the user requested it; report lifecycle "
        "changes only from trusted acknowledged task state."
    )
    assert handler.calls == []
    assert sent[0]["format"] == {"oneOf": [reply_schema(), start_schema()]}
    assert "tools" not in sent[0]


@pytest.mark.asyncio
async def test_ollama_outbound_work_request_carries_objective_context_contract() -> None:
    """Actual request witness; model obedience requires separate qualification."""
    inference, handler, sent, _ = adapter([
        decision_frame({"kind": "reply", "text": "Which baseline should I use?"}),
    ])
    try:
        assert await collect(inference) == ["Which baseline should I use?"]
    finally:
        await inference.close()
    instructions = sent[0]["messages"][0]["content"]
    assert instructions.count(_OBJECTIVE_CONTEXT_CONTRACT) == 1
    assert handler.calls == []
    assert sent[0]["format"] == {"oneOf": [reply_schema(), start_schema()]}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CONTEXT_CASES, ids=lambda case: case["id"])
async def test_ollama_outbound_work_request_retains_actual_user_context(case) -> None:
    store = ConversationContextStore()
    for text in [*case["prior_user_turns"], case["utterance"]]:
        store.record_user_transcript(Transcript(text=text, final=True))
    source = store.snapshot()
    snapshot = ConversationInferenceRequest(
        revision=source.revision,
        messages=source.messages,
        active_tasks=source.active_tasks,
        memory=source.memory,
    )
    inference, handler, sent, _ = adapter([
        decision_frame({"kind": "reply", "text": "Synthetic answer."}),
    ])
    try:
        assert await collect(inference, snapshot) == ["Synthetic answer."]
    finally:
        await inference.close()
    assert sent[0]["messages"][1:] == [
        {"role": "user", "content": text}
        for text in [*case["prior_user_turns"], case["utterance"]]
    ]
    assert handler.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [False, True], ids=["ordinary", "work"])
@pytest.mark.parametrize("phase", ["open", "read"])
async def test_transport_timeout_preserves_authority_scope(bound, phase, capsys) -> None:
    inference, handler, _, response = adapter([])
    if not bound:
        inference._work_tool_handler = None
    error = TimeoutError("synthetic transport timeout")

    def fail(*args):
        raise error

    if phase == "open":
        inference._open_request = fail
    else:
        response.readline = fail
    try:
        with pytest.raises(ValueError if bound else TimeoutError) as caught:
            await collect(inference)
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        if bound:
            assert str(caught.value) == "Ollama work refused: deadline"
            assert markers == [
                '[ollama-work] {"category":"deadline","kind":"start","attempts":0}',
            ]
        else:
            assert type(caught.value) is TimeoutError
            assert caught.value.args == error.args
            assert markers == []
        assert handler.calls == []
        if phase == "read":
            assert response.closed
    finally:
        await inference.close()


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
@pytest.mark.parametrize("reason", [
    "capacity exhausted", "matching work is already active",
    "work control unavailable; restart required",
])
async def test_rejected_ack_reports_one_bounded_marker_before_refusal_speech(
    reason, capsys,
) -> None:
    inference, handler, _, response = adapter([start_frame()])
    handler.result = WorkStartResult(accepted=False, state="rejected", reason=reason)
    stream = inference.stream(request(), turn_id="turn_test")
    try:
        assert await anext(stream) == "The background task was not accepted."
        markers = [json.loads(line.removeprefix("[ollama-work] "))
                   for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        assert markers == [{"category": "ack_rejected", "kind": "start", "attempts": 1}]
        assert len(handler.calls) == 1
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert response.closed
        assert not any(line.startswith("[ollama-work] ")
                       for line in capsys.readouterr().out.splitlines())
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["active", "cancelling"])
async def test_accepted_ack_has_no_rejection_marker(state, capsys) -> None:
    inference, handler, _, _ = adapter([start_frame()])
    handler.result = WorkStartResult(accepted=True, state=state, task_id="task_test")
    assert await collect(inference)
    assert len(handler.calls) == 1
    assert not any(line.startswith("[ollama-work] ")
                   for line in capsys.readouterr().out.splitlines())


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
    monkeypatch.setattr(ollama, "_MAX_WORK_DECISION_BYTES", 262144)
    inference, handler, _, _ = adapter([
        decision_frame({"kind": "reply", "text": "x" * 65537}),
    ])
    with pytest.raises(ValueError, match="reply_length"):
        await collect(inference)
    assert handler.calls == []


@pytest.mark.asyncio
async def test_envelope_only_prefix_does_not_consume_decision_budget() -> None:
    prefix = {"model": "synthetic-model", "created_at": "2026-10-13T00:00:00Z",
              "message": {"content": "", "thinking": "Synthetic reasoning."}, "done": False}
    inference, handler, _, response = adapter([prefix] * 500 + [start_frame()])
    assert sum(len(line) for line in response.lines) > 65536
    assert await collect(inference) == ["I've started that background task."]
    assert len(handler.calls) == 1
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("text_kind", ["ascii", "unicode"])
@pytest.mark.parametrize("overflow", [False, True], ids=["exact", "overflow"])
async def test_decision_utf8_byte_boundary(text_kind, overflow, capsys) -> None:
    overhead = len(json.dumps({"kind": "reply", "text": ""}, separators=(",", ":")).encode())
    remaining = 65536 - overhead
    text = "x" * remaining if text_kind == "ascii" else (
        "é" * (remaining // 2) + "x" * (remaining % 2))
    if overflow:
        text += "x"
    decision = json.dumps({"kind": "reply", "text": text}, ensure_ascii=False,
                          separators=(",", ":"))
    assert len(decision.encode()) == 65536 + int(overflow)
    split = len(decision) // 2
    inference, handler, _, response = adapter([
        {"message": {"content": decision[:split]}, "done": False},
        {"message": {"content": decision[split:]}, "done": True, "done_reason": "stop"},
    ])
    if overflow:
        with pytest.raises(ValueError, match="decision_bytes"):
            await collect(inference)
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        assert markers == [
            '[ollama-work] {"category":"decision_bytes","kind":"start","attempts":0}',
        ]
    else:
        assert "".join(await collect(inference)) == text
    assert handler.calls == []
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [False, True], ids=["exact", "overflow"])
async def test_total_wire_byte_boundary(overflow, capsys) -> None:
    prefix = {"message": {"content": "", "thinking": ""}, "done": False,
              "envelope_padding": ""}
    frame = decision_frame({"kind": "reply", "text": "Synthetic answer."})
    frame["envelope_padding"] = ""
    wire_size = sum(len(json.dumps(row).encode()) + 1 for row in (prefix, frame))
    remaining = 1048576 - wire_size
    prefix["envelope_padding"] = "x" * (remaining // 2)
    frame["envelope_padding"] = "x" * (remaining - remaining // 2 + int(overflow))
    inference, handler, _, response = adapter([prefix, frame])
    assert sum(len(line) for line in response.lines) == 1048576 + int(overflow)
    if overflow:
        with pytest.raises(ValueError, match="response_bytes"):
            await collect(inference)
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        assert markers == [
            '[ollama-work] {"category":"response_bytes","kind":"start","attempts":0}',
        ]
    else:
        assert await collect(inference) == ["Synthetic answer."]
    assert handler.calls == []
    assert response.closed


@pytest.mark.asyncio
async def test_oversized_decision_refuses_before_buffer_append(monkeypatch) -> None:
    class ObserveAppend(str):
        def __radd__(self, other):
            raise AssertionError("oversized decision was appended")

    inference, handler, _, response = adapter([
        {"message": {"content": "x" * 65537}, "done": True, "done_reason": "stop"},
    ])
    original = inference._content

    def observe_validated_content(payload):
        content, done = original(payload)
        return ObserveAppend(content), done

    monkeypatch.setattr(inference, "_content", observe_validated_content)
    with pytest.raises(ValueError, match="decision_bytes"):
        await collect(inference)
    assert handler.calls == []
    assert response.closed


@pytest.mark.asyncio
async def test_decision_content_requires_strict_utf8_before_buffering(capsys) -> None:
    inference, handler, _, response = adapter([
        {"message": {"content": "\ud800"}, "done": True, "done_reason": "stop"},
    ])
    with pytest.raises(ValueError, match="decision_unicode"):
        await collect(inference)
    markers = [line for line in capsys.readouterr().out.splitlines()
               if line.startswith("[ollama-work] ")]
    assert markers == [
        '[ollama-work] {"category":"decision_unicode","kind":"start","attempts":0}',
    ]
    assert handler.calls == []
    assert response.closed


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
        lines = [{"message": {"content": "", "thinking": "Synthetic reasoning."}, "done": False},
                 start_frame()]
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
async def test_preflight_is_tool_less_until_handler_is_bound(capsys) -> None:
    response = Response([{"message": {"content": "Ready."}, "done": True}])
    sent = []

    def open_request(req, timeout):
        sent.append(json.loads(req.data))
        return response

    inference = ollama.OllamaStreamingInference(base_url="http://127.0.0.1:11434", model="test",
                                              open_request=open_request,
                                              metadata_client=supported_ollama_metadata())
    assert [part async for part in inference.stream(request(), turn_id="turn_preflight")] == [
        "Ready.",
    ]
    assert "tools" not in sent[0]
    inference.bind_work_tools(Handler())
    assert _OBJECTIVE_CONTEXT_CONTRACT not in sent[0]["messages"][0]["content"]
    with pytest.raises(RuntimeError, match="before the first inference"):
        await inference.configure_context(4096)
    assert not any(line.startswith("[ollama-work] ")
                   for line in capsys.readouterr().out.splitlines())


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_id", ["turn_user", "turn_preflight_extra"])
async def test_completed_user_turn_prevents_late_work_binding(turn_id, capsys) -> None:
    inference, _, _, _ = adapter([{"message": {"content": "Ready."}, "done": True}])
    inference._work_tool_handler = None
    try:
        assert [part async for part in inference.stream(request(), turn_id=turn_id)] == ["Ready."]
        with pytest.raises(RuntimeError, match="before user-turn admission"):
            inference.bind_work_tools(Handler())
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        assert markers == [
            '[ollama-work] {"category":"binding_after_admission","kind":"start","attempts":0}',
        ]
        assert inference._work_tool_handler is None
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gap", ["state_lock", "context_contract"])
async def test_user_turn_prevents_binding_before_first_request(gap, capsys) -> None:
    inference, _, sent, _ = adapter([{"message": {"content": "Ready."}, "done": True}])
    inference._work_tool_handler = None
    entered, release = asyncio.Event(), asyncio.Event()
    original = inference._ensure_context_contract_locked

    async def waiting_contract():
        entered.set()
        await release.wait()
        return await original()

    inference._ensure_context_contract_locked = waiting_contract
    await inference._state_lock.acquire()
    task = asyncio.create_task(collect(inference))
    try:
        await asyncio.sleep(0)
        if gap == "context_contract":
            inference._state_lock.release()
            await asyncio.wait_for(entered.wait(), 1)
        assert sent == []
        with pytest.raises(RuntimeError, match="before user-turn admission"):
            inference.bind_work_tools(Handler())
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        assert markers == [
            '[ollama-work] {"category":"binding_after_admission","kind":"start","attempts":0}',
        ]
        assert inference._work_tool_handler is None
    finally:
        if gap == "state_lock":
            inference._state_lock.release()
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        await inference.close()
    assert task.result() == ["Ready."]
    assert "format" not in sent[0]


@pytest.mark.asyncio
async def test_preflight_after_user_turn_does_not_reopen_work_binding(capsys) -> None:
    response = Response([{"message": {"content": "Ready."}, "done": True}])
    inference = ollama.OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test",
        metadata_client=supported_ollama_metadata(),
        open_request=lambda req, timeout: response,
    )
    try:
        assert await collect(inference) == ["Ready."]
        inference._open_request = lambda req, timeout: Response([
            {"message": {"content": "Ready."}, "done": True},
        ])
        assert [part async for part in inference.stream(request(), turn_id="turn_preflight")]
        with pytest.raises(RuntimeError, match="before user-turn admission"):
            inference.bind_work_tools(Handler())
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[ollama-work] ")]
        assert markers == [
            '[ollama-work] {"category":"binding_after_admission","kind":"start","attempts":0}',
        ]
    finally:
        await inference.close()


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
async def test_binding_requires_idle_open_and_unbound_adapter(state, capsys) -> None:
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
    try:
        with pytest.raises(RuntimeError, match="only once while idle and open"):
            inference.bind_work_tools(Handler())
    finally:
        await inference.close()
    markers = [line for line in capsys.readouterr().out.splitlines()
               if line.startswith("[ollama-work] ")]
    assert markers == [
        '[ollama-work] {"category":"binding_state","kind":"start","attempts":0}',
    ]


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

