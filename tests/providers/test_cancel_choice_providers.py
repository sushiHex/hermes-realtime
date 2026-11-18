"""Confirmed cancel choices in both providers (#237); no model process is launched.

With several frozen running tasks, cancel_work takes exactly one offered label; its
selection is a terminal host item naming only that label, and the labelled task list is
rendered as reference data. A nomination never cancels anything itself: the host confirms.
"""

import asyncio
import json

import pytest
from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler
from test_ollama_work_tools import adapter, collect, decision_frame

from hermes_realtime.conversation.context import ActiveTaskSummary, ConversationMessage
from hermes_realtime.conversation.natural_cancel import CancelChoice, HostCancelRequest
from hermes_realtime.conversation.streaming import ConversationInferenceRequest, ModelOutputRefused
from hermes_realtime.providers import codex_app_server as codex

_TASKS = (
    ActiveTaskSummary("task_alpha", "Inspect alpha"),
    ActiveTaskSummary("task_beta", "Inspect beta"),
)
_CHOICES = (CancelChoice("choice_1", "Inspect alpha"), CancelChoice("choice_2", "Inspect beta"))
_REFERENCE = [
    {"choice": "choice_1", "task": "Inspect alpha"},
    {"choice": "choice_2", "task": "Inspect beta"},
]


def _request() -> ConversationInferenceRequest:
    return ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage("user", "Stop the beta one."),),
        active_tasks=_TASKS,
        cancel_choices=_CHOICES,
    )


def test_request_choices_must_label_exactly_the_active_tasks_in_order() -> None:
    for choices in (
        tuple(reversed(_CHOICES)),
        _CHOICES[:1],
        (_CHOICES[0], CancelChoice("choice_2", "Inspect gamma")),
    ):
        with pytest.raises(ValueError):
            ConversationInferenceRequest(
                revision=1, messages=(ConversationMessage("user", "Stop one."),),
                active_tasks=_TASKS, cancel_choices=choices,
            )
    with pytest.raises(ValueError):
        ConversationInferenceRequest(
            revision=1, messages=(ConversationMessage("user", "Stop one."),),
            active_tasks=_TASKS, cancel_choices=_CHOICES, cancel_offered=True,
        )


# ------------------------------------------------------------------------------ codex


@pytest.mark.asyncio
async def test_codex_nomination_is_a_terminal_host_item_naming_only_the_label() -> None:
    transport = DynamicStartCodexTransport(
        tool="cancel_work", arguments={"target": "choice_2"},
        assistant_delta="The beta task has been cancelled.",
    )
    handler = FakeWorkToolHandler(can_cancel_work=True)
    inference = codex.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)
    try:
        async with asyncio.timeout(1):
            assert [
                part async for part in inference.stream(_request(), turn_id="turn_c")
            ] == [HostCancelRequest(target="choice_2")]
        assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()
    thread = next(row for row in transport.sent if row.get("method") == "thread/start")
    tools = {
        tool["name"]: tool
        for namespace in thread["params"]["dynamicTools"] for tool in namespace["tools"]
    }
    assert tools["cancel_work"]["inputSchema"] == {
        "type": "object", "additionalProperties": False, "required": ["target"],
        "properties": {"target": {"type": "string", "enum": ["choice_1", "choice_2"]}},
    }
    turn = next(row for row in transport.sent if row.get("method") == "turn/start")
    payload = json.loads(turn["params"]["input"][0]["text"].rsplit("\n", 1)[1])
    assert payload["cancel_choices"] == _REFERENCE
    # Task identities stay with the host.
    assert "task_alpha" not in json.dumps(thread) + json.dumps(turn)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [{}, {"target": "choice_3"}, {"target": "task_beta"}, {"target": "all"},
     {"target": ["choice_2"]}, {"target": "choice_2", "reason": "now"}],
    ids=["missing", "unoffered", "raw_task", "all", "list", "extra"],
)
async def test_codex_foreign_nomination_has_no_effect(arguments: dict[str, object]) -> None:
    transport = DynamicStartCodexTransport(
        tool="cancel_work", arguments=arguments, assistant_delta="Cancelled.",
    )
    handler = FakeWorkToolHandler(can_cancel_work=True)
    inference = codex.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)
    try:
        with pytest.raises(RuntimeError):
            async with asyncio.timeout(1):
                async for part in inference.stream(_request(), turn_id="turn_c"):
                    assert type(part) is not HostCancelRequest
        assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()


# ----------------------------------------------------------------------------- ollama


@pytest.mark.asyncio
async def test_ollama_nomination_is_a_terminal_host_item_naming_only_the_label() -> None:
    inference, handler, sent, _ = adapter(
        [decision_frame({"kind": "cancel_work", "target": "choice_2"})],
    )
    try:
        assert await collect(inference, _request()) == [HostCancelRequest(target="choice_2")]
        assert handler.calls == []
    finally:
        await inference.close()
    assert {
        "type": "object", "additionalProperties": False, "required": ["kind", "target"],
        "properties": {"kind": {"const": "cancel_work"},
                       "target": {"enum": ["choice_1", "choice_2"]}},
    } in sent[0]["format"]["oneOf"]
    system = sent[0]["messages"][0]["content"]
    reference = json.loads(system.split("\n\nReference data:\n", 1)[1])
    assert reference["cancel_choices"] == _REFERENCE
    assert "task_alpha" not in json.dumps(sent[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [{"kind": "cancel_work"}, {"kind": "cancel_work", "target": "choice_3"},
     {"kind": "cancel_work", "target": "task_beta"}, {"kind": "cancel_work", "target": "all"},
     {"kind": "cancel_work", "target": ["choice_2"]},
     {"kind": "cancel_work", "target": "choice_2", "text": "Stopped."}],
    ids=["missing", "unoffered", "raw_task", "all", "list", "extra"],
)
async def test_ollama_foreign_nomination_is_refused(value: dict[str, object]) -> None:
    inference, handler, _sent, _ = adapter([decision_frame(value)])
    try:
        with pytest.raises(ModelOutputRefused, match="decision_schema"):
            await collect(inference, _request())
        assert handler.calls == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_ollama_duplicate_target_keys_are_refused() -> None:
    inference, handler, _sent, _ = adapter([{
        "message": {"content": '{"kind":"cancel_work","target":"choice_1","target":"choice_2"}'},
        "done": True, "done_reason": "stop",
    }])
    try:
        with pytest.raises(ModelOutputRefused, match="duplicate_keys"):
            await collect(inference, _request())
        assert handler.calls == []
    finally:
        await inference.close()
