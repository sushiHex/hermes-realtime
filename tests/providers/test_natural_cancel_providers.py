"""The natural cancel route in both providers (#237); no model process is launched.

The route is advertised only on an admitted user turn whose request carries a host-frozen
cancel target. Its selection carries no task and returns a terminal host item; the
Cancel-control fallback and the route are never rendered together (#317).
"""

import asyncio
import json

import pytest
from test_codex_app_server import (
    DynamicStartCodexTransport,
    FakeCodexTransport,
    FakeWorkToolHandler,
)
from test_ollama_work_tools import adapter, collect, decision_frame

from hermes_realtime.conversation.context import ActiveTaskSummary, ConversationMessage
from hermes_realtime.conversation.natural_cancel import HostCancelRequest
from hermes_realtime.conversation.output_style import communication_policy
from hermes_realtime.conversation.streaming import (
    AcceptedCancelInferenceRequest,
    ConversationInferenceRequest,
    ModelOutputRefused,
)
from hermes_realtime.conversation.work_tools import (
    WORK_ROUTING_POLICY,
    WORK_ROUTING_POLICY_WITH_CANCEL,
)
from hermes_realtime.providers import codex_app_server as codex
from hermes_realtime.providers._call_imitation import ADVERTISED_TOOL_NAMES

# The fallback sentences that must never appear beside the advertised route.
_FALLBACKS = (
    "direct the user to the task's Cancel control",
    "For cancellation or approval, direct the user to the controls",
    "Never cancel tasks",
)
_ROUTE = "select cancel_work"


def _request(*, offered: bool) -> ConversationInferenceRequest:
    return ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage("user", "Stop that, please."),),
        active_tasks=(ActiveTaskSummary("task_public", "Synthetic active work"),),
        cancel_offered=offered,
    )


def _cancel_ack() -> AcceptedCancelInferenceRequest:
    return AcceptedCancelInferenceRequest(
        revision=1,
        messages=(ConversationMessage("user", "Stop that, please."),),
        active_tasks=(ActiveTaskSummary("task_public", "Synthetic active work"),),
        task_id="task_public",
    )


def _assert_exactly_one_cancel_instruction(text: str, *, offered: bool) -> None:
    # Mutation: render the fallback sentence beside the route.
    if offered:
        assert _ROUTE in text
        assert not any(fallback in text for fallback in _FALLBACKS)
    else:
        assert _ROUTE not in text and "cancel_work" not in text
        assert all(fallback in text for fallback in _FALLBACKS[:2])


def test_policies_render_the_route_or_the_fallback_never_both() -> None:
    for offered in (False, True):
        text = communication_policy(
            task_status_available=True, task_cancel_available=offered,
        ) + (WORK_ROUTING_POLICY_WITH_CANCEL if offered else WORK_ROUTING_POLICY)
        _assert_exactly_one_cancel_instruction(text, offered=offered)
    with pytest.raises(ValueError, match="only with the status route"):
        communication_policy(task_cancel_available=True)
    with pytest.raises(TypeError, match="task_cancel_available"):
        communication_policy(task_status_available=True, task_cancel_available=1)  # type: ignore[arg-type]


def test_cancel_route_is_a_guarded_tool_name() -> None:
    assert "cancel_work" in ADVERTISED_TOOL_NAMES
    assert codex._DYNAMIC_TOOL_NAMES == ADVERTISED_TOOL_NAMES


# ------------------------------------------------------------------------------ codex


def _functions(transport) -> list[str]:
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    tools = request["params"]["dynamicTools"]
    return [tool["name"] for namespace in tools for tool in namespace["tools"]]


def _instructions(transport) -> str:
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    return request["params"]["baseInstructions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("offered", [False, True])
async def test_codex_advertises_cancel_only_for_a_frozen_target(offered: bool) -> None:
    transport = FakeCodexTransport()
    inference = codex.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        [part async for part in inference.stream(_request(offered=offered), turn_id="turn_c")]
    finally:
        await inference.close()
    assert _functions(transport) == (
        ["start_work", "task_status", "cancel_work"] if offered else ["start_work", "task_status"]
    )
    _assert_exactly_one_cancel_instruction(_instructions(transport), offered=offered)


@pytest.mark.asyncio
async def test_codex_cancel_selection_is_a_terminal_host_item_without_effects() -> None:
    transport = DynamicStartCodexTransport(
        tool="cancel_work", arguments={}, assistant_delta="The task has been cancelled.",
    )
    handler = FakeWorkToolHandler(can_cancel_work=True)
    inference = codex.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)
    try:
        async with asyncio.timeout(1):
            assert [
                part async for part in inference.stream(_request(offered=True), turn_id="turn_c")
            ] == [HostCancelRequest()]
        assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("offered", "arguments"),
    [(False, {}), (True, {"task_id": "task_public"}), (True, {"reason": "now"})],
    ids=["unadvertised", "named_task", "extra_argument"],
)
async def test_codex_cancel_without_route_or_with_arguments_has_no_effect(
    offered: bool, arguments: dict[str, object],
) -> None:
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
                async for part in inference.stream(_request(offered=offered), turn_id="turn_c"):
                    assert type(part) is not HostCancelRequest
        assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_codex_cancel_acknowledgment_has_no_tools_and_no_stopped_claim() -> None:
    transport = FakeCodexTransport()
    inference = codex.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        [part async for part in inference.stream(_cancel_ack(), turn_id="turn_c")]
    finally:
        await inference.close()
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert request["params"]["dynamicTools"] == []
    assert "may still be stopping" in _instructions(transport)
    assert "Do not say it has stopped" in _instructions(transport)
    turn = next(row for row in transport.sent if row.get("method") == "turn/start")
    assert json.loads(turn["params"]["input"][0]["text"].split("\n", 1)[1]) == {
        "cancelling_background_objective": "Synthetic active work",
    }


# ----------------------------------------------------------------------------- ollama


_CANCEL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["kind"], "properties": {"kind": {"const": "cancel_work"}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("offered", [False, True])
async def test_ollama_offers_cancel_only_for_a_frozen_target(offered: bool) -> None:
    inference, _handler, sent, _ = adapter([decision_frame({"kind": "reply", "text": "Okay."})])
    try:
        assert await collect(inference, _request(offered=offered)) == ["Okay."]
    finally:
        await inference.close()
    options = sent[0]["format"]["oneOf"]
    assert (_CANCEL_SCHEMA in options) is offered
    system = sent[0]["messages"][0]["content"]
    _assert_exactly_one_cancel_instruction(system.split("Response schema:", 1)[0], offered=offered)


@pytest.mark.asyncio
async def test_ollama_cancel_decision_is_a_terminal_host_item() -> None:
    inference, handler, _sent, _ = adapter([decision_frame({"kind": "cancel_work"})])
    try:
        assert await collect(inference, _request(offered=True)) == [HostCancelRequest()]
        assert handler.calls == []
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "offered", "category"),
    [
        ({"kind": "cancel_work"}, False, "unadvertised"),
        ({"kind": "cancel_work", "task_id": "task_public"}, True, "decision_schema"),
        ({"kind": "cancel_work", "text": "Stopped."}, True, "decision_schema"),
    ],
    ids=["unadvertised", "named_task", "with_text"],
)
async def test_ollama_cancel_without_route_or_with_fields_is_refused(
    value: dict[str, object], offered: bool, category: str,
) -> None:
    inference, handler, _sent, _ = adapter([decision_frame(value)])
    try:
        with pytest.raises(ModelOutputRefused, match=category):
            await collect(inference, _request(offered=offered))
        assert handler.calls == []
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_ollama_cancel_acknowledgment_has_no_schema_and_no_stopped_claim() -> None:
    inference, handler, sent, _ = adapter([
        {"message": {"content": "Okay, stopping that now."}, "done": True},
    ])
    try:
        assert await collect(inference, _cancel_ack()) == ["Okay, stopping that now."]
    finally:
        await inference.close()
    assert "format" not in sent[0] and "tools" not in sent[0]
    system = sent[0]["messages"][0]["content"]
    assert "may still be stopping" in system
    # Mutation: keep the Cancel-control direction on an accepted cancel's acknowledgment.
    assert not any(fallback in system for fallback in _FALLBACKS)
    assert _ROUTE not in system
    reference = json.loads(system.split("\n\nReference data:\n", 1)[1])
    assert reference["cancelling_background_objective"] == "Synthetic active work"
    assert handler.calls == []
