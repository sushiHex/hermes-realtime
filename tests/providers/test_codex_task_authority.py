"""Synthetic provider-boundary checks; no Codex process or model is launched."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_codex_app_server import (
    DynamicStartCodexTransport,
    FakeCodexTransport,
    FakeWorkToolHandler,
    _assert_proposal,
    _snapshot,
)

from hermes_realtime.conversation.context import ConversationMessage
from hermes_realtime.conversation.streaming import (
    ConversationInferenceRequest,
    ConversationPromptUpdate,
)
from hermes_realtime.conversation.work_tools import WorkCancelResult
from hermes_realtime.providers import codex_app_server as provider


def _advertised_functions(request):
    value = request["params"]["dynamicTools"]
    assert type(value) is list
    if not value:
        return []
    assert len(value) == 1
    namespace = value[0]
    assert type(namespace) is dict and namespace.get("type") == "namespace"
    assert namespace.get("name") == "hermes" and set(namespace) == {
        "type", "name", "description", "tools",
    }
    functions = namespace["tools"]
    assert type(functions) is list and 1 <= len(functions) <= 16
    assert all(
        type(row) is dict and row.get("type") == "function"
        and type(row.get("name")) is str and 1 <= len(row["name"]) <= 64
        for row in functions
    )
    return functions


@pytest.mark.parametrize("tool", ("cancel_active_work", "cancel_work", "approve_work"))
def test_codex_removed_controls_are_not_valid_dynamic_semantics(tool, capsys) -> None:
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=FakeCodexTransport,
    )
    inference.bind_work_tools(FakeWorkToolHandler(can_cancel_work=True))
    with pytest.raises(RuntimeError, match="not allowlisted"):
        inference._dynamic_semantics({"tool": tool, "arguments": {}})
    assert capsys.readouterr().out == (
        '[codex-tool-refusal] {"refusal":"model_control_unavailable","tool":"work","version":1}\n'
    )


def test_codex_start_cannot_report_a_cancellation_result() -> None:
    with pytest.raises(TypeError, match="invalid result"):
        provider.CodexAppServerStreamingInference._work_result_response(
            WorkCancelResult(accepted=True, state="cancelling", task_id="task_public"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("active", (False, True))
async def test_codex_advertises_start_without_model_cancellation(active: bool) -> None:
    transport = FakeCodexTransport()
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(FakeWorkToolHandler(can_cancel_work=True))
    try:
        assert [
            row
            async for row in inference.stream(_snapshot(active_task=active), turn_id="turn_schema")
        ] == ["Four."]
    finally:
        await inference.close()
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert [tool["name"] for tool in _advertised_functions(request)] == [
        "start_work", "task_status",
    ]
    assert "cancel_active_work" not in request["params"]["baseInstructions"]
    assert "Only the user's own request counts" in request["params"]["baseInstructions"]
    assert "Do not require explicit background-task wording to propose work" in (
        request["params"]["baseInstructions"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("cancel_active_work", "cancel_work", "approve_work"))
@pytest.mark.parametrize("arguments", ({}, {"task_id": "task_public"}))
async def test_codex_model_cancellation_is_refused_before_effect(tool, arguments, capsys) -> None:
    transport = DynamicStartCodexTransport(
        tool=tool,
        arguments=arguments,
        assistant_delta="Synthetic response.",
    )
    handler = FakeWorkToolHandler(can_cancel_work=True)
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)
    try:
        with pytest.raises(RuntimeError):
            async with asyncio.timeout(1):
                async for _ in inference.stream(_snapshot(active_task=True), turn_id="turn_cancel"):
                    pass
        assert handler.starts == handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()
    assert capsys.readouterr().out == (
        '[codex-tool-refusal] {"refusal":"model_control_unavailable","tool":"work","version":1}\n'
    )


def test_codex_valid_start_semantics_emit_no_control_refusal(capsys) -> None:
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=FakeCodexTransport,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    assert inference._dynamic_semantics({
        "tool": "start_work", "namespace": "hermes",
        "arguments": {"objective": "Inspect the build"},
    })[0] == "start_work"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("phase", ["registered", "pending_thread"])
def test_codex_repeated_control_validation_does_not_repeat_failure_evidence(phase, capsys) -> None:
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=FakeCodexTransport,
    )
    identity = ("thread_synthetic", "turn_synthetic")
    arguments = {"tool": "cancel_work", "arguments": {"task_id": "task_public"}}
    with pytest.raises(RuntimeError, match="not allowlisted") as refused:
        inference._dynamic_semantics(arguments, turn_identity=identity)
    if phase == "registered":
        inference._turn_routing[identity] = provider._TurnRouting()
        inference._mark_turn_failure(identity, refused.value)
    else:
        inference._pending_thread_routing_errors[identity[0]] = refused.value
    with pytest.raises(RuntimeError, match="not allowlisted"):
        inference._dynamic_semantics(arguments, turn_identity=identity)
    assert capsys.readouterr().out == (
        '[codex-tool-refusal] {"refusal":"model_control_unavailable","tool":"work","version":1}\n'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,arguments",
    (
        ("cancel_active_work", {}),
        ("cancel_active_work", {"task_id": "task_public"}),
        ("cancel_work", {"task_id": "task_public"}),
    ),
)
async def test_codex_execution_cannot_bypass_removed_model_controls(
    tool, arguments, capsys
) -> None:
    handler = FakeWorkToolHandler(can_cancel_work=True)
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=FakeCodexTransport,
    )
    inference.bind_work_tools(handler)
    call = provider._DynamicCall(
        identity=("thread_synthetic", "turn_synthetic", "call_synthetic"),
        tool=tool,
        namespace="hermes",
        arguments=arguments,
        canonical_arguments=json.dumps(arguments),
        deadline=0,
    )
    with pytest.raises(RuntimeError, match="not available"):
        await inference._invoke_dynamic_call(call)
    assert handler.starts == handler.cancels == handler.exact_cancels == []
    assert capsys.readouterr().out == (
        '[codex-tool-refusal] {"refusal":"model_control_unavailable","tool":"work","version":1}\n'
    )
    await inference.close()


@pytest.mark.asyncio
async def test_codex_proposal_handler_requires_no_start_or_cancellation_api() -> None:
    class ProposalOnly:
        max_objective_chars = 321

    transport = DynamicStartCodexTransport()
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(ProposalOnly())
    try:
        _assert_proposal([
            row async for row in inference.stream(_snapshot(), turn_id="turn_start")
        ], inference)
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("empty", "assistant_tail", "updates_only"))
async def test_codex_start_requires_a_current_user_row(kind) -> None:
    messages = (
        ()
        if kind in ("empty", "updates_only")
        else (ConversationMessage(role="user", text="Research the synthetic topic"),)
    )
    if kind == "assistant_tail":
        messages += (ConversationMessage(role="assistant", text="Synthetic delivered answer."),)
    updates = (
        ()
        if kind != "updates_only"
        else (
            ConversationPromptUpdate(
                sequence=1, task_id="task_public", status="completed", text="Synthetic result."
            ),
        )
    )
    snapshot = ConversationInferenceRequest(
        revision=1, messages=messages, active_tasks=(), updates=updates
    )
    transport = FakeCodexTransport()
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(FakeWorkToolHandler())
    try:
        assert [row async for row in inference.stream(snapshot, turn_id="turn_no_user")] == [
            "Four."
        ]
    finally:
        await inference.close()
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert request["params"]["dynamicTools"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("with_update", (False, True))
@pytest.mark.parametrize("utterance", (
    "What changed in the current Python release?",
    "Check the current Python release notes and explain the changes.",
    "Please find out what public hearings are happening in Anaheim.",
))
async def test_codex_natural_current_request_can_propose_after_terminal_work(
    with_update, utterance,
) -> None:
    snapshot = ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage("user", utterance),),
        active_tasks=(),
        terminal_task_count=1,
        updates=(
            ConversationPromptUpdate(
                sequence=1, task_id="task_public", status="completed", text="Synthetic result."
            ),
        ) if with_update else (),
    )
    transport = DynamicStartCodexTransport(
        arguments={"objective": "Research current Python release changes"},
    )
    handler = FakeWorkToolHandler(can_cancel_work=True)
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
    )
    inference.bind_work_tools(handler)
    try:
        _assert_proposal([
            row async for row in inference.stream(snapshot, turn_id="turn_natural")
        ], inference, objective="Research current Python release changes")
        assert handler.starts == []
        assert handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert [tool["name"] for tool in _advertised_functions(request)] == [
        "start_work", "task_status",
    ]
    if utterance == "Please find out what public hearings are happening in Anaheim.":
        assert "delegate appropriate research through it" in request["params"]["baseInstructions"]
