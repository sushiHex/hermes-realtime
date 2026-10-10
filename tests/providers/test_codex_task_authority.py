"""Synthetic provider-boundary checks; no Codex process or model is launched."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_codex_app_server import (
    DynamicStartCodexTransport,
    FakeCodexTransport,
    FakeWorkToolHandler,
    _snapshot,
)

from hermes_realtime.conversation.context import ConversationMessage
from hermes_realtime.conversation.streaming import (
    ConversationInferenceRequest,
    ConversationPromptUpdate,
)
from hermes_realtime.conversation.work_tools import WorkCancelResult, WorkStartResult
from hermes_realtime.providers import codex_app_server as provider


@pytest.mark.parametrize("tool", ("cancel_active_work", "cancel_work", "approve_work"))
def test_codex_removed_controls_are_not_valid_dynamic_semantics(tool) -> None:
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=FakeCodexTransport,
    )
    inference.bind_work_tools(FakeWorkToolHandler(can_cancel_work=True))
    with pytest.raises(RuntimeError, match="not allowlisted"):
        inference._dynamic_semantics({"tool": tool, "arguments": {}})


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
    assert [tool["name"] for tool in request["params"]["dynamicTools"]] == ["start_work"]
    assert "cancel_active_work" not in request["params"]["baseInstructions"]
    assert "Decide from the current user's intent" in request["params"]["baseInstructions"]
    assert "Do not require explicit background-task wording or confirmation" in (
        request["params"]["baseInstructions"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ({}, {"task_id": "task_public"}))
async def test_codex_model_cancellation_is_refused_before_effect(arguments) -> None:
    transport = DynamicStartCodexTransport(
        tool="cancel_active_work",
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
        namespace=None,
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
async def test_codex_start_only_handler_requires_no_model_cancellation_api() -> None:
    class StartOnly:
        max_objective_chars = 321

        async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
            return WorkStartResult(accepted=True, state="active", task_id="task_public")

    transport = DynamicStartCodexTransport()
    inference = provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: transport,
    )
    inference.bind_work_tools(StartOnly())
    try:
        assert [row async for row in inference.stream(_snapshot(), turn_id="turn_start")] == [
            "I started it."
        ]
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
async def test_codex_natural_current_request_can_start_after_terminal_work(
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
        assert [row async for row in inference.stream(snapshot, turn_id="turn_natural")] == [
            "I started it."
        ]
        assert len(handler.starts) == 1
        assert handler.cancels == handler.exact_cancels == []
    finally:
        await inference.close()
    request = next(row for row in transport.sent if row.get("method") == "thread/start")
    assert [tool["name"] for tool in request["params"]["dynamicTools"]] == ["start_work"]
    if utterance == "Please find out what public hearings are happening in Anaheim.":
        assert "delegate appropriate research through it" in request["params"]["baseInstructions"]
