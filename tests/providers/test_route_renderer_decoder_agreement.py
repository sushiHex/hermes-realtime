"""Each adapter accepts exactly the selections its own request offered, as it behaves today.

Driven from the golden request matrix: for every distinct set of routes either adapter
renders across the matrix, one cell is replayed through the real ``stream()`` and each
route name is then selected. A name the request offered must be accepted with an argument
at the rendered bound and refused one character past it; a name it did not offer must be
refused. The rendered routes are also compared with the golden fixture, so this test and
the request matrix describe the same requests.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any
from urllib.request import Request

import pytest

import hermes_realtime.providers.codex_app_server as codex_app_server
import hermes_realtime.providers.ollama as ollama
from hermes_realtime.conversation.natural_cancel import HostCancelRequest
from hermes_realtime.conversation.streaming import ModelOutputRefused
from hermes_realtime.conversation.task_proposals import (
    MAX_TASK_PROPOSAL_OBJECTIVE_CHARS,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.task_status import HostSpeechRequest
from tests.providers import test_foreground_request_golden as golden
from tests.support.ollama_metadata import supported_ollama_metadata

_ROUTE_NAMES = tuple(sorted(codex_app_server._DYNAMIC_TOOL_NAMES))
_ARGUMENT = {"search_knowledge": "query", "start_work": "objective"}


def _fixture_requests(adapter: str) -> dict[str, Any]:
    return golden._requests(golden._fixture(adapter).read_text(encoding="utf-8"))


def _representatives(adapter: str, rendered: Any) -> list[Any]:
    """The first matrix cell for each distinct rendering and host configuration."""

    requests = _fixture_requests(adapter)
    chosen: dict[str, golden.Cell] = {}
    for cell in golden._MATRIX:
        key = json.dumps(
            [rendered(requests[cell.id]), cell.handler, cell.lookup in ("direct", "both")],
            sort_keys=True,
        )
        chosen.setdefault(key, cell)
    return [pytest.param(cell, id=cell.id) for cell in chosen.values()]


# --------------------------------------------------------------------------- local adapter

def _ollama_format(request: Mapping[str, Any]) -> Any:
    return request["body"].get("format")


def _ollama_branches(schema: Any) -> dict[str, Any]:
    if schema is None:
        return {}
    branches = schema.get("oneOf", [schema])
    return {branch["properties"]["kind"]["const"]: branch for branch in branches}


async def _ollama_select(cell: Any, decision: dict[str, object]) -> tuple[Any, Any]:
    """Send the cell's request and answer it with one decision; return (format, outcome)."""

    sent: list[Request] = []
    structured = cell.handler is not None and cell.purpose not in ("ack", "cancel_ack")
    record: dict[str, object] = {"message": {"content": json.dumps(decision)}, "done": True}
    if structured:
        record["done_reason"] = "stop"

    def open_request(request: Request, timeout: float) -> Any:
        sent.append(request)
        return golden._Lines(record)

    inference = ollama.OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model=golden._OLLAMA_MODELS[cell.model],
        open_request=open_request,
        num_ctx=cell.num_ctx,
        num_gpu=cell.num_gpu,
        think=cell.think,
        output_style=lambda: cell.style,
        metadata_client=supported_ollama_metadata(golden._OLLAMA_MODELS[cell.model]),
    )
    if cell.handler is not None:
        inference.bind_work_tools(golden._Handler(cell.handler))
    outcome: Any
    try:
        outcome = [
            part async for part in inference.stream(
                golden._request(cell),
                turn_id="turn_preflight" if cell.purpose == "preflight" else golden._TURN_ID,
            )
        ]
    except ModelOutputRefused as refusal:
        outcome = refusal
    finally:
        await inference.close()
    (request,) = sent
    assert type(request.data) is bytes
    return json.loads(request.data).get("format"), outcome


def _ollama_decision(
    kind: str, length: int, branches: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    if kind == "reply":
        return {"kind": "reply", "text": "Okay."}
    if kind in _ARGUMENT:
        return {"kind": kind, _ARGUMENT[kind]: "x" * length}
    labels = _ollama_choice_labels(branches or {})
    if kind == "cancel_work" and labels:
        return {"kind": kind, "target": labels[-1]}
    return {"kind": kind}


def _ollama_choice_labels(branches: Mapping[str, Any]) -> list[str]:
    """The labels a cancel branch names, or none for the sole cancel."""
    cancel = branches.get("cancel_work")
    if cancel is None or "target" not in cancel["properties"]:
        return []
    return list(cancel["properties"]["target"]["enum"])


# Every cancel argument the multi-task schema does not name, each refused by the decoder.
_FOREIGN_CANCEL_ARGUMENTS: tuple[dict[str, object], ...] = (
    {}, {"target": "choice_9"}, {"target": "choice_3"}, {"target": "all"},
    {"target": "task_earlier"}, {"target": ["choice_1"]}, {"target": None},
    {"target": "choice_1", "also": "choice_2"}, {"task_id": "task_earlier"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("cell", _representatives("ollama", _ollama_format))
async def test_the_local_adapter_accepts_exactly_the_kinds_its_schema_names(
    cell: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    expected_format = _ollama_format(_fixture_requests("ollama")[cell.id])
    branches = _ollama_branches(expected_format)
    if "start_work" in branches:
        bound = branches["start_work"]["properties"]["objective"]["maxLength"]
        assert bound == min(cell.handler, MAX_TASK_PROPOSAL_OBJECTIVE_CHARS)
    else:
        bound = 1

    labels = _ollama_choice_labels(branches)
    for kind in ("reply", *_ROUTE_NAMES):
        sent_format, outcome = await _ollama_select(
            cell, _ollama_decision(kind, bound, branches),
        )
        assert sent_format == expected_format
        if expected_format is None:
            # No envelope was requested, so nothing the model writes can select a route.
            parts = [] if type(outcome) is ModelOutputRefused else outcome
            assert all(type(part) is str for part in parts), kind
        elif kind not in branches:
            assert type(outcome) is ModelOutputRefused, kind
            assert outcome.category == (
                "decision_schema" if kind == "search_knowledge" else "unadvertised"
            ), kind
        elif kind == "reply":
            assert outcome == ["Okay."]
        elif kind == "start_work":
            (proposal,) = outcome
            assert type(proposal) is HostTaskProposalRequest
            assert proposal.objective == "x" * bound
        elif kind == "cancel_work":
            (cancel,) = outcome
            assert type(cancel) is HostCancelRequest
            # Exactly the label the schema names, or none for the sole cancel.
            assert cancel == HostCancelRequest(target=labels[-1] if labels else None)
        else:
            (status,) = outcome
            assert type(status) is HostSpeechRequest

    if labels:
        for arguments in _FOREIGN_CANCEL_ARGUMENTS:
            _, outcome = await _ollama_select(cell, {"kind": "cancel_work", **arguments})
            assert type(outcome) is ModelOutputRefused, arguments
            assert outcome.category == "decision_schema", arguments
    elif "cancel_work" in branches:
        # The sole cancel names no task, so even a well-formed label is refused.
        _, outcome = await _ollama_select(cell, {"kind": "cancel_work", "target": "choice_1"})
        assert type(outcome) is ModelOutputRefused
        assert outcome.category == "decision_schema"
    if "start_work" in branches:
        _, outcome = await _ollama_select(cell, _ollama_decision("start_work", bound + 1))
        assert type(outcome) is ModelOutputRefused
        assert outcome.category == "objective_length"
    capsys.readouterr()


def test_the_local_matrix_reaches_every_schema_shape() -> None:
    kinds = {
        (
            *sorted(_ollama_branches(_ollama_format(request))),
            *_ollama_choice_labels(_ollama_branches(_ollama_format(request))),
        )
        for request in _fixture_requests("ollama").values()
    }

    assert kinds == {
        (), ("reply",), ("reply", "start_work", "task_status"),
        ("cancel_work", "reply", "start_work", "task_status"),
        # Two frozen running tasks: the cancel names exactly one of their labels.
        ("cancel_work", "reply", "start_work", "task_status", "choice_1", "choice_2"),
    }


# --------------------------------------------------------------------------- Codex adapter

def _codex_tools(request: Mapping[str, Any]) -> Any:
    return request["thread/start"]["dynamicTools"]


def _codex_named(tools: Any) -> dict[str, Any]:
    return {tool["name"]: tool for namespace in tools for tool in namespace["tools"]}


class _CapturingTransport(golden._CodexTransport):
    """The golden transport, also recording what the adapter itself holds as advertised."""

    def __init__(self) -> None:
        super().__init__()
        self.inference: Any = None
        self.advertised: Any = None
        self.advertised_choices: Any = None

    async def send(self, message: Mapping[str, object]) -> None:
        if message.get("method") == "turn/start":
            self.advertised = self.inference._advertised_dynamic_tools.get("thread_golden")
            self.advertised_choices = self.inference._advertised_cancel_choices.get(
                "thread_golden",
            )
        await super().send(message)


def _codex_choice_labels(named: Mapping[str, Any]) -> list[str]:
    """The labels the cancel tool's schema names, or none for the sole cancel."""
    cancel = named.get("cancel_work")
    if cancel is None or "target" not in cancel["inputSchema"]["properties"]:
        return []
    return list(cancel["inputSchema"]["properties"]["target"]["enum"])


def _probe(
    inference: Any, advertised: Any, thread: str, tool: str, argument: str,
    *, choices: Any = frozenset(), arguments: dict[str, object] | None = None,
) -> Any:
    inference._advertised_dynamic_tools[thread] = advertised
    inference._advertised_cancel_choices[thread] = choices
    ready = asyncio.Event()
    ready.set()
    inference._server_turn_ready[thread] = ready
    if arguments is None:
        arguments = {_ARGUMENT[tool]: argument} if tool in _ARGUMENT else {}
        if tool == "cancel_work" and choices:
            arguments = {"target": sorted(choices)[-1]}
    item: dict[str, object] = {
        "id": "call_probe", "type": "dynamicToolCall", "namespace": "hermes", "tool": tool,
        "arguments": arguments, "status": "inProgress",
    }
    try:
        inference._handle_dynamic_lifecycle(
            "item/started",
            {"threadId": thread, "turnId": "turn_probe", "item": item, "startedAtMs": 1},
            item,
        )
    except RuntimeError as refusal:
        return str(refusal)
    return inference._dynamic_calls[(thread, "turn_probe", "call_probe")]


@pytest.mark.asyncio
@pytest.mark.parametrize("cell", _representatives("codex_app_server", _codex_tools))
async def test_the_codex_adapter_accepts_exactly_the_tools_its_request_names(
    cell: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    expected_tools = _codex_tools(_fixture_requests("codex_app_server")[cell.id])
    named = _codex_named(expected_tools)
    direct_lookup = cell.lookup in ("direct", "both")
    transport = _CapturingTransport()
    inference = codex_app_server.CodexAppServerStreamingInference(
        model=golden._CODEX_MODELS[cell.model],
        effort=cell.effort,
        transport_factory=lambda: transport,
        current_fact_lookup=golden._Lookup(cell.evidence) if direct_lookup else None,
        knowledge_coordinator=(
            golden._Coordinator(cell.evidence)
            if cell.lookup in ("coordinator", "both") else None
        ),
        local_now=lambda: golden._CLOCKS[cell.clock],
        hermes_context=golden._PROFILES[cell.profile],
        background_tools=golden._BACKGROUND[cell.background],
        output_style=lambda: cell.style,
    )
    transport.inference = inference
    if cell.handler is not None:
        inference.bind_work_tools(golden._Handler(cell.handler))
    try:
        spoken = [
            part async for part in inference.stream(
                golden._request(cell),
                turn_id="turn_preflight" if cell.purpose == "preflight" else golden._TURN_ID,
            )
        ]
        assert spoken == ["Okay."]
        sent = {message["method"]: message.get("params") for message in transport.sent}
        assert _codex_tools(sent) == expected_tools
        # What the adapter holds as advertised is exactly what its request named.
        assert transport.advertised == frozenset(("hermes", name) for name in named)
        labels = _codex_choice_labels(named)
        assert transport.advertised_choices == frozenset(labels)

        bounds = {
            name: tool["inputSchema"]["properties"][_ARGUMENT[name]]["maxLength"]
            for name, tool in named.items() if name in _ARGUMENT
        }
        if "start_work" in named:
            assert bounds["start_work"] == min(cell.handler, MAX_TASK_PROPOSAL_OBJECTIVE_CHARS)
        for name in _ROUTE_NAMES:
            outcome = _probe(
                inference, transport.advertised, f"thread_{name}", name,
                "x" * bounds.get(name, 1), choices=transport.advertised_choices,
            )
            if name in named:
                assert type(outcome) is not str, (name, outcome)
                assert outcome.tool == name and outcome.response is None
                if name == "cancel_work":
                    expected = {"target": labels[-1]} if labels else {}
                    assert outcome.arguments == expected
            elif cell.handler is None and not direct_lookup:
                assert outcome == "Codex dynamic tool activity is prohibited", name
            elif name == "start_work" and cell.handler is None:
                assert outcome == "Codex work tools are unavailable"
            elif name == "search_knowledge" and not direct_lookup:
                assert outcome == "Codex knowledge search is unavailable"
            else:
                assert outcome == "Codex invoked an unadvertised dynamic tool", name
        if labels:
            for index, arguments in enumerate(_FOREIGN_CANCEL_ARGUMENTS):
                outcome = _probe(
                    inference, transport.advertised, f"thread_foreign_{index}", "cancel_work",
                    "", choices=transport.advertised_choices, arguments=dict(arguments),
                )
                assert outcome == "Codex cancel_work arguments are invalid", arguments
        elif "cancel_work" in named:
            # The sole cancel names no task, so even a well-formed label is refused.
            outcome = _probe(
                inference, transport.advertised, "thread_sole_label", "cancel_work", "",
                arguments={"target": "choice_1"},
            )
            assert outcome == "Codex cancel_work arguments are invalid"
        for name, bound in bounds.items():
            outcome = _probe(
                inference, transport.advertised, f"thread_over_{name}", name, "x" * (bound + 1),
            )
            assert outcome == (
                "Codex start_work objective is invalid" if name == "start_work"
                else "Codex search_knowledge query is invalid"
            )
    finally:
        await inference.close()
    capsys.readouterr()


def test_the_codex_matrix_reaches_every_tool_set_it_can_offer() -> None:
    tool_sets = {
        (
            *sorted(_codex_named(_codex_tools(request))),
            *_codex_choice_labels(_codex_named(_codex_tools(request))),
        )
        for request in _fixture_requests("codex_app_server").values()
    }

    # Search with status but no start does not occur: whatever withdraws the start on an
    # admitted user turn (private material, an attempted lookup) also withdraws the search.
    assert tool_sets == {
        (),
        ("search_knowledge",),
        ("start_work",),
        ("task_status",),
        ("search_knowledge", "start_work"),
        ("start_work", "task_status"),
        ("search_knowledge", "start_work", "task_status"),
        # Only a host-frozen cancel target adds the cancel route, always beside status.
        ("cancel_work", "start_work", "task_status"),
        ("cancel_work", "search_knowledge", "start_work", "task_status"),
        # Two frozen running tasks: the cancel names exactly one of their labels.
        ("cancel_work", "start_work", "task_status", "choice_1", "choice_2"),
        ("cancel_work", "search_knowledge", "start_work", "task_status", "choice_1", "choice_2"),
    }
