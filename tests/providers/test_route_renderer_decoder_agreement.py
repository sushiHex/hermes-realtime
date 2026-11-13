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
    structured = cell.handler is not None and cell.purpose != "ack"
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


def _ollama_decision(kind: str, length: int) -> dict[str, object]:
    if kind == "reply":
        return {"kind": "reply", "text": "Okay."}
    if kind in _ARGUMENT:
        return {"kind": kind, _ARGUMENT[kind]: "x" * length}
    return {"kind": kind}


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

    for kind in ("reply", *_ROUTE_NAMES):
        sent_format, outcome = await _ollama_select(cell, _ollama_decision(kind, bound))
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
        else:
            (status,) = outcome
            assert type(status) is HostSpeechRequest

    if "start_work" in branches:
        _, outcome = await _ollama_select(cell, _ollama_decision("start_work", bound + 1))
        assert type(outcome) is ModelOutputRefused
        assert outcome.category == "objective_length"
    capsys.readouterr()


def test_the_local_matrix_reaches_every_schema_shape() -> None:
    kinds = {
        tuple(sorted(_ollama_branches(_ollama_format(request))))
        for request in _fixture_requests("ollama").values()
    }

    assert kinds == {(), ("reply",), ("reply", "start_work", "task_status")}


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

    async def send(self, message: Mapping[str, object]) -> None:
        if message.get("method") == "turn/start":
            self.advertised = self.inference._advertised_dynamic_tools.get("thread_golden")
        await super().send(message)


def _probe(inference: Any, advertised: Any, thread: str, tool: str, argument: str) -> Any:
    inference._advertised_dynamic_tools[thread] = advertised
    ready = asyncio.Event()
    ready.set()
    inference._server_turn_ready[thread] = ready
    arguments = {_ARGUMENT[tool]: argument} if tool in _ARGUMENT else {}
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

        bounds = {
            name: tool["inputSchema"]["properties"][_ARGUMENT[name]]["maxLength"]
            for name, tool in named.items() if name in _ARGUMENT
        }
        if "start_work" in named:
            assert bounds["start_work"] == min(cell.handler, MAX_TASK_PROPOSAL_OBJECTIVE_CHARS)
        for name in _ROUTE_NAMES:
            outcome = _probe(
                inference, transport.advertised, f"thread_{name}", name,
                "x" * bounds.get(name, 1),
            )
            if name in named:
                assert type(outcome) is not str, (name, outcome)
                assert outcome.tool == name and outcome.response is None
            elif cell.handler is None and not direct_lookup:
                assert outcome == "Codex dynamic tool activity is prohibited", name
            elif name == "start_work" and cell.handler is None:
                assert outcome == "Codex work tools are unavailable"
            elif name == "search_knowledge" and not direct_lookup:
                assert outcome == "Codex knowledge search is unavailable"
            else:
                assert outcome == "Codex invoked an unadvertised dynamic tool", name
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
        tuple(sorted(_codex_named(_codex_tools(request))))
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
    }
