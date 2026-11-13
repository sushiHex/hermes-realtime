"""Golden request matrix: the exact request each foreground model adapter sends.

This is a characterization net for the foreground-model refactor, not a statement that
today's requests are right. One matrix of host configurations and turns is driven
through both real adapters (``providers/ollama.py`` and ``providers/codex_app_server.py``)
with fake transports, an injected clock and fixed identifiers. What each adapter sends is
compared with ``tests/fixtures/foreground_request_golden/<adapter>.json``:

* Ollama: the ``/api/chat`` call (URL, method, headers and the body, whose structured
  form is proven to re-serialize to the exact bytes sent).
* Codex: the ``thread/start`` and ``turn/start`` parameters, and the order of every
  method sent on the wire.

The matrix is the full product of the axes whose conditions interact (request purpose,
work handler, lookup configuration, what the user said, lookup outcome), plus every
other axis varied alone from one anchor cell per instruction branch.

A fixture states every value once. A long text is stored as paragraphs of sentences so a
change shows as the sentence that changed, and a value equal to the value at the same
path in an earlier cell is stored as ``{"$ref": "<that cell>"}``. A cell that is entirely
a reference therefore records that the adapter does not react to that axis.

A normal run only compares. Regeneration is one explicit command, run from the
repository root, that refuses to run in CI and prints which cells changed:

    uv run python -m tests.providers.test_foreground_request_golden --regenerate

Regenerate only for a change that is meant to alter a request, and review the fixture
diff as the behaviour change it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import difflib
import io
import json
import os
import re
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

import hermes_realtime.providers.codex_app_server as codex_app_server
import hermes_realtime.providers.ollama as ollama
from hermes_realtime.conversation.context import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
)
from hermes_realtime.conversation.knowledge import KnowledgeConsumeResult
from hermes_realtime.conversation.streaming import (
    AcceptedStartInferenceRequest,
    ConversationInferenceRequest,
    ConversationPromptUpdate,
)
from hermes_realtime.memory import BuiltinMemorySnapshot
from hermes_realtime.providers.current_facts import (
    CurrentFactEvidence,
    CurrentFactSource,
    EvidencePassage,
    contains_private_material,
    external_search_forbidden,
    foreground_search_query,
)
from hermes_realtime.providers.ollama_models import DEFAULT_OLLAMA_NUM_CTX
from hermes_realtime.speech import Transcript
from tests.support.ollama_metadata import supported_ollama_metadata

pytestmark = pytest.mark.usefixtures("deny_uninjected_ollama_metadata")

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "foreground_request_golden"
_REGENERATE = "uv run python -m tests.providers.test_foreground_request_golden --regenerate"
_CI_VARIABLES = ("CI", "GITHUB_ACTIONS")


# --------------------------------------------------------------------------- the matrix


@dataclass(frozen=True, slots=True)
class Cell:
    """One host configuration and one turn. The defaults are the baseline cell."""

    # The turn.
    purpose: str = "user"
    say: str = "ordinary"
    evidence: str = "sources"
    history: str = "short"
    memory: str = "absent"
    updates: int = 0
    results: int = 0
    tasks: int = 0
    clock: str = "morning"
    # The host.
    handler: int | None = 2048
    lookup: str = "both"
    style: str = "default"
    profile: str = "full"
    background: str = "names"
    effort: str = "low"
    model: str = "first"
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    num_gpu: int | None = None
    think: bool | None = None

    @property
    def id(self) -> str:
        changed = [
            f"{field.name}={getattr(self, field.name)}"
            for field in fields(self)
            if getattr(self, field.name) != field.default
        ]
        return ",".join(changed) or "baseline"


# The axes whose conditions interact inside the adapters, taken as a full product.
_ROUTING_PURPOSES = ("user", "plain", "ends_assistant", "ack")
_HANDLERS = (2048, None)
_LOOKUPS = ("both", "coordinator", "direct", "none")
_SAYS = ("ordinary", "source", "outcome", "private", "local")
_EVIDENCE = ("sources", "conflict", "empty", "error", "unavailable")
# Every other axis, varied one value at a time from each anchor below.
_CONTENT_AXES: dict[str, tuple[Any, ...]] = {
    "history": ("none", "marked_below", "marked_at", "marked_over", "tight"),
    "memory": ("present", "truncated"),
    "updates": (1, 2),
    "results": (1, 3),
    "tasks": (1,),
    "clock": ("night",),
    "handler": (64,),
    "style": ("proactive", "concise", "explanatory", "learning"),
    "profile": ("identity", "none"),
    "background": ("unattested", "off"),
    "effort": ("high",),
    "model": ("second",),
    "num_ctx": (4096,),
    "num_gpu": (0,),
    "think": (True, False),
}
# One anchor per instruction branch: tools, no tools, acknowledgment, unadmitted, private.
_ANCHORS = (
    Cell(),
    Cell(handler=None, lookup="none"),
    Cell(purpose="ack"),
    Cell(purpose="plain"),
    Cell(say="private"),
)


def _coherent(cell: Cell) -> bool:
    """Drop cells whose evidence could not be produced by the configured lookup."""

    if cell.evidence == "sources":
        return True
    if cell.say != "source" or cell.lookup == "none":
        return False
    return cell.evidence != "unavailable" or cell.lookup != "direct"


def _matrix() -> tuple[Cell, ...]:
    cells = [Cell()]
    cells += [
        Cell(purpose=purpose, handler=handler, lookup=lookup, say=say, evidence=evidence)
        for purpose, handler, lookup, say, evidence in product(
            _ROUTING_PURPOSES, _HANDLERS, _LOOKUPS, _SAYS, _EVIDENCE
        )
        if purpose == "user" or evidence == "sources"
    ]
    cells += [
        Cell(purpose=purpose, handler=handler, lookup=lookup)
        for purpose, handler, lookup in product(("empty", "preflight"), _HANDLERS, _LOOKUPS)
    ]
    cells += [
        replace(anchor, **{axis: value})
        for anchor in _ANCHORS
        for axis, values in _CONTENT_AXES.items()
        for value in values
    ]
    cells += [
        Cell(history=history, results=results)
        for history, results in product(_CONTENT_AXES["history"], _CONTENT_AXES["results"])
    ]
    return tuple(dict.fromkeys(cell for cell in cells if _coherent(cell)))


# ------------------------------------------------------------------ synthetic scenarios

_SAY = {
    "ordinary": "Tell me a short story about a lighthouse.",
    "source": "What is the latest stable release of the Example framework?",
    "outcome": "Who won the Example Cup final today?",
    "private": "My password is in the notes, please keep it safe.",
    "local": "Search my repository for the newest change.",
}
# (source-sensitive, private, external search forbidden), as the adapters' predicates see it.
_SAY_CLASS = {
    "ordinary": (False, False, False),
    "source": (True, False, False),
    "outcome": (True, False, False),
    "private": (False, True, True),
    "local": (False, False, True),
}
_TURN_ID = "turn_golden"
_OLLAMA_MODELS = {"first": "golden-model", "second": "golden-model-2"}
_CODEX_MODELS = {"first": "gpt-5.6-terra", "second": "gpt-5.6-sol"}
_ACCEPTED_TASK = "task_golden"
_ACCEPTED_OBJECTIVE = "Draft the example lighthouse story outline for the café."
_RESULT_OBJECTIVE = "Summarize the example report."
_RESULTS = (
    ("completed", "The example report has three sections — all short."),
    ("failed", "The example archive could not be opened."),
    ("interrupted", "The example export stopped early."),
)
_UPDATES = (
    ConversationPromptUpdate(
        sequence=1, task_id="task_update1", status="completed",
        text="The example report summary is ready — three sections.",
    ),
    ConversationPromptUpdate(
        sequence=2, task_id="task_update2", status="failed",
        text="The example archive check did not finish.",
    ),
)
_MEMORY = {
    "present": BuiltinMemorySnapshot(
        memory="The example project uses weekly notes.", user="Prefers short answers and the café.",
    ),
    "truncated": BuiltinMemorySnapshot(
        memory="The example project uses weekly notes.", user="Prefers short answers and the café.",
        truncated=True,
    ),
}
_CLOCKS = {
    "morning": datetime(2031, 3, 14, 9, 26, 53, 589793, timezone(timedelta(hours=-5), "EST")),
    "night": datetime(2031, 12, 31, 23, 59, 59, 0, timezone(timedelta(hours=9), "JST")),
}
_PROFILES = {
    "none": None,
    "identity": codex_app_server.HermesRepresentativeContext(identity="Golden Hermes"),
    "full": codex_app_server.HermesRepresentativeContext(
        identity="Golden Hermes",
        persona="Calm and précis",
        user_preferences="Plain words",
        location="Exampleville, Testland",
    ),
}
_BACKGROUND: dict[str, Callable[[], tuple[str, ...] | None] | None] = {
    "off": None,
    "unattested": lambda: None,
    "names": lambda: ("web_search", "terminal", "file_read"),
}


def _marked(count: int) -> tuple[ConversationMessage, ...]:
    """Alternating rows with host speech at rows 1 and 5 and an interruption at row 3."""

    marks: dict[int, dict[str, Any]] = {
        1: {"source": "host_task_status"},
        3: {"interrupted": True},
        5: {"source": "host_task_proposal"},
    }
    return tuple(
        ConversationMessage(
            "user" if index % 2 == 0 else "assistant",
            f"Earlier {'question' if index % 2 == 0 else 'answer'} {index}.",
            **marks.get(index, {}),
        )
        for index in range(count)
    )


# Rows already heard before this turn, and the store bounds they are held in. The window
# is eight rows so the cells below, at and over it stay small; "tight" is a character
# budget small enough that recent results push heard rows out of a single snapshot.
_HISTORY: dict[str, tuple[dict[str, int], tuple[ConversationMessage, ...]]] = {
    "none": ({}, ()),
    "short": ({}, (
        ConversationMessage("user", "Good morning from the café."),
        ConversationMessage("assistant", "Good morning. How can I help?"),
    )),
    "marked_below": ({"max_messages": 8}, _marked(5)),
    "marked_at": ({"max_messages": 8}, _marked(7)),
    "marked_over": ({"max_messages": 8}, _marked(8)),
    "tight": ({"max_item_chars": 64, "max_window_chars": 128}, _marked(2)),
}


def _request(cell: Cell) -> ConversationContextSnapshot:
    """Build the snapshot the way the host does: a real store, then the request type."""

    if cell.purpose == "preflight":
        # The readiness probe both launchers send before any work tool is bound.
        return ConversationInferenceRequest(
            revision=0,
            messages=(
                ConversationMessage("user", "Reply with one short sentence confirming readiness."),
            ),
            active_tasks=(),
            updates=(),
        )
    if cell.purpose == "empty":
        return ConversationInferenceRequest(revision=0, messages=(), active_tasks=())
    bounds, rows = _HISTORY[cell.history]
    store = ConversationContextStore(**bounds)
    if cell.purpose == "ends_assistant":
        rows = (
            *rows,
            ConversationMessage("user", _SAY[cell.say]),
            ConversationMessage("assistant", "Understood."),
        )
    if rows:
        store.restore(DurableConversation(messages=rows, prior_work=False))
    if cell.memory != "absent":
        store.set_memory(_MEMORY[cell.memory])
    for index, (status, text) in enumerate(_RESULTS[: cell.results], start=1):
        store.record_task_accepted(
            task_id=f"task_result{index}", run_id=f"deleg_result{index}",
            objective=_RESULT_OBJECTIVE,
        )
        store.record_task_completed(
            task_id=f"task_result{index}", run_id=f"deleg_result{index}",
            result_status=status, result_text=text,
        )
    if cell.tasks:
        store.record_task_accepted(
            task_id="task_earlier", run_id="deleg_earlier",
            objective="Collect the example weekly notes.",
        )
    if cell.purpose != "ends_assistant":
        store.record_user_transcript(Transcript(_SAY[cell.say], True))
    if cell.purpose == "ack":
        store.record_task_accepted(
            task_id=_ACCEPTED_TASK, run_id="deleg_golden", objective=_ACCEPTED_OBJECTIVE,
        )
    snapshot = store.snapshot()
    if cell.purpose == "ack":
        return AcceptedStartInferenceRequest(
            revision=snapshot.revision,
            messages=snapshot.messages,
            active_tasks=tuple(
                task for task in snapshot.active_tasks if task.task_id == _ACCEPTED_TASK
            ),
            terminal_task_count=snapshot.terminal_task_count,
            memory=snapshot.memory,
            task_id=_ACCEPTED_TASK,
            recent_task_results=snapshot.recent_task_results,
        )
    if cell.purpose == "plain":
        return snapshot
    return ConversationInferenceRequest(
        revision=snapshot.revision,
        messages=snapshot.messages,
        active_tasks=snapshot.active_tasks,
        terminal_task_count=snapshot.terminal_task_count,
        memory=snapshot.memory,
        recent_task_results=snapshot.recent_task_results,
        updates=_UPDATES[: cell.updates],
    )


def _evidence(kind: str, text: str) -> CurrentFactEvidence | None:
    if kind == "unavailable":
        return None
    query = foreground_search_query(text) or text
    if kind == "error":
        return CurrentFactEvidence(
            query=query, retrieved_date="2031-03-14", sources=(), error="lookup timed out",
        )
    if kind == "empty":
        return CurrentFactEvidence(query=query, retrieved_date="2031-03-14", sources=())
    return CurrentFactEvidence(
        query=query,
        retrieved_date="2031-03-14",
        sources=(
            CurrentFactSource(
                title="Example Framework release notes",
                url="https://example.com/releases",
                snippet=(
                    "Example Framework 4.2 is the latest stable release. "
                    "The Example Cup final was won by the Example Rovers today."
                ),
            ),
            CurrentFactSource(
                title="Example Framework downloads",
                url="https://example.org/downloads",
                snippet="Downloads for every Example Framework version.",
                passages=(EvidencePassage("source_2", "Version 4.2 is listed as stable."),),
            ),
            CurrentFactSource(
                title="Example Cup archive",
                url="https://example.net/cup",
                snippet="The Example Rovers won the Example Cup final today.",
            ),
        ),
        conflict_detected=kind == "conflict",
    )


# ----------------------------------------------------------------------- adapter drivers


class _Handler:
    def __init__(self, max_objective_chars: int) -> None:
        self.max_objective_chars = max_objective_chars


class _Lines:
    def __init__(self, *records: dict[str, object]) -> None:
        self._lines = [json.dumps(record).encode() + b"\n" for record in records]

    def readline(self, limit: int = -1, /) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def close(self) -> None:
        self._lines.clear()


async def _ollama_request(cell: Cell) -> dict[str, Any]:
    sent: list[Request] = []
    # A bound adapter answers with one structured decision, except on the acknowledgment.
    structured = cell.handler is not None and cell.purpose != "ack"
    record: dict[str, object] = (
        {"message": {"content": json.dumps({"kind": "reply", "text": "Okay."})},
         "done": True, "done_reason": "stop"}
        if structured else {"message": {"content": "Okay."}, "done": True}
    )

    def open_request(request: Request, timeout: float) -> _Lines:
        sent.append(request)
        return _Lines(record)

    inference = ollama.OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model=_OLLAMA_MODELS[cell.model],
        open_request=open_request,
        num_ctx=cell.num_ctx,
        num_gpu=cell.num_gpu,
        think=cell.think,
        output_style=lambda: cell.style,
        metadata_client=supported_ollama_metadata(_OLLAMA_MODELS[cell.model]),
    )
    if cell.handler is not None:
        inference.bind_work_tools(_Handler(cell.handler))
    try:
        spoken = [
            part async for part in inference.stream(
                _request(cell),
                turn_id="turn_preflight" if cell.purpose == "preflight" else _TURN_ID,
            )
        ]
    finally:
        await inference.close()
    assert spoken == ["Okay."]
    (request,) = sent
    assert type(request.data) is bytes
    body = json.loads(request.data)
    # The structured body below is the wire body: it re-serializes to the exact bytes sent.
    assert request.data == json.dumps(
        body, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "url": request.full_url,
        "method": request.get_method(),
        "headers": dict(request.header_items()),
        "body": body,
    }


class _CodexTransport:
    """Answers the three requests of one turn; nothing leaves the process."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def send(self, message: Mapping[str, object]) -> None:
        copied = copy.deepcopy(dict(message))
        self.sent.append(copied)
        method, request_id = copied.get("method"), copied.get("id")
        if method == "initialize":
            await self._incoming.put({"id": request_id, "result": {}})
        elif method == "thread/start":
            await self._incoming.put({"id": request_id, "result": {
                "thread": {"id": "thread_golden"},
                "model": copied["params"]["model"],
                "modelProvider": "openai",
            }})
        elif method == "turn/start":
            turn = {"id": "server_turn_golden", "items": []}
            await self._incoming.put(
                {"id": request_id, "result": {"turn": {**turn, "status": "inProgress"}}}
            )
            await self._incoming.put({"method": "item/agentMessage/delta", "params": {
                "threadId": "thread_golden", "turnId": turn["id"], "itemId": "item_golden",
                "delta": "Okay.",
            }})
            await self._incoming.put({"method": "turn/completed", "params": {
                "threadId": "thread_golden", "turn": {**turn, "status": "completed"},
            }})

    async def receive(self) -> Mapping[str, object]:
        message = await self._incoming.get()
        if message is None:
            raise codex_app_server._CodexTransportClosed("golden transport closed")
        return message

    async def close(self) -> None:
        await self._incoming.put(None)


class _Lookup:
    def __init__(self, kind: str) -> None:
        self._kind = kind

    async def lookup(self, query: str) -> CurrentFactEvidence:
        evidence = _evidence(self._kind, query)
        assert evidence is not None
        return evidence

    async def close(self) -> None:
        return None


class _Coordinator:
    def __init__(self, kind: str) -> None:
        self._kind = kind

    async def consume_turn_result(self, turn_id: str, final_text: str) -> KnowledgeConsumeResult:
        return KnowledgeConsumeResult(
            evidence=_evidence(self._kind, final_text), timing=None, speculative=False,
        )

    async def discard_turn(self, turn_id: str, reason: str) -> None:
        return None

    async def close(self) -> None:
        return None


async def _codex_request(cell: Cell) -> dict[str, Any]:
    transport = _CodexTransport()
    inference = codex_app_server.CodexAppServerStreamingInference(
        model=_CODEX_MODELS[cell.model],
        effort=cell.effort,
        transport_factory=lambda: transport,
        current_fact_lookup=(
            _Lookup(cell.evidence) if cell.lookup in ("direct", "both") else None
        ),
        knowledge_coordinator=(
            _Coordinator(cell.evidence)
            if cell.lookup in ("coordinator", "both") else None
        ),
        local_now=lambda: _CLOCKS[cell.clock],
        hermes_context=_PROFILES[cell.profile],
        background_tools=_BACKGROUND[cell.background],
        output_style=lambda: cell.style,
    )
    if cell.handler is not None:
        inference.bind_work_tools(_Handler(cell.handler))
    try:
        spoken = [
            part async for part in inference.stream(
                _request(cell),
                turn_id="turn_preflight" if cell.purpose == "preflight" else _TURN_ID,
            )
        ]
    finally:
        await inference.close()
    assert spoken == ["Okay."]
    by_method = {message["method"]: message.get("params") for message in transport.sent}
    thread_start = by_method["thread/start"]
    # The only machine-specific value in the request; everything else is stored verbatim.
    assert thread_start["cwd"] == str(Path(tempfile.gettempdir()).resolve())
    thread_start["cwd"] = "<conversation_cwd>"
    return {
        "wire": [message["method"] for message in transport.sent],
        "thread/start": thread_start,
        "turn/start": by_method["turn/start"],
    }


_ADAPTERS: dict[str, Callable[[Cell], Awaitable[dict[str, Any]]]] = {
    "codex_app_server": _codex_request,
    "ollama": _ollama_request,
}


async def _capture(adapter: str, cells: tuple[Cell, ...]) -> dict[str, Any]:
    return {cell.id: await _ADAPTERS[adapter](cell) for cell in cells}


# ------------------------------------------------------------------- the fixture format

_LONG = 80
_PARAGRAPH = re.compile(r"(?<=\n\n)")
_SENTENCE = re.compile(r"(?<=[.!?] )|(?<=\n)|(?<=\},)|(?<=\],)")
_MARKERS = frozenset({"$ref", "$text"})


def _same(left: Any, right: Any) -> bool:
    """Equal as sent: key order and the difference between true and 1 both count."""

    return json.dumps(left, ensure_ascii=False) == json.dumps(right, ensure_ascii=False)


def _paragraphs(text: str) -> list[list[str]]:
    """Paragraphs of sentences whose concatenation is exactly ``text``."""

    return [
        [sentence for sentence in _SENTENCE.split(paragraph) if sentence]
        for paragraph in _PARAGRAPH.split(text) if paragraph
    ]


def _tree(value: Any) -> Any:
    """The request with each long text split into paragraphs of sentences."""

    if type(value) is dict:
        assert not _MARKERS & value.keys()
        return {key: _tree(item) for key, item in value.items()}
    if type(value) is list:
        return [_tree(item) for item in value]
    if type(value) is str and (len(value) > _LONG or "\n" in value):
        return {"$text": _paragraphs(value)}
    return value


def _plain(node: Any) -> Any:
    if type(node) is dict:
        if node.keys() == {"$text"}:
            return "".join(sentence for paragraph in node["$text"] for sentence in paragraph)
        return {key: _plain(item) for key, item in node.items()}
    if type(node) is list:
        return [_plain(item) for item in node]
    return node


def _children(node: Any) -> list[tuple[Any, Any]]:
    if type(node) is dict:
        return list(node.items())
    if type(node) is list:
        return list(enumerate(node))
    return []


def _rebuilt(node: Any, items: list[tuple[Any, Any]]) -> Any:
    return dict(items) if type(node) is dict else [item for _, item in items]


def _share(node: Any, path: tuple[Any, ...], cell: str, owners: dict[Any, str]) -> Any:
    """State a large value once: later equal values at the same path point to its cell."""

    if type(node) not in (dict, list):
        return node
    compact = json.dumps(node, ensure_ascii=False, separators=(",", ":"))
    if len(compact) > _LONG and owners.setdefault((path, compact), cell) != cell:
        return {"$ref": owners[(path, compact)]}
    return _rebuilt(node, [
        (key, _share(item, (*path, key), cell, owners)) for key, item in _children(node)
    ])


def _resolve(node: Any, path: tuple[Any, ...], cells: dict[str, Any]) -> Any:
    if type(node) is dict and node.keys() == {"$ref"}:
        target = cells[node["$ref"]]
        for key in path:
            target = target[key]
        return target
    if type(node) not in (dict, list):
        return node
    return _rebuilt(node, [
        (key, _resolve(item, (*path, key), cells)) for key, item in _children(node)
    ])


def _requests(document: str) -> dict[str, Any]:
    """Decode a fixture document into the full request of every cell."""

    resolved: dict[str, Any] = {}
    for cell, node in json.loads(document)["cells"].items():
        resolved[cell] = _resolve(node, (), resolved)
    return {cell: _plain(node) for cell, node in resolved.items()}


def _document(adapter: str, requests: dict[str, Any]) -> str:
    owners: dict[Any, str] = {}
    document = json.dumps(
        {
            "adapter": adapter,
            "format": 1,
            "regenerate": _REGENERATE,
            "baseline": asdict(Cell()),
            "cells": {
                cell: _share(_tree(request), (), cell, owners)
                for cell, request in requests.items()
            },
        },
        ensure_ascii=True,
        indent=1,
    ) + "\n"
    assert _same(_requests(document), requests), "the fixture format must be lossless"
    return document


def _leaf_differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Name each differing field; for a text, show the sentences that differ."""

    if _same(expected, actual):
        return []
    if type(expected) is type(actual) and type(expected) in (dict, list):
        left, right = _children(expected), _children(actual)
        if [key for key, _ in left] == [key for key, _ in right]:
            return [
                line
                for (key, golden), (_, sent) in zip(left, right, strict=True)
                for line in _leaf_differences(golden, sent, f"{path}/{key}")
            ]
    if type(expected) is str and type(actual) is str:
        sentences = [
            [sentence for paragraph in _paragraphs(text) for sentence in paragraph]
            for text in (expected, actual)
        ]
        delta = list(difflib.unified_diff(*sentences, "golden", "sent", n=1, lineterm=""))
        return [f"  {path}:", *(f"    {line!r}" for line in delta[2:14])]
    return [f"  {path}: golden {expected!r:.200} != sent {actual!r:.200}"]


def _differences(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    lines = [f"cell only in the golden: {cell}" for cell in expected if cell not in actual]
    lines += [f"cell missing from the golden: {cell}" for cell in actual if cell not in expected]
    changed = [
        cell for cell in actual if cell in expected and not _same(expected[cell], actual[cell])
    ]
    for cell in changed[:6]:
        lines += [f"cell {cell}:", *_leaf_differences(expected[cell], actual[cell])]
    if len(changed) > 6:
        lines.append("and also: " + "; ".join(changed[6:]))
    return lines


def _fixture(adapter: str) -> Path:
    return _FIXTURES / f"{adapter}.json"


def _assert_golden(adapter: str, sent: dict[str, Any], *, whole_matrix: bool) -> None:
    document = _fixture(adapter).read_text(encoding="utf-8")
    golden = _requests(document)
    if not whole_matrix:
        golden = {cell: golden[cell] for cell in sent}
    lines = _differences(golden, sent)
    if not lines and whole_matrix and document != _document(adapter, sent):
        lines = ["the fixture holds the right requests but is not in its generated form"]
    if lines:
        raise AssertionError("\n".join([
            f"{adapter} requests differ from {_fixture(adapter).name}.",
            *lines,
            f"If the change is intended, regenerate and review the diff: {_REGENERATE}",
        ]))


# -------------------------------------------------------------------------------- tests

_MATRIX = _matrix()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", sorted(_ADAPTERS))
async def test_every_matrix_request_matches_its_golden(adapter: str) -> None:
    _assert_golden(adapter, await _capture(adapter, _MATRIX), whole_matrix=True)


def test_matrix_turns_are_the_classes_the_adapters_see() -> None:
    assert {
        name: (
            foreground_search_query(text) is not None,
            contains_private_material(text),
            external_search_forbidden(text),
        )
        for name, text in _SAY.items()
    } == _SAY_CLASS
    assert len({cell.id for cell in _MATRIX}) == len(_MATRIX)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", sorted(_ADAPTERS))
async def test_one_changed_instruction_character_or_row_fails_the_golden(
    adapter: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = (Cell(),)
    sent = await _capture(adapter, baseline)
    _assert_golden(adapter, sent, whole_matrix=False)

    row_changed = copy.deepcopy(sent)
    if adapter == "ollama":
        row_changed["baseline"]["body"]["messages"][1]["content"] += "!"
    else:
        prompt = row_changed["baseline"]["turn/start"]["input"][0]
        prompt["text"] = prompt["text"].replace("Good morning from", "Good morning to", 1)
    with pytest.raises(AssertionError, match="cell baseline:"):
        _assert_golden(adapter, row_changed, whole_matrix=False)

    module = {"ollama": ollama, "codex_app_server": codex_app_server}[adapter]
    monkeypatch.setattr(
        module, "WORK_ROUTING_POLICY",
        module.WORK_ROUTING_POLICY.replace("propose work, not promises", "propose work"),
    )
    with pytest.raises(AssertionError, match="propose work, not promises"):
        _assert_golden(adapter, await _capture(adapter, baseline), whole_matrix=False)


def test_regeneration_is_explicit_and_refused_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {adapter: _fixture(adapter).read_bytes() for adapter in _ADAPTERS}
    for name in _CI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert main([]) == 2
    for name in _CI_VARIABLES:
        monkeypatch.setenv(name, "true")
        assert main(["--regenerate"]) == 2
        monkeypatch.delenv(name)
    assert {adapter: _fixture(adapter).read_bytes() for adapter in _ADAPTERS} == before


# ------------------------------------------------------------------------- regeneration


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments != ["--regenerate"]:
        print(f"This only compares under pytest. To regenerate: {_REGENERATE}", file=sys.stderr)
        return 2
    if any(os.environ.get(name) for name in _CI_VARIABLES):
        print("Refusing to regenerate golden requests in CI.", file=sys.stderr)
        return 2
    _FIXTURES.mkdir(parents=True, exist_ok=True)
    for adapter in sorted(_ADAPTERS):
        with contextlib.redirect_stdout(io.StringIO()):
            sent = asyncio.run(_capture(adapter, _MATRIX))
        path = _fixture(adapter)
        golden = _requests(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(_document(adapter, sent), encoding="utf-8", newline="\n")
        print(f"{adapter}: wrote {len(sent)} cells to {path.name}")
        for line in _differences(golden, sent):
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
