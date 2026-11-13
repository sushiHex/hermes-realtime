"""Characterization of how the Codex adapter decodes one tool selection, as it behaves today.

These tests pin which selections are accepted, which are refused and for which stated
reason, what each refusal does (raise, raise with a refusal line, or answer the call with
a rejection) and the line it prints. Every refusal is matched by its message, so a
selection refused for a different reason fails here. The refusal table is checked against
the decoder's source, so a refusal added without a row fails.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import textwrap
from typing import Any

import pytest

from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.providers.current_facts import CurrentFactEvidence

_BOUND = 64
_HOST_CAP = 1024
_QUERY_BOUND = 512
_OBJECTIVE = "Inspect the release."
_QUERY = "Latest stable release of the Example framework."
_FORBIDDEN_QUERY = "Search my repository for the newest change."
_PRIVATE_QUERY = "My password is in the notes, please keep it safe."
_RUN_TOKEN = "Use deleg_abc1."

_NO_LINE = ""
_MODEL_CONTROL_LINE = (
    '[codex-tool-refusal] {"refusal":"model_control_unavailable","tool":"work","version":1}\n'
)
_STATUS_ARGUMENTS_LINE = '[codex-task-status] {"category":"arguments"}\n'
_ADDITIONAL_CHOICE_LINE = '[codex-task-status] {"category":"additional_choice"}\n'
_ONE_ROUTE_REASON = "one knowledge search or work start is allowed per turn"


class _Text(str):
    """A string subclass: equal to its text, but not an exact built-in string."""


class _Handler:
    def __init__(self, max_objective_chars: int) -> None:
        self.max_objective_chars = max_objective_chars


class _Lookup:
    async def lookup(self, query: str) -> CurrentFactEvidence:
        raise AssertionError("decoding a selection must not run a lookup")


class _Transport:
    def __init__(self) -> None:
        raise AssertionError("decoding a selection must not open a transport")


def _inference(*, bound: int | None = _BOUND, lookup: bool = True) -> Any:
    inference = CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=_Transport,
        current_fact_lookup=_Lookup() if lookup else None,
    )
    if bound is not None:
        inference.bind_work_tools(_Handler(bound))
    return inference


def _selection(tool: object, arguments: object, namespace: object = "hermes") -> dict[str, object]:
    return {"tool": tool, "namespace": namespace, "arguments": arguments}


_START = {"objective": _OBJECTIVE}
_SEARCH = {"query": _QUERY}

# Every refusal the decoder can produce: reason -> (line printed, selections, setup).
# Setup is (objective bound or None for no work handler, direct lookup configured).
_Row = tuple[str, tuple[dict[str, object], ...], tuple[int | None, bool]]
_REFUSALS: dict[str, _Row] = {
    "Codex dynamic tool name must be an exact string": (_NO_LINE, (
        _selection(7, _START), _selection(None, _START), _selection(["start_work"], _START),
        _selection(_Text("start_work"), _START),
    ), (_BOUND, True)),
    "Codex dynamic tool is not allowlisted": (_MODEL_CONTROL_LINE, (
        _selection("cancel_active_work", {"task_id": "task_1"}),
        _selection("reply", {"text": "Okay."}),
        _selection("Start_Work", _START),
        _selection("", {}),
    ), (_BOUND, True)),
    "Codex dynamic tool namespace is prohibited": (_MODEL_CONTROL_LINE, (
        _selection("start_work", _START, namespace="functions"),
        _selection("task_status", {}, namespace=None),
        _selection("search_knowledge", _SEARCH, namespace=""),
        {"tool": "start_work", "arguments": _START},
        _selection("start_work", _START, namespace=_Text("hermes")),
    ), (_BOUND, True)),
    "Codex dynamic tool arguments must be an exact object": (_NO_LINE, (
        _selection("start_work", [_OBJECTIVE]), _selection("task_status", None),
        _selection("search_knowledge", _QUERY),
    ), (_BOUND, True)),
    "Codex dynamic tool arguments keys must be exact strings": (_NO_LINE, (
        _selection("start_work", {1: _OBJECTIVE}),
    ), (_BOUND, True)),
    "Codex search_knowledge arguments are invalid": (_NO_LINE, (
        _selection("search_knowledge", {}),
        _selection("search_knowledge", {"objective": _OBJECTIVE}),
        _selection("search_knowledge", {"query": _QUERY, "limit": 3}),
    ), (_BOUND, True)),
    "Codex search_knowledge query is invalid": (_NO_LINE, (
        _selection("search_knowledge", {"query": 7}),
        _selection("search_knowledge", {"query": None}),
        _selection("search_knowledge", {"query": _Text(_QUERY)}),
        _selection("search_knowledge", {"query": ""}),
        _selection("search_knowledge", {"query": " \n\t"}),
        _selection("search_knowledge", {"query": "x" * (_QUERY_BOUND + 1)}),
        _selection("search_knowledge", {"query": _RUN_TOKEN}),
        _selection("search_knowledge", {"query": _FORBIDDEN_QUERY}),
        _selection("search_knowledge", {"query": _PRIVATE_QUERY}),
    ), (_BOUND, True)),
    "Codex knowledge search is unavailable": (_NO_LINE, (
        _selection("search_knowledge", _SEARCH),
    ), (_BOUND, False)),
    "Codex start_work arguments are invalid": (_NO_LINE, (
        _selection("start_work", {}),
        _selection("start_work", {"query": _QUERY}),
        _selection("start_work", {"objective": _OBJECTIVE, "task_id": "task_1"}),
    ), (_BOUND, True)),
    "Codex work tools are unavailable": (_NO_LINE, (
        _selection("start_work", _START),
    ), (None, True)),
    "Codex start_work objective is invalid": (_NO_LINE, (
        _selection("start_work", {"objective": 7}),
        _selection("start_work", {"objective": None}),
        _selection("start_work", {"objective": _Text(_OBJECTIVE)}),
        _selection("start_work", {"objective": ""}),
        _selection("start_work", {"objective": "  \n"}),
        _selection("start_work", {"objective": "x" * (_BOUND + 1)}),
        _selection("start_work", {"objective": _RUN_TOKEN}),
    ), (_BOUND, True)),
    "Codex task_status arguments are invalid": (_STATUS_ARGUMENTS_LINE, (
        _selection("task_status", {"task_id": "task_1"}),
        _selection("task_status", {"objective": _OBJECTIVE}),
    ), (_BOUND, True)),
}
_CASES = [
    pytest.param(reason, line, selection, setup, id=f"{reason.removeprefix('Codex ')}-{index}")
    for reason, (line, selections, setup) in _REFUSALS.items()
    for index, selection in enumerate(selections)
]


def _decoder_refusals() -> set[str]:
    """Every refusal reason the decoder's source can raise."""

    reasons: set[str] = set()
    source = inspect.getsource(CodexAppServerStreamingInference._dynamic_semantics)
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Raise):
            call = node.exc
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            assert call.func.id == "RuntimeError"
            (message,) = call.args
            assert isinstance(message, ast.Constant)
            reasons.add(message.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "_refuse_model_control":
                message = node.args[0]
                assert isinstance(message, ast.Constant)
                reasons.add(message.value)
            elif node.func.attr == "_exact_string":
                name = node.args[1]
                assert isinstance(name, ast.Constant)
                reasons.add(f"Codex {name.value} must be an exact string")
            elif node.func.attr == "_exact_mapping":
                name = node.args[1]
                assert isinstance(name, ast.Constant)
                reasons.add(f"Codex {name.value} must be an exact object")
                reasons.add(f"Codex {name.value} keys must be exact strings")
    return reasons


def test_the_refusal_table_has_a_row_for_every_refusal_the_decoder_can_produce() -> None:
    assert _decoder_refusals() == set(_REFUSALS)


@pytest.mark.parametrize(("reason", "line", "selection", "setup"), _CASES)
def test_each_refused_selection_is_refused_for_its_stated_reason(
    reason: str, line: str, selection: dict[str, object], setup: tuple[int | None, bool],
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference = _inference(bound=setup[0], lookup=setup[1])

    with pytest.raises(RuntimeError) as caught:
        inference._dynamic_semantics(selection)

    assert type(caught.value) is RuntimeError
    assert str(caught.value) == reason
    assert capsys.readouterr().out == line


@pytest.mark.parametrize(("handler_bound", "bound"), [(1, 1), (64, 64), (1024, 1024), (2048, 1024)])
def test_an_objective_is_accepted_at_the_bound_and_refused_one_past_it(
    handler_bound: int, bound: int,
) -> None:
    inference = _inference(bound=handler_bound)

    accepted = inference._dynamic_semantics(_selection("start_work", {"objective": "x" * bound}))
    assert accepted[2] == {"objective": "x" * bound}
    with pytest.raises(RuntimeError, match=r"^Codex start_work objective is invalid$"):
        inference._dynamic_semantics(_selection("start_work", {"objective": "x" * (bound + 1)}))
    assert bound == min(handler_bound, _HOST_CAP)


def test_a_query_is_accepted_at_its_bound_and_refused_one_past_it() -> None:
    inference = _inference()

    accepted = inference._dynamic_semantics(
        _selection("search_knowledge", {"query": "x" * _QUERY_BOUND}),
    )
    assert accepted[2] == {"query": "x" * _QUERY_BOUND}
    with pytest.raises(RuntimeError, match=r"^Codex search_knowledge query is invalid$"):
        inference._dynamic_semantics(
            _selection("search_knowledge", {"query": "x" * (_QUERY_BOUND + 1)}),
        )


class _Turn:
    """One advertised thread and turn, driven through the lifecycle handler directly."""

    def __init__(self, inference: Any, *advertised: str, thread: str = "thread_probe") -> None:
        self.inference = inference
        self.thread = thread
        self.turn = "turn_probe"
        inference._advertised_dynamic_tools[thread] = frozenset(
            ("hermes", tool) for tool in advertised
        )
        ready = asyncio.Event()
        ready.set()
        inference._server_turn_ready[thread] = ready

    def start(self, call_id: str, tool: str, arguments: dict[str, object]) -> Any:
        item: dict[str, object] = {
            "id": call_id, "type": "dynamicToolCall", "namespace": "hermes", "tool": tool,
            "arguments": arguments, "status": "inProgress",
        }
        self.inference._handle_dynamic_lifecycle(
            "item/started",
            {"threadId": self.thread, "turnId": self.turn, "item": item, "startedAtMs": 1},
            item,
        )
        return self.inference._dynamic_calls[(self.thread, self.turn, call_id)]


_VALID: dict[str, dict[str, object]] = {
    "search_knowledge": _SEARCH, "start_work": _START, "task_status": {},
}
_ALL = tuple(_VALID)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _ALL)
async def test_a_tool_not_advertised_for_the_thread_is_refused(
    tool: str, capsys: pytest.CaptureFixture[str],
) -> None:
    inference = _inference()
    turn = _Turn(inference, *(name for name in _ALL if name != tool))

    with pytest.raises(RuntimeError) as caught:
        turn.start("call_1", tool, _VALID[tool])

    assert str(caught.value) == "Codex invoked an unadvertised dynamic tool"
    assert capsys.readouterr().out == _MODEL_CONTROL_LINE
    assert inference._dynamic_calls == {}
    await inference.close()


# After the first selection of a turn, which second selection is still admitted.
_SECOND: dict[tuple[str, str], tuple[bool, str]] = {
    ("search_knowledge", "search_knowledge"): (False, _NO_LINE),
    ("search_knowledge", "start_work"): (False, _NO_LINE),
    ("search_knowledge", "task_status"): (True, _NO_LINE),
    ("start_work", "search_knowledge"): (False, _NO_LINE),
    ("start_work", "start_work"): (False, _NO_LINE),
    ("start_work", "task_status"): (False, _ADDITIONAL_CHOICE_LINE),
    ("task_status", "search_knowledge"): (False, _ADDITIONAL_CHOICE_LINE),
    ("task_status", "start_work"): (False, _ADDITIONAL_CHOICE_LINE),
    ("task_status", "task_status"): (False, _ADDITIONAL_CHOICE_LINE),
}


def test_the_second_selection_table_covers_every_pair() -> None:
    assert set(_SECOND) == {(first, second) for first in _ALL for second in _ALL}


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), sorted(_SECOND))
async def test_a_second_selection_in_one_turn_is_answered_with_a_rejection(
    first: str, second: str, capsys: pytest.CaptureFixture[str],
) -> None:
    admitted, line = _SECOND[(first, second)]
    inference = _inference()
    turn = _Turn(inference, *_ALL)

    assert turn.start("call_1", first, _VALID[first]).response is None
    call = turn.start("call_2", second, _VALID[second])

    if admitted:
        assert call.response is None
    else:
        assert call.operation is None
        assert call.response == {
            "contentItems": [{"type": "inputText", "text": json.dumps(
                {"accepted": False, "state": "rejected", "reason": _ONE_ROUTE_REASON},
                separators=(",", ":"),
            )}],
            "success": False,
        }
        assert call.response_ready.is_set()
    assert capsys.readouterr().out == line
    await inference.close()
