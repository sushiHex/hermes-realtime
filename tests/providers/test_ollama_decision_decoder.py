"""Characterization of the local adapter's decision decoder, as it behaves today.

These tests pin how one returned decision is decoded: which selections are accepted,
which are refused and under which category, what each refusal costs (the turn or the
session) and the evidence line it prints. The golden request matrix pins what is sent;
nothing there covers what comes back. The tables are checked against the adapter's
source, so a refusal category added without a row here fails.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest

from hermes_realtime.conversation.streaming import (
    MODEL_OUTPUT_REFUSAL_CATEGORIES,
    ModelOutputRefused,
)
from hermes_realtime.providers import ollama
from hermes_realtime.providers.ollama import OllamaDecisionRefused, OllamaStreamingInference

_BOUND = 64
_REPLY_BOUND = 65_536
_LONE_SURROGATE = "\ud800"

# What a refusal costs: the turn only, or the session.
_TURN = ModelOutputRefused
_SESSION = OllamaDecisionRefused
_CONSEQUENCE: dict[str, type[Exception]] = {
    "call_imitation": _TURN,
    "unverified_claim": _TURN,
    "unexpected_native_calls": _TURN,
    "unadvertised": _TURN,
    "duplicate_keys": _TURN,
    "decision_json": _TURN,
    "decision_schema": _TURN,
    "reply_type": _TURN,
    "reply_blank": _TURN,
    "reply_length": _TURN,
    "reply_unicode": _TURN,
    "objective_type": _TURN,
    "objective_blank": _TURN,
    "objective_length": _TURN,
    "objective_unicode": _TURN,
    "private_authority": _SESSION,
    "calls_type": _SESSION,
    "decision_unicode": _SESSION,
    "completion_reason": _SESSION,
    "incomplete": _SESSION,
}


def _raw(value: object) -> str:
    return json.dumps(value)


# Every refusal the decoder can produce: (category, decision text, start and status offered).
_REFUSALS: tuple[tuple[str, str, bool], ...] = (
    ("duplicate_keys", '{"kind":"reply","kind":"reply","text":"Okay."}', True),
    ("duplicate_keys", '{"kind":"start_work","objective":"A","objective":"B"}', True),
    ("decision_json", "Okay.", True),
    ("decision_json", '{"kind":"reply","text":"Okay."', True),
    ("decision_json", '{"kind":"reply","text":NaN}', True),
    ("decision_json", '{"kind":"reply","text":Infinity}', True),
    ("decision_schema", "[]", True),
    ("decision_schema", '"reply"', True),
    ("decision_schema", "null", True),
    ("decision_schema", _raw({"text": "Okay."}), True),
    ("decision_schema", _raw({"kind": "propose", "objective": "Inspect it."}), True),
    ("decision_schema", _raw({"kind": "search_knowledge", "query": "Releases."}), True),
    ("decision_schema", _raw({"kind": "reply"}), True),
    ("decision_schema", _raw({"kind": "reply", "text": "Okay.", "objective": "Inspect it."}), True),
    ("decision_schema", _raw({"kind": "reply", "objective": "Inspect it."}), True),
    ("decision_schema", _raw({"kind": "start_work"}), True),
    ("decision_schema", _raw({"kind": "start_work", "objective": "Inspect it.", "x": 1}), True),
    ("decision_schema", _raw({"kind": "start_work", "text": "Inspect it."}), True),
    ("decision_schema", _raw({"kind": "task_status", "task_id": "task_1"}), True),
    ("decision_schema", _raw({"kind": "task_status", "text": "Okay."}), False),
    ("unadvertised", _raw({"kind": "start_work", "objective": "Inspect it."}), False),
    ("unadvertised", _raw({"kind": "task_status"}), False),
    ("reply_type", _raw({"kind": "reply", "text": 7}), True),
    ("reply_type", _raw({"kind": "reply", "text": None}), True),
    ("reply_type", _raw({"kind": "reply", "text": ["Okay."]}), True),
    ("reply_blank", _raw({"kind": "reply", "text": ""}), True),
    ("reply_blank", _raw({"kind": "reply", "text": " \n\t"}), True),
    ("reply_length", _raw({"kind": "reply", "text": "x" * (_REPLY_BOUND + 1)}), True),
    ("reply_unicode", _raw({"kind": "reply", "text": "Okay" + _LONE_SURROGATE}), True),
    ("objective_type", _raw({"kind": "start_work", "objective": 7}), True),
    ("objective_type", _raw({"kind": "start_work", "objective": {"text": "Inspect it."}}), True),
    ("objective_blank", _raw({"kind": "start_work", "objective": ""}), True),
    ("objective_blank", _raw({"kind": "start_work", "objective": "  \n"}), True),
    ("objective_length", _raw({"kind": "start_work", "objective": "x" * (_BOUND + 1)}), True),
    ("objective_unicode", _raw({"kind": "start_work", "objective": "A" + _LONE_SURROGATE}), True),
    ("private_authority", _raw({"kind": "start_work", "objective": "Use deleg_abc1."}), True),
)


def _refuse_work_categories(function: object) -> set[str]:
    """Every category the given source can pass to ``_refuse_work``."""

    found: set[str] = set()
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))  # type: ignore[arg-type]
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_refuse_work"
        ):
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant):
            found.add(argument.value)
        elif isinstance(argument, ast.IfExp):
            assert isinstance(argument.body, ast.Constant)
            assert isinstance(argument.orelse, ast.Constant)
            found.update((argument.body.value, argument.orelse.value))
        else:
            # ``label + "_suffix"``: the decoder's two labels.
            assert isinstance(argument, ast.BinOp) and isinstance(argument.right, ast.Constant)
            assert isinstance(argument.left, ast.Name) and argument.left.id == "label"
            found.update(label + argument.right.value for label in ("reply", "objective"))
    return found


def _evidence(category: str) -> str:
    return '[ollama-work] ' + json.dumps(
        {"category": category, "kind": "start", "attempts": 0}, separators=(",", ":"),
    ) + "\n"


def test_the_refusal_table_has_a_row_for_every_category_the_decoder_can_produce() -> None:
    decoder = _refuse_work_categories(OllamaStreamingInference._work_decision)

    assert decoder == {category for category, _, _ in _REFUSALS}
    assert decoder <= set(_CONSEQUENCE)


def test_the_consequence_table_has_a_row_for_every_category_the_adapter_refuses_with() -> None:
    assert _refuse_work_categories(ollama) == set(_CONSEQUENCE)
    turn_only = {category for category, kind in _CONSEQUENCE.items() if kind is _TURN}
    assert turn_only == ollama._TURN_REFUSAL_CATEGORIES
    assert turn_only <= MODEL_OUTPUT_REFUSAL_CATEGORIES


@pytest.mark.parametrize("category", sorted(_CONSEQUENCE))
def test_each_category_costs_the_turn_or_the_session_and_prints_one_line(
    category: str, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises((ModelOutputRefused, OllamaDecisionRefused)) as caught:
        OllamaStreamingInference._refuse_work(category)

    assert type(caught.value) is _CONSEQUENCE[category]
    assert caught.value.category == category
    assert capsys.readouterr().out == _evidence(category)


@pytest.mark.parametrize(
    ("category", "content", "offered"), _REFUSALS,
    ids=[f"{row[0]}-{index}" for index, row in enumerate(_REFUSALS)],
)
def test_each_refused_decision_is_refused_for_its_own_reason(
    category: str, content: str, offered: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises((ModelOutputRefused, OllamaDecisionRefused)) as caught:
        OllamaStreamingInference._work_decision(content, _BOUND, can_start=offered)

    assert type(caught.value) is _CONSEQUENCE[category]
    assert caught.value.category == category
    assert capsys.readouterr().out == _evidence(category)


@pytest.mark.parametrize(
    "text", ["Okay.", "  Two sentences. With edges.  ", "x" * _REPLY_BOUND],
    ids=["short", "edges kept", "at the bound"],
)
@pytest.mark.parametrize("offered", [True, False])
def test_a_reply_is_returned_unchanged_up_to_its_own_bound(
    text: str, offered: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    decision = OllamaStreamingInference._work_decision(
        _raw({"kind": "reply", "text": text}), _BOUND, can_start=offered,
    )

    assert decision == (None, text)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bound", [1, 64, 1024])
def test_an_objective_is_accepted_at_the_bound_and_refused_one_past_it(
    bound: int, capsys: pytest.CaptureFixture[str],
) -> None:
    accepted = OllamaStreamingInference._work_decision(
        _raw({"kind": "start_work", "objective": "x" * bound}), bound, can_start=True,
    )
    assert accepted == ("x" * bound, "")
    assert capsys.readouterr().out == ""

    with pytest.raises(ModelOutputRefused) as caught:
        OllamaStreamingInference._work_decision(
            _raw({"kind": "start_work", "objective": "x" * (bound + 1)}), bound, can_start=True,
        )
    assert caught.value.category == "objective_length"
