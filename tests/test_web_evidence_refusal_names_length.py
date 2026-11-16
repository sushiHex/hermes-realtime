"""The presenter's cut predicates name themselves; all stay ``too_long`` (#159)."""

from __future__ import annotations

from typing import Any

import pytest

from hermes_realtime.integration import evidence_presentation as presenter
from hermes_realtime.integration.native_web_evidence import _CAUSES, _reason
from tests.test_web_evidence_refusal_names import (
    LEAD,
    _answer,
    _Formatted,
    _install_fake_openai_models,
    _markers,
    _observer,
    _refused_chat,
)

CATALOG = ("https://example.com/a",)
LIMITS = presenter.presentation_limits(CATALOG)
SURROGATE = "A split character \ud83d"

CASES = {
    "lead": ("presentation-lead-length", ("x" * (LIMITS["lead"] + 1), "", [0])),
    "support": ("presentation-support-length", (LEAD, "s" * (LIMITS["support"] + 1), [0])),
    "sources": ("presentation-source-count", (LEAD, "", [0] * (LIMITS["sources"] + 1))),
    "unicode": ("presentation-result-unicode", (SURROGATE, "", [0])),
    "unicode-support": ("presentation-result-unicode", (LEAD, SURROGATE, [0])),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("cause", "fields"), CASES.values(), ids=list(CASES))
async def test_each_cut_predicate_has_its_own_cause_and_stays_too_long(
    cause: str, fields: tuple[str, str, list[int]], monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    await _refused_chat(_observer(_Formatted(_answer(*fields))))

    [marker] = _markers(capsys)
    assert (marker["cause"], marker["reason"], marker["phase"], marker["stage"]) == (
        cause, "too_long", "render", "content")


@pytest.mark.parametrize(("first", "second"), [
    ("lead", "support"), ("lead", "sources"), ("lead", "unicode"),
    ("support", "sources"), ("support", "unicode"), ("sources", "unicode"),
])
def test_an_input_breaking_two_predicates_is_named_by_the_first_in_order(
    first: str, second: str,
) -> None:
    # Fixed order: lead length, support length, source count, Unicode.
    lead, support, ids = LEAD, "", [0]
    for name in (first, second):
        if name == "lead":
            lead = "x" * (LIMITS["lead"] + 1)
        elif name == "support":
            support = "s" * (LIMITS["support"] + 1)
        elif name == "sources":
            ids = [0] * (LIMITS["sources"] + 1)
        else:
            support = support + "\ud83d" if support else "\ud83d"
    with pytest.raises(ValueError) as caught:
        presenter.render_presentation_response(
            _chat_response(_answer(lead, support, ids)), CATALOG, "chat_completions", {},
            passages={},
        )
    assert str(caught.value) == CASES[first][0]


def _chat_response(content: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=content, refusal=None, tool_calls=None),
        )],
        usage=None,
    )


def test_new_causes_are_declared_and_map_to_too_long() -> None:
    for cause, _fields in CASES.values():
        assert cause in _CAUSES
        assert _reason(ValueError(cause), "internal") == "too_long"
