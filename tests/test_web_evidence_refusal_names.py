"""A web-evidence refusal names its exact cause in a content-free marker (#550 step 1)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_realtime.integration.native_web_evidence import NativeWebEvidence
from tests.test_native_web_evidence import (
    RUN_ID,
    URL,
    _bind_run,
    _final_response,
    _final_turn,
    _install_fake_openai_models,
    _observer,
    _Presentation,
    _web_turn,
)

LEAD = "The report found a supported result."
FIELDS = {
    "category", "count", "reason", "phase", "cause", "error_class", "stage",
    "selected_source_count", "rendered_detail_length",
}


class _Formatted(_Presentation):
    """A presenter that returns exactly ``content`` as its chat completion."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    async def complete(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=self.content, refusal=None, tool_calls=None),
            )],
            usage=None,
        )


def _answer(lead: str, support: str, source_ids: list[int]) -> str:
    return json.dumps({"answer": {"lead": lead, "support": support}, "source_ids": source_ids})


def _markers(capsys: Any) -> list[dict[str, Any]]:
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("[web-evidence] ")]
    return [json.loads(line.removeprefix("[web-evidence] ")) for line in lines]


async def _refused_chat(observer: NativeWebEvidence) -> None:
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    _web_turn(observer)
    returned = await asyncio.to_thread(_final_turn, observer, _final_response())
    assert returned.choices[0].message.content == '{"error": "policy_refused"}'
    assert observer.reports[RUN_ID]


def _codex_final(*items: Any) -> Any:
    return SimpleNamespace(status="completed", output=list(items))


def _codex_message(text: str) -> Any:
    return SimpleNamespace(
        type="message", role="assistant", status="completed", phase="final_answer",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _codex_turn(observer: NativeWebEvidence, response: Any) -> Any:
    return observer.llm_execution(
        request={"model": "test"}, provider="test", base_url="https://example.com",
        model="test", api_mode="codex_responses", next_call=lambda _request: response,
    )


async def _refused_codex(observer: NativeWebEvidence, response: Any) -> None:
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    _web_turn(observer)
    returned = await asyncio.to_thread(_codex_turn, observer, response)
    assert returned.output_text == '{"error": "policy_refused"}'


@pytest.mark.asyncio
async def test_rendered_detail_over_bound_names_length_stage_and_detail_length(
    monkeypatch: Any, capsys: Any,
) -> None:
    # Fields within the per-request limits always fit (tests/test_presentation_fits.py), so
    # the rendered-detail check is a backstop for a route that ignores the schema bounds.
    # Widen the accepted bounds to reach it.
    _install_fake_openai_models(monkeypatch)
    from hermes_realtime.integration import evidence_presentation

    monkeypatch.setattr(evidence_presentation, "presentation_limits", lambda _catalog: {
        "lead": 220, "support": 1800, "sources": 256})
    support = "s" * 1790
    detail = LEAD + "\n\n" + support + "\n\nSources:\n" + URL
    assert len(detail) > 1800
    await _refused_chat(_observer(_Formatted(_answer(LEAD, support, [0]))))

    assert _markers(capsys) == [{
        "category": "refused", "count": 1, "reason": "too_long", "phase": "render",
        "cause": "presentation-content-length", "error_class": "ValueError",
        "stage": "content", "selected_source_count": 1,
        "rendered_detail_length": len(detail),
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize(("lead", "support", "cause", "reason"), [
    ("x" * 221, "", "presentation-content-length", "too_long"),
    (LEAD, "s" * 1790, "presentation-content-length", "too_long"),
    ("two\nlines", "", "presentation-content-shape", "internal"),
    (LEAD, "<detail>", "presentation-content-shape", "internal"),
    (" ", "", "presentation-content-shape", "internal"),
], ids=["lead-length", "support-length", "lead-newline", "markup", "blank-lead"])
async def test_answer_field_refusals_split_length_from_shape(
    lead: str, support: str, cause: str, reason: str, monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    await _refused_chat(_observer(_Formatted(_answer(lead, support, [0]))))

    [marker] = _markers(capsys)
    assert (marker["cause"], marker["stage"], marker["phase"]) == (cause, "content", "render")
    assert marker["reason"] == reason
    assert marker["rendered_detail_length"] is None


@pytest.mark.asyncio
async def test_answer_object_shape_is_named_shape(monkeypatch: Any, capsys: Any) -> None:
    _install_fake_openai_models(monkeypatch)
    content = json.dumps({"answer": {"lead": LEAD}, "source_ids": []})
    await _refused_chat(_observer(_Formatted(content)))

    [marker] = _markers(capsys)
    assert (marker["cause"], marker["stage"]) == ("presentation-content-shape", "content")


@pytest.mark.asyncio
async def test_presenter_json_failure_is_named_not_a_subclass(
    monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    await _refused_chat(_observer(_Formatted("not a presentation envelope")))

    [marker] = _markers(capsys)
    assert (marker["cause"], marker["error_class"], marker["stage"]) == (
        "presentation-json", "ValueError", "json")


@pytest.mark.asyncio
async def test_codex_native_final_over_bound_names_responses_text_bound(capsys: Any) -> None:
    observer = _observer()
    await _refused_codex(observer, _codex_final(_codex_message("x" * 8193)))

    assert _markers(capsys) == [{
        "category": "refused", "count": 1, "reason": "internal", "phase": "final",
        "cause": "responses-text-bound", "error_class": "ValueError", "stage": "none",
        "selected_source_count": None, "rendered_detail_length": None,
    }]
    assert observer.presentation.calls == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_codex_function_call_argument_json_is_named(capsys: Any) -> None:
    call = SimpleNamespace(type="function_call", status="completed", call_id="call_1",
                           name="web_search", arguments="{not json")
    await _refused_codex(_observer(), _codex_final(call))

    [marker] = _markers(capsys)
    assert (marker["cause"], marker["error_class"], marker["phase"]) == (
        "responses-function-call-json", "ValueError", "final")


@pytest.mark.asyncio
async def test_foreign_exception_is_named_by_allowlisted_class_only(
    monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    import hermes_realtime.integration.native_web_evidence as module

    class PrivateFailure(Exception):
        pass

    def runtime(*_args: Any) -> str:
        raise RuntimeError("private detail")

    def foreign(*_args: Any) -> str:
        raise PrivateFailure("private detail")

    for broken, label in ((runtime, "RuntimeError"), (foreign, "other")):
        monkeypatch.setattr(module, "presentation_input", broken)
        await _refused_chat(_observer())
        out = capsys.readouterr().out
        assert "private" not in out and "PrivateFailure" not in out
        marker = json.loads(out.strip().removeprefix("[web-evidence] "))
        assert (marker["cause"], marker["error_class"], marker["phase"]) == (
            "none", label, "final")


@pytest.mark.asyncio
async def test_undeclared_value_error_literal_is_not_emitted(
    monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    presentation = _Presentation()

    async def failed(*_args: Any) -> Any:
        raise ValueError("fixed-test-private-literal")

    presentation.complete = failed  # type: ignore[method-assign]
    await _refused_chat(_observer(presentation))

    out = capsys.readouterr().out
    assert "fixed-test-private-literal" not in out
    marker = json.loads(out.strip().removeprefix("[web-evidence] "))
    assert set(marker) == FIELDS
    assert (marker["reason"], marker["phase"], marker["cause"], marker["error_class"]) == (
        "presentation_failed", "presentation", "undeclared", "ValueError")
