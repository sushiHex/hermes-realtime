"""A web-evidence refusal names its exact cause in a content-free marker (#550 step 1)."""

from __future__ import annotations

import asyncio
import hashlib
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


class _CodexFormatted(_Presentation):
    """A presenter that answers as a completed codex response and keeps its evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.evidence: dict[str, Any] | None = None

    async def complete(self, *args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        self.evidence = dict(args[4])
        return SimpleNamespace(status="completed", usage=None, output=[
            _codex_message(_answer(LEAD, "The retrieved report supports this result.", [0]))])


async def _codex_run(observer: NativeWebEvidence, response: Any, shape: str) -> Any:
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    if shape in {"web", "mixed"}:
        _web_turn(observer)
    if shape == "mixed":
        observer.tool_execution(tool_name="terminal", args={"command": "..."},
                                next_call=lambda _args: "tool output")
    return await asyncio.to_thread(_codex_turn, observer, response)


# The shared codex parser checks shape only; every presenter path keeps its own bound.
LONG_FINAL = "x" * 8193


@pytest.mark.asyncio
async def test_codex_web_only_long_native_final_is_presented(capsys: Any) -> None:
    presentation = _CodexFormatted()
    observer = _observer(presentation)
    returned = await _codex_run(observer, _codex_final(_codex_message(LONG_FINAL)), "web")

    assert presentation.calls == 1
    assert returned.output[0].content[0].text.startswith("<voice>" + LEAD + "</voice>")
    assert json.loads(observer.reports[RUN_ID])["disposition"] == "presented"
    assert not [marker for marker in _markers(capsys) if marker["category"] == "refused"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [None, "final_answer"])
async def test_codex_intermediate_turn_with_long_message_returns_its_calls(
    phase: str | None, capsys: Any,
) -> None:
    presentation = _CodexFormatted()
    observer = _observer(presentation)
    call = SimpleNamespace(type="function_call", status="completed", call_id="call_1",
                           name="web_search", arguments='{"query": "more"}')
    message = _codex_message(LONG_FINAL)
    message.phase = phase
    response = _codex_final(message, call)
    returned = await _codex_run(observer, response, "web")

    assert returned is response
    assert presentation.calls == 0
    assert observer.rows[RUN_ID]["disposition"] is None
    assert _markers(capsys) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("shape", "disposition"), [
    ("no_web", "no_web"), ("mixed", "mixed_ungrounded")])
async def test_codex_long_native_final_keeps_unpresented_disposition(
    shape: str, disposition: str, capsys: Any,
) -> None:
    presentation = _CodexFormatted()
    observer = _observer(presentation)
    response = _codex_final(_codex_message(LONG_FINAL))
    returned = await _codex_run(observer, response, shape)

    assert returned is response
    assert presentation.calls == 0
    assert json.loads(observer.reports[RUN_ID])["disposition"] == disposition
    assert not [marker for marker in _markers(capsys) if marker["category"] == "refused"]


LONE_SURROGATE = "Native draft \ud800 with a lone surrogate."


@pytest.mark.asyncio
async def test_codex_web_only_lone_surrogate_final_is_presented() -> None:
    # The native text feeds only the local diagnostic hash on a presented run.
    presentation = _CodexFormatted()
    observer = _observer(presentation)
    returned = await _codex_run(
        observer, _codex_final(_codex_message(LONE_SURROGATE)), "web")

    assert returned.output[0].content[0].text.startswith("<voice>" + LEAD + "</voice>")
    assert presentation.evidence is not None
    assert presentation.evidence["source_sha256"] == hashlib.sha256(
        LONE_SURROGATE.encode("utf-8", errors="surrogatepass")).hexdigest()
    assert json.loads(observer.reports[RUN_ID])["disposition"] == "presented"


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["no_web", "mixed"])
async def test_codex_lone_surrogate_unpresented_final_stays_strict(
    shape: str, capsys: Any,
) -> None:
    # The host strict-encodes the output when it checks the attestation, so these stay strict.
    presentation = _CodexFormatted()
    observer = _observer(presentation)
    returned = await _codex_run(
        observer, _codex_final(_codex_message(LONE_SURROGATE)), shape)

    assert returned.output_text == '{"error": "policy_refused"}'
    assert presentation.calls == 0
    [marker] = _markers(capsys)
    assert (marker["category"], marker["reason"], marker["phase"]) == (
        "refused", "internal", "final")


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


# A native final that did not finish is one fact; both routes name it the same way (#557).
def _incomplete_chat_final() -> Any:
    response = _final_response()
    response.choices[0].finish_reason = "length"
    return response


@pytest.mark.asyncio
async def test_codex_incomplete_native_final_names_final_response(capsys: Any) -> None:
    response = SimpleNamespace(status="incomplete", output=[_codex_message("Native draft.")])
    await _refused_codex(_observer(), response)

    [marker] = _markers(capsys)
    assert (marker["reason"], marker["cause"], marker["phase"]) == (
        "final_response", "native-final-response", "final")


@pytest.mark.asyncio
async def test_chat_unfinished_native_final_keeps_final_response(
    monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    observer = _observer()
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    _web_turn(observer)
    returned = await asyncio.to_thread(_final_turn, observer, _incomplete_chat_final())

    assert returned.choices[0].message.content == '{"error": "policy_refused"}'
    [marker] = _markers(capsys)
    assert (marker["reason"], marker["cause"]) == ("final_response", "native-final-response")


class _IncompletePresenter(_Presentation):
    async def complete(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(status="incomplete", usage=None, output=[
            _codex_message(_answer(LEAD, "The retrieved report supports this result.", [0]))])


@pytest.mark.asyncio
async def test_codex_incomplete_presenter_response_keeps_its_own_category(
    capsys: Any,
) -> None:
    observer = _observer(_IncompletePresenter())
    returned = await _codex_run(observer, _codex_final(_codex_message("Native draft.")), "web")

    assert returned.output_text == '{"error": "policy_refused"}'
    [marker] = _markers(capsys)
    assert marker["reason"] != "final_response"
    assert (marker["cause"], marker["phase"]) == ("responses-not-completed", "render")


@pytest.mark.asyncio
async def test_codex_incomplete_intermediate_turn_is_refused_as_a_final(capsys: Any) -> None:
    # An unfinished turn carries no completed calls to forward.
    call = SimpleNamespace(type="function_call", status="completed", call_id="call_1",
                           name="web_search", arguments='{"query": "more"}')
    response = SimpleNamespace(status="incomplete", output=[call])
    await _refused_codex(_observer(), response)

    [marker] = _markers(capsys)
    assert marker["cause"] == "native-final-response"
