"""Offline producer/consumer evidence; no inference server or captured model text."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from hermes_realtime.providers import ollama
from scripts import evaluate_natural_work_scenario as scenario
from tests.support.ollama_metadata import supported_ollama_metadata


class ForeignDecisionRefusal(ollama.OllamaDecisionRefused):
    """A caller-owned exception cannot assert the adapter's refusal provenance."""


@pytest.fixture
def offline_cases(monkeypatch):
    instances = []
    real_factory = scenario.factory

    def install(content=None, *, failure=None, cleanup_failure=None):
        def factory(**kwargs):
            inference = real_factory(**kwargs)
            instances.append(inference)
            inference._model_client = supported_ollama_metadata()

            async def configure_context(*args):
                pass

            inference.configure_context = configure_context

            def opened(*args):
                if failure is not None:
                    raise failure
                return io.BytesIO(json.dumps({
                    "message": {"content": content}, "done": True, "done_reason": "stop",
                }).encode() + b"\n")

            inference._open_request = opened
            if cleanup_failure is not None:
                close = inference.close

                async def failed_close():
                    await close()
                    raise cleanup_failure

                inference.close = failed_close
            return inference

        monkeypatch.setattr(scenario, "factory", factory)

    async def preflight(**kwargs):
        kwargs["inference"].bind_work_tools(kwargs["surface"])

    monkeypatch.setattr(scenario, "preflight", preflight)
    value = scenario.DevelopmentDiagnostic(cases=tuple(
        scenario.legacy.IntentCase(f"synthetic_{i}", "negative", "Synthetic request.", None)
        for i in range(2)
    ), bindings={})
    return install, value, instances


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,category", [
    ({"kind": "reply", "text": "I've started that background task."}, "unverified_claim"),
    ({"kind": "reply", "text": " "}, "reply_blank"),
    ({"kind": "reply", "text": 1}, "reply_type"),
    ({"kind": "reply", "text": "\ud800"}, "reply_unicode"),
    ({"kind": "start_work", "objective": " "}, "objective_blank"),
])
async def test_real_decision_refusals_retain_category_and_continue(
    offline_cases, decision, category,
):
    install, value, instances = offline_cases
    install(json.dumps(decision))
    rows = await scenario.run_cases(value, model="test", base_url="http://127.0.0.1:11434")
    assert len(rows) == len(instances) == 2
    assert all(row["error"] == "decision_refused" for row in rows)
    assert all(row["refusal_category"] == category for row in rows)
    assert all(not row["usable_decision"] for row in rows)
    assert all(row["speech"] == row["proposals"] == row["shadow_dispatches"] == [] for row in rows)
    assert all(instance._closed for instance in instances)


@pytest.mark.asyncio
async def test_valid_reply_remains_usable(offline_cases):
    install, value, instances = offline_cases
    install(json.dumps({"kind": "reply", "text": "Which baseline should I use?"}))
    rows = await scenario.run_cases(value, model="test", base_url="http://127.0.0.1:11434")
    assert len(rows) == len(instances) == 2
    assert all(row["usable_decision"] and row["error"] is None for row in rows)
    assert all(row.get("refusal_category") is None for row in rows)
    assert all(instance._closed for instance in instances)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,category", [
    ("_MAX_WORK_RESPONSE_BYTES", "response_bytes"),
    ("_MAX_WORK_RESPONSE_CHUNKS", "response_chunks"),
    ("_MAX_WORK_DECISION_BYTES", "decision_bytes"),
])
async def test_resource_limits_remain_operational(offline_cases, monkeypatch, limit, category):
    install, value, instances = offline_cases
    monkeypatch.setattr(ollama, limit, 0)
    install(json.dumps({"kind": "reply", "text": "Synthetic answer."}))
    rows = []
    with pytest.raises(ValueError, match=category) as caught:
        await scenario.run_cases(value, model="test", base_url="http://127.0.0.1:11434",
                                 observations=rows)
    assert type(caught.value) is ValueError
    assert len(rows) == len(instances) == 1
    assert rows[0]["error"] == "operational_failure"
    assert not rows[0]["usable_decision"]
    assert rows[0].get("refusal_category") is None
    assert instances[0]._closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    ValueError("Ollama work refused: unverified_claim"),
    ValueError("Ollama work refused: reply_blank"),
    ValueError("synthetic internal error"),
    OSError("synthetic transport error"),
    TimeoutError("synthetic deadline"),
    RuntimeError("Ollama work unavailable: ownership_before_admission"),
    asyncio.CancelledError(),
    ForeignDecisionRefusal("unverified_claim"),
], ids=["spoof_claim", "spoof_schema", "unknown", "transport", "deadline", "ownership", "cancel",
        "foreign_subclass"])
async def test_operational_errors_stop_after_first_case(offline_cases, failure):
    install, value, instances = offline_cases
    install(failure=failure)
    rows = []
    expected = ValueError if type(failure) is TimeoutError else type(failure)
    with pytest.raises(expected) as caught:
        await scenario.run_cases(value, model="test", base_url="http://127.0.0.1:11434",
                                 observations=rows)
    if not isinstance(failure, (TimeoutError, asyncio.CancelledError)):
        assert caught.value is failure
    assert len(rows) == len(instances) == 1
    assert rows[0]["error"] == "operational_failure"
    assert not rows[0]["usable_decision"]
    assert rows[0].get("refusal_category") is None
    assert instances[0]._closed


@pytest.mark.asyncio
async def test_decision_refusal_does_not_hide_cleanup_failure(offline_cases):
    install, value, instances = offline_cases
    failure = RuntimeError("synthetic cleanup failure")
    install(json.dumps({"kind": "reply", "text": " "}), cleanup_failure=failure)
    rows = []
    with pytest.raises(RuntimeError) as caught:
        await scenario.run_cases(value, model="test", base_url="http://127.0.0.1:11434",
                                 observations=rows)
    assert caught.value is failure
    assert len(rows) == len(instances) == 1
    assert rows[0]["error"] == "decision_refused"
    assert rows[0]["refusal_category"] == "reply_blank"
    assert rows[0]["cleanup_failed"] and not rows[0]["usable_decision"]
    assert instances[0]._closed
