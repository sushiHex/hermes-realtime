"""Model-free controls for the existing natural-work evaluator's provider seam."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import time

import pytest

from hermes_realtime.conversation.task_status import HostSpeechRequest
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import OllamaModelClient
from scripts import evaluate_natural_work_routing as evaluator
from tests.support.ollama_metadata import supported_ollama_tags


class _Inference:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.events: list[tuple[str, object]] = []

    async def configure_context(self, tokens: int) -> None:
        self.events.append(("context", tokens))

    async def stream(self, snapshot: object, *, turn_id: str):
        self.events.append(("stream", turn_id))
        if turn_id == "turn_preflight":
            if self.ready:
                yield "Ready."
            return
        yield "Answer."

    def bind_work_tools(self, handler: object) -> None:
        self.events.append(("bind", handler))

    async def close(self) -> None:
        self.events.append(("close", None))


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "codex"])
async def test_status_request_is_separate_from_model_prose_and_effects(
    monkeypatch: pytest.MonkeyPatch, provider: str,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    instances: list[_Inference] = []

    class Status(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            if turn_id == "turn_preflight":
                yield "Ready."
            else:
                yield HostSpeechRequest()

    def factory(**_options: object) -> Status:
        instance = Status()
        instances.append(instance)
        return instance

    monkeypatch.setattr(evaluator, "_build_streaming_inference", factory)
    monkeypatch.setattr(evaluator, "_resolve_codex_executable", lambda _value: "synthetic")
    monkeypatch.setattr(evaluator, "_codex_version", lambda _value: "1.0.0")
    report = await evaluator.run_evaluation(
        corpus=one_case, provider=provider, model="synthetic:latest", effort="low",
        codex_executable=None,
    )
    assert report["schema_version"] == 5
    status = report["host_status_requests"]
    assert status["count"] == 1 and status["case_ids"] == ["nw-001"]
    assert status["scope"] == "selection_only"
    assert status["latency_ms"]["count"] == 1
    assert report["tool_calls"]["total"] == 0
    assert report["latency_ms"]["first_speakable"]["count"] == 0
    assert report["metrics"]["positive_start"]["correct"] == 0
    assert report["semantic_review"]["status"] == "pending"
    assert report["status"] == "failed"
    assert len(instances) == 1 and instances[0].events[-1] == ("close", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("later", [HostSpeechRequest(), "Later prose.", object()])
async def test_status_request_is_terminal_for_evaluator_output(
    monkeypatch: pytest.MonkeyPatch, later: object,
) -> None:
    class Status(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            if turn_id == "turn_preflight":
                yield "Ready."
            else:
                yield HostSpeechRequest()
                yield later

    instance = Status()
    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: instance)
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    with pytest.raises(evaluator.EvaluationFailure) as failure:
        await evaluator.run_evaluation(
            corpus=dataclasses.replace(corpus, cases=corpus.cases[:1]),
            provider="ollama", model="synthetic:latest", effort="low", codex_executable=None,
        )
    assert str(failure.value.__cause__) == "qualification output followed terminal status request"
    assert failure.value.public_error()["completed_cases"] == 0
    assert instance.events[-1] == ("close", None)


@pytest.mark.parametrize("value", [True, -1.0, float("inf"), float("nan"), "0"])
def test_host_status_request_latency_is_strict(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        evaluator.CaseObservation(
            "nw-001", (), (), None, host_status_request_latency_ms=value,
        )


def test_status_observation_requires_owned_case_and_no_lifecycle_effects() -> None:
    handler = evaluator.ShadowWorkHandler(can_cancel_work=False)
    with pytest.raises(RuntimeError, match="active case"):
        handler.observe_host_status_request()
    handler.begin_case("nw-001", started_at=time.perf_counter())
    handler.observe_host_status_request()
    with pytest.raises(RuntimeError, match="terminal status"):
        handler.observe_host_status_request()
    with pytest.raises(RuntimeError, match="terminal status"):
        handler._record("start_work", "tool_" + "a" * 32)
    observation = handler.finish_case(first_speakable_latency_ms=None)
    assert observation.host_status_request_latency_ms is not None
    handler.begin_case("nw-002", started_at=time.perf_counter())
    handler._record("start_work", "tool_" + "a" * 32)
    with pytest.raises(RuntimeError, match="lifecycle"):
        handler.observe_host_status_request()
    handler.abandon_case()
    handler.begin_case("nw-003", started_at=time.perf_counter())
    reset = handler.finish_case(first_speakable_latency_ms=None)
    assert reset.host_status_request_latency_ms is None


def test_observation_host_status_latency_control_and_mixed_effect_refusal() -> None:
    assert evaluator.CaseObservation(
        "nw-001", (), (), None, host_status_request_latency_ms=0.0,
    ).host_status_request_latency_ms == 0.0
    with pytest.raises(ValueError, match="lifecycle"):
        evaluator.CaseObservation(
            "nw-001", ("start_work",), (0.0,), None, host_status_request_latency_ms=0.0,
        )


@pytest.mark.parametrize("mismatch", ["module_path", "class_identity"])
def test_status_source_binding_is_exact(
    monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    from hermes_realtime.conversation import task_status

    evaluator._verify_controller_source()
    if mismatch == "module_path":
        monkeypatch.setattr(task_status, "__file__", __file__)
    else:
        monkeypatch.setattr(evaluator, "HostSpeechRequest", type("Other", (), {}))
    with pytest.raises(ValueError, match="evaluator_(source|script)_binding"):
        evaluator._verify_controller_source()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_prose", [False, True])
async def test_real_ollama_status_selection_reports_no_rendered_host_speech(
    monkeypatch: pytest.MonkeyPatch, prior_prose: bool,
) -> None:
    if prior_prose:
        class Prior(_Inference):
            async def stream(self, snapshot: object, *, turn_id: str):
                if turn_id == "turn_preflight":
                    yield "Ready."
                else:
                    yield "Earlier model prose."
                    yield HostSpeechRequest()

        monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: Prior())
    else:
        async def metadata(_self: OllamaModelClient, endpoint: str, body: object) -> bytes:
            if endpoint == "tags":
                return supported_ollama_tags("synthetic:latest")
            return b'{"version":"0.34.4"}'

        def opening(request: object, _timeout: float) -> io.BytesIO:
            body = json.loads(request.data)  # type: ignore[attr-defined]
            content = json.dumps({"kind": "task_status"}) if "format" in body else "Ready."
            return io.BytesIO(json.dumps({
                "message": {"content": content}, "done": True, "done_reason": "stop",
            }).encode() + b"\n")

        monkeypatch.setattr(OllamaModelClient, "_read_request", metadata)
        monkeypatch.setattr(OllamaStreamingInference, "_urlopen", staticmethod(opening))
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    report = await evaluator.run_evaluation(
        corpus=dataclasses.replace(corpus, cases=corpus.cases[:2]),
        provider="ollama", model="synthetic:latest", effort="low", codex_executable=None,
    )
    assert report["host_status_requests"]["count"] == 2
    assert report["host_status_requests"]["case_ids"] == ["nw-001", "nw-002"]
    assert report["host_status_requests"]["latency_ms"]["count"] == 2
    assert report["latency_ms"]["first_speakable"]["count"] == (2 if prior_prose else 0)
    assert report["tool_calls"]["total"] == 0
    assert report["semantic_review"]["status"] == "pending"
    assert "Earlier model prose." not in json.dumps(report)


@pytest.mark.asyncio
async def test_status_marker_subclass_is_not_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    class Untrusted(HostSpeechRequest):
        pass

    class Invalid(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            yield "Ready." if turn_id == "turn_preflight" else Untrusted()

    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: Invalid())
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    with pytest.raises(evaluator.EvaluationFailure) as failure:
        await evaluator.run_evaluation(
            corpus=dataclasses.replace(corpus, cases=corpus.cases[:1]),
            provider="ollama", model="synthetic:latest", effort="low", codex_executable=None,
        )
    assert str(failure.value.__cause__) == "qualification decision emitted unsupported output"


@pytest.mark.asyncio
async def test_status_report_equals_only_marker_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    class Mixed(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            if turn_id == "turn_preflight":
                yield "Ready."
            elif "nw_002" in turn_id:
                yield HostSpeechRequest()
            else:
                yield "Ordinary model prose."

    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: Mixed())
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    report = await evaluator.run_evaluation(
        corpus=dataclasses.replace(corpus, cases=corpus.cases[:2]),
        provider="ollama", model="synthetic", effort="low", codex_executable=None,
    )
    status = report["host_status_requests"]
    assert status["case_ids"] == ["nw-002"]
    assert status["count"] == status["latency_ms"]["count"] == 1
    assert report["latency_ms"]["first_speakable"]["count"] == 1


@pytest.mark.asyncio
async def test_real_codex_transport_status_selection_remains_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
    from tests.providers.test_codex_app_server import (
        DynamicStartCodexTransport,
        FakeCodexTransport,
    )

    class ReadyThenStatus(DynamicStartCodexTransport):
        def __init__(self) -> None:
            super().__init__(tool="task_status", arguments={})
            self.turns = 0

        async def send(self, message):
            if message.get("method") == "turn/start":
                self.turns += 1
                if self.turns == 1:
                    await FakeCodexTransport.send(self, message)
                    return
            await super().send(message)

    transport = ReadyThenStatus()
    monkeypatch.setattr(evaluator, "_resolve_codex_executable", lambda _value: "synthetic")
    monkeypatch.setattr(evaluator, "_codex_version", lambda _value: "1.0.0")
    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options:
        CodexAppServerStreamingInference(
            model="synthetic", effort="low", transport_factory=lambda: transport,
        )
    )
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    report = await evaluator.run_evaluation(
        corpus=dataclasses.replace(corpus, cases=corpus.cases[:1]),
        provider="codex", model="synthetic", effort="low", codex_executable=None,
    )
    assert report["host_status_requests"]["case_ids"] == ["nw-001"]
    assert report["tool_calls"]["total"] == 0
    assert report["latency_ms"]["first_speakable"]["count"] == 0
    assert transport.closed is True


@pytest.mark.asyncio
async def test_ollama_qualification_uses_host_factory_context_preflight_then_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = _Inference()
    factory_calls: list[dict[str, object]] = []

    def factory(**options: object) -> _Inference:
        factory_calls.append(options)
        return inference

    monkeypatch.setattr(evaluator, "_build_streaming_inference", factory)
    handler = evaluator.ShadowWorkHandler(can_cancel_work=False)
    result = await evaluator._open_qualified_inference(
        provider="ollama", model="synthetic:latest", effort="low",
        codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        ollama_num_ctx=4096, handler=handler,
    )
    assert result is inference
    assert factory_calls[0]["inference_provider"] == "ollama"
    assert factory_calls[0]["ollama_model"] == "synthetic:latest"
    assert inference.events == [
        ("context", 4096), ("stream", "turn_preflight"), ("bind", handler),
    ]


@pytest.mark.asyncio
async def test_failed_readiness_closes_adapter_without_tool_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = _Inference(ready=False)
    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: inference)
    handler = evaluator.ShadowWorkHandler(can_cancel_work=False)
    with pytest.raises(RuntimeError, match="preflight"):
        await evaluator._open_qualified_inference(
            provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
            ollama_num_ctx=4096, handler=handler,
        )
    assert inference.events == [
        ("context", 4096), ("stream", "turn_preflight"), ("close", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("unspeakable", ["", "  ", object(), HostSpeechRequest()])
async def test_preflight_refuses_non_speech_before_tool_binding(
    monkeypatch: pytest.MonkeyPatch, unspeakable: object,
) -> None:
    class Invalid(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            self.events.append(("stream", turn_id))
            if type(unspeakable) is not str:
                yield "Ready."
            yield unspeakable

    inference = Invalid()
    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: inference)
    with pytest.raises(RuntimeError, match="preflight"):
        await evaluator._open_qualified_inference(
            provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
            ollama_num_ctx=None,
            handler=evaluator.ShadowWorkHandler(can_cancel_work=False),
        )
    assert [event[0] for event in inference.events] == ["stream", "close"]


@pytest.mark.asyncio
async def test_evaluator_uses_configured_ollama_adapter_for_decision_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    instances: list[_Inference] = []

    class Decision(_Inference):
        def __init__(self) -> None:
            super().__init__()
            self.handler: evaluator.ShadowWorkHandler | None = None

        def bind_work_tools(self, handler: evaluator.ShadowWorkHandler) -> None:
            super().bind_work_tools(handler)
            self.handler = handler

        async def stream(self, snapshot: object, *, turn_id: str):
            self.events.append(("stream", turn_id))
            if turn_id == "turn_preflight":
                yield "Ready."
                return
            assert self.handler is not None
            await self.handler.start_work(
                objective="Inspect synthetic release evidence",
                invocation_id="tool_" + "a" * 32,
            )
            yield "A" * 5000  # Capture must not add an earlier semantic limit.

    def factory(**options: object) -> Decision:
        assert options["inference_provider"] == "ollama"
        assert options["ollama_num_gpu"] == 0
        value = Decision()
        instances.append(value)
        return value

    monkeypatch.setattr(evaluator, "_build_streaming_inference", factory)
    report = await evaluator.run_evaluation(
        corpus=one_case, provider="ollama", mode="diagnostic_override",
        model="synthetic:latest", effort="low", codex_executable=None,
        ollama_base_url="http://127.0.0.1:11434", ollama_num_ctx=4096,
        ollama_num_gpu=0,
    )
    assert report["provider"] == "ollama"
    assert report["mode"] == "diagnostic_override"
    assert report["ollama_gpu_layers"] == 0
    assert report["tool_calls"]["by_tool"]["start_work"] == 1
    assert report["host_status_requests"]["count"] == 0
    assert report["host_status_requests"]["case_ids"] == []
    assert report["host_status_requests"]["latency_ms"]["count"] == 0
    assert report["process_evidence"]["external_runtime_cache"] == "unknown"
    assert "A" * 5000 not in str(report)
    assert len(instances) == 1
    assert instances[0].events[0:3] == [
        ("context", 4096), ("stream", "turn_preflight"),
        ("bind", instances[0].handler),
    ]
    assert instances[0].events[-1] == ("close", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [True, False])
async def test_real_host_ollama_adapter_observes_effect_free_ack(
    monkeypatch: pytest.MonkeyPatch, accept: bool,
) -> None:
    requests: list[dict[str, object]] = []

    async def metadata(_self: OllamaModelClient, endpoint: str, body: object) -> bytes:
        assert body is None
        if endpoint == "tags":
            return supported_ollama_tags("synthetic:latest")
        assert endpoint == "version"
        return b'{"version":"0.34.4"}'

    def opening(request: object, _timeout: float) -> io.BytesIO:
        body = json.loads(request.data)  # type: ignore[attr-defined]
        requests.append(body)
        decision = (
            {"kind": "start_work", "objective": "Inspect synthetic release"}
            if "format" in body else None
        )
        content = json.dumps(decision) if decision is not None else "Ready."
        payload = {
            "message": {"content": content}, "done": True, "done_reason": "stop",
        }
        return io.BytesIO(json.dumps(payload).encode() + b"\n")

    monkeypatch.setattr(OllamaModelClient, "_read_request", metadata)
    monkeypatch.setattr(OllamaStreamingInference, "_urlopen", staticmethod(opening))
    handler = evaluator.ShadowWorkHandler(
        can_cancel_work=False, accept_start=accept,
    )
    inference = await evaluator._open_qualified_inference(
        provider="ollama", model="synthetic:latest", effort="low",
        codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        ollama_num_ctx=None, handler=handler,
    )
    try:
        handler.begin_case("nw-001", started_at=time.perf_counter())
        case = evaluator.load_corpus(evaluator._DEFAULT_CORPUS).cases[0]
        snapshot = evaluator._snapshot(case, 1)
        spoken = [part async for part in inference.stream(snapshot, turn_id="turn_ack")]
        assert spoken == [
            "I've started that background task." if accept
            else "The background task was not accepted."
        ]
        observation = handler.finish_case(first_speakable_latency_ms=0.0)
        assert observation.observed_tools == ("start_work",)
        assert len(requests) == 2
        assert "format" not in requests[0] and "format" in requests[1]
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_real_ollama_adapter_retains_provider_response_bound_without_capture_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def metadata(_self: OllamaModelClient, endpoint: str, body: object) -> bytes:
        assert body is None
        if endpoint == "tags":
            return supported_ollama_tags("synthetic:latest")
        assert endpoint == "version"
        return b'{"version":"0.34.4"}'

    def opening(request: object, _timeout: float) -> io.BytesIO:
        nonlocal calls
        calls += 1
        body = json.loads(request.data)  # type: ignore[attr-defined]
        content = (
            json.dumps({"kind": "reply", "text": "R" * 5000})
            if "format" in body else "Ready."
        )
        return io.BytesIO(json.dumps({
            "message": {"content": content}, "done": True, "done_reason": "stop",
        }).encode() + b"\n")

    monkeypatch.setattr(OllamaModelClient, "_read_request", metadata)
    monkeypatch.setattr(OllamaStreamingInference, "_urlopen", staticmethod(opening))
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    report = await evaluator.run_evaluation(
        corpus=one_case, provider="ollama", model="synthetic:latest", effort="low",
        codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
    )
    assert calls == 2
    assert report["tool_calls"]["total"] == 0
    assert report["latency_ms"]["first_speakable"]["count"] == 1
    assert "R" * 5000 not in str(report)


@pytest.mark.asyncio
async def test_unknown_decision_item_refuses_without_latency_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    instances: list[_Inference] = []

    class TypedDecision(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            if turn_id == "turn_preflight":
                yield "Ready."
            else:
                yield "Starting."
                yield object()

    def factory(**_options: object) -> TypedDecision:
        instance = TypedDecision()
        instances.append(instance)
        return instance

    monkeypatch.setattr(evaluator, "_build_streaming_inference", factory)
    with pytest.raises(evaluator.EvaluationFailure) as failure:
        await evaluator.run_evaluation(
            corpus=one_case, provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        )
    assert failure.value.public_error() == {
        "code": "provider_turn_failed", "case_id": "nw-001", "completed_cases": 0,
        "attempted_cases": 1, "failed_attempts": 1,
    }
    assert len(instances) == 1
    assert instances[0].events[-1] == ("close", None)


@pytest.mark.asyncio
async def test_cleanup_refusal_has_distinct_bounded_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    attempts: list[_Inference] = []

    class BrokenClose(_Inference):
        async def close(self) -> None:
            await super().close()
            raise RuntimeError("private cleanup path")

    def factory(**_options: object) -> BrokenClose:
        instance = BrokenClose()
        attempts.append(instance)
        return instance

    monkeypatch.setattr(evaluator, "_build_streaming_inference", factory)
    with pytest.raises(evaluator.EvaluationFailure) as failure:
        await evaluator.run_evaluation(
            corpus=one_case, provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        )
    assert failure.value.public_error() == {
        "code": "provider_cleanup_failed", "case_id": "nw-001", "completed_cases": 0,
        "attempted_cases": 1, "failed_attempts": 1,
    }
    assert len(attempts) == 1 and attempts[0].events[-1] == ("close", None)


@pytest.mark.asyncio
async def test_readiness_cleanup_refusal_is_not_reported_as_plain_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])

    class BrokenPreflight(_Inference):
        def __init__(self) -> None:
            super().__init__(ready=False)

        async def close(self) -> None:
            raise RuntimeError("private cleanup path")

    monkeypatch.setattr(
        evaluator, "_build_streaming_inference", lambda **_options: BrokenPreflight(),
    )
    with pytest.raises(evaluator.EvaluationFailure) as failure:
        await evaluator.run_evaluation(
            corpus=one_case, provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        )
    assert failure.value.public_error() == {
        "code": "provider_readiness_failed", "case_id": "nw-001", "completed_cases": 0,
        "attempted_cases": 1, "failed_attempts": 1,
        "secondary_categories": ["provider_cleanup_failed"],
    }


@pytest.mark.asyncio
async def test_final_case_close_crossing_total_deadline_cannot_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    monkeypatch.setattr(evaluator, "_EVALUATION_BUDGET_SECONDS", 0.1)

    class SlowClose(_Inference):
        async def close(self) -> None:
            await asyncio.sleep(0.15)
            await super().close()

    inference = SlowClose()
    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: inference)
    with pytest.raises(evaluator.EvaluationFailure, match="evaluation_timed_out") as failure:
        await evaluator.run_evaluation(
            corpus=one_case, provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        )
    assert failure.value.public_error()["attempted_cases"] == 1
    assert inference.events[-1] == ("close", None)


@pytest.mark.asyncio
async def test_decision_and_cleanup_failures_retain_both_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])

    class BrokenBoth(_Inference):
        async def stream(self, snapshot: object, *, turn_id: str):
            if turn_id == "turn_preflight":
                yield "Ready."
            else:
                raise RuntimeError("private decision detail")
                if False:
                    yield ""

        async def close(self) -> None:
            raise RuntimeError("private cleanup detail")

    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_options: BrokenBoth())
    with pytest.raises(evaluator.EvaluationFailure) as failure:
        await evaluator.run_evaluation(
            corpus=one_case, provider="ollama", model="synthetic:latest", effort="low",
            codex_executable=None, ollama_base_url="http://127.0.0.1:11434",
        )
    assert failure.value.public_error() == {
        "code": "provider_turn_failed", "case_id": "nw-001", "completed_cases": 0,
        "attempted_cases": 1, "failed_attempts": 1,
        "secondary_categories": ["provider_cleanup_failed"],
    }


def test_production_default_refuses_diagnostic_override_flags_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "--provider", "ollama", "--mode", "production_default",
        "--model", "synthetic:latest",
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
    ]
    assert evaluator._parser().parse_args(arguments).provider == "ollama"
    monkeypatch.setattr(evaluator, "_verify_source_binding", lambda _head, _tree: None)
    monkeypatch.setattr(
        evaluator, "run_evaluation",
        lambda **_options: pytest.fail("provider was invoked before mode refusal"),
    )
    assert evaluator.main(arguments) == 2
    output = capsys.readouterr().out
    assert '"status":"error"' in output
    assert "synthetic:latest" not in output


def test_resolved_provider_options_use_configured_host_defaults_for_omitted_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = evaluator._host_provider_defaults()
    defaults["ollama_num_ctx"] = 8192
    defaults["codex_executable"] = "bound-codex"
    monkeypatch.setattr(evaluator, "_host_provider_defaults", lambda: defaults)
    ollama = evaluator._resolved_provider_options(evaluator._parser().parse_args([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40,
    ]))
    codex = evaluator._resolved_provider_options(evaluator._parser().parse_args([
        "--provider", "codex", "--source-head", "a" * 40,
        "--source-tree", "b" * 40,
    ]))
    assert ollama["ollama_num_ctx"] == 8192
    assert codex["codex_executable"] == "bound-codex"


def test_missing_source_binding_refuses_before_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(evaluator, "_verify_controller_source", lambda: None)
    monkeypatch.setattr(
        evaluator, "run_evaluation",
        lambda **_options: pytest.fail("unbound CLI reached provider"),
    )
    assert evaluator.main(["--provider", "ollama"]) == 2
    assert '"status":"error"' in capsys.readouterr().out


@pytest.mark.parametrize("observed", [
    ("c" * 40, "b" * 40, True),
    ("a" * 40, "c" * 40, True),
    ("a" * 40, "b" * 40, False),
])
def test_source_head_tree_or_cleanliness_drift_refuses_before_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    observed: tuple[str, str, bool],
) -> None:
    head, tree = "a" * 40, "b" * 40
    monkeypatch.setattr(evaluator, "_verify_controller_source", lambda: None)
    monkeypatch.setattr(evaluator, "_git_source", lambda: observed)
    monkeypatch.setattr(
        evaluator, "run_evaluation",
        lambda **_options: pytest.fail("wrong tree reached provider"),
    )
    assert evaluator.main([
        "--source-head", head, "--source-tree", tree, "--provider", "ollama",
    ]) == 2
    assert '"code":"source_binding_failed"' in capsys.readouterr().out


@pytest.mark.parametrize("valid", [True, False])
def test_owned_failure_distinguishes_valid_unsuccessful_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str],
    valid: bool,
) -> None:
    from scripts import qualification_launch

    receipt = tmp_path / "receipt.jsonl"
    if valid:
        receipt.write_text(json.dumps({
            "stage": "incomplete", "category": "execution_failed",
            "controller_completed": False, "cleanup_verified": True,
        }) + "\n")
    else:
        receipt.write_text("{private invalid\n")

    def failed(*_args: object, **_options: object) -> None:
        raise RuntimeError("private provider path")

    monkeypatch.setattr(qualification_launch, "run_owned_controller", failed)
    monkeypatch.setattr(evaluator, "_verify_source_binding", lambda _head, _tree: None)
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40, "--launch-binding", str(tmp_path / "binding.json"),
        "--launch-binding-sha256", "c" * 64, "--launch-receipt", str(receipt),
    ]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "owned_launch_failed"
    assert output["error"]["receipt_status"] == (
        "valid_unsuccessful" if valid else "missing_or_malformed"
    )
    if valid:
        assert output["error"]["category"] == "execution_failed"
        assert output["error"]["cleanup_verified"] is True
    assert "private provider path" not in str(output)


def test_source_drift_after_owned_failure_preserves_receipt_category(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import qualification_launch

    receipt = tmp_path / "receipt.jsonl"
    receipt.write_text(json.dumps({
        "stage": "incomplete", "category": "guardian_inherited_job",
        "controller_completed": False, "cleanup_verified": False,
    }) + "\n")
    checks = 0

    def source(_head: str, _tree: str) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise evaluator.SourceBindingFailure("source_binding_failed")

    def failed(*_args: object, **_options: object) -> None:
        raise RuntimeError("private guardian detail")

    monkeypatch.setattr(evaluator, "_verify_source_binding", source)
    monkeypatch.setattr(qualification_launch, "run_owned_controller", failed)
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40, "--launch-binding", str(tmp_path / "binding.json"),
        "--launch-binding-sha256", "c" * 64, "--launch-receipt", str(receipt),
    ]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "owned_launch_failed"
    assert error["category"] == "guardian_inherited_job"
    assert error["secondary_categories"] == ["source_binding_failed"]


def test_source_drift_after_nonzero_owned_result_preserves_result_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import qualification_launch

    checks = 0

    def source(_head: str, _tree: str) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise evaluator.SourceBindingFailure("source_binding_failed")

    monkeypatch.setattr(evaluator, "_verify_source_binding", source)
    monkeypatch.setattr(
        qualification_launch, "run_owned_controller",
        lambda *_args, **_options: qualification_launch.LaunchResult(
            1, b"private provider response"
        ),
    )
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40, "--launch-binding", str(tmp_path / "binding.json"),
        "--launch-binding-sha256", "c" * 64,
        "--launch-receipt", str(tmp_path / "receipt.jsonl"),
    ]) == 2
    output = capsys.readouterr().out
    assert "private provider response" not in output
    error = json.loads(output)["error"]
    assert error["code"] == "owned_result_failed"
    assert error["secondary_categories"] == ["source_binding_failed"]


def test_source_is_rechecked_after_provider_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checks: list[tuple[str, str]] = []

    def source(head: str, tree: str) -> None:
        checks.append((head, tree))

    async def failed(**_options: object) -> dict[str, object]:
        raise evaluator.EvaluationFailure(
            "provider_turn_failed", case_id="nw-002", completed_cases=1,
        )

    monkeypatch.setattr(evaluator, "_verify_source_binding", source)
    monkeypatch.setattr(evaluator, "run_evaluation", failed)
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40,
    ]) == 2
    assert checks == [("a" * 40, "b" * 40)] * 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"] == {
        "code": "provider_turn_failed", "case_id": "nw-002", "completed_cases": 1,
        "attempted_cases": 2, "failed_attempts": 1,
    }
    marker = '[provider-qualification] {"version":1,"category":"provider_turn_failed",'
    assert marker in captured.err


def test_post_execution_source_drift_cannot_report_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checks = 0

    def source(_head: str, _tree: str) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise evaluator.SourceBindingFailure("source_binding_failed")

    async def success(**_options: object) -> dict[str, object]:
        return {"pass": True}

    monkeypatch.setattr(evaluator, "_verify_source_binding", source)
    monkeypatch.setattr(evaluator, "run_evaluation", success)
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40,
    ]) == 2
    assert checks == 2
    output = json.loads(capsys.readouterr().out)
    assert output["pass"] is False
    assert output["error"]["code"] == "source_binding_failed"


def test_source_drift_after_provider_failure_preserves_primary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checks = 0

    def source(_head: str, _tree: str) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise evaluator.SourceBindingFailure("source_binding_failed")

    async def failed(**_options: object) -> dict[str, object]:
        raise evaluator.EvaluationFailure(
            "provider_turn_failed", case_id="nw-002", completed_cases=1,
        )

    monkeypatch.setattr(evaluator, "_verify_source_binding", source)
    monkeypatch.setattr(evaluator, "run_evaluation", failed)
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40,
    ]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "provider_turn_failed"
    assert error["completed_cases"] == 1
    assert error["secondary_categories"] == ["source_binding_failed"]


def test_wrong_checkout_host_factory_refuses_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_realtime import host_launcher

    monkeypatch.setattr(host_launcher, "__file__", str(tmp_path / "host_launcher.py"))

    async def forbidden(**_options: object) -> dict[str, object]:
        pytest.fail("wrong checkout reached provider")

    monkeypatch.setattr(evaluator, "run_evaluation", forbidden)
    assert evaluator.main([
        "--provider", "ollama", "--source-head", "a" * 40,
        "--source-tree", "b" * 40,
    ]) == 2
    output = capsys.readouterr().out
    assert '"status":"error"' in output
    assert str(tmp_path) not in output


@pytest.mark.asyncio
async def test_direct_caller_cannot_relabel_override_as_production_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = evaluator.load_corpus(evaluator._DEFAULT_CORPUS)
    one_case = dataclasses.replace(corpus, cases=corpus.cases[:1])
    monkeypatch.setattr(
        evaluator, "_build_streaming_inference",
        lambda **_options: pytest.fail("mislabel reached provider factory"),
    )
    with pytest.raises(ValueError, match="production_default"):
        await evaluator.run_evaluation(
            corpus=one_case, provider="ollama", mode="production_default",
            model="synthetic:latest", effort="low", codex_executable=None,
            ollama_base_url="http://127.0.0.1:11434", ollama_num_ctx=None,
        )
