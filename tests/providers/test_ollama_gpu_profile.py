"""Model-free controls for an explicit Ollama GPU-layer request profile."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from hermes_realtime import host_launcher as host
from hermes_realtime.conversation.context import ConversationMessage
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import OllamaModelClient
from scripts import evaluate_natural_work_routing as evaluator
from tests.support.ollama_metadata import supported_ollama_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("layers", [None, 0, 17, 1024])
async def test_gpu_layers_reach_readiness_and_structured_decision(layers):
    bodies = []

    def opened(request, timeout):
        body = json.loads(request.data)
        bodies.append(body)
        text = json.dumps({"kind": "reply", "text": "Ready."}) if "format" in body else "Ready."
        return io.BytesIO(json.dumps({
            "message": {"content": text}, "done": True, "done_reason": "stop",
        }).encode() + b"\n")

    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434", model="test", num_ctx=4096,
        num_gpu=layers, open_request=opened, metadata_client=supported_ollama_metadata(),
    )
    snapshot = ConversationInferenceRequest(
        revision=0, messages=(ConversationMessage("user", "Synthetic readiness."),),
        active_tasks=(),
    )

    class Handler:
        max_objective_chars = 64

        async def start_work(self, **kwargs):
            raise AssertionError("synthetic reply must not dispatch")

    try:
        assert [part async for part in inference.stream(
            snapshot, turn_id="turn_preflight")] == ["Ready."]
        inference.bind_work_tools(Handler())
        assert [part async for part in inference.stream(
            snapshot, turn_id="turn_decide")] == ["Ready."]
    finally:
        await inference.close()
    selected = {} if layers is None else {"num_gpu": layers}
    assert bodies[0]["options"] == {"num_ctx": 4096, **selected}
    assert bodies[1]["options"] == {
        "num_ctx": 4096, "num_predict": 1024, "temperature": 0, **selected,
    }
    assert all(body["shift"] is False and body["truncate"] is False for body in bodies)


@pytest.mark.parametrize("layers", [True, False, -1, 1025, 0.0, "0", [], {}])
def test_gpu_layers_refuse_invalid_before_metadata_or_inference(layers, capsys):
    with pytest.raises((TypeError, ValueError), match="num_gpu"):
        OllamaStreamingInference(
            base_url="http://127.0.0.1:11434", model="test", num_gpu=layers,
        )
    assert '"category":"gpu_layers_refused"' in capsys.readouterr().out


def test_host_factory_and_cli_keep_gpu_layers_explicit(monkeypatch):
    seen = {}

    def adapter(**options):
        seen.update(options)
        return object()

    monkeypatch.setattr(host, "OllamaStreamingInference", adapter)
    host._build_streaming_inference(
        inference_provider="ollama", ollama_base_url="http://127.0.0.1:11434",
        ollama_model="test", ollama_num_gpu=0, codex_model="test", codex_effort="low",
        codex_executable=None,
    )
    assert seen["num_gpu"] == 0
    assert host._build_argument_parser().parse_args([]).ollama_num_gpu is None
    assert host._build_argument_parser().parse_args(["--ollama-num-gpu", "0"]).ollama_num_gpu == 0


def test_evaluator_gpu_layers_are_reported_diagnostic_override():
    parser = evaluator._parser()
    arguments = parser.parse_args([
        "--provider", "ollama", "--mode", "diagnostic_override", "--ollama-num-gpu", "0",
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
    ])
    assert evaluator._resolved_provider_options(arguments)["ollama_num_gpu"] == 0
    arguments.mode = "production_default"
    with pytest.raises(ValueError, match="diagnostic overrides"):
        evaluator._resolved_provider_options(arguments)
    arguments.mode = "diagnostic_override"
    arguments.provider = "codex"
    with pytest.raises(ValueError, match="Codex refuses Ollama"):
        evaluator._resolved_provider_options(arguments)


def test_evaluator_uses_gpu_default_from_host_signature(monkeypatch):
    defaults = evaluator._host_provider_defaults()
    defaults["ollama_num_gpu"] = 0
    monkeypatch.setattr(evaluator, "_host_provider_defaults", lambda: defaults)
    arguments = evaluator._parser().parse_args([
        "--provider", "ollama", "--source-head", "a" * 40, "--source-tree", "b" * 40,
    ])
    assert evaluator._resolved_provider_options(arguments)["ollama_num_gpu"] == 0


@pytest.mark.asyncio
async def test_evaluator_factory_cpu_profile_reaches_both_real_adapter_requests(monkeypatch):
    bodies = []

    async def metadata(_self, endpoint, body):
        if endpoint == "show":
            assert body == {"model": "test"}
            return json.dumps({
                "model_info": {
                    "general.architecture": "synthetic", "synthetic.context_length": 4096,
                },
                "capabilities": ["completion"], "details": {"format": "gguf"},
            }).encode()
        assert body is None
        if endpoint == "version":
            return b'{"version":"0.40.4"}'
        from tests.support.ollama_metadata import supported_ollama_tags
        assert endpoint == "tags"
        return supported_ollama_tags("test")

    def opened(request, _timeout):
        body = json.loads(request.data)
        bodies.append(body)
        text = json.dumps({"kind": "reply", "text": "Ready."}) if "format" in body else "Ready."
        return io.BytesIO(json.dumps({
            "message": {"content": text}, "done": True, "done_reason": "stop",
        }).encode() + b"\n")

    monkeypatch.setattr(OllamaModelClient, "_read_request", metadata)
    monkeypatch.setattr(OllamaStreamingInference, "_urlopen", staticmethod(opened))
    handler = evaluator.ShadowWorkHandler(can_cancel_work=False)
    inference = await evaluator._open_qualified_inference(
        provider="ollama", model="test", effort="low", codex_executable=None,
        ollama_base_url="http://127.0.0.1:11434", ollama_num_ctx=4096,
        ollama_num_gpu=0, handler=handler,
    )
    try:
        handler.begin_case("nw-001", started_at=0.0)
        snapshot = ConversationInferenceRequest(
            revision=1, messages=(ConversationMessage("user", "Synthetic question."),),
            active_tasks=(),
        )
        assert [part async for part in inference.stream(
            snapshot, turn_id="turn_decision")] == ["Ready."]
    finally:
        await inference.close()
    assert bodies[0]["options"] == {"num_ctx": 4096, "num_gpu": 0}
    assert bodies[1]["options"] == {
        "num_ctx": 4096, "num_predict": 1024, "temperature": 0, "num_gpu": 0,
    }


def test_owned_evaluator_forwards_zero_gpu_override_without_native_launch(monkeypatch, tmp_path):
    from scripts import qualification_launch

    seen = []

    def launch(script, arguments, **options):
        seen.extend(arguments)
        return SimpleNamespace(exit_code=0, stdout=b"{}\n")

    monkeypatch.setattr(qualification_launch, "run_owned_controller", launch)
    monkeypatch.setattr(evaluator, "_verify_source_binding", lambda *_: None)
    assert evaluator.main([
        "--provider", "ollama", "--mode", "diagnostic_override", "--ollama-num-gpu", "0",
        "--source-head", "a" * 40, "--source-tree", "b" * 40,
        "--launch-binding", str(tmp_path / "binding.json"), "--launch-binding-sha256", "c" * 64,
        "--launch-receipt", str(tmp_path / "receipt.jsonl"),
    ]) == 0
    assert seen[seen.index("--ollama-num-gpu") + 1] == "0"


def test_gpu_profile_factory_refuses_other_provider(monkeypatch):
    monkeypatch.setattr(host, "CodexAppServerStreamingInference",
                        lambda **_: pytest.fail("GPU layers reached Codex constructor"))
    with pytest.raises(ValueError, match="requires --inference-provider ollama"):
        host._build_streaming_inference(
            inference_provider="codex", ollama_base_url="http://127.0.0.1:11434",
            ollama_model="test", ollama_num_gpu=0, codex_model="test", codex_effort="low",
            codex_executable=None,
        )


def test_gpu_profile_host_refuses_other_provider(monkeypatch):
    monkeypatch.setattr(host, "_build_streaming_inference",
                        lambda **_: pytest.fail("GPU layers reached unrelated provider factory"))
    with host._qualification_no_task_composition(), pytest.raises(
        ValueError, match="requires --inference-provider ollama"
    ):
        host.build_local_host_launcher(
            hermes_api_bearer=None, inference_provider="codex", ollama_num_gpu=0,
        )


def test_gpu_profile_refuses_unrelated_qualification_provider_override(monkeypatch):
    monkeypatch.setattr(host, "_current_qualification_full_host_dependencies",
                        lambda: SimpleNamespace(take=lambda: object()))
    with pytest.raises(ValueError, match="qualification provider overrides"):
        host.build_local_host_launcher(hermes_api_bearer=None, ollama_num_gpu=0)


@pytest.mark.asyncio
async def test_evaluator_rejects_gpu_profile_on_codex_before_factory(monkeypatch):
    monkeypatch.setattr(evaluator, "_build_streaming_inference", lambda **_: pytest.fail("factory"))
    with pytest.raises(ValueError, match="not a Codex option"):
        await evaluator.run_evaluation(
            corpus=evaluator.load_corpus(evaluator._DEFAULT_CORPUS), model="test", effort="low",
            codex_executable=None, provider="codex", ollama_num_gpu=0,
        )
