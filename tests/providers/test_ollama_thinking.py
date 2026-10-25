"""Model-free controls for explicit Ollama thinking selection."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from hermes_realtime import host_launcher as host
from hermes_realtime.conversation.context import ConversationMessage
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.providers.ollama import OllamaStreamingInference
from tests.support.ollama_metadata import supported_ollama_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [None, False, True])
async def test_thinking_selection_reaches_readiness_and_structured_decision(thinking):
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
        think=thinking, open_request=opened, metadata_client=supported_ollama_metadata(),
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
            snapshot, turn_id="turn_decision")] == ["Ready."]
    finally:
        await inference.close()

    assert "format" not in bodies[0]
    assert bodies[0]["options"] == {"num_ctx": 4096}
    assert bodies[1]["options"] == {
        "num_ctx": 4096, "num_predict": 1024, "temperature": 0,
    }
    assert bodies[1]["format"] == inference._work_response_schema(64, can_start=True)
    for body in bodies:
        assert "think" not in body["options"]
        if thinking is None:
            assert "think" not in body
        else:
            assert body["think"] is thinking
        assert body["shift"] is False and body["truncate"] is False


@pytest.mark.parametrize("thinking", [0, 1, 0.0, "false", [], {}])
def test_adapter_refuses_non_boolean_thinking_before_requests(thinking):
    with pytest.raises(TypeError, match="think must be an exact boolean or None"):
        OllamaStreamingInference(
            base_url="http://127.0.0.1:11434", model="test", think=thinking,
        )


@pytest.mark.parametrize("thinking", [None, False, True])
def test_host_factory_passes_thinking_choice(monkeypatch, thinking):
    seen = {}
    monkeypatch.setattr(host, "OllamaStreamingInference",
                        lambda **options: seen.update(options) or object())
    host._build_streaming_inference(
        inference_provider="ollama", ollama_base_url="http://127.0.0.1:11434",
        ollama_model="test", ollama_think=thinking, codex_model="test",
        codex_effort="low", codex_executable=None,
    )
    assert seen["think"] is thinking


@pytest.mark.parametrize("thinking", [0, "false"])
def test_host_factory_refuses_non_boolean_thinking_before_composition(monkeypatch, thinking):
    monkeypatch.setattr(host, "OllamaStreamingInference",
                        lambda **_: pytest.fail("invalid thinking reached provider"))
    with pytest.raises(TypeError, match="think must be an exact boolean or None"):
        host._build_streaming_inference(
            inference_provider="ollama", ollama_base_url="http://127.0.0.1:11434",
            ollama_model="test", ollama_think=thinking, codex_model="test",
            codex_effort="low", codex_executable=None,
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_explicit_thinking_refuses_non_ollama_factory(monkeypatch, thinking):
    monkeypatch.setattr(host, "CodexAppServerStreamingInference",
                        lambda **_: pytest.fail("thinking reached unrelated provider"))
    with pytest.raises(ValueError, match="--ollama-think requires --inference-provider ollama"):
        host._build_streaming_inference(
            inference_provider="codex", ollama_base_url="http://127.0.0.1:11434",
            ollama_model="test", ollama_think=thinking, codex_model="test",
            codex_effort="low", codex_executable=None,
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_explicit_thinking_refuses_non_ollama_host(monkeypatch, thinking):
    monkeypatch.setattr(host, "_build_streaming_inference",
                        lambda **_: pytest.fail("thinking reached unrelated provider factory"))
    with host._qualification_no_task_composition(), pytest.raises(
        ValueError, match="--ollama-think requires --inference-provider ollama",
    ):
        host.build_local_host_launcher(
            hermes_api_bearer=None, inference_provider="codex", ollama_think=thinking,
        )


@pytest.mark.parametrize("thinking", [0, "false"])
def test_host_refuses_non_boolean_thinking_before_composition(monkeypatch, thinking):
    monkeypatch.setattr(host, "_compose_public_search",
                        lambda **_: pytest.fail("invalid thinking reached composition"))
    with host._qualification_no_task_composition(), pytest.raises(
        TypeError, match="think must be an exact boolean or None",
    ):
        host.build_local_host_launcher(hermes_api_bearer=None, ollama_think=thinking)


@pytest.mark.asyncio
async def test_explicit_thinking_refuses_qualification_provider_override(monkeypatch):
    class Provider:
        async def close(self):
            pass

        async def aclose(self):
            pass

    standin = SimpleNamespace(
        take=lambda: standin,
        inference_factory=Provider,
        synthesizer_factory=Provider,
        transcriber_factory=Provider,
        speech_presence_factory=Provider,
        vad_factory=Provider,
        identity_factory=lambda: "qualification-identity",
        writer_transport_factory=None,
    )
    monkeypatch.setattr(host, "_current_qualification_full_host_dependencies",
                        lambda: standin)
    launcher = None
    try:
        with pytest.raises(
            ValueError, match="thinking does not accept qualification provider overrides",
        ):
            launcher = host.build_local_host_launcher(hermes_api_bearer=None, ollama_think=False)
    finally:
        if launcher is not None:
            await launcher.close()


def test_host_cli_thinking_is_optional_and_boolean():
    parser = host._build_argument_parser()
    assert parser.parse_args([]).ollama_think is None
    assert parser.parse_args(["--ollama-think"]).ollama_think is True
    assert parser.parse_args(["--no-ollama-think"]).ollama_think is False
