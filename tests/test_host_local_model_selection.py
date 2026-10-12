"""The browser choice must reach host ownership before any model inference."""

from __future__ import annotations

import pytest

from hermes_realtime import host_launcher as host


def test_local_selection_cli_is_opt_in() -> None:
    parser = host._build_argument_parser()
    assert parser.parse_args([]).local_model_selection is False
    assert parser.parse_args(["--local-model-selection"]).local_model_selection is True


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"local_model_selection": 1}, TypeError, "exact boolean"),
        ({"inference_provider": "codex"}, ValueError, "requires.*ollama"),
        ({"ollama_num_ctx": 4096}, ValueError, "conflicts"),
        ({"hermes_context": object()}, ValueError, "representative context"),
    ],
)
def test_local_selection_refuses_incompatible_configuration_before_composition(
    options: dict[str, object], error: type[Exception], message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider composition must not start")

    monkeypatch.setattr(host, "_compose_public_search", unexpected)
    parameters: dict[str, object] = {
        "hermes_api_bearer": "x" * 32,
        "allow_unsandboxed_tasks": True,
        "local_model_selection": True,
        **options,
    }
    with pytest.raises(error, match=message):
        host.build_local_host_launcher(**parameters)  # type: ignore[arg-type]
    assert capsys.readouterr().out == (
        '[local-model-selection] {"version":1,"category":"configuration_refused"}\n'
    )


def test_local_selection_cannot_replace_qualification_provider_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Bundle:
        def take(self) -> Bundle:
            return self

    monkeypatch.setattr(host, "_current_qualification_full_host_dependencies", Bundle)
    with pytest.raises(ValueError, match="qualification provider overrides"):
        host.build_local_host_launcher(hermes_api_bearer=None, local_model_selection=True)


@pytest.mark.asyncio
async def test_host_selection_owns_actual_speech_readiness_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.providers.ollama_models import (
        OllamaContextConfiguration,
        OllamaContextObservation,
        OllamaModelMetadata,
    )
    from hermes_realtime.speech import AudioFrame, SpeechChunk

    events: list[str] = []
    captured: dict[str, object] = {}
    metadata = OllamaModelMetadata("selected:latest", "a" * 64, 32768)

    class Provider:
        async def close(self) -> None:
            events.append("provider_closed")

        async def synthesize(self, text: str, turn_id: str):  # type: ignore[no-untyped-def]
            events.append("tts_ready")
            yield SpeechChunk(
                turn_id=turn_id, chunk_id="chunk_ready", text=text,
                audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=24000, channels=1),
            )

    class Runtime:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def start(self) -> str:
            return "http://127.0.0.1:8765/"

        async def close_ingress(self) -> None:
            events.append("ingress_closed")

        async def close(self) -> None:
            events.append("runtime_closed")

    async def task_start(self: object) -> None:
        events.append("tasks_started")

    async def task_close(self: object) -> None:
        events.append("tasks_closed")

    async def installed(self: object) -> tuple[OllamaModelMetadata, ...]:
        events.append("metadata")
        return (metadata,)

    async def configure(self, requested: int, *, client: object) -> None:  # type: ignore[no-untyped-def]
        events.append("configured")
        configuration = OllamaContextConfiguration(metadata, requested)
        self._context_configuration = configuration
        self._context_observation = OllamaContextObservation(configuration)

    async def observe(self: object) -> None:
        events.append("observed")

    async def stream(self, request: object, *, turn_id: str):  # type: ignore[no-untyped-def]
        assert self._model == metadata.model
        events.append("inference:" + turn_id)
        yield "Ready."

    async def adapter_close(self: object) -> None:
        events.append("adapter_closed")

    original_speech = host.StreamingSpeechLoop
    original_actions_close = host.ConversationUpdateExecutor.close

    async def actions_close(self):  # type: ignore[no-untyped-def]
        events.append("actions_closed")
        await original_actions_close(self)

    def speech(**kwargs: object):  # type: ignore[no-untyped-def]
        captured["speech_inference"] = kwargs["inference"]
        return original_speech(**kwargs)

    def no_default(**kwargs: object) -> None:
        raise AssertionError("selection must not construct a default adapter")

    monkeypatch.setattr(host, "_build_streaming_inference", no_default)
    monkeypatch.setattr(host, "_build_synthesizer", lambda **kwargs: Provider())
    monkeypatch.setattr(host, "_build_streaming_transcriber", lambda **kwargs: Provider())
    monkeypatch.setattr(host, "SileroSpeechPresenceVerifier", Provider)
    monkeypatch.setattr(host, "WebRtcVoiceActivityDetector", Provider)
    monkeypatch.setattr(host, "BrowserClientRuntime", Runtime)
    monkeypatch.setattr(host, "StreamingSpeechLoop", speech)
    monkeypatch.setattr(host.ConversationUpdateExecutor, "close", actions_close)
    monkeypatch.setattr(host.HermesApiTaskSession, "start", task_start)
    monkeypatch.setattr(host.HermesApiTaskSession, "close", task_close)
    monkeypatch.setattr(host.OllamaModelClient, "installed_models", installed)
    monkeypatch.setattr(host.OllamaStreamingInference, "configure_context", configure)
    monkeypatch.setattr(host.OllamaStreamingInference, "observe_context", observe)
    monkeypatch.setattr(host.OllamaStreamingInference, "stream", stream)
    monkeypatch.setattr(host.OllamaStreamingInference, "close", adapter_close)
    launcher = host.build_local_host_launcher(
        hermes_api_bearer="x" * 32, allow_unsandboxed_tasks=True,
        local_model_selection=True,
    )
    owner = captured["local_model_selection"]
    assert type(owner) is host.OllamaSessionSelection
    assert captured["speech_inference"] is owner
    assert captured["model_configuration"] is None
    assert events == []
    await launcher._preflight()
    assert events == ["tasks_started"]
    await owner.prepare(metadata.model, metadata.digest, 4096)
    assert events == ["tasks_started", "metadata", "configured",
                      "inference:turn_preflight", "tts_ready", "observed"]
    assert owner.require_prepared().model == metadata.model
    await owner.prepare(metadata.model, metadata.digest, 4096)
    assert events.count("tasks_started") == events.count("configured") == 1
    launcher._start_attempted = True
    await launcher.close()
    assert events.index("ingress_closed") < events.index("adapter_closed")
    assert events.index("adapter_closed") < events.index("actions_closed")
    assert events.index("adapter_closed") < events.index("tasks_closed")
    assert events.index("adapter_closed") < events.index("provider_closed")
