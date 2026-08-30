from __future__ import annotations

import asyncio
import inspect
import ssl
import sys
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

import hermes_realtime.host_launcher as host_launcher_module
import hermes_realtime.launcher as launcher_module
from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import (
    DEFAULT_MAX_RESPONSE_SEGMENTS,
    ConversationContextStore,
    ConversationUpdateExecutor,
    ForegroundTurnLease,
    StreamingSpeechLoop,
    TaskTerminalOutcome,
    UpdatePolicyInput,
)
from hermes_realtime.conversation.knowledge import KnowledgePrefetchCoordinator
from hermes_realtime.host_launcher import (
    _LIVEKIT_CONFIRMATION_MARGIN_SECONDS,
    _LIVEKIT_CONFIRMATION_TIMEOUT_SECONDS,
    _LIVEKIT_PUBLICATION_TIMEOUT_SECONDS,
    ProactiveCompletionUpdatePolicy,
    _build_streaming_inference,
    _build_streaming_transcriber,
    _build_synthesizer,
    _encode_speech_timing_event,
    _load_api_bearer,
    _load_hermes_context,
    _renderer_authority_ends,
    _renderer_yield_event_data,
    build_local_host_launcher,
)
from hermes_realtime.launcher import (
    ConversationOnlyTaskSession,
    LocalBrowserLauncher,
    _project_conversation_observation,
    build_local_conversation_launcher,
)
from hermes_realtime.livekit import MAX_SPEECH_CHUNK_DURATION_SECONDS
from hermes_realtime.protocol import ControlCancelEvent, WorkDispatchRequestedEvent
from hermes_realtime.providers.codex_app_server import HermesRepresentativeContext
from hermes_realtime.providers.current_facts import (
    DdgsCurrentFactLookup,
)
from hermes_realtime.speech import AudioFrame, SpeechChunk, WordTiming


@pytest.fixture(autouse=True)
def _stub_optional_speech_presence_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = object()
    monkeypatch.setattr(
        host_launcher_module,
        "SileroSpeechPresenceVerifier",
        lambda: verifier,
    )
    monkeypatch.setattr(
        launcher_module,
        "SileroSpeechPresenceVerifier",
        lambda: verifier,
    )


class RuntimeProbe:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self) -> str:
        self.events.append("runtime:start")
        return "http://127.0.0.1:8765/#bootstrap=secret"

    async def close(self) -> None:
        self.events.append("runtime:close")


def test_production_confirmation_budget_covers_maximum_speech_chunk() -> None:
    assert _LIVEKIT_CONFIRMATION_MARGIN_SECONDS >= 5.0
    assert _LIVEKIT_CONFIRMATION_TIMEOUT_SECONDS == (
        MAX_SPEECH_CHUNK_DURATION_SECONDS + _LIVEKIT_CONFIRMATION_MARGIN_SECONDS
    )
    assert _LIVEKIT_PUBLICATION_TIMEOUT_SECONDS == (
        MAX_SPEECH_CHUNK_DURATION_SECONDS + _LIVEKIT_CONFIRMATION_MARGIN_SECONDS
    )


def test_renderer_yield_event_carries_exact_prepared_stream_authority() -> None:
    authority = ("turn_1", 3, "chunk_2", "stream_4")

    assert _renderer_yield_event_data(
        "interrupt_requested",
        {"turnId": "turn_1"},
        authority,
    ) == {
        "turnId": "turn_1",
        "turnGeneration": 3,
        "chunkId": "chunk_2",
        "streamId": "stream_4",
    }
    assert _renderer_yield_event_data(
        "barge_in_non_speech_suppressed",
        {},
        authority,
    ) == {
        "turnId": "turn_1",
        "turnGeneration": 3,
        "chunkId": "chunk_2",
        "streamId": "stream_4",
    }
    assert _renderer_yield_event_data("voice_activity_started", {}, authority) == {}
    assert _renderer_yield_event_data("interrupt_requested", {}, None) == {}
    assert _renderer_yield_event_data("interrupt_requested", {}, authority) == {}
    assert (
        _renderer_yield_event_data(
            "interrupt_requested",
            {"turnId": "other_turn"},
            authority,
        )
        == {}
    )
    assert (
        _renderer_yield_event_data(
            "barge_in_non_speech_suppressed",
            {"reason": "non_speech"},
            None,
        )
        == {}
    )


def test_renderer_authority_survives_interruption_until_exact_receipt() -> None:
    owner = ("presentation_turn", 3)

    assert not _renderer_authority_ends(
        "assistant_turn_interrupted",
        {"turnId": "presentation_turn", "turnGeneration": 3},
        owner,
    )
    assert _renderer_authority_ends(
        "assistant_turn_completed",
        {"turnId": "presentation_turn", "turnGeneration": 3},
        owner,
    )
    assert _renderer_authority_ends("interrupt_requested", {}, owner)


def test_natural_profile_tolerates_cadence_pause_that_legacy_endpoints() -> None:
    build_vad = host_launcher_module._build_voice_activity_detector

    legacy_decisions = deque((True,) * 20 + (False,) * 40)
    legacy = build_vad(
        conversation_profile="legacy",
        classify=lambda _pcm, _rate: legacy_decisions.popleft(),
    )
    assert [legacy.process(_vad_frame()) for _ in range(60)][-1].value == "speech_ended"

    natural_decisions = deque((True,) * 20 + (False,) * 60 + (True,))
    natural = build_vad(
        conversation_profile="natural_v1",
        classify=lambda _pcm, _rate: natural_decisions.popleft(),
    )
    natural_activity = [natural.process(_vad_frame()) for _ in range(81)]
    assert all(activity.value != "speech_ended" for activity in natural_activity)
    assert natural_activity[-1].value == "speech_continued"


def test_browser_speech_runtime_uses_configured_model_identities() -> None:
    moonshine = host_launcher_module._build_browser_speech_runtime(
        stt_provider="moonshine",
        whisper_model="small.en",
        moonshine_model_tier="small",
        tts_provider="kokoro",
        edge_voice="en-GB-SoniaNeural",
    )
    assert moonshine.public_data() == {
        "sttModel": "moonshine-v2-small",
        "sttProvider": "moonshine",
        "ttsModel": "kokoro-v1.0.onnx",
        "ttsProvider": "kokoro",
    }


def test_host_medium_launch_projection_preserves_tiny_defaults() -> None:
    medium = host_launcher_module._build_browser_speech_runtime(
        stt_provider="moonshine",
        whisper_model="base.en",
        moonshine_model_tier="medium",
        tts_provider="edge",
        edge_voice="en-US-AriaNeural",
    )
    defaults = host_launcher_module._build_argument_parser().parse_args([])

    assert medium.public_data()["sttProvider"] == "moonshine"
    assert medium.public_data()["sttModel"] == "moonshine-v2-medium"
    assert defaults.moonshine_model_tier == "tiny"
    assert defaults.tts_provider == host_launcher_module._default_tts_provider()

    remote = host_launcher_module._build_browser_speech_runtime(
        stt_provider="faster-whisper",
        whisper_model="small.en",
        moonshine_model_tier="small",
        tts_provider="edge",
        edge_voice="en-GB-SoniaNeural",
    )
    assert remote.public_data() == {
        "sttModel": "small.en",
        "sttProvider": "faster-whisper",
        "ttsModel": "en-GB-SoniaNeural",
        "ttsProvider": "edge",
    }


def _vad_frame() -> AudioFrame:
    return AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)


def _write_test_certificate(
    root: Path,
    *,
    stem: str,
    hostname: str,
    include_subject_alternative_name: bool,
) -> tuple[Path, Path]:
    """Create an ephemeral, self-signed certificate for one test process."""

    private_key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name((x509.NameAttribute(NameOID.COMMON_NAME, hostname),))
    now = datetime.now(UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
    )
    if include_subject_alternative_name:
        builder = builder.add_extension(
            x509.SubjectAlternativeName((x509.DNSName(hostname),)),
            critical=False,
        )
    certificate = builder.sign(private_key, hashes.SHA256())
    certificate_path = root / f"{stem}-cert.pem"
    private_key_path = root / f"{stem}-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    private_key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, private_key_path


@pytest.fixture
def tls_certificates(tmp_path: Path) -> dict[str, Path]:
    certificate, private_key = _write_test_certificate(
        tmp_path,
        stem="assistant",
        hostname="assistant.example.test",
        include_subject_alternative_name=True,
    )
    cn_only_certificate, cn_only_private_key = _write_test_certificate(
        tmp_path,
        stem="cn-only",
        hostname="assistant.example.test",
        include_subject_alternative_name=False,
    )
    return {
        "certificate": certificate,
        "private_key": private_key,
        "cn_only_certificate": cn_only_certificate,
        "cn_only_private_key": cn_only_private_key,
    }


def test_remote_host_accepts_exact_secure_livekit_origin() -> None:
    assert (
        host_launcher_module._validate_livekit_origin(
            "wss://voice.example.test:7880",
            remote_mode=True,
        )
        == "wss://voice.example.test:7880"
    )


@pytest.mark.parametrize(
    "origin",
    [
        "wss://voice.example.test:abc",
        "wss://voice.example.test:99999",
        "wss://voice.example.test\\forbidden",
        "wss://127.0.0.1:7880",
        "wss://localhost:7880",
        "wss://127.1:7880",
        "wss://2130706433:7880",
    ],
)
def test_remote_host_rejects_unusable_livekit_origins(origin: str) -> None:
    with pytest.raises(ValueError, match="remote LiveKit URL"):
        host_launcher_module._validate_livekit_origin(origin, remote_mode=True)


def test_local_launcher_drops_echo_diagnostics_when_projection_is_full() -> None:
    projection = BrowserEventProjection(capacity=1)
    projection.publish("notification_queued", {})

    _project_conversation_observation(projection, "echo_suppressed", {})
    _project_conversation_observation(projection, "echo_barge_in_confirmed", {})

    assert [event.kind for event in projection.events_after(0)] == ["notification_queued"]


@pytest.mark.parametrize(
    "kind,data",
    (
        ("barge_in_non_speech_suppressed", {}),
        ("barge_in_verifier_unavailable", {"reason": "classification_error"}),
    ),
)
def test_local_launcher_projects_speech_verifier_advisories(
    kind: str,
    data: dict[str, str | int | bool | None],
) -> None:
    projection = BrowserEventProjection()

    _project_conversation_observation(projection, kind, data)

    assert [event.kind for event in projection.events_after(0)] == [kind]


def test_local_launcher_keeps_authoritative_observations_fail_closed() -> None:
    projection = BrowserEventProjection(capacity=1)
    projection.publish("notification_queued", {})

    with pytest.raises(RuntimeError, match="capacity"):
        _project_conversation_observation(
            projection,
            "transcript_final",
            {"text": "hello"},
        )


class VerifierProbe:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None:
        del timeout_seconds
        self.events.append(f"verifier:connect:{room_name}")

    async def disconnect(self, *, timeout_seconds: float = 10) -> None:
        del timeout_seconds
        self.events.append("verifier:disconnect")


class ProviderProbe:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    async def close(self) -> None:
        self.events.append(f"provider:close:{self.name}")


@pytest.mark.asyncio
async def test_local_launcher_preflights_before_exposing_one_use_browser_url() -> None:
    events: list[str] = []

    async def preflight() -> None:
        events.append("preflight")

    launcher = LocalBrowserLauncher(
        runtime=RuntimeProbe(events),
        verifier=VerifierProbe(events),
        room_name="hermes-local",
        preflight=preflight,
        providers=(ProviderProbe(events, "tts"), ProviderProbe(events, "inference")),
    )

    url = await launcher.start()
    with pytest.raises(RuntimeError, match="one-shot"):
        await launcher.start()
    await launcher.close()

    assert url.startswith("http://127.0.0.1:8765/#bootstrap=")
    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close",
        "verifier:disconnect",
        "provider:close:tts",
        "provider:close:inference",
    ]


@pytest.mark.asyncio
async def test_local_launcher_cleans_prepared_resources_when_runtime_start_fails() -> None:
    events: list[str] = []

    class FailingRuntime(RuntimeProbe):
        async def start(self) -> str:
            events.append("runtime:start")
            raise RuntimeError("listener unavailable")

    async def preflight() -> None:
        events.append("preflight")

    launcher = LocalBrowserLauncher(
        runtime=FailingRuntime(events),
        verifier=VerifierProbe(events),
        room_name="hermes-local",
        preflight=preflight,
        providers=(ProviderProbe(events, "provider"),),
    )

    with pytest.raises(RuntimeError, match="listener unavailable"):
        await launcher.start()

    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close",
        "verifier:disconnect",
        "provider:close:provider",
    ]


@pytest.mark.asyncio
async def test_local_launcher_close_survives_caller_cancellation_and_settles_once() -> None:
    events: list[str] = []
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class BlockingRuntime(RuntimeProbe):
        async def close(self) -> None:
            events.append("runtime:close:start")
            close_started.set()
            await release_close.wait()
            events.append("runtime:close:end")

    async def preflight() -> None:
        events.append("preflight")

    launcher = LocalBrowserLauncher(
        runtime=BlockingRuntime(events),
        verifier=VerifierProbe(events),
        room_name="hermes-local",
        preflight=preflight,
        providers=(ProviderProbe(events, "provider"),),
    )
    await launcher.start()
    caller = asyncio.create_task(launcher.close())
    await close_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    release_close.set()
    await launcher.close()

    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close:start",
        "runtime:close:end",
        "verifier:disconnect",
        "provider:close:provider",
    ]


@pytest.mark.asyncio
async def test_local_launcher_closes_provider_authority_when_ingress_close_fails() -> None:
    events: list[str] = []

    class FailOnceRuntime(RuntimeProbe):
        def __init__(self, runtime_events: list[str]) -> None:
            super().__init__(runtime_events)
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            events.append(f"runtime:close:{self.close_calls}")
            if self.close_calls == 1:
                raise RuntimeError("transient browser worker close failure")

    async def preflight() -> None:
        events.append("preflight")

    launcher = LocalBrowserLauncher(
        runtime=FailOnceRuntime(events),
        verifier=VerifierProbe(events),
        room_name="hermes-local",
        preflight=preflight,
        providers=(ProviderProbe(events, "authority"),),
    )
    await launcher.start()

    with pytest.raises(RuntimeError, match="transient browser worker"):
        await launcher.close()
    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close:1",
        "verifier:disconnect",
        "provider:close:authority",
    ]

    await launcher.close()
    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close:1",
        "verifier:disconnect",
        "provider:close:authority",
        "runtime:close:2",
    ]


@pytest.mark.asyncio
async def test_local_launcher_aggregates_ingress_and_every_provider_failure_in_order() -> None:
    events: list[str] = []

    class FailingRuntime(RuntimeProbe):
        async def close(self) -> None:
            events.append("runtime:close")
            raise RuntimeError("browser ingress close failed")

    class FailingProvider(ProviderProbe):
        async def close(self) -> None:
            events.append(f"provider:close:{self.name}")
            raise RuntimeError(f"{self.name} close failed")

    async def preflight() -> None:
        events.append("preflight")

    launcher = LocalBrowserLauncher(
        runtime=FailingRuntime(events),
        verifier=VerifierProbe(events),
        room_name="hermes-local",
        preflight=preflight,
        providers=(
            FailingProvider(events, "work"),
            FailingProvider(events, "evidence-drain"),
        ),
    )
    await launcher.start()

    with pytest.raises(BaseExceptionGroup) as failure:
        await launcher.close()

    assert [str(error) for error in failure.value.exceptions] == [
        "browser ingress close failed",
        "work close failed",
        "evidence-drain close failed",
    ]
    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close",
        "verifier:disconnect",
        "provider:close:work",
        "provider:close:evidence-drain",
    ]


@pytest.mark.asyncio
async def test_local_launcher_aggregates_evidence_close_failure_and_retries_it() -> None:
    events: list[str] = []

    class FailOnceProvider(ProviderProbe):
        def __init__(self, provider_events: list[str], name: str) -> None:
            super().__init__(provider_events, name)
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            events.append(f"provider:close:{self.name}:{self.close_calls}")
            if self.close_calls == 1:
                raise RuntimeError(f"{self.name} close failed")

    async def preflight() -> None:
        events.append("preflight")

    work_owner = FailOnceProvider(events, "work")
    evidence_owner = FailOnceProvider(events, "evidence-drain")
    launcher = LocalBrowserLauncher(
        runtime=RuntimeProbe(events),
        verifier=VerifierProbe(events),
        room_name="hermes-local",
        preflight=preflight,
        providers=(work_owner, evidence_owner),
    )
    await launcher.start()

    with pytest.raises(BaseExceptionGroup) as failure:
        await launcher.close()

    assert [str(error) for error in failure.value.exceptions] == [
        "work close failed",
        "evidence-drain close failed",
    ]
    assert events == [
        "preflight",
        "verifier:connect:hermes-local",
        "runtime:start",
        "runtime:close",
        "verifier:disconnect",
        "provider:close:work:1",
        "provider:close:evidence-drain:1",
    ]

    await launcher.close()

    assert events[-2:] == [
        "provider:close:work:2",
        "provider:close:evidence-drain:2",
    ]


@pytest.mark.asyncio
async def test_conversation_only_task_session_fails_closed_without_transport() -> None:
    session = ConversationOnlyTaskSession()

    with pytest.raises(PermissionError, match="conversation-only"):
        await session.dispatch(cast(WorkDispatchRequestedEvent, object()))
    with pytest.raises(PermissionError, match="conversation-only"):
        await session.cancel(cast(ControlCancelEvent, object()))

    pending = asyncio.create_task(session.next_update())
    await asyncio.sleep(0)
    assert not pending.done()
    await session.close()
    with pytest.raises(RuntimeError, match="closed"):
        await pending


def test_host_inference_selection_is_explicit_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    codex_instance = object()

    def fake_codex(**kwargs: object) -> object:
        calls.append(("codex", kwargs))
        return codex_instance

    def forbidden_ollama(**kwargs: object) -> object:
        calls.append(("ollama", kwargs))
        raise AssertionError("Codex selection must not construct Ollama")

    monkeypatch.setattr(host_launcher_module, "CodexAppServerStreamingInference", fake_codex)
    monkeypatch.setattr(host_launcher_module, "OllamaStreamingInference", forbidden_ollama)

    selected = _build_streaming_inference(
        inference_provider="codex",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="local-model",
        codex_model="gpt-5.6-terra",
        codex_effort="low",
        codex_executable="C:/tools/codex.exe",
        hermes_context=HermesRepresentativeContext(
            identity="MrAnderson",
            persona="composed and precise",
        ),
    )

    assert selected is codex_instance
    assert calls[0][1]["current_fact_lookup"] is None
    assert calls == [
        (
            "codex",
            {
                "model": "gpt-5.6-terra",
                "effort": "low",
                "codex_executable": "C:/tools/codex.exe",
                "hermes_context": HermesRepresentativeContext(
                    identity="MrAnderson",
                    persona="composed and precise",
                ),
                "token_usage_observer": None,
                "current_fact_lookup": calls[0][1]["current_fact_lookup"],
                "knowledge_coordinator": None,
                "knowledge_timing_observer": None,
                "request_timeout_seconds": 60.0,
            },
        )
    ]

    with pytest.raises(ValueError, match="requires --inference-provider codex"):
        _build_streaming_inference(
            inference_provider="ollama",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="local-model",
            codex_model="gpt-5.6-terra",
            codex_effort="low",
            codex_executable=None,
            hermes_context=HermesRepresentativeContext(identity="MrAnderson"),
        )


def test_codex_coordinator_preserves_same_lookup_for_dynamic_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    lookup = DdgsCurrentFactLookup(timeout_seconds=3.5, max_results=5, enrich_results=2)
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        owns_lookup=False,
    )

    def fake_codex(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        host_launcher_module,
        "CodexAppServerStreamingInference",
        fake_codex,
    )

    _build_streaming_inference(
        inference_provider="codex",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="local-model",
        codex_model="gpt-5.6-terra",
        codex_effort="low",
        codex_executable="C:/tools/codex.exe",
        knowledge_coordinator=coordinator,
    )

    assert calls[0]["knowledge_coordinator"] is coordinator
    assert calls[0]["current_fact_lookup"] is lookup


def test_streaming_inference_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="inference_provider"):
        _build_streaming_inference(
            inference_provider="automatic",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="local-model",
            codex_model="gpt-5.6-terra",
            codex_effort="low",
            codex_executable=None,
        )


def test_host_stt_selection_uses_explicit_moonshine_tier_without_whisper_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    moonshine_instance = object()

    def fake_moonshine(**kwargs: object) -> object:
        calls.append(("moonshine", kwargs))
        return moonshine_instance

    def forbidden_whisper(**kwargs: object) -> object:
        calls.append(("whisper", kwargs))
        raise AssertionError("Moonshine selection must not construct Faster Whisper")

    monkeypatch.setattr(host_launcher_module, "MoonshineStreamingTranscriber", fake_moonshine)
    monkeypatch.setattr(host_launcher_module, "FasterWhisperTranscriber", forbidden_whisper)

    selected = _build_streaming_transcriber(
        stt_provider="moonshine",
        whisper_model="base.en",
        moonshine_model_tier="medium",
        moonshine_update_interval_seconds=0.2,
    )

    assert selected is moonshine_instance
    assert calls == [
        (
            "moonshine",
            {
                "language": "en",
                "model_tier": "medium",
                "update_interval_seconds": 0.2,
                "sample_rate_hz": 48_000,
                "channels": 1,
            },
        )
    ]
    with pytest.raises(ValueError, match="stt_provider"):
        _build_streaming_transcriber(
            stt_provider="automatic",
            whisper_model="base.en",
            moonshine_model_tier="tiny",
            moonshine_update_interval_seconds=0.2,
        )
    assert len(calls) == 1


def test_host_tts_selection_uses_kokoro_isabella_without_edge_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    kokoro_instance = object()

    def fake_kokoro(**kwargs: object) -> object:
        calls.append(("kokoro", kwargs))
        return kokoro_instance

    def forbidden_edge(**kwargs: object) -> object:
        calls.append(("edge", kwargs))
        raise AssertionError("Kokoro selection must not construct Edge TTS")

    monkeypatch.setattr(host_launcher_module, "KokoroSynthesizer", fake_kokoro)
    monkeypatch.setattr(host_launcher_module, "EdgeTtsSynthesizer", forbidden_edge)

    selected = _build_synthesizer(
        tts_provider="kokoro",
        edge_voice="en-US-AriaNeural",
        kokoro_voice="bf_isabella",
        kokoro_speed=1.0,
        kokoro_intra_op_threads=8,
        kokoro_pronunciations={"Hermes": "Her-mees", "LiveKit": "Live Kit"},
        kokoro_stream_chunk_chars=320,
        kokoro_worker_python=Path("C:/isolated-kokoro/python.exe"),
    )

    assert selected is kokoro_instance
    assert calls == [
        (
            "kokoro",
            {
                "voice": "bf_isabella",
                "speed": 1.0,
                "intra_op_threads": 8,
                "pronunciations": {"Hermes": "Her-mees", "LiveKit": "Live Kit"},
                "stream_chunk_chars": 320,
                "worker_python": Path("C:/isolated-kokoro/python.exe"),
            },
        )
    ]
    with pytest.raises(ValueError, match="tts_provider"):
        _build_synthesizer(
            tts_provider="automatic",
            edge_voice="en-US-AriaNeural",
            kokoro_voice="bf_isabella",
            kokoro_speed=1.0,
            kokoro_intra_op_threads=8,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "arguments",
    (
        ["--evidence-capture"],
        ["--evidence-retention-hours", "24"],
        ["--evidence-db", r"C:\capture\capture-v1.sqlite3"],
        ["--evidence-status"],
        ["--purge-evidence"],
    ),
)
def test_local_launcher_rejects_host_only_evidence_flags(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes-realtime-local", *arguments],
    )

    with pytest.raises(SystemExit) as raised:
        launcher_module.main()

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments" in captured.err


def test_host_parser_exposes_moonshine_streaming_tiers_with_tiny_default() -> None:
    parser = host_launcher_module._build_argument_parser()

    assert parser.parse_args([]).moonshine_model_tier == "tiny"
    assert parser.parse_args(["--moonshine-model-tier", "small"]).moonshine_model_tier == "small"
    assert parser.parse_args(["--moonshine-model-tier", "medium"]).moonshine_model_tier == "medium"


def test_host_parser_exposes_bounded_default_off_knowledge_controls() -> None:
    parser = host_launcher_module._build_argument_parser()

    defaults = parser.parse_args([])
    enabled = parser.parse_args(
        [
            "--knowledge-speculation",
            "--knowledge-recovery",
            "--knowledge-budget-seconds",
            "2.25",
        ]
    )

    assert defaults.knowledge_speculation is False
    assert defaults.knowledge_recovery is False
    assert defaults.knowledge_budget_seconds == 3.5
    assert enabled.knowledge_speculation is True
    assert enabled.knowledge_recovery is True
    assert enabled.knowledge_budget_seconds == 2.25
    with pytest.raises(SystemExit):
        parser.parse_args(["--knowledge-budget-seconds", "31"])


@pytest.mark.asyncio
async def test_host_cli_passes_moonshine_tier_to_host_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class LauncherStub:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=private"

        async def close(self) -> None:
            return None

    monkeypatch.setattr(host_launcher_module, "_load_tailnet_authorizer", lambda **_kwargs: None)
    monkeypatch.setattr(host_launcher_module, "_load_api_bearer", lambda _path: "b" * 32)
    monkeypatch.setattr(
        host_launcher_module,
        "_load_livekit_credentials",
        lambda **_kwargs: ("key", "s" * 32),
    )
    monkeypatch.setattr(
        host_launcher_module,
        "build_local_host_launcher",
        lambda **kwargs: captured.update(kwargs) or LauncherStub(),
    )
    args = host_launcher_module._build_argument_parser().parse_args(
        [
            "--moonshine-model-tier",
            "small",
            "--evidence-capture",
            "--evidence-retention-hours",
            "48",
            "--evidence-db",
            r"C:\capture\capture-v1.sqlite3",
        ]
    )
    running = asyncio.create_task(host_launcher_module._run_host_cli(args))
    await asyncio.sleep(0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert captured["moonshine_model_tier"] == "small"
    assert captured["knowledge_speculation"] is False
    assert captured["knowledge_recovery"] is False
    assert captured["knowledge_budget_seconds"] == 3.5
    assert captured["evidence_capture"] is True
    assert captured["evidence_retention_hours"] == 48
    assert captured["evidence_database"] == Path(r"C:\capture\capture-v1.sqlite3")


def test_host_parser_accepts_bounded_repeatable_kokoro_pronunciations() -> None:
    args = host_launcher_module._build_argument_parser().parse_args(
        [
            "--kokoro-pronunciation",
            "Hermes=Her-mees",
            "--kokoro-pronunciation",
            "LiveKit=Live Kit",
            "--kokoro-stream-chunk-chars",
            "320",
            "--kokoro-worker-python",
            sys.executable,
        ]
    )

    assert host_launcher_module._parse_kokoro_pronunciations(args.kokoro_pronunciation) == {
        "Hermes": "Her-mees",
        "LiveKit": "Live Kit",
    }
    assert args.kokoro_stream_chunk_chars == 320
    assert args.kokoro_worker_python == Path(sys.executable).resolve()


def test_host_parser_accepts_explicit_remote_profile() -> None:
    args = host_launcher_module._build_argument_parser().parse_args(
        [
            "--remote",
            "--livekit-url",
            "wss://voice.example.test",
            "--browser-host",
            "192.0.2.10",
            "--browser-origin",
            "https://assistant.example.test:8765",
            "--tls-cert",
            "C:/runtime/cert.pem",
            "--tls-key",
            "C:/runtime/key.pem",
        ]
    )

    assert args.remote is True
    assert args.livekit_url == "wss://voice.example.test"
    assert args.browser_host == "192.0.2.10"
    assert args.browser_origin == "https://assistant.example.test:8765"
    assert args.tls_cert == "C:/runtime/cert.pem"
    assert args.tls_key == "C:/runtime/key.pem"


def test_remote_host_requires_both_tls_files_before_startup() -> None:
    with pytest.raises(ValueError, match="both --tls-cert and --tls-key"):
        host_launcher_module._build_server_ssl_context(
            remote_mode=True,
            certificate_file=None,
            private_key_file=Path("key.pem"),
            canonical_origin="https://assistant.example.test:8765",
        )


def test_remote_host_certificate_must_cover_canonical_origin(
    tls_certificates: dict[str, Path],
) -> None:
    context = host_launcher_module._build_server_ssl_context(
        remote_mode=True,
        certificate_file=tls_certificates["certificate"],
        private_key_file=tls_certificates["private_key"],
        canonical_origin="https://assistant.example.test:8765",
    )
    assert type(context) is ssl.SSLContext

    with pytest.raises(ValueError, match="certificate does not cover canonical origin"):
        host_launcher_module._build_server_ssl_context(
            remote_mode=True,
            certificate_file=tls_certificates["certificate"],
            private_key_file=tls_certificates["private_key"],
            canonical_origin="https://other.example.test:8765",
        )

    with pytest.raises(ValueError, match="certificate does not cover canonical origin"):
        host_launcher_module._build_server_ssl_context(
            remote_mode=True,
            certificate_file=tls_certificates["cn_only_certificate"],
            private_key_file=tls_certificates["cn_only_private_key"],
            canonical_origin="https://assistant.example.test:8765",
        )


def test_remote_certificate_validation_does_not_require_removed_ssl_api(
    monkeypatch: pytest.MonkeyPatch,
    tls_certificates: dict[str, Path],
) -> None:
    monkeypatch.delattr(ssl, "match_hostname")

    context = host_launcher_module._build_server_ssl_context(
        remote_mode=True,
        certificate_file=tls_certificates["certificate"],
        private_key_file=tls_certificates["private_key"],
        canonical_origin="https://assistant.example.test:8765",
    )

    assert type(context) is ssl.SSLContext


def test_remote_host_requires_environment_livekit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="LIVEKIT_API_KEY and LIVEKIT_API_SECRET"):
        host_launcher_module._load_livekit_credentials(remote_mode=True)


@pytest.mark.asyncio
async def test_remote_host_wires_secure_browser_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tls_certificates: dict[str, Path],
) -> None:
    captured: dict[str, object] = {}
    transcriber_configuration: dict[str, object] = {}
    foreground_configuration: dict[str, object] = {}
    speech_configuration: dict[str, object] = {}
    foreground_constructor = host_launcher_module.ForegroundTurnCoordinator
    speech_constructor = host_launcher_module.StreamingSpeechLoop

    def capture_foreground(**kwargs: Any) -> Any:
        foreground_configuration.update(kwargs)
        return foreground_constructor(**kwargs)

    def capture_speech(**kwargs: Any) -> Any:
        speech_configuration.update(kwargs)
        return speech_constructor(**kwargs)

    class RuntimeStub:
        async def start(self) -> str:
            return "https://assistant.example.test:8765/#bootstrap=private"

        async def close(self) -> None:
            return None

    def capture_runtime(**kwargs: object) -> RuntimeStub:
        captured.update(kwargs)
        return RuntimeStub()

    def capture_transcriber(**kwargs: object) -> object:
        transcriber_configuration.update(kwargs)
        return object()

    monkeypatch.setattr(host_launcher_module, "BrowserClientRuntime", capture_runtime)
    monkeypatch.setattr(
        host_launcher_module,
        "ForegroundTurnCoordinator",
        capture_foreground,
    )
    monkeypatch.setattr(host_launcher_module, "StreamingSpeechLoop", capture_speech)
    monkeypatch.setattr(
        host_launcher_module,
        "_build_streaming_transcriber",
        capture_transcriber,
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_synthesizer",
        lambda **_kwargs: ProviderProbe([], "synthesizer"),
    )

    launcher = build_local_host_launcher(
        hermes_api_bearer="b" * 32,
        livekit_url="wss://voice.example.test",
        livekit_api_key="remote-key",
        livekit_api_secret="s" * 32,
        browser_host="192.0.2.10",
        browser_canonical_origin="https://assistant.example.test:8765",
        browser_certificate_file=tls_certificates["certificate"],
        browser_private_key_file=tls_certificates["private_key"],
        remote_mode=True,
        inference_provider="ollama",
        stt_provider="faster-whisper",
        moonshine_model_tier="medium",
        tts_provider="edge",
        allow_unsandboxed_tasks=True,
    )

    assert isinstance(launcher, LocalBrowserLauncher)
    assert captured["host"] == "192.0.2.10"
    assert captured["lan_mode"] is True
    assert type(captured["ssl_context"]) is ssl.SSLContext
    assert captured["canonical_origin"] == "https://assistant.example.test:8765"
    assert transcriber_configuration["moonshine_model_tier"] == "medium"
    assert (
        foreground_configuration.get("output_capacity")
        == speech_configuration.get("max_segments")
        == DEFAULT_MAX_RESPONSE_SEGMENTS
    )


@pytest.mark.asyncio
async def test_full_host_capture_rejects_unsupported_platform_before_provider_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden_provider(**_kwargs: object) -> object:
        raise AssertionError("unsupported evidence capture must fail before provider loading")

    monkeypatch.setattr(host_launcher_module.sys, "platform", "linux")
    monkeypatch.setattr(host_launcher_module, "_build_streaming_inference", forbidden_provider)

    with pytest.raises(RuntimeError, match="unsupported_platform"):
        build_local_host_launcher(
            hermes_api_bearer="b" * 32,
            allow_unsandboxed_tasks=True,
            evidence_capture=True,
            evidence_database=tmp_path / "capture-v1.sqlite3",
        )


@pytest.mark.asyncio
async def test_full_host_capture_available_composes_one_unconsented_final_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_owner: dict[str, object] = {}
    invalidation_reasons: list[object] = []
    speech_configuration: dict[str, object] = {}
    action_configuration: dict[str, object] = {}
    browser_configuration: dict[str, object] = {}
    retention_configuration: dict[str, object] = {}
    speech_constructor = host_launcher_module.StreamingSpeechLoop
    action_constructor = host_launcher_module.ConversationUpdateExecutor

    class EvidenceOwnerProbe:
        evidence_lifecycle = None
        evidence_admission = None

        def __init__(self) -> None:
            from hermes_realtime.evidence import (
                ConversationOperationScheduler,
                EvidenceLifecycleOwner,
            )

            self.operation_scheduler = ConversationOperationScheduler(
                owner_generation=77,
                max_operations=16,
            )
            self.lifecycle_owner = EvidenceLifecycleOwner(
                owner_generation=77,
                operation_scheduler=self.operation_scheduler,
            )

        @property
        def conversation_authority(self) -> object:
            return self.lifecycle_owner.conversation_authority

        def resolve_evidence_pair(self) -> tuple[None, None]:
            return None, None

        def capture_status(self, *, disclosure_digest: str) -> object:
            del disclosure_digest
            return object()

        def reserve_browser_revoke(self, **kwargs: object) -> object:
            del kwargs
            return object()

        async def activate_revoke(self, *_args: object, **_kwargs: object) -> object:
            return object()

        def configure_retention_owner(self, **kwargs: object) -> None:
            retention_configuration.update(kwargs)

        async def invalidate_active_binding(self, reason: object) -> None:
            invalidation_reasons.append(reason)

        async def close(self) -> None:
            return None

    evidence_owner = EvidenceOwnerProbe()

    def capture_evidence_owner(**kwargs: object) -> EvidenceOwnerProbe:
        captured_owner.update(kwargs)
        return evidence_owner

    def capture_speech(**kwargs: Any) -> Any:
        speech_configuration.update(kwargs)
        return speech_constructor(**kwargs)

    def capture_actions(**kwargs: Any) -> Any:
        action_configuration.update(kwargs)
        return action_constructor(**kwargs)

    class RuntimeStub:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=private"

        async def close(self) -> None:
            return None

    monkeypatch.setattr(host_launcher_module, "HostEvidenceRuntimeV1", capture_evidence_owner)
    def capture_browser(**kwargs: object) -> RuntimeStub:
        browser_configuration.update(kwargs)
        return RuntimeStub()

    monkeypatch.setattr(host_launcher_module, "BrowserClientRuntime", capture_browser)
    monkeypatch.setattr(host_launcher_module, "StreamingSpeechLoop", capture_speech)
    monkeypatch.setattr(host_launcher_module, "ConversationUpdateExecutor", capture_actions)
    monkeypatch.setattr(
        host_launcher_module,
        "_build_streaming_transcriber",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_synthesizer",
        lambda **_kwargs: ProviderProbe([], "synthesizer"),
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_voice_activity_detector",
        lambda **_kwargs: type("VadProbe", (), {"required_pre_roll_frames": 1})(),
    )

    database = tmp_path / "capture-v1.sqlite3"
    launcher = build_local_host_launcher(
        hermes_api_bearer="b" * 32,
        stt_provider="faster-whisper",
        tts_provider="edge",
        allow_unsandboxed_tasks=True,
        evidence_capture=True,
        evidence_retention_hours=24,
        evidence_database=database,
    )

    assert captured_owner["database"] == database
    assert captured_owner["retention_hours"] == 24
    assert type(captured_owner["owner_generation"]) is int
    assert int(captured_owner["owner_generation"]) > 0
    observations = captured_owner["production_observations"]
    recorder = captured_owner["production_observation_recorder"]
    assert type(observations).__name__ == "ProductionObservationViewV1"
    assert type(recorder).__name__ == "_ProductionObservationRecorderV1"
    assert speech_configuration["evidence_admission"] is None
    assert speech_configuration["evidence_resolver"] == evidence_owner.resolve_evidence_pair
    assert speech_configuration["production_observation_recorder"] is recorder
    cancel = retention_configuration["cancel"]
    assert callable(cancel)
    assert cancel.__func__ is speech_constructor.cancel_for_retention_expiry  # type: ignore[attr-defined]
    assert callable(browser_configuration["evidence_consent"])
    assert callable(browser_configuration["evidence_status"])
    assert callable(browser_configuration["evidence_revoke"])
    invalidate = browser_configuration["evidence_invalidate"]
    assert callable(invalidate)
    from hermes_realtime.evidence.models import BindingCloseReason

    await invalidate(BindingCloseReason.MEDIA_INCARNATION_REPLACED)
    assert invalidation_reasons == [BindingCloseReason.MEDIA_INCARNATION_REPLACED]
    assert action_configuration["operation_scheduler"] is evidence_owner.operation_scheduler
    assert action_configuration["evidence_resolver"] == evidence_owner.resolve_evidence_pair
    assert action_configuration["production_observation_recorder"] is recorder
    binding_configuration: dict[str, object] = {}

    class BindingProbe:
        def __init__(self, **kwargs: object) -> None:
            binding_configuration.update(kwargs)

    monkeypatch.setattr(host_launcher_module, "ConversationSessionWorker", BindingProbe)
    livekit_worker = browser_configuration["worker"]
    conversation_runtime = livekit_worker._runtime  # type: ignore[attr-defined]
    conversation_runtime._binding_factory(  # type: ignore[attr-defined]
        "browser_0123456789abcdef",
        1,
        conversation_runtime._actions,  # type: ignore[attr-defined]
    )
    lifecycle_resolver = binding_configuration["evidence_lifecycle_resolver"]
    assert callable(lifecycle_resolver)
    assert lifecycle_resolver() is None
    assert binding_configuration["production_observation_recorder"] is recorder
    assert launcher.production_observations is observations
    assert launcher._providers[-1] is evidence_owner


@pytest.mark.asyncio
async def test_host_revoke_observer_publishes_terminal_capture_status() -> None:
    from hermes_realtime.client import BrowserBindingSnapshot, BrowserEventProjection
    from hermes_realtime.evidence import models as m

    projection = BrowserEventProjection(capacity=4)
    revoked, terminal = projection.reserve_capture_status_slots(2)
    observed: list[object] = []
    retained = iter((revoked, terminal))

    class Runtime:
        def take_lifecycle_status_reservation(self) -> m.ProjectionReservation:
            return next(retained)

        def reserve_browser_revoke(self, **_kwargs: object) -> object:
            return object()

        def capture_status(self, *, disclosure_digest: str) -> m.CaptureStatusV1:
            state = m.CaptureState.REVOKED_PURGING if not observed else m.CaptureState.IDLE
            return m.CaptureStatusV1(
                available=True,
                capture_state=state,
                retention_hours=24,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest=disclosure_digest,
            )

        async def activate_revoke(
            self,
            _authority: object,
            *,
            timeout_seconds: float,
        ) -> m.RevokeDisposition:
            assert timeout_seconds == 2.0
            return m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        def observe_revoke_terminal(
            self,
            callback: object,
            release_unpublished: object,
        ) -> None:
            assert callable(release_unpublished)
            observed.append(callback)

    operation = host_launcher_module._reserve_host_evidence_revoke(
        runtime=Runtime(),  # type: ignore[arg-type]
        projection=projection,
        binding=BrowserBindingSnapshot(
            participant_identity="browser_0123456789abcdef",
            binding_generation=1,
        ),
        request=m.EvidenceRevokeRequestV1(sequence=2),
    )
    response = await operation.complete()
    assert response.status == 202
    callback = observed[0]
    assert callable(callback)
    assert callback(m.RevokeDisposition.PURGE_COMPLETED) is True
    assert callback(m.RevokeDisposition.PURGE_COMPLETED) is True

    statuses = [event for event in projection.events_after(0) if event.kind == "capture_status"]
    assert [event.data["captureState"] for event in statuses] == ["revoked_purging", "idle"]


@pytest.mark.asyncio
async def test_conversation_only_launcher_aligns_foreground_queue_and_segment_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreground_configuration: dict[str, object] = {}
    speech_configuration: dict[str, object] = {}
    foreground_constructor = launcher_module.ForegroundTurnCoordinator
    speech_constructor = launcher_module.StreamingSpeechLoop

    def capture_foreground(**kwargs: Any) -> Any:
        foreground_configuration.update(kwargs)
        return foreground_constructor(**kwargs)

    def capture_speech(**kwargs: Any) -> Any:
        speech_configuration.update(kwargs)
        return speech_constructor(**kwargs)

    monkeypatch.setattr(launcher_module, "ForegroundTurnCoordinator", capture_foreground)
    monkeypatch.setattr(launcher_module, "StreamingSpeechLoop", capture_speech)
    monkeypatch.setattr(
        launcher_module,
        "OllamaStreamingInference",
        lambda **_kwargs: ProviderProbe([], "inference"),
    )
    monkeypatch.setattr(
        launcher_module,
        "EdgeTtsSynthesizer",
        lambda **_kwargs: ProviderProbe([], "synthesizer"),
    )
    monkeypatch.setattr(launcher_module, "FasterWhisperTranscriber", lambda **_kwargs: object())

    launcher = build_local_conversation_launcher()

    assert isinstance(launcher, LocalBrowserLauncher)
    assert (
        foreground_configuration.get("output_capacity")
        == speech_configuration.get("max_segments")
        == DEFAULT_MAX_RESPONSE_SEGMENTS
    )


@pytest.mark.parametrize("browser_host", ["127.0.0.1", "localhost", "127.1", "2130706433"])
def test_remote_host_rejects_loopback_browser_listener_before_provider_loading(
    browser_host: str,
) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "tls"

    with pytest.raises(ValueError, match="explicitly non-loopback"):
        build_local_host_launcher(
            hermes_api_bearer="b" * 32,
            livekit_url="wss://voice.example.test",
            livekit_api_key="remote-key",
            livekit_api_secret="s" * 32,
            browser_host=browser_host,
            browser_canonical_origin="https://assistant.example.test:8765",
            browser_certificate_file=fixture_root / "assistant-cert.pem",
            browser_private_key_file=fixture_root / "assistant-key.pem",
            remote_mode=True,
            allow_unsandboxed_tasks=True,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--kokoro-stream-chunk-chars", "31"],
        ["--kokoro-stream-chunk-chars", "1025"],
        ["--kokoro-pronunciation", f"{'x' * 65}=spoken"],
        ["--kokoro-worker-python", "relative/python.exe"],
    ],
)
def test_host_parser_rejects_invalid_kokoro_speech_configuration(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        host_launcher_module._build_argument_parser().parse_args(arguments)

    assert raised.value.code == 2


def test_host_main_rejects_duplicate_pronunciations_before_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host_launcher_module.sys,
        "argv",
        [
            "hermes-realtime-host",
            "--kokoro-pronunciation",
            "Hermes=Her-mees",
            "--kokoro-pronunciation",
            "HERMES=other",
        ],
    )

    async def forbidden_run(_args: object) -> None:
        raise AssertionError("invalid CLI input must not start the host")

    monkeypatch.setattr(host_launcher_module, "_run_host_cli", forbidden_run)
    with pytest.raises(SystemExit) as raised:
        host_launcher_module.main()

    assert raised.value.code == 2


def test_host_tts_defaults_are_platform_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    default_provider = host_launcher_module._default_tts_provider

    monkeypatch.setattr(host_launcher_module.sys, "platform", "win32")
    assert default_provider() == "kokoro"

    monkeypatch.setattr(host_launcher_module.sys, "platform", "linux")
    assert default_provider() == "edge"

    monkeypatch.setattr(host_launcher_module.sys, "platform", "darwin")
    assert default_provider() == "edge"


def test_local_launcher_rejects_non_loopback_livekit_before_provider_loading() -> None:
    with pytest.raises(ValueError, match="loopback"):
        build_local_conversation_launcher(livekit_url="wss://livekit.example.com")


def test_speech_timing_event_is_compact_turn_and_stream_correlated() -> None:
    chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="kokoro_abc123",
        text="😀 Hello there",
        audio=AudioFrame(
            pcm=b"\x01\x00" * 240,
            sample_rate_hz=48_000,
            channels=1,
        ),
        word_timings=(
            WordTiming("Hello", 0, 100, 2, 7),
            WordTiming("there", 100, 240, 8, 13),
        ),
        timing_source="estimated",
    )

    assert _encode_speech_timing_event(
        chunk,
        "speech_def456",
        turn_generation=7,
    ) == {
        "turnId": "turn_001",
        "presentationTurnId": "turn_001",
        "turnGeneration": 7,
        "chunkId": "kokoro_abc123",
        "segmentId": "kokoro_abc123",
        "streamId": "speech_def456",
        "sampleRate": 48_000,
        "timingSource": "estimated",
        "timings": "3,8,0,100;9,14,100,240",
    }
    presentation_timing = _encode_speech_timing_event(
        chunk,
        "speech_def456",
        turn_generation=7,
        presentation_turn_id="turn_original",
        segment_id="segment_2",
        segment_text_offset_utf16=20,
    )
    assert presentation_timing is not None
    assert presentation_timing["turnId"] == chunk.turn_id
    assert presentation_timing["presentationTurnId"] == "turn_original"
    assert presentation_timing["chunkId"] == chunk.chunk_id
    assert presentation_timing["segmentId"] == "segment_2"
    assert presentation_timing["timings"] == "23,28,0,100;29,34,100,240"


def test_speech_timing_event_is_absent_without_trustworthy_timing() -> None:
    chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="edge_abc123",
        text="Hello",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=48_000, channels=1),
    )

    assert _encode_speech_timing_event(chunk, "speech_def456", turn_generation=7) is None


def test_full_host_rejects_non_loopback_hermes_api_before_provider_loading() -> None:
    with pytest.raises(ValueError, match="loopback"):
        build_local_host_launcher(
            hermes_api_bearer="host-test-bearer-value-32-characters",
            hermes_api_url="https://hermes.example.com",
            allow_unsandboxed_tasks=True,
        )


def test_full_host_requires_explicit_unsandboxed_task_opt_in() -> None:
    with pytest.raises(PermissionError, match="unsandboxed"):
        build_local_host_launcher(
            hermes_api_bearer="host-test-bearer-value-32-characters",
        )


@pytest.mark.asyncio
async def test_qualification_no_task_composition_routes_through_real_command_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_router = host_launcher_module.ConversationTaskCommandRouter

    def capture_router(**kwargs: object) -> object:
        captured.update(kwargs)
        return original_router(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        host_launcher_module,
        "ConversationTaskCommandRouter",
        capture_router,
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_streaming_transcriber",
        lambda **_kwargs: ProviderProbe([], "transcriber"),
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_synthesizer",
        lambda **_kwargs: ProviderProbe([], "synthesizer"),
    )
    with host_launcher_module._qualification_no_task_composition():
        launcher = build_local_host_launcher(
            hermes_api_bearer=None,
            evidence_capture=True,
            evidence_database=Path("capture-v1.sqlite3"),
        )
    try:
        assert type(captured["surface"]) is host_launcher_module.ConversationWorkControlSurface
        assert captured["lifecycle_owner"] is not None
    finally:
        await launcher.close()


@pytest.mark.asyncio
async def test_qualification_no_task_composition_does_not_require_public_task_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host_launcher_module,
        "_build_streaming_transcriber",
        lambda **_kwargs: ProviderProbe([], "transcriber"),
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_synthesizer",
        lambda **_kwargs: ProviderProbe([], "synthesizer"),
    )
    with host_launcher_module._qualification_no_task_composition():
        launcher = build_local_host_launcher(
            hermes_api_bearer=None,
            evidence_capture=True,
            evidence_database=Path("capture-v1.sqlite3"),
        )

    await launcher.close()


def test_full_host_rejects_natural_work_tools_for_non_codex_before_loading_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_provider_loading(**_kwargs: object) -> object:
        raise AssertionError("unsupported natural work routing must fail before providers load")

    monkeypatch.setattr(
        host_launcher_module,
        "_build_streaming_inference",
        forbidden_provider_loading,
    )

    with pytest.raises(ValueError, match="requires.*codex"):
        build_local_host_launcher(
            hermes_api_bearer="host-test-bearer-value-32-characters",
            inference_provider="ollama",
            natural_work_tools=True,
            allow_unsandboxed_tasks=True,
        )


@pytest.mark.asyncio
async def test_full_host_explicit_commands_share_the_composed_work_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_router = host_launcher_module.ConversationTaskCommandRouter

    def capture_router(**kwargs: object) -> object:
        captured.update(kwargs)
        return original_router(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        host_launcher_module,
        "ConversationTaskCommandRouter",
        capture_router,
    )

    class TranscriberStub:
        async def push(self, _frame: AudioFrame) -> tuple[()]:
            return ()

        async def finish_utterance(self) -> None:
            return None

        async def cancel(self) -> None:
            return None

    monkeypatch.setattr(
        host_launcher_module,
        "FasterWhisperTranscriber",
        lambda **_kwargs: TranscriberStub(),
    )
    monkeypatch.setattr(
        host_launcher_module,
        "_build_synthesizer",
        lambda **_kwargs: ProviderProbe([], "synthesizer"),
    )
    launcher = build_local_host_launcher(
        hermes_api_bearer="host-test-bearer-value-32-characters",
        inference_provider="ollama",
        stt_provider="faster-whisper",
        tts_provider="edge",
        allow_unsandboxed_tasks=True,
    )
    try:
        surface = captured["surface"]
        assert type(surface) is host_launcher_module.ConversationWorkControlSurface
        assert "controller" not in captured
        close_owner = next(
            provider for provider in launcher._providers if hasattr(provider, "_owners")
        )
        assert close_owner._owners[1] is surface
    finally:
        await launcher.close()


def test_full_host_loads_only_one_strong_api_key_from_explicit_env_file(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    bearer = "env-file-test-bearer-value-32-characters"
    env_file.write_text(
        f"DISCORD_BOT_TOKEN=must-not-be-loaded\nAPI_SERVER_KEY={bearer}\n",
        encoding="utf-8",
    )
    assert _load_api_bearer(env_file) == bearer

    env_file.write_text(
        f"API_SERVER_KEY={bearer}\nAPI_SERVER_KEY={bearer}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="one explicit strong"):
        _load_api_bearer(env_file)


def test_full_host_loads_bounded_compact_hermes_context(tmp_path: Path) -> None:
    context_file = tmp_path / "realtime-context.json"
    context_file.write_text(
        '{"version":1,"identity":"MrAnderson","persona":"composed, dry, precise",'
        '"user_preferences":"lead with the answer","location":null}',
        encoding="utf-8",
    )

    assert _load_hermes_context(context_file) == HermesRepresentativeContext(
        identity="MrAnderson",
        persona="composed, dry, precise",
        user_preferences="lead with the answer",
    )
    assert _load_hermes_context(None) is None

    context_file.write_bytes(b" " * 4097)
    with pytest.raises(RuntimeError, match="4096-byte"):
        _load_hermes_context(context_file)

    context_file.write_bytes(b"{\xff}")
    with pytest.raises(RuntimeError, match="UTF-8"):
        _load_hermes_context(context_file)

    for invalid in (
        "",
        '{"version":1,"identity":"Hermes","identity":"Other"}',
        '{"version":1,"identity":"Hermes","unknown":true}',
        '{"version":true,"identity":"Hermes"}',
        '{"version":1.0,"identity":"Hermes"}',
        '{"version":1,"identity":"Hermes\\u202ehidden"}',
    ):
        context_file.write_text(invalid, encoding="utf-8")
        with pytest.raises(RuntimeError):
            _load_hermes_context(context_file)


@pytest.mark.asyncio
async def test_full_host_policy_speaks_completion_when_user_and_foreground_are_idle() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(
        projection=projection,
        floor_available=lambda: True,
    )
    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_release_check",
                status="completed",
                summary="release evidence is consistent",
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.kind == "interrupt"
    assert directive.text == "Release evidence is consistent."
    assert "task_release_check" not in directive.text

    assert [(event.kind, dict(event.data)) for event in projection.events_after(0)] == [
        (
            "completion_received",
            {"status": "completed", "taskId": "task_release_check"},
        ),
        (
            "task_state",
            {"status": "completed", "taskId": "task_release_check"},
        ),
        (
            "task_result",
            {
                "status": "completed",
                "taskId": "task_release_check",
                "text": "release evidence is consistent",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_full_host_policy_defers_completion_while_conversation_owns_floor() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(
        projection=projection,
        floor_available=lambda: False,
    )

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_busy_floor",
                status="completed",
                summary="the current information is ready",
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.kind == "mention_next"
    assert directive.text == "The current information is ready."


@pytest.mark.parametrize(
    ("status", "detail", "expected"),
    (
        (
            "failed",
            "LiveKit handshake failed",
            "I ran into a problem with that work. LiveKit handshake failed.",
        ),
        (
            "interrupted",
            "the user stopped it",
            "That work stopped. The user stopped it.",
        ),
    ),
)
def test_completion_failure_leads_are_conversational(
    status: str,
    detail: str,
    expected: str,
) -> None:
    spoken = host_launcher_module._voice_friendly_completion_text(status, detail)
    assert spoken == expected


def test_completion_with_sanitized_empty_detail_hands_off_to_screen() -> None:
    assert host_launcher_module._voice_friendly_completion_text("completed", "•") == (
        "The full result is on screen."
    )


def test_full_host_contains_no_delayed_progress_advisory_surface() -> None:
    assert not hasattr(host_launcher_module, "_BackgroundProgressCueScheduler")
    assert not hasattr(ConversationUpdateExecutor, "announce_advisory")
    assert not hasattr(StreamingSpeechLoop, "_announce_advisory")
    assert "admit" not in inspect.signature(ForegroundTurnLease.publish).parameters
    run_turn_parameters = inspect.signature(StreamingSpeechLoop._run_turn).parameters
    assert "replace_active" not in run_turn_parameters
    assert "admission_guard" not in run_turn_parameters


@pytest.mark.asyncio
async def test_full_host_policy_keeps_machine_text_visual_and_voice_friendly() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    summary = (
        "See https://github.com/NousResearch/hermes-agent/releases/tag/v0.19.1 "
        + "for details. "
        + ("A substantial stabilization detail follows. " * 15)
    )

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_087a306b24da49c788d2d2cf825c31fd",
                status="completed",
                summary=summary,
            ),
            context=ConversationContextStore(max_item_chars=4096).snapshot(),
        )
    )

    assert directive.text is not None
    assert "task_087a306b24da49c788d2d2cf825c31fd" not in directive.text
    assert "https://" not in directive.text
    assert "the source link" in directive.text
    assert "background result" not in directive.text.casefold()
    assert directive.text.count("A substantial stabilization detail follows.") > 10
    task_result = projection.events_after(0)[-1]
    assert task_result.kind == "task_result"
    assert task_result.data["text"] == summary


@pytest.mark.asyncio
async def test_completed_work_speaks_topic_before_completion_logistics() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_topic_first",
                status="completed",
                summary=(
                    "<voice>The release checks are clean.</voice>"
                    "<detail>All release checks passed.</detail>"
                ),
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == "All release checks passed."
    assert "found" not in directive.text.casefold()


@pytest.mark.asyncio
async def test_completed_work_gives_inference_every_material_background_finding() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    summary = (
        "<voice>Several major stories moved today.</voice>"
        "<detail>AP says Iran and Oman moved closer to reopening the Strait of Hormuz, "
        "but there is no final deal. Reuters reports Russian missiles killed 17 people "
        "near Kyiv. Reuters says Taiwan began major defensive drills. AP reports El Niño "
        "could make 2026 the hottest year on record.</detail>"
    )

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_news_roundup",
                status="completed",
                summary=summary,
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text is not None
    assert "Iran and Oman" in directive.text
    assert "17 people" in directive.text
    assert "Taiwan" in directive.text
    assert "hottest year" in directive.text
    assert len(directive.text) > len("Several major stories moved today.")


@pytest.mark.asyncio
async def test_completion_context_supplies_material_once_without_spoken_scaffolding() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    detail = (
        "The official calendar lists the next council meeting on August 18. "
        "The location is Council Chambers."
    )

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_public_meetings",
                status="completed",
                summary=(
                    "<voice>The next council meeting is August 18.</voice>"
                    f"<detail>{detail}</detail>"
                ),
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == detail
    assert "background result" not in directive.text.casefold()
    assert directive.text.count("August 18") == 1


@pytest.mark.asyncio
async def test_full_host_policy_separates_conversational_voice_from_visual_detail() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    summary = (
        "<voice>Indiana plays Portland tonight at seven Pacific. Indiana is favored by seven "
        "and a half.</voice>\n"
        "<detail>Indiana Fever at Portland Fire, 7:00 PM Pacific, Moda Center. DraftKings: "
        "Indiana -7.5, moneyline -325; Portland +260. Total 186.5.</detail>"
    )

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_wnba",
                status="completed",
                summary=summary,
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == (
        "Indiana Fever at Portland Fire, 7:00 PM Pacific, Moda Center. DraftKings: "
        "Indiana -7.5, moneyline -325; Portland +260. Total 186.5."
    )
    task_result = projection.events_after(0)[-1]
    assert task_result.data["text"] == (
        "Indiana Fever at Portland Fire, 7:00 PM Pacific, Moda Center. DraftKings: "
        "Indiana -7.5, moneyline -325; Portland +260. Total 186.5."
    )
    assert "MLB" not in directive.text


@pytest.mark.asyncio
async def test_full_host_policy_does_not_speak_malformed_structured_result() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    overlong_voice = "x" * 241

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_malformed_voice",
                status="completed",
                summary=(
                    f"<voice>{overlong_voice}</voice>"
                    "<detail>The bounded visual result remains available.</detail>"
                ),
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == "The bounded visual result remains available."
    assert projection.events_after(0)[-1].data["text"] == (
        "The bounded visual result remains available."
    )


@pytest.mark.asyncio
async def test_full_host_policy_supplies_bounded_visual_detail_once() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    detail = "x" * 721

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_long_detail",
                status="completed",
                summary=f"<voice>The short answer is ready.</voice><detail>{detail}</detail>",
            ),
            context=ConversationContextStore(max_item_chars=2048).snapshot(),
        )
    )

    assert directive.text is not None
    assert directive.text == detail
    assert projection.events_after(0)[-1].data["text"] == detail


@pytest.mark.asyncio
async def test_full_host_policy_strips_stray_closing_tag_without_speaking_it() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_stray_tag",
                status="completed",
                summary="Indiana is favored by seven.</detail>",
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == "Indiana is favored by seven."
    assert projection.events_after(0)[-1].data["text"] == "Indiana is favored by seven."


@pytest.mark.asyncio
async def test_full_host_policy_replaces_empty_protocol_elements_without_leaking_tags() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_empty_tags",
                status="completed",
                summary="<voice></voice><detail>   </detail>",
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == "The full result is on screen."
    assert projection.events_after(0)[-1].data["text"] == ("No displayable result was returned.")


@pytest.mark.asyncio
async def test_full_host_policy_reuses_valid_voice_when_detail_is_empty() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_empty_detail",
                status="completed",
                summary="<voice>Indiana is favored by seven.</voice><detail></detail>",
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == "Indiana is favored by seven."
    assert projection.events_after(0)[-1].data["text"] == "Indiana is favored by seven."


@pytest.mark.asyncio
async def test_full_host_policy_strips_nested_protocol_tags_from_valid_visual_detail() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)

    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_nested_tags",
                status="completed",
                summary=(
                    "<voice>Short answer.</voice><detail>Detail one</detail> and more</detail>"
                ),
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text == "Detail one and more"
    assert projection.events_after(0)[-1].data["text"] == "Detail one and more"


@pytest.mark.asyncio
async def test_full_host_policy_redacts_private_run_handles_before_projection() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    directive = await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_ci",
                status="completed",
                summary="CI workflow run_20260731ab finished green.",
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    assert directive.text is not None
    assert "run_20260731ab" not in directive.text
    assert projection.events_after(0)[-1].data["text"] == (
        "CI workflow private reference finished green."
    )


@pytest.mark.asyncio
async def test_full_host_policy_clamps_result_after_private_handle_redaction() -> None:
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    private_prefix = "run_a1b2c3d4 "
    summary = private_prefix + ("x" * (1024 - len(private_prefix)))

    await policy.decide(
        UpdatePolicyInput(
            sequence=1,
            completion=TaskTerminalOutcome(
                task_id="task_ci_bound",
                status="completed",
                summary=summary,
            ),
            context=ConversationContextStore().snapshot(),
        )
    )

    projected = projection.events_after(0)[-1].data["text"]
    assert type(projected) is str
    assert len(projected) <= 1024
    assert "run_a1b2c3d4" not in projected
