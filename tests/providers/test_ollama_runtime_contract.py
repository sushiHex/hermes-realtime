from __future__ import annotations

import asyncio
import io
import json

import pytest

import hermes_realtime.launcher as legacy
from hermes_realtime.conversation import ConversationContextSnapshot, ConversationMessage
from hermes_realtime.host_launcher import _build_streaming_inference
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import OllamaModelClient
from tests.support.ollama_metadata import supported_ollama_row

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("deny_uninjected_ollama_metadata")]
ORIGIN = "http://127.0.0.1:11434"
SNAPSHOT = ConversationContextSnapshot(
    revision=0, messages=(ConversationMessage("user", "Synthetic question."),), active_tasks=()
)


async def consume(adapter, turn="turn_one"):
    return [part async for part in adapter.stream(SNAPSHOT, turn_id=turn)]


def adapter_for(path, monkeypatch):
    if path == "legacy":

        class Synthesizer:
            async def close(self):
                pass

        monkeypatch.setattr(legacy, "EdgeTtsSynthesizer", lambda **kwargs: Synthesizer())
        monkeypatch.setattr(legacy, "FasterWhisperTranscriber", lambda **kwargs: object())
        monkeypatch.setattr(legacy, "SileroSpeechPresenceVerifier", lambda **kwargs: object())
        return legacy.build_local_conversation_launcher()._providers[-1]
    if path == "host":
        return _build_streaming_inference(
            inference_provider="ollama",
            ollama_base_url=ORIGIN,
            ollama_model="test:latest",
            codex_model="unused",
            codex_effort="medium",
            codex_executable=None,
        )
    return OllamaStreamingInference(base_url=ORIGIN, model="test:latest")


@pytest.mark.parametrize("path", ["direct", "host", "legacy"])
@pytest.mark.parametrize("failure", ["unsupported", "unavailable"])
async def test_every_ollama_path_requires_runtime_proof_before_opening(path, failure, monkeypatch):
    requests = []
    openings = []

    async def metadata(self, endpoint, body):
        requests.append((endpoint, body))
        if failure == "unavailable":
            raise TimeoutError("synthetic private detail")
        return b'{"version":"0.1.0"}'

    monkeypatch.setattr(OllamaModelClient, "_read_request", metadata)
    adapter = adapter_for(path, monkeypatch)
    adapter._open_request = lambda request, timeout: (
        openings.append(request) or io.BytesIO(b'{"message":{"content":"Ready."},"done":true}\n')
    )
    try:
        with pytest.raises(ValueError if failure == "unsupported" else TimeoutError):
            await consume(adapter)
        assert requests == [("version", None)]
        assert openings == []
        assert adapter._inference_started is False
    finally:
        await adapter.close()


@pytest.mark.parametrize("configured", [False, True], ids=["default", "explicit"])
async def test_runtime_proof_is_cached_across_turns_and_explicit_configuration(configured, capsys):
    requests = []
    bodies = []

    async def read(endpoint, body):
        requests.append(endpoint)
        if endpoint == "version":
            payload = {"version": "0.34.4"}
        elif endpoint == "show":
            payload = {
                "capabilities": ["completion"],
                "details": {"format": "gguf"},
                "model_info": {},
            }
        else:
            payload = {"models": [supported_ollama_row("test:latest")]}
        return json.dumps(payload).encode()

    def opening(request, timeout):
        bodies.append(json.loads(request.data))
        return io.BytesIO(b'{"message":{"content":"Ready."},"done":true}\n')

    client = OllamaModelClient(base_url=ORIGIN, read_request=read)
    adapter = OllamaStreamingInference(
        base_url=ORIGIN, model="test:latest", open_request=opening, metadata_client=client
    )
    try:
        if configured:
            await adapter.configure_context(4096, client=client)
        assert await consume(adapter) == ["Ready."]
        assert await consume(adapter, "turn_two") == ["Ready."]
        assert requests == (
            ["version", "tags", "tags", "show", "tags"] if configured else ["version", "tags"]
        )
        assert len(bodies) == 2
        assert all(body["truncate"] is False and body["shift"] is False for body in bodies)
        categories = [
            json.loads(line.removeprefix("[ollama-runtime-contract] "))
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("[ollama-runtime-contract] ")
        ]
        assert categories == [
            {"version": 1, "category": category}
            for category in (
                ["verified", "cached", "cached"] if configured else ["verified", "cached"]
            )
        ]
    finally:
        await adapter.close()


async def test_concurrent_first_turns_share_one_successful_runtime_proof():
    entered = asyncio.Event()
    release = asyncio.Event()
    requests = []
    openings = []

    async def read(endpoint, body):
        requests.append(endpoint)
        entered.set()
        await release.wait()
        if endpoint == "tags":
            return json.dumps({"models": [supported_ollama_row("test")]}).encode()
        return b'{"version":"0.34.4"}'

    def opening(request, timeout):
        openings.append(request)
        return io.BytesIO(b'{"message":{"content":"Ready."},"done":true}\n')

    adapter = OllamaStreamingInference(
        base_url=ORIGIN,
        model="test",
        open_request=opening,
        metadata_client=OllamaModelClient(base_url=ORIGIN, read_request=read),
    )
    first = asyncio.create_task(consume(adapter))
    await entered.wait()
    second = asyncio.create_task(consume(adapter, "turn_two"))
    await asyncio.sleep(0)
    try:
        assert openings == []
        release.set()
        assert await asyncio.gather(first, second) == [["Ready."], ["Ready."]]
        assert requests == ["version", "tags"]
        assert len(openings) == 2
    finally:
        release.set()
        await adapter.close()


async def test_cancelled_runtime_proof_does_not_cache_or_start_inference():
    entered = asyncio.Event()
    requests = []
    openings = []

    async def read(endpoint, body):
        requests.append(endpoint)
        if len(requests) == 1:
            entered.set()
            await asyncio.Event().wait()
        if endpoint == "tags":
            return json.dumps({"models": [supported_ollama_row("test")]}).encode()
        return b'{"version":"0.34.4"}'

    def opening(request, timeout):
        openings.append(request)
        return io.BytesIO(b'{"message":{"content":"Ready."},"done":true}\n')

    adapter = OllamaStreamingInference(
        base_url=ORIGIN,
        model="test",
        open_request=opening,
        metadata_client=OllamaModelClient(base_url=ORIGIN, read_request=read),
    )
    attempt = asyncio.create_task(consume(adapter))
    await entered.wait()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    try:
        assert not adapter._inference_started
        assert openings == []
        assert await consume(adapter) == ["Ready."]
        assert requests == ["version", "version", "tags"]
        assert len(openings) == 1
    finally:
        await adapter.close()


@pytest.mark.parametrize("invalid", ["type", "port", "host"])
async def test_runtime_proof_refuses_foreign_or_forged_metadata_client(invalid, capsys):
    requests = []
    openings = []

    async def read(endpoint, body):
        requests.append(endpoint)
        return b'{"version":"0.34.4"}'

    client = (
        object()
        if invalid == "type"
        else OllamaModelClient(
            base_url="http://127.0.0.1:12345" if invalid == "port" else "http://localhost:11434",
            read_request=read,
        )
    )
    adapter = OllamaStreamingInference(
        base_url=ORIGIN,
        model="test",
        metadata_client=client,
        open_request=lambda request, timeout: openings.append(request) or io.BytesIO(),
    )
    try:
        with pytest.raises(TypeError if invalid == "type" else ValueError):
            await consume(adapter)
        assert requests == [] and openings == []
        assert adapter._inference_started is False
        assert capsys.readouterr().out.splitlines() == [
            '[ollama-runtime-contract] {"version":1,"category":"contract_refused"}'
        ]
    finally:
        await adapter.close()


async def test_successful_runtime_proof_cannot_be_reused_with_another_client():
    requests = []

    async def read(endpoint, body):
        requests.append(endpoint)
        if endpoint == "version":
            return b'{"version":"0.34.4"}'
        if endpoint == "show":
            raise ValueError("synthetic discovery refusal")
        return json.dumps({"models": [supported_ollama_row("test")]}).encode()

    first = OllamaModelClient(base_url=ORIGIN, read_request=read)
    second = OllamaModelClient(base_url=ORIGIN, read_request=read)
    adapter = OllamaStreamingInference(base_url=ORIGIN, model="test", metadata_client=first)
    try:
        with pytest.raises(ValueError, match="discovery refusal"):
            await adapter.configure_context(4096, client=first)
        before = requests.copy()
        with pytest.raises(ValueError, match="already bound"):
            await adapter.configure_context(4096, client=second)
        assert requests == before
    finally:
        await adapter.close()
