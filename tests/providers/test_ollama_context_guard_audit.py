from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_realtime.client.session import BrowserModelConfiguration
from hermes_realtime.host_launcher import build_local_host_launcher
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import (
    OllamaContextConfiguration,
    OllamaContextObservation,
    OllamaModelClient,
    OllamaModelMetadata,
)

pytestmark = pytest.mark.asyncio
MODEL = OllamaModelMetadata("synthetic:latest", "a" * 64, 1_048_576)
CONFIGURATION = OllamaContextConfiguration(MODEL, 4096)
OBSERVATION = OllamaContextObservation(CONFIGURATION, 4096)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", ""),
        ("model", " " * 3),
        ("model", "x" * 257),
        ("model", False),
        ("digest", "invalid"),
        ("digest", 123),
        ("advertised_context_tokens", True),
    ],
)
async def test_metadata_constructor_cannot_admit_forged_identity_or_capacity(field, value):
    with pytest.raises(ValueError):
        replace(MODEL, **{field: value})


async def test_setting_constructor_requires_validated_metadata():
    with pytest.raises(TypeError):
        OllamaContextConfiguration(
            SimpleNamespace(
                model=MODEL.model,
                digest=MODEL.digest,
                advertised_context_tokens=MODEL.advertised_context_tokens,
            ),
            4096,
        )


async def test_observation_constructor_requires_validated_configuration():
    with pytest.raises(TypeError):
        OllamaContextObservation(SimpleNamespace(model=MODEL, requested_context_tokens=4096), 4096)


async def test_observation_constructor_refuses_boolean_allocation():
    with pytest.raises(ValueError):
        OllamaContextObservation(CONFIGURATION, True)


async def test_origin_constructor_refuses_non_string():
    with pytest.raises(TypeError):
        OllamaModelClient(base_url=123)


async def test_origin_constructor_refuses_fragment():
    with pytest.raises(ValueError):
        OllamaModelClient(base_url="http://localhost:11434#private")


async def test_malformed_model_rows_make_optional_allocation_unknown():
    async def read(endpoint, body):
        if endpoint == "version":
            payload = {"version": "0.34.4"}
        elif endpoint == "show":
            payload = {"capabilities": ["completion"], "model_info": {}}
        elif endpoint == "ps":
            payload = {"models": [False]}
        else:
            payload = {
                "models": [{"name": MODEL.model, "model": MODEL.model, "digest": MODEL.digest}]
            }
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://localhost:11434", read_request=read)
    inference = OllamaStreamingInference(base_url="http://localhost:11434", model=MODEL.model)
    await inference.configure_context(4096, client=client)
    await inference.observe_context()
    assert inference.context_public_data()["effectiveContextTokens"] is None
    await inference.close()


@pytest.mark.parametrize(
    "info",
    [
        False,
        {"general.architecture": False, "False.context_length": 4096},
        {"general.architecture": "", ".context_length": 4096},
        {"general.architecture": "x" * 129, f"{'x' * 129}.context_length": 4096},
    ],
)
async def test_capacity_stays_unknown_without_bounded_architecture_metadata(info):
    async def read(endpoint, body):
        if endpoint == "show":
            payload = {"capabilities": ["completion"], "model_info": info}
        else:
            payload = {
                "models": [{"name": MODEL.model, "model": MODEL.model, "digest": MODEL.digest}]
            }
        return json.dumps(payload).encode()

    metadata = await OllamaModelClient(
        base_url="http://localhost:11434", read_request=read
    ).discover(MODEL.model)
    assert metadata.advertised_context_tokens is None
    assert OllamaContextConfiguration(metadata, 4096).requested_context_tokens == 4096


async def test_closed_adapter_refuses_context_configuration_before_metadata_io():
    called = False

    async def read(endpoint, body):
        nonlocal called
        called = True
        if endpoint == "version":
            payload = {"version": "0.34.4"}
        elif endpoint == "show":
            payload = {"capabilities": ["completion"], "model_info": {}}
        else:
            payload = {
                "models": [{"name": MODEL.model, "model": MODEL.model, "digest": MODEL.digest}]
            }
        return json.dumps(payload).encode()

    inference = OllamaStreamingInference(base_url="http://localhost:11434", model=MODEL.model)
    await inference.close()
    with pytest.raises(RuntimeError, match="idle open"):
        await inference.configure_context(
            4096, client=OllamaModelClient(base_url="http://localhost:11434", read_request=read)
        )
    assert not called


async def test_unconfigured_observation_preserves_the_default_path():
    inference = OllamaStreamingInference(base_url="http://localhost:11434", model=MODEL.model)
    await inference.observe_context()
    with pytest.raises(RuntimeError, match="not been configured"):
        inference.context_observation()
    await inference.close()


def browser_configuration(**kwargs):
    return BrowserModelConfiguration(
        authentication="local",
        provider="ollama",
        transport="local-http",
        model=MODEL.model,
        effort=None,
        context_window_tokens=None,
        reports_token_usage=False,
        **kwargs,
    )


async def test_browser_callback_requires_callable():
    with pytest.raises(TypeError):
        browser_configuration(local_context=123)


async def test_browser_callback_requires_validated_observation():
    class UnverifiedObservation:
        configuration = CONFIGURATION

        def public_data(self):
            return {**OBSERVATION.public_data(), "modelDigest": "b" * 64}

    with pytest.raises(TypeError):
        browser_configuration(local_context=UnverifiedObservation).public_data()


async def test_malformed_empty_models_cannot_be_reported_as_no_allocation():
    async def read(endpoint, body):
        return b'{"models":{}}'

    with pytest.raises(ValueError):
        await OllamaModelClient(
            base_url="http://localhost:11434", read_request=read
        ).effective_context(MODEL)


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "other"),
        ("authentication", "subscription"),
        ("transport", "other"),
        ("effort", "low"),
        ("reports_token_usage", True),
        ("context_window_tokens", 4096),
    ],
)
async def test_browser_refuses_local_budget_bound_to_an_incompatible_session(field, value):
    configuration = browser_configuration(local_context=lambda: OBSERVATION)
    with pytest.raises(ValueError, match="identity"):
        replace(configuration, **{field: value}).public_data()


async def test_host_refuses_ollama_context_configuration_for_another_provider(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("provider construction must follow context validation")

    monkeypatch.setattr("hermes_realtime.host_launcher._build_streaming_inference", forbidden)
    with pytest.raises(ValueError, match="ollama-num-ctx"):
        build_local_host_launcher(
            hermes_api_bearer="b" * 32,
            inference_provider="codex",
            ollama_num_ctx=4096,
            allow_unsandboxed_tasks=True,
        )


async def test_host_refuses_non_integer_context_before_provider_construction(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("provider construction must follow context validation")

    monkeypatch.setattr("hermes_realtime.host_launcher._build_streaming_inference", forbidden)
    with pytest.raises(TypeError):
        build_local_host_launcher(
            hermes_api_bearer="b" * 32, ollama_num_ctx=4096.5, allow_unsandboxed_tasks=True
        )


async def test_metadata_http_failure_never_becomes_an_installed_identity(monkeypatch):
    import aiohttp

    import hermes_realtime.providers.ollama_models as module

    class Response:
        def __init__(self):
            self.content = self
            self.sent = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            raise aiohttp.ClientResponseError(request_info=None, history=(), status=500)

        async def read(self, size):
            if self.sent:
                return b""
            self.sent = True
            return json.dumps({"models": [{"name": MODEL.model, "digest": MODEL.digest}]}).encode()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def request(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(module.aiohttp, "ClientSession", Client)
    with pytest.raises(aiohttp.ClientResponseError):
        await OllamaModelClient(base_url="http://localhost:11434").identity(MODEL.model)
