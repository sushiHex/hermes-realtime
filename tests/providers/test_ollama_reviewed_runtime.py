"""The context contract admits only source-reviewed runtimes, weights and runners (#282)."""

from __future__ import annotations

import json

import pytest

from hermes_realtime.providers.ollama_models import OllamaModelClient, OllamaModelMetadata

pytestmark = pytest.mark.asyncio

DIGEST = "a" * 64


def _version_client(version: str) -> OllamaModelClient:
    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        assert (endpoint, body) == ("version", None)
        return json.dumps({"version": version}).encode()

    return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)


@pytest.mark.parametrize("version", ["0.34.4", "0.40.4"])
async def test_context_contract_admits_each_source_reviewed_runtime(version: str) -> None:
    await _version_client(version).require_context_contract()


@pytest.mark.parametrize(
    "version", ["0.34.3", "0.34.5", "0.40.3", "0.40.5", "0.41.0", "0.40.4-rc1", " 0.40.4"],
)
async def test_context_contract_refuses_every_other_runtime(version: str) -> None:
    with pytest.raises(ValueError, match="source-reviewed"):
        await _version_client(version).require_context_contract()


def _backend(*, tags: dict[str, object], show: dict[str, object], ps: dict[str, object]):
    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        row: dict[str, object] = {"name": "test:latest", "model": "test:latest", "digest": DIGEST}
        if endpoint == "show":
            return json.dumps(
                {
                    "model_info": {
                        "general.architecture": "synthetic",
                        "synthetic.context_length": 131072,
                    },
                    "capabilities": ["completion"],
                    **show,
                }
            ).encode()
        if endpoint == "ps":
            return json.dumps({"models": [{**row, "context_length": 4096, **ps}]}).encode()
        assert endpoint == "tags"
        return json.dumps({"models": [{**row, **tags}]}).encode()

    return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)


async def test_reviewed_gguf_weights_and_llama_runners_are_admitted() -> None:
    for runner in ("llamacpp", "ggml"):
        client = _backend(
            tags={"details": {"format": "gguf"}},
            show={"details": {"format": "gguf"}},
            ps={"details": {"format": "gguf"}, "runner": runner},
        )
        model = await client.discover("test:latest")
        assert await client.effective_context(model) == 4096


@pytest.mark.parametrize("endpoint", ["tags", "show"])
@pytest.mark.parametrize(
    "details",
    [{"format": "safetensors"}, {"format": None}, {"format": ["gguf"]}, ["gguf"], "gguf"],
)
async def test_unreviewed_weight_format_refuses_before_any_prompt(
    endpoint: str, details: object,
) -> None:
    fields = {"details": details}
    client = _backend(
        tags=fields if endpoint == "tags" else {},
        show=fields if endpoint == "show" else {},
        ps={},
    )
    with pytest.raises(ValueError, match="GGUF"):
        await client.discover("test:latest")


@pytest.mark.parametrize("runner", ["mlx", "ollama", "", None, ["llamacpp"]])
async def test_unreviewed_runner_allocation_refuses(runner: object) -> None:
    client = _backend(tags={}, show={}, ps={"runner": runner})
    model = await client.discover("test:latest")
    with pytest.raises(ValueError, match="runner"):
        await client.effective_context(model)


async def test_catalog_excludes_unreviewed_weights_without_refusing_the_rest() -> None:
    rows = [
        {"name": "small", "model": "small", "digest": "a" * 64, "details": {"format": "gguf"}},
        {"name": "mlx", "model": "mlx", "digest": "b" * 64, "details": {"format": "safetensors"}},
    ]
    shown: list[object] = []

    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        if endpoint == "tags":
            return json.dumps({"models": rows}).encode()
        assert endpoint == "show" and body is not None
        shown.append(body["model"])
        return json.dumps(
            {
                "capabilities": ["completion"],
                "model_info": {"general.architecture": "g", "g.context_length": 4096},
                "details": {"format": "gguf"},
            }
        ).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert await client.installed_models() == (OllamaModelMetadata("small", "a" * 64, 4096),)
    assert shown == ["small"]
