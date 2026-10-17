"""The context contract admits only source-reviewed runtimes and weights (#282)."""

from __future__ import annotations

import json

import pytest

from hermes_realtime.conversation import ConversationInferenceRequest, ConversationMessage
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.providers.ollama_models import OllamaModelClient, OllamaModelMetadata

pytestmark = pytest.mark.asyncio

DIGEST = "a" * 64
MODEL = "test:latest"


def _contract_client(
    version: str, details: object = None, *, requests: list[str] | None = None,
) -> OllamaModelClient:
    seen = requests if requests is not None else []

    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        seen.append(endpoint)
        assert body is None
        if endpoint == "version":
            return json.dumps({"version": version}).encode()
        assert endpoint == "tags"
        row: dict[str, object] = {"name": MODEL, "model": MODEL, "digest": DIGEST}
        if details is not None:
            row["details"] = details
        return json.dumps({"models": [row]}).encode()

    return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)


@pytest.mark.parametrize("version", ["0.34.4", "0.40.4"])
async def test_context_contract_admits_each_source_reviewed_runtime(version: str) -> None:
    requests: list[str] = []
    await _contract_client(version, {"format": "gguf"}, requests=requests).require_context_contract(
        MODEL
    )
    assert requests == ["version", "tags"]


@pytest.mark.parametrize(
    "version",
    ["0.34.3", "0.34.5", "0.40.3", "0.40.5", "0.41.0", "0.40.4-rc1", " 0.40.4", None, 404],
)
async def test_context_contract_refuses_every_other_runtime_before_reading_models(
    version: object,
) -> None:
    requests: list[str] = []

    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        requests.append(endpoint)
        return json.dumps({"version": version}).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ValueError, match="source-reviewed"):
        await client.require_context_contract(MODEL)
    assert requests == ["version"]


_UNREVIEWED_DETAILS = [
    None,  # no details at all
    {},
    {"format": ""},
    {"format": "safetensors"},
    {"format": "GGUF"},
    {"format": ["gguf"]},
    ["gguf"],
    "gguf",
]


@pytest.mark.parametrize("details", _UNREVIEWED_DETAILS)
async def test_context_contract_refuses_unreviewed_or_undeclared_weights(details: object) -> None:
    with pytest.raises(ValueError, match="GGUF"):
        await _contract_client("0.40.4", details).require_context_contract(MODEL)


def _request() -> ConversationInferenceRequest:
    return ConversationInferenceRequest(
        revision=1,
        messages=(ConversationMessage(role="user", text="Hello."),),
        active_tasks=(),
        updates=(),
    )


async def test_default_path_refuses_unreviewed_weights_before_any_prompt() -> None:
    # No context size is configured: the contract alone must refuse before /api/chat.
    requests: list[str] = []
    inference = OllamaStreamingInference(
        base_url="http://127.0.0.1:11434",
        model=MODEL,
        metadata_client=_contract_client(
            "0.40.4", {"format": "safetensors"}, requests=requests,
        ),
    )
    try:
        with pytest.raises(ValueError, match="GGUF"):
            async for _segment in inference.stream(
                _request(), turn_id="turn_mlx",
            ):
                pass
        assert requests == ["version", "tags"]
    finally:
        await inference.close()


def _metadata(*, tags: dict[str, object], show: dict[str, object]) -> OllamaModelClient:
    gguf = {"details": {"format": "gguf"}}

    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        row: dict[str, object] = {"name": MODEL, "model": MODEL, "digest": DIGEST, **gguf}
        if endpoint == "show":
            return json.dumps(
                {
                    "model_info": {
                        "general.architecture": "synthetic",
                        "synthetic.context_length": 131072,
                    },
                    "capabilities": ["completion"],
                    **gguf,
                    **show,
                }
            ).encode()
        assert endpoint == "tags"
        return json.dumps({"models": [{**row, **tags}]}).encode()

    return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)


async def test_discovery_admits_declared_gguf_weights() -> None:
    model = await _metadata(tags={}, show={}).discover(MODEL)
    assert (model.model, model.digest) == (MODEL, DIGEST)


@pytest.mark.parametrize("endpoint", ["tags", "show"])
@pytest.mark.parametrize("details", [{"format": "safetensors"}, {}])
async def test_discovery_refuses_unreviewed_weights(endpoint: str, details: object) -> None:
    fields = {"details": details}
    client = _metadata(
        tags=fields if endpoint == "tags" else {},
        show=fields if endpoint == "show" else {},
    )
    with pytest.raises(ValueError, match="GGUF"):
        await client.discover(MODEL)


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
