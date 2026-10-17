from __future__ import annotations

import asyncio
import json

import pytest

from hermes_realtime.providers.ollama_models import OllamaModelClient, OllamaModelMetadata

pytestmark = pytest.mark.asyncio

GGUF = {"format": "gguf"}


class CatalogBackend:
    def __init__(self):
        self.rows = [
            {"name": "small", "model": "small", "digest": "a" * 64, "details": GGUF},
            {"name": "embedding", "digest": "b" * 64, "details": GGUF},
            {"name": "cloud", "digest": "c" * 64, "details": GGUF, "remote_host": "remote"},
        ]
        self.requests = []
        self.active = self.peak = 0
        self.changed = False
        self.block = False

    async def read(self, endpoint, body):
        self.requests.append((endpoint, body))
        if endpoint == "tags":
            rows = [dict(row) for row in self.rows]
            if self.changed and len(self.requests) > 1:
                rows[0]["digest"] = "d" * 64
            return json.dumps({"models": rows}).encode()
        assert endpoint == "show"
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01 if not self.block else 0.1)
            return json.dumps(
                {
                    "capabilities": ["embedding"]
                    if body["model"] == "embedding"
                    else ["completion"],
                    "details": GGUF,
                    "model_info": {
                        "general.architecture": "generic",
                        "generic.context_length": 4096,
                    },
                }
            ).encode()
        finally:
            self.active -= 1

    def client(self):
        return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=self.read)


async def test_catalog_is_local_completion_only_without_loading():
    backend = CatalogBackend()
    assert await backend.client().installed_models() == (
        OllamaModelMetadata("small", "a" * 64, 4096),
    )
    assert all(path in {"tags", "show"} for path, _ in backend.requests)
    assert [body["model"] for path, body in backend.requests if path == "show"] == [
        "small",
        "embedding",
    ]


async def test_catalog_identity_snapshot_requires_equality():
    backend = CatalogBackend()
    backend.changed = True
    with pytest.raises(ValueError, match="changed"):
        await backend.client().installed_models()


async def test_catalog_bounds_concurrency():
    backend = CatalogBackend()
    backend.rows = [
        {"name": str(index), "digest": "a" * 64, "details": GGUF} for index in range(12)
    ]
    assert len(await backend.client().installed_models()) == 12
    assert backend.peak == 4


async def test_catalog_has_one_overall_deadline(monkeypatch, capsys):
    import hermes_realtime.providers.ollama_models as module

    monkeypatch.setattr(module, "_CATALOG_TIMEOUT_SECONDS", 0.02)
    backend = CatalogBackend()
    backend.block = True
    with pytest.raises(TimeoutError):
        await backend.client().installed_models()
    assert backend.active == 0
    assert '[ollama-catalog] {"version":1,"category":"deadline"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "bad\nname"),
        ("name", "bad\x80name"),
        ("model", "conflicting"),
        ("digest", "invalid"),
        ("remote_host", None),
    ],
)
async def test_catalog_refuses_malformed_identity(field, value):
    backend = CatalogBackend()
    backend.rows[0][field] = value
    if field == "name":
        backend.rows[0]["model"] = value
    with pytest.raises(ValueError):
        await backend.client().installed_models()


async def test_catalog_refuses_duplicate_names():
    backend = CatalogBackend()
    backend.rows.append(dict(backend.rows[0]))
    with pytest.raises(ValueError, match="ambiguous"):
        await backend.client().installed_models()


@pytest.mark.parametrize("capabilities", [None, [False], ["x" * 65], ["completion"] * 33])
async def test_catalog_refuses_malformed_capabilities(capabilities):
    backend = CatalogBackend()
    original = backend.read

    async def read(endpoint, body):
        payload = json.loads(await original(endpoint, body))
        if endpoint == "show":
            payload["capabilities"] = capabilities
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ExceptionGroup) as caught:
        await client.installed_models()
    assert all(isinstance(error, ValueError) for error in caught.value.exceptions)


async def test_catalog_refused_capacity_remains_explicitly_usable(capsys):
    backend = CatalogBackend()
    original = backend.read

    async def read(endpoint, body):
        payload = json.loads(await original(endpoint, body))
        if endpoint == "show":
            payload["model_info"] = False
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    assert await client.installed_models() == (OllamaModelMetadata("small", "a" * 64, None),)
    assert '[ollama-catalog] {"version":1,"category":"capacity_refused"' in capsys.readouterr().out


async def test_catalog_show_cannot_switch_to_remote_backing():
    backend = CatalogBackend()
    original = backend.read

    async def read(endpoint, body):
        payload = json.loads(await original(endpoint, body))
        if endpoint == "show":
            payload["remote_model"] = "remote"
        return json.dumps(payload).encode()

    client = OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
    with pytest.raises(ExceptionGroup) as caught:
        await client.installed_models()
    assert all(isinstance(error, ValueError) for error in caught.value.exceptions)
