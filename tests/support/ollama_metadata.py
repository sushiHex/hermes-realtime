"""Synthetic supported metadata through the production client's normal verifier."""

import json

from hermes_realtime.providers.ollama_models import OllamaModelClient


def supported_ollama_row(model: str) -> dict[str, object]:
    """One installed local GGUF tags row for ``model``."""

    return {"name": model, "model": model, "digest": "a" * 64, "details": {"format": "gguf"}}


def supported_ollama_tags(model: str) -> bytes:
    """A tags payload with exactly one row, for ``model``."""

    return json.dumps({"models": [supported_ollama_row(model)]}).encode()


def supported_ollama_metadata(model: str = "test") -> OllamaModelClient:
    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        assert body is None
        if endpoint == "version":
            return b'{"version":"0.34.4"}'
        assert endpoint == "tags"
        return supported_ollama_tags(model)

    return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
