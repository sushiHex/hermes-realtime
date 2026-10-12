"""Explicit transport denial for synthetic Ollama streaming tests."""

import pytest

from hermes_realtime.providers.ollama_models import OllamaModelClient


@pytest.fixture
def deny_uninjected_ollama_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def denied(self: OllamaModelClient, endpoint: str, body: object) -> bytes:
        raise AssertionError("synthetic Ollama tests must inject a metadata reader")

    monkeypatch.setattr(OllamaModelClient, "_read_request", denied)
