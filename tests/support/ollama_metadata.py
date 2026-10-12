"""Synthetic supported metadata through the production client's normal verifier."""

from hermes_realtime.providers.ollama_models import OllamaModelClient


def supported_ollama_metadata() -> OllamaModelClient:
    async def read(endpoint: str, body: dict[str, object] | None) -> bytes:
        assert (endpoint, body) == ("version", None)
        return b'{"version":"0.34.4"}'

    return OllamaModelClient(base_url="http://127.0.0.1:11434", read_request=read)
