from __future__ import annotations

import pytest

from hermes_realtime.providers.current_facts import CurrentFactEvidence
from hermes_realtime.providers.knowledge_backends import (
    DdgsKnowledgeBackend,
    ExaInstantKnowledgeBackend,
    OpenAIHostedSearchBackend,
    TavilyKnowledgeBackend,
)


class FakeJsonTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, object], float]] = []

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_seconds: float,
    ) -> object:
        self.calls.append((url, headers, body, timeout_seconds))
        return self.response

    async def close(self) -> None:
        return None


class FakeLookup:
    def __init__(self) -> None:
        self.closed = False

    async def lookup(self, query: str) -> CurrentFactEvidence:
        return CurrentFactEvidence(
            query=query,
            retrieved_date="2026-08-02",
            sources=(),
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ddgs_baseline_uses_same_normalized_boundary_and_owns_lookup() -> None:
    lookup = FakeLookup()
    backend = DdgsKnowledgeBackend(lookup=lookup)

    evidence = await backend.lookup("current fact")
    await backend.close()

    assert evidence.query == "current fact"
    assert backend.authentication_mode == "none"
    assert backend.cost_mode == "unmetered_third_party"
    assert lookup.closed is True


@pytest.mark.asyncio
async def test_tavily_maps_raw_content_without_provider_objects() -> None:
    transport = FakeJsonTransport(
        {
            "results": [
                {
                    "title": "LiveKit docs",
                    "url": "https://docs.livekit.io/audio",
                    "content": "AudioStream API reference.",
                    "raw_content": (
                        "AudioStream accepts a capacity argument. "
                        "A capacity of zero means the queue is unbounded."
                    ),
                }
            ]
        }
    )
    backend = TavilyKnowledgeBackend(
        api_key="tavily-secret",
        search_depth="ultra-fast",
        transport=transport,
    )

    evidence = await backend.lookup("What does LiveKit AudioStream capacity zero mean?")

    assert evidence.backend == "tavily-ultra-fast"
    assert evidence.sources[0].passages
    assert "unbounded" in evidence.sources[0].passages[0].text
    assert backend.authentication_mode == "api_key"
    assert backend.cost_mode == "provider_metered"
    _, headers, body, _ = transport.calls[0]
    assert headers["Authorization"] == "Bearer tavily-secret"
    assert body["search_depth"] == "ultra-fast"


@pytest.mark.asyncio
async def test_exa_maps_highlights_and_scores_to_plain_passages() -> None:
    transport = FakeJsonTransport(
        {
            "results": [
                {
                    "title": "Python release",
                    "url": "https://python.org/downloads/",
                    "highlights": ["Python 3.14 is the latest stable release."],
                    "highlightScores": [0.91],
                }
            ]
        }
    )
    backend = ExaInstantKnowledgeBackend(api_key="exa-secret", transport=transport)

    evidence = await backend.lookup("latest stable Python release")

    passage = evidence.sources[0].passages[0]
    assert evidence.backend == "exa-instant"
    assert passage.score == 0.91
    assert type(evidence.tool_result()["sources"][0]) is dict  # type: ignore[index]
    _, headers, body, _ = transport.calls[0]
    assert headers["x-api-key"] == "exa-secret"
    assert body["type"] == "instant"


@pytest.mark.asyncio
async def test_openai_arm_forces_hosted_search_and_requires_explicit_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAIHostedSearchBackend(api_key="", transport=FakeJsonTransport({}))
    transport = FakeJsonTransport(
        {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"type": "search", "query": "current value"},
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The current value is 42.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "Official source",
                                    "url": "https://example.com/current",
                                    "start_index": 0,
                                    "end_index": 24,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    backend = OpenAIHostedSearchBackend(api_key="openai-secret", transport=transport)

    evidence = await backend.lookup("What is the current value?")

    assert evidence.backend == "openai-hosted-search"
    _, headers, body, _ = transport.calls[0]
    assert headers["Authorization"] == "Bearer openai-secret"
    assert body["tools"] == [{"type": "web_search"}]
    assert body["tool_choice"] == {"type": "web_search"}
    assert body["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_malformed_experimental_response_fails_closed() -> None:
    backend = TavilyKnowledgeBackend(
        api_key="secret",
        search_depth="fast",
        transport=FakeJsonTransport({"results": "not-a-list"}),
    )

    evidence = await backend.lookup("current fact")

    assert evidence.sources == ()
    assert evidence.error == "tavily-fast returned an invalid response."
