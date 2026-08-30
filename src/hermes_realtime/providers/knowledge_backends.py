"""Experimental provider-neutral foreground knowledge backends.

These adapters normalize provider JSON into ``CurrentFactEvidence`` and are not
selected by the production launcher. Credentials are accepted only explicitly.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Protocol

import aiohttp

from hermes_realtime import __version__

from .current_facts import (
    CurrentFactEvidence,
    CurrentFactLookup,
    CurrentFactSource,
    DdgsCurrentFactLookup,
    EvidencePassage,
    _select_evidence_passages,
)

_MAX_QUERY_CHARS = 512
_MAX_RESULTS = 8
_MAX_SNIPPET_CHARS = 1_200
_MAX_PASSAGE_CHARS = 720


class JsonPostTransport(Protocol):
    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_seconds: float,
    ) -> object: ...

    async def close(self) -> None: ...


class AiohttpJsonTransport:
    """Small reusable raw-HTTP transport with one bounded connection pool."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict[str, object],
        timeout_seconds: float,
    ) -> object:
        session = self._session
        if session is None:
            session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=4),
                trust_env=False,
                headers={"User-Agent": f"Hermes-Realtime-Knowledge/{__version__}"},
            )
            self._session = session
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with session.post(
            url,
            headers=headers,
            json=body,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            return await response.json(content_type=None)

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()


class _ExperimentalKnowledgeBackend:
    authentication_mode = "api_key"
    cost_mode = "provider_metered"
    backend_name: str

    def __init__(
        self,
        *,
        api_key: str,
        transport: JsonPostTransport | None,
        timeout_seconds: float,
    ) -> None:
        if type(api_key) is not str or not api_key.strip() or len(api_key) > 512:
            raise ValueError("api_key must be an explicit non-empty string")
        if transport is not None and not callable(getattr(transport, "post_json", None)):
            raise TypeError("transport must provide post_json()")
        if (
            type(timeout_seconds) not in (int, float)
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or not 0.1 <= timeout_seconds <= 30.0
        ):
            raise ValueError("timeout_seconds must be between 0.1 and 30.0")
        self._api_key = api_key
        self._transport = transport or AiohttpJsonTransport()
        self._owns_transport = transport is None
        self._timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _query(query: str) -> str:
        if type(query) is not str:
            raise TypeError("query must be an exact string")
        normalized = " ".join(query.split())
        if not normalized or len(normalized) > _MAX_QUERY_CHARS:
            raise ValueError("query must contain between 1 and 512 characters")
        return normalized

    @staticmethod
    def _date() -> str:
        return datetime.now(UTC).date().isoformat()

    def _evidence(
        self,
        query: str,
        sources: tuple[CurrentFactSource, ...],
    ) -> CurrentFactEvidence:
        return CurrentFactEvidence(
            query=query,
            retrieved_date=self._date(),
            sources=sources,
            backend=self.backend_name,
        )

    def _invalid(self, query: str) -> CurrentFactEvidence:
        return CurrentFactEvidence(
            query=query,
            retrieved_date=self._date(),
            sources=(),
            error=f"{self.backend_name} returned an invalid response.",
            backend=self.backend_name,
        )

    async def close(self) -> None:
        if self._owns_transport:
            await self._transport.close()


class DdgsKnowledgeBackend:
    """Normalize the maintained DDGS baseline to the experimental arm boundary."""

    authentication_mode = "none"
    cost_mode = "unmetered_third_party"
    backend_name = "ddgs"

    def __init__(self, *, lookup: CurrentFactLookup | None = None) -> None:
        if lookup is not None and not callable(getattr(lookup, "lookup", None)):
            raise TypeError("lookup must provide lookup()")
        self._lookup = lookup or DdgsCurrentFactLookup(
            timeout_seconds=3.5,
            max_results=5,
            enrich_results=2,
        )

    async def lookup(self, query: str) -> CurrentFactEvidence:
        return await self._lookup.lookup(query)

    async def close(self) -> None:
        await self._lookup.close()


class TavilyKnowledgeBackend(_ExperimentalKnowledgeBackend):
    """Experimental Tavily Search adapter for ``fast`` and ``ultra-fast`` arms."""

    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key: str,
        search_depth: str,
        transport: JsonPostTransport | None = None,
        timeout_seconds: float = 3.5,
        max_results: int = 5,
    ) -> None:
        super().__init__(
            api_key=api_key,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
        if type(search_depth) is not str or search_depth not in {"fast", "ultra-fast"}:
            raise ValueError("search_depth must be 'fast' or 'ultra-fast'")
        if (
            type(max_results) is not int
            or isinstance(max_results, bool)
            or not 1 <= max_results <= 8
        ):
            raise ValueError("max_results must be between 1 and 8")
        self._search_depth = search_depth
        self._max_results = max_results
        self.backend_name = f"tavily-{search_depth}"

    async def lookup(self, query: str) -> CurrentFactEvidence:
        normalized = self._query(query)
        try:
            payload = await self._transport.post_json(
                self._ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                body={
                    "query": normalized,
                    "search_depth": self._search_depth,
                    "max_results": self._max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout_seconds=self._timeout_seconds,
            )
            sources = self._parse(payload, normalized)
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, KeyError):
            return self._invalid(normalized)
        return self._evidence(normalized, sources)

    def _parse(self, payload: object, query: str) -> tuple[CurrentFactSource, ...]:
        if type(payload) is not dict or type(payload.get("results")) is not list:
            raise ValueError("invalid Tavily response")
        sources: list[CurrentFactSource] = []
        for item in payload["results"][: self._max_results]:
            if type(item) is not dict:
                continue
            title = item.get("title")
            url = item.get("url")
            content = item.get("content")
            raw = item.get("raw_content")
            if not all(type(value) is str and value.strip() for value in (title, url, content)):
                continue
            evidence_text = raw if type(raw) is str and raw.strip() else content
            snippet = str(content).strip()[:_MAX_SNIPPET_CHARS]
            source_id = f"source_{len(sources) + 1}"
            passages = _select_evidence_passages(
                str(evidence_text),
                query,
                source_id=source_id,
            )
            sources.append(
                CurrentFactSource(
                    title=str(title).strip()[:240],
                    url=str(url).strip()[:2_048],
                    snippet=snippet,
                    backend=self.backend_name,
                    passages=passages,
                )
            )
        return tuple(sources)


class ExaInstantKnowledgeBackend(_ExperimentalKnowledgeBackend):
    """Experimental Exa ``instant`` search arm with normalized highlights."""

    _ENDPOINT = "https://api.exa.ai/search"
    backend_name = "exa-instant"

    def __init__(
        self,
        *,
        api_key: str,
        transport: JsonPostTransport | None = None,
        timeout_seconds: float = 3.5,
        max_results: int = 5,
    ) -> None:
        super().__init__(
            api_key=api_key,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
        if (
            type(max_results) is not int
            or isinstance(max_results, bool)
            or not 1 <= max_results <= 8
        ):
            raise ValueError("max_results must be between 1 and 8")
        self._max_results = max_results

    async def lookup(self, query: str) -> CurrentFactEvidence:
        normalized = self._query(query)
        try:
            payload = await self._transport.post_json(
                self._ENDPOINT,
                headers={
                    "x-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                body={
                    "query": normalized,
                    "type": "instant",
                    "numResults": self._max_results,
                    "contents": {"highlights": True},
                },
                timeout_seconds=self._timeout_seconds,
            )
            sources = self._parse(payload)
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, KeyError):
            return self._invalid(normalized)
        return self._evidence(normalized, sources)

    def _parse(self, payload: object) -> tuple[CurrentFactSource, ...]:
        if type(payload) is not dict or type(payload.get("results")) is not list:
            raise ValueError("invalid Exa response")
        sources: list[CurrentFactSource] = []
        for item in payload["results"][: self._max_results]:
            if type(item) is not dict:
                continue
            title = item.get("title")
            url = item.get("url")
            highlights = item.get("highlights")
            scores = item.get("highlightScores", [])
            if (
                type(title) is not str
                or not title.strip()
                or type(url) is not str
                or not url.strip()
                or type(highlights) is not list
            ):
                continue
            passages: list[EvidencePassage] = []
            for highlight_index, highlight in enumerate(highlights[:3]):
                if type(highlight) is not str or not highlight.strip():
                    continue
                score: float | None = None
                if type(scores) is list and highlight_index < len(scores):
                    candidate = scores[highlight_index]
                    if type(candidate) in (int, float) and not isinstance(candidate, bool):
                        numeric = float(candidate)
                        if math.isfinite(numeric) and 0 <= numeric <= 100:
                            score = numeric
                passages.append(
                    EvidencePassage(
                        source_id=f"source_{len(sources) + 1}",
                        text=" ".join(highlight.split())[:_MAX_PASSAGE_CHARS],
                        score=score,
                    )
                )
            if not passages:
                continue
            snippet = " ".join(passage.text for passage in passages)[:_MAX_SNIPPET_CHARS]
            sources.append(
                CurrentFactSource(
                    title=title.strip()[:240],
                    url=url.strip()[:2_048],
                    snippet=snippet,
                    backend=self.backend_name,
                    passages=tuple(passages),
                )
            )
        return tuple(sources)


class OpenAIHostedSearchBackend(_ExperimentalKnowledgeBackend):
    """Explicitly API-key-authenticated forced hosted-search experiment."""

    _ENDPOINT = "https://api.openai.com/v1/responses"
    backend_name = "openai-hosted-search"

    def __init__(
        self,
        *,
        api_key: str,
        transport: JsonPostTransport | None = None,
        timeout_seconds: float = 3.5,
        model: str = "gpt-5.6",
    ) -> None:
        super().__init__(
            api_key=api_key,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
        if type(model) is not str or not model.strip() or len(model) > 120:
            raise ValueError("model must be an explicit non-empty string")
        self._model = model

    async def lookup(self, query: str) -> CurrentFactEvidence:
        normalized = self._query(query)
        try:
            payload = await self._transport.post_json(
                self._ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                body={
                    "model": self._model,
                    "input": normalized,
                    "reasoning": {"effort": "none"},
                    "tools": [{"type": "web_search"}],
                    "tool_choice": {"type": "web_search"},
                },
                timeout_seconds=self._timeout_seconds,
            )
            sources = self._parse(payload)
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, KeyError):
            return self._invalid(normalized)
        return self._evidence(normalized, sources)

    def _parse(self, payload: object) -> tuple[CurrentFactSource, ...]:
        if type(payload) is not dict or type(payload.get("output")) is not list:
            raise ValueError("invalid OpenAI response")
        sources: list[CurrentFactSource] = []
        seen_urls: set[str] = set()
        for item in payload["output"]:
            if type(item) is not dict or item.get("type") != "message":
                continue
            content = item.get("content")
            if type(content) is not list:
                continue
            for part in content:
                if type(part) is not dict or part.get("type") != "output_text":
                    continue
                text = part.get("text")
                annotations = part.get("annotations")
                if type(text) is not str or type(annotations) is not list:
                    continue
                for annotation in annotations:
                    if type(annotation) is not dict or annotation.get("type") != "url_citation":
                        continue
                    title = annotation.get("title")
                    url = annotation.get("url")
                    if (
                        type(title) is not str
                        or not title.strip()
                        or type(url) is not str
                        or not url.strip()
                        or url in seen_urls
                    ):
                        continue
                    seen_urls.add(url)
                    snippet = _citation_text(text, annotation)
                    source_id = f"source_{len(sources) + 1}"
                    sources.append(
                        CurrentFactSource(
                            title=title.strip()[:240],
                            url=url.strip()[:2_048],
                            snippet=snippet,
                            backend=self.backend_name,
                            passages=(EvidencePassage(source_id=source_id, text=snippet),),
                        )
                    )
                    if len(sources) == _MAX_RESULTS:
                        return tuple(sources)
        return tuple(sources)


def _citation_text(text: str, annotation: dict[object, object]) -> str:
    start = annotation.get("start_index")
    end = annotation.get("end_index")
    if type(start) is int and type(end) is int and 0 <= start < end <= len(text):
        selected = text[start:end].strip()
        if selected:
            return selected[:_MAX_PASSAGE_CHARS]
    normalized = " ".join(text.split())
    return normalized[:_MAX_PASSAGE_CHARS] or "Source cited by hosted web search."


__all__ = [
    "AiohttpJsonTransport",
    "DdgsKnowledgeBackend",
    "ExaInstantKnowledgeBackend",
    "JsonPostTransport",
    "OpenAIHostedSearchBackend",
    "TavilyKnowledgeBackend",
]
