"""Owned tool-free native formatter transport; native SDK imports are lazy."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import importlib
import json
import os
import re
import time
from collections.abc import AsyncIterable
from types import MappingProxyType
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from .evidence_presentation import (
    CODEX_REJECTED_EVENT_KINDS,
    MAX_CODEX_STREAM_EVENTS,
    MAX_CODEX_STREAM_TEXT_CHARS,
    PRESENTATION_INSTRUCTIONS,
    PRESENTATION_MODES,
    PRESENTATION_SCHEMA,
    frozen,
)


class PresentationOwner(Protocol):
    """Existing run owner surface; no tool admission or dispatch authority."""

    @property
    def lock(self) -> contextlib.AbstractContextManager[object]: ...

    def scope(self) -> tuple[str, dict[str, Any] | None]: ...


async def _consume_codex_stream(
    stream: AsyncIterable[Any], evidence: dict[str, Any] | None = None,
) -> Any:
    """Return only the completed terminal object, bounded and never delta-reconstructed."""
    terminal = None
    terminal_count = 0
    event_count = 0
    text_chars = 0
    allowed = {
        "response.created", "response.in_progress", "response.queued",
        "response.output_item.added", "response.output_item.done",
        "response.content_part.added", "response.content_part.done",
        "response.output_text.delta", "response.output_text.done",
        "response.output_text.annotation.added",
        "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
        "response.completed",
        # Observed Codex subscription extension. Its body is intentionally opaque.
        "response.metadata",
    }
    async for event in stream:
        event_count += 1
        if event_count > MAX_CODEX_STREAM_EVENTS:
            raise ValueError("responses-event-bound")
        event_type = getattr(event, "type", None)
        if type(event_type) is not str:
            raise ValueError("responses-event-type")
        if event_type in {"response.failed", "response.incomplete", "error"}:
            raise ValueError("responses-terminal-failure")
        if event_type not in allowed:
            if evidence is not None:
                evidence["provider_event_kind"] = (
                    event_type if event_type in CODEX_REJECTED_EVENT_KINDS else "other"
                )
            raise ValueError("responses-event-unknown")
        if terminal_count:
            raise ValueError("responses-event-after-terminal")
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if type(delta) is not str:
                raise ValueError("responses-delta-shape")
            text_chars += len(delta)
            if text_chars > MAX_CODEX_STREAM_TEXT_CHARS:
                raise ValueError("responses-stream-text-bound")
        elif event_type in {"response.output_item.added", "response.output_item.done"}:
            item = getattr(event, "item", None)
            item_type = getattr(item, "type", None)
            if item_type not in {"message", "reasoning"}:
                raise ValueError("responses-stream-output-item")
        elif event_type in {"response.content_part.added", "response.content_part.done"}:
            part = getattr(event, "part", None)
            if getattr(part, "type", None) != "output_text":
                raise ValueError("responses-stream-content-part")
        if event_type == "response.completed":
            terminal_count += 1
            terminal = getattr(event, "response", None)
            if terminal is None:
                raise ValueError("responses-terminal-missing")
    if terminal_count != 1 or terminal is None:
        raise ValueError("responses-terminal-count")
    return terminal



class Presentation:
    """One explicit same-route SDK call, independent of native tool dispatch."""

    def __init__(self, config: dict[str, str]) -> None:
        required = {"provider", "base_url", "model", "api_key_env"}
        if (type(config) is not dict or not required <= set(config)
                or set(config) - required not in (set(), {"api_mode"})):
            raise ValueError("presentation-config")
        normalized = dict(config)
        normalized.setdefault("api_mode", "chat_completions")
        if any(type(normalized[key]) is not str
               or not 1 <= len(normalized[key]) <= 2048
               for key in required | {"api_mode"}):
            raise ValueError("presentation-config")
        if normalized["api_mode"] not in PRESENTATION_MODES:
            raise ValueError("presentation-config")
        url = urlsplit(normalized["base_url"])
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", normalized["api_key_env"]) is None):
            raise ValueError("presentation-config")
        self.config = MappingProxyType(normalized)

    def validate_route(self, kw: dict[str, Any]) -> None:
        if any(kw.get(key) != self.config[key]
               for key in ("provider", "base_url", "model", "api_mode")):
            raise ValueError("presentation-route")
        if kw["request"].get("model") != self.config["model"]:
            raise ValueError("presentation-model")

    async def complete(
        self, generation: PresentationOwner, run_id: str, row: dict[str, Any],
        source_input: str, evidence: dict[str, Any],
    ) -> Any:
        # These packages belong to the configured native runtime, not the host wheel.
        httpx = importlib.import_module("httpx")
        AsyncOpenAI = importlib.import_module("openai").AsyncOpenAI

        key = os.environ.get(self.config["api_key_env"])
        if not key:
            raise ValueError("presentation-credential")
        if self.config["api_mode"] == "codex_responses":
            return await self._complete_codex(generation, row, source_input, evidence, key,
                                               httpx, AsyncOpenAI)
        body = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": PRESENTATION_INSTRUCTIONS},
                {"role": "user", "content": source_input},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "voice_detail", "strict": True, "schema": PRESENTATION_SCHEMA}},
            "max_tokens": 1024,
            "stream": False,
            "reasoning_effort": "none",
        }

        async def observe(request: Any) -> None:
            # Actual SDK serialization; fixed fields/hashes only, never prompt or auth.
            raw = request.content
            serialized = json.loads(raw)
            actual_input = serialized["messages"][-1]["content"]
            payload = json.loads(actual_input)
            evidence["wire"] = {
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "model_matches": serialized.get("model") == self.config["model"],
                "tools_absent": not any(name in serialized for name in (
                    "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call")),
                "nonstreaming": serialized.get("stream") is False,
                "reasoning_effort_none": serialized.get("reasoning_effort") == "none",
                "schema_matches": serialized.get("response_format") == body["response_format"],
                "source_sha256": evidence.get("source_sha256", ""),
                "input_sha256": hashlib.sha256(actual_input.encode()).hexdigest(),
                "catalog_sha256": hashlib.sha256(frozen(payload["sources"]).encode()).hexdigest(),
                "catalog_count": len(payload["sources"]),
                "instruction_sha256": hashlib.sha256(
                    serialized["messages"][0]["content"].encode()).hexdigest(),
            }
            evidence["wire_requests"] += 1

        active = {"loop": asyncio.get_running_loop(), "task": asyncio.current_task(),
                  "cancel_requested": False, "closing": False}
        settled = False
        failure = None
        try:
            remaining = row["deadline"] - time.monotonic()
            if remaining <= 0:
                raise ValueError("presentation-deadline")
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=remaining, event_hooks={"request": [observe]}
            ) as http, AsyncOpenAI(
                base_url=self.config["base_url"], api_key=key, http_client=http,
                max_retries=0, timeout=remaining,
            ) as client:
                with generation.lock:
                    _, current = generation.scope()
                    if current is not row or row["cancelled"] or row.get("presentation_active"):
                        raise ValueError("presentation-ownership")
                    remaining = row["deadline"] - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("presentation-deadline")
                    row["presentation_active"] = active
                try:
                    async with asyncio.timeout(remaining):
                        result = await client.chat.completions.create(**body)
                except BaseException as error:
                    failure = error
                finally:
                    with generation.lock:
                        active["closing"] = True
            settled = True
            if failure is not None:
                raise failure
            return result
        finally:
            # Timeout and repeat stop requests cannot cancel the context-manager cleanup.
            evidence["local_http_settled"] = settled
            with generation.lock:
                if settled and row.get("presentation_active") is active:
                    row["presentation_active"] = None

    async def _complete_codex(
        self, generation: PresentationOwner, row: dict[str, Any], source_input: str,
        evidence: dict[str, Any], key: str, httpx: Any, AsyncOpenAI: Any,
    ) -> Any:
        codex_cloudflare_headers = importlib.import_module(
            "agent.codex_headers").codex_cloudflare_headers

        body = {
            "model": self.config["model"],
            "instructions": PRESENTATION_INSTRUCTIONS,
            "input": [{"role": "user", "content": source_input}],
            "store": False,
            "stream": True,
            "text": {"format": {
                "type": "json_schema", "name": "voice_detail", "strict": True,
                "schema": PRESENTATION_SCHEMA,
            }},
            "reasoning": {"effort": "low"},
        }
        expected_text_format = copy.deepcopy(body["text"])

        async def observe(request: Any) -> None:
            raw = request.content
            serialized = json.loads(raw)
            input_items = serialized.get("input")
            if (type(input_items) is not list or len(input_items) != 1
                    or type(input_items[0]) is not dict
                    or input_items[0].get("role") != "user"
                    or type(input_items[0].get("content")) is not str):
                raise ValueError("presentation-wire-input")
            actual_input = input_items[0]["content"]
            payload = json.loads(actual_input)
            instructions = serialized.get("instructions")
            text = serialized.get("text")
            evidence["wire"] = {
                "body_sha256": hashlib.sha256(raw).hexdigest(),
                "input_matches": actual_input == source_input,
                "instructions_match": instructions == PRESENTATION_INSTRUCTIONS,
                "model_matches": serialized.get("model") == self.config["model"],
                "tools_absent": not any(name in serialized for name in (
                    "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call")),
                "streaming": serialized.get("stream") is True,
                "store_disabled": serialized.get("store") is False,
                "reasoning_effort_low": serialized.get("reasoning") == {"effort": "low"},
                "schema_matches": text == expected_text_format,
                "source_sha256": evidence.get("source_sha256", ""),
                "input_sha256": hashlib.sha256(actual_input.encode()).hexdigest(),
                "catalog_sha256": hashlib.sha256(frozen(payload["sources"]).encode()).hexdigest(),
                "catalog_count": len(payload["sources"]),
                "instruction_sha256": (hashlib.sha256(instructions.encode()).hexdigest()
                                        if type(instructions) is str else ""),
            }
            if (set(serialized) != set(body)
                    or not evidence["wire"]["input_matches"]
                    or not evidence["wire"]["instructions_match"]
                    or not evidence["wire"]["model_matches"]
                    or not evidence["wire"]["tools_absent"]
                    or not evidence["wire"]["streaming"]
                    or not evidence["wire"]["store_disabled"]
                    or not evidence["wire"]["reasoning_effort_low"]
                    or not evidence["wire"]["schema_matches"]):
                raise ValueError("presentation-wire-contract")
            evidence["wire_requests"] += 1

        active = {"loop": asyncio.get_running_loop(), "task": asyncio.current_task(),
                  "cancel_requested": False, "closing": False}
        settled = False
        failure = None
        stream = None
        response = None
        try:
            remaining = row["deadline"] - time.monotonic()
            if remaining <= 0:
                raise ValueError("presentation-deadline")
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=remaining, event_hooks={"request": [observe]}
            ) as http, AsyncOpenAI(
                base_url=self.config["base_url"], api_key=key, http_client=http,
                default_headers=codex_cloudflare_headers(
                    key, base_url=self.config["base_url"]),
                max_retries=0, timeout=remaining,
            ) as client:
                with generation.lock:
                    _, current = generation.scope()
                    if current is not row or row["cancelled"] or row.get("presentation_active"):
                        raise ValueError("presentation-ownership")
                    remaining = row["deadline"] - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("presentation-deadline")
                    row["presentation_active"] = active
                try:
                    async with asyncio.timeout(remaining):
                        stream = await client.responses.create(**body)
                        response = await _consume_codex_stream(stream, evidence)
                except BaseException as error:
                    failure = error
                finally:
                    if stream is not None:
                        with generation.lock:
                            active["closing"] = True
                        await stream.close()
                    else:
                        with generation.lock:
                            active["closing"] = True
            settled = True
            if failure is not None:
                raise failure
            return response
        finally:
            evidence["local_http_settled"] = settled
            with generation.lock:
                if settled and row.get("presentation_active") is active:
                    row["presentation_active"] = None



def cancel_presentation(owner: PresentationOwner, row: dict[str, Any]) -> None:
    active = cast(dict[str, Any], row.get("presentation_active"))
    if active and not active["cancel_requested"]:
        active["cancel_requested"] = True

        def cancel() -> None:
            with owner.lock:
                if row.get("presentation_active") is active and not active["closing"]:
                    active["task"].cancel()

        with contextlib.suppress(RuntimeError):
            # The owning finally may already have settled a closed event loop.
            active["loop"].call_soon_threadsafe(cancel)
