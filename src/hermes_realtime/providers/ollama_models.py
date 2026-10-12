"""Bounded, metadata-only Ollama discovery and immutable session context choices.

No family/name table and no tokenizer estimate: advertised capacity is a metadata
claim; ps reports an observed allocation, not a guarantee about the next request.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

import aiohttp

DEFAULT_OLLAMA_NUM_CTX = 16_384
_MAX_BYTES = 1_048_576
_MAX_MODELS = 256
_MAX_CAPACITY = 100_000_000
_REQUEST_TIMEOUT_SECONDS = 5.0
_DISCOVERY_TIMEOUT_SECONDS = 10.0
_CATALOG_TIMEOUT_SECONDS = 8.0
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


_ReadRequest = Callable[[str, dict[str, object] | None], Awaitable[bytes]]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("metadata JSON object keys must be unique")
        value[key] = item
    return value


def _capacity(value: object) -> int | None:
    return value if type(value) is int and 1 <= value <= _MAX_CAPACITY else None


def _optional_capacity(value: object) -> int | None:
    if value is None:
        return None
    capacity = _capacity(value)
    if capacity is None:
        raise ValueError("metadata capacity must be a bounded exact integer or unavailable")
    return capacity


def _advertised_capacity(info: object) -> int | None:
    if info is None:
        return None
    if type(info) is not dict:
        raise ValueError("model_info must be an exact object or unavailable")
    architecture = info.get("general.architecture")
    if architecture is None:
        return None
    if type(architecture) is not str or not 1 <= len(architecture) <= 128:
        raise ValueError("advertised architecture must be a bounded exact string")
    return _optional_capacity(info.get(f"{architecture}.context_length"))


def _require_local(row: dict[str, object]) -> None:
    for field in ("remote_host", "remote_model"):
        if field in row and row[field] != "":
            raise ValueError("context configuration requires local model weights")


def _completion_capability(payload: dict[str, object]) -> bool:
    capabilities = payload.get("capabilities")
    if (
        type(capabilities) is not list
        or len(capabilities) > 32
        or any(type(item) is not str or len(item) > 64 for item in capabilities)
    ):
        raise ValueError("model completion capabilities must be a bounded exact string list")
    return "completion" in capabilities


def require_num_ctx(value: object) -> int:
    if type(value) is not int:
        raise TypeError("num_ctx must be an exact integer")
    if not 2048 <= value <= 262_144:
        raise ValueError("num_ctx must be between 2048 and 262144")
    return value


@dataclass(frozen=True, slots=True)
class OllamaModelMetadata:
    model: str
    digest: str
    advertised_context_tokens: int | None

    def __post_init__(self) -> None:
        _require_model(self.model)
        if type(self.digest) is not str or _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("model digest must be an exact SHA256 digest")
        if self.advertised_context_tokens is not None and (
            _capacity(self.advertised_context_tokens) is None
        ):
            raise ValueError("advertised context must be a bounded exact integer or unknown")


@dataclass(frozen=True, slots=True)
class OllamaContextConfiguration:
    model: OllamaModelMetadata
    requested_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self.model) is not OllamaModelMetadata:
            raise TypeError("model must be exact OllamaModelMetadata")
        requested = self.requested_context_tokens
        maximum = self.model.advertised_context_tokens
        if requested is None:
            if maximum is None:
                raise ValueError("unknown capacity requires an explicit context setting")
            requested = DEFAULT_OLLAMA_NUM_CTX
        requested = require_num_ctx(requested)
        if maximum is not None and requested > maximum:
            raise ValueError("requested context exceeds advertised model maximum")
        object.__setattr__(self, "requested_context_tokens", requested)


@dataclass(frozen=True, slots=True)
class OllamaContextObservation:
    """One immutable configuration and separately observed runtime allocation."""

    configuration: OllamaContextConfiguration
    effective_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self.configuration) is not OllamaContextConfiguration:
            raise TypeError("configuration must be exact OllamaContextConfiguration")
        if (
            self.effective_context_tokens is not None
            and _capacity(self.effective_context_tokens) is None
        ):
            raise ValueError("effective context must be a bounded exact integer or unknown")

    def public_data(self) -> dict[str, str | int | bool | None]:
        return {
            "version": 2,
            "requestedContextTokens": self.configuration.requested_context_tokens,
            "advertisedContextTokens": self.configuration.model.advertised_context_tokens,
            "effectiveContextTokens": self.effective_context_tokens,
            "modelDigest": self.configuration.model.digest,
        }


def _require_model(value: object) -> str:
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError("model must be an exact nonblank bounded string")
    if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value):
        raise ValueError("model identity cannot contain control characters")
    return value


class OllamaModelClient:
    """Read tags/show/ps/version without loading, pulling or generating a model."""

    def __init__(
        self,
        *,
        base_url: str,
        read_request: _ReadRequest | None = None,
    ) -> None:
        if type(base_url) is not str:
            raise TypeError("base_url must be an exact built-in string")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.port is None
            or parsed.username is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama metadata requires an exact loopback HTTP origin")
        self._origin = base_url.rstrip("/")
        self._read = read_request or self._read_request

    async def _read_request(self, endpoint: str, body: dict[str, object] | None) -> bytes:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS),
                trust_env=False,
            ) as client,
            client.request(
                "GET" if body is None else "POST",
                f"{self._origin}/api/{endpoint}",
                json=body,
                allow_redirects=False,
            ) as response,
        ):
            if response.status != 200:
                raise aiohttp.ClientResponseError(
                    response.request_info, response.history, status=response.status
                )
            chunks = bytearray()
            while len(chunks) <= _MAX_BYTES:
                chunk = await response.content.read(min(65536, _MAX_BYTES + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            return bytes(chunks)

    async def _request(
        self,
        endpoint: str,
        body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        category = "unavailable"
        count = 0
        try:
            async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
                raw = await self._read(endpoint, body)
            category = "malformed"
            if type(raw) is not bytes:
                raise TypeError("metadata read must return exact bytes")
            count = min(len(raw), _MAX_BYTES + 1)
            if len(raw) > _MAX_BYTES:
                category = "oversized"
                raise ValueError("metadata response exceeds supported size")
            value = json.loads(raw, object_pairs_hook=_unique_object)
            if type(value) is not dict:
                raise TypeError("metadata response must be an exact object")
            category = "accepted"
            return cast(dict[str, object], value)
        finally:
            print(
                "[ollama-metadata] "
                + json.dumps(
                    {"version": 1, "kind": endpoint, "category": category, "bytes": count},
                    separators=(",", ":"),
                ),
                flush=True,
            )

    @staticmethod
    def _models(payload: dict[str, object]) -> list[dict[str, object]]:
        models = payload.get("models")
        if type(models) is not list or len(models) > _MAX_MODELS:
            raise ValueError("metadata models must be a bounded exact list")
        if any(type(row) is not dict for row in models):
            raise TypeError("metadata model rows must be exact objects")
        return cast(list[dict[str, object]], models)

    @classmethod
    def _selected_row(cls, payload: dict[str, object], model: str) -> dict[str, object] | None:
        rows = [
            row
            for row in cls._models(payload)
            if row.get("name") == model or row.get("model") == model
        ]
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("selected model identity is ambiguous")
        row = rows[0]
        if row.get("name") != model or row.get("model", model) != model:
            raise ValueError("selected model identity fields disagree")
        return row

    async def identity(self, model: str) -> str:
        _require_model(model)
        row = self._selected_row(await self._request("tags"), model)
        if row is None:
            raise ValueError("selected model must have one consistent installed identity")
        _require_local(row)
        digest = row.get("digest")
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise ValueError("model identity requires an exact SHA256 digest")
        return digest

    @classmethod
    def _catalog_rows(cls, payload: dict[str, object]) -> dict[str, dict[str, object]]:
        rows = cls._models(payload)
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            name = _require_model(row.get("name"))
            # Use the same name/model ambiguity rule as individual discovery.
            selected = cast(dict[str, object], cls._selected_row(payload, name))
            OllamaModelMetadata(name, cast(str, selected.get("digest")), None)
            for field in ("remote_host", "remote_model"):
                if field in selected and type(selected[field]) is not str:
                    raise ValueError("remote model declarations must be exact strings")
            result[name] = selected
        return result

    async def installed_models(self) -> tuple[OllamaModelMetadata, ...]:
        """Discover the whole installed catalog without loading any model."""
        category = "metadata_refused"
        count = excluded = refused_capacity = 0
        try:
            async with asyncio.timeout(_CATALOG_TIMEOUT_SECONDS):
                before = self._catalog_rows(await self._request("tags"))
                count = len(before)
                semaphore = asyncio.Semaphore(4)

                async def discover(name: str, row: dict[str, object]) -> OllamaModelMetadata | None:
                    nonlocal excluded, refused_capacity
                    if any(row.get(field, "") != "" for field in ("remote_host", "remote_model")):
                        excluded += 1
                        return None
                    async with semaphore:
                        payload = await self._request("show", {"model": name})
                    _require_local(payload)
                    if not _completion_capability(payload):
                        excluded += 1
                        return None
                    try:
                        maximum = _advertised_capacity(payload.get("model_info"))
                    except ValueError:
                        maximum = None
                        refused_capacity += 1
                    return OllamaModelMetadata(name, cast(str, row["digest"]), maximum)

                tasks: list[asyncio.Task[OllamaModelMetadata | None]] = []
                async with asyncio.TaskGroup() as group:
                    for name, row in before.items():
                        tasks.append(group.create_task(discover(name, row)))
                after = self._catalog_rows(await self._request("tags"))
                if after != before:
                    raise ValueError("installed model identity changed during catalog discovery")
                models = tuple(task.result() for task in tasks if task.result() is not None)
                category = "capacity_refused" if refused_capacity else "discovered"
                return cast(tuple[OllamaModelMetadata, ...], models)
        except TimeoutError:
            category = "deadline"
            raise
        finally:
            print(
                "[ollama-catalog] "
                + json.dumps(
                    {
                        "version": 1,
                        "category": category,
                        "models": count,
                        "excluded": excluded,
                        "capacityRefused": refused_capacity,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    async def discover(self, model: str) -> OllamaModelMetadata:
        category = "metadata_refused"
        try:
            async with asyncio.timeout(_DISCOVERY_TIMEOUT_SECONDS):
                metadata, category = await self._discover(model)
            return metadata
        except TimeoutError:
            category = "deadline"
            raise
        finally:
            print(
                "[ollama-discovery] "
                + json.dumps({"version": 1, "category": category}, separators=(",", ":")),
                flush=True,
            )

    async def _discover(self, model: str) -> tuple[OllamaModelMetadata, str]:
        digest = await self.identity(model)
        payload = await self._request("show", {"model": model})
        _require_local(payload)
        if not _completion_capability(payload):
            raise ValueError("selected model must advertise completion capability")
        try:
            maximum = _advertised_capacity(payload.get("model_info"))
            category = "discovered" if maximum is not None else "capacity_unknown"
        except ValueError:
            maximum = None
            category = "capacity_refused"
        if await self.identity(model) != digest:
            raise ValueError("model identity changed during discovery")
        return OllamaModelMetadata(model, digest, maximum), category

    async def effective_context(self, model: OllamaModelMetadata) -> int | None:
        row = self._selected_row(await self._request("ps"), model.model)
        if row is None:
            return None
        if row.get("digest") != model.digest:
            raise ValueError("allocation model identity differs from session configuration")
        _require_local(row)
        return _optional_capacity(row.get("context_length"))

    async def require_context_contract(self) -> None:
        # Exact source-reviewed runtime contract, independent of model identity.
        # Unknown/older servers may silently ignore the no-truncation fields.
        if (await self._request("version")).get("version") != "0.34.4":
            raise ValueError("context safety requires source-reviewed Ollama runtime 0.34.4")
