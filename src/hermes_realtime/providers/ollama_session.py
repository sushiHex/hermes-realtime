"""One pre-connect local choice, frozen for the lifetime of its host."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

from hermes_realtime.client.session import BrowserModelConfiguration
from hermes_realtime.conversation import ConversationContextSnapshot, ConversationInferenceRequest

from .ollama import OllamaStreamingInference
from .ollama_models import (
    DEFAULT_OLLAMA_NUM_CTX,
    OllamaContextConfiguration,
    OllamaContextObservation,
    OllamaModelClient,
    OllamaModelMetadata,
    _require_model,
)

PREPARATION_TIMEOUT_SECONDS = 120.0
CLEANUP_TIMEOUT_SECONDS = 3.0


class OllamaSessionSelection:
    """Own metadata, candidate readiness, the committed adapter and its cleanup."""

    def __init__(
        self,
        *,
        client: OllamaModelClient,
        default_model: str,
        adapter_factory: Callable[[str], OllamaStreamingInference],
        readiness: Callable[[OllamaStreamingInference], Awaitable[None]],
    ) -> None:
        if type(client) is not OllamaModelClient:
            raise TypeError("client must be an exact Ollama metadata client")
        _require_model(default_model)
        if not callable(adapter_factory) or not callable(readiness):
            raise TypeError("adapter factory and readiness must be callable")
        self._client = client
        self._default_model = default_model
        self._factory = adapter_factory
        self._readiness = readiness
        self._models: tuple[OllamaModelMetadata, ...] | None = None
        self._catalog_task: asyncio.Task[tuple[OllamaModelMetadata, ...]] | None = None
        self._preparation: asyncio.Task[dict[str, object]] | None = None
        self._adapter: OllamaStreamingInference | None = None
        self._configuration: BrowserModelConfiguration | None = None
        self._selection: dict[str, object] | None = None
        self._closed = False
        self._owned: set[OllamaStreamingInference] = set()
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_tasks: dict[OllamaStreamingInference, asyncio.Task[None]] = {}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("local selection host is closed")

    async def catalog(self) -> dict[str, object]:
        category = "refused"
        try:
            result = await self._catalog()
            category = "available"
            return result
        finally:
            print(
                "[local-model-catalog] "
                + json.dumps({"version": 1, "category": category}, separators=(",", ":")),
                flush=True,
            )

    async def _catalog(self) -> dict[str, object]:
        self._require_open()
        if self._models is None:
            if self._catalog_task is None:
                self._catalog_task = asyncio.create_task(self._client.installed_models())
            operation = self._catalog_task
            try:
                self._models = await asyncio.shield(operation)
            finally:
                if operation.done() and self._catalog_task is operation:
                    self._catalog_task = None
            self._require_open()
        models = self._models
        return {
            "version": 1,
            "available": True,
            "provider": "ollama",
            "state": "prepared"
            if self._selection is not None
            else ("preparing" if self._preparation is not None else "configuring"),
            "models": [
                {
                    "model": row.model,
                    "digest": row.digest,
                    "advertisedContextTokens": row.advertised_context_tokens,
                }
                for row in models
            ],
            "defaultModel": self._default_model
            if any(row.model == self._default_model for row in models)
            else None,
            "defaultRequestedContextTokens": DEFAULT_OLLAMA_NUM_CTX,
            "minimumContextTokens": 2048,
            "maximumContextTokens": 262144,
            "preparationTimeoutMs": int(PREPARATION_TIMEOUT_SECONDS * 1000),
            "selection": None if self._selection is None else dict(self._selection),
        }

    async def prepare(
        self,
        model: str,
        digest: str,
        requested_context_tokens: int,
    ) -> dict[str, object]:
        category = "choice_refused"
        try:
            async with asyncio.timeout(PREPARATION_TIMEOUT_SECONDS):
                await self.catalog()
                models = cast(tuple[OllamaModelMetadata, ...], self._models)
                claimed = OllamaModelMetadata(model, digest, None)
                row = next((item for item in models if item.model == claimed.model), None)
                if row is None or row.digest != claimed.digest:
                    raise ValueError("selection identity does not match the installed catalog")
                choice = OllamaContextConfiguration(row, requested_context_tokens)
                self._require_open()
                if self._selection is not None:
                    if (model, digest, requested_context_tokens) != (
                        self._selection["model"],
                        self._selection["digest"],
                        self._selection["requestedContextTokens"],
                    ):
                        raise RuntimeError(
                            "local selection is frozen; restart the host to change it"
                        )
                    category = "prepared"
                    return dict(self._selection)
                if self._preparation is not None:
                    raise RuntimeError("local selection is already preparing")
                operation = asyncio.create_task(self._prepare(choice))
                self._preparation = operation
                category = "readiness_refused"
                try:
                    result = await operation
                    category = "prepared"
                    return result
                finally:
                    self._preparation = None
        except TimeoutError:
            category = "deadline"
            raise
        except asyncio.CancelledError:
            category = "cancelled"
            raise
        finally:
            print(
                "[local-model-prepare] "
                + json.dumps({"version": 1, "category": category}, separators=(",", ":")),
                flush=True,
            )

    async def _prepare(self, choice: OllamaContextConfiguration) -> dict[str, object]:
        candidate = self._factory(choice.model.model)
        if type(candidate) is not OllamaStreamingInference:
            raise TypeError("local preparation requires an exact Ollama inference adapter")
        self._owned.add(candidate)
        committed = False
        try:
            await candidate.configure_context(
                cast(int, choice.requested_context_tokens),
                client=self._client,
            )
            observation = candidate.context_observation()
            configured = observation.configuration
            if configured != choice:
                raise ValueError("selected model identity changed before preparation")
            await self._readiness(candidate)
            await candidate.observe_context()
            self._require_open()
            observation = candidate.context_observation()
            if observation.configuration != configured:
                raise ValueError("prepared model identity changed during readiness")
            metadata = observation.configuration.model
            selection: dict[str, object] = {
                "model": metadata.model,
                "digest": metadata.digest,
                "requestedContextTokens": observation.configuration.requested_context_tokens,
                "advertisedContextTokens": metadata.advertised_context_tokens,
                "effectiveContextTokens": observation.effective_context_tokens,
            }
            configuration = BrowserModelConfiguration(
                model=metadata.model,
                provider="ollama",
                authentication="local",
                transport="local-http",
                effort=None,
                context_window_tokens=None,
                reports_token_usage=False,
                local_context=candidate.context_observation,
            )
            models = cast(tuple[OllamaModelMetadata, ...], self._models)
            self._models = tuple(metadata if row.model == metadata.model else row for row in models)
            self._adapter = candidate
            self._configuration = configuration
            self._selection = selection
            committed = True
            return dict(selection)
        finally:
            if not committed:
                await self._close_adapter(candidate)

    async def _close_adapter(self, adapter: OllamaStreamingInference) -> None:
        try:
            async with asyncio.timeout(CLEANUP_TIMEOUT_SECONDS), self._cleanup_lock:
                if adapter in self._owned:
                    operation = self._cleanup_tasks.get(adapter)
                    if operation is None or (
                        operation.done()
                        and (operation.cancelled() or operation.exception() is not None)
                    ):
                        operation = asyncio.create_task(adapter.close())
                        self._cleanup_tasks[adapter] = operation
                    await asyncio.shield(operation)
                    self._owned.remove(adapter)
                    del self._cleanup_tasks[adapter]
        except BaseException:
            self._closed = True
            raise

    def require_prepared(self) -> BrowserModelConfiguration:
        category = "refused"
        try:
            self._require_open()
            if self._configuration is None:
                raise RuntimeError("local selection must be prepared before connecting")
            category = "prepared"
            return self._configuration
        finally:
            print(
                "[local-model-admission] "
                + json.dumps({"version": 1, "category": category}, separators=(",", ":")),
                flush=True,
            )

    def context_observation(self) -> OllamaContextObservation:
        self.require_prepared()
        return cast(OllamaStreamingInference, self._adapter).context_observation()

    async def stream(
        self,
        snapshot: ConversationContextSnapshot | ConversationInferenceRequest,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        self.require_prepared()
        adapter = cast(OllamaStreamingInference, self._adapter)
        async for segment in adapter.stream(snapshot, turn_id=turn_id):
            yield segment

    async def cancel(self, turn_id: str) -> None:
        if self._adapter is not None:
            await self._adapter.cancel(turn_id)

    async def close(self) -> None:
        category = "cleanup_refused"
        try:
            self._closed = True
            tasks = [task for task in (self._catalog_task, self._preparation) if task is not None]
            for task in tasks:
                task.cancel()
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=CLEANUP_TIMEOUT_SECONDS)
                if pending:
                    raise TimeoutError("local preparation cleanup is still pending")
                # Observe owned task failures; retained adapters are retried below.
                for task in tasks:
                    if not task.cancelled():
                        task.exception()
            for adapter in tuple(self._owned):
                await self._close_adapter(adapter)
            self._adapter = None
            category = "closed"
        finally:
            print(
                "[local-model-close] "
                + json.dumps(
                    {"version": 1, "category": category, "remaining": min(len(self._owned), 256)},
                    separators=(",", ":"),
                ),
                flush=True,
            )
