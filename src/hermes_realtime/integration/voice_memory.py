"""Off-turn companion memory subscription for one foreground conversation binding."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Protocol

from hermes_realtime.conversation.context import ConversationContextStore
from hermes_realtime.integration.bridge import LocalHermesBridgeClient
from hermes_realtime.memory import BuiltinMemorySnapshot
from hermes_realtime.protocol import (
    VOICE_MEMORY_CAPABILITY,
    VoiceMemoryEvent,
    VoiceMemoryRefusedEvent,
    VoiceMemorySnapshotEvent,
)

_MARKER = "[voice-memory-receive] "
_RETAINING_REFUSALS = frozenset({"pending", "capacity"})


class VoiceMemoryLink(Protocol):
    @property
    def capabilities(self) -> frozenset[str]: ...

    def memory(
        self, event: VoiceMemoryEvent
    ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]: ...

    async def close(self) -> None: ...


def memory_connector(
    *, host: str, port: int, token: str
) -> Callable[[], Awaitable[VoiceMemoryLink]]:
    async def connect() -> VoiceMemoryLink:
        return await LocalHermesBridgeClient.connect(
            host=host,
            port=port,
            token=token,
            participant_id="voice-memory",
            capabilities=(VOICE_MEMORY_CAPABILITY,),
        )

    return connect


def _marker(evidence: dict[str, str | int]) -> None:
    print(_MARKER + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


class VoiceMemoryReceiver:
    """Apply only ordered pushes from the current, exact companion binding."""

    def __init__(
        self,
        context: ConversationContextStore,
        connect: Callable[[], Awaitable[VoiceMemoryLink]],
        binding: Callable[[], tuple[str, int]],
        *,
        connect_timeout_seconds: float = 30.0,
        close_timeout_seconds: float = 5.0,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        if type(context) is not ConversationContextStore:
            raise TypeError("memory receiver needs an exact context store")
        if not callable(connect) or not callable(binding):
            raise TypeError("memory connector and binding must be callable")
        times = (
            connect_timeout_seconds,
            close_timeout_seconds,
            initial_backoff_seconds,
            max_backoff_seconds,
        )
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in times):
            raise TypeError("memory receiver timing must be finite exact numbers")
        if not (0 < connect_timeout_seconds <= 300 and 0 < close_timeout_seconds <= 30):
            raise ValueError("memory receiver timeouts must be positive and bounded")
        if not 0 < initial_backoff_seconds <= max_backoff_seconds <= 300:
            raise ValueError("memory receiver backoff must be positive and bounded")
        self._context = context
        self._connect = connect
        self._binding = binding
        self._connect_timeout = float(connect_timeout_seconds)
        self._close_timeout = float(close_timeout_seconds)
        self._initial_backoff = float(initial_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._task: asyncio.Task[None] | None = None
        self._link: VoiceMemoryLink | None = None
        self._request: VoiceMemoryEvent | None = None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("memory receiver starts at most once")
        identity = self._binding()
        if type(identity) is not tuple or len(identity) != 2:
            raise TypeError("memory binding must be an exact pair")
        self._request = VoiceMemoryEvent(
            protocol_version="0.3",
            type="voice_memory",
            conversation_id=identity[0],
            generation=identity[1],
        )
        self._context.set_memory(None)
        self._task = asyncio.create_task(self._run(), name="voice-memory-receiver")

    async def close(self) -> None:
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._close_timeout)
            except asyncio.CancelledError:
                owner = asyncio.current_task()
                if not task.done() or (owner is not None and owner.cancelling()):
                    raise
            except TimeoutError:
                raise RuntimeError("memory receiver did not stop before close timeout") from None
            if not task.done():
                raise RuntimeError("memory receiver is still running")
            self._task = None
        await self._drop()
        self._context.set_memory(None)

    async def _drop(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            with suppress(Exception):
                async with asyncio.timeout(self._close_timeout):
                    await link.close()

    def _same_binding(self, request: VoiceMemoryEvent) -> bool:
        try:
            current = self._binding()
        except Exception:
            return False
        return (
            type(current) is tuple
            and len(current) == 2
            and type(current[0]) is str
            and type(current[1]) is int
            and current == (request.conversation_id, request.generation)
        )

    async def _run(self) -> None:
        request = self._request
        assert request is not None
        backoff = self._initial_backoff
        try:
            while True:
                self._context.set_memory(None)
                if not self._same_binding(request):
                    _marker({"refusal": "binding", "version": 1})
                    return
                try:
                    async with asyncio.timeout(self._connect_timeout):
                        link = await self._connect()
                    self._link = link
                    if VOICE_MEMORY_CAPABILITY not in link.capabilities:
                        _marker({"refusal": "capability", "version": 1})
                        return
                    stream = link.memory(request)
                    if not hasattr(stream, "__aiter__"):
                        _marker({"refusal": "stream", "version": 1})
                        return
                    revision: int | None = None
                    async for event in stream:
                        if not self._same_binding(request) or (
                            type(event) not in (VoiceMemorySnapshotEvent, VoiceMemoryRefusedEvent)
                            or (event.conversation_id, event.generation)
                            != (request.conversation_id, request.generation)
                        ):
                            _marker({"refusal": "binding", "version": 1})
                            return
                        if type(event) is VoiceMemoryRefusedEvent:
                            if event.category not in _RETAINING_REFUSALS:
                                self._context.set_memory(None)
                            _marker({"refusal": event.category, "version": 1})
                            continue
                        assert type(event) is VoiceMemorySnapshotEvent
                        if revision is not None and event.revision < revision:
                            _marker({"refusal": "revision", "version": 1})
                            return
                        self._context.set_memory(
                            BuiltinMemorySnapshot(
                                memory=event.memory,
                                user=event.user,
                                truncated=event.truncated,
                            )
                        )
                        if revision is None:
                            backoff = self._initial_backoff
                        revision = event.revision
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    _marker(
                        {"outcome": "disconnected", "cause": type(error).__name__, "version": 1}
                    )
                finally:
                    self._context.set_memory(None)
                    await self._drop()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
        finally:
            self._context.set_memory(None)
            await self._drop()


__all__ = ["VoiceMemoryLink", "VoiceMemoryReceiver", "memory_connector"]
