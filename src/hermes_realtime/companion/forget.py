"""Durable, idempotent deletion of a voice generation's Hermes sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Literal, Protocol, TypeVar

from hermes_realtime.companion.integrity import ArchiveRefusal, validate_conversation_id
from hermes_realtime.companion.store import CompanionStore, DeleteTarget


class DeletePort(Protocol):
    def capture_delete_targets(
        self, voice_session_id: str | None, allow_missing_voice: bool,
    ) -> tuple[DeleteTarget, ...]: ...

    def delete_target(self, target: DeleteTarget) -> bool: ...

    def absent(self, session_ids: tuple[str, ...]) -> bool: ...


_T = TypeVar("_T")


class VoiceForgetReconciler:
    """Tombstone first; replay the same frozen manifest after failure or restart."""

    def __init__(
        self,
        store: CompanionStore,
        port: DeletePort,
        review_admitted: Callable[[str], bool],
        guard: Callable[[str], object] | None = None,
        on_tombstone: Callable[[str], None] | None = None,
        on_complete: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._port = port
        self._review_admitted = review_admitted
        self._guard = guard
        self._on_tombstone = on_tombstone
        self._on_complete = on_complete
        self._fallback_lock = asyncio.Lock()

    async def _call(self, function: Callable[..., _T], *arguments: object) -> _T:
        """Keep the conversation lock until a native worker ends, even on cancellation."""
        work = asyncio.create_task(asyncio.to_thread(function, *arguments))
        cancelled = False
        while not work.done():
            try:
                await asyncio.wait({work})
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            work.exception()
            raise asyncio.CancelledError
        return work.result()

    @contextlib.asynccontextmanager
    async def _conversation(self, conversation_id: str) -> AsyncIterator[None]:
        if self._guard is not None:
            async with self._guard(conversation_id):  # type: ignore[attr-defined]
                yield
            return
        async with self._fallback_lock:
            yield

    async def forget(
        self, conversation_id: str, generation: int,
    ) -> Literal["pending", "complete"]:
        validate_conversation_id(conversation_id)
        async with self._conversation(conversation_id):
            self._store.tombstone(conversation_id, generation)
            if self._on_tombstone is not None:
                self._on_tombstone(conversation_id)
            return await self._reconcile(conversation_id)

    async def reconcile(self, conversation_id: str) -> Literal["pending", "complete"]:
        validate_conversation_id(conversation_id)
        async with self._conversation(conversation_id):
            return await self._reconcile(conversation_id)

    async def reconcile_all(self) -> None:
        for conversation_id in self._store.pending_deletions():
            await self.reconcile(conversation_id)

    async def _reconcile(self, conversation_id: str) -> Literal["pending", "complete"]:
        deletion = self._store.deletion(conversation_id)
        if deletion is None:
            raise ArchiveRefusal("unbound")
        if deletion.complete and deletion.targets is not None:
            ids = tuple(target.session_id for target in deletion.targets)
            try:
                if await self._call(self._port.absent, ids):
                    return "complete"
            except Exception as error:
                _marker("verify", error)
                return "pending"
        if self._review_admitted(conversation_id):
            return "pending"
        if deletion.targets is None:
            binding = self._store.read(conversation_id)
            voice_id = None if binding is None else binding.session_id
            allow_missing_voice = (
                binding is not None and binding.committed is None
                and binding.pending is not None and binding.pending.fingerprint.count == 0
            )
            try:
                targets = await self._call(
                    self._port.capture_delete_targets, voice_id, allow_missing_voice,
                )
                self._store.set_delete_manifest(conversation_id, targets)
            except Exception as error:
                _marker("capture", error)
                return "pending"
            deletion = self._store.deletion(conversation_id)
            assert deletion is not None
        assert deletion.targets is not None
        for target in deletion.targets:
            try:
                await self._call(self._port.delete_target, target)
            except Exception as error:
                _marker("delete", error)
                return "pending"
        ids = tuple(target.session_id for target in deletion.targets)
        try:
            if not await self._call(self._port.absent, ids):
                return "pending"
            if self._on_complete is not None:
                await self._on_complete(conversation_id)
            self._store.mark_delete_complete(conversation_id)
        except Exception as error:
            _marker("verify", error)
            return "pending"
        return "complete"


def _marker(stage: str, error: Exception) -> None:
    outcome = (
        {"refusal": error.category} if isinstance(error, ArchiveRefusal)
        else {"failure": type(error).__name__}
    )
    print(
        "[voice-forget] "
        + json.dumps({"stage": stage, "version": 1, **outcome}, sort_keys=True),
        flush=True,
    )


__all__ = ["DeleteTarget", "VoiceForgetReconciler"]
