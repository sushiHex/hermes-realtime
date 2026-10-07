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

    def branch_copies_absent(self, session_ids: tuple[str, ...]) -> bool: ...


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
        """One conversation's failure stays its own: it never stops the others or a start."""
        for conversation_id in self._store.pending_deletions():
            try:
                await self.reconcile(conversation_id)
            except Exception as error:
                outcome: dict[str, str | int] = (
                    {"refusal": error.category} if isinstance(error, ArchiveRefusal)
                    else {"failure": type(error).__name__}
                )
                _marker(outcome | {"stage": "load", "version": 1})

    async def _reconcile(self, conversation_id: str) -> Literal["pending", "complete"]:
        """Every decision that leaves the delete pending leaves through one marker."""

        deletion = self._store.deletion(conversation_id)
        if deletion is None:
            raise ArchiveRefusal("unbound")
        evidence: dict[str, str | int] | None = None
        stage = "verify"
        try:
            if (
                deletion.complete
                and deletion.targets is not None
                and await self._remaining(deletion.targets) is None
            ):
                return "complete"
            stage = "defer"
            if self._review_admitted(conversation_id):
                evidence = {"category": "review", "count": 1, "kind": "admitted"}
                return "pending"
            if deletion.targets is None:
                stage = "capture"
                binding = self._store.read(conversation_id)
                voice_id = None if binding is None else binding.session_id
                allow_missing_voice = (
                    binding is not None and binding.committed is None
                    and binding.pending is not None and binding.pending.fingerprint.count == 0
                )
                targets = await self._call(
                    self._port.capture_delete_targets, voice_id, allow_missing_voice,
                )
                self._store.set_delete_manifest(conversation_id, targets)
                deletion = self._store.deletion(conversation_id)
                assert deletion is not None and deletion.targets is not None
            stage = "delete"
            for target in deletion.targets:
                await self._call(self._port.delete_target, target)
            stage = "verify"
            remaining = await self._remaining(deletion.targets)
            if remaining is not None:
                evidence = {"category": remaining}
                return "pending"
            if self._on_complete is not None:
                await self._on_complete(conversation_id)
            self._store.mark_delete_complete(conversation_id)
            return "complete"
        except ArchiveRefusal as refusal:
            evidence = {"refusal": refusal.category}
            return "pending"
        except Exception as error:
            evidence = {"failure": type(error).__name__}
            return "pending"
        finally:
            if evidence is not None:
                _marker(evidence | {"stage": stage, "version": 1})

    async def _remaining(self, targets: tuple[DeleteTarget, ...]) -> str | None:
        """What of the generation is left: None when nothing is, else its category.

        Hermes's delete keeps the copies ``/branch`` made of a chain session as
        independent conversations; while any remains, the delete is not complete.
        """

        ids = tuple(target.session_id for target in targets)
        if not await self._call(self._port.absent, ids):
            return "present"
        if not await self._call(self._port.branch_copies_absent, ids):
            return "branch_copies"
        return None


def _marker(evidence: dict[str, str | int]) -> None:
    print("[voice-forget] " + json.dumps(evidence, sort_keys=True), flush=True)


__all__ = ["DeleteTarget", "VoiceForgetReconciler"]
