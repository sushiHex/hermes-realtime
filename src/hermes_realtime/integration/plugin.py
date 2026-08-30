"""Adapter for dispatching realtime work through Hermes's plugin API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, cast
from uuid import uuid4

from hermes_realtime.protocol import Durability

from .service import HermesDispatchCommand, HermesDispatchRejected

logger = logging.getLogger(__name__)
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class HermesPluginContext(Protocol):
    """Public Hermes plugin surface used by the realtime integration."""

    def dispatch_reserved_delegation(
        self,
        args: dict[str, object],
        delegation_id: str,
    ) -> str:
        """Dispatch background work with a trusted pre-reserved handle."""

    def register_hook(self, name: str, callback: Callable[..., None]) -> None:
        """Register a Hermes lifecycle observer."""

    def interrupt_delegation(self, delegation_id: str) -> bool:
        """Signal one exact asynchronous delegation to stop."""


@dataclass(frozen=True, slots=True)
class HermesRunCompletion:
    """Exact terminal callback for one asynchronous Hermes delegation."""

    run_id: str
    status: str
    summary: str | None
    reason: str | None
    authoritative: bool = False


class HermesCompletionSource(Protocol):
    """Thread-safe subscription surface consumed by the local bridge."""

    def subscribe_completion(
        self,
        listener: Callable[[HermesRunCompletion], None],
    ) -> Callable[[], None]:
        """Subscribe and return an idempotent unsubscribe callback."""


class HermesPluginRuntime:
    """Own the public dispatcher and translate Hermes completion hooks."""

    def __init__(self, context: object) -> None:
        legacy_required = (
            "dispatch_reserved_delegation",
            "register_hook",
            "interrupt_delegation",
        )
        missing_legacy = [
            name for name in legacy_required if not callable(getattr(context, name, None))
        ]
        has_legacy = not missing_legacy
        lifecycle_descriptor = getattr(type(context), "subagent_lifecycle", None)
        has_lifecycle = isinstance(lifecycle_descriptor, property)
        if not has_legacy and not has_lifecycle:
            raise RuntimeError(
                "Hermes PluginContext provides neither the v0.20 lifecycle API nor "
                "the complete v0.19 exact delegation compatibility APIs; missing: "
                + ", ".join(missing_legacy)
            )
        self._listeners: set[Callable[[HermesRunCompletion], None]] = set()
        self._reserved_run_ids: set[str] = set()
        self._lock = RLock()
        if has_lifecycle:
            self.dispatcher = HermesPluginDispatcher(
                None,
                unavailable_reason=(
                    "Hermes v0.20 local bridge dispatch is unavailable outside an active "
                    "parent turn; use the authenticated full-host /v1/runs adapter"
                ),
            )
        else:
            legacy_context = cast(HermesPluginContext, context)
            self.dispatcher = HermesPluginDispatcher(
                legacy_context,
                reserve=self._reserve,
                publish=self._publish,
            )
            legacy_context.register_hook("subagent_stop", self._on_subagent_stop)

    def _reserve(self, run_id: str) -> None:
        with self._lock:
            self._reserved_run_ids.add(run_id)

    def _publish(self, run_id: str) -> None:
        with self._lock:
            self._reserved_run_ids.discard(run_id)

    def subscribe_completion(
        self,
        listener: Callable[[HermesRunCompletion], None],
    ) -> Callable[[], None]:
        """Subscribe to exact async completions; return an idempotent unsubscribe."""

        with self._lock:
            self._listeners.add(listener)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.discard(listener)

        return unsubscribe

    def _on_subagent_stop(
        self,
        *,
        delegation_id: object = None,
        child_status: object = None,
        child_summary: object = None,
        **_: object,
    ) -> None:
        if (
            not isinstance(delegation_id, str)
            or _IDENTIFIER_PATTERN.fullmatch(delegation_id) is None
        ):
            return
        raw_status = child_status.strip().lower() if isinstance(child_status, str) else ""
        raw_summary = child_summary.strip() if isinstance(child_summary, str) else ""
        with self._lock:
            authoritative = delegation_id in self._reserved_run_ids
        if raw_status in {"completed", "success"}:
            if raw_summary:
                completion = HermesRunCompletion(
                    run_id=delegation_id,
                    status="completed",
                    summary=raw_summary,
                    reason=None,
                    authoritative=authoritative,
                )
            else:
                completion = HermesRunCompletion(
                    run_id=delegation_id,
                    status="failed",
                    summary=None,
                    reason="Hermes delegation completed without a summary",
                    authoritative=authoritative,
                )
        elif raw_status in {"interrupted", "cancelled", "canceled"}:
            completion = HermesRunCompletion(
                run_id=delegation_id,
                status="interrupted",
                summary=None,
                reason=raw_summary or "Hermes delegation was interrupted",
                authoritative=authoritative,
            )
        else:
            completion = HermesRunCompletion(
                run_id=delegation_id,
                status="failed",
                summary=None,
                reason=(
                    raw_summary or f"Hermes delegation ended with status {raw_status or 'unknown'}"
                ),
                authoritative=authoritative,
            )
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(completion)
            except Exception:  # noqa: BLE001 - observer failures must not break Hermes
                logger.exception("Hermes realtime completion listener failed")


class HermesPluginDispatcher:
    """Submit work through Hermes's authoritative delegate_task tool."""

    def __init__(
        self,
        context: HermesPluginContext | None,
        *,
        reserve: Callable[[str], None] | None = None,
        publish: Callable[[str], None] | None = None,
        unavailable_reason: str | None = None,
    ) -> None:
        self._context = context
        self._reserve = reserve or (lambda _: None)
        self._publish = publish or (lambda _: None)
        self._unavailable_reason = unavailable_reason

    def publish(self, run_id: str) -> None:
        """Release a reservation after the service publishes active ownership."""

        self._publish(run_id)

    async def cancel(self, run_id: str) -> bool:
        """Interrupt the exact asynchronous delegation acknowledged as run_id."""

        if self._context is None:
            return False
        return await asyncio.to_thread(self._context.interrupt_delegation, run_id)

    async def dispatch(self, command: HermesDispatchCommand) -> str:
        if self._context is None:
            raise HermesDispatchRejected(
                self._unavailable_reason or "Hermes plugin dispatch is unavailable"
            )
        if command.durability is Durability.DURABLE:
            raise HermesDispatchRejected(
                "durable dispatch is not supported by Hermes background delegation"
            )
        reserved_run_id = f"deleg_{uuid4().hex}"
        self._reserve(reserved_run_id)
        args: dict[str, object] = {
            "goal": command.objective,
            "context": (
                "Dispatched by hermes-realtime. "
                f"Hermes session_id={command.session_id}; "
                f"realtime task_id={command.task_id}."
            ),
            "background": True,
        }
        try:
            raw_result = await asyncio.to_thread(
                self._context.dispatch_reserved_delegation,
                args,
                reserved_run_id,
            )
        except BaseException:
            self._publish(reserved_run_id)
            raise
        try:
            payload = json.loads(raw_result)
        except (TypeError, json.JSONDecodeError) as exc:
            self._publish(reserved_run_id)
            raise HermesDispatchRejected("Hermes returned an invalid dispatch response") from exc

        if not isinstance(payload, dict) or payload.get("status") != "dispatched":
            self._publish(reserved_run_id)
            raise HermesDispatchRejected("Hermes did not accept background dispatch")
        delegation_id = payload.get("delegation_id")
        if not isinstance(delegation_id, str) or delegation_id != reserved_run_id:
            self._publish(reserved_run_id)
            raise HermesDispatchRejected("Hermes accepted dispatch without a delegation ID")
        return delegation_id
