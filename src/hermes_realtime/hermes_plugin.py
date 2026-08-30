"""Pip entry point loaded by Hermes Agent's plugin manager."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from .integration import (
    EventSequencer,
    HermesCompletionRouter,
    HermesIntegrationService,
    HermesPluginDispatcher,
    HermesPluginRuntime,
    LocalHermesBridgeServer,
    SessionBindings,
)

_runtime: HermesPluginRuntime | None = None
_lock = RLock()


def register(context: object) -> None:
    """Bind this standalone plugin to Hermes's public PluginContext."""

    global _runtime
    with _lock:
        _runtime = HermesPluginRuntime(context)


def get_dispatcher() -> HermesPluginDispatcher:
    """Return the adapter initialized by Hermes's plugin loader."""

    with _lock:
        if _runtime is None:
            raise RuntimeError("hermes-realtime plugin has not been registered")
        return _runtime.dispatcher


def get_runtime() -> HermesPluginRuntime:
    """Return the initialized plugin runtime and completion source."""

    with _lock:
        if _runtime is None:
            raise RuntimeError("hermes-realtime plugin has not been registered")
        return _runtime


def create_local_bridge(
    *,
    bindings: SessionBindings,
    token: str,
    host: str = "127.0.0.1",
    port: int = 0,
    event_id_factory: Callable[[], str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> LocalHermesBridgeServer:
    """Build an authenticated bridge around the registered Hermes runtime."""

    runtime = get_runtime()
    sequencer = EventSequencer()
    make_event_id = event_id_factory or (lambda: f"evt_{uuid4().hex}")
    now = clock or (lambda: datetime.now(UTC))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=runtime.dispatcher,
        canceller=runtime.dispatcher,
        event_id_factory=make_event_id,
        clock=now,
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=make_event_id,
        clock=now,
    )
    return LocalHermesBridgeServer(
        service=service,
        completions=completions,
        completion_source=runtime,
        token=token,
        host=host,
        port=port,
    )
