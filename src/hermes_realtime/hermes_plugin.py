"""Pip entry point loaded by Hermes Agent's plugin manager."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .companion.archive import ArchivePort
from .companion.host import (
    CompanionEndpoint,
    VoiceCompanionHost,
    VoiceCompanionService,
    companion_endpoint,
)
from .integration import (
    EventSequencer,
    HermesCompletionRouter,
    HermesIntegrationService,
    HermesPluginDispatcher,
    HermesPluginRuntime,
    LocalHermesBridgeServer,
    SessionBindings,
)
from .integration.bridge import VoiceArchiveHandler

_runtime: HermesPluginRuntime | None = None
_companion: VoiceCompanionHost | None = None
_lock = RLock()
_COMPANION_STORE = "voice-companion.db"
_MARKER = "[voice-companion] "


def _open_archive_port() -> ArchivePort:
    """The companion's port over this profile's Hermes database (Hermes imported lazily)."""

    from .companion.hermes_compat import HermesArchivePort, open_session_db

    return HermesArchivePort(open_session_db())


def _refuse(category: str) -> None:
    evidence = json.dumps({"refusal": category, "version": 1}, separators=(",", ":"))
    print(_MARKER + evidence, flush=True)


def register(context: object) -> None:
    """Bind this standalone plugin to Hermes's public PluginContext.

    Registration also builds the voice companion when its endpoint is configured, and
    begins its owned start; readiness is announced only once that start completes. A
    companion that cannot be owned is refused with a marker, and dispatch still registers.
    """

    global _runtime, _companion
    with _lock:
        _runtime = HermesPluginRuntime(context)
        try:
            endpoint = companion_endpoint(os.environ)
        except ValueError:
            _refuse("endpoint")
            return
        if endpoint is None:
            return
        on_unload = getattr(context, "on_unload", None)
        state = getattr(context, "state", None)
        data_dir = getattr(state, "data_dir", None)
        if not callable(on_unload) or not isinstance(data_dir, Path):
            _refuse("context")
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        companion = VoiceCompanionHost(
            store_path=data_dir / _COMPANION_STORE,
            open_port=_open_archive_port,
            bridge_factory=lambda service: _companion_bridge(endpoint, service),
        )
        try:
            companion.start()
        except RuntimeError:
            return  # Multiplexing: the refusal marker is already printed.
        _companion = companion
        on_unload(_close_companion)


def _companion_bridge(
    endpoint: CompanionEndpoint, service: VoiceCompanionService
) -> LocalHermesBridgeServer:
    return create_local_bridge(
        bindings=SessionBindings(),
        token=endpoint.token,
        port=endpoint.port,
        voice=service,
    )


def _close_companion() -> None:
    global _companion
    with _lock:
        companion, _companion = _companion, None
    if companion is not None:
        companion.close()


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
    voice: VoiceArchiveHandler | None = None,
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
    bridge_options: dict[str, Any] = {} if voice is None else {"voice": voice}
    return LocalHermesBridgeServer(
        service=service,
        completions=completions,
        completion_source=runtime,
        token=token,
        host=host,
        port=port,
        **bridge_options,
    )
