"""Hermes integration boundary for the realtime runtime."""

from .api import HermesApiConfig, HermesApiTaskSession
from .bridge import (
    BridgeAuthenticationError,
    BridgeProtocolError,
    LocalHermesBridgeClient,
    LocalHermesBridgeServer,
)
from .completion import HermesCompletionRouter
from .plugin import (
    HermesCompletionSource,
    HermesPluginContext,
    HermesPluginDispatcher,
    HermesPluginRuntime,
    HermesRunCompletion,
)
from .sequencing import EventSequencer
from .service import (
    HermesDispatchCommand,
    HermesDispatchRejected,
    HermesIntegrationService,
    HermesWorkDispatcher,
)
from .session import SessionBinding, SessionBindings
from .worker import RealtimeHermesSession

__all__ = [
    "BridgeAuthenticationError",
    "BridgeProtocolError",
    "EventSequencer",
    "HermesApiConfig",
    "HermesApiTaskSession",
    "HermesCompletionRouter",
    "HermesCompletionSource",
    "HermesDispatchCommand",
    "HermesDispatchRejected",
    "HermesIntegrationService",
    "HermesPluginContext",
    "HermesPluginDispatcher",
    "HermesPluginRuntime",
    "HermesRunCompletion",
    "HermesWorkDispatcher",
    "LocalHermesBridgeClient",
    "LocalHermesBridgeServer",
    "RealtimeHermesSession",
    "SessionBinding",
    "SessionBindings",
]
