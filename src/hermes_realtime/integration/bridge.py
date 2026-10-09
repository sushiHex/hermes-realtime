"""Authenticated loopback IPC for the realtime worker and Hermes plugin."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import re
import secrets
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, cast

from pydantic import ValidationError

from hermes_realtime.protocol import (
    BRIDGE_CAPABILITIES,
    BRIDGE_PROTOCOL_VERSION,
    MUTUAL_AUTH_CAPABILITY,
    RUNTIME_ATTESTATION_CAPABILITY,
    VOICE_ARCHIVE_CAPABILITY,
    VOICE_EVENT_TYPES,
    VOICE_FORGET_CAPABILITY,
    VOICE_MEMORY_CAPABILITY,
    VOICE_REVIEW_CAPABILITY,
    ControlCancelAcknowledgedEvent,
    ControlCancelEvent,
    ProtocolEvent,
    RuntimeAttestation,
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceForgetAckEvent,
    VoiceForgetEvent,
    VoiceForgetRefusedEvent,
    VoiceMemoryEvent,
    VoiceMemoryRefusedEvent,
    VoiceMemorySnapshotEvent,
    VoiceReviewAckEvent,
    VoiceReviewEvent,
    VoiceReviewRefusedEvent,
    WorkCompletedEvent,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
    parse_event,
    parse_voice_event,
)

from .completion import HermesCompletionRouter
from .plugin import HermesCompletionSource, HermesRunCompletion
from .service import HermesIntegrationService
from .session import SessionBinding

_MAX_LINE_BYTES = 64 * 1024
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
# The hello carries no secret: each side proves the token with an HMAC over the handshake.
_HELLO_FIELDS = frozenset({"participant_id", "protocol_version", "capabilities", "client_nonce"})
_WELCOME_FIELDS = frozenset({"ok", "protocol_version", "capabilities", "server_nonce", "proof"})
# A welcome field that accompanies exactly one negotiated capability.
_WELCOME_EXTRAS = {
    "review_interval": VOICE_REVIEW_CAPABILITY,
    "runtime": RUNTIME_ATTESTATION_CAPABILITY,
}
# 32 random bytes from ``secrets``, lowercase hex; an HMAC-SHA256 proof has the same shape.
_NONCE_PATTERN = re.compile(r"[0-9a-f]{64}")
_HELLO_MARKER = "[hermes-bridge-hello] "
_WELCOME_MARKER = "[hermes-bridge-welcome] "
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Handshake:
    """Everything one handshake's proofs bind, so no part can be downgraded or spliced."""

    participant_id: str
    client_nonce: str
    server_nonce: str
    requested: frozenset[str]
    negotiated: frozenset[str]
    # The welcome's metadata exactly as sent: ``review_interval`` and ``runtime``.
    metadata: dict[str, object]

    def proof(self, token: str, role: str) -> str:
        """The proof of one role: ``server`` (welcome), ``client`` or ``accept`` (final)."""

        transcript = json.dumps(
            {
                "client_nonce": self.client_nonce,
                "metadata": self.metadata,
                "negotiated": sorted(self.negotiated),
                "participant_id": self.participant_id,
                "protocol_version": BRIDGE_PROTOCOL_VERSION,
                "requested": sorted(self.requested),
                "role": role,
                "server_nonce": self.server_nonce,
            },
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("ascii")
        return hmac.new(
            token.encode("utf-8", errors="surrogatepass"), transcript, hashlib.sha256
        ).hexdigest()

    def verifies(self, token: str, role: str, received: object) -> bool:
        """Constant-time check of a received proof, refusing anything of another shape."""

        return (
            type(received) is str
            and _NONCE_PATTERN.fullmatch(received) is not None
            and hmac.compare_digest(
                self.proof(token, role).encode("ascii"), received.encode("ascii")
            )
        )


def _nonce() -> str:
    return secrets.token_hex(32)


async def _read_handshake_line(reader: asyncio.StreamReader) -> Any:
    """One handshake line as JSON; a closed or garbled peer fails authentication."""

    try:
        return json.loads(await reader.readline())
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise BridgeAuthenticationError("bridge authentication failed") from error


def _marker(prefix: str, refusal: str) -> None:
    """The rejection category only: never the token, a nonce, a proof or the participant."""
    print(prefix + json.dumps({"refusal": refusal, "version": 1}, separators=(",", ":")),
          flush=True)


class VoiceArchiveHandler(Protocol):
    """The companion side of ``voice_archive``: answer a batch, or None when unknown.

    None means the outcome is unknown (the commit may or may not have landed); the bridge
    then closes the connection without a reply, and realtime resends the frozen batch.
    """

    async def archive(
        self, event: VoiceArchiveEvent
    ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent | None: ...

    @property
    def review_interval(self) -> int | None: ...

    async def review(
        self, event: VoiceReviewEvent
    ) -> VoiceReviewAckEvent | VoiceReviewRefusedEvent | None: ...

    @property
    def memory_available(self) -> bool: ...

    def memory(
        self, event: VoiceMemoryEvent,
    ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]: ...

    @property
    def forget_available(self) -> bool: ...

    async def forget(
        self, event: VoiceForgetEvent,
    ) -> VoiceForgetAckEvent | VoiceForgetRefusedEvent | None: ...


def _capabilities(value: object) -> frozenset[str] | None:
    """An exact list of distinct known capabilities, or None."""
    if type(value) is not list or any(type(item) is not str for item in value):
        return None
    if len(set(value)) != len(value) or not set(value) <= BRIDGE_CAPABILITIES:
        return None
    return frozenset(value)


class BridgeAuthenticationError(Exception):
    """The peer did not prove possession of the configured bridge token."""


class BridgeProtocolError(Exception):
    """The peer sent malformed or unsupported bridge data."""


@dataclass(eq=False, slots=True)
class _Connection:
    writer: asyncio.StreamWriter
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_sequence: int = -1


@dataclass(frozen=True, slots=True)
class _Route:
    connection: _Connection
    binding: SessionBinding


def _event_type(raw: bytes) -> str | None:
    """The ``type`` of a JSON object line, or None; the full parse happens after routing."""
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None
    kind = document.get("type") if type(document) is dict else None
    return kind if type(kind) is str else None


class LocalHermesBridgeServer:
    """Loopback-only event server hosted inside the Hermes plugin process."""

    def __init__(
        self,
        *,
        service: HermesIntegrationService,
        completions: HermesCompletionRouter,
        token: str,
        completion_source: HermesCompletionSource | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_pending_completions: int = 256,
        authentication_timeout: float = 5.0,
        max_connections: int = 32,
        shutdown_timeout: float = 5.0,
        voice: VoiceArchiveHandler | None = None,
        runtime: RuntimeAttestation | None = None,
    ) -> None:
        if len(token) < 24:
            raise ValueError("bridge token must contain at least 24 characters")
        if not 0 <= port <= 65535:
            raise ValueError("bridge port is outside the valid range")
        if max_pending_completions < 1:
            raise ValueError("max_pending_completions must be positive")
        if authentication_timeout <= 0:
            raise ValueError("authentication_timeout must be positive")
        if max_connections < 1:
            raise ValueError("max_connections must be positive")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")
        self._service = service
        self._completions = completions
        self._completions.set_expiration_callback(service.expire_terminal_replay)
        self._token = token
        self._completion_source = completion_source
        self._host = host
        self._configured_port = port
        self._port = port
        self._max_pending_completions = max_pending_completions
        self._authentication_timeout = authentication_timeout
        self._max_connections = max_connections
        self._shutdown_timeout = shutdown_timeout
        # The bridge starts only once its companion is ready, so it offers what it holds.
        self._voice = voice
        interval = getattr(voice, "review_interval", None)
        if interval is not None and (type(interval) is not int or not 1 <= interval <= 1000):
            raise ValueError("review interval must be an exact integer from 1 to 1000")
        self._review_interval: int | None = interval
        if runtime is not None and type(runtime) is not RuntimeAttestation:
            raise TypeError("runtime must be an exact RuntimeAttestation or None")
        self._runtime = runtime
        self._offered = (
            frozenset({VOICE_ARCHIVE_CAPABILITY, VOICE_REVIEW_CAPABILITY})
            if voice is not None
            and interval is not None
            and callable(getattr(voice, "review", None))
            else frozenset({VOICE_ARCHIVE_CAPABILITY})
            if voice is not None
            else frozenset()
        ) | (frozenset({RUNTIME_ATTESTATION_CAPABILITY}) if runtime is not None else frozenset())
        self._server: asyncio.Server | None = None
        if (getattr(voice, "memory_available", False) is True
                and callable(getattr(voice, "memory", None))):
            self._offered |= frozenset({VOICE_MEMORY_CAPABILITY})
        if (getattr(voice, "forget_available", False) is True
                and callable(getattr(voice, "forget", None))):
            self._offered |= frozenset({VOICE_FORGET_CAPABILITY})
        # Required, never merely offered: a hello without it is refused.
        self._offered |= frozenset({MUTUAL_AUTH_CAPABILITY})
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe_completion: Callable[[], None] | None = None
        self._completion_tasks: set[asyncio.Task[WorkCompletedEvent | None]] = set()
        self._completion_ingress: set[str] = set()
        self._authoritative_ingress: set[str] = set()
        self._completion_ingress_lock = threading.Lock()
        self._routes: dict[str, _Route] = {}
        self._connections: set[_Connection] = set()
        self._handler_tasks: set[asyncio.Task[None]] = set()
        self._pending: OrderedDict[str, tuple[WorkTerminalStatus | str, str | None, str | None]] = (
            OrderedDict()
        )
        self._terminal_runs: OrderedDict[str, None] = OrderedDict()
        self._completing_runs: set[str] = set()
        self._state_lock = asyncio.Lock()
        self._registration_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._server_generation = 0
        self._closed_permanently = False

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Start listening after enforcing the loopback-only boundary."""

        async with self._lifecycle_lock:
            await self._start_locked()

    async def _start_locked(self) -> None:
        """Start while holding the lifecycle serialization boundary."""

        if self._closed_permanently:
            raise RuntimeError("a closed Hermes bridge instance cannot be restarted")
        if not ipaddress.ip_address(self._host).is_loopback:
            raise ValueError("Hermes bridge must bind to a loopback address")
        if self._server is not None:
            return
        async with self._state_lock:
            self._closing = False
            self._server_generation += 1
            generation = self._server_generation
        self._server = await asyncio.start_server(
            lambda reader, writer: self._handle_connection(
                reader,
                writer,
                generation,
            ),
            self._host,
            self._configured_port,
            limit=_MAX_LINE_BYTES,
        )
        sockets: list[Any] = list(self._server.sockets or ())
        if not sockets:
            server, self._server = self._server, None
            server.close()
            await server.wait_closed()
            async with self._state_lock:
                self._closing = True
            raise RuntimeError("Hermes bridge started without a listening socket")
        self._port = int(sockets[0].getsockname()[1])
        self._loop = asyncio.get_running_loop()
        if self._completion_source is not None:
            self._unsubscribe_completion = self._completion_source.subscribe_completion(
                self._on_completion
            )

    async def close(self) -> None:
        """Stop accepting peers and bound shutdown of all bridge tasks."""

        async with self._lifecycle_lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        """Close while holding the lifecycle serialization boundary."""

        self._closed_permanently = True
        async with self._state_lock:
            self._closing = True
        unsubscribe, self._unsubscribe_completion = self._unsubscribe_completion, None
        if unsubscribe is not None:
            unsubscribe()
        self._loop = None
        server, self._server = self._server, None
        if server is not None:
            server.close()

        async with self._state_lock:
            connections = tuple(self._connections)
            tasks: tuple[asyncio.Task[object], ...] = tuple(self._completion_tasks) + tuple(
                self._handler_tasks
            )
            self._connections.clear()
            self._handler_tasks.clear()
            self._routes.clear()
            self._pending.clear()
            self._terminal_runs.clear()
            self._completing_runs.clear()
        for connection in connections:
            connection.writer.close()
        for task in tasks:
            task.cancel()
        if server is not None:
            await server.wait_closed()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self._shutdown_timeout)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                logger.warning(
                    "Hermes bridge abandoned %d task(s) that ignored cancellation",
                    len(pending),
                )
        await asyncio.gather(
            *(self._wait_writer_closed(connection.writer) for connection in connections),
            return_exceptions=True,
        )

    def _on_completion(self, completion: HermesRunCompletion) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        active_run_ids = self._service.active_run_ids_now()
        with self._completion_ingress_lock:
            if completion.run_id in self._completion_ingress:
                return
            if completion.authoritative or completion.run_id in active_run_ids:
                unrelated_run_id = next(
                    (
                        run_id
                        for run_id in self._completion_ingress
                        if run_id not in active_run_ids
                        and run_id not in self._authoritative_ingress
                    ),
                    None,
                )
                if (
                    len(self._completion_ingress) >= self._max_pending_completions
                    and unrelated_run_id is not None
                ):
                    self._completion_ingress.remove(unrelated_run_id)
                    self._authoritative_ingress.discard(unrelated_run_id)
            elif (
                sum(
                    run_id not in active_run_ids and run_id not in self._authoritative_ingress
                    for run_id in self._completion_ingress
                )
                >= self._max_pending_completions
            ):
                return
            self._completion_ingress.add(completion.run_id)
            if completion.authoritative:
                self._authoritative_ingress.add(completion.run_id)
        loop.call_soon_threadsafe(self._schedule_completion, completion)

    def _schedule_completion(self, completion: HermesRunCompletion) -> None:
        with self._completion_ingress_lock:
            if completion.run_id not in self._completion_ingress:
                return
        if self._server is None:
            with self._completion_ingress_lock:
                self._completion_ingress.discard(completion.run_id)
                self._authoritative_ingress.discard(completion.run_id)
            return
        task = asyncio.create_task(
            self.complete(
                completion.run_id,
                status=completion.status,
                summary=completion.summary,
                reason=completion.reason,
            )
        )
        self._completion_tasks.add(task)
        task.add_done_callback(
            lambda completed_task: self._completion_finished(
                completion.run_id,
                completed_task,
            )
        )

    def _completion_finished(
        self,
        run_id: str,
        task: asyncio.Task[WorkCompletedEvent | None],
    ) -> None:
        self._completion_tasks.discard(task)
        with self._completion_ingress_lock:
            self._completion_ingress.discard(run_id)
            self._authoritative_ingress.discard(run_id)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Hermes bridge completion routing failed: %s",
                task.exception(),
            )

    async def complete(
        self,
        run_id: str,
        *,
        status: WorkTerminalStatus | str,
        summary: str | None = None,
        reason: str | None = None,
    ) -> WorkCompletedEvent | None:
        """Record exact terminal evidence and emit it when a live route exists."""

        summary = self._bound_completion_text(summary)
        reason = self._bound_completion_text(reason)
        claimed, route = await self._claim_completion(
            run_id,
            status=status,
            summary=summary,
            reason=reason,
        )
        if not claimed:
            return None
        try:
            if route is None:
                return await self._finish_completion(
                    run_id,
                    status=status,
                    summary=summary,
                    reason=reason,
                    route=None,
                )
            async with route.connection.event_lock:
                return await self._finish_completion(
                    run_id,
                    status=status,
                    summary=summary,
                    reason=reason,
                    route=route,
                )
        except BaseException:
            async with self._state_lock:
                self._completing_runs.discard(run_id)
            raise

    async def _claim_completion(
        self,
        run_id: str,
        *,
        status: WorkTerminalStatus | str,
        summary: str | None,
        reason: str | None,
    ) -> tuple[bool, _Route | None]:
        async with self._registration_lock:
            async with self._state_lock:
                if (
                    run_id in self._terminal_runs
                    or run_id in self._completing_runs
                    or run_id in self._pending
                ):
                    return False, None
                route = self._routes.get(run_id)
                if route is not None:
                    self._completing_runs.add(run_id)
                    return True, route
            tracked = await self._completions.is_tracked(run_id)
            active_run_ids = await self._service.active_run_ids()
            async with self._state_lock:
                if run_id in self._terminal_runs or run_id in self._completing_runs:
                    return False, None
                route = self._routes.get(run_id)
                if route is None and not tracked:
                    if run_id in self._pending:
                        return False, None
                    self._pending[run_id] = (status, summary, reason)
                    self._pending.move_to_end(run_id)
                    while (
                        sum(
                            pending_run_id not in active_run_ids for pending_run_id in self._pending
                        )
                        > self._max_pending_completions
                    ):
                        unrelated_run_id = next(
                            pending_run_id
                            for pending_run_id in self._pending
                            if pending_run_id not in active_run_ids
                        )
                        del self._pending[unrelated_run_id]
                    return False, None
                self._completing_runs.add(run_id)
                return True, route

    async def _finish_completion(
        self,
        run_id: str,
        *,
        status: WorkTerminalStatus | str,
        summary: str | None,
        reason: str | None,
        route: _Route | None,
    ) -> WorkCompletedEvent | None:
        try:
            event = await self._completions.complete(
                run_id,
                status=status,
                summary=summary,
                reason=reason,
            )
        except Exception:
            async with self._state_lock:
                self._completing_runs.discard(run_id)
            raise
        async with self._state_lock:
            self._completing_runs.discard(run_id)
            self._routes.pop(run_id, None)
            self._terminal_runs[run_id] = None
            self._terminal_runs.move_to_end(run_id)
            while len(self._terminal_runs) > self._max_pending_completions:
                self._terminal_runs.popitem(last=False)
        try:
            sent = route is not None and await self._send_bound(route, event)
        finally:
            await self._mark_terminal_cancellation_safe(run_id)
        return event if sent else None

    async def _mark_terminal_cancellation_safe(self, run_id: str) -> None:
        settlement = asyncio.create_task(self._service.mark_terminal(run_id))
        try:
            await asyncio.shield(settlement)
        except asyncio.CancelledError:
            # Terminal evidence already committed. Do not let one or repeated
            # caller cancellations strand authoritative active-run ownership.
            while not settlement.done():
                try:
                    await asyncio.shield(settlement)
                except asyncio.CancelledError:
                    continue
            settlement.result()
            raise

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        generation: int,
    ) -> None:
        connection = _Connection(writer)
        handler_task = asyncio.current_task()
        if handler_task is None:
            writer.close()
            await self._wait_writer_closed(writer)
            return
        async with self._state_lock:
            admitted = (
                not self._closing
                and generation == self._server_generation
                and len(self._connections) < self._max_connections
            )
            if admitted:
                self._connections.add(connection)
                self._handler_tasks.add(handler_task)
        if not admitted:
            writer.close()
            await self._wait_writer_closed(writer)
            return

        try:
            participant_id, negotiated = await self._authenticate(reader, connection)
            while raw := await reader.readline():
                if _event_type(raw) == "voice_memory":
                    await self._route_memory(connection, negotiated, raw, reader)
                    return
                if _event_type(raw) in VOICE_EVENT_TYPES:
                    await self._route_voice(connection, negotiated, raw)
                    continue
                event = parse_event(raw)
                pending = None
                async with connection.event_lock:
                    request_binding = self._service.current_binding(participant_id)
                    if request_binding is None:
                        raise BridgeProtocolError("participant binding is no longer active")
                    response_route = _Route(connection, request_binding)
                    if isinstance(event, ControlCancelEvent):
                        cancel_acknowledgment = await self._service.cancel(
                            participant_id,
                            event,
                        )
                        await self._send_bound(response_route, cancel_acknowledgment)
                        continue
                    if not isinstance(event, WorkDispatchRequestedEvent):
                        raise BridgeProtocolError(
                            "bridge accepts only dispatch or cancellation requests"
                        )
                    acknowledgment = await self._service.dispatch(participant_id, event)
                    retained_terminal = None
                    run_id = acknowledgment.payload.run_id
                    if acknowledgment.payload.accepted and run_id is not None:
                        async with self._registration_lock:
                            await self._completions.track(acknowledgment)
                            binding = await self._service.binding_for_run(run_id)
                            if binding is None:
                                raise BridgeProtocolError(
                                    "acknowledged Hermes run has no admitted binding"
                                )
                            response_route = _Route(connection, binding)
                            retained_terminal = await self._completions.completed(run_id)
                            if retained_terminal is None:
                                async with self._state_lock:
                                    self._routes[run_id] = response_route
                                    pending = self._pending.get(run_id)
                    emitted = await self._send_bound(response_route, acknowledgment)
                    if not emitted:
                        continue
                    if retained_terminal is not None:
                        await self._send_bound(response_route, retained_terminal)
                if pending is not None:
                    assert run_id is not None
                    async with self._state_lock:
                        if self._pending.get(run_id) == pending:
                            del self._pending[run_id]
                        else:
                            pending = None
                if pending is not None:
                    assert run_id is not None
                    status, summary, reason = pending
                    await self.complete(
                        run_id,
                        status=status,
                        summary=summary,
                        reason=reason,
                    )
        except (
            asyncio.IncompleteReadError,
            TimeoutError,
            BridgeAuthenticationError,
            BridgeProtocolError,
            ConnectionError,
            PermissionError,
            ValueError,
        ):
            pass
        finally:
            async with self._state_lock:
                self._connections.discard(connection)
                self._handler_tasks.discard(handler_task)
                stale = [
                    run_id
                    for run_id, route in self._routes.items()
                    if route.connection is connection
                ]
                for run_id in stale:
                    del self._routes[run_id]
            writer.close()
            await self._wait_writer_closed(writer)

    async def _authenticate(
        self,
        reader: asyncio.StreamReader,
        connection: _Connection,
    ) -> tuple[str, frozenset[str]]:
        """Prove the token first, then require the client's proof before routing anything.

        The welcome carries the server's proof over the whole handshake; the client answers
        with its own, and only then does the server send its final authenticated acceptance.
        """

        # The category of the stage in progress: every exit short of acceptance, whatever
        # raised it, leaves exactly one marker with it from the ``finally`` below.
        refusal: str | None = "shape"
        try:
            async with asyncio.timeout(self._authentication_timeout):
                raw = await reader.readline()
                try:
                    hello = json.loads(raw)
                    if type(hello) is not dict or set(hello) != _HELLO_FIELDS:
                        # A hello that still carries a token is refused here, unanswered by it.
                        raise TypeError("the hello must carry exactly the 0.3 fields")
                    participant_id = hello["participant_id"]
                except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
                    await self._send_json(connection, {"ok": False})
                    raise BridgeAuthenticationError("invalid bridge handshake") from exc
                requested = _capabilities(hello["capabilities"])
                client_nonce = hello["client_nonce"]
                rejected: str | None = None
                if (
                    not isinstance(participant_id, str)
                    or _IDENTIFIER_PATTERN.fullmatch(participant_id) is None
                ):
                    rejected = "participant"
                elif (
                    type(hello["protocol_version"]) is not str
                    or hello["protocol_version"] != BRIDGE_PROTOCOL_VERSION
                ):
                    rejected = "version"
                elif requested is None or MUTUAL_AUTH_CAPABILITY not in requested:
                    rejected = "capability"
                elif (
                    type(client_nonce) is not str
                    or _NONCE_PATTERN.fullmatch(client_nonce) is None
                ):
                    rejected = "nonce"
                if rejected is not None:
                    refusal = rejected
                    await self._send_json(connection, {"ok": False})
                    raise BridgeAuthenticationError("bridge authentication failed")
                # From here the client has only to prove the token.
                refusal = "abandoned"
                assert requested is not None  # Refused above otherwise.
                negotiated = requested & self._offered
                metadata: dict[str, object] = {}
                if VOICE_REVIEW_CAPABILITY in negotiated:
                    metadata["review_interval"] = self._review_interval
                if RUNTIME_ATTESTATION_CAPABILITY in negotiated:
                    assert self._runtime is not None  # Offered only with an attestation.
                    metadata["runtime"] = self._runtime.model_dump(mode="json")
                handshake = _Handshake(
                    participant_id, client_nonce, _nonce(), requested, negotiated, metadata
                )
                await self._send_json(connection, {
                    "ok": True,
                    "protocol_version": BRIDGE_PROTOCOL_VERSION,
                    "capabilities": sorted(negotiated),
                    "server_nonce": handshake.server_nonce,
                    "proof": handshake.proof(self._token, "server"),
                    **metadata,
                })
                raw = await reader.readline()
                if not raw:
                    # The client walked away, as one does on refusing the welcome.
                    raise BridgeAuthenticationError("bridge authentication was abandoned")
                try:
                    answer = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
                    answer = None
                if (
                    type(answer) is not dict
                    or set(answer) != {"proof"}
                    or not handshake.verifies(self._token, "client", answer["proof"])
                ):
                    # Nothing is routed for a client that cannot prove the token.
                    refusal = "proof"
                    raise BridgeAuthenticationError("bridge authentication failed")
                await self._send_json(
                    connection, {"accepted": handshake.proof(self._token, "accept")}
                )
                refusal = None
        except TimeoutError:
            refusal = "deadline"
            raise
        finally:
            if refusal is not None:
                _marker(_HELLO_MARKER, refusal)
        return participant_id, negotiated

    async def _route_voice(
        self, connection: _Connection, negotiated: frozenset[str], raw: bytes
    ) -> None:
        """Answer private voice traffic without routing it to work or speech."""

        voice = self._voice
        event = parse_voice_event(raw)
        if voice is None:
            raise BridgeProtocolError("voice capability was not negotiated")
        async with connection.event_lock:
            reply: (
                VoiceArchiveAckEvent | VoiceArchiveRefusedEvent
                | VoiceReviewAckEvent | VoiceReviewRefusedEvent
                | VoiceForgetAckEvent | VoiceForgetRefusedEvent | None
            )
            if type(event) is VoiceArchiveEvent and VOICE_ARCHIVE_CAPABILITY in negotiated:
                reply = await voice.archive(event)
            elif type(event) is VoiceReviewEvent and VOICE_REVIEW_CAPABILITY in negotiated:
                reply = await voice.review(event)
            elif type(event) is VoiceForgetEvent and VOICE_FORGET_CAPABILITY in negotiated:
                reply = await voice.forget(event)
            else:
                raise BridgeProtocolError("voice request was not negotiated")
            if reply is None:
                raise BridgeProtocolError("the voice outcome is unknown")
            await self._send_bytes(connection, reply.model_dump_json().encode("utf-8") + b"\n")

    async def _route_memory(
        self, connection: _Connection, negotiated: frozenset[str], raw: bytes,
        reader: asyncio.StreamReader,
    ) -> None:
        event = parse_voice_event(raw)
        voice = self._voice
        if (VOICE_MEMORY_CAPABILITY not in negotiated or voice is None
                or type(event) is not VoiceMemoryEvent):
            raise BridgeProtocolError("memory capability was not negotiated")

        async def publish() -> None:
            async for reply in voice.memory(event):
                if (type(reply) not in (VoiceMemorySnapshotEvent, VoiceMemoryRefusedEvent)
                        or reply.conversation_id != event.conversation_id
                        or reply.generation != event.generation):
                    raise BridgeProtocolError("invalid memory response")
                await self._send_bytes(connection, reply.model_dump_json().encode("utf-8") + b"\n")

        producer = asyncio.create_task(publish())
        self._handler_tasks.add(producer)

        def forget_producer(task: asyncio.Task[None]) -> None:
            self._handler_tasks.discard(task)
            if not task.cancelled():
                task.exception()

        producer.add_done_callback(forget_producer)
        disconnected = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait(
                (producer, disconnected), return_when=asyncio.FIRST_COMPLETED,
            )
            if producer in done:
                producer.result()
        finally:
            producer.cancel()
            disconnected.cancel()
            joined, pending = await asyncio.wait(
                (producer, disconnected), timeout=self._shutdown_timeout,
            )
            for task in joined:
                if not task.cancelled():
                    task.exception()
            if pending:
                logger.warning(
                    "[voice-memory-stream] %s",
                    json.dumps(
                        {"refusal": "shutdown_deadline", "tasks": len(pending), "version": 1},
                        separators=(",", ":"), sort_keys=True,
                    ),
                )
                raise BridgeProtocolError("memory subscription did not stop")

    async def _send_bound(self, route: _Route, event: ProtocolEvent) -> bool:
        async with route.connection.write_lock:
            if event.sequence <= route.connection.last_sequence:
                event = await self._completions.resequence(
                    event,
                    after=route.connection.last_sequence,
                )
            payload = event.model_dump_json().encode("utf-8") + b"\n"
            self._validate_payload_size(payload)
            emitted = self._service.emit_if_binding_current(
                route.binding,
                lambda: route.connection.writer.write(payload),
            )
            if not emitted:
                return False
            await route.connection.writer.drain()
            route.connection.last_sequence = event.sequence
            return True

    async def _send_json(self, connection: _Connection, value: object) -> None:
        await self._send_bytes(
            connection,
            json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n",
        )

    async def _send_bytes(self, connection: _Connection, payload: bytes) -> None:
        self._validate_payload_size(payload)
        async with connection.write_lock:
            connection.writer.write(payload)
            await connection.writer.drain()

    @staticmethod
    def _validate_payload_size(payload: bytes) -> None:
        if len(payload) > _MAX_LINE_BYTES:
            raise BridgeProtocolError("Hermes bridge event exceeds the line limit")

    @staticmethod
    def _bound_completion_text(value: str | None) -> str | None:
        # Two independently escaped fields must still fit one 64-KiB NDJSON line.
        return None if value is None else value[:4096]

    async def _wait_writer_closed(self, writer: asyncio.StreamWriter) -> None:
        with suppress(asyncio.TimeoutError, ConnectionError):
            await asyncio.wait_for(
                writer.wait_closed(),
                timeout=self._shutdown_timeout,
            )


class LocalHermesBridgeClient:
    """Persistent loopback client used by the realtime conversation worker."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._capabilities: frozenset[str] = frozenset()
        self._review_interval: int | None = None
        self._runtime: RuntimeAttestation | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        """The capabilities the companion advertised to this connection."""
        return self._capabilities

    @property
    def runtime(self) -> RuntimeAttestation | None:
        """What the companion's process attested it loaded, when negotiated."""
        return self._runtime

    @property
    def review_interval(self) -> int | None:
        return self._review_interval

    @classmethod
    async def connect(
        cls,
        *,
        host: str,
        port: int,
        token: str,
        participant_id: str,
        capabilities: tuple[str, ...] = (),
        handshake_timeout: float = 10.0,
    ) -> LocalHermesBridgeClient:
        """Connect, and return only after both sides proved the token and the server accepted.

        The token never crosses the wire. Nothing is sent after the hello, and nothing the
        welcome carries is accepted, until the server's proof over the whole handshake
        verifies; the server's final acceptance is itself a proof.
        """

        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Hermes bridge client must connect to loopback")
        requested = _capabilities(list(capabilities)) if type(capabilities) is tuple else None
        if requested is None:
            raise ValueError("capabilities must be a tuple of distinct known capabilities")
        if (
            type(handshake_timeout) not in (int, float)
            or not math.isfinite(handshake_timeout)
            or handshake_timeout <= 0
        ):
            raise ValueError("handshake_timeout must be a positive finite number")
        requested |= frozenset({MUTUAL_AUTH_CAPABILITY})
        client: LocalHermesBridgeClient | None = None
        refusal: str | None = None
        try:
            async with asyncio.timeout(handshake_timeout):
                reader, writer = await asyncio.open_connection(
                    host,
                    port,
                    limit=_MAX_LINE_BYTES,
                )
                client = cls(reader, writer)
                client_nonce = _nonce()
                await client._send_json(
                    {
                        "participant_id": participant_id,
                        "protocol_version": BRIDGE_PROTOCOL_VERSION,
                        "capabilities": sorted(requested),
                        "client_nonce": client_nonce,
                    }
                )
                refusal = "shape"
                response = await _read_handshake_line(reader)
                fields = frozenset(response) if type(response) is dict else frozenset()
                offered = (
                    _capabilities(response["capabilities"]) if "capabilities" in fields else None
                )
                # Each extra is present exactly when its capability was negotiated.
                extras = (
                    frozenset()
                    if offered is None
                    else frozenset(
                        extra for extra, capability in _WELCOME_EXTRAS.items()
                        if capability in offered
                    )
                )
                interval = response.get("review_interval") if type(response) is dict else None
                if (
                    offered is None
                    or fields != _WELCOME_FIELDS | extras
                    or response["ok"] is not True
                    or response["protocol_version"] != BRIDGE_PROTOCOL_VERSION
                    or not offered <= requested
                    or MUTUAL_AUTH_CAPABILITY not in offered
                    or type(response["server_nonce"]) is not str
                    or _NONCE_PATTERN.fullmatch(response["server_nonce"]) is None
                    or (
                        VOICE_REVIEW_CAPABILITY in offered
                        and (type(interval) is not int or not 1 <= interval <= 1000)
                    )
                ):
                    raise BridgeAuthenticationError("bridge authentication failed")
                handshake = _Handshake(
                    participant_id,
                    client_nonce,
                    response["server_nonce"],
                    requested,
                    offered,
                    {extra: response[extra] for extra in sorted(extras)},
                )
                refusal = "proof"
                if not handshake.verifies(token, "server", response["proof"]):
                    raise BridgeAuthenticationError("bridge authentication failed")
                # Only now is anything the welcome carries believed.
                refusal = "runtime"
                runtime = None
                if RUNTIME_ATTESTATION_CAPABILITY in offered:
                    try:
                        runtime = RuntimeAttestation.model_validate(response["runtime"])
                    except ValidationError as error:
                        raise BridgeAuthenticationError(
                            "bridge authentication failed"
                        ) from error
                await client._send_json({"proof": handshake.proof(token, "client")})
                refusal = "acceptance"
                accepted = await _read_handshake_line(reader)
                if (
                    type(accepted) is not dict
                    or set(accepted) != {"accepted"}
                    or not handshake.verifies(token, "accept", accepted["accepted"])
                ):
                    raise BridgeAuthenticationError("bridge authentication failed")
                refusal = None
            client._capabilities = offered - {MUTUAL_AUTH_CAPABILITY}
            client._review_interval = interval
            client._runtime = runtime
            return client
        except TimeoutError:
            refusal = "deadline"
            if client is not None:
                await client.close()
            raise
        except Exception:
            if client is not None:
                await client.close()
            raise
        finally:
            if refusal is not None:
                _marker(_WELCOME_MARKER, refusal)

    async def archive(
        self, event: VoiceArchiveEvent
    ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent:
        """Send one batch and read its reply; only on a connection that negotiated it.

        A voice connection carries nothing else, so this call owns its reader.
        """

        if VOICE_ARCHIVE_CAPABILITY not in self._capabilities:
            raise BridgeProtocolError("the companion did not advertise the voice capability")
        if type(event) is not VoiceArchiveEvent:
            raise TypeError("event must be an exact VoiceArchiveEvent")
        payload = event.model_dump_json().encode("utf-8") + b"\n"
        if len(payload) > _MAX_LINE_BYTES:
            raise BridgeProtocolError("Hermes bridge event exceeds the line limit")
        async with self._write_lock:
            self._writer.write(payload)
            await self._writer.drain()
        raw = await self._reader.readline()
        if not raw:
            raise BridgeProtocolError("the companion closed before answering")
        reply = parse_voice_event(raw)
        if not isinstance(reply, (VoiceArchiveAckEvent, VoiceArchiveRefusedEvent)):
            raise BridgeProtocolError("the companion answered with a request")
        return reply

    async def review(
        self, event: VoiceReviewEvent
    ) -> VoiceReviewAckEvent | VoiceReviewRefusedEvent:
        if VOICE_REVIEW_CAPABILITY not in self._capabilities:
            raise BridgeProtocolError("the companion did not advertise voice review")
        if type(event) is not VoiceReviewEvent:
            raise TypeError("event must be an exact VoiceReviewEvent")
        payload = event.model_dump_json().encode("utf-8") + b"\n"
        if len(payload) > _MAX_LINE_BYTES:
            raise BridgeProtocolError("Hermes bridge event exceeds the line limit")
        async with self._write_lock:
            self._writer.write(payload)
            await self._writer.drain()
        raw = await self._reader.readline()
        if not raw:
            raise BridgeProtocolError("the companion closed before answering")
        reply = parse_voice_event(raw)
        if not isinstance(reply, (VoiceReviewAckEvent, VoiceReviewRefusedEvent)):
            raise BridgeProtocolError("the companion answered with another voice event")
        return reply

    async def forget(
        self, event: VoiceForgetEvent,
    ) -> VoiceForgetAckEvent | VoiceForgetRefusedEvent:
        if VOICE_FORGET_CAPABILITY not in self._capabilities:
            raise BridgeProtocolError("the companion did not advertise voice forget")
        if type(event) is not VoiceForgetEvent:
            raise TypeError("forget request must be exact")
        await self._send_json(event.model_dump(mode="json"))
        raw = await self._reader.readline()
        if not raw:
            raise BridgeProtocolError("the companion closed before answering")
        reply = parse_voice_event(raw)
        if type(reply) not in (VoiceForgetAckEvent, VoiceForgetRefusedEvent):
            raise BridgeProtocolError("the companion answered with another voice event")
        return cast(VoiceForgetAckEvent | VoiceForgetRefusedEvent, reply)

    async def memory(
        self, event: VoiceMemoryEvent,
    ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]:
        """Subscribe on a dedicated connection; no foreground turn performs a read."""
        if VOICE_MEMORY_CAPABILITY not in self._capabilities:
            raise BridgeProtocolError("the companion did not advertise memory")
        if type(event) is not VoiceMemoryEvent:
            raise TypeError("memory request must be exact")
        await self._send_json(event.model_dump(mode="json"))
        while raw := await self._reader.readline():
            reply = parse_voice_event(raw)
            if (type(reply) not in (VoiceMemorySnapshotEvent, VoiceMemoryRefusedEvent)
                    or reply.conversation_id != event.conversation_id
                    or reply.generation != event.generation):
                raise BridgeProtocolError("invalid memory response")
            assert isinstance(reply, (VoiceMemorySnapshotEvent, VoiceMemoryRefusedEvent))
            yield reply
        raise BridgeProtocolError("the memory subscription closed")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._writer.close()
        with suppress(ConnectionError):
            await self._writer.wait_closed()

    async def send(
        self,
        event: WorkDispatchRequestedEvent | ControlCancelEvent,
        *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None:
        payload = event.model_dump_json().encode("utf-8") + b"\n"
        if len(payload) > _MAX_LINE_BYTES:
            raise BridgeProtocolError("Hermes bridge event exceeds the line limit")
        async with self._write_lock:
            if admission_guard is not None:
                admission_guard()
            self._writer.write(payload)
            await self._writer.drain()

    async def receive(self) -> ProtocolEvent:
        raw = await self._reader.readline()
        if not raw:
            raise BridgeProtocolError("Hermes bridge closed before the next event")
        event = parse_event(raw)
        if not isinstance(
            event,
            (
                WorkDispatchAcknowledgedEvent,
                WorkCompletedEvent,
                ControlCancelAcknowledgedEvent,
            ),
        ):
            raise BridgeProtocolError("Hermes bridge returned an unsupported event")
        return event

    async def _send_json(self, value: object) -> None:
        async with self._write_lock:
            self._writer.write(json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n")
            await self._writer.drain()
