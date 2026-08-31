"""Authenticated loopback IPC for the realtime worker and Hermes plugin."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Self

from hermes_realtime.protocol import (
    ControlCancelAcknowledgedEvent,
    ControlCancelEvent,
    ProtocolEvent,
    WorkCompletedEvent,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
    parse_event,
)

from .completion import HermesCompletionRouter
from .plugin import HermesCompletionSource, HermesRunCompletion
from .service import HermesIntegrationService
from .session import SessionBinding

_MAX_LINE_BYTES = 64 * 1024
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
logger = logging.getLogger(__name__)


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
        self._server: asyncio.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe_completion: Callable[[], None] | None = None
        self._completion_tasks: set[asyncio.Task[WorkCompletedEvent | None]] = set()
        self._completion_ingress: set[str] = set()
        self._authoritative_ingress: set[str] = set()
        self._completion_ingress_lock = threading.Lock()
        self._routes: dict[str, _Route] = {}
        self._connections: set[_Connection] = set()
        self._handler_tasks: set[asyncio.Task[None]] = set()
        self._pending: OrderedDict[
            str, tuple[WorkTerminalStatus | str, str | None, str | None]
        ] = OrderedDict()
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
                    run_id not in active_run_ids
                    and run_id not in self._authoritative_ingress
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
                            pending_run_id not in active_run_ids
                            for pending_run_id in self._pending
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

        participant_id: str | None = None
        try:
            participant_id = await asyncio.wait_for(
                self._authenticate(reader, connection),
                timeout=self._authentication_timeout,
            )
            while raw := await reader.readline():
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
    ) -> str:
        raw = await reader.readline()
        try:
            hello = json.loads(raw)
            token = hello["token"]
            participant_id = hello["participant_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            await self._send_json(connection, {"ok": False})
            raise BridgeAuthenticationError("invalid bridge handshake") from exc
        if (
            not isinstance(token, str)
            or not hmac.compare_digest(
                token.encode("utf-8", errors="surrogatepass"),
                self._token.encode("utf-8", errors="surrogatepass"),
            )
            or not isinstance(participant_id, str)
            or _IDENTIFIER_PATTERN.fullmatch(participant_id) is None
        ):
            await self._send_json(connection, {"ok": False})
            raise BridgeAuthenticationError("bridge authentication failed")
        await self._send_json(connection, {"ok": True})
        return participant_id

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

    @classmethod
    async def connect(
        cls,
        *,
        host: str,
        port: int,
        token: str,
        participant_id: str,
    ) -> LocalHermesBridgeClient:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Hermes bridge client must connect to loopback")
        reader, writer = await asyncio.open_connection(
            host,
            port,
            limit=_MAX_LINE_BYTES,
        )
        client = cls(reader, writer)
        try:
            await client._send_json(
                {"token": token, "participant_id": participant_id}
            )
            response = json.loads(await reader.readline())
            if response != {"ok": True}:
                raise BridgeAuthenticationError("bridge authentication failed")
            return client
        except Exception:
            await client.close()
            raise

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
            self._writer.write(
                json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            await self._writer.drain()
