"""Production composition for the secure local browser conversation client."""

from __future__ import annotations

import asyncio
import math
import re
import ssl
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

from hermes_realtime.evidence.models import (
    BindingCloseReason,
    CaptureStatusV1,
    EvidenceConsentRequestV1,
    EvidenceRevokeRequestV1,
)
from hermes_realtime.livekit import LiveKitConnection, LiveKitRoomPeer
from hermes_realtime.livekit.worker import LiveKitConversationWorker
from hermes_realtime.network_origin import canonical_remote_hostname
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    _ProductionObservationRecorderV1,
)
from hermes_realtime.search_egress import SearchEgressAuthority

from .bootstrap import (
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    OneTimeBootstrapCapability,
)
from .http import BrowserBootstrapApplication
from .loopback import LoopbackPeerAuthorizer
from .projection import BrowserEventProjection, BrowserPublicEvent, PublicValue
from .server import BrowserHttpServer
from .session import (
    BrowserBindingSnapshot,
    BrowserEvidenceConsentOperation,
    BrowserEvidenceRevokeOperation,
    BrowserModelCatalog,
    BrowserModelConfiguration,
    BrowserSessionDirector,
    BrowserSpeechRuntime,
)
from .tailnet import TailnetPeerAuthorizer


class BrowserClientRuntime:
    """Compose one browser HTTP surface around one conversation worker."""

    _WORKER_IDENTITY = re.compile(r"worker_[A-Za-z0-9_-]{8,64}\Z")

    def __init__(
        self,
        *,
        connection: LiveKitConnection,
        room_name: str,
        worker_identity: str,
        worker: LiveKitConversationWorker,
        approval: Callable[[str, int, int, str, str], Awaitable[None]],
        static_root: Path,
        host: str = "127.0.0.1",
        port: int = 8765,
        lan_mode: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        canonical_origin: str | None = None,
        peer_factory: Callable[[], LiveKitRoomPeer] | None = None,
        capability_factory: Callable[[], str] | None = None,
        bootstrap_ttl_seconds: int = 120,
        speech_runtime: BrowserSpeechRuntime | None = None,
        model_configuration: BrowserModelConfiguration | None = None,
        model_catalog: Callable[[], Awaitable[BrowserModelCatalog]] | None = None,
        select_model: Callable[[str, str], Awaitable[BrowserModelCatalog]] | None = None,
        activate_media: Callable[[str, int, int], Awaitable[None]] | None = None,
        yield_speech: Callable[[str, int, str, int, str, str], Awaitable[bool]] | None = None,
        conversation_profile: str = "legacy",
        on_session_started: Callable[[str, int], None] | None = None,
        browser_identity_factory: Callable[[], str] | None = None,
        projection: BrowserEventProjection | None = None,
        voice_configuration: Callable[[], tuple[tuple[str, ...], str | None]] | None = None,
        select_voice: Callable[[str], Awaitable[None]] | None = None,
        evidence_consent: Callable[
            [BrowserBindingSnapshot, EvidenceConsentRequestV1],
            BrowserEvidenceConsentOperation,
        ]
        | None = None,
        evidence_status: Callable[[], CaptureStatusV1] | None = None,
        evidence_revoke: Callable[
            [BrowserBindingSnapshot, EvidenceRevokeRequestV1],
            BrowserEvidenceRevokeOperation,
        ]
        | None = None,
        evidence_invalidate: Callable[[BindingCloseReason], Awaitable[None]] | None = None,
        search_egress_authority: SearchEgressAuthority | None = None,
        token_ttl_seconds: int = 60,
        inactivity_timeout_seconds: float = 300.0,
        lease_check_interval_seconds: float = 5.0,
        tailnet_authorizer: TailnetPeerAuthorizer | None = None,
        persistent_tailnet_mode: bool = False,
        loopback_authorizer: LoopbackPeerAuthorizer | None = None,
        persistent_loopback_mode: bool = False,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
    ) -> None:
        if type(connection) is not LiveKitConnection:
            raise TypeError("connection must be an exact LiveKitConnection")
        if type(worker) is not LiveKitConversationWorker:
            raise TypeError("worker must be an exact LiveKitConversationWorker")
        self._validate_worker_identity(worker_identity)
        if not callable(approval):
            raise TypeError("approval must be callable")
        if type(port) is not int or not 1 <= port <= 65_535:
            raise ValueError("runtime port must be an exact integer from 1 to 65535")
        if peer_factory is not None and not callable(peer_factory):
            raise TypeError("peer_factory must be callable")
        if capability_factory is not None and not callable(capability_factory):
            raise TypeError("capability_factory must be callable")
        if projection is not None and type(projection) is not BrowserEventProjection:
            raise TypeError("projection must be an exact BrowserEventProjection")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        if type(lease_check_interval_seconds) not in (int, float):
            raise TypeError("lease check interval must be an exact number")
        if (
            not math.isfinite(lease_check_interval_seconds)
            or not 1 <= lease_check_interval_seconds <= 60
        ):
            raise ValueError("lease check interval must be from 1 through 60 seconds")
        if type(persistent_tailnet_mode) is not bool:
            raise TypeError("persistent_tailnet_mode must be an exact boolean")
        if persistent_tailnet_mode != (tailnet_authorizer is not None):
            raise ValueError("persistent Tailnet mode requires exactly one Tailnet authorizer")
        if persistent_tailnet_mode and not lan_mode:
            raise ValueError("persistent Tailnet mode requires LAN mode")
        if type(persistent_loopback_mode) is not bool:
            raise TypeError("persistent_loopback_mode must be an exact boolean")
        if persistent_loopback_mode != (loopback_authorizer is not None):
            raise ValueError("persistent loopback mode requires exactly one loopback authorizer")
        if persistent_loopback_mode and lan_mode:
            raise ValueError("persistent loopback mode requires loopback mode")
        if persistent_tailnet_mode and persistent_loopback_mode:
            raise ValueError("persistent browser front doors are mutually exclusive")
        persistent_mode = persistent_tailnet_mode or persistent_loopback_mode

        origin = self._canonical_origin(
            host=host,
            port=port,
            lan_mode=lan_mode,
            ssl_context=ssl_context,
            configured=canonical_origin,
        )
        projection = projection or BrowserEventProjection()
        issuer = BrowserTokenIssuer(
            connection=connection,
            room_name=room_name,
            ttl_seconds=token_ttl_seconds,
            identity_factory=browser_identity_factory,
        )
        make_peer = peer_factory or (lambda: LiveKitRoomPeer(connection, identity=worker_identity))

        async def provision(participant_identity: str) -> int:
            peer = make_peer()
            if type(peer) is not LiveKitRoomPeer:
                raise TypeError("peer_factory must return an exact LiveKitRoomPeer")
            return await worker.connect(
                peer,
                room_name=room_name,
                participant_identity=participant_identity,
            )

        async def reprovision(participant_identity: str) -> int:
            peer = make_peer()
            if type(peer) is not LiveKitRoomPeer:
                raise TypeError("peer_factory must return an exact LiveKitRoomPeer")
            return await worker.reconnect(
                peer,
                room_name=room_name,
                participant_identity=participant_identity,
            )

        async def submit(
            identity: str,
            generation: int,
            sequence: int,
            text: str,
        ) -> None:
            await worker.submit_final_transcript(
                participant_identity=identity,
                session_generation=generation,
                typed_sequence=sequence,
                text=text,
            )

        async def stop(_identity: str, _generation: int) -> None:
            if persistent_mode:
                await worker.disconnect()
            else:
                await worker.close()

        sessions = BrowserSessionDirector(
            issuer=issuer,
            provision=provision,
            reprovision=reprovision,
            submit=submit,
            stop=stop,
            approval=approval,
            projection=projection,
            speech_runtime=speech_runtime,
            model_configuration=model_configuration,
            model_catalog=model_catalog,
            select_model=select_model,
            activate_media=activate_media,
            yield_speech=yield_speech,
            conversation_profile=conversation_profile,
            on_session_started=on_session_started,
            voice_configuration=voice_configuration,
            select_voice=select_voice,
            evidence_consent=evidence_consent,
            evidence_status=evidence_status,
            evidence_revoke=evidence_revoke,
            evidence_invalidate=evidence_invalidate,
            search_egress_authority=search_egress_authority,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
        )
        capability = (
            None
            if persistent_mode
            else OneTimeBootstrapCapability(
                ttl_seconds=bootstrap_ttl_seconds,
                token_factory=capability_factory,
            )
        )
        application = BrowserBootstrapApplication(
            sessions=sessions,
            verifier=BrowserTokenVerifier(connection=connection, room_name=room_name),
            capability=capability,
            allowed_origin=origin,
            worker_identity=worker_identity,
            tailnet_authorizer=tailnet_authorizer,
            loopback_authorizer=loopback_authorizer,
        )
        self._server = BrowserHttpServer(
            application=application,
            static_root=static_root,
            livekit_url=connection.url,
            host=host,
            port=port,
            lan_mode=lan_mode,
            ssl_context=ssl_context,
        )
        self._worker = worker
        self._sessions = sessions
        self._projection = projection
        self._lease_check_interval = float(lease_check_interval_seconds)
        self._lease_watchdog: asyncio.Task[None] | None = None
        self._close_operation: asyncio.Task[None] | None = None
        self._server_closed = False
        self._worker_closed = False
        self._ingress_close_recorded = False
        self._origin = origin
        self._capability = capability
        self._started = False
        self._persistent_mode = persistent_mode
        self._production_observation_recorder = production_observation_recorder

    @classmethod
    def _validate_worker_identity(cls, value: object) -> None:
        if type(value) is not str or cls._WORKER_IDENTITY.fullmatch(value) is None:
            raise ValueError("worker_identity must use the server-only worker namespace")

    @staticmethod
    def _canonical_origin(
        *,
        host: str,
        port: int,
        lan_mode: bool,
        ssl_context: ssl.SSLContext | None,
        configured: str | None,
    ) -> str:
        if lan_mode:
            if type(configured) is not str:
                raise ValueError("LAN mode requires an explicit canonical HTTPS origin")
            try:
                canonical_remote_hostname(host, boundary="LAN listener")
                if (
                    any(ord(character) < 33 or ord(character) > 126 for character in configured)
                    or "\\" in configured
                ):
                    raise ValueError
                parsed = urlsplit(configured)
                hostname = parsed.hostname
                configured_port = parsed.port
            except ValueError:
                raise ValueError("canonical LAN origin must be an exact HTTPS origin") from None
            if (
                parsed.scheme != "https"
                or hostname is None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError("canonical LAN origin must be an exact HTTPS origin")
            try:
                hostname = canonical_remote_hostname(hostname, boundary="canonical LAN origin")
            except ValueError:
                raise ValueError("canonical LAN origin must be an exact HTTPS origin") from None
            effective_port = configured_port if configured_port is not None else 443
            if effective_port != port:
                raise ValueError("canonical LAN origin port must match the listener")
            bracketed = f"[{hostname}]" if ":" in hostname else hostname
            port_suffix = "" if port == 443 else f":{port}"
            return f"https://{bracketed}{port_suffix}"
        if configured is not None:
            raise ValueError("loopback origin is derived rather than configured")
        if host not in {"127.0.0.1", "::1"}:
            raise ValueError("non-LAN browser runtime binds only literal loopback")
        scheme = "https" if ssl_context is not None else "http"
        bracketed = f"[{host}]" if ":" in host else host
        return f"{scheme}://{bracketed}:{port}"

    @property
    def projection(self) -> BrowserEventProjection:
        """Return the bounded public sink for authoritative subsystem markers."""

        return self._projection

    def publish(self, kind: str, data: dict[str, PublicValue]) -> BrowserPublicEvent:
        """Publish one validated public marker from an authoritative subsystem."""

        return self._projection.publish(kind, data)

    async def start(self) -> str:
        """Start serving and return the one-use fragment launch URL."""

        if self._started:
            raise RuntimeError("browser client runtime is already started")
        await self._server.start()
        self._started = True
        self._lease_watchdog = asyncio.create_task(
            self._watch_browser_lease(),
            name="browser-session-lease-watchdog",
        )
        if self._persistent_mode:
            return f"{self._origin}/"
        assert self._capability is not None
        return f"{self._origin}/#bootstrap={self._capability.token}"

    async def _watch_browser_lease(self) -> None:
        while True:
            await asyncio.sleep(self._lease_check_interval)
            try:
                expired = await self._sessions.expire_if_inactive()
            except Exception:
                continue
            if expired and not self._persistent_mode:
                await self._server.close()
                self._started = False
                return

    async def close(self) -> None:
        """Close browser ingress before closing conversation authority."""

        operation = self._close_operation
        if operation is None or self._close_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="browser-client-runtime-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    async def close_ingress(self) -> None:
        """Stop browser admission without tearing down the active conversation worker."""

        errors: list[BaseException] = []
        watchdog = self._lease_watchdog
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                if self._lease_watchdog is watchdog:
                    self._lease_watchdog = None
            except BaseException as error:
                errors.append(error)
            else:
                if self._lease_watchdog is watchdog:
                    self._lease_watchdog = None
        if not self._server_closed:
            try:
                await self._server.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._server_closed = True
                self._started = False
                self._record_ingress_close_result(CloseResultV1.SUCCEEDED)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("browser ingress close failed", errors)

    @staticmethod
    def _close_failed(operation: asyncio.Task[None]) -> bool:
        if not operation.done():
            return False
        if operation.cancelled():
            return True
        return operation.exception() is not None

    async def _close_owned(self) -> None:
        """Settle each shutdown owner once, retaining only failed-owner retries."""

        errors: list[BaseException] = []
        try:
            await self.close_ingress()
        except BaseException as error:
            errors.append(error)
        if not self._worker_closed:
            try:
                await self._worker.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._worker_closed = True
        if len(errors) == 1:
            self._record_close_result(CloseResultV1.FAILED)
            raise errors[0]
        if errors:
            self._record_close_result(CloseResultV1.FAILED)
            raise BaseExceptionGroup("browser client runtime close failed", errors)
        self._record_close_result(CloseResultV1.SUCCEEDED)

    def _record_close_result(self, result: CloseResultV1) -> None:
        self._record_ingress_close_result(result)

    def _record_ingress_close_result(self, result: CloseResultV1) -> None:
        if getattr(self, "_ingress_close_recorded", False):
            return
        if result is CloseResultV1.SUCCEEDED:
            self._ingress_close_recorded = True
        recorder = getattr(self, "_production_observation_recorder", None)
        if recorder is not None:
            recorder.record_close_stage(stage=CloseStageV1.BROWSER_CLIENT, result=result)
