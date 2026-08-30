from __future__ import annotations

import asyncio
import json
import socket
import ssl
from pathlib import Path
from types import MethodType
from urllib.parse import urlsplit

import pytest

from hermes_realtime.client import (
    BrowserClientRuntime,
    BrowserEventProjection,
    BrowserModelCatalog,
    BrowserModelConfiguration,
    BrowserSelectableModel,
    LoopbackPeerAddress,
    LoopbackPeerAuthorizer,
    TailnetPeerAddress,
    TailnetPeerAuthorizer,
)
from hermes_realtime.livekit import LiveKitConnection, LiveKitRoomPeer
from hermes_realtime.livekit.worker import LiveKitConversationWorker


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.asyncio
async def test_runtime_close_aggregates_ordered_owner_failures_and_retries_only_failed_owner(
) -> None:
    events: list[str] = []
    server_failures = 1
    worker_failures = 1

    class Server:
        async def close(self) -> None:
            nonlocal server_failures
            events.append("server")
            if server_failures:
                server_failures -= 1
                raise RuntimeError("server close failed")

    class Worker:
        async def close(self) -> None:
            nonlocal worker_failures
            events.append("worker")
            if worker_failures:
                worker_failures -= 1
                raise RuntimeError("worker close failed")

    runtime = object.__new__(BrowserClientRuntime)
    runtime._server = Server()  # type: ignore[assignment]
    runtime._worker = Worker()  # type: ignore[assignment]
    runtime._lease_watchdog = None
    runtime._started = True
    runtime._close_operation = None
    runtime._server_closed = False
    runtime._worker_closed = False

    with pytest.raises(BaseExceptionGroup) as failure:
        await runtime.close()

    assert events == ["server", "worker"]
    assert [str(error) for error in failure.value.exceptions] == [
        "server close failed",
        "worker close failed",
    ]
    assert runtime._server_closed is False
    assert runtime._worker_closed is False
    assert runtime._started is True

    await runtime.close()

    assert events == ["server", "worker", "server", "worker"]
    assert runtime._server_closed is True
    assert runtime._worker_closed is True
    assert runtime._started is False


@pytest.mark.asyncio
async def test_runtime_close_observation_retains_failed_then_retry_success_attempts() -> None:
    from hermes_realtime.production_observation import _new_observation_channel

    failures = 1

    class Owner:
        async def close(self) -> None:
            nonlocal failures
            if failures:
                failures -= 1
                raise RuntimeError("first close fails")

    observations, recorder = _new_observation_channel()
    runtime = object.__new__(BrowserClientRuntime)
    runtime._server = Owner()  # type: ignore[assignment]
    runtime._worker = Owner()  # type: ignore[assignment]
    runtime._lease_watchdog = None
    runtime._started = True
    runtime._close_operation = None
    runtime._server_closed = False
    runtime._worker_closed = False
    runtime._production_observation_recorder = recorder

    with pytest.raises(RuntimeError, match="first close fails"):
        await runtime.close()
    await runtime.close()

    assert [(item.stage.value, item.result.value) for item in observations.records()] == [
        ("browser_client", "failed"),
        ("browser_client", "succeeded")
    ]


@pytest.mark.asyncio
async def test_runtime_close_coalesces_concurrent_callers_and_shields_owner_close_from_cancellation(
) -> None:
    events: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    class Server:
        async def close(self) -> None:
            events.append("server")
            entered.set()
            await release.wait()

    class Worker:
        async def close(self) -> None:
            events.append("worker")

    runtime = object.__new__(BrowserClientRuntime)
    runtime._server = Server()  # type: ignore[assignment]
    runtime._worker = Worker()  # type: ignore[assignment]
    runtime._lease_watchdog = None
    runtime._started = True
    runtime._close_operation = None
    runtime._server_closed = False
    runtime._worker_closed = False

    first = asyncio.create_task(runtime.close())
    await entered.wait()
    cancelled_waiter = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    release.set()
    await first

    assert events == ["server", "worker"]
    assert runtime._server_closed is True
    assert runtime._worker_closed is True


@pytest.mark.asyncio
async def test_runtime_provisions_exact_browser_identity_before_bootstrap_response(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Hermes</title>", encoding="utf-8")
    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-runtime-secret-32-bytes",
    )
    worker = object.__new__(LiveKitConversationWorker)
    events: list[tuple[object, ...]] = []

    async def connect(
        self: LiveKitConversationWorker,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float = 10,
    ) -> int:
        del self, peer, timeout_seconds
        events.append(("connect", room_name, participant_identity))
        return 7

    async def submit_final_transcript(
        self: LiveKitConversationWorker,
        *,
        participant_identity: str,
        session_generation: int,
        text: str,
    ) -> None:
        del self
        events.append(("submit", participant_identity, session_generation, text))

    async def reconnect(
        self: LiveKitConversationWorker,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float = 10,
    ) -> int:
        del self, peer, timeout_seconds
        events.append(("reconnect", room_name, participant_identity))
        return 8

    async def close(self: LiveKitConversationWorker) -> None:
        del self
        events.append(("close",))

    worker.connect = MethodType(connect, worker)  # type: ignore[method-assign]
    worker.reconnect = MethodType(reconnect, worker)  # type: ignore[method-assign]
    worker.submit_final_transcript = MethodType(  # type: ignore[method-assign]
        submit_final_transcript,
        worker,
    )
    worker.close = MethodType(close, worker)  # type: ignore[method-assign]

    async def approval(
        identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        events.append(("approval", identity, generation, sequence, approval_id, decision))

    peer = object.__new__(LiveKitRoomPeer)
    projection = BrowserEventProjection()
    port = _available_port()
    model_configuration = BrowserModelConfiguration(
        authentication="subscription",
        provider="openai-codex",
        transport="subscription-app-server",
        model="gpt-5.6-terra",
        effort="medium",
        context_window_tokens=None,
        reports_token_usage=False,
    )
    catalog_value = BrowserModelCatalog(
        models=(
            BrowserSelectableModel(
                model="gpt-5.6-terra",
                display_name="GPT-5.6 Terra",
                description="Fast coding model",
                supported_efforts=("medium",),
                default_effort="medium",
            ),
        ),
        selected_model="gpt-5.6-terra",
        selected_effort="medium",
    )

    async def model_catalog() -> BrowserModelCatalog:
        return catalog_value

    async def select_model(_model: str, _effort: str) -> BrowserModelCatalog:
        return catalog_value

    runtime = BrowserClientRuntime(
        connection=connection,
        room_name="hermes-local",
        worker_identity="worker_hermes_browser",
        worker=worker,
        peer_factory=lambda: peer,
        approval=approval,
        static_root=static,
        host="127.0.0.1",
        port=port,
        capability_factory=lambda: "r" * 43,
        bootstrap_ttl_seconds=3_600,
        model_configuration=model_configuration,
        model_catalog=model_catalog,
        select_model=select_model,
        browser_identity_factory=iter(
            ("browser_0123456789abcdef", "browser_fedcba9876543210")
        ).__next__,
        projection=projection,
    )

    assert runtime.projection is projection
    assert runtime._sessions._model_configuration is model_configuration
    assert runtime._sessions._model_catalog is model_catalog
    assert runtime._sessions._select_model is select_model
    remaining_bootstrap_seconds = runtime._capability._expires_at - runtime._capability._clock()
    assert 3_599 <= remaining_bootstrap_seconds <= 3_600.001
    launch_url = await runtime.start()
    assert runtime._lease_watchdog is not None
    parsed = urlsplit(launch_url)
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    request = (
        "POST /api/v1/bootstrap HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Origin: http://127.0.0.1:{port}\r\n"
        "Authorization: Bearer "
        f"{parsed.fragment.removeprefix('bootstrap=')}\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()

    try:
        head, body = response.split(b"\r\n\r\n", 1)
        assert head.startswith(b"HTTP/1.1 200 OK\r\n")
        payload = json.loads(body)
        assert payload["participantIdentity"] == "browser_0123456789abcdef"
        assert events == [("connect", "hermes-local", "browser_0123456789abcdef")]
        replacement = await runtime._sessions.rebind(
            participant_identity="browser_0123456789abcdef",
        )
        assert replacement.participant_identity == "browser_fedcba9876543210"
        assert events[-1] == (
            "reconnect",
            "hermes-local",
            "browser_fedcba9876543210",
        )
        assert launch_url == f"http://127.0.0.1:{port}/#bootstrap={'r' * 43}"
    finally:
        await runtime.close()

    assert events[-1] == ("close",)
    assert runtime._lease_watchdog is None


def test_runtime_rejects_browser_namespace_for_server_worker() -> None:
    with pytest.raises(ValueError, match="worker namespace"):
        BrowserClientRuntime._validate_worker_identity("browser_0123456789abcdef")


@pytest.mark.asyncio
async def test_persistent_tailnet_runtime_serves_two_sessions_without_terminal_close(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("client", encoding="utf-8")
    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-persistent-runtime-secret-32-bytes",
    )
    worker = object.__new__(LiveKitConversationWorker)
    events: list[tuple[object, ...]] = []

    async def connect(
        self: LiveKitConversationWorker,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float = 10,
    ) -> int:
        del self, peer, timeout_seconds
        events.append(("connect", room_name, participant_identity))
        return len([event for event in events if event[0] == "connect"])

    async def submit_final_transcript(
        self: LiveKitConversationWorker,
        *,
        participant_identity: str,
        session_generation: int,
        text: str,
    ) -> None:
        del self, participant_identity, session_generation, text

    async def disconnect(
        self: LiveKitConversationWorker,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, timeout_seconds
        events.append(("disconnect",))

    async def close(self: LiveKitConversationWorker) -> None:
        del self
        events.append(("close",))

    worker.connect = MethodType(connect, worker)  # type: ignore[method-assign]
    worker.submit_final_transcript = MethodType(  # type: ignore[method-assign]
        submit_final_transcript,
        worker,
    )
    worker.disconnect = MethodType(disconnect, worker)  # type: ignore[method-assign]
    worker.close = MethodType(close, worker)  # type: ignore[method-assign]
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        return b'{"Node":{"StableID":"allowed-node"}}'

    async def approve(
        identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        del identity, generation, sequence, approval_id, decision

    runtime = BrowserClientRuntime(
        connection=connection,
        room_name="room",
        worker_identity="worker_hermes_browser",
        worker=worker,
        peer_factory=lambda: object.__new__(LiveKitRoomPeer),
        approval=approve,
        static_root=static,
        host="192.0.2.10",
        port=8765,
        lan_mode=True,
        ssl_context=ssl.create_default_context(),
        canonical_origin="https://phone.example:8765",
        browser_identity_factory=lambda: next(identities),
        tailnet_authorizer=TailnetPeerAuthorizer(
            allowed_stable_id="allowed-node", resolver=resolve
        ),
        persistent_tailnet_mode=True,
    )
    app = runtime._server._application
    peer = TailnetPeerAddress.from_peername(("100.64.1.2", 54321))
    launch_headers = {
        "content-length": "0",
        "origin": "https://phone.example:8765",
        "sec-fetch-site": "same-origin",
    }

    first = await app.handle(
        method="POST",
        path="/api/v1/tailnet-bootstrap",
        headers=launch_headers,
        body=b"",
        peer=peer,
    )
    first_credential = json.loads(first.body)
    await app.handle(
        method="POST",
        path="/api/v1/stop",
        headers={
            "authorization": f"Bearer {first_credential['token']}",
            "content-length": "0",
            "origin": "https://phone.example:8765",
        },
        body=b"",
    )
    second = await app.handle(
        method="POST",
        path="/api/v1/tailnet-bootstrap",
        headers=launch_headers,
        body=b"",
        peer=peer,
    )
    second_credential = json.loads(second.body)

    assert first_credential["participantIdentity"] != second_credential["participantIdentity"]
    assert events == [
        ("connect", "room", "browser_0123456789abcdef"),
        ("disconnect",),
        ("connect", "room", "browser_fedcba9876543210"),
    ]
    await runtime.close()
    assert events[-1] == ("close",)


@pytest.mark.asyncio
async def test_persistent_loopback_runtime_serves_two_sessions_from_one_stable_url(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("client", encoding="utf-8")
    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-persistent-runtime-secret-32-bytes",
    )
    worker = object.__new__(LiveKitConversationWorker)
    events: list[tuple[object, ...]] = []

    async def connect(
        self: LiveKitConversationWorker,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float = 10,
    ) -> int:
        del self, peer, timeout_seconds
        events.append(("connect", room_name, participant_identity))
        return len([event for event in events if event[0] == "connect"])

    async def submit_final_transcript(
        self: LiveKitConversationWorker,
        *,
        participant_identity: str,
        session_generation: int,
        text: str,
    ) -> None:
        del self, participant_identity, session_generation, text

    async def disconnect(
        self: LiveKitConversationWorker,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, timeout_seconds
        events.append(("disconnect",))

    async def close(self: LiveKitConversationWorker) -> None:
        del self
        events.append(("close",))

    worker.connect = MethodType(connect, worker)  # type: ignore[method-assign]
    worker.submit_final_transcript = MethodType(  # type: ignore[method-assign]
        submit_final_transcript,
        worker,
    )
    worker.disconnect = MethodType(disconnect, worker)  # type: ignore[method-assign]
    worker.close = MethodType(close, worker)  # type: ignore[method-assign]
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))

    async def approve(
        identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        del identity, generation, sequence, approval_id, decision

    port = _available_port()
    runtime = BrowserClientRuntime(
        connection=connection,
        room_name="room",
        worker_identity="worker_hermes_browser",
        worker=worker,
        peer_factory=lambda: object.__new__(LiveKitRoomPeer),
        approval=approve,
        static_root=static,
        host="127.0.0.1",
        port=port,
        browser_identity_factory=lambda: next(identities),
        loopback_authorizer=LoopbackPeerAuthorizer(),
        persistent_loopback_mode=True,
    )
    launch_url = await runtime.start()
    app = runtime._server._application
    peer = LoopbackPeerAddress.from_peername(("127.0.0.1", 54321))
    headers = {
        "content-length": "0",
        "origin": f"http://127.0.0.1:{port}",
        "sec-fetch-site": "same-origin",
    }

    for rejected_path, rejected_peer in (
        ("/api/v1/stable-bootstrap", None),
        (
            "/api/v1/stable-bootstrap",
            TailnetPeerAddress.from_peername(("100.64.1.2", 54321)),
        ),
        ("/api/v1/tailnet-bootstrap", peer),
    ):
        with pytest.raises(PermissionError):
            await app.handle(
                method="POST",
                path=rejected_path,
                headers=headers,
                body=b"",
                peer=rejected_peer,
            )

    first = await app.handle(
        method="POST",
        path="/api/v1/stable-bootstrap",
        headers=headers,
        body=b"",
        peer=peer,
    )
    first_credential = json.loads(first.body)
    await app.handle(
        method="POST",
        path="/api/v1/stop",
        headers={
            "authorization": f"Bearer {first_credential['token']}",
            "content-length": "0",
            "origin": f"http://127.0.0.1:{port}",
        },
        body=b"",
    )
    second = await app.handle(
        method="POST",
        path="/api/v1/stable-bootstrap",
        headers=headers,
        body=b"",
        peer=peer,
    )
    second_credential = json.loads(second.body)

    assert launch_url == f"http://127.0.0.1:{port}/"
    assert first_credential["participantIdentity"] != second_credential["participantIdentity"]
    assert events == [
        ("connect", "room", "browser_0123456789abcdef"),
        ("disconnect",),
        ("connect", "room", "browser_fedcba9876543210"),
    ]
    await runtime.close()
    assert events[-1] == ("close",)


def test_runtime_normalizes_default_https_port_for_browser_origin() -> None:
    assert (
        BrowserClientRuntime._canonical_origin(
            host="192.0.2.10",
            port=443,
            lan_mode=True,
            ssl_context=ssl.create_default_context(),
            configured="https://Phone.Example:443/",
        )
        == "https://phone.example"
    )


@pytest.mark.parametrize(
    "configured",
    [
        "https://*:8765",
        "https://phone.example\\forbidden:8765",
        "https://127.0.0.1:8765",
        "https://localhost:8765",
        "https://127.1:8765",
        "https://2130706433:8765",
    ],
)
def test_runtime_rejects_unusable_remote_browser_origins(configured: str) -> None:
    with pytest.raises(ValueError, match="canonical LAN origin"):
        BrowserClientRuntime._canonical_origin(
            host="192.0.2.10",
            port=8765,
            lan_mode=True,
            ssl_context=ssl.create_default_context(),
            configured=configured,
        )
