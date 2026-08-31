from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hermes_realtime.client import (
    BrowserBootstrapApplication,
    BrowserEventProjection,
    BrowserHttpServer,
    BrowserSessionDirector,
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    LoopbackPeerAuthorizer,
    OneTimeBootstrapCapability,
    TailnetPeerAddress,
    TailnetPeerAuthorizer,
)
from hermes_realtime.livekit import LiveKitConnection


async def _submit(identity: str, generation: int, text: str) -> None:
    del identity, generation, text


async def _stop(identity: str, generation: int) -> None:
    del identity, generation


async def _approval(
    identity: str,
    generation: int,
    sequence: int,
    approval_id: str,
    decision: str,
) -> None:
    del identity, generation, sequence, approval_id, decision


async def _raw_request(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


@pytest.mark.asyncio
async def test_server_serves_only_allowlisted_shell_with_security_headers(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Hermes</title>", encoding="utf-8")
    assets = static / "assets"
    assets.mkdir()
    (assets / "styles.css").write_text(":root { color-scheme: dark; }", encoding="utf-8")
    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-bootstrap-secret-32-bytes",
    )

    async def provision(identity: str) -> int:
        del identity
        return 1

    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(connection=connection, room_name="hermes-local"),
            provision=provision,
            submit=_submit,
            stop=_stop,
            approval=_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "d" * 43),
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
    )
    server = BrowserHttpServer(
        application=app,
        static_root=static,
        host="127.0.0.1",
        port=0,
    )
    await server.start()
    try:
        response = await _raw_request(
            server.port,
            b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        )
        stylesheet_response = await _raw_request(
            server.port,
            b"GET /assets/styles.css HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        )
    finally:
        await server.close()

    headers, body = response.split(b"\r\n\r\n", 1)
    assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
    assert b"content-security-policy:" in headers.lower()
    assert b"cache-control: no-store" in headers.lower()
    assert b"access-control-allow-origin" not in headers.lower()
    assert body == b"<!doctype html><title>Hermes</title>"
    stylesheet_headers, stylesheet_body = stylesheet_response.split(b"\r\n\r\n", 1)
    assert b"cache-control: no-store" in stylesheet_headers.lower()
    assert stylesheet_body == b":root { color-scheme: dark; }"


@pytest.mark.asyncio
async def test_server_accepts_browser_bodyless_bootstrap_request(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Hermes</title>", encoding="utf-8")
    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-bootstrap-secret-32-bytes",
    )

    async def provision(identity: str) -> int:
        del identity
        return 1

    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(connection=connection, room_name="hermes-local"),
            provision=provision,
            submit=_submit,
            stop=_stop,
            approval=_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "d" * 43),
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
    )
    server = BrowserHttpServer(
        application=app,
        static_root=static,
        host="127.0.0.1",
        port=0,
    )
    await server.start()
    try:
        response = await _raw_request(
            server.port,
            (
                b"POST /api/v1/bootstrap HTTP/1.1\r\n"
                b"Host: 127.0.0.1:8765\r\n"
                b"Origin: http://127.0.0.1:8765\r\n"
                b"Accept: */*\r\n"
                b"Accept-Encoding: gzip, deflate\r\n"
                b"Accept-Language: en-US\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Connection: close\r\n"
                b"Pragma: no-cache\r\n"
                b"Sec-CH-UA: browser\r\n"
                b"Sec-CH-UA-Mobile: ?0\r\n"
                b"Sec-CH-UA-Platform: Windows\r\n"
                b"Sec-Fetch-Dest: empty\r\n"
                b"Sec-Fetch-Mode: cors\r\n"
                b"Sec-Fetch-Site: same-origin\r\n"
                b"User-Agent: browser\r\n" + b"Authorization: Bearer " + b"d" * 43 + b"\r\n\r\n"
            ),
        )
    finally:
        await server.close()

    headers, body = response.split(b"\r\n\r\n", 1)
    assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
    assert b'"participantIdentity":"browser_' in body


@pytest.mark.asyncio
async def test_server_uses_socket_peer_and_ignores_forwarding_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("client", encoding="utf-8")
    connection = LiveKitConnection("wss://livekit.test", "key", "s" * 32)
    observed: list[TailnetPeerAddress] = []

    async def provision(_identity: str) -> int:
        return 1

    async def resolve(peer: TailnetPeerAddress) -> bytes:
        observed.append(peer)
        return b'{"Node":{"StableID":"allowed-node"}}'

    real_peer = TailnetPeerAddress.from_peername(("100.64.1.2", 54321))
    monkeypatch.setattr(
        TailnetPeerAddress,
        "from_peername",
        classmethod(lambda _cls, _peername: real_peer),
    )
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(connection=connection, room_name="room"),
            provision=provision,
            submit=_submit,
            stop=_stop,
            approval=_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None,
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
        tailnet_authorizer=TailnetPeerAuthorizer(
            allowed_stable_id="allowed-node", resolver=resolve
        ),
    )
    server = BrowserHttpServer(application=app, static_root=static, port=0)
    await server.start()
    try:
        response = await _raw_request(
            server.port,
            (
                b"POST /api/v1/tailnet-bootstrap HTTP/1.1\r\n"
                b"Host: 127.0.0.1:8765\r\n"
                b"Origin: http://127.0.0.1:8765\r\n"
                b"Sec-Fetch-Site: same-origin\r\n"
                b"X-Forwarded-For: 100.64.9.9\r\n"
                b"Forwarded: for=100.64.9.9\r\n"
                b"X-Real-IP: 100.64.9.9\r\n"
                b"Content-Length: 0\r\n\r\n"
            ),
        )
    finally:
        await server.close()

    assert response.startswith(b"HTTP/1.1 200 OK\r\n")
    assert observed == [real_peer]


@pytest.mark.asyncio
async def test_server_authorizes_stable_loopback_from_the_socket_peer(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("client", encoding="utf-8")
    connection = LiveKitConnection("wss://livekit.test", "key", "s" * 32)
    provisioned: list[str] = []

    async def provision(identity: str) -> int:
        provisioned.append(identity)
        return 1

    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(connection=connection, room_name="room"),
            provision=provision,
            submit=_submit,
            stop=_stop,
            approval=_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None,
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
        loopback_authorizer=LoopbackPeerAuthorizer(),
    )
    server = BrowserHttpServer(
        application=app,
        static_root=static,
        host="127.0.0.1",
        port=0,
    )
    await server.start()
    try:
        response = await _raw_request(
            server.port,
            (
                b"POST /api/v1/stable-bootstrap HTTP/1.1\r\n"
                b"Host: 127.0.0.1:8765\r\n"
                b"Origin: http://127.0.0.1:8765\r\n"
                b"Sec-Fetch-Site: same-origin\r\n"
                b"X-Forwarded-For: 100.64.9.9\r\n"
                b"Forwarded: for=100.64.9.9\r\n"
                b"X-Real-IP: 100.64.9.9\r\n"
                b"Content-Length: 0\r\n\r\n"
            ),
        )
    finally:
        await server.close()

    assert response.startswith(b"HTTP/1.1 200 OK\r\n")
    assert len(provisioned) == 1
    assert provisioned[0].startswith("browser_")
