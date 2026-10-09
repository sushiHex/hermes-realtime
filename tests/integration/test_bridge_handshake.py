"""The 0.3 hello authenticates both sides without the token ever crossing the wire."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import socket
import sys
from typing import Any

import pytest

from hermes_realtime.integration import BridgeAuthenticationError, LocalHermesBridgeClient, bridge
from tests.integration.test_bridge_voice import (
    _ATTESTATION,
    _TOKEN,
    _batch,
    _ReviewVoice,
    _server,
    _Voice,
)
from tests.support import bridge_hello

_HELLO_MARKER = "[hermes-bridge-hello] "
_WELCOME_MARKER = "[hermes-bridge-welcome] "


def _markers(output: str, prefix: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(prefix)) for line in output.splitlines()
        if line.startswith(prefix)
    ]


def _review_welcome(hello: dict[str, object]) -> dict[str, object]:
    return {
        "ok": True,
        "protocol_version": "0.3",
        "capabilities": ["mutual_auth", "voice_review"],
        "review_interval": 10,
    }


async def _connect(port: int, **options: Any) -> LocalHermesBridgeClient:
    return await LocalHermesBridgeClient.connect(
        host="127.0.0.1", port=port, token=_TOKEN, participant_id="voice-review",
        capabilities=options.pop("capabilities", ("voice_review",)), **options,
    )


@pytest.mark.asyncio
async def test_the_token_never_crosses_the_wire_in_either_direction() -> None:
    seen = bytearray()

    async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection(server.host, server.port)

        async def pump(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
            while chunk := await source.read(65536):
                seen.extend(chunk)
                sink.write(chunk)
                await sink.drain()
            sink.close()

        await asyncio.gather(pump(reader, upstream_writer), pump(upstream_reader, writer),
                             return_exceptions=True)

    async with _server(voice=_Voice()) as server:
        relay_server = await asyncio.start_server(relay, "127.0.0.1", 0)
        async with relay_server:
            port = relay_server.sockets[0].getsockname()[1]
            client = await LocalHermesBridgeClient.connect(
                host="127.0.0.1", port=port, token=_TOKEN, participant_id="voice-archive",
                capabilities=("voice_archive",),
            )
            async with client:
                assert client.capabilities == frozenset({"voice_archive"})
                await client.archive(_batch())

    assert seen and _TOKEN.encode() not in seen
    assert b"token" not in seen.split(b"\n", 1)[0]


@pytest.mark.asyncio
async def test_the_client_sends_nothing_after_its_hello_before_the_server_proves_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A companion that does not hold the token: its welcome is well formed, its proof is not.
    impostor = bridge_hello.FakeCompanion("not-the-shared-token-" * 2, _review_welcome)
    server, port = await impostor.start()
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port)
        await asyncio.sleep(0.05)

    # Exactly the hello, which carries no secret, and no proof or event after it.
    assert len(impostor.received) == 1
    hello = impostor.received[0]
    assert type(hello) is dict and set(hello) == {
        "participant_id", "protocol_version", "capabilities", "client_nonce",
    }
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "proof", "version": 1}
    ]


def _splice(field: str, value: object) -> Any:
    return lambda covered: covered | {field: value}


_OPENING_SPLICES = [
    pytest.param(_splice("participant_id", "someone-else"), id="participant"),
    pytest.param(_splice("client_nonce", "0" * 64), id="replayed-client-nonce"),
    pytest.param(_splice("server_nonce", "1" * 64), id="server-nonce"),
    pytest.param(_splice("requested", ["mutual_auth"]), id="requested"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sign", _OPENING_SPLICES)
async def test_a_server_proof_spliced_from_another_handshake_is_refused(
    sign: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome, sign=sign)
    server, port = await companion.start()
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port)
        await asyncio.sleep(0.05)
    assert len(companion.received) == 1
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "proof", "version": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sign_accept",
    [
        *_OPENING_SPLICES,
        pytest.param(_splice("negotiated", ["mutual_auth"]), id="negotiated"),
        pytest.param(_splice("metadata", {"review_interval": 999}), id="metadata"),
    ],
)
async def test_an_acceptance_proof_spliced_from_another_handshake_is_refused(
    sign_accept: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome, sign_accept=sign_accept)
    server, port = await companion.start()
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port)
        await asyncio.sleep(0.05)
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "acceptance", "version": 1}
    ]


@pytest.mark.asyncio
async def test_a_signed_acceptance_with_a_field_no_proof_covers_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome, accept_extra={"x": 1})
    server, port = await companion.start()
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port)
        await asyncio.sleep(0.05)
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "acceptance", "version": 1}
    ]


def test_metadata_too_deep_to_re_encode_fails_the_proof_check_instead_of_raising() -> None:
    deep: object = []
    for _ in range(sys.getrecursionlimit() + 100):
        deep = [deep]
    handshake = bridge._Handshake(
        "voice-review", "0" * 64, "1" * 64, frozenset({"mutual_auth", "runtime_attestation"})
    ).accepting(frozenset({"mutual_auth", "runtime_attestation"}), {"runtime": deep})

    assert handshake.verifies(_TOKEN, "accept", "2" * 64) is False


@pytest.mark.asyncio
async def test_the_welcome_discloses_nothing_but_a_nonce_and_a_proof() -> None:
    sent = bytearray()
    async with _server(voice=_ReviewVoice(), runtime=_ATTESTATION) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        hello = bridge_hello.hello(
            "voice-review", ("voice_archive", "voice_review", "runtime_attestation")
        )
        writer.write(json.dumps(hello).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
        sent.extend(line)
        welcome = json.loads(line)
        assert set(welcome) == {"ok", "protocol_version", "server_nonce", "proof"}
        # A peer without the token proves nothing and is told nothing more.
        writer.write(json.dumps({"proof": "0" * 64}).encode() + b"\n")
        await writer.drain()
        sent.extend(await asyncio.wait_for(reader.read(), 2))
        writer.close()
    assert b"runtime" not in sent and b"review_interval" not in sent
    assert b"capabilities" not in sent and _ATTESTATION.hermes_commit.encode() not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize("nonce", ["0" * 63, "A" * 64, "0" * 65])
async def test_a_welcome_with_a_malformed_server_nonce_is_refused_even_when_signed(
    nonce: str,
) -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome, server_nonce=nonce)
    server, port = await companion.start()
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port)
    assert len(companion.received) == 1


@pytest.mark.asyncio
async def test_no_attestation_is_read_before_the_server_proof_verifies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def welcome(hello: dict[str, object]) -> dict[str, object]:
        return {
            "ok": True, "protocol_version": "0.3",
            "capabilities": ["mutual_auth", "runtime_attestation"],
            "runtime": _ATTESTATION.model_dump(mode="json") | {"pid": "not-a-pid"},
        }

    impostor = bridge_hello.FakeCompanion("not-the-shared-token-" * 2, welcome)
    server, port = await impostor.start()
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port, capabilities=("runtime_attestation",))
        await asyncio.sleep(0.05)
    # The forged proof refuses it, never the attestation it carried.
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "proof", "version": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accept", "category", "error"),
    [
        pytest.param("forged", "acceptance", BridgeAuthenticationError, id="forged"),
        pytest.param("missing", "acceptance", BridgeAuthenticationError, id="closed"),
        pytest.param("silent", "deadline", TimeoutError, id="silent"),
    ],
)
async def test_connect_returns_only_after_an_authenticated_acceptance(
    accept: str, category: str, error: type[Exception], capsys: pytest.CaptureFixture[str]
) -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome, accept=accept)
    server, port = await companion.start()
    async with server:
        with pytest.raises(error):
            await _connect(port, handshake_timeout=0.5)
        await asyncio.sleep(0.05)
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": category, "version": 1}
    ]


@pytest.mark.asyncio
async def test_the_client_handshake_has_a_deadline(capsys: pytest.CaptureFixture[str]) -> None:
    async def mute(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()

    server = await asyncio.start_server(mute, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        # The outer bound only keeps a missing deadline from hanging the suite: then the
        # client is cancelled from outside and never reports its own deadline.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(_connect(port, handshake_timeout=0.2), 2)
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "deadline", "version": 1}
    ]


@pytest.mark.asyncio
async def test_a_cancelled_connect_is_no_refusal_and_closes_its_socket(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    closes: list[LocalHermesBridgeClient] = []
    close = LocalHermesBridgeClient.close

    async def recording_close(self: LocalHermesBridgeClient) -> None:
        closes.append(self)
        await close(self)

    monkeypatch.setattr(LocalHermesBridgeClient, "close", recording_close)

    async def mute(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        with contextlib.suppress(ConnectionError):
            await reader.read()

    server = await asyncio.start_server(mute, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        # The caller's own bound cancels the connect from outside its handshake deadline.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(_connect(port, handshake_timeout=10), 0.2)
    # Closed by the handshake itself, not left for garbage collection to find.
    assert len(closes) == 1
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "welcome",
    [
        pytest.param(b'{"ok": 1' + b"0" * 5000 + b"}\n", id="huge-integer"),
        pytest.param(b"[" * 5000 + b"]" * 5000 + b"\n", id="deep-nesting"),
        pytest.param(b"x" * (64 * 1024 + 1) + b"\n", id="oversized"),
    ],
)
async def test_an_unreadable_welcome_is_refused_with_one_true_marker(
    welcome: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    async def companion(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.write(welcome)
        with contextlib.suppress(ConnectionError):
            await writer.drain()
            await reader.read()
        writer.close()

    server = await asyncio.start_server(companion, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        with pytest.raises(BridgeAuthenticationError):
            await _connect(port)
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "shape", "version": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "category"),
    [
        ("before-welcome", "shape"),
        ("after-welcome", "acceptance"),
        ("after-proof", "acceptance"),
    ],
)
async def test_a_companion_reset_mid_handshake_leaves_one_true_marker(
    stage: str, category: str, capsys: pytest.CaptureFixture[str]
) -> None:
    loop = asyncio.get_running_loop()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)

    async def companion() -> None:
        sock, _ = await loop.sock_accept(listener)
        sock.setblocking(False)
        hello = json.loads(await bridge_hello.raw_line(sock))
        if stage != "before-welcome":
            server_nonce = "2" * 64
            opening = {
                "participant_id": hello["participant_id"],
                "client_nonce": hello["client_nonce"],
                "server_nonce": server_nonce,
                "requested": hello["capabilities"],
            }
            welcome = {
                "ok": True, "protocol_version": "0.3", "server_nonce": server_nonce,
                "proof": bridge_hello.proof(_TOKEN, "server", **opening),
            }
            await loop.sock_sendall(sock, json.dumps(welcome).encode() + b"\n")
            if stage == "after-proof":
                await bridge_hello.raw_line(sock)
            else:
                await asyncio.sleep(0.05)
        bridge_hello.reset(sock)

    task = asyncio.create_task(companion())
    try:
        with pytest.raises((BridgeAuthenticationError, ConnectionError)):
            await _connect(listener.getsockname()[1])
        await task
    finally:
        listener.close()
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": category, "version": 1}
    ]


@pytest.mark.asyncio
async def test_a_connection_dropped_after_the_welcome_is_refused_as_the_acceptance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome, accept="dropped")
    server, port = await companion.start()
    async with server:
        with pytest.raises((BridgeAuthenticationError, ConnectionError)):
            await _connect(port)
    # Whether the client's proof send or its read of the acceptance meets the drop, the
    # failure is that no authenticated acceptance arrived.
    assert _markers(capsys.readouterr().out, _WELCOME_MARKER) == [
        {"refusal": "acceptance", "version": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "1"])
async def test_the_handshake_deadline_must_be_a_positive_finite_number(timeout: object) -> None:
    with pytest.raises(ValueError, match="handshake_timeout"):
        await _connect(1, handshake_timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"proof": "0" * 64}, id="wrong-proof"),
        pytest.param({"proof": "reflected"}, id="the-servers-own-proof"),
        pytest.param({"proof": "0" * 64, "extra": 1}, id="extra-key"),
        pytest.param(None, id="an-event-instead-of-a-proof"),
    ],
)
async def test_the_server_routes_nothing_until_the_client_proves_the_token(
    answer: dict[str, object] | None, capsys: pytest.CaptureFixture[str]
) -> None:
    voice = _Voice()
    async with _server(voice=voice) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        hello = bridge_hello.hello("voice-archive", ("voice_archive",))
        writer.write(json.dumps(hello).encode() + b"\n")
        await writer.drain()
        welcome = json.loads(await reader.readline())
        assert welcome["ok"] is True
        if answer is not None:
            if answer.get("proof") == "reflected":
                # Role separation: the server's proof is no client proof.
                answer = {"proof": welcome["proof"]}
            writer.write(json.dumps(answer).encode() + b"\n")
        writer.write(_batch().model_dump_json().encode() + b"\n")
        await writer.drain()
        assert await reader.readline() == b""
        writer.close()
        await asyncio.sleep(0.05)
    assert voice.events == []
    assert _markers(capsys.readouterr().out, _HELLO_MARKER) == [
        {"refusal": "proof", "version": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stall", ["before-hello", "after-welcome"])
async def test_the_server_marks_an_expired_authentication_deadline(
    stall: str, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _server(voice=_Voice(), authentication_timeout=0.2) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        if stall == "after-welcome":
            hello = bridge_hello.hello("voice-archive", ("voice_archive",))
            writer.write(json.dumps(hello).encode() + b"\n")
            await writer.drain()
            assert json.loads(await reader.readline())["ok"] is True
        # The outer bound only keeps a missing server deadline from hanging the suite.
        assert await asyncio.wait_for(reader.readline(), 2) == b""
        writer.close()
        await asyncio.sleep(0.05)
    assert _markers(capsys.readouterr().out, _HELLO_MARKER) == [
        {"refusal": "deadline", "version": 1}
    ]


# Lines whose read or parse fails before any refusal is decided: past the line bound, an
# integer past Python's digit limit (a plain ValueError), and nesting past the recursion limit.
_UNREADABLE = {
    "oversized": b"x" * (64 * 1024 + 1) + b"\n",
    "huge-integer": b'{"proof": 1' + b"0" * 5000 + b"}\n",
    "deep-nesting": b"[" * 5000 + b"]" * 5000 + b"\n",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "ending", "category"),
    [
        *[
            pytest.param("hello", line, "shape", id=f"{line}-hello")
            for line in _UNREADABLE
        ],
        *[
            pytest.param("answer", line, "proof", id=f"{line}-answer")
            for line in _UNREADABLE
        ],
        pytest.param("answer", "closed", "abandoned", id="closed-after-welcome"),
        pytest.param("accepted", None, None, id="accepted"),
    ],
)
async def test_every_server_handshake_exit_but_acceptance_leaves_one_true_marker(
    stage: str, ending: str | None, category: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _server(voice=_Voice()) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        if stage == "accepted":
            await bridge_hello.authenticate(reader, writer, _TOKEN, bridge_hello.hello())
            writer.close()
        else:
            if stage == "answer":
                hello = bridge_hello.hello("voice-archive", ("voice_archive",))
                writer.write(json.dumps(hello).encode() + b"\n")
                await writer.drain()
                assert json.loads(await reader.readline())["ok"] is True
            if ending == "closed":
                writer.write_eof()
            else:
                assert ending is not None
                writer.write(_UNREADABLE[ending])
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), 2) in (b"", b'{"ok":false}\n')
            writer.close()
        await asyncio.sleep(0.1)
    expected = [] if category is None else [{"refusal": category, "version": 1}]
    assert _markers(capsys.readouterr().out, _HELLO_MARKER) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "category"), [("before-hello", "shape"), ("after-welcome", "abandoned")]
)
async def test_a_client_reset_mid_handshake_leaves_one_true_marker(
    stage: str, category: str, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _server(voice=_Voice()) as server:
        sock = await bridge_hello.raw_connect(server.host, server.port)
        if stage == "after-welcome":
            hello = bridge_hello.hello("voice-archive", ("voice_archive",))
            await asyncio.get_running_loop().sock_sendall(sock, json.dumps(hello).encode() + b"\n")
            assert json.loads(await bridge_hello.raw_line(sock))["ok"] is True
        else:
            await asyncio.sleep(0.05)
        bridge_hello.reset(sock)
        await asyncio.sleep(0.1)
    assert _markers(capsys.readouterr().out, _HELLO_MARKER) == [
        {"refusal": category, "version": 1}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before-hello", "after-welcome"])
async def test_closing_the_server_mid_handshake_is_no_refusal(
    stage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    server = _server(voice=_Voice())
    await server.start()
    try:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        if stage == "after-welcome":
            hello = bridge_hello.hello("voice-archive", ("voice_archive",))
            writer.write(json.dumps(hello).encode() + b"\n")
            await writer.drain()
            assert json.loads(await reader.readline())["ok"] is True
        else:
            await asyncio.sleep(0.05)
    finally:
        await server.close()
    assert await asyncio.wait_for(reader.read(), 2) == b""
    writer.close()
    await asyncio.sleep(0.05)
    assert _markers(capsys.readouterr().out, _HELLO_MARKER) == []


@pytest.mark.asyncio
async def test_every_handshake_uses_fresh_random_nonces() -> None:
    companion = bridge_hello.FakeCompanion(_TOKEN, _review_welcome)
    server, port = await companion.start()
    async with server:
        for _ in range(2):
            client = await _connect(port)
            await client.close()
    client_nonces = [
        hello["client_nonce"] for hello in companion.received
        if type(hello) is dict and "client_nonce" in hello
    ]
    assert len(set(client_nonces)) == 2
    assert all(re.fullmatch(r"[0-9a-f]{64}", nonce) for nonce in client_nonces)

    async with _server(voice=_Voice()) as real:
        welcomes = []
        for _ in range(2):
            reader, writer = await asyncio.open_connection(real.host, real.port)
            welcomes.append(await bridge_hello.authenticate(
                reader, writer, _TOKEN, bridge_hello.hello()
            ))
            writer.close()
    server_nonces = [welcome["server_nonce"] for welcome in welcomes]
    assert len(set(server_nonces)) == 2
    assert all(re.fullmatch(r"[0-9a-f]{64}", str(nonce)) for nonce in server_nonces)
