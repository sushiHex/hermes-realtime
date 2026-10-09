"""The 0.3 hello authenticates both sides without the token ever crossing the wire."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest

from hermes_realtime.integration import BridgeAuthenticationError, LocalHermesBridgeClient
from tests.integration.test_bridge_voice import _ATTESTATION, _TOKEN, _batch, _server, _Voice
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sign",
    [
        pytest.param(_splice("participant_id", "someone-else"), id="participant"),
        pytest.param(_splice("client_nonce", "0" * 64), id="replayed-client-nonce"),
        pytest.param(_splice("server_nonce", "1" * 64), id="server-nonce"),
        pytest.param(_splice("negotiated", ["mutual_auth"]), id="negotiated"),
        pytest.param(_splice("requested", ["mutual_auth"]), id="requested"),
        pytest.param(_splice("metadata", {"review_interval": 999}), id="metadata"),
    ],
)
async def test_a_proof_spliced_from_another_handshake_is_refused(
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
async def test_the_client_handshake_has_a_deadline() -> None:
    async def mute(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()

    server = await asyncio.start_server(mute, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await _connect(port, handshake_timeout=0.2)
        assert asyncio.get_running_loop().time() - started < 2


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
