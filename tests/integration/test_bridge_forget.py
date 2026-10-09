"""Deletion is a negotiated private request, never a work or speech event."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from hermes_realtime.integration.bridge import BridgeProtocolError, LocalHermesBridgeClient
from hermes_realtime.protocol import (
    VoiceArchiveAckEvent,
    VoiceForgetAckEvent,
    VoiceForgetEvent,
    VoiceReviewAckEvent,
    VoiceReviewEvent,
    parse_voice_event,
)
from tests.integration.test_bridge_voice import _TOKEN, _batch, _ReviewVoice, _server, _Voice
from tests.support import bridge_hello


class ForgetVoice(_Voice):
    forget_available = True

    async def forget(self, event: VoiceForgetEvent) -> VoiceForgetAckEvent:
        return VoiceForgetAckEvent(protocol_version="0.3", type="voice_forget_ack",
                                   conversation_id=event.conversation_id,
                                   generation=event.generation, state="pending")


def request() -> VoiceForgetEvent:
    return VoiceForgetEvent(protocol_version="0.3", type="voice_forget",
                            conversation_id="synthetic", generation=2)


@pytest.mark.asyncio
async def test_forget_negotiates_and_returns_bound_pending_receipt() -> None:
    async with _server(ForgetVoice()) as server, await LocalHermesBridgeClient.connect(
        host=server.host, port=server.port, token=_TOKEN,
        participant_id="delete", capabilities=("voice_forget",),
    ) as client:
        assert client.capabilities == frozenset({"voice_forget"})
        reply = await client.forget(request())
        assert reply.model_dump() == dict(protocol_version="0.3", type="voice_forget_ack",
                                          conversation_id="synthetic", generation=2,
                                          state="pending")


@pytest.mark.asyncio
async def test_forget_without_negotiation_refuses_before_sending() -> None:
    async with _server(ForgetVoice()) as server, await LocalHermesBridgeClient.connect(
        host=server.host, port=server.port, token=_TOKEN, participant_id="old",
    ) as client:
        with pytest.raises(BridgeProtocolError, match="forget"):
            await client.forget(request())


@pytest.mark.asyncio
async def test_raw_forget_without_negotiation_closes_connection() -> None:
    async with _server(ForgetVoice()) as server, await LocalHermesBridgeClient.connect(
        host=server.host, port=server.port, token=_TOKEN, participant_id="old",
    ) as client:
        client._writer.write(request().model_dump_json().encode() + b"\n")
        await client._writer.drain()
        assert await asyncio.wait_for(client._reader.readline(), 2) == b""


def isolated_client(raw: bytes) -> LocalHermesBridgeClient:
    client = object.__new__(LocalHermesBridgeClient)
    client._capabilities = frozenset({"voice_forget"})
    client._send_json = AsyncMock()  # type: ignore[method-assign]
    client._reader = asyncio.StreamReader()
    client._reader.feed_data(raw)
    client._reader.feed_eof()
    return client


def receipt() -> bytes:
    return VoiceForgetAckEvent(protocol_version="0.3", type="voice_forget_ack",
                               conversation_id="synthetic", generation=2,
                               state="complete").model_dump_json().encode() + b"\n"


@pytest.mark.asyncio
async def test_forget_client_capability_guard_prevents_even_a_valid_reply() -> None:
    client = isolated_client(receipt())
    client._capabilities = frozenset()
    with pytest.raises(BridgeProtocolError, match="advertise voice forget"):
        await client.forget(request())
    client._send_json.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_forget_client_refuses_subclass_before_send() -> None:
    class Subclass(VoiceForgetEvent):
        pass

    client = isolated_client(receipt())
    event = Subclass(**request().model_dump())
    with pytest.raises(TypeError, match="exact"):
        await client.forget(event)
    client._send_json.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_forget_client_refuses_closed_connection() -> None:
    client = isolated_client(b"")
    with pytest.raises(BridgeProtocolError, match="closed"):
        await client.forget(request())


@pytest.mark.asyncio
async def test_forget_client_refuses_other_event() -> None:
    client = isolated_client(request().model_dump_json().encode() + b"\n")
    with pytest.raises(BridgeProtocolError, match="another voice event"):
        await client.forget(request())


@pytest.mark.parametrize("available", [False, 1, "true", None])
def test_forget_availability_must_be_exact_true(available: object) -> None:
    voice = ForgetVoice()
    voice.forget_available = available  # type: ignore[assignment]
    assert "voice_forget" not in _server(voice)._offered


def test_forget_availability_requires_callable_handler() -> None:
    voice = ForgetVoice()
    voice.forget = None  # type: ignore[assignment]
    assert "voice_forget" not in _server(voice)._offered


@pytest.mark.asyncio
async def test_old_0_3_peer_keeps_archive_review_without_forget() -> None:
    class AllVoice(ForgetVoice, _ReviewVoice):
        pass

    async with _server(AllVoice()) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        try:
            welcome = await bridge_hello.authenticate(reader, writer, _TOKEN, bridge_hello.hello(
                "older-peer", ("voice_archive", "voice_review"),
            ))
            assert {k: v for k, v in welcome.items() if k not in {"server_nonce", "proof"}} == dict(
                ok=True, protocol_version="0.3",
                capabilities=["mutual_auth", "voice_archive", "voice_review"],
                review_interval=10,
            )
            writer.write(_batch().model_dump_json().encode() + b"\n")
            await writer.drain()
            assert type(parse_voice_event(await reader.readline())) is VoiceArchiveAckEvent
            writer.write(VoiceReviewEvent(
                protocol_version="0.3", type="voice_review", conversation_id="conv",
                generation=0, seq_from=0, seq_through=1, memory=True, skills=True, closing=False,
            ).model_dump_json().encode() + b"\n")
            await writer.drain()
            assert type(parse_voice_event(await reader.readline())) is VoiceReviewAckEvent
        finally:
            writer.close()
            await writer.wait_closed()
