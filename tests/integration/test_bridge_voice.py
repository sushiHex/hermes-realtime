"""Bridge protocol 0.3: the capability hello and voice archive routing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from hermes_realtime.integration import (
    BridgeAuthenticationError,
    BridgeProtocolError,
    EventSequencer,
    HermesCompletionRouter,
    HermesIntegrationService,
    LocalHermesBridgeClient,
    LocalHermesBridgeServer,
    SessionBindings,
)
from hermes_realtime.protocol import (
    RuntimeAttestation,
    VoiceArchiveAckEvent,
    VoiceArchiveEvent,
    VoiceArchiveRefusedEvent,
    VoiceArchiveRow,
    VoiceReviewAckEvent,
    VoiceReviewEvent,
    VoiceReviewRefusedEvent,
)

_TOKEN = "correct-test-token-with-sufficient-entropy"


class _Dispatcher:
    async def dispatch(self, command: object) -> str:
        raise AssertionError("no work is dispatched here")

    async def cancel(self, run_id: str) -> bool:
        raise AssertionError("no work is cancelled here")


def _server(
    voice: Any = None, runtime: RuntimeAttestation | None = None
) -> LocalHermesBridgeServer:
    from datetime import UTC, datetime

    sequencer = EventSequencer()
    now = lambda: datetime(2026, 9, 26, tzinfo=UTC)  # noqa: E731
    service = HermesIntegrationService(
        bindings=SessionBindings(),
        dispatcher=_Dispatcher(),  # type: ignore[arg-type]
        canceller=_Dispatcher(),  # type: ignore[arg-type]
        event_id_factory=lambda: "evt_1",
        clock=now,
        sequencer=sequencer,
    )
    completions = HermesCompletionRouter(
        sequencer=sequencer, event_id_factory=lambda: "evt_2", clock=now
    )
    return LocalHermesBridgeServer(
        service=service,
        completions=completions,
        token=_TOKEN,
        voice=voice,
        runtime=runtime,
    )


def _batch(seq_through: int = 1) -> VoiceArchiveEvent:
    return VoiceArchiveEvent(
        protocol_version="0.3",
        type="voice_archive",
        conversation_id="conv",
        generation=0,
        seq_from=0,
        seq_through=seq_through,
        rows=[
            VoiceArchiveRow(
                seq=0, role="user", text="Hi", interrupted=False, ts=1.0, gap_before=None
            ),
            VoiceArchiveRow(
                seq=1, role="assistant", text="Hello", interrupted=True, ts=2.0, gap_before=None
            ),
        ],
    )


class _Voice:
    def __init__(self, reply: str = "ack") -> None:
        self.reply = reply
        self.events: list[VoiceArchiveEvent] = []

    async def archive(
        self, event: VoiceArchiveEvent
    ) -> VoiceArchiveAckEvent | VoiceArchiveRefusedEvent | None:
        self.events.append(event)
        fields: dict[str, Any] = {
            "protocol_version": "0.3",
            "conversation_id": event.conversation_id,
            "generation": event.generation,
            "seq_from": event.seq_from,
            "seq_through": event.seq_through,
        }
        if self.reply == "ack":
            return VoiceArchiveAckEvent(type="voice_archive_ack", **fields)
        if self.reply == "refuse":
            return VoiceArchiveRefusedEvent(
                type="voice_archive_refused", category="partition", **fields
            )
        return None


class _ReviewVoice(_Voice):
    review_interval = 10

    def __init__(self) -> None:
        super().__init__()
        self.reviews: list[VoiceReviewEvent] = []

    async def review(
        self, event: VoiceReviewEvent
    ) -> VoiceReviewAckEvent | VoiceReviewRefusedEvent | None:
        self.reviews.append(event)
        return VoiceReviewAckEvent(
            protocol_version="0.3",
            type="voice_review_ack",
            conversation_id=event.conversation_id,
            generation=event.generation,
            seq_from=event.seq_from,
            seq_through=event.seq_through,
            closing=event.closing,
            review_id="vr_0123456789abcdef0123456789abcdef",
            status="accepted",
        )


async def _hello(server: LocalHermesBridgeServer, hello: dict[str, object]) -> object:
    reader, writer = await asyncio.open_connection(server.host, server.port)
    try:
        writer.write(json.dumps(hello).encode("utf-8") + b"\n")
        await writer.drain()
        return json.loads(await reader.readline())
    finally:
        writer.close()


def _valid_hello(**overrides: object) -> dict[str, object]:
    hello: dict[str, object] = {
        "token": _TOKEN,
        "participant_id": "voice-archive",
        "protocol_version": "0.3",
        "capabilities": ["voice_archive"],
    }
    hello.update(overrides)
    return hello


@pytest.mark.asyncio
async def test_the_hello_negotiates_voice_archive_only_when_the_companion_offers_it() -> None:
    async with _server(voice=_Voice()) as offering, _server() as plain:
        offered = await LocalHermesBridgeClient.connect(
            host=offering.host,
            port=offering.port,
            token=_TOKEN,
            participant_id="voice-archive",
            capabilities=("voice_archive",),
        )
        withheld = await LocalHermesBridgeClient.connect(
            host=plain.host,
            port=plain.port,
            token=_TOKEN,
            participant_id="voice-archive",
            capabilities=("voice_archive",),
        )
        unasked = await LocalHermesBridgeClient.connect(
            host=offering.host,
            port=offering.port,
            token=_TOKEN,
            participant_id="participant_001",
        )
        try:
            assert offered.capabilities == frozenset({"voice_archive"})
            assert withheld.capabilities == frozenset()
            assert unasked.capabilities == frozenset()
        finally:
            for client in (offered, withheld, unasked):
                await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hello",
    [
        pytest.param({"token": _TOKEN, "participant_id": "p1"}, id="a-0.1-hello"),
        pytest.param(_valid_hello(protocol_version="0.4"), id="wrong-version"),
        pytest.param(_valid_hello(capabilities=["voice_forget"]), id="unknown-capability"),
        pytest.param(_valid_hello(capabilities=["voice_archive", "voice_archive"]), id="repeated"),
        pytest.param(_valid_hello(capabilities="voice_archive"), id="capabilities-not-a-list"),
        pytest.param(_valid_hello(extra=1), id="extra-key"),
        pytest.param(_valid_hello(token="wrong-token-wrong-token-wrong"), id="wrong-token"),
    ],
)
async def test_every_malformed_hello_is_refused(hello: dict[str, object]) -> None:
    async with _server(voice=_Voice()) as server:
        assert await _hello(server, hello) == {"ok": False}


_BRIDGE_MARKER = "[hermes-bridge-hello] "


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hello", "category"),
    [
        pytest.param(_valid_hello(protocol_version="0.4"), "version", id="version"),
        pytest.param(_valid_hello(capabilities=["voice_forget"]), "capability", id="unknown"),
        pytest.param(
            _valid_hello(capabilities=["voice_archive", "voice_archive"]),
            "capability",
            id="repeated",
        ),
        pytest.param({"token": _TOKEN, "participant_id": "p1"}, "shape", id="shape"),
        pytest.param(_valid_hello(token="wrong-token-wrong-token-wrong"), "token", id="token"),
    ],
)
async def test_a_refused_hello_leaves_one_marker_with_its_category_only(
    hello: dict[str, object], category: str, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _server(voice=_Voice()) as server:
        assert await _hello(server, hello) == {"ok": False}
        await asyncio.sleep(0.05)

    output = capsys.readouterr().out
    markers = [
        json.loads(line.removeprefix(_BRIDGE_MARKER))
        for line in output.splitlines()
        if line.startswith(_BRIDGE_MARKER)
    ]
    assert markers == [{"refusal": category, "version": 1}]
    assert _TOKEN not in output and "voice-archive" not in output and "p1" not in output


@pytest.mark.asyncio
async def test_the_welcome_names_the_version_and_the_offered_capabilities() -> None:
    async with _server(voice=_Voice()) as server:
        assert await _hello(server, _valid_hello()) == {
            "ok": True,
            "protocol_version": "0.3",
            "capabilities": ["voice_archive"],
        }
        assert await _hello(server, _valid_hello(capabilities=[])) == {
            "ok": True,
            "protocol_version": "0.3",
            "capabilities": [],
        }


@pytest.mark.asyncio
async def test_voice_archive_is_answered_by_the_companion_without_a_participant_binding() -> None:
    voice = _Voice()
    async with _server(voice=voice) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="voice-archive",
            capabilities=("voice_archive",),
        )
        try:
            reply = await client.archive(_batch())
            assert type(reply) is VoiceArchiveAckEvent
            assert (reply.seq_from, reply.seq_through) == (0, 1)
            voice.reply = "refuse"
            refused = await client.archive(_batch())
            assert type(refused) is VoiceArchiveRefusedEvent
            assert refused.category == "partition"
        finally:
            await client.close()
    assert voice.events == [_batch(), _batch()]


@pytest.mark.asyncio
async def test_review_is_private_and_welcome_binds_verified_interval() -> None:
    voice = _ReviewVoice()
    async with _server(voice=voice) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="voice-review",
            capabilities=("voice_review",),
        )
        try:
            assert client.capabilities == frozenset({"voice_review"})
            assert client.review_interval == 10
            event = VoiceReviewEvent(
                protocol_version="0.3",
                type="voice_review",
                conversation_id="conv",
                generation=0,
                seq_from=0,
                seq_through=1,
                memory=True,
                skills=True,
                closing=True,
            )
            reply = await client.review(event)
            assert type(reply) is VoiceReviewAckEvent
            assert reply.closing
            assert voice.reviews == [event]
            assert voice.events == []
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_review_without_negotiated_capability_never_reaches_companion() -> None:
    voice = _ReviewVoice()
    async with _server(voice=voice) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="voice-review",
            capabilities=(),
        )
        try:
            event = VoiceReviewEvent(
                protocol_version="0.3",
                type="voice_review",
                conversation_id="conv",
                generation=0,
                seq_from=0,
                seq_through=0,
                memory=True,
                skills=True,
                closing=False,
            )
            with pytest.raises(BridgeProtocolError):
                await client.review(event)
            assert voice.reviews == []
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_raw_unnegotiated_review_is_rejected_before_service() -> None:
    voice = _ReviewVoice()
    async with _server(voice=voice) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        writer.write(json.dumps(_valid_hello(capabilities=[])).encode() + b"\n")
        await writer.drain()
        assert json.loads(await reader.readline())["capabilities"] == []
        event = VoiceReviewEvent(
            protocol_version="0.3",
            type="voice_review",
            conversation_id="conv",
            generation=0,
            seq_from=0,
            seq_through=0,
            memory=True,
            skills=True,
            closing=False,
        )
        writer.write(event.model_dump_json().encode() + b"\n")
        await writer.drain()
        assert await reader.readline() == b""
        writer.close()
    assert voice.reviews == []


@pytest.mark.asyncio
async def test_client_guard_sends_zero_unnegotiated_review_bytes() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()

    class Writer:
        writes = 0

        def write(self, payload: bytes) -> None:
            del payload
            self.writes += 1

        async def drain(self) -> None:
            pass

    writer = Writer()
    client = LocalHermesBridgeClient(reader, writer)  # type: ignore[arg-type]
    event = VoiceReviewEvent(
        protocol_version="0.3",
        type="voice_review",
        conversation_id="conv",
        generation=0,
        seq_from=0,
        seq_through=0,
        memory=True,
        skills=True,
        closing=False,
    )
    with pytest.raises(BridgeProtocolError):
        await client.review(event)
    assert writer.writes == 0


@pytest.mark.asyncio
async def test_an_unknown_outcome_closes_the_connection_without_a_reply() -> None:
    async with _server(voice=_Voice(reply="unknown")) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="voice-archive",
            capabilities=("voice_archive",),
        )
        try:
            with pytest.raises(BridgeProtocolError):
                await asyncio.wait_for(client.archive(_batch()), timeout=2)
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_a_voice_event_without_the_negotiated_capability_closes_the_connection() -> None:
    voice = _Voice()
    async with _server(voice=voice) as server:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        writer.write(json.dumps(_valid_hello(capabilities=[])).encode("utf-8") + b"\n")
        await writer.drain()
        assert json.loads(await reader.readline())["ok"] is True
        writer.write(_batch().model_dump_json().encode("utf-8") + b"\n")
        await writer.drain()
        assert await reader.readline() == b""
        writer.close()
    assert voice.events == []


@pytest.mark.asyncio
async def test_the_client_never_sends_a_voice_event_the_companion_did_not_offer() -> None:
    async with _server() as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="voice-archive",
            capabilities=("voice_archive",),
        )
        try:
            with pytest.raises(BridgeProtocolError, match="capability"):
                await client.archive(_batch())
        finally:
            await client.close()


async def _fake_companion(welcome: object) -> tuple[asyncio.Server, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.write(json.dumps(welcome).encode("utf-8") + b"\n")
        await writer.drain()
        await reader.read()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "welcome",
    [
        pytest.param({"ok": True}, id="a-0.1-welcome"),
        pytest.param(
            {"ok": True, "protocol_version": "0.4", "capabilities": []}, id="wrong-version"
        ),
        pytest.param(
            {"ok": True, "protocol_version": "0.3", "capabilities": ["voice_archive"]},
            id="unrequested-capability",
        ),
        pytest.param(
            {"ok": True, "protocol_version": "0.3", "capabilities": [], "x": 1}, id="extra-key"
        ),
        pytest.param({"ok": False}, id="refused"),
    ],
)
async def test_the_client_refuses_a_welcome_it_did_not_ask_for(welcome: object) -> None:
    server, port = await _fake_companion(welcome)
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await LocalHermesBridgeClient.connect(
                host="127.0.0.1", port=port, token=_TOKEN, participant_id="p1"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [True, 0, 1.0, 1001])
async def test_review_welcome_refuses_unverified_interval(interval: object) -> None:
    server, port = await _fake_companion(
        {
            "ok": True,
            "protocol_version": "0.3",
            "capabilities": ["voice_review"],
            "review_interval": interval,
        }
    )
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await LocalHermesBridgeClient.connect(
                host="127.0.0.1",
                port=port,
                token=_TOKEN,
                participant_id="voice-review",
                capabilities=("voice_review",),
            )


_ATTESTATION = RuntimeAttestation(
    pid=1,
    hermes_version="0.21.0",
    hermes_commit="0123456789abcdef0123456789abcdef01234567",
    realtime_version="0.0.3",
    realtime_install="wheel",
    realtime_record="0123456789abcdef" * 4,
)


@pytest.mark.asyncio
async def test_the_welcome_attests_the_runtime_only_when_asked() -> None:
    async with _server(voice=_ReviewVoice(), runtime=_ATTESTATION) as server:
        asked = await _hello(server, _valid_hello(capabilities=["runtime_attestation"]))
        unasked = await _hello(server, _valid_hello())

    assert asked == {
        "ok": True,
        "protocol_version": "0.3",
        "capabilities": ["runtime_attestation"],
        "runtime": _ATTESTATION.model_dump(mode="json"),
    }
    assert unasked == {"ok": True, "protocol_version": "0.3", "capabilities": ["voice_archive"]}


@pytest.mark.asyncio
async def test_a_bridge_without_an_attestation_never_offers_one() -> None:
    async with _server(voice=_ReviewVoice()) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="voice-review",
            capabilities=("voice_review", "runtime_attestation"),
        )
        async with client:
            assert client.capabilities == frozenset({"voice_review"})
            assert client.runtime is None


@pytest.mark.asyncio
async def test_the_client_holds_the_attested_runtime() -> None:
    async with _server(voice=_ReviewVoice(), runtime=_ATTESTATION) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host,
            port=server.port,
            token=_TOKEN,
            participant_id="real-hermes-gate",
            capabilities=("voice_archive", "voice_review", "runtime_attestation"),
        )
        async with client:
            assert client.capabilities == frozenset(
                {"voice_archive", "voice_review", "runtime_attestation"}
            )
            assert client.runtime == _ATTESTATION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "welcome",
    [
        pytest.param(
            {"ok": True, "protocol_version": "0.3", "capabilities": ["runtime_attestation"]},
            id="negotiated-without-runtime",
        ),
        pytest.param(
            {
                "ok": True,
                "protocol_version": "0.3",
                "capabilities": [],
                "runtime": _ATTESTATION.model_dump(mode="json"),
            },
            id="runtime-not-negotiated",
        ),
        pytest.param(
            {
                "ok": True,
                "protocol_version": "0.3",
                "capabilities": ["runtime_attestation"],
                "runtime": _ATTESTATION.model_dump(mode="json") | {"pid": "1"},
            },
            id="malformed-runtime",
        ),
    ],
)
async def test_the_client_refuses_an_unverified_attestation(welcome: object) -> None:
    server, port = await _fake_companion(welcome)
    async with server:
        with pytest.raises(BridgeAuthenticationError):
            await LocalHermesBridgeClient.connect(
                host="127.0.0.1",
                port=port,
                token=_TOKEN,
                participant_id="real-hermes-gate",
                capabilities=("runtime_attestation",),
            )
