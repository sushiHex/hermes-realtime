import asyncio
import gc
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from hermes_realtime.integration.bridge import BridgeProtocolError, LocalHermesBridgeClient
from hermes_realtime.protocol import (
    VoiceMemoryEvent,
    VoiceMemoryRefusedEvent,
    VoiceMemorySnapshotEvent,
)
from tests.integration.test_bridge_voice import (
    _ATTESTATION,
    _TOKEN,
    _ReviewVoice,
    _server,
    _Voice,
)


class MemoryVoice(_Voice):
    memory_available = True

    def __init__(self) -> None:
        super().__init__()
        self.changed = asyncio.Event()
        self.closed = asyncio.Event()

    async def memory(self, request: VoiceMemoryEvent) -> AsyncIterator[VoiceMemorySnapshotEvent]:
        try:
            for revision in range(2):
                if revision:
                    await self.changed.wait()
                yield VoiceMemorySnapshotEvent(
                    protocol_version="0.4", type="voice_memory_snapshot",
                    conversation_id=request.conversation_id, generation=request.generation,
                    revision=revision, memory="", user="", truncated=False,
                )
            await asyncio.Event().wait()
        finally:
            self.closed.set()


@pytest.mark.asyncio
async def test_memory_stream_is_negotiated_and_disconnect_joins_subscription() -> None:
    voice = MemoryVoice()
    server = _server(voice)
    await server.start()
    client = await LocalHermesBridgeClient.connect(
        host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
        capabilities=("voice_memory",),
    )
    try:
        assert client.capabilities == frozenset({"voice_memory"})
        event = VoiceMemoryEvent(protocol_version="0.4", type="voice_memory",
                                 conversation_id="conv", generation=0)
        stream = client.memory(event)
        assert (await anext(stream)).revision == 0
        voice.changed.set()
        assert (await anext(stream)).revision == 1
        await client.close()
        await asyncio.wait_for(voice.closed.wait(), 2)
        await stream.aclose()
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_memory_subscription_preserves_review_and_runtime_negotiation() -> None:
    class CompleteVoice(MemoryVoice, _ReviewVoice):
        pass

    voice = CompleteVoice()
    capabilities = ("voice_archive", "voice_review", "voice_memory", "runtime_attestation")
    async with (
        _server(voice, runtime=_ATTESTATION) as server,
        await LocalHermesBridgeClient.connect(
            host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
            capabilities=capabilities,
        ) as client,
    ):
        assert client.capabilities == frozenset(capabilities)
        assert client.runtime == _ATTESTATION
        assert client.review_interval == 10
        stream = client.memory(VoiceMemoryEvent(
            protocol_version="0.4", type="voice_memory",
            conversation_id="conv", generation=0,
        ))
        assert (await anext(stream)).revision == 0
        await stream.aclose()
    await asyncio.wait_for(voice.closed.wait(), 2)


@pytest.mark.asyncio
async def test_memory_disconnect_bounds_uncooperative_producer_but_keeps_ownership(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SlowMemoryVoice(MemoryVoice):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()
            self.producer: asyncio.Task[object] | None = None

        async def memory(
            self, request: VoiceMemoryEvent,
        ) -> AsyncIterator[VoiceMemorySnapshotEvent]:
            self.producer = asyncio.current_task()
            try:
                yield VoiceMemorySnapshotEvent(
                    protocol_version="0.4", type="voice_memory_snapshot",
                    conversation_id=request.conversation_id, generation=request.generation,
                    revision=0, memory="", user="", truncated=False,
                )
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    await self.release.wait()
                    raise RuntimeError("late memory failure") from None
            finally:
                self.finished.set()

    voice = SlowMemoryVoice()
    server = _server(voice)
    server._shutdown_timeout = 0.05
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    await server.start()
    client = await LocalHermesBridgeClient.connect(
        host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
        capabilities=("voice_memory",),
    )
    try:
        request = VoiceMemoryEvent(
            protocol_version="0.4", type="voice_memory",
            conversation_id="conv", generation=0,
        )
        stream = client.memory(request)
        assert (await anext(stream)).revision == 0
        assert voice.producer is not None
        handler = next(task for task in server._handler_tasks if task is not voice.producer)
        await client.close()
        await asyncio.wait_for(voice.cancelled.wait(), 1)
        await asyncio.wait_for(asyncio.shield(handler), 0.5)
        assert voice.producer in server._handler_tasks
        assert not voice.finished.is_set()
        assert (
            '[voice-memory-stream] {"refusal":"shutdown_deadline","tasks":1,"version":1}'
            in caplog.text
        )
        voice.release.set()
        await asyncio.wait_for(voice.finished.wait(), 1)
        await stream.aclose()
        await asyncio.sleep(0)
        assert voice.producer not in server._handler_tasks
        voice.producer = None
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled
    finally:
        voice.release.set()
        await client.close()
        await server.close()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("available,method", [(False, True), (1, True), (True, False)])
async def test_partial_memory_capability_is_not_offered(available, method) -> None:
    voice = MemoryVoice()
    voice.memory_available = available
    if not method:
        voice.memory = None
    async with _server(voice) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
            capabilities=("voice_memory",),
        )
        try:
            assert not client.capabilities
            event = VoiceMemoryEvent(protocol_version="0.4", type="voice_memory",
                                     conversation_id="conv", generation=0)
            with pytest.raises(BridgeProtocolError):
                await anext(client.memory(event))
        finally:
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("conversation_id", "other"), ("generation", 1),
                                        ("type", "voice_memory")])
async def test_memory_client_rejects_other_binding_or_event(field, value) -> None:
    class Writer:
        def write(self, data):
            pass

        async def drain(self):
            pass

    reader = asyncio.StreamReader()
    event = VoiceMemoryEvent(protocol_version="0.4", type="voice_memory",
                             conversation_id="conv", generation=0)
    reply = VoiceMemorySnapshotEvent(
        protocol_version="0.4", type="voice_memory_snapshot", conversation_id="conv",
        generation=0, revision=0, memory="", user="", truncated=False,
    ).model_dump()
    reply[field] = value
    if field == "type":
        reply = event.model_dump()
    import json
    reader.feed_data(json.dumps(reply).encode() + b"\n")
    reader.feed_eof()
    client = LocalHermesBridgeClient(reader, Writer())
    client._capabilities = frozenset({"voice_memory"})
    with pytest.raises(BridgeProtocolError, match="invalid memory response"):
        await anext(client.memory(event))


@pytest.mark.asyncio
async def test_server_refuses_unnegotiated_subscription() -> None:
    voice = MemoryVoice()
    async with _server(voice) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
        )
        try:
            event = VoiceMemoryEvent(protocol_version="0.4", type="voice_memory",
                                     conversation_id="conv", generation=0)
            await client._send_json(event.model_dump())
            assert await asyncio.wait_for(client._reader.readline(), 1) == b""
            assert not voice.closed.is_set()
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_memory_stream_rejects_a_second_request_and_joins_producer() -> None:
    voice = MemoryVoice()
    async with _server(voice) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
            capabilities=("voice_memory",),
        )
        try:
            request = VoiceMemoryEvent(
                protocol_version="0.4", type="voice_memory",
                conversation_id="conv", generation=0,
            )
            await client._send_json(request.model_dump())
            assert json.loads(await asyncio.wait_for(client._reader.readline(), 1))["revision"] == 0
            await client._send_json(request.model_dump())
            assert await asyncio.wait_for(client._reader.readline(), 1) == b""
            await asyncio.wait_for(voice.closed.wait(), 1)
        finally:
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["type", "conversation_id", "generation"])
async def test_server_never_publishes_an_invalid_memory_reply(kind: str) -> None:
    class InvalidMemoryVoice(MemoryVoice):
        async def memory(
            self, request: VoiceMemoryEvent
        ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]:
            if kind == "type":
                yield cast(Any, request)
            else:
                yield VoiceMemorySnapshotEvent(
                    protocol_version="0.4", type="voice_memory_snapshot",
                    conversation_id=(
                        "other" if kind == "conversation_id" else request.conversation_id
                    ),
                    generation=1 if kind == "generation" else request.generation,
                    revision=0, memory="", user="", truncated=False,
                )

    async with _server(InvalidMemoryVoice()) as server:
        client = await LocalHermesBridgeClient.connect(
            host=server.host, port=server.port, token=_TOKEN, participant_id="memory",
            capabilities=("voice_memory",),
        )
        try:
            request = VoiceMemoryEvent(
                protocol_version="0.4", type="voice_memory",
                conversation_id="conv", generation=0,
            )
            await client._send_json(request.model_dump())
            assert await asyncio.wait_for(client._reader.readline(), 1) == b""
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_client_requires_exact_request_and_closed_stream_is_an_error() -> None:
    class Writer:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.sent.append(data)

        async def drain(self) -> None:
            pass

    reader = asyncio.StreamReader()
    writer = Writer()
    client = LocalHermesBridgeClient(reader, cast(Any, writer))
    request = VoiceMemoryEvent(
        protocol_version="0.4", type="voice_memory",
        conversation_id="conv", generation=0,
    )
    with pytest.raises(BridgeProtocolError, match="did not advertise"):
        await asyncio.wait_for(anext(client.memory(request)), 0.2)
    assert writer.sent == []

    client._capabilities = frozenset({"voice_memory"})
    wrong = VoiceMemorySnapshotEvent(
        protocol_version="0.4", type="voice_memory_snapshot",
        conversation_id="conv", generation=0, revision=0,
        memory="", user="", truncated=False,
    )
    with pytest.raises(TypeError, match="exact"):
        await asyncio.wait_for(anext(client.memory(cast(Any, wrong))), 0.2)
    assert writer.sent == []

    reader.feed_eof()
    with pytest.raises(BridgeProtocolError, match="closed"):
        await anext(client.memory(request))
