from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import struct
import urllib.parse
import uuid
from collections.abc import AsyncIterator

import pytest
from livekit import rtc

from hermes_realtime.conversation import ConversationContextSnapshot
from hermes_realtime.launcher import build_local_conversation_launcher
from hermes_realtime.speech import AudioFrame, SpeechChunk, Transcript

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_LIVEKIT_LOCAL") != "1",
        reason="set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit",
    ),
]


def _tone_frame() -> AudioFrame:
    sample_rate_hz = 48_000
    samples = [
        round(12_000 * math.sin(2 * math.pi * 440.0 * index / sample_rate_hz))
        for index in range(480)
    ]
    return AudioFrame(
        pcm=struct.pack(f"<{len(samples)}h", *samples),
        sample_rate_hz=sample_rate_hz,
        channels=1,
    )


class _DeterministicInference:
    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "Local voice is online."

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return


class _ToneSynthesizer:
    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id="launcher_tone_1",
            text=text,
            audio=_tone_frame(),
        )

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return


class _UnexpectedTranscriber:
    def __init__(self, **kwargs: object) -> None:
        del kwargs

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        raise AssertionError("typed-input launcher path unexpectedly invoked microphone STT")

    async def finish_utterance(self) -> Transcript | None:
        raise AssertionError("typed-input launcher path unexpectedly finished microphone STT")

    async def cancel(self) -> None:
        return


def _available_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


async def _request(
    *,
    port: int,
    origin: str,
    path: str,
    bearer: str,
    body: bytes,
) -> tuple[int, dict[str, object]]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    headers = [
        f"POST {path} HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        f"Origin: {origin}",
        f"Authorization: Bearer {bearer}",
        f"Content-Length: {len(body)}",
    ]
    if body:
        headers.append("Content-Type: application/json")
    wire = ("\r\n".join(headers) + "\r\n\r\n").encode() + body
    writer.write(wire)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    head, payload = response.split(b"\r\n\r\n", 1)
    status = int(head.split(b" ", 2)[1])
    decoded = json.loads(payload)
    assert type(decoded) is dict
    return status, decoded


async def test_local_conversation_launcher_typed_turn_reaches_livekit_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hermes_realtime.launcher.OllamaStreamingInference",
        _DeterministicInference,
    )
    monkeypatch.setattr("hermes_realtime.launcher.EdgeTtsSynthesizer", _ToneSynthesizer)
    monkeypatch.setattr(
        "hermes_realtime.launcher.FasterWhisperTranscriber",
        _UnexpectedTranscriber,
    )
    suffix = uuid.uuid4().hex[:10]
    port = _available_port()

    launcher = build_local_conversation_launcher(
        livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "local-" + ("x" * 32)),
        room_name=f"m61-launcher-{suffix}",
        worker_identity=f"worker_{suffix}",
        browser_port=port,
    )
    room = rtc.Room()
    audio_received = asyncio.Event()
    audio_formats: set[tuple[int, int]] = set()
    audio_tasks: set[asyncio.Task[None]] = set()

    async def consume_audio(track: rtc.RemoteAudioTrack) -> None:
        stream = rtc.AudioStream(track)
        try:
            async for event in stream:
                if any(event.frame.data):
                    audio_formats.add((event.frame.sample_rate, event.frame.num_channels))
                    audio_received.set()
                    return
        finally:
            await stream.aclose()

    @room.on("track_subscribed")  # type: ignore[untyped-decorator]
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        del publication
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity != f"worker_{suffix}":
            return
        assert isinstance(track, rtc.RemoteAudioTrack)
        task = asyncio.create_task(consume_audio(track))
        audio_tasks.add(task)
        task.add_done_callback(audio_tasks.discard)

    try:
        launch_url = await launcher.start()
        parsed = urllib.parse.urlsplit(launch_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        capability_values = urllib.parse.parse_qs(parsed.fragment).get("bootstrap")
        assert capability_values is not None and len(capability_values) == 1
        status, credential = await _request(

            port=port,
            origin=origin,
            path="/api/v1/bootstrap",
            bearer=capability_values[0],
            body=b"",
        )
        assert status == 200
        token = credential["token"]
        assert type(token) is str
        await room.connect(str(credential["url"]), token)

        status, _ = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"Reply with exactly: Local voice is online."}',
        )
        assert status == 202
        await asyncio.wait_for(audio_received.wait(), timeout=45)
        assert audio_formats == {(48_000, 1)}

        assistant_texts: list[str] = []
        after = 0
        async with asyncio.timeout(45):
            while not assistant_texts:
                body = json.dumps({"after": after}, separators=(",", ":")).encode()
                status, event_payload = await _request(
                    port=port,
                    origin=origin,
                    path="/api/v1/events",
                    bearer=token,
                    body=body,
                )
                assert status == 200
                events = event_payload["events"]
                assert type(events) is list
                for event in events:
                    assert type(event) is dict
                    after = max(after, int(event["sequence"]))
                    if event["kind"] == "transcript_final":
                        data = event["data"]
                        assert type(data) is dict
                        if data.get("role") == "assistant":
                            assistant_texts.append(str(data["text"]))
                if not assistant_texts:
                    await asyncio.sleep(0.05)

        assert assistant_texts == ["Local voice is online."]
        status, stopped = await _request(
            port=port,
            origin=origin,
            path="/api/v1/stop",
            bearer=token,
            body=b"",
        )
        assert status == 200
        assert stopped == {"stopped": True, "version": 1}
    finally:
        await room.disconnect()
        for task in tuple(audio_tasks):
            task.cancel()
        await asyncio.gather(*audio_tasks, return_exceptions=True)
        await launcher.close()
