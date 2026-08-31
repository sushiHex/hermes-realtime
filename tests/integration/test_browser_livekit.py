from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from pathlib import Path
from types import MethodType

import pytest
from livekit import rtc

from hermes_realtime.client import BrowserClientRuntime
from hermes_realtime.conversation import ReconnectSafeConversationWorker
from hermes_realtime.livekit import (
    LiveKitConnection,
    LiveKitConversationWorker,
    LiveKitRoomPeer,
)
from hermes_realtime.speech import ParticipantAudioFrame

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_LIVEKIT_LOCAL") != "1",
        reason="set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit",
    ),
]


def _local_connection() -> LiveKitConnection:
    return LiveKitConnection(
        url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
        api_secret=os.getenv("LIVEKIT_API_SECRET", "local-" + ("x" * 32)),
    )


def _available_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


async def _raw_request(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


async def _approval(
    identity: str,
    generation: int,
    sequence: int,
    approval_id: str,
    decision: str,
) -> None:
    del identity, generation, sequence, approval_id, decision


async def test_browser_bootstrap_token_joins_and_delivers_bound_microphone_pcm(
    tmp_path: Path,
) -> None:
    connection = _local_connection()
    room_name = f"hermes-browser-{uuid.uuid4().hex}"
    received = asyncio.Event()
    packet: ParticipantAudioFrame | None = None
    runtime_worker = object.__new__(ReconnectSafeConversationWorker)

    async def bind(self: object, participant_identity: str) -> int:
        del self, participant_identity
        return 1

    async def run_source(
        self: object,
        source: LiveKitRoomPeer,
        generation: int,
        *,
        receive_timeout_seconds: float = 1.0,
    ) -> None:
        nonlocal packet
        del self, generation
        while True:
            try:
                candidate = await source.receive_participant_audio(
                    timeout_seconds=receive_timeout_seconds
                )
            except TimeoutError:
                continue
            if any(candidate.frame.pcm):
                packet = candidate
                received.set()
                return

    async def submit_final_transcript(
        self: object,
        *,
        participant_identity: str,
        session_generation: int,
        text: str,
    ) -> None:
        del self, participant_identity, session_generation, text

    async def close_runtime(self: object) -> None:
        del self

    async def close_binding(self: object) -> None:
        del self

    runtime_worker.bind = MethodType(bind, runtime_worker)  # type: ignore[method-assign]
    runtime_worker.run_source = MethodType(  # type: ignore[method-assign]
        run_source, runtime_worker
    )
    runtime_worker.submit_final_transcript = MethodType(  # type: ignore[method-assign]
        submit_final_transcript,
        runtime_worker,
    )
    runtime_worker.close_binding = MethodType(  # type: ignore[method-assign]
        close_binding,
        runtime_worker,
    )
    runtime_worker.close = MethodType(  # type: ignore[method-assign]
        close_runtime, runtime_worker
    )
    worker = LiveKitConversationWorker(runtime=runtime_worker)

    static_root = tmp_path / "static"
    (static_root / "assets").mkdir(parents=True)
    (static_root / "index.html").write_text("client", encoding="utf-8")
    (static_root / "assets" / "app.js").write_text("void 0;", encoding="utf-8")
    (static_root / "assets" / "styles.css").write_text("body{}", encoding="utf-8")
    port = _available_port()
    browser = BrowserClientRuntime(
        connection=connection,
        room_name=room_name,
        worker=worker,
        worker_identity="worker_hermes_runtime",
        approval=_approval,
        static_root=static_root,
        port=port,
        capability_factory=lambda: "B" * 43,
        browser_identity_factory=lambda: "browser_0123456789abcdef",
    )
    room = rtc.Room()

    async with asyncio.timeout(30):
        try:
            launch_url = await browser.start()
            origin = launch_url.split("/#", 1)[0]
            request = (
                b"POST /api/v1/bootstrap HTTP/1.1\r\n"
                + f"Host: 127.0.0.1:{port}\r\n".encode()
                + f"Origin: {origin}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                + b"Content-Length: 0\r\n"
                + b"Authorization: Bearer "
                + (b"B" * 43)
                + b"\r\n\r\n"
            )
            response = await _raw_request(port, request)
            assert response.startswith(b"HTTP/1.1 200 OK\r\n")
            credential = json.loads(response.split(b"\r\n\r\n", 1)[1])

            await room.connect(connection.url, credential["token"])
            source = rtc.AudioSource(48_000, 1)
            track = rtc.LocalAudioTrack.create_audio_track("browser-microphone", source)
            await room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            pcm = b"\x00\x20" * 480
            frame = rtc.AudioFrame(
                data=pcm,
                sample_rate=48_000,
                num_channels=1,
                samples_per_channel=480,
            )
            for _ in range(100):
                await source.capture_frame(frame)
                if received.is_set():
                    break
            await received.wait()

            assert packet is not None
            assert packet.participant_identity == credential["participantIdentity"]
            assert any(packet.frame.pcm)
        finally:
            await room.disconnect()
            await browser.close()
