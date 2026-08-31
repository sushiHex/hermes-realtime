"""Synthetic full-host audio transport qualification.

This test feeds the provenance-tracked SAPI ``stop`` fixture through the real
local LiveKit SDK ``rtc.AudioSource.capture_frame()`` boundary.  It qualifies
synthetic PCM transport, deterministic VAD/STT endpointing, and public marker
ordering only.  It makes no physical transducer/AEC, subjective audibility, or
naturalness claim.  The synthetic echo/noise sweep and mixed near/far barge-in
nodes remain deliberately deferred rather than represented by flaky tests.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import uuid
import wave
from pathlib import Path

import pytest
from livekit import rtc

from hermes_realtime.host_launcher import build_local_host_launcher
from hermes_realtime.speech import Transcript
from tests.integration.test_qualification_full_host_ingress import (
    _available_port,
    _connect_and_activate,
    _Inference,
    _Presence,
    _Synthesizer,
    _Transcriber,
    _Vad,
    _wait_event,
)
from tests.support.qualification import InProcessQualificationComposition

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(sys.platform != "win32", reason="full-host qualification is Windows-only"),
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_LIVEKIT_LOCAL") != "1",
        reason="set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit",
    ),
]

_STOP_FIXTURE = Path(__file__).parents[1] / "fixtures" / "audio" / "sapi_stop.wav"
_STOP_FIXTURE_SHA256 = "3ef6a5a2465c5e4669e1998a3cbfdbdf7a15d39e81f6eb5ecd0ee47cdfc84233"
_SAMPLE_RATE = 48_000
_FRAME_SAMPLES = 480


class _SapiStopTranscriber(_Transcriber):
    """Keep the ingress helper's deterministic endpoint, but name its SAPI stimulus."""

    async def finish_utterance(self) -> Transcript | None:
        if not self._armed:
            return None
        self._armed = False
        return Transcript("stop", final=True)


def _sapi_stop_frames() -> tuple[rtc.AudioFrame, ...]:
    """Verify provenance and deterministically upsample mono signed-16-bit fixture PCM."""

    assert hashlib.sha256(_STOP_FIXTURE.read_bytes()).hexdigest() == _STOP_FIXTURE_SHA256
    with wave.open(str(_STOP_FIXTURE), "rb") as source:
        assert (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
        ) == (1, 2, 22_050)
        original = struct.unpack(f"<{source.getnframes()}h", source.readframes(source.getnframes()))
    upsampled = tuple(original[index * 22_050 // _SAMPLE_RATE] for index in range(
        len(original) * _SAMPLE_RATE // 22_050
    ))
    frames: list[rtc.AudioFrame] = []
    for offset in range(0, len(upsampled), _FRAME_SAMPLES):
        samples = upsampled[offset : offset + _FRAME_SAMPLES]
        if len(samples) < _FRAME_SAMPLES:
            samples += (0,) * (_FRAME_SAMPLES - len(samples))
        frames.append(
            rtc.AudioFrame(
                data=struct.pack(f"<{_FRAME_SAMPLES}h", *samples),
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=_FRAME_SAMPLES,
            )
        )
    return tuple(frames)


async def test_sapi_stop_pcm_crosses_full_host_endpoint_and_orders_public_markers() -> None:
    """Node 1: local SDK PCM reaches the owner-bound full host without human speech."""

    suffix, port = uuid.uuid4().hex[:10], _available_port()
    composition = InProcessQualificationComposition()
    transcriber = _SapiStopTranscriber()
    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"synthetic-audio-{suffix}",
            worker_identity=f"worker_{suffix}",
        ),
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
    )
    running: object | None = None
    room = rtc.Room()
    try:
        running = await composition.start_host(registration)
        origin, token, source = await _connect_and_activate(
            port=port,
            launch_url=running.url,
            room=room,
        )
        ready = await _wait_event(port=port, origin=origin, token=token, kind="voice_input_ready")
        ready_sequence = ready["sequence"]
        assert type(ready_sequence) is int

        transcriber._armed = True
        for frame in _sapi_stop_frames():
            await source.capture_frame(frame)

        user_final = await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="transcript_final",
            after=ready_sequence,
            timeout=30,
        )
        assert user_final["data"] == {"role": "user", "text": "stop"}
        user_sequence = user_final["sequence"]
        assert type(user_sequence) is int
        first_token = await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="first_foreground_token",
            after=user_sequence,
            timeout=30,
        )
        token_sequence = first_token["sequence"]
        assert type(token_sequence) is int and token_sequence > user_sequence
        first_audio = await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="first_playable_audio",
            after=token_sequence,
            timeout=30,
        )
        audio_sequence = first_audio["sequence"]
        assert type(audio_sequence) is int and audio_sequence > token_sequence
        completed = await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="assistant_turn_completed",
            after=audio_sequence,
            timeout=30,
        )
        completed_sequence = completed["sequence"]
        assert type(completed_sequence) is int and completed_sequence > audio_sequence
        assert transcriber._armed is False
    finally:
        await room.disconnect()
        if running is not None:
            await composition.close_host(running)  # type: ignore[arg-type]
    assert composition.trace.status().trace_complete
