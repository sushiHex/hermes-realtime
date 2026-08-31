"""Real full-host consent qualification through HTTP, LiveKit, and SQLite evidence."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import struct
import sys
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from livekit import rtc

from hermes_realtime.conversation import ConversationContextSnapshot
from hermes_realtime.evidence.admission import (
    MAX_CANONICAL_RECORD_BYTES,
    MAX_QUEUE_CANONICAL_BYTES,
    MAX_QUEUE_PHYSICAL_ITEMS,
    MAX_QUEUE_RECORDS,
)
from hermes_realtime.evidence.models import DrainDisposition, TerminalReason
from hermes_realtime.host_launcher import build_local_host_launcher
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageObservationV1,
    CloseStageV1,
)
from hermes_realtime.speech import (
    AudioFrame,
    SpeechChunk,
    SpeechPresence,
    Transcript,
    VoiceActivity,
)
from tests.support.qualification import InProcessQualificationComposition

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(sys.platform != "win32", reason="evidence capture is Windows-only"),
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_LIVEKIT_LOCAL") != "1",
        reason="set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit",
    ),
]


def _available_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


async def _request(
    *, port: int, origin: str, path: str, bearer: str, body: bytes
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
    writer.write(("\r\n".join(headers) + "\r\n\r\n").encode() + body)
    await writer.drain()
    response = await asyncio.wait_for(reader.read(), timeout=10)
    writer.close()
    await writer.wait_closed()
    head, payload = response.split(b"\r\n\r\n", 1)
    decoded = json.loads(payload)
    assert type(decoded) is dict
    return int(head.split(b" ", 2)[1]), decoded


async def _events(
    *, port: int, origin: str, token: str, after: int
) -> tuple[int, list[dict[str, object]]]:
    status, payload = await _request(
        port=port,
        origin=origin,
        path="/api/v1/events",
        bearer=token,
        body=json.dumps({"after": after}, separators=(",", ":")).encode(),
    )
    events = payload["events"]
    assert type(events) is list
    assert all(type(event) is dict for event in events)
    return status, events


async def _wait_event(
    *,
    port: int,
    origin: str,
    token: str,
    kind: str,
    after: int = 0,
    timeout: float = 30,
) -> dict[str, object]:
    async with asyncio.timeout(timeout):
        while True:
            status, events = await _events(port=port, origin=origin, token=token, after=after)
            assert status == 200
            for event in events:
                sequence = event["sequence"]
                assert type(sequence) is int
                after = max(after, sequence)
                if event["kind"] == kind:
                    return event
            await asyncio.sleep(0.05)
    raise AssertionError(f"missing event {kind}")


def _pcm() -> bytes:
    return struct.pack("<480h", *([8_000] * 480))


class _Inference:
    def stream(self, snapshot: ConversationContextSnapshot, *, turn_id: str) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id)

    async def _stream(
        self, snapshot: ConversationContextSnapshot, turn_id: str
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "Qualification response."

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


class _Synthesizer:
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"chunk-{turn_id}",
            text=text,
            audio=AudioFrame(pcm=_pcm(), sample_rate_hz=48_000, channels=1),
        )

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


class _Transcriber:
    def __init__(self) -> None:
        self._armed = False

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        if not self._armed:
            return ()
        return (Transcript("microphone evidence", final=False),)

    async def finish_utterance(self) -> Transcript | None:
        if not self._armed:
            return None
        self._armed = False
        return Transcript("microphone evidence turn", final=True)

    async def cancel(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _Vad:
    required_pre_roll_frames = 0

    def __init__(self, transcriber: _Transcriber) -> None:
        self._transcriber = transcriber
        self._calls = 0

    def process(self, frame: AudioFrame) -> VoiceActivity:
        del frame
        if not self._transcriber._armed:
            self._calls = 0
            return VoiceActivity.SILENCE
        self._calls += 1
        return VoiceActivity.SPEECH_STARTED if self._calls == 1 else VoiceActivity.SPEECH_ENDED


class _Presence:
    def classify(self, frames: tuple[AudioFrame, ...]) -> SpeechPresence:
        assert frames
        return SpeechPresence.CONFIRMED_SPEECH

    async def close(self) -> None:
        return None


async def _connect_and_activate(
    *, port: int, launch_url: str, room: rtc.Room
) -> tuple[str, str, rtc.AudioSource]:
    parsed = urllib.parse.urlsplit(launch_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    bootstrap = urllib.parse.parse_qs(parsed.fragment)["bootstrap"][0]
    status, credential = await _request(
        port=port, origin=origin, path="/api/v1/bootstrap", bearer=bootstrap, body=b""
    )
    assert status == 200
    token = credential["token"]
    assert type(token) is str
    await room.connect(str(credential["url"]), token)
    source = rtc.AudioSource(48_000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("microphone-1", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    status, payload = await _request(
        port=port, origin=origin, path="/api/v1/media", bearer=token, body=b'{"mediaIncarnation":1}'
    )
    assert status == 202 and payload == {"mediaIncarnation": 1, "version": 1}
    return origin, token, source


@pytest.mark.parametrize("capture", [False, True], ids=["disabled", "capture-enabled-unconsented"])
async def test_full_host_typed_ingress_is_identical_without_consent(
    tmp_path: Path, capture: bool
) -> None:
    """Both no-capture and unconsented capture route typed ingress but persist no raw evidence."""
    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database_root = tmp_path / suffix
    database_root.mkdir()
    database = database_root / "capture-v1.sqlite3"
    composition = InProcessQualificationComposition()
    transcriber = _Transcriber()
    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=capture,
            evidence_database=database if capture else None,
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
        origin, token, _source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        status, payload = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"typed unconsented turn"}',
        )
        assert status == 202 and payload == {"sequence": 1, "version": 1}
        final = await _wait_event(port=port, origin=origin, token=token, kind="transcript_final")
        assert final["data"] == {"role": "user", "text": "typed unconsented turn"}
        if capture:
            # Capture is intentionally inert until the public consent arm commits.
            assert not database.exists()
    finally:
        await room.disconnect()
        if running is not None:
            await composition.close_host(running)  # type: ignore[arg-type]
    assert composition.trace.status().trace_complete


async def test_full_host_consented_typed_and_livekit_pcm_ingress_are_durable(
    tmp_path: Path,
) -> None:
    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database_root = tmp_path / suffix
    database_root.mkdir()
    database = database_root / "capture-v1.sqlite3"
    composition = InProcessQualificationComposition()
    transcriber = _Transcriber()
    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=True,
            evidence_database=database,
        ),
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
    )
    room = rtc.Room()
    running: object | None = None
    try:
        running = await composition.start_host(registration)
        origin, token, source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        capture = await _wait_event(port=port, origin=origin, token=token, kind="capture_status")
        capture_data = capture["data"]
        assert type(capture_data) is dict and capture_data["captureState"] == "idle"
        consent = {
            "sequence": 1,
            "accepted": True,
            "consentVersion": "realtime-evidence-consent-v1",
            "disclosureDigest": capture_data["disclosureDigest"],
            "retentionHours": 24,
            "sources": {"microphone": True, "typed": True},
        }
        status, consent_result = await _request(
            port=port,
            origin=origin,
            path="/api/v1/evidence-consent",
            bearer=token,
            body=json.dumps(consent, separators=(",", ":")).encode(),
        )
        assert status == 200 and consent_result == {
            "captureState": "active",
            "result": "consent_activated",
            "sequence": 1,
        }
        status, _ = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"typed evidence turn"}',
        )
        assert status == 202
        await _wait_event(port=port, origin=origin, token=token, kind="assistant_turn_completed")
        await _wait_event(port=port, origin=origin, token=token, kind="voice_input_ready")
        transcriber._armed = True
        frame = rtc.AudioFrame(
            data=_pcm(), sample_rate=48_000, num_channels=1, samples_per_channel=480
        )
        for _ in range(100):
            await source.capture_frame(frame)
        async with asyncio.timeout(10):
            while transcriber._armed:
                await asyncio.sleep(0.05)
        async with asyncio.timeout(30):
            while True:
                with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                    rows = connection.execute(
                        "SELECT canonical_payload FROM evidence_events "
                        "WHERE event_kind = 'user_final_accepted'"
                    ).fetchall()
                if len(rows) >= 2:
                    break
                await asyncio.sleep(0.1)
        payloads = [json.loads(row[0]) for row in rows]
        assert {payload["source"] for payload in payloads} == {"typed", "microphone"}
    finally:
        await room.disconnect()
        if running is not None:
            await composition.close_host(running)  # type: ignore[arg-type]
    assert composition.trace.status().trace_complete


class _FaultAtWriterCompletionTransport:
    """Fault only at the real writer's owned terminal completion boundary."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.drain_calls = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def drain_and_close(self, _command: object) -> DrainDisposition:
        self.drain_calls += 1
        return DrainDisposition.WRITER_FAULT


class _BlockFirstWriterRecordTransport:
    """Hold every real ordinary writer completion until the qualification releases it."""

    def __init__(self, delegate: object) -> None:
        from threading import Event

        self._delegate = delegate
        self.entered = Event()
        self.release = Event()
        self._blocked = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def append_record(self, item: object) -> object:
        self._blocked = True
        self.entered.set()
        assert self.release.wait(timeout=60.0), "writer block was not released"
        return cast(Any, self._delegate).append_record(item)


async def _run_non_mutation_arm(
    *,
    tmp_path: Path,
    capture: bool,
    consent: bool,
    writer_fault: bool,
    writer_block: bool = False,
) -> tuple[object, ...]:
    """Run one real host/browser/LiveKit flow with fixed typed and PCM stimuli."""

    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database = tmp_path / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition = InProcessQualificationComposition()
    transcriber = _Transcriber()
    fault_transports: list[_FaultAtWriterCompletionTransport] = []
    block_transports: list[_BlockFirstWriterRecordTransport] = []

    def writer_factory(runtime: object) -> object:
        delegate = cast(Any, runtime).create_sqlite_transport()
        if writer_fault:
            fault_transport = _FaultAtWriterCompletionTransport(delegate)
            fault_transports.append(fault_transport)
            return fault_transport
        if writer_block:
            block_transport = _BlockFirstWriterRecordTransport(delegate)
            block_transports.append(block_transport)
            return block_transport
        return delegate

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=capture,
            evidence_database=database if capture else None,
        ),
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory if writer_fault or writer_block else None,
    )
    room = rtc.Room()
    running: object | None = None
    try:
        running = await composition.start_host(registration)
        origin, token, source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        if consent:
            capture_status = await _wait_event(
                port=port, origin=origin, token=token, kind="capture_status"
            )
            capture_data = capture_status["data"]
            assert type(capture_data) is dict
            status, result = await _request(
                port=port,
                origin=origin,
                path="/api/v1/evidence-consent",
                bearer=token,
                body=json.dumps(
                    {
                        "sequence": 1,
                        "accepted": True,
                        "consentVersion": "realtime-evidence-consent-v1",
                        "disclosureDigest": capture_data["disclosureDigest"],
                        "retentionHours": 24,
                        "sources": {"microphone": True, "typed": True},
                    },
                    separators=(",", ":"),
                ).encode(),
            )
            assert status == 200 and result == {
                "captureState": "active",
                "result": "consent_activated",
                "sequence": 1,
            }
        status, result = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"paired typed stimulus"}',
        )
        assert status == 202 and result == {"sequence": 1, "version": 1}
        if writer_block:
            assert len(block_transports) == 1
            assert await asyncio.wait_for(
                asyncio.to_thread(block_transports[0].entered.wait), timeout=5.0
            )
        typed_completed = await _wait_event(
            port=port, origin=origin, token=token, kind="assistant_turn_completed"
        )
        typed_sequence = typed_completed["sequence"]
        assert type(typed_sequence) is int
        voice_ready = await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="voice_input_ready",
            after=typed_sequence,
        )
        transcriber._armed = True
        frame = rtc.AudioFrame(
            data=_pcm(), sample_rate=48_000, num_channels=1, samples_per_channel=480
        )
        for _ in range(100):
            await source.capture_frame(frame)
        voice_ready_sequence = voice_ready["sequence"]
        assert type(voice_ready_sequence) is int
        await _wait_event(
            port=port,
            origin=origin,
            token=token,
            kind="assistant_turn_completed",
            after=voice_ready_sequence,
        )
    finally:
        for transport in block_transports:
            transport.release.set()
        await room.disconnect()
        if running is not None:
            if writer_fault:
                with pytest.raises(
                    RuntimeError,
                    match="evidence drain did not reach terminal stopped state",
                ):
                    await composition.close_host(running)  # type: ignore[arg-type]
            else:
                await composition.close_host(running)  # type: ignore[arg-type]
    assert composition.trace.status().trace_complete
    if writer_fault:
        assert len(fault_transports) == 1
        assert fault_transports[0].drain_calls == 1
    if writer_block:
        assert len(block_transports) == 1
        assert block_transports[0].entered.is_set()
        assert block_transports[0].release.is_set()
    return composition.trace.records()


async def test_full_host_capture_arms_do_not_mutate_paired_typed_or_pcm_conversation(
    tmp_path: Path,
) -> None:
    """Capture state and a real writer completion fault cannot change the conversation host."""

    disabled = await _run_non_mutation_arm(
        tmp_path=tmp_path, capture=False, consent=False, writer_fault=False
    )
    unconsented = await _run_non_mutation_arm(
        tmp_path=tmp_path, capture=True, consent=False, writer_fault=False
    )
    consented = await _run_non_mutation_arm(
        tmp_path=tmp_path, capture=True, consent=True, writer_fault=False
    )
    writer_blocked = await _run_non_mutation_arm(
        tmp_path=tmp_path, capture=True, consent=True, writer_fault=False, writer_block=True
    )
    writer_faulted = await _run_non_mutation_arm(
        tmp_path=tmp_path, capture=True, consent=True, writer_fault=True
    )

    # These raw owner records are the production-authored comparison boundary:
    # adapter snapshots, generation, confirmed transport, cancellation/cleanup,
    # and the host return. Browser capture status and SQLite contents stay out.
    assert unconsented == disabled
    assert consented == disabled
    assert writer_blocked == disabled
    # The terminal host outcome cannot be normalized away: the production fault
    # contract currently propagates the failed DRAIN_AND_STOP as host failure.
    assert writer_faulted[:-1] == disabled[:-1]
    assert cast(Any, writer_faulted[-1]).outcome.value == "failed"
    assert cast(Any, disabled[-1]).outcome.value == "returned"


class _ResponseCheckpointInference:
    """Pass host preflight, then hold the submitted response before any output."""

    def __init__(self) -> None:
        self.response_entered = asyncio.Event()
        self.release = asyncio.Event()

    def stream(self, snapshot: ConversationContextSnapshot, *, turn_id: str) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id)

    async def _stream(
        self, snapshot: ConversationContextSnapshot, turn_id: str
    ) -> AsyncIterator[str]:
        del snapshot
        if turn_id == "turn_preflight":
            yield "Ready."
            return
        self.response_entered.set()
        await self.release.wait()
        yield "This response must never be generated."

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


_EVIDENCE_ONLY_CLOSE_STAGES = frozenset(
    {
        CloseStageV1.WRITER_DRAIN,
        CloseStageV1.WRITER_STOP,
        CloseStageV1.TRANSPORT_CLOSE,
        CloseStageV1.EVIDENCE_RUNTIME,
        CloseStageV1.RETENTION_CANCELLATION,
    }
)


def _normalized_production_records(
    records: tuple[object, ...],
    *,
    remove_evidence_only: bool,
) -> tuple[object, ...]:
    close_records = tuple(record for record in records if type(record) is CloseStageObservationV1)
    if not remove_evidence_only:
        return close_records
    return tuple(
        record
        for record in close_records
        if not (
            type(record) is CloseStageObservationV1 and record.stage in _EVIDENCE_ONLY_CLOSE_STAGES
        )
    )


def _qualification_fact(record: object) -> tuple[str, object]:
    value = cast(Any, record)
    return (
        value.kind.value,
        getattr(record, "reason", getattr(record, "succeeded", getattr(record, "outcome", None))),
    )


async def _run_active_response_host_shutdown_arm(
    *, tmp_path: Path, capture: bool, consent: bool
) -> tuple[tuple[object, ...], tuple[object, ...], Path]:
    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database = tmp_path / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition = InProcessQualificationComposition()
    transcriber = _Transcriber()
    checkpoints: list[_ResponseCheckpointInference] = []

    def inference_factory() -> _ResponseCheckpointInference:
        checkpoint = _ResponseCheckpointInference()
        checkpoints.append(checkpoint)
        return checkpoint

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=capture,
            evidence_database=database if capture else None,
        ),
        inference_factory=inference_factory,
        speech_presence_factory=_Presence,
        synthesizer_factory=_Synthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
    )
    room = rtc.Room()
    running: object | None = None
    closed = False
    try:
        running = await composition.start_host(registration)
        origin, token, _source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        if consent:
            capture_status = await _wait_event(
                port=port, origin=origin, token=token, kind="capture_status"
            )
            capture_data = capture_status["data"]
            assert type(capture_data) is dict
            status, result = await _request(
                port=port,
                origin=origin,
                path="/api/v1/evidence-consent",
                bearer=token,
                body=json.dumps(
                    {
                        "sequence": 1,
                        "accepted": True,
                        "consentVersion": "realtime-evidence-consent-v1",
                        "disclosureDigest": capture_data["disclosureDigest"],
                        "retentionHours": 24,
                        "sources": {"microphone": True, "typed": True},
                    },
                    separators=(",", ":"),
                ).encode(),
            )
            assert status == 200 and result == {
                "captureState": "active",
                "result": "consent_activated",
                "sequence": 1,
            }
        status, result = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"paired host shutdown stimulus"}',
        )
        assert status == 202 and result == {"sequence": 1, "version": 1}
        assert len(checkpoints) == 1
        await asyncio.wait_for(checkpoints[0].response_entered.wait(), timeout=10.0)

        await composition.close_host(running)  # type: ignore[arg-type]
        closed = True
        observations = composition.production_observations(running)  # type: ignore[arg-type]
        assert observations.status().trace_complete
        return composition.trace.records(), observations.records(), database
    finally:
        for checkpoint in checkpoints:
            checkpoint.release.set()
        await room.disconnect()
        if running is not None and not closed:
            await composition.close_host(running)  # type: ignore[arg-type]


async def test_full_host_active_response_host_shutdown_is_capture_equivalent(
    tmp_path: Path,
) -> None:
    """Public host close cancels an active response identically across capture arms."""

    disabled = await _run_active_response_host_shutdown_arm(
        tmp_path=tmp_path, capture=False, consent=False
    )
    unconsented = await _run_active_response_host_shutdown_arm(
        tmp_path=tmp_path, capture=True, consent=False
    )
    consented = await _run_active_response_host_shutdown_arm(
        tmp_path=tmp_path, capture=True, consent=True
    )

    disabled_raw, disabled_metadata, disabled_database = disabled
    unconsented_raw, unconsented_metadata, unconsented_database = unconsented
    consented_raw, consented_metadata, consented_database = consented
    assert not disabled_database.exists()
    assert not unconsented_database.exists()
    assert disabled_raw == unconsented_raw == consented_raw
    raw_facts = [_qualification_fact(record) for record in disabled_raw]
    assert raw_facts.count(("cancellation", TerminalReason.HOST_SHUTDOWN)) == 1
    assert raw_facts.count(("foreground_cleanup", True)) == 1
    assert raw_facts.count(("host_return", "returned")) == 1
    assert not any(
        cast(Any, record).kind.value in {"generated_text", "transport_confirmed_chunk"}
        for record in disabled_raw
    )

    disabled_close = _normalized_production_records(disabled_metadata, remove_evidence_only=False)
    assert (
        _normalized_production_records(unconsented_metadata, remove_evidence_only=True)
        == disabled_close
    )
    assert (
        _normalized_production_records(consented_metadata, remove_evidence_only=True)
        == disabled_close
    )
    for metadata in (disabled_metadata, unconsented_metadata, consented_metadata):
        close_records = _normalized_production_records(metadata, remove_evidence_only=False)
        assert close_records
        assert all(
            cast(CloseStageObservationV1, record).result is CloseResultV1.SUCCEEDED
            for record in close_records
        )
        assert (
            close_records.count(
                CloseStageObservationV1(CloseStageV1.LAUNCHER, CloseResultV1.SUCCEEDED)
            )
            == 1
        )
        close_stages = [
            record.stage for record in metadata if type(record) is CloseStageObservationV1
        ]
        browser_ingress_close = close_stages.index(CloseStageV1.BROWSER_CLIENT)
        assert browser_ingress_close < close_stages.index(CloseStageV1.FOREGROUND_CLOSE)
        if CloseStageV1.HOST_WORK in close_stages:
            assert browser_ingress_close < close_stages.index(CloseStageV1.HOST_WORK)
            assert close_stages.index(CloseStageV1.HOST_WORK) < close_stages.index(
                CloseStageV1.SPEECH_LOOP
            )
        else:
            assert browser_ingress_close < close_stages.index(CloseStageV1.SPEECH_LOOP)
    consented_close = [
        record for record in consented_metadata if type(record) is CloseStageObservationV1
    ]
    for stage in _EVIDENCE_ONLY_CLOSE_STAGES:
        assert consented_close.count(CloseStageObservationV1(stage, CloseResultV1.SUCCEEDED)) == 1
    assert (
        _normalized_production_records(consented_metadata, remove_evidence_only=True)
        == disabled_metadata
    )
    assert consented_database.exists()


class _CapacityInference:
    """Emit separate valid generated segments until ordinary credits saturate."""

    def stream(self, snapshot: ConversationContextSnapshot, *, turn_id: str) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id)

    async def _stream(
        self, snapshot: ConversationContextSnapshot, turn_id: str
    ) -> AsyncIterator[str]:
        del snapshot
        if turn_id == "turn_preflight":
            yield "Ready."
            return
        for index in range(80):
            yield f"capacity generated segment {index}."

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


class _CapacitySynthesizer:
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        if turn_id == "turn_tts_preflight":
            yield SpeechChunk(
                turn_id=turn_id,
                chunk_id="capacity-preflight",
                text=text,
                audio=AudioFrame(pcm=_pcm(), sample_rate_hz=48_000, channels=1),
            )
            return
        del text, turn_id
        if False:
            yield cast(SpeechChunk, None)

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


class _BlockRevokeAndDrainTransport:
    """Block only real writer control commits, never controller or queue behavior."""

    def __init__(self, delegate: object) -> None:
        from threading import Event

        self._delegate = delegate
        self.revoke_entered = Event()
        self.revoke_release = Event()
        self.drain_entered = Event()
        self.drain_release = Event()
        self.revoke_calls = 0
        self.drain_calls = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def commit_revoke_request(self, command: object) -> object:
        self.revoke_calls += 1
        self.revoke_entered.set()
        assert self.revoke_release.wait(timeout=10.0), "revoke commit block was not released"
        return cast(Any, self._delegate).commit_revoke_request(command)

    def drain_and_close(self, command: object) -> object:
        self.drain_calls += 1
        self.drain_entered.set()
        assert self.drain_release.wait(timeout=10.0), "drain block was not released"
        return cast(Any, self._delegate).drain_and_close(command)


async def _consent_capture(*, port: int, origin: str, token: str) -> None:
    capture = await _wait_event(port=port, origin=origin, token=token, kind="capture_status")
    data = capture["data"]
    assert type(data) is dict
    status, result = await _request(
        port=port,
        origin=origin,
        path="/api/v1/evidence-consent",
        bearer=token,
        body=json.dumps(
            {
                "sequence": 1,
                "accepted": True,
                "consentVersion": "realtime-evidence-consent-v1",
                "disclosureDigest": data["disclosureDigest"],
                "retentionHours": 24,
                "sources": {"microphone": True, "typed": True},
            },
            separators=(",", ":"),
        ).encode(),
    )
    assert status == 200 and result == {
        "captureState": "active",
        "result": "consent_activated",
        "sequence": 1,
    }


def _capacity_vector(
    observations: tuple[object, ...],
) -> list[tuple[str, str, int, int, int, bool]]:
    return [
        (
            value.kind,
            value.rejection_source,
            value.queue_record_count,
            value.queue_canonical_bytes,
            value.queue_physical_count,
            value.all_capacity_released,
        )
        for value in cast(list[Any], list(observations))
    ]


async def test_full_host_capacity_saturation_reports_record_binding_and_release(
    tmp_path: Path,
) -> None:
    """A real batched turn reaches 64 records before physical or aggregate-byte capacity."""

    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database = tmp_path / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition, transports = InProcessQualificationComposition(), []
    transcriber = _Transcriber()

    def writer_factory(runtime: object) -> object:
        transport = _BlockFirstWriterRecordTransport(cast(Any, runtime).create_sqlite_transport())
        transports.append(transport)
        return transport

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=True,
            evidence_database=database,
        ),
        inference_factory=_CapacityInference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_CapacitySynthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory,
    )
    room, running = rtc.Room(), None
    try:
        running = await composition.start_host(registration)
        origin, token, _source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        await _consent_capture(port=port, origin=origin, token=token)
        status, _ = await _request(
            port=port,
            origin=origin,
            path="/api/v1/input",
            bearer=token,
            body=b'{"sequence":1,"text":"capacity typed turn"}',
        )
        assert status == 202
        assert len(transports) == 1
        assert await asyncio.wait_for(asyncio.to_thread(transports[0].entered.wait), timeout=5)
        await asyncio.sleep(2)
        vector = _capacity_vector(composition.capacity_observations(running))
        admitted = [item for item in vector if item[0] == "ordinary_admitted"]
        rejected = [item for item in vector if item[0] == "ordinary_rejected"]
        assert rejected, vector
        assert admitted
        assert admitted[0][2:5:2] == (7, 2)
        assert [(item[2], item[4]) for item in admitted[1:]] == [
            (7 + index, 2 + index) for index in range(1, 58)
        ]
        assert MAX_QUEUE_RECORDS * MAX_CANONICAL_RECORD_BYTES == MAX_QUEUE_CANONICAL_BYTES
        capacity_evidence = {
            "ordinaryRecordCount": admitted[-1][2],
            "ordinaryCanonicalBytes": admitted[-1][3],
            "ordinaryPhysicalItems": admitted[-1][4],
            "aggregateByteSaturation": "unreachable under the frozen valid-record schema",
            "physicalItemSaturation": (
                "unreachable before record capacity under the frozen production batched trace"
            ),
        }
        assert capacity_evidence == {
            "ordinaryRecordCount": MAX_QUEUE_RECORDS,
            "ordinaryCanonicalBytes": 193_497,
            "ordinaryPhysicalItems": 59,
            "aggregateByteSaturation": "unreachable under the frozen valid-record schema",
            "physicalItemSaturation": (
                "unreachable before record capacity under the frozen production batched trace"
            ),
        }
        assert admitted[-1][3] == 193_497
        assert admitted[-1][3] < 2 * 1024 * 1024
        assert admitted[-1][4] < MAX_QUEUE_PHYSICAL_ITEMS
        assert all(
            item == ("ordinary_rejected", "record_capacity", 64, 193_497, 59, False)
            for item in rejected
        )
        transports[0].release.set()
        await _wait_event(port=port, origin=origin, token=token, kind="assistant_turn_completed")
        async with asyncio.timeout(20):
            while (
                sum(
                    item[0] == "ordinary_completed"
                    for item in _capacity_vector(composition.capacity_observations(running))
                )
                < 59
            ):
                await asyncio.sleep(0.05)
        await composition.close_host(running)
        final_vector = _capacity_vector(composition.capacity_observations(running))
        assert final_vector[-1] == ("drain_terminal", "none", 0, 0, 0, True)
    finally:
        for transport in transports:
            transport.release.set()
        await room.disconnect()
        if running is not None and not running._closed:
            await composition.close_host(running)
    assert composition.trace.status().trace_complete


async def test_full_host_http_revoke_duplicate_coalesces_without_admission_retry(
    tmp_path: Path,
) -> None:
    """An in-flight duplicate HTTP revoke shares its owner operation and commits once."""

    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database = tmp_path / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition, transports = InProcessQualificationComposition(), []
    transcriber = _Transcriber()

    def writer_factory(runtime: object) -> object:
        transport = _BlockRevokeAndDrainTransport(cast(Any, runtime).create_sqlite_transport())
        transports.append(transport)
        return transport

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=True,
            evidence_database=database,
        ),
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_CapacitySynthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory,
    )
    room, running = rtc.Room(), None
    try:
        running = await composition.start_host(registration)
        origin, token, _source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        await _consent_capture(port=port, origin=origin, token=token)
        first = asyncio.create_task(
            _request(
                port=port,
                origin=origin,
                path="/api/v1/evidence-revoke",
                bearer=token,
                body=b'{"sequence":2}',
            )
        )
        assert await asyncio.wait_for(
            asyncio.to_thread(transports[0].revoke_entered.wait), timeout=5
        )
        second = asyncio.create_task(
            _request(
                port=port,
                origin=origin,
                path="/api/v1/evidence-revoke",
                bearer=token,
                body=b'{"sequence":2}',
            )
        )
        await asyncio.sleep(0.1)
        assert transports[0].revoke_calls == 1
        transports[0].revoke_release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result == second_result
        assert first_result[0] in {200, 202}
        async with asyncio.timeout(20):
            while not any(
                item[0] == "revoke_terminal" and item[-1]
                for item in _capacity_vector(composition.capacity_observations(running))
            ):
                await asyncio.sleep(0.05)
        vector = _capacity_vector(composition.capacity_observations(running))
        assert vector == [
            ("revoke_accepted", "none", 3, 98_304, 0, False),
            ("revoke_terminal", "none", 0, 0, 0, True),
        ]
        assert sum(item[0] == "revoke_coalesced" for item in vector) == 0
    finally:
        for transport in transports:
            transport.revoke_release.set()
            transport.drain_release.set()
        await room.disconnect()
        if running is not None:
            await composition.close_host(running)


async def test_full_host_owner_close_coalesces_drain_and_releases(tmp_path: Path) -> None:
    """Only the same registered owner can join its in-flight real host close."""

    suffix, port = uuid.uuid4().hex[:10], _available_port()
    database = tmp_path / suffix / "capture-v1.sqlite3"
    database.parent.mkdir()
    composition, transports = InProcessQualificationComposition(), []
    transcriber = _Transcriber()

    def writer_factory(runtime: object) -> object:
        transport = _BlockRevokeAndDrainTransport(cast(Any, runtime).create_sqlite_transport())
        transports.append(transport)
        return transport

    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"qualification-{suffix}",
            worker_identity=f"worker_{suffix}",
            evidence_capture=True,
            evidence_database=database,
        ),
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_CapacitySynthesizer,
        transcriber_factory=lambda: transcriber,
        vad_factory=lambda: _Vad(transcriber),
        identity_factory=lambda: f"verifier_{suffix}",
        writer_transport_factory=writer_factory,
    )
    room, running = rtc.Room(), None
    try:
        running = await composition.start_host(registration)
        origin, token, _source = await _connect_and_activate(
            port=port, launch_url=running.url, room=room
        )
        await _consent_capture(port=port, origin=origin, token=token)
        first = asyncio.create_task(composition.close_host(running))
        assert await asyncio.wait_for(
            asyncio.to_thread(transports[0].drain_entered.wait), timeout=10
        )
        second = asyncio.create_task(composition.retry_owned_close(running))
        await asyncio.sleep(0.1)
        assert transports[0].drain_calls == 1
        transports[0].drain_release.set()
        close_results = await asyncio.gather(first, second, return_exceptions=True)
        vector = _capacity_vector(composition.capacity_observations(running))
        assert close_results == [None, None], (close_results, vector)
        assert vector == [
            ("drain_accepted", "none", 3, 98_304, 0, False),
            ("drain_terminal", "none", 0, 0, 0, True),
        ]
        assert sum(item[0] == "drain_coalesced" for item in vector) == 0
        await composition.retry_owned_close(running)
        assert transports[0].drain_calls == 1
        assert _capacity_vector(composition.capacity_observations(running)) == vector
        with pytest.raises(TypeError, match="test-owner running host"):
            await composition.retry_owned_close(cast(Any, object()))
    finally:
        for transport in transports:
            transport.revoke_release.set()
            transport.drain_release.set()
        await room.disconnect()
        if running is not None and not running._closed:
            await composition.close_host(running)
