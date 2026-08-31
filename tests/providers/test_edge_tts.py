from __future__ import annotations

import asyncio

import pytest

from hermes_realtime.providers import EdgeTtsSynthesizer
from hermes_realtime.speech import AudioFrame, SpeechChunk


@pytest.mark.asyncio
async def test_edge_tts_yields_one_bounded_pcm_chunk_with_exact_text_authority() -> None:
    calls: list[tuple[object, ...]] = []

    async def synthesize_mp3(text: str, voice: str) -> bytes:
        calls.append(("synthesize", text, voice))
        return b"synthetic-mp3"

    async def decode_mp3(payload: bytes) -> bytes:
        calls.append(("decode", payload))
        return b"\x01\x00" * 480

    tts = EdgeTtsSynthesizer(
        voice="en-US-AriaNeural",
        synthesize_mp3=synthesize_mp3,
        decode_mp3=decode_mp3,
        sample_rate_hz=48_000,
        channels=1,
    )

    chunks = [chunk async for chunk in tts.synthesize("Hello there.", "turn_1")]

    assert len(chunks) == 1
    assert chunks[0] == SpeechChunk(
        turn_id="turn_1",
        chunk_id=chunks[0].chunk_id,
        text="Hello there.",
        audio=AudioFrame(pcm=b"\x01\x00" * 480, sample_rate_hz=48_000, channels=1),
    )
    assert chunks[0].chunk_id.startswith("edge_")
    assert calls == [
        ("synthesize", "Hello there.", "en-US-AriaNeural"),
        ("decode", b"synthetic-mp3"),
    ]


@pytest.mark.asyncio
async def test_edge_tts_suppresses_matched_markdown_delimiters_from_speech() -> None:
    synthesized_text: list[str] = []

    async def synthesize_mp3(text: str, _voice: str) -> bytes:
        synthesized_text.append(text)
        return b"synthetic-mp3"

    async def decode_mp3(_payload: bytes) -> bytes:
        return b"\x01\x00" * 480

    tts = EdgeTtsSynthesizer(
        synthesize_mp3=synthesize_mp3,
        decode_mp3=decode_mp3,
    )
    source = "*italic* and **bold** and ***both***; unmatched * remains."

    chunk = await anext(tts.synthesize(source, "turn_markdown"))

    assert synthesized_text == [
        "italic and bold and both; unmatched * remains."
    ]
    assert chunk.text == source


@pytest.mark.asyncio
async def test_edge_tts_pads_pcm_to_exact_ten_millisecond_transport_boundary() -> None:
    async def synthesize_mp3(_text: str, _voice: str) -> bytes:
        return b"synthetic-mp3"

    async def decode_mp3(_payload: bytes) -> bytes:
        return b"\x01\x00" * 481

    tts = EdgeTtsSynthesizer(
        synthesize_mp3=synthesize_mp3,
        decode_mp3=decode_mp3,
        sample_rate_hz=48_000,
        channels=1,
    )

    chunk = await anext(tts.synthesize("Aligned text.", "turn_aligned"))

    assert len(chunk.audio.pcm) == 480 * 2 * 2
    assert chunk.audio.pcm[: 481 * 2] == b"\x01\x00" * 481
    assert chunk.audio.pcm[481 * 2 :] == b"\x00" * ((960 - 481) * 2)


@pytest.mark.asyncio
async def test_edge_tts_cancel_settles_owned_synthesis() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def synthesize_mp3(_text: str, _voice: str) -> bytes:
        started.set()
        await release.wait()
        return b"late-mp3"

    async def decode_mp3(_payload: bytes) -> bytes:
        raise AssertionError("cancelled synthesis must not decode")

    tts = EdgeTtsSynthesizer(
        synthesize_mp3=synthesize_mp3,
        decode_mp3=decode_mp3,
    )
    iterator = tts.synthesize("Cancelled text.", "turn_cancel")
    pending: asyncio.Future[SpeechChunk] = asyncio.ensure_future(anext(iterator))
    await started.wait()

    cancelling = asyncio.create_task(tts.cancel("turn_cancel"))
    await cancelling
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await pending


@pytest.mark.asyncio
async def test_ffmpeg_cancellation_kills_and_waits_without_reentering_communicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self) -> None:
            self.stdin = object()
            self.stdout = object()
            self.stderr = object()
            self.returncode: int | None = None
            self.communicate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0

        async def communicate(self, payload: bytes | None = None) -> tuple[bytes, bytes]:
            del payload
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise asyncio.CancelledError
            raise AssertionError("communicate must not be re-entered during cleanup")

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

        async def wait(self) -> int:
            self.wait_calls += 1
            return -9

    process = Process()

    async def create_subprocess_exec(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr("hermes_realtime.providers.edge_tts.shutil.which", lambda path: path)
    tts = EdgeTtsSynthesizer(ffmpeg_path="ffmpeg")

    with pytest.raises(asyncio.CancelledError):
        await tts._ffmpeg_decode(b"mp3")

    assert process.communicate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_ffmpeg_cleanup_timeout_retains_original_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        stdin = object()
        stdout = object()
        stderr = object()
        returncode: int | None = None

        async def communicate(self, payload: bytes | None = None) -> tuple[bytes, bytes]:
            del payload
            raise asyncio.CancelledError

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            return -9

    process = Process()

    async def create_subprocess_exec(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return process

    async def timeout_wait(awaitable: object, *, timeout: float) -> object:
        del timeout
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError("cleanup timed out")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(asyncio, "wait_for", timeout_wait)
    monkeypatch.setattr("hermes_realtime.providers.edge_tts.shutil.which", lambda path: path)
    tts = EdgeTtsSynthesizer(ffmpeg_path="ffmpeg")

    with pytest.raises(BaseExceptionGroup) as caught:
        await tts._ffmpeg_decode(b"mp3")

    assert any(isinstance(error, asyncio.CancelledError) for error in caught.value.exceptions)
    assert any(isinstance(error, TimeoutError) for error in caught.value.exceptions)


@pytest.mark.asyncio
async def test_ffmpeg_kill_race_retains_original_decode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        stdin = object()
        stdout = object()
        stderr = object()
        returncode: int | None = None

        async def communicate(self, payload: bytes | None = None) -> tuple[bytes, bytes]:
            del payload
            raise ValueError("decode failed")

        def kill(self) -> None:
            raise ProcessLookupError("already exited")

        async def wait(self) -> int:
            raise AssertionError("wait must not run after kill fails")

    process = Process()

    async def create_subprocess_exec(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr("hermes_realtime.providers.edge_tts.shutil.which", lambda path: path)
    tts = EdgeTtsSynthesizer(ffmpeg_path="ffmpeg")

    with pytest.raises(BaseExceptionGroup) as caught:
        await tts._ffmpeg_decode(b"mp3")

    assert any(isinstance(error, ValueError) for error in caught.value.exceptions)
    assert any(isinstance(error, ProcessLookupError) for error in caught.value.exceptions)
