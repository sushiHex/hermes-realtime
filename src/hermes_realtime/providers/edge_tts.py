"""Cancellable Edge neural TTS adapter with bounded FFmpeg PCM decoding."""

from __future__ import annotations

import asyncio
import importlib
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

from hermes_realtime.providers._speech_text import strip_markdown_emphasis_for_speech
from hermes_realtime.speech import AudioFrame, SpeechChunk

_MAX_TEXT_CHARS = 4096
_MAX_VOICE_CHARS = 128
_MAX_MP3_BYTES = 8 * 1024 * 1024
_MAX_PCM_BYTES = 64 * 1024 * 1024
_MAX_ACTIVE_TURNS = 16

_SynthesizeMp3 = Callable[[str, str], Awaitable[bytes]]
_DecodeMp3 = Callable[[bytes], Awaitable[bytes]]


class EdgeTtsSynthesizer:
    """Synthesize one inference segment into one independently cancellable PCM chunk."""

    def __init__(
        self,
        *,
        voice: str = "en-US-AriaNeural",
        ffmpeg_path: str = "ffmpeg",
        sample_rate_hz: int = 48_000,
        channels: int = 1,
        max_mp3_bytes: int = 2 * 1024 * 1024,
        max_pcm_bytes: int = 16 * 1024 * 1024,
        max_active_turns: int = 4,
        synthesize_mp3: _SynthesizeMp3 | None = None,
        decode_mp3: _DecodeMp3 | None = None,
    ) -> None:
        if type(voice) is not str:
            raise TypeError("voice must be an exact built-in string")
        if not voice.strip() or len(voice) > _MAX_VOICE_CHARS:
            raise ValueError("voice must contain 1 to 128 characters")
        if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be an exact positive integer")
        if type(channels) is not int or not 1 <= channels <= 8:
            raise ValueError("channels must be an exact integer from 1 through 8")
        for name, value, ceiling in (
            ("max_mp3_bytes", max_mp3_bytes, _MAX_MP3_BYTES),
            ("max_pcm_bytes", max_pcm_bytes, _MAX_PCM_BYTES),
            ("max_active_turns", max_active_turns, _MAX_ACTIVE_TURNS),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= ceiling:
                raise ValueError(f"{name} exceeds supported bounds")
        if synthesize_mp3 is not None and not callable(synthesize_mp3):
            raise TypeError("synthesize_mp3 must be callable")
        if decode_mp3 is not None and not callable(decode_mp3):
            raise TypeError("decode_mp3 must be callable")
        if decode_mp3 is None:
            if type(ffmpeg_path) is not str or not ffmpeg_path.strip():
                raise ValueError("ffmpeg_path must be an exact nonblank string")
            resolved = shutil.which(ffmpeg_path)
            if resolved is None:
                raise RuntimeError("ffmpeg executable is required for Edge TTS")
            ffmpeg_path = resolved

        self._voice = voice
        self._ffmpeg_path = ffmpeg_path
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._max_mp3_bytes = max_mp3_bytes
        self._max_pcm_bytes = max_pcm_bytes
        self._max_active_turns = max_active_turns
        self._synthesize_mp3 = synthesize_mp3 or self._edge_synthesize
        self._decode_mp3 = decode_mp3 or self._ffmpeg_decode
        self._operations: dict[str, asyncio.Task[SpeechChunk]] = {}
        self._lock = asyncio.Lock()

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self._validate_text(text)
        self._validate_turn_id(turn_id)
        async with self._lock:
            if turn_id in self._operations:
                raise RuntimeError("turn already owns an Edge TTS operation")
            if len(self._operations) >= self._max_active_turns:
                raise RuntimeError("Edge TTS operation capacity exhausted")
            operation = asyncio.create_task(
                self._build_chunk(text, turn_id),
                name=f"edge-tts-synthesis:{turn_id}",
            )
            self._operations[turn_id] = operation
        try:
            try:
                chunk = await operation
            except asyncio.CancelledError:
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise
            yield chunk
        finally:
            async with self._lock:
                if self._operations.get(turn_id) is operation:
                    del self._operations[turn_id]

    async def cancel(self, turn_id: str) -> None:
        self._validate_turn_id(turn_id)
        async with self._lock:
            operation = self._operations.get(turn_id)
        if operation is not None:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            async with self._lock:
                if self._operations.get(turn_id) is operation:
                    del self._operations[turn_id]

    async def close(self) -> None:
        async with self._lock:
            operations = tuple(self._operations.values())
        for operation in operations:
            operation.cancel()
        await asyncio.gather(*operations, return_exceptions=True)
        async with self._lock:
            self._operations.clear()

    async def _build_chunk(self, text: str, turn_id: str) -> SpeechChunk:
        spoken_text = strip_markdown_emphasis_for_speech(text)
        mp3 = await self._synthesize_mp3(spoken_text, self._voice)
        if type(mp3) is not bytes:
            raise TypeError("Edge TTS backend must return exact MP3 bytes")
        if not mp3 or len(mp3) > self._max_mp3_bytes:
            raise ValueError("Edge TTS MP3 output exceeds supported bounds")
        pcm = await self._decode_mp3(mp3)
        if type(pcm) is not bytes:
            raise TypeError("Edge TTS decoder must return exact PCM bytes")
        if not pcm or len(pcm) > self._max_pcm_bytes:
            raise ValueError("Edge TTS PCM output exceeds supported bounds")
        transport_frame_bytes = (self._sample_rate_hz // 100) * self._channels * 2
        remainder = len(pcm) % transport_frame_bytes
        if remainder:
            padding_bytes = transport_frame_bytes - remainder
            if len(pcm) + padding_bytes > self._max_pcm_bytes:
                raise ValueError("aligned Edge TTS PCM output exceeds supported bounds")
            pcm += b"\x00" * padding_bytes
        audio = AudioFrame(
            pcm=pcm,
            sample_rate_hz=self._sample_rate_hz,
            channels=self._channels,
        )
        return SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"edge_{uuid.uuid4().hex}",
            text=text,
            audio=audio,
        )

    async def _edge_synthesize(self, text: str, voice: str) -> bytes:
        try:
            edge_tts = importlib.import_module("edge_tts")
        except ImportError as error:
            raise RuntimeError(
                "Edge TTS requires the 'edge-tts' local extra"
            ) from error
        communication = edge_tts.Communicate(text=text, voice=voice)
        output = bytearray()
        async for event in communication.stream():
            if type(event) is not dict:
                raise TypeError("Edge TTS stream event must be an exact object")
            event_type = event.get("type")
            if type(event_type) is not str:
                raise TypeError("Edge TTS event type must be an exact string")
            if event_type != "audio":
                continue
            data = event.get("data")
            if type(data) is not bytes:
                raise TypeError("Edge TTS audio event must contain exact bytes")
            if len(output) + len(data) > self._max_mp3_bytes:
                raise ValueError("Edge TTS MP3 output exceeds supported bounds")
            output.extend(data)
        return bytes(output)

    async def _ffmpeg_decode(self, payload: bytes) -> bytes:
        max_duration_seconds = self._max_pcm_bytes / (
            self._sample_rate_hz * self._channels * 2
        )
        process = await asyncio.create_subprocess_exec(
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-t",
            f"{max_duration_seconds:.6f}",
            "-f",
            "s16le",
            "-ar",
            str(self._sample_rate_hz),
            "-ac",
            str(self._channels),
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            pcm, error_output = await process.communicate(payload)
            if len(pcm) > self._max_pcm_bytes:
                raise ValueError("Edge TTS PCM output exceeds supported bounds")
            if process.returncode != 0:
                detail = error_output.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg Edge TTS decode failed: {detail[:1024]}")
            return pcm
        except BaseException as original_error:
            try:
                if process.returncode is None:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "FFmpeg decode and cleanup failed",
                    [original_error, cleanup_error],
                ) from None
            raise

    @staticmethod
    def _validate_text(text: object) -> None:
        if type(text) is not str:
            raise TypeError("text must be an exact built-in string")
        if not text.strip() or len(text) > _MAX_TEXT_CHARS:
            raise ValueError("text must contain 1 to 4096 characters")

    @staticmethod
    def _validate_turn_id(turn_id: object) -> None:
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id.strip() or len(turn_id) > 128:
            raise ValueError("turn_id must contain 1 to 128 characters")
