"""Bounded faster-whisper utterance transcriber with retained cancellation ownership."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import wave
from collections.abc import Awaitable, Callable
from typing import cast

from hermes_realtime.speech import AudioFrame, Transcript

_MAX_UTTERANCE_BYTES = 64 * 1024 * 1024
_MAX_TRANSCRIPT_CHARS = 65_536
_MAX_MODEL_NAME_CHARS = 256

_TranscribeWav = Callable[[bytes], str | Awaitable[str]]


class FasterWhisperTranscriber:
    """Buffer one bounded utterance and transcribe it outside the event loop."""

    def __init__(
        self,
        *,
        model_size_or_path: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = "en",
        sample_rate_hz: int = 48_000,
        channels: int = 1,
        max_utterance_bytes: int = 16 * 1024 * 1024,
        transcribe_wav: _TranscribeWav | None = None,
    ) -> None:
        for name, value in (
            ("model_size_or_path", model_size_or_path),
            ("device", device),
            ("compute_type", compute_type),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact built-in string")
            if not value.strip() or len(value) > _MAX_MODEL_NAME_CHARS:
                raise ValueError(f"{name} must contain 1 to 256 characters")
        if language is not None:
            if type(language) is not str:
                raise TypeError("language must be an exact built-in string or None")
            if not language.strip() or len(language) > 32:
                raise ValueError("language must contain 1 to 32 characters")
        if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be an exact positive integer")
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be an exact positive integer")
        if type(max_utterance_bytes) is not int:
            raise TypeError("max_utterance_bytes must be an exact integer")
        if not 1 <= max_utterance_bytes <= _MAX_UTTERANCE_BYTES:
            raise ValueError("max_utterance_bytes exceeds supported bounds")
        if transcribe_wav is not None and not callable(transcribe_wav):
            raise TypeError("transcribe_wav must be callable")

        if transcribe_wav is None:
            try:
                faster_whisper = importlib.import_module("faster_whisper")
            except ImportError as error:
                raise RuntimeError(
                    "faster-whisper STT requires the 'faster-whisper' local extra"
                ) from error
            model = faster_whisper.WhisperModel(
                model_size_or_path,
                device=device,
                compute_type=compute_type,
            )

            def transcribe(payload: bytes) -> str:
                segments, _info = model.transcribe(
                    io.BytesIO(payload),
                    language=language,
                    beam_size=1,
                    best_of=1,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                return "".join(cast(str, segment.text) for segment in segments)

            transcribe_wav = transcribe

        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._max_utterance_bytes = max_utterance_bytes
        self._transcribe_wav = transcribe_wav
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._epoch = 0
        self._operation: asyncio.Task[str] | None = None

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        if type(frame) is not AudioFrame:
            raise TypeError("frame must be an exact AudioFrame")
        frame = AudioFrame(
            pcm=frame.pcm,
            sample_rate_hz=frame.sample_rate_hz,
            channels=frame.channels,
        )
        if frame.sample_rate_hz != self._sample_rate_hz or frame.channels != self._channels:
            raise ValueError("STT frame format does not match its configuration")
        async with self._lock:
            operation = self._operation
            if operation is not None and not operation.done():
                raise RuntimeError("an utterance transcription is already active")
            projected = len(self._buffer) + len(frame.pcm)
            if projected > self._max_utterance_bytes:
                raise RuntimeError("STT utterance byte capacity exhausted")
            self._buffer.extend(frame.pcm)
        return ()

    async def finish_utterance(self) -> Transcript | None:
        async with self._lock:
            operation = self._operation
            if operation is not None and not operation.done():
                raise RuntimeError("an utterance transcription is already active")
            if not self._buffer:
                raise RuntimeError("no buffered audio is available for transcription")
            pcm = bytes(self._buffer)
            self._buffer.clear()
            epoch = self._epoch
            operation = asyncio.create_task(
                self._transcribe_owned(self._wav(pcm)),
                name=f"faster-whisper-transcription:{epoch}",
            )
            self._operation = operation
        try:
            text = await asyncio.shield(operation)
            async with self._lock:
                if epoch != self._epoch:
                    raise RuntimeError("utterance transcription was cancelled")
            if not text:
                return None
            return Transcript(text=text, final=True)
        finally:
            if operation.done():
                async with self._lock:
                    if self._operation is operation:
                        self._operation = None

    async def cancel(self) -> None:
        async with self._lock:
            self._epoch += 1
            self._buffer.clear()
            operation = self._operation
        if operation is not None:
            await asyncio.shield(operation)
            async with self._lock:
                if self._operation is operation:
                    self._operation = None

    async def _transcribe_owned(self, payload: bytes) -> str:
        result = await asyncio.to_thread(self._transcribe_wav, payload)
        if inspect.isawaitable(result):
            result = await result
        if type(result) is not str:
            raise TypeError("faster-whisper backend must return an exact string")
        text = result.strip()
        if len(text) > _MAX_TRANSCRIPT_CHARS:
            raise ValueError("faster-whisper transcript exceeds supported size")
        return text

    def _wav(self, pcm: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as recording:
            recording.setnchannels(self._channels)
            recording.setsampwidth(2)
            recording.setframerate(self._sample_rate_hz)
            recording.writeframes(pcm)
        return output.getvalue()
