"""Bounded Moonshine native-streaming transcription."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from typing import Any, Protocol

from hermes_realtime.speech import AudioFrame, Transcript

_MAX_PENDING_BYTES = 64 * 1024 * 1024
_MAX_TRANSCRIPT_CHARS = 4096


class _MoonshineBackend(Protocol):
    def add_pcm(self, pcm: bytes, sample_rate_hz: int, channels: int) -> str | None: ...

    def finish(self) -> str: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class _NativeMoonshineBackend:
    def __init__(
        self,
        *,
        language: str,
        model_tier: str,
        update_interval_seconds: float,
    ) -> None:
        moonshine = importlib.import_module("moonshine_voice")
        numpy = importlib.import_module("numpy")
        architecture = getattr(
            moonshine.ModelArch,
            {
                "tiny": "TINY_STREAMING",
                "small": "SMALL_STREAMING",
                "medium": "MEDIUM_STREAMING",
            }[model_tier],
        )
        model_path, selected_architecture = moonshine.get_model_for_language(
            wanted_language=language,
            wanted_model_arch=architecture,
        )
        if selected_architecture != architecture:
            raise RuntimeError("Moonshine selected an unexpected model architecture")
        self._numpy = numpy
        self._transcriber = moonshine.Transcriber(
            model_path=model_path,
            model_arch=architecture,
            update_interval=update_interval_seconds,
        )
        self._update_interval_seconds = update_interval_seconds
        self._stream: Any | None = None
        self._lines: dict[int, tuple[float, str]] = {}
        self._dirty = False

    def _ensure_stream(self) -> Any:
        stream = self._stream
        if stream is not None:
            return stream
        stream = self._transcriber.create_stream(
            update_interval=self._update_interval_seconds
        )
        stream.add_listener(self._on_event)
        stream.start()
        self._stream = stream
        self._lines.clear()
        self._dirty = False
        return stream

    def _on_event(self, event: object) -> None:
        line = getattr(event, "line", None)
        if line is None:
            return
        line_id = getattr(line, "line_id", None)
        start_time = getattr(line, "start_time", None)
        text = getattr(line, "text", None)
        if type(line_id) is not int:
            raise TypeError("Moonshine line metadata is invalid")
        if type(start_time) is int:
            normalized_start_time = float(start_time)
        elif type(start_time) is float:
            normalized_start_time = start_time
        else:
            raise TypeError("Moonshine line metadata is invalid")
        if type(text) is not str:
            raise TypeError("Moonshine line text is invalid")
        cleaned = text.strip()
        if cleaned:
            self._lines[line_id] = (normalized_start_time, cleaned)
        else:
            self._lines.pop(line_id, None)
        self._dirty = True

    def _snapshot(self) -> str:
        return " ".join(
            text
            for _line_id, (_start, text) in sorted(
                self._lines.items(), key=lambda item: (item[1][0], item[0])
            )
        ).strip()

    def add_pcm(self, pcm: bytes, sample_rate_hz: int, channels: int) -> str | None:
        if channels != 1:
            raise ValueError("Moonshine backend requires mono PCM")
        audio = self._numpy.frombuffer(pcm, dtype=self._numpy.int16).astype(
            self._numpy.float32
        )
        audio /= 32768.0
        self._dirty = False
        self._ensure_stream().add_audio(audio, sample_rate_hz)
        return self._snapshot() if self._dirty else None

    def finish(self) -> str:
        stream = self._stream
        if stream is None:
            return ""
        try:
            result = stream.stop()
            if result is None:
                return self._snapshot()
            lines = getattr(result, "lines", None)
            if not isinstance(lines, list):
                raise TypeError("Moonshine final transcript lines are invalid")
            texts: list[str] = []
            for line in lines:
                text = getattr(line, "text", None)
                if type(text) is not str:
                    raise TypeError("Moonshine final transcript text is invalid")
                cleaned = text.strip()
                if cleaned:
                    texts.append(cleaned)
            final_text = " ".join(texts)
            return final_text or self._snapshot()
        finally:
            stream.close()
            self._stream = None
            self._lines.clear()
            self._dirty = False

    def cancel(self) -> None:
        stream = self._stream
        if stream is not None:
            stream.close()
        self._stream = None
        self._lines.clear()
        self._dirty = False


    def close(self) -> None:
        self.cancel()
        self._transcriber.close()


class MoonshineStreamingTranscriber:
    """Stream bounded PCM through Moonshine without blocking the asyncio loop."""

    def __init__(
        self,
        *,
        language: str = "en",
        model_tier: str = "tiny",
        update_interval_seconds: float = 0.2,
        sample_rate_hz: int = 48_000,
        channels: int = 1,
        max_pending_bytes: int = 16 * 1024 * 1024,
        max_add_bytes: int = 192_000,
        max_transcript_chars: int = _MAX_TRANSCRIPT_CHARS,
        backend: _MoonshineBackend | None = None,
        backend_factory: Callable[[], _MoonshineBackend] | None = None,
    ) -> None:
        if type(language) is not str or not language.strip() or len(language) > 32:
            raise ValueError("language must contain 1 to 32 characters")
        if type(model_tier) is not str:
            raise TypeError("model_tier must be an exact string")
        if model_tier not in {"tiny", "small", "medium"}:
            raise ValueError("model_tier must be exactly 'tiny', 'small', or 'medium'")
        if type(update_interval_seconds) not in (int, float):
            raise TypeError("update_interval_seconds must be an exact number")
        if not 0.05 <= update_interval_seconds <= 2.0:
            raise ValueError("update_interval_seconds must be between 0.05 and 2")
        if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be an exact positive integer")
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be an exact positive integer")
        if type(max_pending_bytes) is not int:
            raise TypeError("max_pending_bytes must be an exact integer")
        if not 1 <= max_pending_bytes <= _MAX_PENDING_BYTES:
            raise ValueError("max_pending_bytes exceeds supported bounds")
        if type(max_add_bytes) is not int:
            raise TypeError("max_add_bytes must be an exact integer")
        if not 1 <= max_add_bytes <= _MAX_PENDING_BYTES:
            raise ValueError("max_add_bytes exceeds supported bounds")
        if type(max_transcript_chars) is not int:
            raise TypeError("max_transcript_chars must be an exact integer")
        if not 1 <= max_transcript_chars <= _MAX_TRANSCRIPT_CHARS:
            raise ValueError("max_transcript_chars exceeds supported bounds")
        if backend is not None and backend_factory is not None:
            raise ValueError("provide backend or backend_factory, not both")
        if backend is not None:
            self._validate_backend(backend)
        elif backend_factory is not None:
            if not callable(backend_factory):
                raise TypeError("backend_factory must be callable")
            backend = backend_factory()
            self._validate_backend(backend)
        else:
            backend = _NativeMoonshineBackend(
                language=language,
                model_tier=model_tier,
                update_interval_seconds=float(update_interval_seconds),
            )

        self._backend = backend
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._max_pending_bytes = max_pending_bytes
        self._max_add_bytes = max_add_bytes
        self._max_transcript_chars = max_transcript_chars
        self._pending = bytearray()
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._epoch = 0
        self._operation: asyncio.Task[tuple[int, str | None]] | None = None
        self._finish_operation: asyncio.Task[str] | None = None
        self._last_partial: str | None = None
        self._settling = False
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @staticmethod
    def _validate_backend(backend: object) -> None:
        for method in ("add_pcm", "finish", "cancel", "close"):
            if not callable(getattr(backend, method, None)):
                raise TypeError(f"backend must provide {method}()")

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
            if self._closed:
                raise RuntimeError("Moonshine transcriber is closed")
            if self._settling or self._closing:
                raise RuntimeError("Moonshine utterance is settling")
            partial = self._harvest_locked()
            projected = len(self._pending) + len(frame.pcm)
            if projected > self._max_pending_bytes:
                raise RuntimeError("Moonshine pending PCM capacity exhausted")
            self._pending.extend(frame.pcm)
            self._launch_locked()
        if partial is None:
            return ()
        return (Transcript(text=partial, final=False),)

    async def finish_utterance(self) -> Transcript | None:
        operation: asyncio.Task[str] | None = None
        async with self._lifecycle_lock:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("Moonshine transcriber is closed")
                if self._settling or self._closing:
                    raise RuntimeError("Moonshine utterance is already settling")
                self._settling = True
                epoch = self._epoch
            try:
                await self._drain_pending(epoch)
                operation = asyncio.create_task(
                    asyncio.to_thread(self._backend.finish),
                    name=f"moonshine-finish:{epoch}",
                )
                async with self._lock:
                    self._finish_operation = operation
                text = await asyncio.shield(operation)
                cleaned = self._clean_text(text, final=True)
                async with self._lock:
                    if epoch != self._epoch:
                        raise RuntimeError("Moonshine utterance was cancelled")
                    final_text = cleaned if cleaned is not None else self._last_partial
                    self._last_partial = None
                return (
                    None
                    if final_text is None
                    else Transcript(text=final_text, final=True)
                )
            finally:
                async with self._lock:
                    if operation is None or operation.done():
                        if epoch == self._epoch:
                            self._last_partial = None
                        if self._finish_operation is operation:
                            self._finish_operation = None
                        self._settling = False

    async def cancel(self) -> None:
        await self._cancel(allow_closing=False)

    async def _cancel(self, *, allow_closing: bool) -> None:
        async with self._lifecycle_lock:
            async with self._lock:
                if self._closed or (self._closing and not allow_closing):
                    return
                self._settling = True
                self._epoch += 1
                self._pending.clear()
                operation = self._operation
                finish_operation = self._finish_operation
            errors: list[BaseException] = []

            async def settle_owned(owned_operation: asyncio.Task[Any]) -> None:
                try:
                    await asyncio.shield(owned_operation)
                except BaseException as error:
                    errors.append(error)
                    if not owned_operation.done():
                        try:
                            await asyncio.shield(owned_operation)
                        except BaseException as settle_error:
                            if settle_error is not error:
                                errors.append(settle_error)

            try:
                if operation is not None:
                    await settle_owned(operation)
                if finish_operation is not None:
                    await settle_owned(finish_operation)
                cancel_operation = asyncio.create_task(
                    asyncio.to_thread(self._backend.cancel),
                    name=f"moonshine-cancel:{self._epoch}",
                )
                await settle_owned(cancel_operation)
            finally:
                async with self._lock:
                    if self._operation is operation:
                        self._operation = None
                    if self._finish_operation is finish_operation:
                        self._finish_operation = None
                    self._last_partial = None
                    self._settling = False
            if errors:
                raise BaseExceptionGroup("Moonshine cancellation failed", errors)

    async def close(self) -> None:
        async with self._close_lock:
            async with self._lock:
                if self._closed:
                    return
                self._closing = True
            errors: list[BaseException] = []
            cancel_operation = asyncio.create_task(
                self._cancel(allow_closing=True),
                name="moonshine-close-drain",
            )
            try:
                await asyncio.shield(cancel_operation)
            except BaseException as error:
                errors.append(error)
                if not cancel_operation.done():
                    try:
                        await asyncio.shield(cancel_operation)
                    except BaseException as settle_error:
                        if settle_error is not error:
                            errors.append(settle_error)
            close_operation = asyncio.create_task(
                asyncio.to_thread(self._backend.close),
                name="moonshine-close",
            )
            try:
                await asyncio.shield(close_operation)
            except BaseException as error:
                errors.append(error)
                if not close_operation.done():
                    try:
                        await asyncio.shield(close_operation)
                    except BaseException as settle_error:
                        if settle_error is not error:
                            errors.append(settle_error)
            native_close_succeeded = (
                close_operation.done()
                and not close_operation.cancelled()
                and close_operation.exception() is None
            )
            async with self._lock:
                self._closed = native_close_succeeded
                if native_close_succeeded:
                    self._closing = False
            if errors:
                raise BaseExceptionGroup("Moonshine close failed", errors)

    async def _drain_pending(self, epoch: int) -> None:
        while True:
            async with self._lock:
                self._harvest_locked()
                self._launch_locked(allow_settling=True)
                operation = self._operation
                if operation is None:
                    return
            await asyncio.shield(operation)
            async with self._lock:
                if epoch != self._epoch:
                    raise RuntimeError("Moonshine utterance was cancelled")

    def _launch_locked(self, *, allow_settling: bool = False) -> None:
        operation = self._operation
        if operation is not None:
            return
        if not self._pending or (self._settling and not allow_settling):
            return
        add_bytes = min(len(self._pending), self._max_add_bytes)
        pcm = bytes(self._pending[:add_bytes])
        del self._pending[:add_bytes]
        epoch = self._epoch
        self._operation = asyncio.create_task(
            self._add_owned(epoch, pcm),
            name=f"moonshine-add-audio:{epoch}",
        )

    async def _add_owned(self, epoch: int, pcm: bytes) -> tuple[int, str | None]:
        text = await asyncio.to_thread(
            self._backend.add_pcm,
            pcm,
            self._sample_rate_hz,
            self._channels,
        )
        cleaned = self._clean_text(text, final=False)
        return epoch, cleaned

    def _harvest_locked(self) -> str | None:
        operation = self._operation
        if operation is None or not operation.done():
            return None
        self._operation = None
        epoch, text = operation.result()
        if epoch != self._epoch or text is None or text == self._last_partial:
            return None
        self._last_partial = text
        return text

    def _clean_text(self, value: object, *, final: bool) -> str | None:
        if type(value) is not str and value is not None:
            label = "final" if final else "partial"
            raise TypeError(f"Moonshine {label} transcript must be an exact string or None")
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if len(cleaned) > self._max_transcript_chars:
            raise ValueError("Moonshine transcript exceeds supported size")
        return cleaned
