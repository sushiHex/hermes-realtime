"""Provider-neutral rendered-duration speech segmentation."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from .ports import StreamingSynthesizer
from .types import AudioFrame, SpeechChunk, WordTiming

_MAX_TEXT_CHARS = 65_536
_MAX_TURN_ID_CHARS = 128
_MAX_INITIAL_SEGMENT_CHARS = 4096
_MAX_SPLIT_DEPTH = 16
_MAX_PROVIDER_ATTEMPTS = 512
_MAX_PROVIDER_CHUNKS_PER_ATTEMPT = 256
_BOUNDARY_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+")


class SpeechDurationLimitError(RuntimeError):
    """Provider output cannot satisfy the rendered-duration contract safely."""


@dataclass(slots=True, eq=False)
class _SynthesisOperation:
    provider_prefix: str
    attempts: int = 0
    output_sequence: int = 0
    active_provider_turn_id: str | None = None
    cancelled: bool = False


class DurationBoundedSynthesizer:
    """Validate and re-segment timingless provider output to a PCM duration bound."""

    def __init__(
        self,
        synthesizer: StreamingSynthesizer,
        *,
        max_duration_seconds: float = 3.0,
        initial_segment_chars: int = 160,
        max_split_depth: int = 8,
        max_provider_attempts: int = 64,
    ) -> None:
        if not callable(getattr(synthesizer, "synthesize", None)) or not callable(
            getattr(synthesizer, "cancel", None)
        ):
            raise TypeError("synthesizer must provide synthesize() and cancel()")
        if type(max_duration_seconds) is not float:
            raise TypeError("max_duration_seconds must be an exact float")
        if not math.isfinite(max_duration_seconds) or not (0 < max_duration_seconds <= 3.0):
            raise ValueError("max_duration_seconds must be finite and no greater than 3.0")
        for name, value, ceiling in (
            ("initial_segment_chars", initial_segment_chars, _MAX_INITIAL_SEGMENT_CHARS),
            ("max_split_depth", max_split_depth, _MAX_SPLIT_DEPTH),
            ("max_provider_attempts", max_provider_attempts, _MAX_PROVIDER_ATTEMPTS),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= ceiling:
                raise ValueError(f"{name} is outside the supported range")
        self._synthesizer = synthesizer
        self._max_duration_seconds = max_duration_seconds
        self._initial_segment_chars = initial_segment_chars
        self._max_split_depth = max_split_depth
        self._max_provider_attempts = max_provider_attempts
        self._operations: dict[str, _SynthesisOperation] = {}
        self._operation_sequence = 0
        self._lock = asyncio.Lock()

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self._validate_text(text)
        self._validate_turn_id(turn_id)
        async with self._lock:
            if turn_id in self._operations:
                raise RuntimeError("turn already owns duration-bounded synthesis")
            self._operation_sequence += 1
            digest = hashlib.sha256(
                f"{self._operation_sequence}\0{turn_id}".encode()
            ).hexdigest()[:20]
            operation = _SynthesisOperation(
                provider_prefix=f"bounded_{digest}",
            )
            self._operations[turn_id] = operation
        try:
            for segment in self._split_to_limit(text, self._initial_segment_chars):
                async for chunk in self._synthesize_bounded(
                    segment,
                    public_turn_id=turn_id,
                    operation=operation,
                    depth=0,
                ):
                    yield chunk
        finally:
            async with self._lock:
                if self._operations.get(turn_id) is operation:
                    del self._operations[turn_id]

    async def cancel(self, turn_id: str) -> None:
        self._validate_turn_id(turn_id)
        async with self._lock:
            operation = self._operations.get(turn_id)
            if operation is None:
                return
            operation.cancelled = True
            provider_turn_id = operation.active_provider_turn_id
        if provider_turn_id is not None:
            await self._synthesizer.cancel(provider_turn_id)

    async def _synthesize_bounded(
        self,
        text: str,
        *,
        public_turn_id: str,
        operation: _SynthesisOperation,
        depth: int,
    ) -> AsyncIterator[SpeechChunk]:
        if depth > self._max_split_depth:
            raise SpeechDurationLimitError("speech split depth exhausted")
        async with self._lock:
            self._require_operation(public_turn_id, operation)
            if operation.attempts >= self._max_provider_attempts:
                raise SpeechDurationLimitError("speech provider attempt budget exhausted")
            operation.attempts += 1
            provider_turn_id = f"{operation.provider_prefix}_{operation.attempts:04x}"
            operation.active_provider_turn_id = provider_turn_id
        stream = self._synthesizer.synthesize(text, provider_turn_id)
        chunks: list[SpeechChunk] = []
        iteration_error: BaseException | None = None
        try:
            async for candidate in stream:
                if type(candidate) is not SpeechChunk:
                    raise TypeError("synthesizer must yield exact SpeechChunk values")
                chunk = SpeechChunk(
                    turn_id=candidate.turn_id,
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    audio=candidate.audio,
                    word_timings=candidate.word_timings,
                    timing_source=candidate.timing_source,
                )
                if chunk.turn_id != provider_turn_id:
                    raise ValueError("provider chunk turn_id does not match private operation")
                chunks.append(chunk)
                if len(chunks) > _MAX_PROVIDER_CHUNKS_PER_ATTEMPT:
                    raise SpeechDurationLimitError("provider chunk count exceeded")
        except BaseException as error:
            iteration_error = error
        provider_cancel_error: BaseException | None = None
        if isinstance(iteration_error, asyncio.CancelledError):
            async with self._lock:
                cancellation_already_requested = operation.cancelled
            if not cancellation_already_requested:
                try:
                    await self._synthesizer.cancel(provider_turn_id)
                except BaseException as error:
                    provider_cancel_error = error
        close_error: BaseException | None = None
        close_candidate = getattr(stream, "aclose", None)
        if callable(close_candidate):
            close = cast(Callable[[], Awaitable[None]], close_candidate)
            try:
                await close()
            except BaseException as error:
                close_error = error
        async with self._lock:
            if operation.active_provider_turn_id == provider_turn_id:
                operation.active_provider_turn_id = None
            operation_valid = (
                self._operations.get(public_turn_id) is operation
                and not operation.cancelled
            )
        cleanup_errors = tuple(
            error
            for error in (provider_cancel_error, close_error)
            if error is not None
        )
        if iteration_error is not None and cleanup_errors:
            raise BaseExceptionGroup(
                "speech synthesis iteration and cleanup failed",
                [iteration_error, *cleanup_errors],
            ) from None
        if iteration_error is not None:
            raise iteration_error
        if cleanup_errors:
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            raise BaseExceptionGroup(
                "speech synthesis cleanup failed",
                list(cleanup_errors),
            ) from None
        if not operation_valid:
            raise asyncio.CancelledError
        if not chunks:
            raise SpeechDurationLimitError("provider returned no speech chunks")
        if "".join(chunk.text for chunk in chunks) != text:
            raise SpeechDurationLimitError("provider chunks do not preserve exact source text")

        for chunk in chunks:
            duration_seconds = len(chunk.audio.pcm) / (
                2 * chunk.audio.channels * chunk.audio.sample_rate_hz
            )
            if duration_seconds <= self._max_duration_seconds:
                async with self._lock:
                    self._require_operation(public_turn_id, operation)
                    operation.output_sequence += 1
                    output_sequence = operation.output_sequence
                yield SpeechChunk(
                    turn_id=public_turn_id,
                    chunk_id=f"{operation.provider_prefix}_out_{output_sequence:04x}",
                    text=chunk.text,
                    audio=chunk.audio,
                    word_timings=chunk.word_timings,
                    timing_source=chunk.timing_source,
                )
                continue
            if chunk.timing_source == "provider" and chunk.word_timings:
                qualified_chunks = self._split_qualified_chunk(chunk)
                for qualified in qualified_chunks:
                    async with self._lock:
                        self._require_operation(public_turn_id, operation)
                        operation.output_sequence += 1
                        output_sequence = operation.output_sequence
                    yield SpeechChunk(
                        turn_id=public_turn_id,
                        chunk_id=(
                            f"{operation.provider_prefix}_out_{output_sequence:04x}"
                        ),
                        text=qualified.text,
                        audio=qualified.audio,
                        word_timings=qualified.word_timings,
                        timing_source=qualified.timing_source,
                    )
                continue
            split = self._bisect_text(chunk.text)
            if split is None:
                raise SpeechDurationLimitError(
                    "overlong provider output has no safe text boundary"
                )
            if depth >= self._max_split_depth:
                raise SpeechDurationLimitError("speech split depth exhausted")
            for child in split:
                async for bounded in self._synthesize_bounded(
                    child,
                    public_turn_id=public_turn_id,
                    operation=operation,
                    depth=depth + 1,
                ):
                    yield bounded

    def _split_qualified_chunk(self, chunk: SpeechChunk) -> tuple[SpeechChunk, ...]:
        sample_rate_hz = chunk.audio.sample_rate_hz
        channels = chunk.audio.channels
        sample_width_bytes = 2 * channels
        total_samples = len(chunk.audio.pcm) // sample_width_bytes
        max_samples = int(self._max_duration_seconds * sample_rate_hz)
        if max_samples < 1:
            raise SpeechDurationLimitError("speech duration bound has no sample capacity")
        sample_start = 0
        text_start = 0
        boundaries: list[tuple[int, int]] = []
        while total_samples - sample_start > max_samples:
            eligible = tuple(
                timing
                for timing in chunk.word_timings
                if timing.end_sample > sample_start
                and timing.end_sample - sample_start <= max_samples
                and timing.text_end > text_start
            )
            if not eligible:
                raise SpeechDurationLimitError(
                    "qualified timing map has no safe boundary inside the duration limit"
                )
            boundary = eligible[-1]
            boundaries.append((boundary.end_sample, boundary.text_end))
            sample_start = boundary.end_sample
            text_start = boundary.text_end
        boundaries.append((total_samples, len(chunk.text)))

        output: list[SpeechChunk] = []
        sample_start = 0
        text_start = 0
        for index, (sample_end, text_end) in enumerate(boundaries, start=1):
            if (
                sample_end <= sample_start
                or sample_end - sample_start > max_samples
                or text_end <= text_start
                or not chunk.text[text_start:text_end].strip()
            ):
                raise SpeechDurationLimitError("qualified timing split is not monotonic")
            timings = tuple(
                WordTiming(
                    word=timing.word,
                    start_sample=timing.start_sample - sample_start,
                    end_sample=timing.end_sample - sample_start,
                    text_start=timing.text_start - text_start,
                    text_end=timing.text_end - text_start,
                )
                for timing in chunk.word_timings
                if timing.start_sample >= sample_start
                and timing.end_sample <= sample_end
                and timing.text_start >= text_start
                and timing.text_end <= text_end
            )
            if not timings:
                raise SpeechDurationLimitError(
                    "qualified timing split produced a chunk without word authority"
                )
            pcm_start = sample_start * sample_width_bytes
            pcm_end = sample_end * sample_width_bytes
            output.append(
                SpeechChunk(
                    turn_id=chunk.turn_id,
                    chunk_id=f"{chunk.chunk_id}_split_{index:04x}",
                    text=chunk.text[text_start:text_end],
                    audio=AudioFrame(
                        pcm=chunk.audio.pcm[pcm_start:pcm_end],
                        sample_rate_hz=sample_rate_hz,
                        channels=channels,
                    ),
                    word_timings=timings,
                    timing_source="provider",
                )
            )
            sample_start = sample_end
            text_start = text_end
        candidate = tuple(output)
        if (
            b"".join(part.audio.pcm for part in candidate) != chunk.audio.pcm
            or "".join(part.text for part in candidate) != chunk.text
        ):
            raise SpeechDurationLimitError(
                "qualified timing split did not preserve exact text and PCM"
            )
        return candidate

    def _require_operation(
        self,
        turn_id: str,
        operation: _SynthesisOperation,
    ) -> None:
        if self._operations.get(turn_id) is not operation or operation.cancelled:
            raise asyncio.CancelledError

    @classmethod
    def _split_to_limit(cls, text: str, limit: int) -> tuple[str, ...]:
        pending = [text]
        output: list[str] = []
        while pending:
            candidate = pending.pop()
            if len(candidate) <= limit:
                output.append(candidate)
                continue
            split = cls._bisect_text(candidate, target=limit)
            if split is None:
                raise SpeechDurationLimitError(
                    "speech text exceeds the initial segment bound without a safe boundary"
                )
            pending.extend(reversed(split))
        segments = tuple(output)
        if "".join(segments) != text or any(not part.strip() for part in segments):
            raise SpeechDurationLimitError("speech segmentation did not preserve exact text")
        return segments

    @staticmethod
    def _bisect_text(
        text: str,
        *,
        target: int | None = None,
    ) -> tuple[str, str] | None:
        desired = min(target or (len(text) // 2), len(text) - 1)
        sentence_boundaries = tuple(match.end() for match in _SENTENCE_BOUNDARY_RE.finditer(text))
        whitespace_boundaries = tuple(match.end() for match in _BOUNDARY_RE.finditer(text))
        candidates = sentence_boundaries or whitespace_boundaries
        candidates = tuple(
            offset
            for offset in candidates
            if 0 < offset < len(text) and text[:offset].strip() and text[offset:].strip()
        )
        if not candidates and sentence_boundaries:
            candidates = tuple(
                offset
                for offset in whitespace_boundaries
                if 0 < offset < len(text) and text[:offset].strip() and text[offset:].strip()
            )
        if not candidates:
            return None
        boundary = min(candidates, key=lambda offset: (abs(offset - desired), offset))
        return text[:boundary], text[boundary:]

    @staticmethod
    def _validate_text(text: object) -> None:
        if type(text) is not str:
            raise TypeError("text must be an exact built-in string")
        if not text.strip() or len(text) > _MAX_TEXT_CHARS:
            raise ValueError("text must contain 1 to 65536 characters")

    @staticmethod
    def _validate_turn_id(turn_id: object) -> None:
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id.strip() or len(turn_id) > _MAX_TURN_ID_CHARS:
            raise ValueError("turn_id must contain 1 to 128 characters")
