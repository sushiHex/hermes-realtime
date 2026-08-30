"""Bounded outbound-reference residual echo classification for local playback."""

from __future__ import annotations

import logging
import math
import time
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from hermes_realtime.speech import AudioFrame

_LOGGER = logging.getLogger(__name__)
_MAX_REFERENCE_BYTES = 64 * 1024 * 1024
_MAX_RETIRED_REFERENCES = 256
_DEFAULT_MAX_ANALYSIS_BYTES = 256 * 1024
_MAX_ANALYSIS_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_REFERENCE_ANALYSIS_BYTES = 4 * 1024 * 1024
_MAX_REFERENCE_ANALYSIS_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_CORRELATION_WORK = 16_000_000
_MAX_CORRELATION_WORK = 64_000_000
_MAX_REFERENCE_TEXT_CHARS = 65_536
_MINIMUM_TRANSCRIPT_ECHO_ALNUM = 8
_MINIMUM_SHORT_MULTIWORD_TRANSCRIPT_ECHO_ALNUM = 7
_ROBUST_WINDOW_SECONDS = 0.24
_ROBUST_MINIMUM_SECONDS = 0.16
_ROBUST_FRAME_SECONDS = 0.02
_ROBUST_HOP_SECONDS = 0.01
_MINIMUM_ROBUST_ENVELOPE_COHERENCE = 0.90
_MINIMUM_ROBUST_BAND_TEMPORAL_COHERENCE = 0.80
_MAXIMUM_ROBUST_RELATIVE_MIC_LEVEL = 0.20
_MAX_ROBUST_FEATURE_WORK = 4_000_000


@dataclass(slots=True)
class _Reference:
    frame: AudioFrame
    normalized_text: str
    token: object
    started_at: float
    ended_at: float | None = None


class PlaybackEchoGuard:
    """Classify mic windows against exact active/recent outbound PCM."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        tail_seconds: float = 1.5,
        max_delay_seconds: float = 1.5,
        analysis_rate_hz: int = 8_000,
        delay_step_ms: int = 10,
        initial_max_residual: float = 0.28,
        minimum_coherence: float = 0.55,
        maximum_relative_mic_level: float = 0.05,
        max_retired_references: int = _MAX_RETIRED_REFERENCES,
        max_reference_bytes: int = _MAX_REFERENCE_BYTES,
        max_analysis_bytes: int = _DEFAULT_MAX_ANALYSIS_BYTES,
        max_reference_analysis_bytes: int = _DEFAULT_MAX_REFERENCE_ANALYSIS_BYTES,
        max_correlation_work: int = _DEFAULT_MAX_CORRELATION_WORK,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        for name, value, lower, upper in (
            ("tail_seconds", tail_seconds, 0.1, 5.0),
            ("max_delay_seconds", max_delay_seconds, 0.1, 3.0),
            ("initial_max_residual", initial_max_residual, 0.01, 0.75),
            ("minimum_coherence", minimum_coherence, 0.1, 0.99),
            ("maximum_relative_mic_level", maximum_relative_mic_level, 0.001, 0.5),
        ):
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be an exact number")
            if not math.isfinite(value) or not lower <= float(value) <= upper:
                raise ValueError(f"{name} is outside the supported range")
        if type(analysis_rate_hz) is not int or not 4_000 <= analysis_rate_hz <= 24_000:
            raise ValueError("analysis_rate_hz is outside the supported range")
        if type(delay_step_ms) is not int or not 2 <= delay_step_ms <= 50:
            raise ValueError("delay_step_ms is outside the supported range")
        if tail_seconds > max_delay_seconds:
            raise ValueError("max_delay_seconds must cover tail_seconds")
        if (
            type(max_retired_references) is not int
            or not 1 <= max_retired_references <= _MAX_RETIRED_REFERENCES
        ):
            raise ValueError("max_retired_references is outside the supported range")
        if (
            type(max_reference_bytes) is not int
            or not 1 <= max_reference_bytes <= _MAX_REFERENCE_BYTES
        ):
            raise ValueError("max_reference_bytes is outside the supported range")
        if (
            type(max_analysis_bytes) is not int
            or not 1 <= max_analysis_bytes <= _MAX_ANALYSIS_BYTES
        ):
            raise ValueError("max_analysis_bytes is outside the supported range")
        if (
            type(max_reference_analysis_bytes) is not int
            or not 1 <= max_reference_analysis_bytes <= _MAX_REFERENCE_ANALYSIS_BYTES
        ):
            raise ValueError("max_reference_analysis_bytes is outside the supported range")
        if (
            type(max_correlation_work) is not int
            or not 1 <= max_correlation_work <= _MAX_CORRELATION_WORK
        ):
            raise ValueError("max_correlation_work is outside the supported range")
        self._clock = clock
        self._tail_seconds = float(tail_seconds)
        self._max_delay_seconds = float(max_delay_seconds)
        self._analysis_rate_hz = analysis_rate_hz
        self._delay_step_ms = delay_step_ms
        self._initial_max_residual = float(initial_max_residual)
        self._minimum_coherence = float(minimum_coherence)
        self._maximum_relative_mic_level = float(maximum_relative_mic_level)
        self._max_retired_references = max_retired_references
        self._max_reference_bytes = max_reference_bytes
        self._max_analysis_bytes = max_analysis_bytes
        self._max_reference_analysis_bytes = max_reference_analysis_bytes
        self._max_correlation_work = max_correlation_work
        self._active: _Reference | None = None
        self._retired: deque[_Reference] = deque()
        self._echo_residual_baseline: float | None = None

    def begin_playback(self, frame: AudioFrame, text: str = "") -> object:
        trusted = self._trusted_frame(frame)
        normalized_text = self._normalized_text(text)
        if self._active is not None:
            raise RuntimeError("echo reference already has active playback")
        now = float(self._clock())
        self._purge_retired(now)
        retained_bytes = sum(
            len(reference.frame.pcm) + len(reference.normalized_text.encode("utf-8"))
            for reference in self._retired
        )
        if (
            len(self._retired) >= self._max_retired_references
            or retained_bytes + len(trusted.pcm) + len(normalized_text.encode("utf-8"))
            > self._max_reference_bytes
        ):
            raise RuntimeError("echo reference retention capacity exhausted")
        token = object()
        self._active = _Reference(
            frame=trusted,
            normalized_text=normalized_text,
            token=token,
            started_at=now,
        )
        return token

    def is_transcript_echo(self, text: str) -> bool:
        candidate = self._normalized_text(text)
        if not candidate:
            return False
        now = float(self._clock())
        self._purge_retired(now)
        references = (
            *((self._active,) if self._active is not None else ()),
            *self._retired,
        )
        compact_length = sum(character.isalnum() for character in candidate)
        if compact_length < _MINIMUM_SHORT_MULTIWORD_TRANSCRIPT_ECHO_ALNUM:
            return False
        if (
            compact_length < _MINIMUM_TRANSCRIPT_ECHO_ALNUM
            and len(candidate.split()) < 2
        ):
            return False
        for reference in references:
            outbound = reference.normalized_text
            if not outbound:
                continue
            if candidate == outbound or f" {candidate} " in f" {outbound} ":
                return True
        return False

    def has_recent_playback(self) -> bool:
        """Whether active or retained playback can still contaminate microphone PCM."""

        now = float(self._clock())
        self._purge_retired(now)
        return self._active is not None or bool(self._retired)

    def end_playback(self, token: object) -> None:
        active = self._active
        if active is None or active.token is not token:
            return
        active.ended_at = float(self._clock())
        self._retired.append(active)
        self._active = None
        self._purge_retired(active.ended_at)

    def is_echo_dominated(self, frames: tuple[AudioFrame, ...]) -> bool:
        if type(frames) is not tuple or not frames:
            return False
        trusted = tuple(self._trusted_frame(frame) for frame in frames)
        if sum(len(frame.pcm) for frame in trusted) > self._max_analysis_bytes:
            raise ValueError("echo analysis window exceeds capacity")
        sample_rate_hz = trusted[0].sample_rate_hz
        if any(frame.sample_rate_hz != sample_rate_hz or frame.channels != 1 for frame in trusted):
            raise ValueError("echo analysis frames must share one mono format")
        if sample_rate_hz % self._analysis_rate_hz != 0:
            raise ValueError("echo analysis sample rate is not evenly downsampleable")
        now = float(self._clock())
        self._purge_retired(now)
        references = tuple(
            reference
            for reference in (
                *((self._active,) if self._active is not None else ()),
                *self._retired,
            )
            if reference.frame.sample_rate_hz == sample_rate_hz
        )
        if not references:
            return False
        mic = np.concatenate(
            [np.frombuffer(frame.pcm, dtype="<i2").astype(np.float64) for frame in trusted]
        )
        factor = sample_rate_hz // self._analysis_rate_hz
        mic = self._downsample(mic, factor)
        if len(mic) < self._analysis_rate_hz // 20:
            return False
        mic -= float(mic.mean())
        mic_energy = float(mic @ mic)
        if mic_energy < len(mic) * 16.0:
            return False

        max_delay_samples = round(self._max_delay_seconds * self._analysis_rate_hz)
        reference_windows: list[tuple[memoryview, int]] = []
        reference_analysis_bytes = 0
        for reference in references:
            reference_time = now if reference.ended_at is None else min(now, reference.ended_at)
            elapsed_samples = round(
                (reference_time - reference.started_at) * self._analysis_rate_hz
            )
            available_samples = len(reference.frame.pcm) // (2 * factor)
            window_start = max(0, elapsed_samples - max_delay_samples - len(mic))
            window_end = min(available_samples, elapsed_samples)
            if window_start >= window_end:
                continue
            raw_start = window_start * factor * 2
            raw_end = window_end * factor * 2
            window = memoryview(reference.frame.pcm)[raw_start:raw_end]
            reference_analysis_bytes += window.nbytes
            if reference_analysis_bytes > self._max_reference_analysis_bytes:
                raise ValueError("echo reference analysis exceeds capacity")
            reference_windows.append((window, elapsed_samples - window_start))

        step_samples = max(1, round(self._delay_step_ms * self._analysis_rate_hz / 1_000))
        correlation_work = 0
        for window, elapsed_samples in reference_windows:
            outbound_length = window.nbytes // (2 * factor)
            for delay_samples in range(0, max_delay_samples + 1, step_samples):
                end = elapsed_samples - delay_samples
                start = end - len(mic)
                if start < 0 or end > outbound_length:
                    continue
                correlation_work += len(mic)
                if correlation_work > self._max_correlation_work:
                    raise ValueError("echo correlation work exceeds capacity")

        best_coherence = 0.0
        best_delay_ms = 0
        best_reference_rms = 0.0
        robust_window_samples = min(
            len(mic),
            round(_ROBUST_WINDOW_SECONDS * self._analysis_rate_hz),
        )
        robust_minimum_samples = round(
            _ROBUST_MINIMUM_SECONDS * self._analysis_rate_hz
        )
        mic_recent = (
            mic[-robust_window_samples:]
            if robust_window_samples >= robust_minimum_samples
            else None
        )
        mic_recent_rms = (
            float(np.sqrt(np.mean(mic_recent * mic_recent)))
            if mic_recent is not None
            else 0.0
        )
        robust_candidates: list[
            tuple[float, NDArray[np.float64], float, int]
        ] = []
        robust_feature_work = 0
        for window, elapsed_samples in reference_windows:
            pcm = np.frombuffer(window, dtype="<i2").astype(np.float64)
            outbound = self._downsample(pcm, factor)
            for delay_samples in range(0, max_delay_samples + 1, step_samples):
                end = elapsed_samples - delay_samples
                start = end - len(mic)
                if start < 0 or end > len(outbound):
                    continue
                candidate = outbound[start:end]
                candidate_mean = float(candidate.mean())
                reference_energy = float(candidate @ candidate) - len(candidate) * candidate_mean**2
                if reference_energy < len(candidate) * 16.0:
                    continue
                dot = float(mic @ candidate)
                coherence = (dot * dot) / (mic_energy * reference_energy)
                if coherence > best_coherence:
                    best_coherence = min(1.0, coherence)
                    best_delay_ms = round(delay_samples * 1_000 / self._analysis_rate_hz)
                    best_reference_rms = float(np.sqrt(reference_energy / len(candidate)))
                if mic_recent is not None:
                    candidate_recent = candidate[-len(mic_recent) :]
                    envelope_coherence = self._envelope_coherence(
                        mic_recent,
                        candidate_recent,
                        self._analysis_rate_hz,
                    )
                    candidate_rms = float(
                        np.sqrt(np.mean(candidate_recent * candidate_recent))
                    )
                    robust_ratio = mic_recent_rms / max(candidate_rms, 1.0)
                    if (
                        envelope_coherence
                        < _MINIMUM_ROBUST_ENVELOPE_COHERENCE
                        or robust_ratio > _MAXIMUM_ROBUST_RELATIVE_MIC_LEVEL
                    ):
                        continue
                    robust_feature_work += len(candidate_recent)
                    if robust_feature_work > _MAX_ROBUST_FEATURE_WORK:
                        raise ValueError("echo robust feature work exceeds capacity")
                    robust_candidates.append(
                        (
                            envelope_coherence,
                            candidate_recent,
                            candidate_rms,
                            delay_samples,
                        )
                    )

        residual = 1.0 - best_coherence
        baseline = self._echo_residual_baseline
        residual_limit = (
            self._initial_max_residual
            if baseline is None
            else min(0.45, max(0.10, baseline * 1.75 + 0.03))
        )
        mic_rms = float(np.sqrt(mic_energy / len(mic)))
        mic_reference_ratio = mic_rms / max(best_reference_rms, 1.0)
        coherent_echo = (
            best_coherence >= self._minimum_coherence and residual <= residual_limit
        )
        relative_level_echo = (
            best_reference_rms > 0.0
            and mic_reference_ratio <= self._maximum_relative_mic_level
        )
        best_robust_coherence = 0.0
        best_robust_envelope_coherence = 0.0
        best_robust_band_temporal_coherence = 0.0
        best_robust_ratio = math.inf
        best_robust_delay_ms = 0
        robust_echo = False
        if mic_recent is not None:
            for (
                envelope_coherence,
                candidate,
                candidate_rms,
                delay_samples,
            ) in robust_candidates:
                band_temporal_coherence = self._band_temporal_coherence(
                    mic_recent,
                    candidate,
                    self._analysis_rate_hz,
                )
                robust_coherence = min(
                    envelope_coherence / _MINIMUM_ROBUST_ENVELOPE_COHERENCE,
                    band_temporal_coherence
                    / _MINIMUM_ROBUST_BAND_TEMPORAL_COHERENCE,
                )
                robust_ratio = mic_recent_rms / max(candidate_rms, 1.0)
                candidate_is_echo = (
                    envelope_coherence
                    >= _MINIMUM_ROBUST_ENVELOPE_COHERENCE
                    and band_temporal_coherence
                    >= _MINIMUM_ROBUST_BAND_TEMPORAL_COHERENCE
                    and robust_ratio <= _MAXIMUM_ROBUST_RELATIVE_MIC_LEVEL
                )
                if candidate_is_echo and (
                    not robust_echo or robust_coherence > best_robust_coherence
                ):
                    robust_echo = True
                    best_robust_coherence = robust_coherence
                    best_robust_envelope_coherence = envelope_coherence
                    best_robust_band_temporal_coherence = band_temporal_coherence
                    best_robust_ratio = robust_ratio
                    best_robust_delay_ms = round(
                        delay_samples * 1_000 / self._analysis_rate_hz
                    )
        echo_dominated = coherent_echo or relative_level_echo or robust_echo
        if coherent_echo:
            if baseline is None:
                self._echo_residual_baseline = residual
            else:
                bounded = min(residual, baseline * 1.2 + 0.01)
                self._echo_residual_baseline = baseline * 0.9 + bounded * 0.1
        mic_reference_ratio_milli = round(mic_reference_ratio * 1_000)
        _LOGGER.info(
            "playback echo classification echo=%s coherence_milli=%d residual_milli=%d "
            "limit_milli=%d delay_ms=%d mic_rms=%d reference_rms=%d "
            "mic_reference_ratio_milli=%d robust_echo=%s "
            "robust_envelope_milli=%d robust_band_temporal_milli=%d "
            "robust_delay_ms=%d robust_ratio_milli=%d",
            echo_dominated,
            round(best_coherence * 1_000),
            round(residual * 1_000),
            round(residual_limit * 1_000),
            best_delay_ms,
            round(mic_rms),
            round(best_reference_rms),
            mic_reference_ratio_milli,
            robust_echo,
            round(best_robust_envelope_coherence * 1_000),
            round(best_robust_band_temporal_coherence * 1_000),
            best_robust_delay_ms,
            round(best_robust_ratio * 1_000) if math.isfinite(best_robust_ratio) else -1,
        )
        return echo_dominated

    def _purge_retired(self, now: float) -> None:
        while self._retired:
            ended_at = self._retired[0].ended_at
            if ended_at is None or now - ended_at <= self._tail_seconds:
                break
            self._retired.popleft()

    @staticmethod
    def _downsample(samples: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
        usable = len(samples) - (len(samples) % factor)
        if usable == 0:
            return np.empty(0, dtype=np.float64)
        return cast(
            NDArray[np.float64],
            samples[:usable].reshape(-1, factor).mean(axis=1),
        )

    @classmethod
    def _envelope_coherence(
        cls,
        microphone: NDArray[np.float64],
        reference: NDArray[np.float64],
        sample_rate_hz: int,
    ) -> float:
        microphone_envelope = cls._energy_envelope(microphone, sample_rate_hz)
        reference_envelope = cls._energy_envelope(reference, sample_rate_hz)
        return cls._normalized_correlation(microphone_envelope, reference_envelope)

    @classmethod
    def _band_temporal_coherence(
        cls,
        microphone: NDArray[np.float64],
        reference: NDArray[np.float64],
        sample_rate_hz: int,
    ) -> float:
        microphone_features = cls._band_temporal_features(microphone, sample_rate_hz)
        reference_features = cls._band_temporal_features(reference, sample_rate_hz)
        if (
            microphone_features.shape != reference_features.shape
            or len(microphone_features) == 0
        ):
            return 0.0
        band_coherence = np.sum(
            microphone_features * reference_features,
            axis=0,
        )
        return max(0.0, min(1.0, float(np.mean(band_coherence))))

    @classmethod
    def _energy_envelope(
        cls,
        samples: NDArray[np.float64],
        sample_rate_hz: int,
    ) -> NDArray[np.float64]:
        blocks = cls._feature_blocks(samples, sample_rate_hz)
        if len(blocks) == 0:
            return np.empty(0, dtype=np.float64)
        return cast(
            NDArray[np.float64],
            np.sqrt(np.mean(blocks * blocks, axis=1)),
        )

    @classmethod
    def _band_temporal_features(
        cls,
        samples: NDArray[np.float64],
        sample_rate_hz: int,
    ) -> NDArray[np.float64]:
        blocks = cls._feature_blocks(samples, sample_rate_hz)
        if len(blocks) == 0:
            return np.empty((0, 7), dtype=np.float64)
        window = np.hanning(blocks.shape[1])
        spectrum = np.abs(np.fft.rfft(blocks * window, axis=1)) ** 2
        bin_count = spectrum.shape[1]
        edges = (
            0,
            bin_count // 20,
            bin_count // 10,
            bin_count // 5,
            bin_count * 2 // 5,
            bin_count * 3 // 5,
            bin_count * 4 // 5,
            bin_count,
        )
        bands = np.stack(
            [
                spectrum[:, start : max(start + 1, end)].sum(axis=1)
                for start, end in zip(edges[:-1], edges[1:], strict=True)
            ],
            axis=1,
        )
        bands = np.log1p(bands)
        bands -= bands.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(bands, axis=0, keepdims=True)
        np.divide(bands, norms, out=bands, where=norms > 1e-12)
        return cast(NDArray[np.float64], bands)

    @staticmethod
    def _feature_blocks(
        samples: NDArray[np.float64],
        sample_rate_hz: int,
    ) -> NDArray[np.float64]:
        frame_samples = round(_ROBUST_FRAME_SECONDS * sample_rate_hz)
        hop_samples = round(_ROBUST_HOP_SECONDS * sample_rate_hz)
        if len(samples) < frame_samples:
            return np.empty((0, frame_samples), dtype=np.float64)
        return np.stack(
            [
                samples[start : start + frame_samples]
                for start in range(
                    0,
                    len(samples) - frame_samples + 1,
                    hop_samples,
                )
            ]
        )

    @staticmethod
    def _normalized_correlation(
        left: NDArray[np.float64],
        right: NDArray[np.float64],
    ) -> float:
        if left.shape != right.shape or left.size == 0:
            return 0.0
        centered_left = left - float(left.mean())
        centered_right = right - float(right.mean())
        denominator = float(
            np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
        )
        if denominator <= 1e-12:
            return 0.0
        correlation = float(centered_left @ centered_right) / denominator
        return max(0.0, min(1.0, correlation))


    @staticmethod
    def _trusted_frame(frame: AudioFrame) -> AudioFrame:
        if type(frame) is not AudioFrame:
            raise TypeError("echo reference requires exact AudioFrame values")
        if frame.channels != 1:
            raise ValueError("echo reference requires mono PCM")
        if len(frame.pcm) > _MAX_REFERENCE_BYTES:
            raise ValueError("echo reference frame exceeds capacity")
        return AudioFrame(
            pcm=frame.pcm,
            sample_rate_hz=frame.sample_rate_hz,
            channels=frame.channels,
        )

    @staticmethod
    def _normalized_text(text: str) -> str:
        if type(text) is not str:
            raise TypeError("echo reference text must be an exact built-in string")
        if len(text) > _MAX_REFERENCE_TEXT_CHARS:
            raise ValueError("echo reference text exceeds capacity")
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(
            "".join(character if character.isalnum() else " " for character in normalized).split()
        )
