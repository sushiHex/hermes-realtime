"""Independent bounded speech-presence verification with pinned Silero VAD."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
from pathlib import Path
from typing import Any, Protocol

from hermes_realtime.speech import AudioFrame, SpeechPresence

_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
_ANALYSIS_RATE_HZ = 16_000
_CHUNK_SAMPLES = 512
_CONTEXT_SAMPLES = 64
_DEFAULT_ANALYSIS_WINDOW_MS = 800
_DEFAULT_EVIDENCE_CHUNKS = 2

_LOGGER = logging.getLogger(__name__)


class _InferenceSession(Protocol):
    def run(self, outputs: object, inputs: dict[str, Any]) -> list[Any]: ...


class SileroSpeechPresenceVerifier:
    """Classify candidate PCM without transcript text or retained model state."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        speech_threshold: float = 0.50,
        non_speech_threshold: float = 0.10,
        analysis_window_ms: int = _DEFAULT_ANALYSIS_WINDOW_MS,
        evidence_chunks: int = _DEFAULT_EVIDENCE_CHUNKS,
        session: _InferenceSession | None = None,
    ) -> None:
        for name, value in (
            ("speech_threshold", speech_threshold),
            ("non_speech_threshold", non_speech_threshold),
        ):
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be an exact number")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if non_speech_threshold >= speech_threshold:
            raise ValueError("non_speech_threshold must be below speech_threshold")
        if type(analysis_window_ms) is not int:
            raise TypeError("analysis_window_ms must be an exact int")
        if not 320 <= analysis_window_ms <= 2_000:
            raise ValueError("analysis_window_ms must be between 320 and 2000")
        if type(evidence_chunks) is not int:
            raise TypeError("evidence_chunks must be an exact int")
        if not 2 <= evidence_chunks <= 8:
            raise ValueError("evidence_chunks must be between 2 and 8")
        if session is not None and not callable(getattr(session, "run", None)):
            raise TypeError("session must provide run()")

        if session is None:
            selected_path = (
                Path(__file__).with_name("models") / "silero_vad.onnx"
                if model_path is None
                else Path(model_path)
            )
            if not selected_path.is_file():
                raise FileNotFoundError(f"Silero VAD model is missing: {selected_path}")
            if model_path is None:
                digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
                if digest != _MODEL_SHA256:
                    raise RuntimeError("packaged Silero VAD model checksum mismatch")
            onnxruntime = importlib.import_module("onnxruntime")
            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(selected_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

        assert session is not None
        self._session = session
        self._speech_threshold = float(speech_threshold)
        self._non_speech_threshold = float(non_speech_threshold)
        self._analysis_window_ms = analysis_window_ms
        self._evidence_chunks = evidence_chunks

    def classify(self, frames: tuple[AudioFrame, ...]) -> SpeechPresence:
        if type(frames) is not tuple:
            raise TypeError("frames must be an exact tuple")
        if not frames:
            raise ValueError("frames must not be empty")

        first = frames[0]
        if type(first) is not AudioFrame:
            raise TypeError("frames must contain exact AudioFrame values")
        sample_rate_hz = first.sample_rate_hz
        if first.channels != 1:
            raise ValueError("Silero speech verification requires mono PCM")
        if sample_rate_hz < _ANALYSIS_RATE_HZ or sample_rate_hz % _ANALYSIS_RATE_HZ:
            raise ValueError("Silero speech verification requires a 16 kHz-multiple sample rate")
        for frame in frames:
            if type(frame) is not AudioFrame:
                raise TypeError("frames must contain exact AudioFrame values")
            if frame.sample_rate_hz != sample_rate_hz:
                raise ValueError("speech candidate sample rate changed")
            if frame.channels != 1:
                raise ValueError("Silero speech verification requires mono PCM")

        numpy = importlib.import_module("numpy")
        pcm = numpy.concatenate(
            [numpy.frombuffer(frame.pcm, dtype="<i2") for frame in frames]
        ).astype(numpy.float32)
        analysis_samples = sample_rate_hz * self._analysis_window_ms // 1_000
        pcm = pcm[-analysis_samples:]
        factor = sample_rate_hz // _ANALYSIS_RATE_HZ
        usable_samples = len(pcm) - len(pcm) % factor
        audio = pcm[:usable_samples].reshape(-1, factor).mean(axis=1) / 32768.0
        state = numpy.zeros((2, 1, 128), dtype=numpy.float32)
        context = numpy.zeros((1, _CONTEXT_SAMPLES), dtype=numpy.float32)
        sample_rate = numpy.array(_ANALYSIS_RATE_HZ, dtype=numpy.int64)
        probabilities: list[float] = []
        chunks = 0

        for offset in range(0, len(audio), _CHUNK_SAMPLES):
            chunk = audio[offset : offset + _CHUNK_SAMPLES]
            if len(chunk) < _CHUNK_SAMPLES:
                chunk = numpy.pad(chunk, (0, _CHUNK_SAMPLES - len(chunk)))
            model_input = numpy.concatenate((context, chunk.reshape(1, -1)), axis=1)
            outputs = self._session.run(
                None,
                {"input": model_input, "state": state, "sr": sample_rate},
            )
            if type(outputs) is not list or len(outputs) != 2:
                raise RuntimeError("Silero VAD returned an invalid output set")
            probability = float(outputs[0][0][0])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise RuntimeError("Silero VAD returned an invalid speech probability")
            state = outputs[1]
            context = model_input[:, -_CONTEXT_SAMPLES:]
            probabilities.append(probability)
            chunks += 1

        if len(probabilities) < self._evidence_chunks:
            evidence_probability = max(probabilities)
            if evidence_probability <= self._non_speech_threshold:
                presence = SpeechPresence.CONFIRMED_NON_SPEECH
            else:
                presence = SpeechPresence.UNCERTAIN
        else:
            evidence_probability = max(
                min(probabilities[offset : offset + self._evidence_chunks])
                for offset in range(len(probabilities) - self._evidence_chunks + 1)
            )
            if evidence_probability >= self._speech_threshold:
                presence = SpeechPresence.CONFIRMED_SPEECH
            elif evidence_probability <= self._non_speech_threshold:
                presence = SpeechPresence.CONFIRMED_NON_SPEECH
            else:
                presence = SpeechPresence.UNCERTAIN
        _LOGGER.debug(
            "speech presence classification presence=%s sustained_probability_milli=%d "
            "chunks=%d duration_ms=%d",
            presence.value,
            round(evidence_probability * 1_000),
            chunks,
            round(len(pcm) * 1_000 / sample_rate_hz),
        )
        return presence
