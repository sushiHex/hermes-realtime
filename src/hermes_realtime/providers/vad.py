"""Stateful WebRTC VAD boundary for exact fixed-duration PCM frames."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from hermes_realtime.speech import AudioFrame, VoiceActivity

_SUPPORTED_SAMPLE_RATES = frozenset((8000, 16_000, 32_000, 48_000))
_SUPPORTED_FRAME_DURATIONS = frozenset((10, 20, 30))


class WebRtcVoiceActivityDetector:
    """Debounce WebRTC speech decisions into authoritative turn boundaries."""

    def __init__(
        self,
        *,
        sample_rate_hz: int = 48_000,
        frame_duration_ms: int = 10,
        mode: int = 2,
        speech_start_frames: int = 20,
        speech_end_frames: int = 40,
        classify: Callable[[bytes, int], bool] | None = None,
    ) -> None:
        if type(sample_rate_hz) is not int:
            raise TypeError("sample_rate_hz must be an exact integer")
        if sample_rate_hz not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate_hz is unsupported by WebRTC VAD")
        if type(frame_duration_ms) is not int:
            raise TypeError("frame_duration_ms must be an exact integer")
        if frame_duration_ms not in _SUPPORTED_FRAME_DURATIONS:
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        if type(mode) is not int:
            raise TypeError("mode must be an exact integer")
        if mode not in range(4):
            raise ValueError("mode must be from 0 through 3")
        for name, value in (
            ("speech_start_frames", speech_start_frames),
            ("speech_end_frames", speech_end_frames),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= 1000:
                raise ValueError(f"{name} must be between 1 and 1000")
        if classify is not None and not callable(classify):
            raise TypeError("classify must be callable")
        if classify is None:
            try:
                webrtcvad = importlib.import_module("webrtcvad")
            except ImportError as error:
                raise RuntimeError(
                    "WebRTC VAD requires the 'webrtcvad-wheels' local extra"
                ) from error
            detector = webrtcvad.Vad(mode)
            classify = cast(Callable[[bytes, int], bool], detector.is_speech)
        assert classify is not None

        self._sample_rate_hz = sample_rate_hz
        self._frame_duration_ms = frame_duration_ms
        self._frame_bytes = sample_rate_hz * frame_duration_ms // 1000 * 2
        self._speech_start_frames = speech_start_frames
        self._speech_end_frames = speech_end_frames
        self._classify = classify
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0

    @property
    def required_pre_roll_frames(self) -> int:
        return self._speech_start_frames - 1

    def process(self, frame: AudioFrame) -> VoiceActivity:
        if type(frame) is not AudioFrame:
            raise TypeError("frame must be an exact AudioFrame")
        frame = AudioFrame(
            pcm=frame.pcm,
            sample_rate_hz=frame.sample_rate_hz,
            channels=frame.channels,
        )
        if frame.sample_rate_hz != self._sample_rate_hz or frame.channels != 1:
            raise ValueError("WebRTC VAD frame format does not match its configuration")
        if len(frame.pcm) != self._frame_bytes:
            raise ValueError(f"WebRTC VAD requires exact {self._frame_duration_ms} ms PCM frames")
        voiced = self._classify(frame.pcm, frame.sample_rate_hz)
        if type(voiced) is not bool:
            raise TypeError("WebRTC VAD classifier must return an exact boolean")

        if self._speaking:
            if voiced:
                self._silence_frames = 0
                return VoiceActivity.SPEECH_CONTINUED
            self._silence_frames += 1
            if self._silence_frames < self._speech_end_frames:
                return VoiceActivity.SPEECH_CONTINUED
            self._speaking = False
            self._speech_frames = 0
            self._silence_frames = 0
            return VoiceActivity.SPEECH_ENDED

        if not voiced:
            self._speech_frames = 0
            return VoiceActivity.SILENCE
        self._speech_frames += 1
        if self._speech_frames < self._speech_start_frames:
            return VoiceActivity.SILENCE
        self._speaking = True
        self._speech_frames = 0
        self._silence_frames = 0
        return VoiceActivity.SPEECH_STARTED

    def reset(self) -> None:
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
