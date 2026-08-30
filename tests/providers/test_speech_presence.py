from __future__ import annotations

import importlib
import math
import os
import random
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.providers.speech_presence import SileroSpeechPresenceVerifier
from hermes_realtime.providers.vad import WebRtcVoiceActivityDetector
from hermes_realtime.speech import AudioFrame, SpeechPresence, VoiceActivity


def _optional_module(name: str):
    if os.environ.get("HERMES_RELEASE_SPEECH_VERIFICATION") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


_optional_module("numpy")


class ScriptedSession:
    def __init__(self, *scores: float) -> None:
        self._scores = iter(scores)
        self.calls: list[dict[str, Any]] = []

    def run(self, _outputs: object, inputs: dict[str, Any]) -> list[Any]:
        import numpy as np

        self.calls.append(inputs)
        score = next(self._scores)
        return [
            np.array([[score]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]


def _frame(samples: array[int] | None = None) -> AudioFrame:
    selected = samples if samples is not None else array("h", [0] * 480)
    return AudioFrame(pcm=selected.tobytes(), sample_rate_hz=48_000, channels=1)


@pytest.mark.parametrize(
    ("score", "expected"),
    (
        (0.09, SpeechPresence.CONFIRMED_NON_SPEECH),
        (0.40, SpeechPresence.UNCERTAIN),
        (0.80, SpeechPresence.CONFIRMED_SPEECH),
    ),
)
def test_silero_verifier_projects_conservative_tri_state(
    score: float,
    expected: SpeechPresence,
) -> None:
    session = ScriptedSession(score, score, score, score)
    verifier = SileroSpeechPresenceVerifier(session=session)

    assert verifier.classify((_frame(),) * 10) is expected
    assert len(session.calls) == 4
    assert session.calls[0]["input"].shape == (1, 576)
    assert session.calls[0]["state"].shape == (2, 1, 128)
    assert session.calls[0]["sr"].item() == 16_000


def test_silero_verifier_requires_temporally_sustained_candidate_probability() -> None:
    session = ScriptedSession(0.12, 0.74, 0.80, 0.82)
    verifier = SileroSpeechPresenceVerifier(session=session)

    assert verifier.classify((_frame(),) * 10) is SpeechPresence.CONFIRMED_SPEECH
    assert len(session.calls) == 4


def test_silero_verifier_rejects_one_isolated_probability_spike() -> None:
    session = ScriptedSession(0.01, 0.99, 0.01, 0.01)
    verifier = SileroSpeechPresenceVerifier(session=session)

    assert verifier.classify((_frame(),) * 10) is SpeechPresence.CONFIRMED_NON_SPEECH


def test_silero_verifier_bounds_inference_to_recent_window() -> None:
    session = ScriptedSession(*([0.09] * 25))
    verifier = SileroSpeechPresenceVerifier(session=session)

    assert verifier.classify((_frame(),) * 200) is SpeechPresence.CONFIRMED_NON_SPEECH
    assert len(session.calls) == 25


def test_silero_verifier_rejects_incompatible_audio_before_inference() -> None:
    session = ScriptedSession(0.9)
    verifier = SileroSpeechPresenceVerifier(session=session)

    with pytest.raises(ValueError, match="mono"):
        verifier.classify(
            (AudioFrame(pcm=b"\x00\x00" * 960, sample_rate_hz=48_000, channels=2),)
        )
    with pytest.raises(ValueError, match="sample rate"):
        verifier.classify(
            (AudioFrame(pcm=b"\x00\x00" * 441, sample_rate_hz=44_100, channels=1),)
        )

    assert session.calls == []


def test_official_silero_model_rejects_damped_contact_resonance() -> None:
    _optional_module("onnxruntime")
    model = (
        Path(__file__).parents[2]
        / "src"
        / "hermes_realtime"
        / "providers"
        / "models"
        / "silero_vad.onnx"
    )
    verifier = SileroSpeechPresenceVerifier(model_path=model)
    sample_rate_hz = 48_000
    sample_count = round(0.6 * sample_rate_hz)
    rng = random.Random(42)
    samples = array("h")
    for index in range(sample_count):
        timestamp = index / sample_rate_hz
        value = 8_000 * math.exp(-timestamp / 2) * (
            0.7 * math.sin(2 * math.pi * 180 * timestamp)
            + 0.3 * math.sin(2 * math.pi * 1_200 * timestamp)
        )
        if timestamp < 0.03:
            value += 8_000 * 0.3 * rng.uniform(-1, 1)
        samples.append(max(-32_768, min(32_767, round(value))))
    frames = tuple(
        _frame(array("h", samples[offset : offset + 480]))
        for offset in range(0, len(samples), 480)
    )

    assert verifier.classify(frames) is SpeechPresence.CONFIRMED_NON_SPEECH


@pytest.mark.parametrize(
    ("level", "expected"),
    (
        (0.05, SpeechPresence.CONFIRMED_SPEECH),
        (0.03, SpeechPresence.UNCERTAIN),
        (0.01, SpeechPresence.UNCERTAIN),
    ),
)
def test_official_silero_model_preserves_level_swept_short_speech(
    level: float,
    expected: SpeechPresence,
) -> None:
    np = _optional_module("numpy")
    _optional_module("onnxruntime")
    fixture = Path(__file__).parents[1] / "fixtures" / "audio" / "sapi_stop.wav"
    with wave.open(str(fixture), "rb") as source:
        source_rate = source.getframerate()
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    active = np.flatnonzero(np.abs(pcm) > 32)
    start = max(0, int(active[0]) - round(0.2 * source_rate))
    stop = min(len(pcm), int(active[-1]) + round(0.4 * source_rate))
    selected = pcm[start:stop].astype(np.float32)
    source_axis = np.arange(len(selected)) / source_rate
    target_axis = np.arange(round(len(selected) / source_rate * 48_000)) / 48_000
    resampled = np.clip(
        np.interp(target_axis, source_axis, selected) * level,
        -32_768,
        32_767,
    ).astype("<i2")
    frames = tuple(
        AudioFrame(
            pcm=resampled[offset : offset + 480].tobytes(),
            sample_rate_hz=48_000,
            channels=1,
        )
        for offset in range(0, len(resampled) - 479, 480)
    )

    vad = WebRtcVoiceActivityDetector()
    activities = [vad.process(frame) for frame in frames]

    assert VoiceActivity.SPEECH_STARTED in activities
    assert SileroSpeechPresenceVerifier().classify(frames) is expected
