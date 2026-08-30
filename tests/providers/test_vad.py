from __future__ import annotations

from collections import deque

import pytest

from hermes_realtime.providers import WebRtcVoiceActivityDetector
from hermes_realtime.speech import AudioFrame, VoiceActivity


def _frame() -> AudioFrame:
    return AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)


def test_webrtc_vad_emits_stable_start_continue_and_end_boundaries() -> None:
    decisions = deque((True, True, True, False, False))
    calls: list[tuple[bytes, int]] = []

    def classify(pcm: bytes, sample_rate_hz: int) -> bool:
        calls.append((pcm, sample_rate_hz))
        return decisions.popleft()

    vad = WebRtcVoiceActivityDetector(
        sample_rate_hz=48_000,
        frame_duration_ms=10,
        speech_start_frames=2,
        speech_end_frames=2,
        classify=classify,
    )

    assert vad.required_pre_roll_frames == 1

    assert [vad.process(_frame()) for _ in range(5)] == [
        VoiceActivity.SILENCE,
        VoiceActivity.SPEECH_STARTED,
        VoiceActivity.SPEECH_CONTINUED,
        VoiceActivity.SPEECH_CONTINUED,
        VoiceActivity.SPEECH_ENDED,
    ]
    assert calls == [(_frame().pcm, 48_000)] * 5


def test_webrtc_vad_defaults_admit_short_speech_and_tolerate_normal_cadence_pauses() -> None:
    short_noise = deque((True,) * 19 + (False,))
    vad = WebRtcVoiceActivityDetector(classify=lambda _pcm, _rate: short_noise.popleft())

    assert [vad.process(_frame()) for _ in range(20)] == [VoiceActivity.SILENCE] * 20

    decisions = deque((True,) * 20 + (False,) * 30 + (True,) + (False,) * 40)
    vad = WebRtcVoiceActivityDetector(classify=lambda _pcm, _rate: decisions.popleft())
    activities = [vad.process(_frame()) for _ in range(91)]

    assert activities[:19] == [VoiceActivity.SILENCE] * 19
    assert activities[19] is VoiceActivity.SPEECH_STARTED
    assert activities[20:50] == [VoiceActivity.SPEECH_CONTINUED] * 30
    assert activities[50] is VoiceActivity.SPEECH_CONTINUED
    assert activities[51:90] == [VoiceActivity.SPEECH_CONTINUED] * 39
    assert activities[90] is VoiceActivity.SPEECH_ENDED
    assert vad.required_pre_roll_frames == 19


def test_webrtc_vad_defaults_endpoint_after_400ms_of_silence() -> None:
    decisions = deque((True,) * 20 + (False,) * 40)
    vad = WebRtcVoiceActivityDetector(classify=lambda _pcm, _rate: decisions.popleft())

    activities = [vad.process(_frame()) for _ in range(60)]

    assert activities[19] is VoiceActivity.SPEECH_STARTED
    assert activities[20:59] == [VoiceActivity.SPEECH_CONTINUED] * 39
    assert activities[59] is VoiceActivity.SPEECH_ENDED


def test_webrtc_vad_rejects_noncanonical_frame_before_classifier() -> None:
    called = False

    def classify(_pcm: bytes, _sample_rate_hz: int) -> bool:
        nonlocal called
        called = True
        return False

    vad = WebRtcVoiceActivityDetector(classify=classify)

    with pytest.raises(ValueError, match="10 ms"):
        vad.process(AudioFrame(pcm=b"\x00\x00" * 479, sample_rate_hz=48_000, channels=1))

    assert not called
