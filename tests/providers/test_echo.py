from typing import Any

import numpy as np
import pytest

from hermes_realtime.providers import PlaybackEchoGuard
from hermes_realtime.providers import echo as echo_module
from hermes_realtime.speech import AudioFrame


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _speech(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, samples + 16)
    shaped = np.convolve(noise, np.hanning(17), mode="valid")
    shaped /= max(abs(shaped))
    return shaped


def _syllabic_speech(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_rate_hz = 48_000
    timeline = np.arange(samples) / sample_rate_hz
    duration = samples / sample_rate_hz
    signal = np.zeros(samples)
    for center in np.linspace(0.04, duration - 0.04, 8):
        width = rng.uniform(0.018, 0.045)
        pulse = rng.uniform(0.4, 1.0) * np.exp(
            -0.5 * ((timeline - center - rng.uniform(-0.01, 0.01)) / width) ** 2
        )
        fundamental = rng.uniform(100.0, 240.0)
        carrier = sum(
            np.sin(
                2 * np.pi * fundamental * harmonic * timeline
                + rng.uniform(0.0, 2 * np.pi)
            )
            / harmonic
            for harmonic in range(1, 13)
        )
        signal += pulse * carrier
    signal /= max(float(np.max(np.abs(signal))), 1e-12)
    return signal


def _frame(values: np.ndarray) -> AudioFrame:
    clipped = np.clip(values * 12_000, -32_768, 32_767).astype("<i2")
    return AudioFrame(pcm=clipped.tobytes(), sample_rate_hz=48_000, channels=1)


def test_playback_echo_guard_suppresses_echo_but_not_double_talk() -> None:
    clock = Clock()
    reference = _speech(96_000, 1)
    user = _speech(19_200, 2)
    guard = PlaybackEchoGuard(clock=clock)
    guard.begin_playback(_frame(reference))
    source = reference[19_200:38_400]
    echo = 0.42 * source
    clock.now = 0.98  # 800 ms source progress plus 180 ms acoustic delay.

    assert guard.is_echo_dominated((_frame(echo),)) is True
    assert guard.is_echo_dominated((_frame(echo + 0.35 * user),)) is False


def test_playback_echo_guard_exposes_active_and_retained_acoustic_authority() -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock, tail_seconds=1.5)

    assert guard.has_recent_playback() is False
    token = guard.begin_playback(_frame(_speech(48_000, 101)))
    assert guard.has_recent_playback() is True

    clock.now = 0.5
    guard.end_playback(token)
    assert guard.has_recent_playback() is True

    clock.now = 1.99
    assert guard.has_recent_playback() is True
    clock.now = 2.01
    assert guard.has_recent_playback() is False


def test_playback_echo_guard_uses_relative_level_for_decorrelated_aec_residual() -> None:
    reference = _speech(96_000, 51)
    residual = _speech(19_200, 52)

    def classify(level: float) -> bool:
        clock = Clock()
        guard = PlaybackEchoGuard(clock=clock)
        guard.begin_playback(_frame(reference))
        clock.now = 0.98
        return guard.is_echo_dominated((_frame(level * residual),))

    assert classify(0.02) is True
    assert classify(0.15) is False


def test_playback_echo_guard_suppresses_filtered_echo_but_promotes_quiet_double_talk() -> None:
    reference = _syllabic_speech(96_000, 71)
    source = reference[30_720:38_400]
    user = _syllabic_speech(len(source), 72)
    rng = np.random.default_rng(1634)
    impulse = np.zeros(108)
    impulse[::6] = rng.normal(size=18) * np.exp(-np.arange(18) / 5.0)
    impulse[0] += 0.6
    distorted = np.convolve(source, impulse, mode="same")
    distorted *= float(np.sqrt(np.mean(source * source))) / max(
        float(np.sqrt(np.mean(distorted * distorted))),
        1e-12,
    )
    residual = 0.15 * distorted

    def classify(microphone: np.ndarray) -> bool:
        clock = Clock()
        guard = PlaybackEchoGuard(clock=clock)
        guard.begin_playback(_frame(reference))
        clock.now = 0.98
        return guard.is_echo_dominated((_frame(microphone),))

    assert classify(residual) is True
    assert classify(0.12 * user) is False
    assert classify(residual + 0.12 * user) is False


def test_playback_echo_guard_accepts_any_fully_qualified_robust_candidate(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reference = 0.2 * _speech(96_000, 81)
    reference[34_560:39_360] = _speech(4_800, 82)
    microphone = 0.1 * _speech(7_680, 83)
    monkeypatch.setattr(
        PlaybackEchoGuard,
        "_envelope_coherence",
        classmethod(lambda _cls, _mic, _ref, _rate: 1.0),
    )
    candidate_levels: list[float] = []

    def band_temporal(
        _cls: type[PlaybackEchoGuard],
        _microphone: np.ndarray,
        candidate: np.ndarray,
        _sample_rate_hz: int,
    ) -> float:
        candidate_levels.append(float(np.sqrt(np.mean(candidate * candidate))))
        return 0.0 if len(candidate_levels) == 1 else 1.0

    monkeypatch.setattr(
        PlaybackEchoGuard,
        "_band_temporal_coherence",
        classmethod(band_temporal),
    )
    caplog.set_level("INFO", logger="hermes_realtime.providers.echo")
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock)
    guard.begin_playback(_frame(reference))
    clock.now = 0.98

    assert guard.is_echo_dominated((_frame(microphone),)) is True
    assert len(candidate_levels) >= 2
    assert "robust_echo=True" in caplog.text


def test_playback_echo_guard_checks_long_buffers_on_recent_validation_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_lengths: list[int] = []

    def envelope_coherence(
        _cls: type[PlaybackEchoGuard],
        left: np.ndarray,
        right: np.ndarray,
        _sample_rate_hz: int,
    ) -> float:
        assert left.shape == right.shape
        observed_lengths.append(len(left))
        return 0.0

    monkeypatch.setattr(
        PlaybackEchoGuard,
        "_envelope_coherence",
        classmethod(envelope_coherence),
    )
    monkeypatch.setattr(
        PlaybackEchoGuard,
        "_band_temporal_coherence",
        classmethod(lambda _cls, _mic, _ref, _rate: 0.0),
    )
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock)
    guard.begin_playback(_frame(_speech(144_000, 84)))
    clock.now = 2.0

    guard.is_echo_dominated((_frame(_speech(48_000, 85)),))

    assert observed_lengths
    assert set(observed_lengths) == {1_920}


def test_playback_echo_guard_fails_closed_when_robust_feature_work_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _speech(96_000, 86)
    microphone = 0.15 * reference[30_720:38_400]
    monkeypatch.setattr(
        PlaybackEchoGuard,
        "_envelope_coherence",
        classmethod(lambda _cls, _mic, _ref, _rate: 1.0),
    )
    monkeypatch.setattr(echo_module, "_MAX_ROBUST_FEATURE_WORK", 1)
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock)
    guard.begin_playback(_frame(reference))
    clock.now = 0.98

    with pytest.raises(ValueError, match="robust feature work exceeds capacity"):
        guard.is_echo_dominated((_frame(microphone),))


def test_playback_echo_guard_matches_active_and_recent_outbound_text() -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock, tail_seconds=1.0)
    token = guard.begin_playback(
        _frame(_speech(48_000, 61)),
        "Yes, I can switch models when that option is available.",
    )

    assert guard.is_transcript_echo("yes i can switch models") is True
    assert guard.is_transcript_echo("stop") is False
    assert guard.is_transcript_echo("tell me something different") is False

    clock.now = 1.0
    guard.end_playback(token)
    clock.now = 1.8
    assert guard.is_transcript_echo("switch models when that option is available") is True
    clock.now = 2.01
    assert guard.is_transcript_echo("switch models when that option is available") is False

    short_guard = PlaybackEchoGuard(clock=clock)
    short_guard.begin_playback(_frame(_speech(48_000, 62)), "Yes.")
    assert short_guard.is_transcript_echo("yes") is False
    assert short_guard.is_transcript_echo("Yes, but please stop now.") is False
    quotation_guard = PlaybackEchoGuard(clock=clock)
    quotation_guard.begin_playback(
        _frame(_speech(48_000, 63)),
        "The remote model is unavailable.",
    )
    assert quotation_guard.is_transcript_echo("the remote model is unavailable") is True
    assert (
        quotation_guard.is_transcript_echo(
            "The remote model is unavailable, but stop and listen to my correction."
        )
        is False
    )


def test_playback_echo_guard_matches_short_multiword_outbound_text() -> None:
    guard = PlaybackEchoGuard()
    guard.begin_playback(
        _frame(_speech(48_000, 64)),
        "One, two, three, four, five, six. Say go back to restart.",
    )

    assert guard.is_transcript_echo("Five, six.") is True
    assert guard.is_transcript_echo("Go back") is False
    assert guard.is_transcript_echo("Restart") is False


def test_playback_echo_guard_retains_bounded_acoustic_tail() -> None:
    clock = Clock()
    reference = _speech(96_000, 3)
    guard = PlaybackEchoGuard(clock=clock, tail_seconds=1.0)
    token = guard.begin_playback(_frame(reference))
    source = reference[19_200:38_400]
    clock.now = 0.98
    assert guard.is_echo_dominated((_frame(0.4 * source),)) is True

    clock.now = 1.0
    guard.end_playback(token)
    clock.now = 1.4
    assert guard.is_echo_dominated((_frame(0.4 * source),)) is True
    clock.now = 2.01
    assert guard.is_echo_dominated((_frame(0.4 * source),)) is False


def test_retired_echo_reference_never_exposes_unplayed_audio() -> None:
    clock = Clock()
    reference = _speech(96_000, 4)
    guard = PlaybackEchoGuard(clock=clock, tail_seconds=1.0)
    token = guard.begin_playback(_frame(reference))
    clock.now = 0.5
    guard.end_playback(token)

    # This segment begins after playback was retired and must never become eligible
    # merely because wall time continues advancing.
    unplayed = reference[33_600:43_200]
    clock.now = 1.08

    assert guard.is_echo_dominated((_frame(0.4 * unplayed),)) is False


def test_playback_echo_guard_retains_every_reference_inside_tail() -> None:
    clock = Clock()
    references = [_speech(4_800, seed) for seed in (11, 12, 13)]
    guard = PlaybackEchoGuard(clock=clock, tail_seconds=1.5, max_delay_seconds=1.5)
    for index, reference in enumerate(references):
        clock.now = index * 0.1
        token = guard.begin_playback(_frame(reference))
        clock.now = (index + 1) * 0.1
        guard.end_playback(token)

    clock.now = 0.4
    assert guard.is_echo_dominated((_frame(0.4 * references[0][2_400:]),)) is True


def test_playback_echo_guard_fails_closed_when_retention_capacity_is_exhausted() -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(
        clock=clock,
        tail_seconds=1.5,
        max_delay_seconds=1.5,
        max_retired_references=2,
    )
    for seed in (21, 22):
        token = guard.begin_playback(_frame(_speech(4_800, seed)))
        clock.now += 0.1
        guard.end_playback(token)

    with pytest.raises(RuntimeError, match="retention capacity"):
        guard.begin_playback(_frame(_speech(4_800, 23)))


def test_playback_echo_guard_requires_delay_horizon_to_cover_tail() -> None:
    with pytest.raises(ValueError, match="max_delay_seconds must cover tail_seconds"):
        PlaybackEchoGuard(tail_seconds=1.5, max_delay_seconds=1.2)


def test_playback_echo_guard_rejects_oversized_aggregate_analysis_before_allocation() -> None:
    guard = PlaybackEchoGuard(max_analysis_bytes=1_000)
    frame = _frame(_speech(600, 31))

    with pytest.raises(ValueError, match="analysis window exceeds capacity"):
        guard.is_echo_dominated((frame, frame))


def test_playback_echo_guard_slices_large_reference_before_numpy_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    reference_frame = _frame(_speech(2_000_000, 41))
    microphone_frame = _frame(_speech(4_800, 42))
    guard = PlaybackEchoGuard(clock=clock)
    guard.begin_playback(reference_frame)
    clock.now = 20.0
    converted_bytes: list[int] = []
    original_frombuffer = np.frombuffer

    def tracked_frombuffer(buffer: bytes | memoryview, *args: Any, **kwargs: Any) -> np.ndarray:
        converted_bytes.append(memoryview(buffer).nbytes)
        return original_frombuffer(buffer, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", tracked_frombuffer)

    guard.is_echo_dominated((microphone_frame,))

    assert converted_bytes[0] == len(microphone_frame.pcm)
    assert max(converted_bytes[1:]) <= len(microphone_frame.pcm) + 144_000


def test_playback_echo_guard_rejects_aggregate_reference_work_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock, max_reference_analysis_bytes=15_000)
    for seed in (43, 44):
        token = guard.begin_playback(_frame(_speech(4_800, seed)))
        clock.now += 0.1
        guard.end_playback(token)
    clock.now = 0.3
    microphone_frame = _frame(_speech(4_800, 45))
    original_frombuffer = np.frombuffer
    converted_bytes: list[int] = []

    def tracked_frombuffer(buffer: bytes | memoryview, *args: Any, **kwargs: Any) -> np.ndarray:
        converted_bytes.append(memoryview(buffer).nbytes)
        return original_frombuffer(buffer, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", tracked_frombuffer)

    with pytest.raises(ValueError, match="reference analysis exceeds capacity"):
        guard.is_echo_dominated((microphone_frame,))

    assert converted_bytes == [len(microphone_frame.pcm)]


def test_playback_echo_guard_rejects_excessive_correlation_work_before_reference_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(
        clock=clock,
        analysis_rate_hz=24_000,
        delay_step_ms=2,
        max_delay_seconds=3.0,
        max_correlation_work=16_000_000,
    )
    guard.begin_playback(_frame(_speech(600_000, 46)))
    clock.now = 10.0
    microphone_frame = _frame(_speech(100_000, 47))
    original_frombuffer = np.frombuffer
    converted_bytes: list[int] = []

    def tracked_frombuffer(buffer: bytes | memoryview, *args: Any, **kwargs: Any) -> np.ndarray:
        converted_bytes.append(memoryview(buffer).nbytes)
        return original_frombuffer(buffer, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", tracked_frombuffer)

    with pytest.raises(ValueError, match="correlation work exceeds capacity"):
        guard.is_echo_dominated((microphone_frame,))

    assert converted_bytes == [len(microphone_frame.pcm)]
