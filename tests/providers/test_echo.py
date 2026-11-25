import json
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


def test_playback_echo_guard_reports_a_word_prefix_in_recent_outbound_text() -> None:
    # #576: the transcript-echo match cannot see a four-character word; this query can.
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock, tail_seconds=1.0)
    assert guard.has_recent_outbound_word_prefix("stop") is False
    token = guard.begin_playback(_frame(_speech(48_000, 65)), "Say STOP, and I'll stop.")

    assert guard.has_recent_outbound_word_prefix("stop") is True
    clock.now = 1.0
    guard.end_playback(token)
    clock.now = 1.9
    assert guard.has_recent_outbound_word_prefix("stop") is True
    clock.now = 2.01
    assert guard.has_recent_outbound_word_prefix("stop") is False

    for outbound in ("Unstoppable.", "Nonstop.", "st op", "Halt."):
        word_guard = PlaybackEchoGuard(clock=clock)
        word_guard.begin_playback(_frame(_speech(48_000, 66)), outbound)
        assert word_guard.has_recent_outbound_word_prefix("stop") is False, outbound
    for outbound in (
        "Don't stop.", "stop", "Stop-start.", "It stopped.", "Stops here.", "Keep stopping.",
    ):
        word_guard = PlaybackEchoGuard(clock=clock)
        word_guard.begin_playback(_frame(_speech(48_000, 67)), outbound)
        assert word_guard.has_recent_outbound_word_prefix("stop") is True, outbound

    for prefix in ("", "two words", "Stop"):
        with pytest.raises(ValueError, match="one normalized word"):
            guard.has_recent_outbound_word_prefix(prefix)
    with pytest.raises(TypeError):
        guard.has_recent_outbound_word_prefix(b"stop")  # type: ignore[arg-type]


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


_TAIL_ACOUSTIC_DELAY_SECONDS = 0.06
_TAIL_LEAK_SECONDS = 0.4
_TAIL_PLAYBACK_SECONDS = 2.0


def _tail_capture(
    kind: str,
    source: np.ndarray,
    level: float,
    endpoint_after_playback: float,
    *,
    ended_at: float = _TAIL_PLAYBACK_SECONDS,
    leak_seconds: float = _TAIL_LEAK_SECONDS,
) -> np.ndarray:
    """A capture ending ``endpoint_after_playback`` after playback ended.

    ``leak``/``reverb`` hold the end of ``source`` after the acoustic delay; ``unrelated``
    holds the end of an unrelated signal at the same place; ``user`` holds unrelated speech
    that starts 100 ms after playback ended.
    """

    noise = np.random.default_rng(5)
    if kind == "user":
        length = round((endpoint_after_playback - 0.1) * 48_000)
        capture = noise.normal(0.0, 0.002, length)
        speech = level * source[: round(0.4 * 48_000)]
        capture[: len(speech)] += speech
        return capture
    start = ended_at - leak_seconds + _TAIL_ACOUSTIC_DELAY_SECONDS
    length = round((ended_at + endpoint_after_playback - start) * 48_000)
    capture = noise.normal(0.0, 0.002, length)
    leak = source[-round(leak_seconds * 48_000) :]
    if kind == "reverb":
        rng = np.random.default_rng(6)
        response = rng.normal(0.0, 1.0, 5_760) * np.exp(-np.arange(5_760) / 1_200.0)
        response[0] = 3.0
        wet = np.convolve(leak, response)[: len(leak)]
        leak = wet * float(np.sqrt(np.mean(leak * leak)) / np.sqrt(np.mean(wet * wet)))
    capture[: len(leak)] += level * leak
    return capture


def _classify_tail(
    kind: str,
    level: float,
    endpoint_after_playback: float,
    *,
    guard_type: Any = PlaybackEchoGuard,
    leak_seconds: float = _TAIL_LEAK_SECONDS,
) -> bool:
    clock = Clock()
    guard = guard_type(clock=clock)
    reference = _syllabic_speech(round(_TAIL_PLAYBACK_SECONDS * 48_000), 301)
    token = guard.begin_playback(_frame(reference))
    clock.now = _TAIL_PLAYBACK_SECONDS
    guard.end_playback(token)
    source = reference if kind in ("leak", "reverb") else _syllabic_speech(96_000, 302)
    capture = _tail_capture(
        kind, source, level, endpoint_after_playback, leak_seconds=leak_seconds
    )
    clock.now = _TAIL_PLAYBACK_SECONDS + endpoint_after_playback
    return bool(guard.is_echo_dominated((_frame(capture),)))


def _tail_grid() -> list[tuple[str, float, float]]:
    cells: list[tuple[str, float, float]] = []
    for level in (0.03, 0.05, 0.3, 0.8):
        cells += [("leak", level, after) for after in (0.06, 0.3, 0.6, 0.9, 1.2)]
    for level in (0.05, 0.3):
        cells += [("reverb", level, after) for after in (0.06, 0.3, 0.6, 1.2)]
    for level in (0.05, 0.1, 0.2):
        cells += [("unrelated", level, after) for after in (0.06, 0.3, 0.6, 1.2)]
    for level in (0.05, 0.2, 0.5, 1.0):
        cells += [("user", level, after) for after in (0.6, 0.9, 1.2)]
    return cells


def _tail_key(cell: tuple[str, float, float]) -> str:
    return f"{cell[0]}/{cell[1]}/{cell[2]}"


# Cells the guard flagged before the after-playback scan existed (generated once from the
# pre-change implementation); the scan may only add to them.
_MAIN_FLAGGED: frozenset[str] = frozenset(
    {
        "leak/0.03/0.06",
        "leak/0.03/0.3",
        "leak/0.03/0.6",
        "leak/0.03/0.9",
        "leak/0.03/1.2",
        "leak/0.05/0.06",
        "leak/0.05/0.3",
        "leak/0.05/0.6",
        "leak/0.05/0.9",
        "leak/0.05/1.2",
        "leak/0.3/0.06",
        "leak/0.8/0.06",
        "reverb/0.05/0.3",
        "reverb/0.05/0.6",
        "reverb/0.05/1.2",
        "unrelated/0.05/0.06",
        "unrelated/0.05/0.3",
        "unrelated/0.05/0.6",
        "unrelated/0.05/1.2",
        "unrelated/0.1/0.6",
        "unrelated/0.1/1.2",
        "user/0.05/1.2",
    }
)


@pytest.mark.parametrize("endpoint_after_playback", [0.3, 0.6])
@pytest.mark.parametrize("level", [0.3, 0.8])
def test_playback_echo_guard_catches_leaked_last_word_when_endpoint_follows_playback(
    level: float,
    endpoint_after_playback: float,
) -> None:
    assert _classify_tail("leak", level, endpoint_after_playback) is True


@pytest.mark.parametrize("level", [0.3, 0.8])
def test_playback_echo_guard_catches_leaked_last_word_when_endpoint_is_aligned(
    level: float,
) -> None:
    assert _classify_tail("leak", level, _TAIL_ACOUSTIC_DELAY_SECONDS) is True


@pytest.mark.parametrize("endpoint_after_playback", [0.06, 0.3, 0.6])
@pytest.mark.parametrize("level", [0.3, 0.8])
def test_playback_echo_guard_does_not_flag_unrelated_capture_after_playback(
    level: float,
    endpoint_after_playback: float,
) -> None:
    assert _classify_tail("unrelated", level, endpoint_after_playback) is False


def test_after_playback_scan_only_adds_to_the_verdicts_of_the_pre_change_guard() -> None:
    grid = _tail_grid()
    for cell in grid:
        verdict = _classify_tail(*cell)
        if _tail_key(cell) in _MAIN_FLAGGED:
            assert verdict is True, _tail_key(cell)
        elif cell[0] in ("unrelated", "user"):
            assert verdict is False, _tail_key(cell)


def test_after_playback_scan_writes_no_baseline() -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock)
    reference = _syllabic_speech(96_000, 301)
    token = guard.begin_playback(_frame(reference))
    clock.now = _TAIL_PLAYBACK_SECONDS
    guard.end_playback(token)

    clock.now = _TAIL_PLAYBACK_SECONDS + 0.06
    aligned = _tail_capture("leak", reference, 0.8, 0.06)
    assert guard.is_echo_dominated((_frame(aligned),)) is True
    baseline = guard._echo_residual_baseline
    assert baseline is not None

    clock.now = _TAIL_PLAYBACK_SECONDS + 0.6
    late = _tail_capture("leak", reference, 0.8, 0.6)
    assert guard.is_echo_dominated((_frame(late),)) is True
    assert guard._echo_residual_baseline == baseline


def test_after_playback_scan_stops_at_its_own_budget_without_raising() -> None:
    def classify(budget: int) -> bool:
        clock = Clock()
        guard = PlaybackEchoGuard(clock=clock, max_correlation_work=budget)
        reference = _syllabic_speech(48_000, 311)
        token = guard.begin_playback(_frame(reference))
        clock.now = 1.0
        guard.end_playback(token)
        capture = _tail_capture("leak", reference, 0.8, 0.75, ended_at=1.0)
        clock.now = 1.75
        return bool(guard.is_echo_dominated((_frame(capture),)))

    assert classify(16_000_000) is True
    assert classify(200_000) is False


def test_after_playback_scan_never_raises_over_many_short_references() -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock)
    for seed in range(200):
        token = guard.begin_playback(_frame(_speech(480, seed)))
        clock.now += 0.01
        guard.end_playback(token)
    clock.now += 0.75
    capture = _speech(36_000, 7) * 0.5

    assert guard.is_echo_dominated((_frame(capture),)) is False


def test_after_playback_scan_ignores_candidates_with_too_little_played_audio() -> None:
    assert _classify_tail("leak", 0.8, 0.6, leak_seconds=0.2) is True
    assert _classify_tail("leak", 0.8, 0.6, leak_seconds=0.1) is False


def _tail_scan_lines(captured: str) -> list[dict[str, Any]]:
    prefix = "[echo-tail-scan] "
    return [
        json.loads(line[len(prefix) :]) for line in captured.splitlines() if line.startswith(prefix)
    ]


def _exhausted_scan(guard: PlaybackEchoGuard, clock: Clock, *, begin: bool) -> bool:
    reference = _syllabic_speech(48_000, 311)
    if begin:
        clock.now = 0.0
        token = guard.begin_playback(_frame(reference))
        clock.now = 1.0
        guard.end_playback(token)
    capture = _tail_capture("leak", reference, 0.8, 0.75, ended_at=1.0)
    clock.now = 1.75
    return bool(guard.is_echo_dominated((_frame(capture),)))


def test_exhausted_after_playback_scan_prints_one_bounded_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock, max_correlation_work=200_000)

    assert _exhausted_scan(guard, clock, begin=True) is False

    lines = _tail_scan_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert set(lines[0]) == {"category", "work", "reference_bytes", "references", "suppressed"}
    assert lines[0]["category"] == "work_budget"
    assert lines[0]["references"] == 1
    assert lines[0]["suppressed"] == 0
    assert all(type(lines[0][key]) is int for key in ("work", "reference_bytes", "suppressed"))
    assert lines[0]["work"] > 200_000


def test_completed_after_playback_scan_prints_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock)

    assert _exhausted_scan(guard, clock, begin=True) is True
    assert _classify_tail("unrelated", 0.8, 0.6) is False

    assert _tail_scan_lines(capsys.readouterr().out) == []


def test_exhausted_after_playback_scan_prints_once_per_playback_and_counts_the_rest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    guard = PlaybackEchoGuard(clock=clock, max_correlation_work=200_000)

    assert _exhausted_scan(guard, clock, begin=True) is False
    assert _exhausted_scan(guard, clock, begin=False) is False
    assert len(_tail_scan_lines(capsys.readouterr().out)) == 1

    guard.begin_playback(_frame(_speech(4_800, 5)))
    assert _exhausted_scan(guard, clock, begin=False) is False

    lines = _tail_scan_lines(capsys.readouterr().out)
    assert [line["suppressed"] for line in lines] == [1]
