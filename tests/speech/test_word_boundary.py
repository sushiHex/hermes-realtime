import math
import struct

import pytest

from hermes_realtime.speech import AudioFrame
from hermes_realtime.speech.word_boundary import (
    FADE_OUT_SECONDS,
    MAX_WORD_TAIL_SECONDS,
    fade_out,
    word_boundary_stop,
)

RATE = 48_000


def _samples(ms: float) -> int:
    return round(RATE * ms / 1000)


def _word(ms: float, amplitude: int = 12_000) -> list[int]:
    return [
        round(amplitude * math.sin(2 * math.pi * 220 * index / RATE))
        for index in range(_samples(ms))
    ]


def _gap(ms: float) -> list[int]:
    return [0] * _samples(ms)


def _audio(samples: list[int], channels: int = 1) -> AudioFrame:
    interleaved = [sample for sample in samples for _ in range(channels)]
    return AudioFrame(
        pcm=struct.pack(f"<{len(interleaved)}h", *interleaved),
        sample_rate_hz=RATE,
        channels=channels,
    )


def _pcm(frame: AudioFrame) -> list[int]:
    return list(struct.unpack(f"<{len(frame.pcm) // 2}h", frame.pcm))


def test_stop_lands_in_the_gap_after_the_interrupted_word() -> None:
    audio = _audio(_word(150) + _gap(60) + _word(400))

    stop = word_boundary_stop(audio, _samples(20))

    assert _samples(150) <= stop < _samples(150) + _samples(5)


def test_stop_without_a_gap_lands_at_the_cap() -> None:
    audio = _audio(_word(1_000))

    stop = word_boundary_stop(audio, _samples(100))

    assert stop == _samples(100) + round(MAX_WORD_TAIL_SECONDS * RATE)


def test_gap_threshold_follows_the_chunk_level_not_absolute_volume() -> None:
    loud = _audio(_word(150, 12_000) + _gap(60) + _word(400, 12_000))
    quiet = _audio(_word(150, 150) + _gap(60) + _word(400, 150))

    assert word_boundary_stop(quiet, _samples(20)) == word_boundary_stop(loud, _samples(20))


def test_a_dip_shorter_than_a_word_gap_is_not_a_boundary() -> None:
    audio = _audio(_word(80) + _gap(10) + _word(80) + _gap(60) + _word(400))

    stop = word_boundary_stop(audio, _samples(20))

    assert _samples(170) <= stop < _samples(175)


def test_a_gap_already_played_is_not_chosen() -> None:
    audio = _audio(_word(100) + _gap(60) + _word(150) + _gap(60) + _word(400))

    stop = word_boundary_stop(audio, _samples(180))

    assert _samples(310) <= stop < _samples(315)


def test_a_gap_beyond_the_cap_is_not_chosen() -> None:
    audio = _audio(_word(500) + _gap(60) + _word(200))

    stop = word_boundary_stop(audio, _samples(100))

    assert stop == _samples(100) + round(MAX_WORD_TAIL_SECONDS * RATE)


def test_chunk_end_inside_the_cap_is_a_natural_stop() -> None:
    audio = _audio(_word(200))

    assert word_boundary_stop(audio, _samples(100)) == _samples(200)


def test_stop_rejects_positions_outside_the_chunk() -> None:
    audio = _audio(_word(50))

    with pytest.raises(ValueError, match="position"):
        word_boundary_stop(audio, _samples(51))
    with pytest.raises(TypeError, match="position"):
        word_boundary_stop(audio, True)


def test_fade_reaches_exactly_zero_without_a_discontinuity() -> None:
    constant = [20_000] * _samples(100)
    audio = _audio(constant)
    position = _samples(40)

    faded = fade_out(audio, position)

    assert faded is not None
    pcm = _pcm(faded)
    assert len(pcm) == round(FADE_OUT_SECONDS * RATE)
    assert pcm[-1] == 0
    # Continuous with the sample before the splice, then monotone to zero.
    assert abs(pcm[0] - constant[position - 1]) <= 20_000 * math.pi / len(pcm)
    steps = [abs(after - before) for before, after in zip(pcm, pcm[1:], strict=False)]
    assert max(steps) <= 20_000 * math.pi / len(pcm)
    assert all(after <= before for before, after in zip(pcm, pcm[1:], strict=False))


def test_fade_reads_the_continuation_from_the_position() -> None:
    samples = list(range(0, _samples(100)))
    samples = [value % 30_000 for value in samples]
    audio = _audio(samples)
    position = _samples(30)

    faded = fade_out(audio, position)

    assert faded is not None
    pcm = _pcm(faded)
    assert abs(pcm[0] - samples[position]) <= 1
    assert all(abs(out) <= abs(src) for out, src in zip(pcm, samples[position:], strict=False))


def test_a_fade_cut_short_by_the_chunk_end_still_ends_at_zero() -> None:
    audio = _audio([20_000] * 10)

    faded = fade_out(audio, 8)

    assert faded is not None
    assert _pcm(faded) == [10_000, 0]


def test_fade_zeroes_every_channel_and_stops_at_the_chunk_end() -> None:
    audio = _audio([10_000] * _samples(20), channels=2)

    faded = fade_out(audio, _samples(15))

    assert faded is not None
    pcm = _pcm(faded)
    assert len(pcm) == 2 * _samples(5)
    assert pcm[-2:] == [0, 0]
    assert fade_out(audio, _samples(20)) is None
