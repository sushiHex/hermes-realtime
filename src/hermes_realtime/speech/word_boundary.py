"""Provider-neutral word-boundary stop policy for interrupted speech PCM.

An interrupted chunk keeps playing its already-synthesized PCM to the next gap
between words, then fades out. A gap is a run of short frames whose energy sits
far below the chunk's own speech level, so detection is independent of absolute
volume and of the provider that rendered the audio. The constants were chosen by
measuring real Kokoro output against its own phoneme durations (see #211).
"""

from __future__ import annotations

import math
from operator import mul

from .types import AudioFrame

# Short-frame grid for energy analysis.
FRAME_SECONDS = 0.005
# The chunk's speech level is this percentile of its frame energies.
CHUNK_LEVEL_PERCENTILE = 0.9
# A frame belongs to a gap when its energy is this far below the chunk level.
GAP_DEPTH_DB = 20.0
# The shortest quiet run accepted as a gap between words.
MIN_GAP_SECONDS = 0.025
# The most extra audio an interruption may play while looking for a gap.
MAX_WORD_TAIL_SECONDS = 0.3
# Every stop ends with a raised-cosine fade of this length.
FADE_OUT_SECONDS = 0.015
# At most this many samples are read to estimate the chunk level, which keeps
# the analysis on the cancellation path to a few milliseconds for any chunk.
_LEVEL_SAMPLE_BUDGET = 48_000


def word_boundary_stop(audio: AudioFrame, position: int) -> int:
    """Return the sample at which playback interrupted at ``position`` should stop.

    The result is the start of the first gap at or after ``position`` when that
    gap starts within ``MAX_WORD_TAIL_SECONDS``; otherwise the cap, or the end of
    the chunk when the chunk ends first.
    """

    total = _validate(audio, position)
    rate = audio.sample_rate_hz
    frame = max(1, round(FRAME_SECONDS * rate))
    cap = position + round(MAX_WORD_TAIL_SECONDS * rate)
    samples = memoryview(audio.pcm).cast("h")
    width = frame * audio.channels
    frames = len(samples) // width
    if frames:
        # The level is a percentile over every frame, so a bounded subsample of
        # each frame estimates it as well as reading the whole chunk would.
        stride = min(width, max(1, math.ceil(len(samples) / _LEVEL_SAMPLE_BUDGET)))
        ranked = sorted(
            _mean_square(samples, index * width, width, stride) for index in range(frames)
        )
        level = ranked[max(0, math.ceil(CHUNK_LEVEL_PERCENTILE * frames) - 1)]
        threshold = level * 10 ** (-GAP_DEPTH_DB / 10)
        required = max(1, math.ceil(MIN_GAP_SECONDS / FRAME_SECONDS - 1e-9))
        run = 0
        for index in range(-(-position // frame), frames):
            if (index - run) * frame >= cap:
                break
            quiet = _mean_square(samples, index * width, width, 1) <= threshold
            run = run + 1 if quiet else 0
            if run == required:
                return (index - required + 1) * frame
    return min(total, cap)


def fade_out(audio: AudioFrame, position: int) -> AudioFrame | None:
    """Return the next ``FADE_OUT_SECONDS`` of PCM from ``position``, faded to zero.

    The gain follows a raised cosine that starts just below one and ends exactly
    at zero on every channel. ``None`` means nothing remains to fade.
    """

    total = _validate(audio, position)
    length = min(round(FADE_OUT_SECONDS * audio.sample_rate_hz), total - position)
    if length <= 0:
        return None
    channels = audio.channels
    samples = memoryview(audio.pcm).cast("h")
    faded = memoryview(bytearray(2 * length * channels)).cast("h")
    for offset in range(length):
        gain = 0.5 * (1.0 + math.cos(math.pi * (offset + 1) / length))
        base = (position + offset) * channels
        for channel in range(channels):
            faded[offset * channels + channel] = round(samples[base + channel] * gain)
    return AudioFrame(
        pcm=faded.tobytes(),
        sample_rate_hz=audio.sample_rate_hz,
        channels=channels,
    )


def _mean_square(samples: memoryview, start: int, width: int, stride: int) -> float:
    window = samples[start : start + width : stride]
    energy: int = sum(map(mul, window, window))
    return energy / len(window)


def _validate(audio: AudioFrame, position: int) -> int:
    if type(audio) is not AudioFrame:
        raise TypeError("audio must be an exact AudioFrame")
    if type(position) is not int:
        raise TypeError("position must be an exact integer sample index")
    total = len(audio.pcm) // (2 * audio.channels)
    if not 0 <= position <= total:
        raise ValueError("position must lie within the chunk")
    return total
