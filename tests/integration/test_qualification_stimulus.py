from __future__ import annotations

import struct

from hermes_realtime.livekit.playback import _MATERIAL_SAMPLE_THRESHOLD
from tests.integration.test_qualification_full_host_ingress import _speech_pcm

_SAMPLE_RATE_HZ = 48_000
_FRAME_SAMPLES = 480  # 10 ms at 48 kHz, the codec's frame.
# Telephone band: every Opus mode, narrowband included, passes it without attenuation.
_PRESERVED_BAND_HZ = (300, 3_400)


def test_synthesized_qualification_speech_is_material_in_every_codec_frame() -> None:
    """Delivery confirmation must never depend on one marginal frame surviving the codec (#183).

    No Opus codec is available to this suite, so the stimulus is held to the raw properties
    that make its decoded frames material: many frames, each peaking far above the material
    threshold, no DC (Opus attenuates it), and a frequency inside the band every Opus mode
    preserves. Decoded delivery itself is exercised by the LiveKit integration gates.
    """

    pcm = _speech_pcm()
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    frames = [
        samples[start : start + _FRAME_SAMPLES] for start in range(0, len(samples), _FRAME_SAMPLES)
    ]

    assert len(frames) >= 10
    assert all(len(frame) == _FRAME_SAMPLES for frame in frames)
    assert min(max(abs(sample) for sample in frame) for frame in frames) >= (
        8 * _MATERIAL_SAMPLE_THRESHOLD
    )
    assert abs(sum(samples)) / len(samples) < _MATERIAL_SAMPLE_THRESHOLD / 10
    for frame in frames:
        crossings = sum(
            (earlier < 0) != (later < 0) for earlier, later in zip(frame, frame[1:], strict=False)
        )
        frequency_hz = crossings / 2 * _SAMPLE_RATE_HZ / _FRAME_SAMPLES
        assert _PRESERVED_BAND_HZ[0] <= frequency_hz <= _PRESERVED_BAND_HZ[1]
