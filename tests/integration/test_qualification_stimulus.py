from __future__ import annotations

import struct

from hermes_realtime.livekit.playback import _MATERIAL_SAMPLE_THRESHOLD
from tests.integration.test_qualification_full_host_ingress import _speech_pcm

_FRAME_SAMPLES = 480  # 10 ms at 48 kHz, the codec's frame.


def test_synthesized_qualification_speech_is_material_in_every_codec_frame() -> None:
    """Delivery confirmation must never depend on one marginal frame surviving the codec (#183).

    The stimulus spans many frames, each peaking far above the material threshold, and carries
    no DC, which Opus attenuates.
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
