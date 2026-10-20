"""Known TTS text projection is required for spoken confirmation eligibility."""

import pytest

from hermes_realtime import host_launcher
from hermes_realtime.providers.edge_tts import EdgeTtsSynthesizer
from hermes_realtime.providers.kokoro import KokoroSynthesizer


@pytest.mark.parametrize("provider", [EdgeTtsSynthesizer, KokoroSynthesizer])
def test_readback_projection_matches_actual_provider_normalization(provider):
    synth = object.__new__(provider)
    if provider is KokoroSynthesizer:
        synth._pronunciation_pattern = None
    assert host_launcher._proposal_spoken_text(synth, "Inspect release") == "Inspect release"
    assert (
        host_launcher._proposal_spoken_text(synth, "Inspect **release**") != "Inspect **release**"
    )


def test_unknown_tts_cannot_attest_exact_spoken_readback():
    assert host_launcher._proposal_spoken_text(object(), "Inspect release") is None
