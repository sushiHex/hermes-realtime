"""Known TTS text projection is required for spoken confirmation eligibility."""

import asyncio

import pytest

from hermes_realtime import host_launcher
from hermes_realtime.providers.edge_tts import EdgeTtsSynthesizer
from hermes_realtime.providers.kokoro import KokoroSynthesizer
from hermes_realtime.speech import AudioFrame, SpeechChunk


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


def test_projection_capability_is_shared_with_non_provider_wrappers():
    class Wrapper:
        def project_spoken_text(self, text):
            return text

    assert host_launcher._proposal_spoken_text(Wrapper(), "Inspect release") == "Inspect release"


@pytest.mark.parametrize("failure", ["raises", "wrong_type"])
def test_failed_projection_keeps_confirmation_card_available(failure):
    class Wrapper:
        called = False

        def project_spoken_text(self, text):
            self.called = True
            if failure == "raises":
                raise RuntimeError("synthetic projection failure")
            return 1

    wrapper = Wrapper()
    assert host_launcher._proposal_spoken_text(wrapper, "Inspect release") is None
    assert wrapper.called


@pytest.mark.asyncio
async def test_delayed_readiness_cue_invalidates_confirmation_immediately_before_playback():
    entered, release = asyncio.Event(), asyncio.Event()

    class ProposalBinding:
        eligible = False

        def speech_started(self):
            self.eligible = False

    proposals = ProposalBinding()

    class Conversation:
        async def suppress_audio_input(self, **kwargs):
            entered.set()
            await release.wait()
            return object()

        async def resume_audio_input(self, token):
            pass

    class Playback:
        async def play(self, chunk, *, is_valid):
            assert not proposals.eligible

    chunk = SpeechChunk(
        turn_id="ready",
        chunk_id="ready",
        text="Ready.",
        audio=AudioFrame(
            pcm=b"\x00\x00",
            sample_rate_hz=48_000,
            channels=1,
        ),
    )
    operation = asyncio.create_task(
        host_launcher._play_readiness_cue(
            conversation=Conversation(),
            playback=Playback(),
            chunk=chunk,
            participant_identity="participant",
            generation=1,
            is_valid=lambda: True,
            task_proposals=proposals,
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    # The cue was scheduled before a readback settled, but starts afterwards.
    proposals.eligible = True
    release.set()
    await operation
