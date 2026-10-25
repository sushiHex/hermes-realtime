"""Readback completion is an exact transport boundary, not generated prose."""

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
from tests.conversation.test_task_proposals import Surface


class ProposalInference:
    async def stream(self, _snapshot, *, turn_id):
        yield HostTaskProposalRequest("Inspect **the exact** release")

    async def cancel(self, turn_id):
        pass


@pytest.mark.asyncio
async def test_loop_renders_host_readback_and_arms_only_after_playback():
    surface = Surface()
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, speech_projection=lambda text: text
    )
    proposals.bind("participant", 1)
    synth = SegmentSynthesizer()
    context = ConversationContextStore()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=ProposalInference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    await loop.respond("turn_propose", Transcript(text="Please inspect the release", final=True))
    assert synth.texts == [
        "Proposed background task. Inspect **the exact** release Shall I send this task?"
    ]
    assert surface.calls == []
    assert context.snapshot().messages[-1].source == "host_task_proposal"
    proposal = proposals.confirmation_target("yes")
    assert proposal is not None
    proposals.capture_started(participant="participant", generation=1, clean=True)
    await proposals.consume_input("yes", expected_proposal=proposal)
    assert surface.calls[0][0] == "Inspect **the exact** release"


@pytest.mark.asyncio
async def test_interrupted_readback_never_arms_confirmation():
    surface = Surface()
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, speech_projection=lambda text: text
    )
    proposals.bind("participant", 1)

    class FailedPlayback(RecordingPlayback):
        async def play(self, chunk, *, is_valid=None):
            raise RuntimeError("synthetic playback failure")

    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=ProposalInference(),
        synthesizer=SegmentSynthesizer(),
        playback=FailedPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    with pytest.raises(RuntimeError, match="synthetic playback failure"):
        await loop.respond("turn_propose", Transcript(text="Inspect release", final=True))
    proposal = proposals.confirmation_target("yes")
    assert proposal is not None
    proposals.capture_started(participant="participant", generation=1, clean=True)
    await proposals.consume_input("yes", expected_proposal=proposal)
    assert surface.calls == []
