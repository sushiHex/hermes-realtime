"""Host-composed readback is not model output and is never call-imitation checked (#311)."""

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
from tests.conversation.test_work_tools import _surface


@pytest.mark.asyncio
async def test_proposal_readback_quoting_a_call_is_still_spoken() -> None:
    context = ConversationContextStore()
    surface, _, _ = _surface(context=context)
    proposals = ConversationTaskProposals(surface=surface, observer=lambda *_: None)
    proposals.bind("participant", 1)

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield HostTaskProposalRequest("Add a start_work(objective) guard to the adapter")

        async def cancel(self, turn_id):
            pass

    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    try:
        await loop.respond("turn_readback", Transcript(text="Guard the adapter", final=True))
    finally:
        await surface.close()

    assert any("start_work(objective)" in text for text in synth.texts)
