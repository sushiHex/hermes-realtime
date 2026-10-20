"""Confirmation eligibility belongs to captured input, never delayed STT timing."""

import pytest

from hermes_realtime.conversation.commands import ConversationTaskCommandRouter
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.speech import AudioFrame, VoiceActivity
from tests.conversation.test_task_proposals import Surface
from tests.conversation.test_worker import (
    CapturingTextFinalTranscriber,
    ScriptedEchoGuard,
    ScriptedVad,
    action_executor_probe,
)


class CommandSurface(Surface):
    active_task_ids = ()

    async def submit_cancel_command(self, **kwargs):
        raise AssertionError("proposal confirmation must never cancel tasks")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocker", [None, "no_guard", "recent", "echo", "interrupting", "pre_roll"]
)
async def test_only_clean_onset_can_confirm_after_complete_readback(blocker):
    surface = CommandSurface()
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, speech_projection=lambda text: text
    )
    proposals.bind("browser_user", 1)
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release", "call"))
    proposals.complete_readback(proposal, proposal.readback, queued_chunks=1, confirmed_chunks=1)
    actions, _cancelled, _responses = action_executor_probe()
    actions._speech.foreground_active = blocker == "interrupting"
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda *_: None,
        reserve_observer_capacity=lambda: None,
        task_proposals=proposals,
    )
    # The test's surface needs explicit cancellation only for router construction.
    guard = ScriptedEchoGuard(blocker == "echo", False, False, recent_playback=blocker == "recent")
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            *(
                ([VoiceActivity.SILENCE] if blocker == "pre_roll" else [])
                + [VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED]
            )
        ),
        stt=CapturingTextFinalTranscriber("yes"),
        stt_streams_partials=False,
        actions=actions,
        command_router=router,
        task_proposals=proposals,
        echo_guard=None if blocker == "no_guard" else guard,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)
    try:
        if blocker == "pre_roll":
            guard._recent_playback = True
            await worker.receive_audio("browser_user", 1, frame)
            guard._recent_playback = False
        await worker.receive_audio("browser_user", 1, frame)
        # The output tail can expire while STT runs; it must not upgrade onset.
        guard._recent_playback = False
        await worker.receive_audio("browser_user", 1, frame)
        assert len(surface.calls) == (1 if blocker is None else 0)
    finally:
        await worker.close()
