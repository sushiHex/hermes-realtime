"""Synthetic confirmation timing against the real recent-playback and capture latch."""
from __future__ import annotations

import pytest

from hermes_realtime.conversation import ConversationSessionWorker, ConversationTaskCommandRouter
from hermes_realtime.conversation.natural_cancel import HostCancelRequest
from hermes_realtime.providers.echo import PlaybackEchoGuard
from hermes_realtime.providers.moonshine import MoonshineStreamingTranscriber
from hermes_realtime.providers.vad import WebRtcVoiceActivityDetector
from hermes_realtime.speech import AudioFrame, SpeechPresence, VoiceActivity
from tests.conversation.test_confirmed_multi_cancel import Nominations, _owner, _say, _settle
from tests.conversation.test_worker import (
    ScriptedSpeechPresenceVerifier,
    ScriptedVad,
    action_executor_probe,
)
from tests.providers.test_moonshine import ScriptedMoonshineBackend


def test_confirmation_schedule_clears_default_tail_and_pre_roll():
    from scripts.native_typed_journey import (
        _SPEAK_DRAIN_SECONDS,
        _SPEAK_FRAME_SECONDS,
        _SPEAK_LEAD_FRAMES,
        _SPEAK_SETTLE_SECONDS,
    )

    now = [0.0]
    guard = PlaybackEchoGuard(clock=lambda: now[0])
    frame = AudioFrame(pcm=bytes(960), sample_rate_hz=48000, channels=1)
    guard.end_playback(guard.begin_playback(frame))
    vad = WebRtcVoiceActivityDetector(classify=lambda *_: False)
    # Earliest retained frame, conservatively subtracting all onset pre-roll.
    now[0] = (_SPEAK_DRAIN_SECONDS + _SPEAK_SETTLE_SECONDS
              + _SPEAK_LEAD_FRAMES * _SPEAK_FRAME_SECONDS
              - vad.required_pre_roll_frames * _SPEAK_FRAME_SECONDS)
    assert not guard.has_recent_playback()


@pytest.mark.asyncio
@pytest.mark.parametrize(('onset', 'accepted'), [(0.8, False), (1.65, False), (1.8, True)])
async def test_affirmative_capture_requires_expired_playback_and_clean_pre_roll(onset, accepted):
    loop, _, controller, proposals, _, _ = _owner(
        'alpha', 'beta', inference=Nominations(HostCancelRequest(target='choice_1')))
    await _say(loop, proposals, 'Stop alpha.', 'initial')
    assert proposals.has_pending
    now = [0.0]
    guard = PlaybackEchoGuard(clock=lambda: now[0])
    reference = AudioFrame(pcm=bytes(960), sample_rate_hz=48000, channels=1)
    guard.end_playback(guard.begin_playback(reference, 'Stop background task. Inspect alpha?'))
    actions, _, responses = action_executor_probe()
    actions._speech.foreground_active = False
    router = ConversationTaskCommandRouter(observer=lambda *_: None,
        reserve_observer_capacity=lambda: None, surface=proposals._surface,
        task_proposals=proposals)
    worker = ConversationSessionWorker(participant_identity='participant', session_generation=1,
        vad=ScriptedVad(*([VoiceActivity.SILENCE] * 19),
                        VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        pre_roll_frames=19,
        stt=MoonshineStreamingTranscriber(backend=ScriptedMoonshineBackend('Yes', final='Yes.')),
        actions=actions, task_proposals=proposals, command_router=router, echo_guard=guard,
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH))
    try:
        worker.start()
        for i in range(21):
            now[0] = onset + (i - 19) / 100
            await worker.receive_audio('participant', 1, reference)
        await worker.wait_for_responses()
        await _settle()
        assert bool(controller.cancellations) is accepted
        assert not proposals.has_pending
        if accepted:
            assert controller.cancellations == [('task_alpha', 'user requested task cancellation')]
        else:
            assert [t.text for _, t in responses] == ['Yes.']
    finally:
        await worker.close()
