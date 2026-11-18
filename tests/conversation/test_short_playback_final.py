"""Two-character finals retain host speech and same-utterance streaming evidence."""
from __future__ import annotations

import pytest

from hermes_realtime.conversation import ConversationSessionWorker, ConversationTaskCommandRouter
from hermes_realtime.providers.moonshine import MoonshineStreamingTranscriber
from hermes_realtime.speech import AudioFrame, SpeechPresence, Transcript, VoiceActivity
from tests.conversation.test_worker import (
    ScriptedEchoGuard,
    ScriptedSpeechPresenceVerifier,
    ScriptedVad,
    action_executor_probe,
)
from tests.providers.test_moonshine import ScriptedMoonshineBackend

FRAMES = tuple(AudioFrame(pcm=bytes([i, 0]) * 480, sample_rate_hz=48000, channels=1)
               for i in (3, 4))


async def utterance(*partials, final="No.", verify=True, echoes=(), proposals=None, router=None,
                    foreground=False, on_finish=None, presence=SpeechPresence.CONFIRMED_SPEECH):
    actions, cancelled, responses = action_executor_probe()
    actions._speech.foreground_active = foreground
    actions._speech.active_turn_id = "foreground_1" if foreground else None
    events = []
    backend = ScriptedMoonshineBackend(*partials, final=final)
    if on_finish is not None:
        finish = backend.finish
        def intercepted_finish():
            on_finish(actions)
            return finish()
        backend.finish = intercepted_finish
    worker = ConversationSessionWorker(
        participant_identity="participant", session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(
            backend=backend, max_add_bytes=960),
        actions=actions, task_proposals=proposals, command_router=router,
        echo_guard=ScriptedEchoGuard(False, False, recent_playback=True,
                                     transcript_echoes=echoes),
        speech_presence_verifier=(ScriptedSpeechPresenceVerifier(presence)
                                  if verify else None),
        observer=lambda kind, data: events.append((kind, data)),
    )
    try:
        worker.start()
        for frame in FRAMES:
            await worker.receive_audio("participant", 1, frame)
        await worker.wait_for_responses()
        return responses, events, cancelled.is_set()
    finally:
        await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("partial", "final"), [("No", "No."), ("ok", "OK?!"), ("42", "42.")])
async def test_verified_two_character_final_agrees_with_its_own_partial(partial, final):
    responses, events, cancelled = await utterance(partial, final=final)
    assert [t.text for _, t in responses] == [final]
    assert sum(k == "transcript_final" for k, _ in events) == 1
    assert not cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize(("partials", "final", "verify", "echoes"), [
    ((), "No.", True, ()),
    (("On",), "No.", True, ()),
    (("No",), "On.", True, ()),
    (("##",), "No.", True, ()),
    (("##",), "##.", True, ()),
    (("N-o",), "No.", True, ()),
    (("N O",), "No.", True, ()),
    (("No",), "N-o.", True, ()),
    (("No",), "No please.", True, ()),
    (("N",), "N.", True, ()),
    (("No", "On"), "No.", True, ()),
    (("No",), "No.", False, ()),
    (("No",), "No.", True, ("No",)),
    (("No",), "No.", True, ("No.",)),
])
async def test_short_final_without_exact_verified_non_echo_evidence_refuses(
    partials, final, verify, echoes,
):
    responses, events, cancelled = await utterance(
        *partials, final=final, verify=verify, echoes=echoes)
    assert responses == []
    assert not any(k == "transcript_final" for k, _ in events)
    assert not cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize("presence", [
    SpeechPresence.UNCERTAIN, SpeechPresence.CONFIRMED_NON_SPEECH,
])
async def test_explicit_verifier_must_confirm_speech_for_short_final(presence):
    responses, events, cancelled = await utterance("No", presence=presence)
    assert responses == []
    assert not any(k == "transcript_final" for k, _ in events)
    assert not cancelled


@pytest.mark.asyncio
async def test_no_retires_the_real_pending_cancel_without_cancelling_work():
    from hermes_realtime.conversation.natural_cancel import HostCancelRequest
    from tests.conversation.test_confirmed_multi_cancel import Nominations, _owner, _say

    loop, context, controller, proposals, _, _ = _owner(
        "alpha", "beta", inference=Nominations(HostCancelRequest(target="choice_2")))
    await _say(loop, proposals, "Stop beta.", "initial")
    assert proposals.has_pending
    router = ConversationTaskCommandRouter(observer=lambda *_: None,
        reserve_observer_capacity=lambda: None, surface=proposals._surface,
        task_proposals=proposals)
    responses, _, cancelled = await utterance("No", proposals=proposals, router=router)
    assert [t.text for _, t in responses] == ["No."]
    assert not proposals.has_pending and controller.cancellations == [] and not cancelled
    # This input cannot confirm the retired object, even with a later clean capture.
    proposals.capture_started(participant="participant", generation=1, clean=True)
    assert proposals.confirmation_target("yes") is None
    assert len(context.snapshot().active_tasks) == 2


@pytest.mark.asyncio
async def test_recent_playback_affirmative_still_cannot_confirm_a_pending_cancel():
    from hermes_realtime.conversation.natural_cancel import HostCancelRequest
    from tests.conversation.test_confirmed_multi_cancel import Nominations, _owner, _say

    loop, _, controller, proposals, _, _ = _owner(
        "alpha", "beta", inference=Nominations(HostCancelRequest(target="choice_1")))
    await _say(loop, proposals, "Stop alpha.", "initial")
    router = ConversationTaskCommandRouter(observer=lambda *_: None,
        reserve_observer_capacity=lambda: None, surface=proposals._surface,
        task_proposals=proposals)
    await utterance("Yes", final="Yes.", proposals=proposals, router=router)
    assert controller.cancellations == [] and not proposals.has_pending


@pytest.mark.asyncio
async def test_short_final_cannot_route_against_a_replaced_foreground():
    def replace(actions):
        actions._speech.active_turn_id = "replacement"
    responses, events, cancelled = await utterance("No", foreground=True, on_finish=replace)
    assert not cancelled
    assert responses == []
    assert not any(k == "transcript_final" for k, _ in events)


class Drains:
    def __init__(self): self.index = 0
    async def push(self, frame): return ()
    async def drain_partials(self):
        return (Transcript(text="No", final=False),) if self.index == 0 else ()
    async def finish_utterance(self):
        self.index += 1
        return None if self.index == 1 else Transcript(text="No.", final=True)
    async def cancel(self): pass


@pytest.mark.asyncio
async def test_short_evidence_is_cleared_at_endpoint_and_cannot_cross_utterances():
    actions, _, responses = action_executor_probe()
    actions._speech.foreground_active = False
    worker = ConversationSessionWorker(participant_identity="participant", session_generation=1,
        vad=ScriptedVad(*([VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED] * 2)),
        stt=Drains(), actions=actions,
        echo_guard=ScriptedEchoGuard(False, False, False, False, recent_playback=True),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(
            SpeechPresence.CONFIRMED_SPEECH, SpeechPresence.CONFIRMED_SPEECH))
    try:
        worker.start()
        for frame in FRAMES:
            await worker.receive_audio("participant", 1, frame)
        assert worker._playback_candidate_short_partial is None
        # Also prove onset resets evidence, independently of endpoint cleanup.
        worker._playback_candidate_short_partial = "no"
        for frame in FRAMES:
            await worker.receive_audio("participant", 1, frame)
        await worker.wait_for_responses()
        assert responses == []
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_refusal_evidence_is_counts_and_fixed_categories_only(capsys):
    await utterance("No", final="PRIVATE-BAIT words")
    output = capsys.readouterr().out
    import json
    prefix = "[playback-final-refused] "
    rows = [json.loads(line[len(prefix):]) for line in output.splitlines()
            if line.startswith(prefix)]
    assert rows == [{"version": 1, "category": "streaming_evidence_missing",
                    "short_partial_chars": 2, "final_chars": 18, "speech_verifier": True}]
    assert "PRIVATE-BAIT" not in output
