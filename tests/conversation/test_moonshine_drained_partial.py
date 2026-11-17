"""A short barge-in's own streaming partial must reach the worker before its final (#576).

The real Moonshine wrapper with a scripted backend. A promoted echo candidate's frames
are pushed in one burst: each push only schedules the backend add, and the next push
runs before that task can start, so no push can return its partial. The partial is
harvested only while the endpoint drains, before the final.
"""

from __future__ import annotations

import pytest

from hermes_realtime.conversation import ConversationSessionWorker
from hermes_realtime.providers.moonshine import MoonshineStreamingTranscriber
from hermes_realtime.speech import AudioFrame, SpeechPresence, Transcript, VoiceActivity
from tests.conversation.test_worker import (
    ScriptedEchoGuard,
    ScriptedSpeechPresenceVerifier,
    ScriptedVad,
    action_executor_probe,
)
from tests.providers.test_moonshine import ScriptedMoonshineBackend

_FRAMES = (
    AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
    AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
)


async def _barge_in(*, streams_partials: bool) -> tuple[
    ScriptedMoonshineBackend, bool, list[tuple[str, Transcript]], list[str]
]:
    executor, foreground_cancelled, responses = action_executor_probe()
    backend = ScriptedMoonshineBackend("Stop", final="Stop.")
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(backend=backend),
        stt_streams_partials=streams_partials,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH),
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    for frame in _FRAMES:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()
    await worker.close()
    return backend, foreground_cancelled.is_set(), responses, observed


@pytest.mark.asyncio
async def test_control_without_the_evidence_rule_the_same_barge_in_is_admitted() -> None:
    # Everything but the streaming-evidence rule admits this short barge-in, so a
    # failure of the test below is that rule meeting a partial the worker never saw.
    backend, cancelled, responses, observed = await _barge_in(streams_partials=False)

    assert backend.add_sizes and backend.finish_calls == 1
    assert cancelled is True
    assert "interrupt_requested" in observed
    assert [transcript for _turn, transcript in responses] == [
        Transcript(text="Stop.", final=True)
    ]


@pytest.mark.asyncio
async def test_endpoint_promoted_short_word_keeps_its_drained_streaming_partial() -> None:
    backend, cancelled, responses, observed = await _barge_in(streams_partials=True)

    # The backend decoded "Stop" from this utterance's own audio before its final.
    assert backend.add_sizes and backend.finish_calls == 1
    assert "transcript_partial" in observed
    assert cancelled is True
    assert "interrupt_requested" in observed
    assert [transcript for _turn, transcript in responses] == [
        Transcript(text="Stop.", final=True)
    ]
