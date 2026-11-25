"""A short barge-in's own streaming partial must reach the worker before its final (#576).

The real Moonshine wrapper with a scripted backend. A promoted echo candidate's frames
are pushed in one burst: each push only schedules the backend add, and the next push
runs before that task can start, so no push can return its partial. The partial is
harvested only while the endpoint drains, before the final.
"""

from __future__ import annotations

import asyncio
import json

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


async def _barge_in(
    *,
    streams_partials: bool = True,
    partial: str | None = "Stop",
    final: str = "Stop.",
    transcript_echoes: tuple[str, ...] = (),
) -> tuple[ScriptedMoonshineBackend, bool, list[tuple[str, Transcript]], list[str]]:
    executor, foreground_cancelled, responses = action_executor_probe()
    backend = ScriptedMoonshineBackend(partial, final=final)
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(backend=backend),
        stt_streams_partials=streams_partials,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False, transcript_echoes=transcript_echoes),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH),
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    for frame in _FRAMES:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()
    cancelled = foreground_cancelled.is_set()  # before close, which ends the foreground
    await worker.close()
    return backend, cancelled, responses, observed


@pytest.mark.asyncio
async def test_control_without_the_evidence_rule_the_same_barge_in_is_admitted() -> None:
    # Everything but the streaming-evidence rule admits this short barge-in, so a
    # failure of the test below is that rule meeting a partial the worker never saw.
    backend, cancelled, responses, observed = await _barge_in(streams_partials=False)

    assert backend.add_sizes and backend.finish_calls == 1
    assert cancelled is True
    assert "interrupt_requested" in observed
    # Admitted, then consumed as speech-only: a playback stop word starts no turn.
    assert "transcript_final" in observed
    assert responses == []


@pytest.mark.asyncio
async def test_endpoint_promoted_short_word_keeps_its_drained_streaming_partial() -> None:
    backend, cancelled, responses, observed = await _barge_in()

    # The backend decoded "Stop" from this utterance's own audio before its final.
    assert backend.add_sizes and backend.finish_calls == 1
    assert "transcript_partial" in observed
    assert cancelled is True
    assert "interrupt_requested" in observed
    assert "transcript_final" in observed
    assert responses == []


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (None, "No"), ids=("no-partial", "subthreshold-partial"))
async def test_an_exact_stop_word_final_needs_no_drained_evidence(partial: str | None) -> None:
    # The single stop word is relieved of partial evidence under its acoustic gates (#576).
    backend, cancelled, responses, observed = await _barge_in(partial=partial)

    assert backend.add_sizes and backend.finish_calls == 1
    assert cancelled is True
    assert "interrupt_requested" in observed
    assert "transcript_final" in observed
    assert responses == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial", "final", "transcript_echoes", "echo_suppressed"),
    (
        pytest.param(None, "Chapter 3.", (), False, id="no-partial"),
        pytest.param("No", "Chapter 3.", (), False, id="subthreshold-partial"),
        # Long enough for the real guard's text match, which never matches a short word.
        pytest.param(
            "yes i can switch models", "Chapter 3.", ("yes i can switch models",), True,
            id="echoed-partial",
        ),
    ),
)
async def test_drained_partial_without_evidence_leaves_the_final_unauthoritative(
    partial: str | None, final: str, transcript_echoes: tuple[str, ...], echo_suppressed: bool
) -> None:
    backend, cancelled, responses, observed = await _barge_in(
        partial=partial, final=final, transcript_echoes=transcript_echoes
    )

    assert backend.add_sizes and backend.finish_calls == 1
    assert cancelled is False
    assert "interrupt_requested" not in observed
    assert responses == []
    assert ("transcript_echo_suppressed" in observed) is echo_suppressed


@pytest.mark.asyncio
async def test_an_utterance_outside_playback_is_not_drained() -> None:
    executor, _cancelled, responses = action_executor_probe()
    executor._speech.foreground_active = False  # type: ignore[attr-defined]
    backend = ScriptedMoonshineBackend("Hello", final="Hello there.")
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(backend=backend),
        actions=executor,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    for frame in _FRAMES:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()
    await worker.close()

    # Unchanged behaviour: the final is admitted, and no endpoint drain surfaced a partial.
    assert [transcript for _turn, transcript in responses] == [
        Transcript(text="Hello there.", final=True)
    ]
    assert "transcript_partial" not in observed


class _DrainingTranscriber:
    def __init__(self, drained: object) -> None:
        self._drained = drained

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        return ()

    async def drain_partials(self) -> object:
        return self._drained

    async def finish_utterance(self) -> Transcript:
        return Transcript(text="Stop.", final=True)

    async def cancel(self) -> None:
        return None


_BAIT = "SECRET-BAIT private words"


def _drained_partial_markers(output: str) -> list[dict[str, object]]:
    prefix = "[drained-partials] "
    return [
        json.loads(line[len(prefix) :])
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("drained", "error", "marker"),
    (
        pytest.param(
            (Transcript(text="Stop", final=False), Transcript(text=_BAIT, final=True)),
            ValueError,
            {"category": "final", "items": 2, "version": 1},
            id="final",
        ),
        pytest.param(
            [Transcript(text=_BAIT, final=False)],
            TypeError,
            {"category": "not_a_tuple", "items": 0, "version": 1},
            id="not-a-tuple",
        ),
        pytest.param(
            (_BAIT,),
            TypeError,
            {"category": "not_a_transcript", "items": 1, "version": 1},
            id="not-a-transcript",
        ),
    ),
)
async def test_a_drain_that_is_not_exact_non_final_partials_fails_closed(
    drained: object,
    error: type[Exception],
    marker: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=_DrainingTranscriber(drained),
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH),
    )
    worker.start()
    await worker.receive_audio("browser_user", 1, _FRAMES[0])

    with pytest.raises(error):
        await worker.receive_audio("browser_user", 1, _FRAMES[1])

    assert foreground_cancelled.is_set() is False
    assert responses == []
    # The refusal leaves exactly one bounded line: category and item count, no content.
    output = capsys.readouterr().out
    assert _drained_partial_markers(output) == [marker]
    assert "SECRET-BAIT" not in output


@pytest.mark.asyncio
async def test_a_refusal_marker_that_cannot_be_written_does_not_mask_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_print(*args: object, **kwargs: object) -> None:
        raise OSError("stdout unavailable")

    monkeypatch.setattr("builtins.print", broken_print)
    with pytest.raises(ValueError, match="non-final"):
        ConversationSessionWorker._trusted_partials((Transcript(text="Stop.", final=True),))


@pytest.mark.asyncio
async def test_an_admitted_drain_writes_no_refusal_marker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _barge_in()
    assert _drained_partial_markers(capsys.readouterr().out) == []


@pytest.mark.asyncio
async def test_a_binding_closed_during_the_drain_creates_no_interrupt_or_turn() -> None:
    executor, _foreground_cancelled, responses = action_executor_probe()
    backend = ScriptedMoonshineBackend("Stop", final="Stop.")
    backend.release.clear()
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(backend=backend),
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH),
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    await worker.receive_audio("browser_user", 1, _FRAMES[0])
    endpoint = asyncio.create_task(worker.receive_audio("browser_user", 1, _FRAMES[1]))
    assert await asyncio.to_thread(backend.added.wait, 1)

    closing = asyncio.create_task(worker.close())
    await asyncio.sleep(0)
    backend.release.set()
    await asyncio.gather(endpoint, return_exceptions=True)
    await closing

    # Close itself ends the foreground; what must not happen is a barge-in or a turn.
    # The probe's close cleanup leaves this flag alone, so only a barge-in cancel clears it.
    assert executor._speech.foreground_active is True  # type: ignore[attr-defined]
    assert "interrupt_requested" not in observed
    assert "transcript_final" not in observed
    assert responses == []
