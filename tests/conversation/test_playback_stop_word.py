"""A bare "stop" during playback silences speech and starts no model turn (#576).

Moonshine can finish a short word without ever raising a partial, so the streaming-evidence
rule refused a spoken "Stop" during playback. Only that exact single word is relieved of the
partial-evidence requirement, and only after echo promotion with an explicit verifier that
confirmed speech, when no transcript in the capture matched outbound text, and when no
recently played assistant text contains the word. Every other short final without evidence
stays refused (R2-14).

However it was admitted, an exact stop word captured during playback is speech-only (owner
decision): it stops the current speech exactly as any barge-in does, is shown as the user's
final, and is then consumed. It reaches no command router and no model turn, so it can never
become a cancel selection; background tasks are untouched. Outside playback it is an ordinary
turn.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationSessionWorker,
    ConversationTaskCommandRouter,
    ConversationTaskController,
    ConversationUpdateDirector,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation import worker as worker_module
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.providers.moonshine import MoonshineStreamingTranscriber
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    SpeechPresence,
    Transcript,
    VoiceActivity,
)
from tests.conversation.test_actions import RecordingInference, Synthesizer
from tests.conversation.test_deferred_cancel_note import _GatedPlayback
from tests.conversation.test_idle_cancel_note import _until
from tests.conversation.test_tasks import QueuedUpdateSession
from tests.conversation.test_worker import (
    ScriptedEchoGuard,
    ScriptedSpeechPresenceVerifier,
    ScriptedVad,
    action_executor_probe,
)
from tests.providers.test_moonshine import ScriptedMoonshineBackend

_FRAMES = tuple(
    AudioFrame(pcm=bytes([i, 0]) * 480, sample_rate_hz=48_000, channels=1) for i in (3, 4)
)
_REFUSAL_PREFIX = "[playback-final-refused] "


async def _capture(
    final: str,
    *partials: str,
    verify: bool = True,
    presence: SpeechPresence = SpeechPresence.CONFIRMED_SPEECH,
    endpoint_echo: bool = False,
    echoes: tuple[str, ...] = (),
    outbound_words: tuple[str, ...] = (),
    foreground: bool = True,
    recent_playback: bool = True,
) -> tuple[list[str], bool, list[str]]:
    actions, cancelled, responses = action_executor_probe()
    actions._speech.foreground_active = foreground  # type: ignore[attr-defined]
    actions._speech.active_turn_id = "foreground_1" if foreground else None  # type: ignore[attr-defined]
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="participant",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(
            backend=ScriptedMoonshineBackend(*partials, final=final), max_add_bytes=960
        ),
        actions=actions,
        echo_guard=ScriptedEchoGuard(
            False,
            endpoint_echo,
            recent_playback=recent_playback,
            transcript_echoes=echoes,
            outbound_words=outbound_words,
        ),
        speech_presence_verifier=(
            ScriptedSpeechPresenceVerifier(presence) if verify else None
        ),
        observer=lambda kind, _data: observed.append(kind),
    )
    try:
        worker.start()
        for frame in _FRAMES:
            await worker.receive_audio("participant", 1, frame)
        await worker.wait_for_responses()
        interrupted = cancelled.is_set()  # before close, which ends the foreground
        return [transcript.text for _turn, transcript in responses], interrupted, observed
    finally:
        await worker.close()


def _refusals(output: str) -> list[dict[str, object]]:
    return [
        json.loads(line[len(_REFUSAL_PREFIX) :])
        for line in output.splitlines()
        if line.startswith(_REFUSAL_PREFIX)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["Stop.", "stop", "STOP!", " Stop ", "Stop?", "stop!!"])
async def test_final_only_stop_word_interrupts_playback_and_starts_no_turn(
    final: str, capsys: pytest.CaptureFixture[str]
) -> None:
    texts, interrupted, observed = await _capture(final)

    assert texts == []
    assert interrupted is True
    assert observed.count("interrupt_requested") == 1
    assert "playback_silenced" in observed
    assert observed.count("transcript_final") == 1
    # Admitted, so it is not reported as missing evidence.
    assert _refusals(capsys.readouterr().out) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["St", "No"])
async def test_a_subthreshold_partial_neither_helps_nor_blocks_the_stop_word(partial: str) -> None:
    texts, interrupted, observed = await _capture("Stop.", partial)

    assert texts == []
    assert interrupted is True
    assert observed.count("transcript_final") == 1


@pytest.mark.asyncio
async def test_stop_word_with_streaming_evidence_is_speech_only_too() -> None:
    # The partial path: "Stop" is evidence enough on its own and confirms the barge-in.
    texts, interrupted, observed = await _capture("Stop.", "Stop")

    assert texts == []
    assert interrupted is True
    assert observed.count("interrupt_requested") == 1
    assert observed.count("transcript_final") == 1


@pytest.mark.asyncio
async def test_stop_word_in_the_playback_tail_starts_no_turn() -> None:
    texts, interrupted, observed = await _capture("Stop.", foreground=False)

    assert texts == []
    assert interrupted is False
    assert "interrupt_requested" not in observed
    assert observed.count("transcript_final") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["Stop.", "stop"])
async def test_stop_word_outside_playback_is_an_ordinary_turn(final: str) -> None:
    texts, interrupted, observed = await _capture(final, foreground=False, recent_playback=False)

    assert texts == [final]
    assert interrupted is False
    assert observed.count("transcript_final") == 1


@pytest.mark.asyncio
async def test_a_longer_final_starting_with_stop_is_unchanged_during_playback() -> None:
    texts, interrupted, observed = await _capture("Stop that task.", "Stop that")

    assert texts == ["Stop that task."]
    assert interrupted is True
    assert observed.count("transcript_final") == 1


async def _assert_refused(
    capsys: pytest.CaptureFixture[str], *, reaches_stt: bool = True, **kwargs: object
) -> None:
    final = kwargs.pop("final", "Stop.")
    partials = kwargs.pop("partials", ())
    assert type(final) is str and type(partials) is tuple
    texts, interrupted, observed = await _capture(final, *partials, **kwargs)  # type: ignore[arg-type]

    assert texts == []
    assert interrupted is False
    assert "interrupt_requested" not in observed
    assert "transcript_final" not in observed
    refusals = _refusals(capsys.readouterr().out)
    if reaches_stt:
        assert [row["category"] for row in refusals] == ["streaming_evidence_missing"]
    else:
        assert refusals == []


@pytest.mark.asyncio
async def test_echo_dominated_endpoint_still_refuses_the_stop_word(
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _assert_refused(capsys, endpoint_echo=True, reaches_stt=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "presence", [SpeechPresence.UNCERTAIN, SpeechPresence.CONFIRMED_NON_SPEECH]
)
async def test_unconfirmed_speech_still_refuses_the_stop_word(
    presence: SpeechPresence, capsys: pytest.CaptureFixture[str]
) -> None:
    await _assert_refused(capsys, presence=presence, reaches_stt=False)


@pytest.mark.asyncio
async def test_no_explicit_verifier_still_refuses_the_stop_word(
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _assert_refused(capsys, verify=False)


@pytest.mark.asyncio
async def test_a_text_echo_in_the_same_capture_refuses_the_stop_word(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The capture's own partial matched outbound text (long enough for the real guard's
    # text match): its final is not trusted as the user's.
    echo = "yes i can switch models"
    await _assert_refused(capsys, partials=(echo,), echoes=(echo,))


@pytest.mark.asyncio
async def test_recently_played_stop_word_refuses_the_final_only_stop_word(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Self-interruption: the assistant's own "stop" could leak back as a partial-less final.
    await _assert_refused(capsys, outbound_words=("stop",))


@pytest.mark.asyncio
async def test_recently_played_stop_word_does_not_affect_the_partial_path() -> None:
    texts, interrupted, observed = await _capture("Stop.", "Stop", outbound_words=("stop",))

    assert texts == []
    assert interrupted is True
    assert observed.count("transcript_final") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final",
    [
        # Other short words: the closed set is the single word "stop".
        "No.", "Go.", "Wait.", "Halt.", "Chapter 3.", "Stops.", "Stopp.", "Sto.",
        # More than one token, even when one of them is "stop".
        "Don't stop.", "Stop now.", "Stop stop.", "Stop, please.", "S top.", "Stop-stop.",
        # Punctuation outside the fixed ASCII set is not stripped.
        "¡Stop!", "“Stop.”",
    ],
)
async def test_any_other_final_without_evidence_is_still_refused(
    final: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _assert_refused(capsys, final=final)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Stop.", True), (" stop ", True), ("\tSTOP!\n", True), ("'Stop.'", True),
        ("stop...", True), ("Stop stop", False), ("st op", False), ("ſtop", False),
        ("Ｓtop", False), ("", False), ("Stops", False),
    ],
)
def test_stop_word_normalisation_is_one_ascii_token(text: str, expected: bool) -> None:
    assert worker_module._is_playback_stop_word(text) is expected


@pytest.mark.asyncio
async def test_refusal_and_admission_evidence_stays_content_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _capture("Stop.")
    await _capture("Stop PRIVATE-BAIT.")
    output = capsys.readouterr().out
    assert _refusals(output) == [{
        "version": 1, "category": "streaming_evidence_missing",
        "short_partial_chars": 0, "final_chars": 18, "speech_verifier": True,
    }]
    assert "PRIVATE-BAIT" not in output


class _CancellableEndpointTranscriber:
    """First capture: one text-echo partial, then an endpoint that never finishes."""

    def __init__(self, echo: str) -> None:
        self._pushes: list[tuple[Transcript, ...]] = [
            (Transcript(text=echo, final=False),), (), (), (),
        ]
        self._finals = iter((None, Transcript(text="Stop.", final=True)))
        self.first_endpoint = asyncio.Event()

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        return self._pushes.pop(0)

    async def finish_utterance(self) -> Transcript | None:
        final = next(self._finals)
        if final is None:
            self.first_endpoint.set()
            await asyncio.Event().wait()
        return final

    async def cancel(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_text_echo_from_an_unfinished_capture_does_not_refuse_the_next_stop_word() -> None:
    # The text-echo latch is reset at speech onset, not only at the endpoint: a caller that
    # cancels the endpoint leaves the worker usable, and the next capture starts clean.
    echo = "yes i can switch models"
    actions, cancelled, responses = action_executor_probe()
    observed: list[str] = []
    stt = _CancellableEndpointTranscriber(echo)
    worker = ConversationSessionWorker(
        participant_identity="participant",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED,
            VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED,
        ),
        stt=stt,
        actions=actions,
        echo_guard=ScriptedEchoGuard(False, False, False, False, transcript_echoes=(echo,)),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(
            SpeechPresence.CONFIRMED_SPEECH, SpeechPresence.CONFIRMED_SPEECH
        ),
        observer=lambda kind, _data: observed.append(kind),
    )
    try:
        worker.start()
        await worker.receive_audio("participant", 1, _FRAMES[0])
        endpoint = asyncio.create_task(worker.receive_audio("participant", 1, _FRAMES[1]))
        await asyncio.wait_for(stt.first_endpoint.wait(), timeout=2)
        endpoint.cancel()
        with pytest.raises(asyncio.CancelledError):
            await endpoint
        assert "transcript_echo_suppressed" in observed
        assert cancelled.is_set() is False

        for frame in _FRAMES:
            await worker.receive_audio("participant", 1, frame)
        await worker.wait_for_responses()

        assert cancelled.is_set() is True
        assert observed.count("transcript_final") == 1
        assert responses == []
    finally:
        await worker.close()


# --- Production wiring: a running background task survives a spoken "Stop." -------------

_TASK_TEXT = "task: Keep working on the synthetic background task."


class _BargeInPlayback(_GatedPlayback):
    """Held playback that also accepts the barge-in's finish-the-word request."""

    def __init__(self) -> None:
        super().__init__()
        self.barge_in_turn_ids: list[str] = []

    def finish_word_on_cancel(self, turn_id: str) -> None:
        self.barge_in_turn_ids.append(turn_id)


@dataclass
class _Stack:
    worker: ConversationSessionWorker
    executor: ConversationUpdateExecutor
    context: ConversationContextStore
    session: QueuedUpdateSession
    inference: RecordingInference
    playback: _BargeInPlayback
    states: list[dict[str, object]]
    sequence: itertools.count[int]

    async def say(self, text: str) -> None:
        await self.worker.submit_final_transcript(
            participant_identity="browser_user", session_generation=7,
            typed_sequence=next(self.sequence), text=text,
        )

    async def speak(self) -> None:
        for frame in _FRAMES:
            await self.worker.receive_audio("browser_user", 7, frame)


@asynccontextmanager
async def _stack(*partials: str, final: str) -> AsyncIterator[_Stack]:
    context = ConversationContextStore()
    session = QueuedUpdateSession()
    counter = itertools.count(1)
    controller = ConversationTaskController(
        context=context, session=session, session_id="session_001",
        id_factory=lambda: f"{next(counter):04d}",
    )
    states: list[dict[str, object]] = []

    def observe(kind: str, data: dict[str, object]) -> None:
        if kind == "task_state":
            states.append(dict(data))

    surface = ConversationWorkControlSurface(
        controller=controller, context=context, observer=observe,
        reserve_observer_capacity=lambda: None,
    )
    inference = RecordingInference()
    playback = _BargeInPlayback()
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=Synthesizer(), playback=playback, ledger=DeliveredSpeechLedger(),
    )

    async def record_command_input(text: str) -> None:
        await speech.record_user_input(Transcript(text=text, final=True))

    router = ConversationTaskCommandRouter(
        surface=surface, observer=observe, reserve_observer_capacity=lambda: None,
        validate_user_input=context.validate_user_text, record_user_input=record_command_input,
    )
    director = ConversationUpdateDirector(
        context=context, completions=controller,
        policy=ProactiveCompletionUpdatePolicy(
            projection=BrowserEventProjection(),
            floor_available=lambda: not speech.foreground_active,
        ),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    worker = ConversationSessionWorker(
        participant_identity="browser_user", session_generation=7,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=MoonshineStreamingTranscriber(
            backend=ScriptedMoonshineBackend(*partials, final=final), max_add_bytes=960
        ),
        actions=executor, command_router=router,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH),
    )
    worker.start()
    try:
        yield _Stack(
            worker, executor, context, session, inference, playback, states,
            itertools.count(1),
        )
    finally:
        playback.release.set()
        await worker.close()
        await surface.close()
        await director.close()
        await controller.close()


def _user_messages(context: ConversationContextStore) -> list[str]:
    return [
        message.text for message in context.snapshot().messages if message.role == "user"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partials", [(), ("Stop",)], ids=["no-partial", "streaming-partial"]
)
async def test_spoken_stop_during_playback_leaves_a_running_task_running(
    partials: tuple[str, ...],
) -> None:
    async with _stack(*partials, final="Stop.") as stack:
        await stack.say(_TASK_TEXT)
        await _until(lambda: bool(stack.context.snapshot().active_tasks), "task_active")
        (task,) = stack.context.snapshot().active_tasks

        stack.playback.armed = True
        await stack.say("Tell me something while that runs.")
        await asyncio.wait_for(stack.playback.started.wait(), timeout=2)
        assert stack.executor.foreground_active
        assert len(stack.inference.requests) == 1

        await stack.speak()
        await stack.worker.wait_for_responses()

        # Speech stopped exactly as a barge-in stops it.
        assert stack.executor.foreground_active is False
        assert len(stack.playback.barge_in_turn_ids) == 1
        # No model turn, so no cancel selection; the task was never touched.
        assert len(stack.inference.requests) == 1
        assert stack.context.snapshot().active_tasks == (task,)
        assert stack.session.cancel_calls == 0
        assert not any(state.get("status") == "cancelling" for state in stack.states)
        assert "Stop." not in _user_messages(stack.context)

        # Back to listening: the next utterance is an ordinary turn.
        await stack.say("What were you saying?")
        await stack.worker.wait_for_responses()
        assert len(stack.inference.requests) == 2
        assert stack.context.snapshot().active_tasks == (task,)


