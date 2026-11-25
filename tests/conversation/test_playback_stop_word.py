"""A final-only exact "stop" during playback is admitted under its acoustic gates (#576).

Moonshine can finish a short word without ever raising a partial, so the streaming-evidence
rule refused a spoken "Stop" during playback. Only that exact single word is relieved of the
partial-evidence requirement, and only after echo promotion with an explicit verifier that
confirmed speech, and only when no transcript in the capture matched outbound text. Every
other short final without evidence stays refused (R2-14).
"""

from __future__ import annotations

import json

import pytest

from hermes_realtime.conversation import ConversationSessionWorker
from hermes_realtime.conversation import worker as worker_module
from hermes_realtime.providers.moonshine import MoonshineStreamingTranscriber
from hermes_realtime.speech import AudioFrame, SpeechPresence, VoiceActivity
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
    foreground: bool = True,
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
            False, endpoint_echo, recent_playback=True, transcript_echoes=echoes
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
async def test_final_only_stop_word_interrupts_playback(
    final: str, capsys: pytest.CaptureFixture[str]
) -> None:
    texts, interrupted, observed = await _capture(final)

    assert texts == [final.strip()]  # the Moonshine wrapper trims its final
    assert interrupted is True
    assert observed.count("interrupt_requested") == 1
    assert "playback_silenced" in observed
    assert observed.count("transcript_final") == 1
    # Admitted, so it is not reported as missing evidence.
    assert _refusals(capsys.readouterr().out) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["St", "No"])
async def test_a_subthreshold_partial_neither_helps_nor_blocks_the_stop_word(partial: str) -> None:
    texts, interrupted, _observed = await _capture("Stop.", partial)

    assert texts == ["Stop."]
    assert interrupted is True


@pytest.mark.asyncio
async def test_stop_word_in_the_playback_tail_is_admitted_without_a_foreground() -> None:
    texts, interrupted, observed = await _capture("Stop.", foreground=False)

    assert texts == ["Stop."]
    assert interrupted is False
    assert "interrupt_requested" not in observed


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
    # The capture's own partial matched outbound text: its final is not trusted as the user's.
    await _assert_refused(capsys, partials=("tell me to stop",), echoes=("tell me to stop",))


@pytest.mark.asyncio
async def test_a_text_echo_final_is_still_suppressed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    texts, interrupted, observed = await _capture("Stop.", echoes=("Stop.",))

    assert texts == [] and interrupted is False
    assert "transcript_echo_suppressed" in observed
    assert _refusals(capsys.readouterr().out) == []


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
