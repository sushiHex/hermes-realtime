"""Bounded consumer exception evidence without changing terminal or cleanup behavior."""

import asyncio
import json

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer


def consumer_markers(output):
    prefix = "[foreground-failure] "
    return [value for line in output.splitlines() if line.startswith(prefix)
            if (value := json.loads(line[len(prefix):]))["stage"] == "consumer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_pretext_consumer_error_retains_original_and_cleanup_outcome(
    monkeypatch, capsys, logging_fails, cleanup_fails,
):
    import builtins

    class EmptyInference:
        async def stream(self, snapshot, *, turn_id):
            if False:
                yield "unreachable"

        async def cancel(self, turn_id):
            pass

    error = ValueError("PRIVATE transcript https://private.invalid/?token=SECRET")
    cleanup_error = RuntimeError("PRIVATE cleanup")
    ledger = DeliveredSpeechLedger()
    synthesizer = SegmentSynthesizer()
    events = []

    def failed_counts(turn_id):
        raise error

    async def failed_cancel(turn_id):
        raise cleanup_error

    monkeypatch.setattr(ledger, "snapshot_turn_counts", failed_counts)
    if cleanup_fails:
        monkeypatch.setattr(synthesizer, "cancel", failed_cancel)
    original_print = builtins.print

    def failed_print(*args, **kwargs):
        if args and str(args[0]).startswith("[foreground-failure] "):
            raise OSError("PRIVATE diagnostic failure")
        return original_print(*args, **kwargs)

    if logging_fails:
        monkeypatch.setattr(builtins, "print", failed_print)
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(), foreground=ForegroundTurnCoordinator(),
        inference=EmptyInference(), synthesizer=synthesizer, playback=RecordingPlayback(),
        ledger=ledger, observer=lambda kind, data: events.append((kind, data)),
    )
    try:
        with pytest.raises(BaseException) as caught:
            await loop._run_turn(
                "synthetic_consumer", transcript=Transcript("Synthetic input", final=True),
                announcement=None, updates=(), update_operation=None, typed_input=(3, 7),
            )
        if cleanup_fails:
            assert isinstance(caught.value, BaseExceptionGroup)
            assert caught.value.exceptions == (error, cleanup_error)
        else:
            assert caught.value is error
        assert not any(kind in {"first_foreground_token", "assistant_text_generated"}
                       for kind, _ in events)
        assert [data for kind, data in events if kind == "assistant_turn_interrupted"] == [{
            "turnId": "synthetic_consumer", "turnGeneration": 1,
            "inputSequence": 3, "bindingGeneration": 7,
        }]
        assert ledger.pending() == () and ledger.retained_chunk_count == 0
        output = capsys.readouterr().out
        assert consumer_markers(output) == ([] if logging_fails else [{
            "version": 1, "stage": "consumer", "failure_kind": "value",
            "text_phase": "before_publication",
        }])
        assert "PRIVATE" not in output and "SECRET" not in output
    finally:
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_publication", [False, True])
async def test_consumer_cancellation_is_not_classified_as_provider_failure(
    capsys, after_publication,
):
    entered = asyncio.Event()
    published = asyncio.Event()

    class WaitingInference:
        async def stream(self, snapshot, *, turn_id):
            entered.set()
            if after_publication:
                yield "Synthetic publication with enough words to pass the initial speech buffer."
            await asyncio.Future()
            yield "unreachable"

        async def cancel(self, turn_id):
            pass

    loop = StreamingSpeechLoop(
        context=ConversationContextStore(), foreground=ForegroundTurnCoordinator(),
        inference=WaitingInference(), synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(), ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: published.set() if kind == "assistant_text_generated" else None,
    )
    response = asyncio.create_task(loop.respond(
        "synthetic_cancel", Transcript("Synthetic input", final=True)))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if after_publication:
            await asyncio.wait_for(published.wait(), 1)
        await loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response
        assert consumer_markers(capsys.readouterr().out) == [{
            "version": 1, "stage": "consumer", "failure_kind": "cancelled",
            "text_phase": "after_publication" if after_publication else "before_publication",
        }]
    finally:
        if not response.done():
            response.cancel()
        await asyncio.gather(response, return_exceptions=True)
        await loop.close()


@pytest.mark.asyncio
async def test_consumer_phase_is_frozen_before_cleanup_allows_a_late_publication(
    monkeypatch, capsys,
):
    release = asyncio.Event()
    published = asyncio.Event()
    error = ValueError("PRIVATE consumer failure")

    class LateInference:
        async def stream(self, snapshot, *, turn_id):
            await release.wait()
            yield "Synthetic late publication with enough words to pass the initial speech buffer."

        async def cancel(self, turn_id):
            pass

    synthesizer = SegmentSynthesizer()

    async def late_cancel(turn_id):
        release.set()
        await asyncio.wait_for(published.wait(), 1)

    async def failed_next(*args):
        raise error

    monkeypatch.setattr(synthesizer, "cancel", late_cancel)
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(), foreground=ForegroundTurnCoordinator(),
        inference=LateInference(), synthesizer=synthesizer, playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: published.set() if kind == "assistant_text_generated" else None,
    )
    monkeypatch.setattr(loop, "_next_prefetched_speech", failed_next)
    try:
        with pytest.raises(ValueError) as caught:
            await loop.respond("synthetic_late_publication", Transcript("Synthetic", final=True))
        assert caught.value is error and published.is_set()
        assert consumer_markers(capsys.readouterr().out) == [{
            "version": 1, "stage": "consumer", "failure_kind": "value",
            "text_phase": "before_publication",
        }]
    finally:
        release.set()
        await loop.close()
