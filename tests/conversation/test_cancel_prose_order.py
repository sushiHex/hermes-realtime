"""Prose streamed before a cancel selection (#582): pinned as it is, not held.

A provider that streams (Codex) can emit prose before it selects ``cancel_work``. That prose
is published as it arrives and cannot be withdrawn; no stream hold is applied to ordinary
replies (#588 was rejected for delaying every reply while work runs). A short first segment
still held for coalescing is dropped when the selection arrives. With several running tasks
the selection cancels nothing and the host's own question follows; with one task the host's
fixed refusal or the fact-grounded acknowledgment follows.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_realtime.conversation import ForegroundTurnCoordinator, StreamingSpeechLoop
from hermes_realtime.conversation.context import ConversationContextStore
from hermes_realtime.conversation.natural_cancel import CANCEL_NOT_ACCEPTED, HostCancelRequest
from hermes_realtime.conversation.streaming import (
    AcceptedCancelInferenceRequest,
    ConversationInferenceRequest,
)
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_confirmed_multi_cancel import _READBACK_A, _owner, _say, _settle
from tests.conversation.test_natural_cancel import _ACK, _accept, _surface
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer

_PROSE = "Okay, let me take care of that for you now."
_SHORT = "Sure."


class GatedProse:
    """Yields the scripted items in order; an ``asyncio.Event`` item holds the stream open
    until it is set. An accepted-cancel continuation yields the acknowledgment."""

    def __init__(self, *script: object) -> None:
        self.calls: list[object] = []
        self.held = asyncio.Event()
        self._script = script

    async def stream(self, snapshot, *, turn_id):
        self.calls.append(snapshot)
        if type(snapshot) is AcceptedCancelInferenceRequest:
            yield _ACK
            return
        assert type(snapshot) is ConversationInferenceRequest
        for item in self._script:
            if type(item) is asyncio.Event:
                self.held.set()
                await item.wait()
                continue
            yield item

    async def cancel(self, turn_id):
        pass


def _sole(*, cancel_accepted: bool, inference: GatedProse):
    context = ConversationContextStore()
    _accept(context, "a")
    surface, controller, _events = _surface(context, cancel_accepted=cancel_accepted)
    synth = SegmentSynthesizer()
    playback = RecordingPlayback()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=synth,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        task_cancellation=surface,
    )
    return loop, surface, controller, synth, playback


# ------------------------------------------------------------------ A: several tasks


@pytest.mark.asyncio
async def test_prose_before_a_nomination_is_spoken_then_the_hosts_own_question() -> None:
    inference = GatedProse(_PROSE, HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop the alpha one.", "turn_1")
    await _settle()
    assert inference.calls[0].cancel_choices != ()
    assert synth.texts == [_PROSE, _READBACK_A]
    assert controller.cancellations == []
    assert proposals.has_pending


# ---------------------------------------------------------------------- B: one task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accepted", "follow_up", "calls"),
    [(True, _ACK, 2), (False, CANCEL_NOT_ACCEPTED, 1)],
    ids=["accepted", "refused"],
)
async def test_prose_before_a_sole_selection_is_followed_by_the_host_controlled_outcome(
    accepted: bool, follow_up: str, calls: int,
) -> None:
    inference = GatedProse(_PROSE, HostCancelRequest())
    loop, surface, controller, synth, _playback = _sole(
        cancel_accepted=accepted, inference=inference,
    )
    try:
        await loop.respond("turn_stop", Transcript(text="Stop that.", final=True))
        assert inference.calls[0].cancel_offered is True
        assert controller.cancellations == [("task_a", "user requested task cancellation")]
        assert synth.texts == [_PROSE, follow_up]
        assert len(inference.calls) == calls
    finally:
        await loop.close()
        await surface.close()


# --------------------------------------------------------- C: short first segment


@pytest.mark.asyncio
async def test_a_short_first_segment_is_dropped_when_a_nomination_arrives() -> None:
    inference = GatedProse(_SHORT, HostCancelRequest(target="choice_1"))
    loop, _context, controller, proposals, synth, _events = _owner(
        "alpha", "beta", inference=inference,
    )
    await _say(loop, proposals, "Stop the alpha one.", "turn_1")
    await _settle()
    assert synth.texts == [_READBACK_A]
    assert controller.cancellations == []


@pytest.mark.asyncio
async def test_a_short_first_segment_is_dropped_when_a_sole_selection_arrives() -> None:
    inference = GatedProse(_SHORT, HostCancelRequest())
    loop, surface, _controller, synth, _playback = _sole(cancel_accepted=True, inference=inference)
    try:
        await loop.respond("turn_stop", Transcript(text="Stop that.", final=True))
        assert synth.texts == [_ACK]
    finally:
        await loop.close()
        await surface.close()


# ------------------------------------------------------ D: no hold on ordinary replies


@pytest.mark.asyncio
async def test_an_ordinary_reply_is_spoken_while_the_stream_is_open_when_cancel_is_offered():
    """Regression guard against the rejected #588 hold."""
    gate = asyncio.Event()
    first = "The build is still running and should finish soon."
    second = "I will tell you when it does."
    inference = GatedProse(first, gate, second)
    loop, surface, controller, synth, playback = _sole(cancel_accepted=True, inference=inference)
    response = asyncio.create_task(
        loop.respond("turn_ask", Transcript(text="How is it going?", final=True))
    )
    try:
        await asyncio.wait_for(inference.held.wait(), timeout=1)
        await asyncio.wait_for(playback.first_chunk_played.wait(), timeout=1)
        assert inference.calls[0].cancel_offered is True
        assert not response.done()
        assert synth.texts == [first]
        gate.set()
        await asyncio.wait_for(response, timeout=1)
        assert synth.texts == [first, second]
        assert controller.cancellations == []
    finally:
        gate.set()
        if not response.done():
            response.cancel()
        await loop.close()
        await surface.close()
