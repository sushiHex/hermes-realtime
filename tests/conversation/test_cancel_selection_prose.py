"""Cancel-capable turns quarantine model prose until the terminal selection settles."""
from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from hermes_realtime.conversation.natural_cancel import HostCancelRequest
from hermes_realtime.conversation.streaming import AcceptedCancelInferenceRequest
from tests.conversation.test_confirmed_multi_cancel import _owner, _say, _settle

PREFIX = "Done, I have stopped that background task and nothing remains to cancel."
ORDINARY = "These are the relevant considerations for your ordinary question."
ACK = "The cancellation request was accepted and the task is stopping."


class HeldStream:
    def __init__(self, texts, selection, *, failure=None, close_failure=False, hold_close=False):
        self.texts = iter(texts)
        self.selection = selection
        self.failure = failure
        self.close_failure = close_failure
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.resolved = False
        self.close_waiting = asyncio.Event()
        self.close_release = asyncio.Event()
        if not hold_close:
            self.close_release.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        text = next(self.texts, None)
        if text is not None:
            return text
        if self.resolved:
            raise StopAsyncIteration
        self.waiting.set()
        await self.release.wait()
        self.resolved = True
        if self.failure is not None:
            raise self.failure
        if self.selection is not None:
            return self.selection
        raise StopAsyncIteration

    async def aclose(self):
        self.close_waiting.set()
        await self.close_release.wait()
        if self.close_failure:
            raise ValueError("scripted provider close failure")


class Inference:
    def __init__(self, texts=(PREFIX,), selection=None, **kwargs):
        self.initial = HeldStream(texts, selection, **kwargs)

    def stream(self, snapshot, *, turn_id):
        if type(snapshot) is AcceptedCancelInferenceRequest:
            async def acknowledgment():
                yield ACK
            return acknowledgment()
        return self.initial

    async def cancel(self, turn_id):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("names,selection,expected", [
    (("alpha",), HostCancelRequest(), ACK),
    (("alpha", "beta"), HostCancelRequest(target="choice_2"),
     "Stop background task. Inspect beta?"),
])
async def test_cancel_selection_discards_prose_before_any_publication(names, selection, expected):
    inference = Inference(selection=selection)
    loop, _, controller, proposals, synth, _ = _owner(*names, inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "Stop that task.", "cancel_turn"))
    try:
        await inference.initial.waiting.wait()
        await _settle()
        assert synth.texts == []
        assert controller.cancellations == []
        inference.initial.release.set()
        await turn
        assert synth.texts == [expected]
        assert controller.cancellations == (
            [("task_alpha", "user requested task cancellation")] if len(names) == 1 else [])
        assert proposals.has_pending is (len(names) == 2)
    finally:
        inference.initial.release.set()
        with suppress(asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
async def test_cancel_capable_ordinary_reply_waits_for_clean_close_then_keeps_text():
    inference = Inference(texts=(ORDINARY, "A second ordinary sentence follows."))
    loop, _, controller, proposals, synth, _ = _owner("alpha", inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "An ordinary question.", "ordinary"))
    try:
        await inference.initial.waiting.wait()
        await _settle()
        assert synth.texts == []
        inference.initial.release.set()
        await turn
        assert synth.texts == [ORDINARY, "A second ordinary sentence follows."]
        assert controller.cancellations == [] and not proposals.has_pending
    finally:
        inference.initial.release.set()
        with suppress(asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,close_failure", [(ValueError("iteration failed"), False),
                                                    (None, True)])
async def test_failed_initial_stream_never_releases_quarantined_prose(failure, close_failure):
    inference = Inference(failure=failure, close_failure=close_failure)
    loop, _, controller, proposals, synth, _ = _owner("alpha", inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "Stop it.", "failed"))
    try:
        await inference.initial.waiting.wait()
        await _settle()
        inference.initial.release.set()
        with pytest.raises(ValueError):
            await turn
        assert synth.texts == [] and controller.cancellations == []
    finally:
        inference.initial.release.set()
        with suppress(ValueError, asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["stale", "cancelled"])
async def test_withdrawn_turn_never_releases_quarantined_prose(withdrawal, capsys):
    inference = Inference()
    loop, _, controller, proposals, synth, _ = _owner("alpha", inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "Stop it.", "withdrawn"))
    try:
        await inference.initial.waiting.wait()
        await _settle()
        if withdrawal == "stale":
            loop._foreground._is_current = lambda _lease: False
        else:
            await loop.cancel()
        inference.initial.release.set()
        with suppress(asyncio.CancelledError):
            await turn
        assert synth.texts == [] and controller.cancellations == []
        if withdrawal == "stale":
            import json
            prefix = "[cancel-prose-refused] "
            rows = [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
                    if line.startswith(prefix)]
            assert rows == [{"version": 1, "category": "stale_turn", "buffered_segments": 1}]
    finally:
        inference.initial.release.set()
        with suppress(asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
async def test_turn_without_a_cancel_offer_keeps_streaming():
    inference = Inference(texts=(ORDINARY,))
    loop, _, _, proposals, synth, _ = _owner(inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "An ordinary question.", "unoffered"))
    try:
        await inference.initial.waiting.wait()
        await _settle()
        assert synth.texts == [ORDINARY]
        inference.initial.release.set()
        await turn
    finally:
        inference.initial.release.set()
        with suppress(asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
async def test_quarantine_retains_existing_segment_capacity():
    inference = Inference(texts=(ORDINARY, "Capacity overflow."))
    inference.initial.release.set()
    loop, _, controller, proposals, synth, _ = _owner("alpha", inference=inference)
    loop._max_segments = 1
    try:
        with pytest.raises(ValueError, match="segment count capacity"):
            await _say(loop, proposals, "An ordinary question.", "bounded")
        assert synth.texts == [] and controller.cancellations == []
    finally:
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("short", [False, True])
async def test_ordinary_text_is_not_published_while_stream_close_is_pending(short):
    text = "Brief." if short else ORDINARY
    inference = Inference(texts=(text,), hold_close=True)
    inference.initial.release.set()
    loop, _, _, proposals, synth, _ = _owner("alpha", inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "An ordinary question.", "closing"))
    try:
        await inference.initial.close_waiting.wait()
        await _settle()
        assert synth.texts == []
        inference.initial.close_release.set()
        await turn
        assert synth.texts == [text]
    finally:
        inference.initial.release.set()
        inference.initial.close_release.set()
        with suppress(asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
async def test_coalescing_fallback_cannot_publish_a_model_prefix_before_host_status():
    from hermes_realtime.conversation.task_status import HostSpeechRequest

    inference = Inference(texts=("Brief.", HostSpeechRequest()), hold_close=True)
    inference.initial.release.set()
    loop, _, _, proposals, synth, _ = _owner("alpha", inference=inference)
    turn = asyncio.create_task(_say(loop, proposals, "How is the task?", "status"))
    try:
        await inference.initial.close_waiting.wait()
        await _settle()
        assert synth.texts == []
        inference.initial.close_release.set()
        await turn
        assert synth.texts[0] == "Brief."
        assert len(synth.texts) == 2
    finally:
        inference.initial.release.set()
        inference.initial.close_release.set()
        with suppress(asyncio.CancelledError):
            await turn
        await loop.close()


@pytest.mark.asyncio
async def test_quarantine_retains_existing_segment_character_capacity():
    inference = Inference(texts=("x" * 101,))
    inference.initial.release.set()
    loop, _, controller, proposals, synth, _ = _owner(
        "alpha", inference=inference, max_segment_chars=100)
    try:
        with pytest.raises(ValueError, match="segment character capacity"):
            await _say(loop, proposals, "An ordinary question.", "bounded_chars")
        assert synth.texts == [] and controller.cancellations == []
    finally:
        await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("write_fails", "close_failure"),
                         [(False, False), (True, False), (False, True)])
async def test_stale_quarantine_reports_once_without_text_or_changing_outcome(
    monkeypatch, capsys, write_fails, close_failure,
):
    import builtins
    import json

    def texts():
        yield PREFIX
        yield ORDINARY
        # Deliberately stub only the observational lease predicate. Atomic owner
        # identity remains current, isolating the two new stale-quarantine gates.
        loop._foreground._is_current = lambda _lease: False
        yield "This segment must be refused, not buffered or published."

    inference = Inference(texts=texts(), close_failure=close_failure)
    loop, _, controller, proposals, synth, _ = _owner("alpha", inference=inference)
    rows = []
    original_print = builtins.print

    def print_diagnostic(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("[cancel-prose-refused] "):
            rows.append(json.loads(args[0].split("] ", 1)[1]))
            if write_fails:
                raise OSError("diagnostic output unavailable")
        return original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", print_diagnostic)
    try:
        if close_failure:
            with pytest.raises(ValueError, match="scripted provider close failure"):
                await _say(loop, proposals, "An ordinary question.", "observational_stub")
        else:
            await _say(loop, proposals, "An ordinary question.", "observational_stub")
        assert rows == [{"version": 1, "category": "stale_turn", "buffered_segments": 2}]
        assert synth.texts == [] and controller.cancellations == []
        assert not proposals.has_pending
        output = capsys.readouterr().out
        assert output.count("[cancel-prose-refused] ") == (0 if write_fails else 1)
        assert PREFIX not in output and ORDINARY not in output
        assert "observational_stub" not in output and "task_alpha" not in output
    finally:
        await loop.close()
