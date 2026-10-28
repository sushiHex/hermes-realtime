import asyncio

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.streaming import AcceptedStartInferenceRequest
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
from tests.conversation.test_work_tools import _surface


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_automatic_ack_follows_real_dispatch_ack(accepted):
    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, dispatch_gate=gate, accepted=accepted)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    calls = []

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            calls.append(snapshot)
            if len(calls) == 1:
                yield HostTaskProposalRequest("Inspect the release")
            else:
                yield "I have started the work."

        async def cancel(self, turn_id):
            pass

    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    operation = asyncio.create_task(
        loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
    )
    try:
        await controller.dispatch_started.wait()
        assert len(calls) == 1 and synth.texts == []
        gate.set()
        await operation
        assert len(calls) == (2 if accepted else 1)
        if accepted:
            assert type(calls[1]) is AcceptedStartInferenceRequest
            assert calls[1].task_id == "task_release_check"
        assert synth.texts == (["I have started the work."] if accepted else [])
        assert controller.cancellations == []
    finally:
        gate.set()
        await surface.close()


@pytest.mark.asyncio
async def test_barge_in_while_waiting_does_not_cancel_owned_dispatch():
    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, dispatch_gate=gate)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    calls = []

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            calls.append(snapshot)
            yield HostTaskProposalRequest("Inspect the release")

        async def cancel(self, turn_id):
            pass

    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    operation = asyncio.create_task(
        loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
    )
    try:
        await controller.dispatch_started.wait()
        await loop.cancel()
        gate.set()
        await surface.close()
        await asyncio.gather(operation, return_exceptions=True)
        assert len(calls) == 1 and len(controller.dispatches) == 1
        assert controller.cancellations == []
    finally:
        gate.set()
        await surface.close()


@pytest.mark.asyncio
async def test_barge_in_closes_continuation_generator():
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    entered, closed = asyncio.Event(), asyncio.Event()

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            if type(snapshot) is not AcceptedStartInferenceRequest:
                yield HostTaskProposalRequest("Inspect the release")
                return
            try:
                entered.set()
                await asyncio.Future()
                yield "unreachable"
            finally:
                closed.set()

        async def cancel(self, turn_id):
            pass

    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    operation = asyncio.create_task(
        loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
    )
    try:
        await entered.wait()
        await loop.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        assert closed.is_set() and controller.cancellations == []
    finally:
        await surface.close()


@pytest.mark.asyncio
async def test_terminal_before_continuation_text_suppresses_acknowledgment():
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    entered, release = asyncio.Event(), asyncio.Event()

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            if type(snapshot) is not AcceptedStartInferenceRequest:
                yield HostTaskProposalRequest("Inspect the release")
                return
            entered.set()
            await release.wait()
            yield "I am looking into that now."

        async def cancel(self, turn_id):
            pass

    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    operation = asyncio.create_task(
        loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
    )
    try:
        await entered.wait()
        context.record_task_completed(task_id="task_release_check", run_id="deleg_private_1")
        release.set()
        await operation
        assert synth.texts == [] and controller.cancellations == []
    finally:
        release.set()
        await surface.close()


@pytest.mark.asyncio
async def test_rejected_dispatch_cannot_borrow_an_existing_active_task():
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_release_check", run_id="deleg_existing", objective="Different work"
    )
    surface, controller, _ = _surface(context=context, accepted=False)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    calls = []

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            calls.append(snapshot)
            yield HostTaskProposalRequest("Inspect the release")

        async def cancel(self, turn_id):
            pass

    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    try:
        await loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
        assert len(controller.dispatches) == 1 and len(calls) == 1 and synth.texts == []
    finally:
        await surface.close()


@pytest.mark.asyncio
async def test_superseded_lease_does_not_start_ack_continuation():
    gate = asyncio.Event()
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, dispatch_gate=gate)
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic"
    )
    proposals.bind("participant", 1)
    foreground = ForegroundTurnCoordinator()
    calls = []

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            calls.append(snapshot)
            yield HostTaskProposalRequest("Inspect the release")

        async def cancel(self, turn_id):
            pass

    loop = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    operation = asyncio.create_task(
        loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
    )
    try:
        await controller.dispatch_started.wait()
        foreground._is_current = lambda _lease: False
        gate.set()
        await operation
        assert len(calls) == 1 and controller.cancellations == []
    finally:
        gate.set()
        await surface.close()
