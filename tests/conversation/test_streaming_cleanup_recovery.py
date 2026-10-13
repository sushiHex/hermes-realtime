"""Synthetic held-synthesis probes for the binding cleanup investigation."""

import asyncio
from contextlib import suppress
from types import MethodType

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.conversation.worker import (
    ConversationSessionWorker,
    ReconnectSafeConversationWorker,
)
from hermes_realtime.livekit.worker import LiveKitConversationWorker
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_actions import ManyCompletionSource, director_for_test
from tests.conversation.test_streaming import (
    BlockingSynthesisSynthesizer,
    PerTurnInference,
    RecordingPlayback,
    SegmentSynthesizer,
)
from tests.conversation.test_worker import SilenceVad
from tests.livekit.test_conversation_worker import peer_probe


class IdleTranscriber:
    async def push(self, frame):
        return ()

    async def finish_utterance(self):
        return None

    async def cancel(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [False, True], ids=["cooperative", "held-then-release"])
async def test_binding_reconnect_retains_held_synthesis_and_recovers_after_release(held):
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_synthetic_work",
        run_id="deleg_synthetic_run",
        objective="Synthetic independent work",
    )
    synth = BlockingSynthesisSynthesizer()
    if not held:
        synth = SegmentSynthesizer()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=PerTurnInference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        cleanup_timeout_seconds=0.02,
    )
    actions = ConversationUpdateExecutor(
        director=director_for_test(context, ManyCompletionSource([])),
        speech=speech,
        cleanup_timeout_seconds=0.1,
    )

    def binding_factory(identity, generation, owner):
        return ConversationSessionWorker(
            participant_identity=identity,
            session_generation=generation,
            vad=SilenceVad(),
            stt=IdleTranscriber(),
            actions=owner,
            cleanup_timeout_seconds=0.1,
        )

    runtime = ReconnectSafeConversationWorker(actions=actions, binding_factory=binding_factory)
    worker = LiveKitConversationWorker(runtime=runtime)
    first, _ = peer_probe("synthetic_first")
    replacement, replacement_events = peer_probe("synthetic_replacement")

    async def receive(self, **kwargs):
        await asyncio.Event().wait()

    first.receive_participant_audio = MethodType(receive, first)
    replacement.receive_participant_audio = MethodType(receive, replacement)
    generation = await worker.connect(
        first, room_name="synthetic_room", participant_identity="synthetic_user"
    )
    await worker.submit_final_transcript(
        participant_identity="synthetic_user",
        session_generation=generation,
        typed_sequence=1,
        text="Synthetic input",
    )
    try:
        if held:
            await asyncio.wait_for(synth.iterator.started.wait(), 1)
            with pytest.raises(TimeoutError, match="speech consumer cleanup"):
                await asyncio.wait_for(
                    worker.reconnect(
                        replacement,
                        room_name="synthetic_room",
                        participant_identity="synthetic_user",
                    ),
                    1,
                )
            assert not any(event.startswith("connect:") for event in replacement_events)
            assert synth.iterator.close_count == 0
            retained = tuple(speech._provider_cleanup_tasks)
            assert len(retained) == 1
            synth.iterator.release.set()
            await asyncio.wait_for(asyncio.gather(*retained, return_exceptions=True), 1)
        else:
            async with asyncio.timeout(1):
                while not synth.texts:
                    await asyncio.sleep(0)
        new_generation = await asyncio.wait_for(
            worker.reconnect(
                replacement, room_name="synthetic_room", participant_identity="synthetic_user"
            ),
            1,
        )
        assert new_generation == generation + 1
        assert context.snapshot().active_tasks[0].task_id == "task_synthetic_work"
        if held:
            assert synth.iterator.close_count == 1
    finally:
        if held:
            synth.iterator.release.set()
        await asyncio.wait_for(worker.close(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancel_caller", "fail_marker"),
    [(False, False), (True, False), (False, True)],
    ids=["timeout", "cancelled-caller", "marker-writer-refusal"],
)
async def test_new_inference_remains_refused_until_held_prefetch_releases(
    capsys,
    monkeypatch,
    cancel_caller,
    fail_marker,
):
    opened_second = asyncio.Event()

    class CountingInference(PerTurnInference):
        async def _stream(self, snapshot, *, turn_id):
            if turn_id == "turn_second":
                opened_second.set()
            async for value in super()._stream(snapshot, turn_id=turn_id):
                yield value

    synth = BlockingSynthesisSynthesizer()
    speech = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=CountingInference(),
        synthesizer=synth,
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        cleanup_timeout_seconds=0.02,
    )
    first = asyncio.create_task(
        speech.respond("turn_first", Transcript(text="Synthetic", final=True))
    )
    second = None
    try:
        await asyncio.wait_for(synth.iterator.started.wait(), 1)
        if fail_marker:
            import hermes_realtime.conversation.streaming as module

            def refuse_marker(*args, **kwargs):
                raise OSError("synthetic diagnostic sink refusal")

            monkeypatch.setattr(module, "print", refuse_marker, raising=False)
            with pytest.raises(BaseExceptionGroup) as captured:
                await speech.cancel()
            assert any(isinstance(error, OSError) for error in captured.value.exceptions)
            assert any(isinstance(error, TimeoutError) for error in captured.value.exceptions)
            assert speech._resumable_cleanup_count == 0
        elif cancel_caller:
            cancellation = asyncio.create_task(speech.cancel_for_binding_close())
            async with asyncio.timeout(1):
                while not speech._cleanup_tasks:
                    await asyncio.sleep(0)
            cancellation.cancel()
            with pytest.raises(BaseExceptionGroup) as captured:
                await asyncio.wait_for(cancellation, 1)
            assert any(
                isinstance(error, asyncio.CancelledError) for error in captured.value.exceptions
            )
            assert any(isinstance(error, TimeoutError) for error in captured.value.exceptions)
        else:
            with pytest.raises(TimeoutError, match="speech consumer cleanup"):
                await speech.cancel_for_binding_close()
        if not fail_marker:
            assert (
                '[speech-consumer-cleanup] {"version":1,"category":"retained","pending":1}'
                in capsys.readouterr().out
            )
        second = asyncio.create_task(
            speech.respond("turn_second", Transcript(text="Synthetic", final=True))
        )
        with pytest.raises(RuntimeError, match="provider cleanup is still pending"):
            await asyncio.wait_for(asyncio.shield(second), 0.2)
        assert not opened_second.is_set()
        assert len(speech._provider_cleanup_tasks) == 1
        synth.iterator.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        assert speech._provider_cleanup_tasks == set()
        await asyncio.wait_for(
            speech.respond("turn_second", Transcript(text="Synthetic", final=True)), 1
        )
        assert opened_second.is_set()
    finally:
        synth.iterator.release.set()
        for response in (first, second):
            if response is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(response, 1)
        await speech.close()
