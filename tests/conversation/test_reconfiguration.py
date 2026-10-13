import asyncio

import pytest
from test_actions import (
    BlockingPlayback,
    GatedCompletionSource,
    Playback,
    RecordingInference,
    Synthesizer,
    director_for_test,
)

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
    TaskTerminalOutcome,
    UpdateDirective,
)
from hermes_realtime.evidence import ConversationOperationKind, ConversationOperationScheduler
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript


def test_quiescent_scheduler_blocks_all_kinds_and_restores_after_failure():
    scheduler = ConversationOperationScheduler(owner_generation=1)
    with pytest.raises(RuntimeError, match="synthetic"), scheduler.quiescent():
        for kind in ConversationOperationKind:
            assert scheduler.try_reserve(kind) is None
        raise RuntimeError("synthetic body failure")
    reservation = scheduler.try_reserve(ConversationOperationKind.RESPONSE)
    assert reservation is not None
    scheduler.release(reservation)


@pytest.mark.parametrize("state", ["reserved", "closed", "nested"])
def test_quiescent_scheduler_refuses_nonidle_ownership(state):
    scheduler = ConversationOperationScheduler(owner_generation=1)
    if state == "reserved":
        scheduler.try_reserve(ConversationOperationKind.REPLAY)
    elif state == "closed":
        scheduler.close()
    if state == "nested":
        with scheduler.quiescent():
            with pytest.raises(RuntimeError), scheduler.quiescent():
                pytest.fail("competing scope was admitted")
            assert scheduler.try_reserve(ConversationOperationKind.RESPONSE) is None
    else:
        with pytest.raises(RuntimeError), scheduler.quiescent():
            pytest.fail("nonidle scheduler was admitted")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mention_next", "interrupt"])
async def test_quiescent_actions_defer_completion_without_dropping_it(kind):
    class Policy:
        async def decide(self, _):
            return UpdateDirective(kind=kind, text="Synthetic report ready.")

    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_quiescent",
            status="completed",
            summary="Synthetic report",
        )
    )
    context = ConversationContextStore()
    director = director_for_test(context, source, policy=Policy())
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    actions = ConversationUpdateExecutor(director=director, speech=speech)
    actions.start()
    try:
        with actions.quiescent():
            source.release.set()
            for _ in range(20):
                await asyncio.sleep(0)
            assert source._delivered
            assert actions.action_records() == ()
            assert not actions.foreground_active
            assert actions.active_operation_count == 0
        record = await asyncio.wait_for(actions.wait_for_action_record(1), 1)
        assert record.disposition == "delivered"
    finally:
        await actions.close()


@pytest.mark.asyncio
async def test_consecutive_pauses_recheck_awakened_interrupt_before_admission():
    class Policy:
        async def decide(self, _):
            return UpdateDirective(kind="interrupt", text="Synthetic report ready.")

    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_second_pause",
            status="completed",
            summary="Synthetic report",
        )
    )
    context = ConversationContextStore()
    director = director_for_test(context, source, policy=Policy())
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    actions = ConversationUpdateExecutor(director=director, speech=speech)
    actions.start()
    try:
        with actions.quiescent():
            source.release.set()
            for _ in range(20):
                await asyncio.sleep(0)
        # The first event waiter has been awakened, but has not resumed yet.
        with actions.quiescent():
            for _ in range(20):
                await asyncio.sleep(0)
            assert actions.action_records() == ()
            assert actions.active_operation_count == 0
        record = await asyncio.wait_for(actions.wait_for_action_record(1), 1)
        assert record.disposition == "delivered"
    finally:
        await actions.close()


@pytest.mark.asyncio
async def test_quiescence_refuses_owned_cleanup_without_a_reservation(monkeypatch):
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_idle_guard",
            status="completed",
            summary="Synthetic report",
        )
    )
    context = ConversationContextStore()
    director = director_for_test(context, source)
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    actions = ConversationUpdateExecutor(director=director, speech=speech)
    entered, release = asyncio.Event(), asyncio.Event()
    original = speech._announce_idle_update

    async def unavailable(*args, **kwargs):
        return False

    async def cancel(*args, **kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr(speech, "cancel", cancel)
    monkeypatch.setattr(speech, "_announce_idle_update", unavailable)
    actions.start()
    source.release.set()
    assert (await asyncio.wait_for(actions.wait_for_action_record(1), 1)).disposition == "requeued"
    for _ in range(20):
        await asyncio.sleep(0)
    monkeypatch.setattr(speech, "_announce_idle_update", original)
    operation = asyncio.create_task(actions.cancel_foreground())
    await asyncio.wait_for(entered.wait(), 1)
    assert not speech.foreground_active
    assert actions.reserved_operation_count == 0
    with pytest.raises(RuntimeError, match="quiescent"), actions.quiescent():
        pytest.fail("owned action was ignored")
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(actions.action_records()) == 1
    assert len(director.pending_mentions()) == 1
    release.set()
    await operation
    await actions.close()


@pytest.mark.asyncio
async def test_restoring_quiescence_restarts_an_already_requeued_mention(monkeypatch):
    source = GatedCompletionSource(TaskTerminalOutcome(
        task_id="task_requeued_scope", status="completed", summary="Synthetic report",
    ))
    context = ConversationContextStore()
    director = director_for_test(context, source)
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=RecordingInference(),
        synthesizer=Synthesizer(), playback=Playback(), ledger=DeliveredSpeechLedger(),
    )
    original = speech._announce_idle_update

    async def unavailable(*args, **kwargs):
        return False

    monkeypatch.setattr(speech, "_announce_idle_update", unavailable)
    actions = ConversationUpdateExecutor(director=director, speech=speech)
    actions.start()
    source.release.set()
    assert (await asyncio.wait_for(actions.wait_for_action_record(1), 1)).disposition == "requeued"
    for _ in range(20):
        await asyncio.sleep(0)
    with actions.quiescent():
        monkeypatch.setattr(speech, "_announce_idle_update", original)
        assert len(director.pending_mentions()) == 1
    for _ in range(100):
        if len(actions.action_records()) == 2:
            break
        await asyncio.sleep(0)
    assert len(actions.action_records()) == 2
    assert actions.action_records()[-1].disposition == "delivered"
    await actions.close()


@pytest.mark.asyncio
async def test_quiescence_refuses_speech_owned_outside_an_action():
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_raw_speech",
            status="completed",
            summary="Synthetic report",
        )
    )
    context = ConversationContextStore()
    director = director_for_test(context, source)
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    actions = ConversationUpdateExecutor(director=director, speech=speech)
    operation = asyncio.create_task(
        speech.respond("turn_raw_speech", Transcript(text="Synthetic request", final=True))
    )
    await asyncio.wait_for(playback.first_started.wait(), 1)
    assert actions.active_operation_count == 0
    with pytest.raises(RuntimeError, match="quiescent"), actions.quiescent():
        pytest.fail("live speech was ignored")
    await actions.close()
    await asyncio.gather(operation, return_exceptions=True)
