"""Deterministic task status; no inference service or background executor."""

import asyncio

import pytest
from test_tasks import TaskSessionStub

from hermes_realtime.conversation.context import ConversationContextStore, ConversationMessage
from hermes_realtime.conversation.task_status import HostSpeechRequest, TaskStatusSnapshot
from hermes_realtime.conversation.tasks import ConversationTaskController, TaskCancelOutcome


def controller():
    return ConversationTaskController(
        context=ConversationContextStore(),
        session=TaskSessionStub(),
        session_id="session_status",
        id_factory=lambda: "unused",
    )


def test_status_empty_and_unavailable_are_distinct():
    owner = controller()
    assert owner.task_status() == TaskStatusSnapshot()
    owner._terminal_error = RuntimeError("synthetic")
    assert owner.task_status() == TaskStatusSnapshot(available=False)
    assert "unavailable" in owner.task_status().render()


def test_status_source_is_exact_and_preserved_by_snapshot():
    row = ConversationMessage(
        "assistant", "No confirmed active background tasks.", source="host_task_status"
    )
    assert row.source == "host_task_status"
    with pytest.raises(ValueError):
        ConversationMessage("user", "Synthetic", source="host_task_status")
    assert HostSpeechRequest() == HostSpeechRequest()


@pytest.mark.asyncio
async def test_status_distinguishes_cancellation_pending_and_acknowledged():
    owner = controller()
    owner._active_run_ids["task_test"] = "deleg_private"
    operation = asyncio.create_task(asyncio.sleep(0, result=TaskCancelOutcome("task_test", True)))
    owner._cancel_operations["task_test"] = operation
    assert owner.task_status().cancel_pending == 1
    assert owner.task_status().active == 0
    await operation
    assert owner.task_status().cancelling == 1
    owner._active_run_ids.clear()
    assert owner.task_status() == TaskStatusSnapshot()


@pytest.mark.asyncio
async def test_loop_renders_host_status_and_records_only_heard_source():
    from test_streaming import RecordingPlayback, SegmentSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield HostSpeechRequest()

        async def cancel(self, turn_id):
            pass

    context = ConversationContextStore()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_status=lambda: TaskStatusSnapshot(active=1),
    )
    await loop.respond("turn_status", Transcript(text="How is work?", final=True))
    heard = context.snapshot().messages[-1]
    assert heard.source == "host_task_status"
    assert heard.text == TaskStatusSnapshot(active=1).render()
    assert context.durable_view().messages[-1].source == "host_task_status"
    await loop.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("available", 1),
        ("active", True),
        ("pending", -1),
        ("cancelling", 257),
        ("cancel_pending", float("inf")),
    ],
)
def test_status_rejects_invalid_counts(field, value):
    with pytest.raises((ValueError, TypeError)):
        TaskStatusSnapshot(**{field: value})


@pytest.mark.asyncio
async def test_status_unknown_cancel_outcome_is_unavailable():
    owner = controller()
    owner._active_run_ids["task_test"] = "deleg_private"

    async def fail():
        raise RuntimeError("synthetic")

    operation = asyncio.create_task(fail())
    owner._cancel_operations["task_test"] = operation
    await asyncio.gather(operation, return_exceptions=True)
    assert owner.task_status() == TaskStatusSnapshot(available=False)


@pytest.mark.asyncio
async def test_status_pending_ack_and_terminal_before_ack_never_claim_active():
    from test_tasks import PreAckTerminalSession, terminal_event

    session = PreAckTerminalSession()
    owner = ConversationTaskController(
        context=ConversationContextStore(),
        session=session,
        session_id="session_001",
        id_factory=iter(("preack", "dispatch")).__next__,
    )
    operation = asyncio.create_task(
        owner.dispatch(objective="Synthetic work", utterance_id="user_1")
    )
    await session.dispatch_started.wait()
    assert owner.task_status() == TaskStatusSnapshot(pending=1)
    await session.updates.put(terminal_event(task_id="task_preack", run_id="deleg_preack_001"))
    await session.update_returned.wait()
    await asyncio.sleep(0)
    assert owner.task_status() == TaskStatusSnapshot(pending=1)
    session.release_ack.set()
    await operation
    assert owner.task_status() == TaskStatusSnapshot()
    await owner.close()


@pytest.mark.asyncio
async def test_loop_refuses_text_after_status_with_bounded_evidence(capsys):
    from test_streaming import RecordingPlayback, SegmentSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield HostSpeechRequest()
            yield "The task was cancelled."

        async def cancel(self, turn_id):
            pass

    context = ConversationContextStore()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    with pytest.raises(ValueError, match="after_terminal"):
        await loop.respond("turn_terminal", Transcript(text="Status?", final=True))
    assert [row for row in context.snapshot().messages if row.role == "assistant"] == []
    assert capsys.readouterr().out == (
        '[host-speech] {"category":"after_terminal","kind":"task_status"}\n'
    )
    await loop.close()


def test_durable_wire_downgrades_host_source_without_inventing_authority():
    from hermes_realtime.conversation.context import DurableConversation
    from hermes_realtime.integration.voice_tail import (
        ArchiveOutbox,
        parse_voice_tail,
        voice_tail_bytes,
    )

    view = DurableConversation(
        messages=(ConversationMessage("assistant", "Observed state.", source="host_task_status"),),
        prior_work=False,
    )
    raw = voice_tail_bytes(view, ArchiveOutbox("conv", 0, 0, 0, None, (), 0, None))
    assert b"host_task_status" not in raw
    tail = parse_voice_tail(raw, max_messages=16, max_item_chars=64)
    assert tail is not None
    assert tail.conversation.messages[0].source == "conversation"


@pytest.mark.asyncio
async def test_stale_marker_does_not_read_task_authority():
    from contextlib import suppress

    from test_streaming import RecordingPlayback, SegmentSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    entered = asyncio.Event()
    reads = []

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            entered.set()
            with suppress(asyncio.CancelledError):
                await asyncio.Future()
            yield HostSpeechRequest()

        async def cancel(self, turn_id):
            pass

    context = ConversationContextStore()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_status=lambda: reads.append(1) or TaskStatusSnapshot(),
    )
    response = asyncio.create_task(
        loop.respond("turn_stale", Transcript(text="Status?", final=True))
    )
    await entered.wait()
    await loop.cancel()
    await asyncio.gather(response, return_exceptions=True)
    assert reads == []
    assert [row for row in context.snapshot().messages if row.role == "assistant"] == []
    await loop.close()


@pytest.mark.asyncio
async def test_publication_rejects_invalid_source_before_audio():
    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator

    foreground = ForegroundTurnCoordinator()

    async def run(lease):
        with pytest.raises(ValueError, match="source"):
            await lease.publish("Synthetic.", source="unknown")

    await foreground.start("turn_source", run)
    await foreground._active_task


@pytest.mark.asyncio
async def test_model_and_host_speech_have_separate_heard_rows():
    from test_streaming import RecordingPlayback, SegmentSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield "Earlier model speech."
            yield HostSpeechRequest()

        async def cancel(self, turn_id):
            pass

    context = ConversationContextStore()
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    await loop.respond("turn_source", Transcript(text="Status?", final=True))
    rows = context.snapshot().messages
    assert [(row.source, row.text) for row in rows[1:]] == [
        ("conversation", "Earlier model speech."),
        ("host_task_status", TaskStatusSnapshot(available=False).render()),
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_interrupted_host_speech_records_heard_prefix_and_replays_source():
    from test_streaming import ContextProbePlayback, SentenceSliceSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield HostSpeechRequest()

        async def cancel(self, turn_id):
            pass

    context = ConversationContextStore()
    playback = ContextProbePlayback(context, block=(("turn_001", "slice_1_2"),))
    loop = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SentenceSliceSynthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        task_status=lambda: TaskStatusSnapshot(),
    )
    response = asyncio.create_task(loop.respond("turn_001", Transcript(text="Status?", final=True)))
    await playback.blocked.wait()
    await loop.cancel()
    await asyncio.gather(response, return_exceptions=True)
    heard = context.snapshot().messages[-1]
    assert (heard.source, heard.text, heard.interrupted) == (
        "host_task_status",
        "No confirmed active background tasks.",
        True,
    )
    assert await loop.resume_interrupted()
    replayed = context.snapshot().messages[-1]
    assert replayed.source == "host_task_status"
    assert replayed.text == " To cancel work, use the task's Cancel control."
    assert not replayed.interrupted
    await loop.close()


def test_delivery_invalid_source_does_not_consume_confirmation():
    from test_context import _prepare

    from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk

    context = ConversationContextStore()
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_test",
        chunk_id="chunk_test",
        text="Synthetic.",
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16000, channels=1),
    )
    admission = _prepare(context, chunk.text)
    ledger.queue(chunk, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(
        ledger.mark_started(chunk.turn_id, chunk.chunk_id)
    )
    with pytest.raises(ValueError, match="source"):
        context.record_assistant_delivery(
            admission=admission, ledger=ledger, confirmation=confirmation, source="unknown"
        )
    assert ledger.confirmed_text(confirmation, admission) == chunk.text
    assert (
        context.record_assistant_delivery(
            admission=admission, ledger=ledger, confirmation=confirmation, source="host_task_status"
        )
        == chunk.text
    )


@pytest.mark.parametrize(
    "state", ["unsupported", "closed", "pending_gap", "forged", "subclass", "valid"]
)
def test_surface_status_observes_owner_or_reports_unavailable(state):
    from test_work_tools import _surface

    from hermes_realtime.conversation.work_tools import WorkControlHealth

    surface, probe, _ = _surface()

    class Forged(TaskStatusSnapshot):
        pass

    status = Forged(active=1) if state == "subclass" else TaskStatusSnapshot(active=1)
    if state != "unsupported":
        probe.task_status = lambda: status
    if state == "closed":
        surface._health = WorkControlHealth.CLOSED
    if state == "pending_gap":
        surface._pending_starts["synthetic"] = object()
    if state == "forged":
        object.__setattr__(status, "active", -1)
    assert surface.task_status() == (
        TaskStatusSnapshot(active=1) if state == "valid" else TaskStatusSnapshot(available=False)
    )


@pytest.mark.asyncio
async def test_committed_context_evidence_binds_heard_host_source(monkeypatch):
    import json

    from test_streaming import RecordingPlayback, SegmentSynthesizer

    from hermes_realtime.conversation import streaming
    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    captured = []
    encode = streaming._committed_conversation_context_snapshot_bytes

    def capture(**fields):
        result = encode(**fields)
        captured.append(json.loads(result))
        return result

    monkeypatch.setattr(streaming, "_committed_conversation_context_snapshot_bytes", capture)

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield HostSpeechRequest()

        async def cancel(self, turn_id):
            pass

    loop = streaming.StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    await loop.respond("turn_first", Transcript(text="Status?", final=True))
    await loop.respond("turn_second", Transcript(text="Now?", final=True))
    assert "hostSpeechMessageIndices" not in captured[0]
    assert captured[1]["hostSpeechMessageIndices"] == [1]
    assert captured[1]["messages"][1][1] == TaskStatusSnapshot(available=False).render()
    await loop.close()


@pytest.mark.parametrize("changes", [{"pending": 257}, {"available": False, "active": 1}])
def test_snapshot_refuses_contradictory_or_unbounded_counts(changes):
    with pytest.raises(ValueError):
        TaskStatusSnapshot(**changes)


def test_context_refuses_unknown_source():
    with pytest.raises(ValueError, match="source"):
        ConversationMessage("assistant", "Synthetic.", source="unknown")


@pytest.mark.asyncio
async def test_announcement_cannot_select_host_status():
    from test_streaming import RecordingPlayback, RequestRecordingInference, SegmentSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger

    async def marker(text):
        yield HostSpeechRequest()

    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=RequestRecordingInference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
    )
    loop._announcement_stream = marker
    with pytest.raises(ValueError, match="unadvertised"):
        await loop.announce("turn_update", "Synthetic update.")
    await loop.close()


@pytest.mark.asyncio
async def test_loop_rejects_projection_subclass_with_model_text():
    from test_streaming import RecordingPlayback, SegmentSynthesizer

    from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
    from hermes_realtime.conversation.streaming import StreamingSpeechLoop
    from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

    class Forged(TaskStatusSnapshot):
        def render(self):
            return "The task has been cancelled."

    class Inference:
        async def stream(self, snapshot, *, turn_id):
            yield HostSpeechRequest()

        async def cancel(self, turn_id):
            pass

    loop = StreamingSpeechLoop(
        context=ConversationContextStore(),
        foreground=ForegroundTurnCoordinator(),
        inference=Inference(),
        synthesizer=SegmentSynthesizer(),
        playback=RecordingPlayback(),
        ledger=DeliveredSpeechLedger(),
        task_status=Forged,
    )
    with pytest.raises(ValueError, match="projection_type"):
        await loop.respond("turn_forged", Transcript(text="Status?", final=True))
    await loop.close()
