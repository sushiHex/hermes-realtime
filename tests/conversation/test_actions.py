import asyncio
from collections.abc import AsyncIterator
from types import MethodType
from typing import Any

import pytest

from hermes_realtime.conversation import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationInferenceRequest,
    ConversationTaskController,
    ConversationUpdateDirector,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
    TaskTerminalOutcome,
    UpdateDirective,
    UpdatePolicyInput,
)
from hermes_realtime.speech import (
    AudioFrame,
    DeliveredSpeechLedger,
    SpeechChunk,
    Transcript,
)


class CompletionSource:
    def __init__(self, completion: TaskTerminalOutcome) -> None:
        self._completion = completion
        self._delivered = False
        self._blocked = asyncio.Event()

    async def next_completion(self) -> TaskTerminalOutcome:
        if not self._delivered:
            self._delivered = True
            return self._completion
        await self._blocked.wait()
        raise AssertionError("unreachable")


class GatedCompletionSource(CompletionSource):
    def __init__(self, completion: TaskTerminalOutcome) -> None:
        super().__init__(completion)
        self.release = asyncio.Event()

    async def next_completion(self) -> TaskTerminalOutcome:
        await self.release.wait()
        return await super().next_completion()


class ManyCompletionSource:
    def __init__(self, completions: list[TaskTerminalOutcome]) -> None:
        self._completions = list(completions)
        self._blocked = asyncio.Event()

    async def next_completion(self) -> TaskTerminalOutcome:
        if self._completions:
            return self._completions.pop(0)
        await self._blocked.wait()
        raise AssertionError("unreachable")


class TwoStageCompletionSource:
    def __init__(
        self,
        first: TaskTerminalOutcome,
        second: TaskTerminalOutcome,
    ) -> None:
        self._first = first
        self._second = second
        self._stage = 0
        self.release_second = asyncio.Event()
        self._blocked = asyncio.Event()

    async def next_completion(self) -> TaskTerminalOutcome:
        if self._stage == 0:
            self._stage = 1
            return self._first
        if self._stage == 1:
            await self.release_second.wait()
            self._stage = 2
            return self._second
        await self._blocked.wait()
        raise AssertionError("unreachable")


class SessionStub:
    async def dispatch(self, request: Any) -> Any:
        raise AssertionError(request)

    async def cancel(self, request: Any) -> Any:
        raise AssertionError(request)

    async def next_update(self) -> Any:
        raise AssertionError("not used")

    async def close(self) -> None:
        return None


class MentionPolicy:
    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        return UpdateDirective(
            kind="mention_next",
            text="The requested report is ready.",
        )


class InterruptPolicy:
    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        return UpdateDirective(
            kind="interrupt",
            text="Stop: the selected bridge is closed.",
        )


class MentionThenInterruptPolicy:
    def __init__(self) -> None:
        self._calls = 0

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        self._calls += 1
        if self._calls == 1:
            return UpdateDirective(
                kind="mention_next",
                text="The lower-priority report is ready.",
            )
        return UpdateDirective(
            kind="interrupt",
            text="Urgent update delivered.",
        )


class RecordingInference:
    def __init__(self) -> None:
        self.requests: list[ConversationInferenceRequest] = []

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del turn_id
        if type(snapshot) is not ConversationInferenceRequest:
            raise TypeError("expected exact ConversationInferenceRequest")
        self.requests.append(snapshot)
        yield "I have the report."

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class BlockingInference:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        self.started.set()
        await self.release.wait()
        yield "Late answer."

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id
        self.release.set()


class FirstThenFailInference:
    def __init__(self) -> None:
        self._calls = 0

    async def _stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        self._calls += 1
        if self._calls == 2:
            raise RuntimeError("second inference failed")
        yield "First update delivered."

    def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id=turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class Synthesizer:
    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"chunk_{turn_id}",
            text=text,
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class Playback:
    async def play(self, chunk: SpeechChunk, *, is_valid: Any) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        del chunk

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class FailFirstPlayback(Playback):
    """Make an evidence-backed source turn replayable, then allow its replay."""

    def __init__(self) -> None:
        self.calls = 0

    async def play(self, chunk: SpeechChunk, *, is_valid: Any) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        del chunk
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("injected transport failure")


class BlockingPlayback(Playback):
    def __init__(self) -> None:
        self.played_text: list[str] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.update_played = asyncio.Event()

    async def play(self, chunk: SpeechChunk, *, is_valid: Any) -> None:
        if not is_valid():
            raise asyncio.CancelledError
        self.played_text.append(chunk.text)
        if len(self.played_text) == 1:
            self.first_started.set()
            await self.release_first.wait()
            return
        self.update_played.set()

    async def cancel(self, turn_id: str) -> None:
        del turn_id
        self.release_first.set()


def director_for_test(
    context: ConversationContextStore,
    source: Any,
    *,
    policy: Any | None = None,
    max_pending_mentions: int = 16,
) -> ConversationUpdateDirector:
    authority = ConversationTaskController(
        context=context,
        session=SessionStub(),
        session_id="session_actions",
        id_factory=lambda: "unused",
    )
    director = ConversationUpdateDirector(
        context=context,
        completions=authority,
        policy=MentionPolicy() if policy is None else policy,
        max_pending_mentions=max_pending_mentions,
    )
    object.__setattr__(director, "_completions", source)
    return director


def _active_replay_evidence(
    *,
    owner_generation: int,
) -> tuple[Any, Any, Any, Any]:
    """Build one real, active controller/lifecycle pair for replay ingress."""

    from uuid import UUID

    from hermes_realtime.evidence import (
        BoundedEvidenceWriterQueueV1,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        StoreDisposition,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    def uid(value: int) -> str:
        return str(UUID(int=value, version=4))

    installation_id = uid(901)
    producer_instance_id = uid(902)
    consent_epoch_id = uid(903)
    logical_session_id = uid(904)
    binding_id = uid(905)
    disclosure_digest = "a" * 64
    command = m.CreateEpochV1(
        protocol_version=1,
        installation_id=installation_id,
        producer_instance_id=producer_instance_id,
        consent_epoch_id=consent_epoch_id,
        logical_session_id=logical_session_id,
        binding_id=binding_id,
        binding_generation=1,
        consent_version=m.CONSENT_VERSION,
        disclosure_digest=disclosure_digest,
        retention_hours=24,
        microphone_accepted=True,
        typed_accepted=True,
        control_sequence=1,
        control_fingerprint_hash="b" * 64,
        session_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            logical_session_id=logical_session_id,
            event_id=uid(906),
            event_sequence=1,
            event_kind=m.EventKind.SESSION_OPENED,
            payload=m.SessionOpenedPayloadV1(
                consent_epoch_id=consent_epoch_id,
                binding_id=binding_id,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest=disclosure_digest,
                retention_hours=24,
                microphone_accepted=True,
                typed_accepted=True,
                predecessor_session_id=None,
            ),
        ),
        binding_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            logical_session_id=logical_session_id,
            event_id=uid(907),
            event_sequence=2,
            event_kind=m.EventKind.BINDING_OPENED,
            payload=m.BindingOpenedPayloadV1(
                binding_id=binding_id,
                binding_generation=1,
                microphone_available=True,
                typed_available=True,
            ),
        ),
    )
    scheduler = ConversationOperationScheduler(
        owner_generation=owner_generation,
        max_operations=1,
    )
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=owner_generation,
        operation_scheduler=scheduler,
    )
    writer = BoundedEvidenceWriterQueueV1()
    admission = EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=owner_generation,
        writer_sink=writer,
        operation_scheduler=scheduler,
        conversation_authority_is_current=lifecycle.conversation_authority_is_current,
    )
    reservation = admission.try_reserve_create_epoch(command)
    assert reservation is not None
    create_item = admission.try_enqueue_create_epoch(reservation)
    assert create_item is not None
    assert writer.get_nowait() is create_item
    assert (
        admission.complete_create_epoch(
            create_item,
            disposition=StoreDisposition.COMMITTED,
            binding_current=True,
        )
        is not None
    )
    lifecycle.activate_binding(
        binding_id=binding_id,
        binding_generation=1,
        consent_epoch_id=consent_epoch_id,
        logical_session_id=logical_session_id,
    )
    return lifecycle, admission, writer, command


@pytest.mark.asyncio
async def test_executor_claims_mentions_once_for_next_real_user_turn() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_report",
                status="completed",
                summary="The report is ready.",
            )
        ),
    )
    inference = RecordingInference()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    director.start()
    await director.next_decision()
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()

    await executor.respond(
        "turn_user_1",
        Transcript(text="What is next?", final=True),
    )

    assert [update.sequence for update in inference.requests[0].updates] == [1]
    assert director.pending_mentions() == ()
    assert executor.action_records()[0].disposition == "delivered"
    await executor.close()


@pytest.mark.asyncio
async def test_mid_turn_mention_drains_when_foreground_releases_without_another_user_turn() -> None:
    context = ConversationContextStore()
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_mid_turn_report",
            status="completed",
            summary="The report is ready.",
        )
    )
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
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    response = asyncio.create_task(
        executor.respond(
            "turn_counting",
            Transcript(text="Count to thirty.", final=True),
        )
    )
    await playback.first_started.wait()

    source.release.set()
    for _ in range(100):
        if director.pending_mentions():
            break
        await asyncio.sleep(0)
    assert director.pending_mentions()

    playback.release_first.set()
    await asyncio.wait_for(response, timeout=1)
    await asyncio.wait_for(playback.update_played.wait(), timeout=1)

    assert playback.played_text == [
        "I have the report.",
        "The requested report is ready.",
    ]
    assert director.pending_mentions() == ()
    record = await asyncio.wait_for(executor.wait_for_action_record(1), timeout=1)
    assert record.disposition == "delivered"
    await executor.close()


@pytest.mark.asyncio
async def test_idle_mention_cannot_preempt_foreground_during_startup() -> None:
    context = ConversationContextStore()
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_startup_report",
            status="completed",
            summary="The startup report is ready.",
        )
    )
    director = director_for_test(context, source)
    foreground = ForegroundTurnCoordinator()
    original_start = foreground.start
    start_entered = asyncio.Event()
    release_start = asyncio.Event()

    async def gated_start(_self: Any, turn_id: str, runner: Any) -> Any:
        start_entered.set()
        await release_start.wait()
        return await original_start(turn_id, runner)

    foreground.start = MethodType(gated_start, foreground)  # type: ignore[method-assign]
    class RecordingPlayback(Playback):
        def __init__(self) -> None:
            self.played_text: list[str] = []

        async def play(self, chunk: SpeechChunk, *, is_valid: Any) -> None:
            if not is_valid():
                raise asyncio.CancelledError
            self.played_text.append(chunk.text)

    playback = RecordingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    response = asyncio.create_task(
        executor.respond(
            "turn_starting",
            Transcript(text="Tell me something first.", final=True),
        )
    )
    await start_entered.wait()

    source.release.set()
    requeued = await asyncio.wait_for(executor.wait_for_action_record(1), timeout=1)
    assert requeued.disposition == "requeued"
    assert not response.done()
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(executor.action_records()) == 1

    foreground.start = original_start  # type: ignore[method-assign]
    release_start.set()
    await asyncio.wait_for(response, timeout=1)
    for _ in range(100):
        if len(executor.action_records()) == 2:
            break
        await asyncio.sleep(0)
    assert len(executor.action_records()) == 2
    delivered = executor.action_records()[-1]
    assert delivered.disposition == "delivered"
    assert playback.played_text == [
        "I have the report.",
        "The requested report is ready.",
    ]
    assert director.pending_mentions() == ()
    await executor.close()


@pytest.mark.asyncio
async def test_foreground_cancel_does_not_drain_pending_mention_over_barge_in() -> None:
    context = ConversationContextStore()
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_barge_in_report",
            status="completed",
            summary="The barge-in report is ready.",
        )
    )
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
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    response = asyncio.create_task(
        executor.respond(
            "turn_barge_in",
            Transcript(text="Keep talking.", final=True),
        )
    )
    await playback.first_started.wait()

    source.release.set()
    for _ in range(100):
        if director.pending_mentions():
            break
        await asyncio.sleep(0)
    assert director.pending_mentions()

    await executor.cancel_foreground()
    with pytest.raises(asyncio.CancelledError):
        await response
    for _ in range(10):
        await asyncio.sleep(0)

    assert [decision.sequence for decision in director.pending_mentions()] == [1]
    assert playback.played_text == ["I have the report."]
    assert executor.action_records() == ()
    await executor.close()


@pytest.mark.asyncio
async def test_cancelled_idle_mention_is_not_also_resumable() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_no_duplicate",
                status="completed",
                summary="The no-duplicate report is ready.",
            )
        ),
    )
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    await playback.first_started.wait()

    await executor.cancel_foreground()
    for _ in range(100):
        if executor.action_records():
            break
        await asyncio.sleep(0)

    assert executor.action_records()[-1].disposition == "requeued"
    assert await speech.resume_interrupted() is False
    assert [decision.sequence for decision in director.pending_mentions()] == [1]
    await executor.close()


@pytest.mark.asyncio
async def test_interrupt_preempts_idle_drain_at_single_operation_capacity() -> None:
    context = ConversationContextStore()
    source = TwoStageCompletionSource(
        TaskTerminalOutcome(
            task_id="task_low_priority",
            status="completed",
            summary="The lower-priority report is ready.",
        ),
        TaskTerminalOutcome(
            task_id="task_urgent",
            status="completed",
            summary="The urgent report is ready.",
        ),
    )
    director = director_for_test(
        context,
        source,
        policy=MentionThenInterruptPolicy(),
    )
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
    )
    executor.start()
    await playback.first_started.wait()

    source.release_second.set()
    urgent = await asyncio.wait_for(executor.wait_for_action_record(2), timeout=1)
    assert urgent.disposition == "delivered"
    for _ in range(100):
        if not director.pending_mentions() and len(executor.action_records()) >= 3:
            break
        await asyncio.sleep(0)

    executor.start()
    assert director.pending_mentions() == ()
    low_dispositions = [
        record.disposition
        for record in executor.action_records()
        if record.decision.sequence == 1
    ]
    assert low_dispositions == ["requeued", "delivered"]
    assert playback.played_text == [
        "The lower-priority report is ready.",
        "Urgent update delivered.",
        "The lower-priority report is ready.",
    ]
    await executor.close()


@pytest.mark.asyncio
async def test_completion_interrupt_replaces_active_process_commentary() -> None:
    class ProcessCommentaryInference(RecordingInference):
        async def _stream(
            self,
            snapshot: ConversationContextSnapshot,
            *,
            turn_id: str,
        ) -> AsyncIterator[str]:
            del turn_id
            if type(snapshot) is not ConversationInferenceRequest:
                raise TypeError("expected exact ConversationInferenceRequest")
            self.requests.append(snapshot)
            yield "My plan is to verify the relevant evidence."

    context = ConversationContextStore()
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_route",
            status="completed",
            summary="The selected bridge is closed.",
        )
    )
    director = director_for_test(context, source, policy=InterruptPolicy())
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=ProcessCommentaryInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    response = asyncio.create_task(
        executor.respond(
            "turn_user_1",
            Transcript(text="Tell me the route.", final=True),
        )
    )
    await playback.first_started.wait()

    source.release.set()
    await asyncio.wait_for(playback.update_played.wait(), timeout=1)
    record = await asyncio.wait_for(executor.wait_for_action_record(1), timeout=1)

    assert response.done()
    assert response.cancelled()
    assert playback.played_text == [
        "My plan is to verify the relevant evidence.",
        "Stop: the selected bridge is closed.",
    ]
    assert [message.text for message in context.snapshot().messages] == [
        "Tell me the route.",
        "My plan is to verify the relevant evidence.",
        "Stop: the selected bridge is closed.",
    ]
    assert record.disposition == "delivered"
    await executor.close()


@pytest.mark.asyncio
async def test_invalid_user_turn_does_not_claim_pending_mention() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_report",
                status="completed",
                summary="The report is ready.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    director.start()
    decision = await director.next_decision()
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()

    with pytest.raises(ValueError, match="only final transcripts"):
        await executor.respond(
            "turn_partial",
            Transcript(text="Still speaking", final=False),
        )

    assert director.pending_mentions() == (decision,)
    assert executor.action_records() == ()
    await executor.close()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_orphan_claimed_response_action() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_report",
                status="completed",
                summary="The report is ready.",
            )
        ),
    )
    director.start()
    await director.next_decision()
    inference = BlockingInference()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    caller = asyncio.create_task(
        executor.respond(
            "turn_cancelled_caller",
            Transcript(text="What is next?", final=True),
        )
    )
    await inference.started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert executor.active_operation_count == 1
    assert director.pending_mentions() == ()
    await executor.close()

    assert executor.active_operation_count == 0
    assert director.pending_mentions()[0].sequence == 1
    assert executor.action_records()[-1].disposition == "requeued"


@pytest.mark.asyncio
async def test_foreground_barge_in_cancels_only_active_response() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        GatedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_unrelated",
                status="completed",
                summary="Unrelated work completed.",
            )
        ),
    )
    inference = BlockingInference()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    response = asyncio.create_task(
        executor.respond(
            "turn_barge_in",
            Transcript(text="Keep talking.", final=True),
        )
    )
    await inference.started.wait()

    await executor.cancel_foreground()

    with pytest.raises(asyncio.CancelledError):
        await response
    assert executor.active_operation_count == 0
    assert context.snapshot().active_tasks == ()
    await executor.close()


@pytest.mark.asyncio
async def test_bound_executor_rejects_competing_action_and_speech_consumers() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_report",
                status="completed",
                summary="The report is ready.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)

    with pytest.raises(RuntimeError, match="bound to an action executor"):
        await director.next_decision()
    with pytest.raises(RuntimeError, match="does not own proactive speech"):
        await speech._respond_with_updates(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            "turn_forged",
            Transcript(text="Try to replay it.", final=True),
            (),
        )
    with pytest.raises(RuntimeError, match="already has an action executor"):
        ConversationUpdateExecutor(director=director, speech=speech)

    await executor.close()


@pytest.mark.asyncio
async def test_mentions_over_prompt_capacity_drain_overflow_when_floor_is_idle() -> None:
    context = ConversationContextStore()
    completions = [
        TaskTerminalOutcome(
            task_id=f"task_report_{index}",
            status="completed",
            summary=f"Report {index} is ready.",
        )
        for index in range(17)
    ]
    director = director_for_test(
        context,
        ManyCompletionSource(completions),
        max_pending_mentions=32,
    )
    director.start()
    for _ in range(17):
        await director.next_decision()
    inference = RecordingInference()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()

    await executor.respond(
        "turn_batch_1",
        Transcript(text="Give me the first batch.", final=True),
    )

    assert len(inference.requests[0].updates) == 16
    overflow = await asyncio.wait_for(executor.wait_for_action_record(17), timeout=1)
    assert overflow.disposition == "delivered"
    assert len(inference.requests) == 1
    assert director.pending_mentions() == ()
    assert len(executor.action_records()) == 17
    await executor.close()


@pytest.mark.asyncio
async def test_reused_turn_id_cannot_settle_an_undelivered_mention() -> None:
    context = ConversationContextStore()
    source = TwoStageCompletionSource(
        TaskTerminalOutcome(
            task_id="task_first",
            status="completed",
            summary="First report ready.",
        ),
        TaskTerminalOutcome(
            task_id="task_second",
            status="completed",
            summary="Second report ready.",
        ),
    )
    director = director_for_test(context, source)
    director.start()
    await director.next_decision()
    inference = FirstThenFailInference()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()
    await executor.respond(
        "reused_turn",
        Transcript(text="Give me the first report.", final=True),
    )
    source.release_second.set()

    async def wait_for_second_mention() -> None:
        while not director.pending_mentions():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_second_mention(), timeout=1)
    with pytest.raises(RuntimeError, match="second inference failed"):
        await executor.respond(
            "reused_turn",
            Transcript(text="Give me the second report.", final=True),
        )

    assert [decision.sequence for decision in director.pending_mentions()] == [2]
    assert executor.action_records()[-1].disposition == "requeued"
    await executor.close()


@pytest.mark.asyncio
async def test_interrupt_claim_settles_when_task_is_cancelled_before_entry() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_interrupt",
                status="completed",
                summary="The bridge closed.",
            )
        ),
        policy=InterruptPolicy(),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    director.start()
    decision, claim = await director._next_execution(executor)
    assert claim is not None

    operation = executor._start_interrupt_operation(claim, decision)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert executor.action_records()[-1].disposition == "superseded"
    await executor.close()


@pytest.mark.asyncio
async def test_executor_close_uses_host_shutdown_revocation_not_stop_speaking() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_host_shutdown",
                status="completed",
                summary="Host shutdown fixture.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    calls: list[str] = []

    async def reject_stop(self: StreamingSpeechLoop) -> None:
        del self
        raise AssertionError("executor close used stop-speaking cancellation")

    async def host_shutdown(self: StreamingSpeechLoop) -> None:
        del self
        calls.append("host_shutdown")

    speech.cancel = MethodType(reject_stop, speech)  # type: ignore[method-assign]
    speech.cancel_for_host_shutdown = MethodType(  # type: ignore[attr-defined]
        host_shutdown,
        speech,
    )
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()

    await executor.close()

    assert calls == ["host_shutdown"]


@pytest.mark.asyncio
async def test_executor_retries_transient_final_close_failure() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_close",
                status="completed",
                summary="Close retry fixture.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    original_close = speech.close
    close_attempts = 0

    async def flaky_close(self: StreamingSpeechLoop) -> None:
        nonlocal close_attempts
        del self
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("temporary speech close failure")
        await original_close()

    speech.close = MethodType(flaky_close, speech)  # type: ignore[method-assign]
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()

    with pytest.raises(RuntimeError, match="temporary speech close failure"):
        await executor.close()
    await executor.close()

    assert close_attempts == 2


@pytest.mark.asyncio
async def test_response_capacity_is_reserved_before_spawn_and_released_exactly_once() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        GatedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_capacity",
                status="completed",
                summary="Capacity fixture.",
            )
        ),
    )
    inference = BlockingInference()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=inference,
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
    )
    executor.start()
    first = asyncio.create_task(
        executor.respond(
            "turn_reserved_first",
            Transcript(text="Hold this response.", final=True),
        )
    )
    await inference.started.wait()
    assert executor.reserved_operation_count == 1

    with pytest.raises(RuntimeError, match="operation capacity exhausted"):
        await executor.respond(
            "turn_rejected_before_spawn",
            Transcript(text="Do not spawn this response.", final=True),
        )
    assert executor.reserved_operation_count == 1

    await executor.cancel_foreground()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert executor.reserved_operation_count == 0
    await executor.close()


@pytest.mark.asyncio
async def test_response_replay_and_proactive_share_one_explicit_reservation_owner() -> None:
    from hermes_realtime.evidence import ConversationOperationKind

    context = ConversationContextStore()
    director = director_for_test(
        context,
        GatedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_shared_limit",
                status="completed",
                summary="Shared limit fixture.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
    )
    replay = executor.operation_scheduler.try_reserve(ConversationOperationKind.REPLAY)
    assert replay is not None
    executor.start()

    with pytest.raises(RuntimeError, match="operation capacity exhausted"):
        await executor.respond(
            "turn_blocked_by_replay",
            Transcript(text="This cannot reserve yet.", final=True),
        )
    executor.operation_scheduler.release(replay)
    await executor.respond(
        "turn_after_replay_release",
        Transcript(text="This can reserve now.", final=True),
    )
    assert executor.reserved_operation_count == 0
    await executor.close()
    assert executor.operation_scheduler.closed is True


@pytest.mark.asyncio
async def test_pre_reserved_response_releases_capacity_on_early_validation_failure() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_pre_reserved_validation",
                status="completed",
                summary="Validation must release the bearer.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
    )
    executor.start()
    reservation = executor.reserve_response()

    with pytest.raises(ValueError, match="only final transcripts"):
        await executor.respond(
            "turn_invalid_pre_reserved",
            Transcript(text="not final", final=False),
            reservation=reservation,
        )

    assert executor.reserved_operation_count == 0
    await executor.close()


@pytest.mark.asyncio
async def test_user_authority_reserves_exact_evidence_lease_before_response_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        InputSource,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_evidence_lease",
                status="completed",
                summary="The exact lease reaches speech.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=141, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=141,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=142, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=143, version=4)),
        logical_session_id=str(UUID(int=144, version=4)),
    )
    authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    lease = object.__new__(EvidenceTurnLease)
    trace: list[tuple[str, object]] = []

    def reserve(
        self: EvidenceAdmissionControllerV1,
        supplied_authority: object,
        operation: object,
    ) -> EvidenceLeaseResultV1:
        del self
        assert supplied_authority is authority
        trace.append(("admit", operation))
        return EvidenceLeaseResultV1(
            lease=lease,
            disposition=AppendDisposition.ADMITTED,
        )

    def admit_final(
        self: EvidenceAdmissionControllerV1,
        supplied_lease: object,
        supplied_authority: object,
        text: str,
    ) -> AppendDisposition:
        del self
        assert supplied_lease is lease
        assert supplied_authority is authority
        trace.append(("final", text))
        return AppendDisposition.ADMITTED

    async def respond_with_updates(
        self: StreamingSpeechLoop,
        speech_authority: object,
        operation: object,
        turn_id: str,
        transcript: Transcript,
        updates: object,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> None:
        del self, speech_authority, operation, turn_id, transcript, updates
        trace.append(("speech", evidence_lease))

    monkeypatch.setattr(EvidenceAdmissionControllerV1, "try_reserve_user_turn", reserve)
    monkeypatch.setattr(EvidenceAdmissionControllerV1, "try_admit_user_final", admit_final)
    monkeypatch.setattr(StreamingSpeechLoop, "_respond_with_updates", respond_with_updates)
    published: list[
        tuple[EvidenceLifecycleOwner | None, EvidenceAdmissionControllerV1 | None]
    ] = [(None, None)]
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_resolver=lambda: published[0],
    )
    executor.start()
    reservation = executor.reserve_response()
    published[0] = (lifecycle.conversation_authority, admission)

    await executor.respond(
        "turn_exact_evidence_lease",
        Transcript(text="capture this response", final=True),
        authority,
        reservation=reservation,
    )

    assert trace == [
        ("admit", reservation),
        ("final", "capture this response"),
        ("speech", lease),
    ]
    assert executor.reserved_operation_count == 0
    await executor.close()


@pytest.mark.asyncio
async def test_nonadmitted_evidence_result_continues_response_without_fake_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        InputSource,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_capture_rejected",
                status="completed",
                summary="Natural response remains available.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=151, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=151,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=152, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=153, version=4)),
        logical_session_id=str(UUID(int=154, version=4)),
    )
    authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    observed: list[EvidenceTurnLease | None] = []

    def reserve(
        self: EvidenceAdmissionControllerV1,
        supplied_authority: object,
        operation: object,
    ) -> EvidenceLeaseResultV1:
        del self, operation
        assert supplied_authority is authority
        return EvidenceLeaseResultV1(
            lease=None,
            disposition=AppendDisposition.DISABLED,
        )

    async def respond_with_updates(
        self: StreamingSpeechLoop,
        speech_authority: object,
        operation: object,
        turn_id: str,
        transcript: Transcript,
        updates: object,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> None:
        del self, speech_authority, operation, turn_id, transcript, updates
        observed.append(evidence_lease)

    monkeypatch.setattr(EvidenceAdmissionControllerV1, "try_reserve_user_turn", reserve)
    monkeypatch.setattr(StreamingSpeechLoop, "_respond_with_updates", respond_with_updates)
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
    )
    executor.start()
    reservation = executor.reserve_response()

    await executor.respond(
        "turn_capture_rejected",
        Transcript(text="respond without capture", final=True),
        authority,
        reservation=reservation,
    )

    assert observed == [None]
    assert executor.reserved_operation_count == 0
    await executor.close()


@pytest.mark.asyncio
async def test_replay_resolves_late_evidence_pair_and_drops_invalidated_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.conversation.streaming import _ReplayIdentity
    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        ReplayTurnAuthorityV1,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_replay_authority",
                status="completed",
                summary="Replay authority is reserved.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=161, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=161,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=162, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=163, version=4)),
        logical_session_id=str(UUID(int=164, version=4)),
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    source_evidence_id = str(UUID(int=165, version=4))
    replay_identity = _ReplayIdentity(
        source_evidence_id,
        1,
        str(UUID(int=162, version=4)),
        1,
        str(UUID(int=164, version=4)),
    )
    lease = object.__new__(EvidenceTurnLease)
    trace: list[tuple[str, object]] = []
    resume_outcome = [True]

    def eligible(self: StreamingSpeechLoop) -> _ReplayIdentity:
        del self
        return replay_identity

    def reserve(
        self: EvidenceAdmissionControllerV1,
        authority: ReplayTurnAuthorityV1,
        operation: object,
    ) -> EvidenceLeaseResultV1:
        del self
        assert authority.replay_of_evidence_turn_id == source_evidence_id
        assert authority.replay_generation == 1
        trace.append(("admit", operation))
        return EvidenceLeaseResultV1(
            lease=lease,
            disposition=AppendDisposition.ADMITTED,
        )

    async def resume(
        self: StreamingSpeechLoop,
        evidence_lease: EvidenceTurnLease | None = None,
    ) -> bool:
        del self
        trace.append(("speech", evidence_lease))
        return resume_outcome[0]

    def discard(
        self: EvidenceAdmissionControllerV1,
        replay_lease: EvidenceTurnLease,
        authority: ReplayTurnAuthorityV1,
    ) -> bool:
        del self
        assert replay_lease is lease
        assert authority.replay_of_evidence_turn_id == source_evidence_id
        trace.append(("discard", authority))
        return True

    monkeypatch.setattr(StreamingSpeechLoop, "_eligible_replay_identity", eligible)
    monkeypatch.setattr(StreamingSpeechLoop, "resume_interrupted", resume)
    monkeypatch.setattr(EvidenceAdmissionControllerV1, "try_reserve_replay_turn", reserve)
    monkeypatch.setattr(EvidenceAdmissionControllerV1, "discard_unopened_replay_turn", discard)
    published: list[
        tuple[EvidenceLifecycleOwner | None, EvidenceAdmissionControllerV1 | None]
    ] = [(None, None)]
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_resolver=lambda: published[0],
    )
    executor.start()
    published[0] = (lifecycle.conversation_authority, admission)

    assert await executor.resume_foreground() is True

    assert trace[0][0] == "admit"
    assert trace[1] == ("speech", lease)
    assert executor.reserved_operation_count == 0
    resume_outcome[0] = False
    assert await executor.resume_foreground() is False
    assert trace[-2] == ("speech", lease)
    assert trace[-1][0] == "discard"
    await asyncio.sleep(0)
    assert executor.reserved_operation_count == 0
    resume_outcome[0] = True
    published[0] = (None, None)
    assert await executor.resume_foreground() is True
    assert trace[-1] == ("speech", None)
    await executor.close()


@pytest.mark.asyncio
async def test_resume_foreground_real_replay_lease_opens_settles_and_unblocks_revoke_drain(
) -> None:
    """A successful replay owns its actual evidence lease through terminal drain."""

    from hermes_realtime.evidence import (
        DrainDisposition,
        InputSource,
        RevokeDisposition,
    )
    from hermes_realtime.evidence import models as m

    owner_generation = 166
    lifecycle, admission, writer, command = _active_replay_evidence(
        owner_generation=owner_generation,
    )
    context = ConversationContextStore()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=FailFirstPlayback(),
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
    )
    executor = ConversationUpdateExecutor(
        director=director_for_test(
            context,
            CompletionSource(
                TaskTerminalOutcome(
                    task_id="task_replay_real_lease",
                    status="completed",
                    summary="The replay must settle its evidence lease.",
                )
            ),
        ),
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=lifecycle.operation_scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
    )
    executor.start()
    source_authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )

    with pytest.raises(RuntimeError, match="injected transport failure"):
        await executor.respond(
            "turn_replay_source",
            Transcript(text="Please replay if delivery fails.", final=True),
            source_authority,
            reservation=executor.reserve_response(),
        )

    assert speech._eligible_replay_identity() is not None
    assert admission.diagnostics().active_lease_count == 0
    while writer.ordered_count:
        admission.complete_ordered_item(writer.get_nowait())

    assert await executor.resume_foreground() is True
    assert admission.diagnostics().active_lease_count == 0

    replay_items = [writer.get_nowait() for _ in range(writer.ordered_count)]
    assert [item.payload.snapshot.event_kind for item in replay_items] == [
        m.EventKind.TURN_OPENED,
        m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
        m.EventKind.TURN_SNAPSHOT,
        m.EventKind.TURN_SETTLED,
    ]
    for item in replay_items:
        admission.complete_ordered_item(item)

    # The browser-owned revoke bearer has no lifecycle-owner minting surface;
    # build its exact model boundary as the host runtime does.
    projection_reservation = object.__new__(m.ProjectionReservation)
    revoke_authority = object.__new__(m.ConsentRevokeAuthorityV1)
    for name, value in {
        "protocol_version": 1,
        "binding_id": command.binding_id,
        "binding_generation": command.binding_generation,
        "consent_epoch_id": command.consent_epoch_id,
        "revoke_gate_generation": owner_generation,
        "control_sequence": 2,
        "control_fingerprint_hash": "c" * 64,
        "projection_reservation": projection_reservation,
    }.items():
        object.__setattr__(revoke_authority, name, value)
    revoke_authority._validate()
    ticket = admission.begin_revoke(revoke_authority)
    revoke_request = writer.get_nowait()
    admission.complete_revoke_request(
        revoke_request,
        RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )
    finalizer = writer.get_nowait()
    assert type(finalizer.payload) is m.RevokeFinalizeV1
    admission.complete_revoke_finalize(finalizer, RevokeDisposition.PURGE_COMPLETED)
    assert ticket.terminal_event.is_set()

    lifecycle.close_binding(m.BindingCloseReason.CONSENT_REVOKED)
    drain_authority = lifecycle.mint_drain_authority(
        final_admission_ordinal=admission.final_admission_ordinal,
    )
    drain_ticket = admission.request_drain(drain_authority)
    drain_item = writer.get_nowait()
    admission.complete_drain(drain_item, DrainDisposition.STOPPED)
    assert drain_ticket.terminal_event.is_set()
    await executor.close()


@pytest.mark.asyncio
async def test_replay_rejects_dynamic_partial_evidence_pair_at_operation_admission() -> None:
    from uuid import UUID

    from hermes_realtime.evidence import ConversationOperationScheduler
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_dynamic_partial_replay",
                status="completed",
                summary="Partial evidence publication must fail closed.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=166, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=166,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=167, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=168, version=4)),
        logical_session_id=str(UUID(int=169, version=4)),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_resolver=lambda: (lifecycle.conversation_authority, None),
    )
    executor.start()

    with pytest.raises(RuntimeError, match="publish atomically"):
        await executor.resume_foreground()

    assert executor.reserved_operation_count == 0
    await executor.close()


@pytest.mark.asyncio
async def test_proactive_drops_dynamic_evidence_before_operation_admission() -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        ConversationOperationKind,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_dynamic_proactive_invalidation",
                status="completed",
                summary="Invalidated evidence must not mint a stale proactive authority.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=171, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=171,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=172, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=173, version=4)),
        logical_session_id=str(UUID(int=174, version=4)),
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    published: list[
        tuple[EvidenceLifecycleOwner | None, EvidenceAdmissionControllerV1 | None]
    ] = [(lifecycle.conversation_authority, admission)]
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_resolver=lambda: published[0],
    )
    executor.start()
    reservation = scheduler.try_reserve(ConversationOperationKind.PROACTIVE)
    assert reservation is not None
    published[0] = (None, None)

    assert executor._admit_proactive_evidence(reservation) == (None, None)

    scheduler.release(reservation)
    await executor.close()


@pytest.mark.asyncio
async def test_rejected_user_final_retires_unopened_lease_before_speech_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        InputSource,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_rejected_user_final",
                status="completed",
                summary="Speech must not start.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=145, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=145,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=146, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=147, version=4)),
        logical_session_id=str(UUID(int=148, version=4)),
    )
    authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    lease = object.__new__(EvidenceTurnLease)
    retired: list[object] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_reserve_user_turn",
        lambda self, supplied, operation: EvidenceLeaseResultV1(
            lease=lease,
            disposition=AppendDisposition.ADMITTED,
        ),
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_user_final",
        lambda self, supplied_lease, supplied_authority, text: (
            AppendDisposition.REJECTED_OVERSIZE
        ),
    )

    def retire(
        self: EvidenceAdmissionControllerV1,
        supplied_lease: object,
        supplied_authority: object,
    ) -> bool:
        del self
        assert supplied_lease is lease
        assert supplied_authority is authority
        retired.append(supplied_lease)
        return True

    monkeypatch.setattr(EvidenceAdmissionControllerV1, "discard_unopened_user_turn", retire)
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
    )
    executor.start()
    reservation = executor.reserve_response()

    with pytest.raises(RuntimeError, match="evidence final-input admission rejected"):
        await executor.respond(
            "turn_rejected_user_final",
            Transcript(text="x" * 4_097, final=True),
            authority,
            reservation=reservation,
        )

    assert retired == [lease]
    assert executor.reserved_operation_count == 0
    assert speech.active_turn_id is None
    await executor.close()


@pytest.mark.asyncio
async def test_response_spawn_failure_settles_admitted_lease_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        InputSource,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner
    from hermes_realtime.evidence.models import (
        SettledTerminalOutcomeV1,
        TerminalDisposition,
        TerminalReason,
    )
    from hermes_realtime.production_observation import _new_observation_channel

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_response_spawn_failure",
                status="completed",
                summary="The response task must fail to spawn.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=155, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=155,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=156, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=157, version=4)),
        logical_session_id=str(UUID(int=158, version=4)),
    )
    authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    lease = object.__new__(EvidenceTurnLease)
    observations, recorder = _new_observation_channel()
    trace: list[object] = []

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_reserve_user_turn",
        lambda self, supplied, operation: EvidenceLeaseResultV1(
            lease=lease,
            disposition=AppendDisposition.ADMITTED,
        ),
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_user_final",
        lambda self, supplied_lease, supplied_authority, text: AppendDisposition.ADMITTED,
    )

    def settle_spawn_failed(
        self: EvidenceAdmissionControllerV1,
        supplied_lease: object,
    ) -> AppendDisposition:
        del self
        assert supplied_lease is lease
        trace.append("settled")
        return AppendDisposition.ADMITTED

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_spawn_failed",
        settle_spawn_failed,
        raising=False,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settled_terminal_outcome",
        lambda _self, supplied_lease: (
            SettledTerminalOutcomeV1(
                terminal_disposition=TerminalDisposition.FAILED,
                terminal_reason=TerminalReason.TASK_SPAWN_FAILED,
                context_committed=False,
            )
            if supplied_lease is lease
            else None
        ),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
        production_observation_recorder=recorder,
    )
    executor.start()
    reservation = executor.reserve_response()
    original_create_task = asyncio.create_task

    def fail_response_spawn(coroutine: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("name") == "conversation-update-response:turn_spawn_failure":
            trace.append("spawn_failed")
            raise RuntimeError("injected response spawn failure")
        return original_create_task(coroutine, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_response_spawn)
    with pytest.raises(RuntimeError, match="injected response spawn failure"):
        await executor.respond(
            "turn_spawn_failure",
            Transcript(text="Admitted before spawn.", final=True),
            authority,
            reservation=reservation,
        )

    assert trace == ["spawn_failed", "settled"]
    assert [record.kind.value for record in observations.records()] == ["terminal_settled"]
    [settled] = observations.records()
    assert settled.context_committed is False
    assert executor.reserved_operation_count == 0
    monkeypatch.setattr(asyncio, "create_task", original_create_task)
    await executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settlement_disposition", "expected_observation_kinds"),
    (
        ("admitted", ("terminal_settled", "context_committed")),
        ("dropped_capacity", ()),
    ),
)
async def test_public_response_ingress_observes_only_admitted_settlement_once(
    monkeypatch: pytest.MonkeyPatch,
    settlement_disposition: str,
    expected_observation_kinds: tuple[str, ...],
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        InputSource,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner
    from hermes_realtime.evidence.models import (
        SettledTerminalOutcomeV1,
        TerminalDisposition,
        TerminalReason,
    )
    from hermes_realtime.production_observation import _new_observation_channel

    context = ConversationContextStore()
    scheduler = ConversationOperationScheduler(owner_generation=171, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=171,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=172, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=173, version=4)),
        logical_session_id=str(UUID(int=174, version=4)),
    )
    user_authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    lease = object.__new__(EvidenceTurnLease)
    object.__setattr__(lease, "evidence_turn_id", str(UUID(int=175, version=4)))
    object.__setattr__(lease, "binding_id", str(UUID(int=172, version=4)))
    object.__setattr__(lease, "binding_generation", 1)
    object.__setattr__(lease, "logical_session_id", str(UUID(int=174, version=4)))
    observations, recorder = _new_observation_channel()
    settlements: list[AppendDisposition] = []
    advisory: list[str] = []
    disposition = AppendDisposition(settlement_disposition)

    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_reserve_user_turn",
        lambda _self, authority, _reservation: EvidenceLeaseResultV1(
            lease=lease if authority is user_authority else None,
            disposition=AppendDisposition.ADMITTED,
        ),
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_user_final",
        lambda _self, _lease, _authority, _text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_generated",
        lambda _self, _lease, _text: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_admit_transport_confirmed_full",
        lambda _self, *_args, **_kwargs: AppendDisposition.ADMITTED,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_completed",
        lambda _self, _lease, **_kwargs: settlements.append(disposition) or disposition,
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settle_terminal",
        lambda _self, _lease, **_kwargs: pytest.fail("settlement must not repeat"),
    )
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "settled_terminal_outcome",
        lambda _self, _lease: SettledTerminalOutcomeV1(
            terminal_disposition=TerminalDisposition.COMPLETED,
            terminal_reason=TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
            context_committed=True,
        ),
    )

    def observe(kind: str, _data: dict[str, str | int | bool | None]) -> None:
        advisory.append(kind)
        if kind == "assistant_turn_completed" and disposition is AppendDisposition.ADMITTED:
            raise RuntimeError("completion observer failed after settlement")

    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
        production_observation_recorder=recorder,
        observer=observe,
    )
    executor = ConversationUpdateExecutor(
        director=director_for_test(
            context,
            GatedCompletionSource(
                TaskTerminalOutcome(
                    task_id="task_advisory_settlement",
                    status="completed",
                    summary="The response is settled before advisory observation.",
                )
            ),
        ),
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
        production_observation_recorder=recorder,
    )
    executor.start()
    reservation = executor.reserve_response()
    try:
        await executor.respond(
            "turn_public_advisory_settlement",
            Transcript(text="Question?", final=True),
            user_authority,
            reservation=reservation,
        )
    finally:
        await executor.close()

    assert settlements == [disposition]
    assert advisory.count("assistant_turn_completed") == 1
    assert "assistant_turn_interrupted" not in advisory
    assert tuple(
        record.kind.value
        for record in observations.records()
        if record.kind.value in {"terminal_settled", "context_committed"}
    ) == expected_observation_kinds


@pytest.mark.asyncio
async def test_idle_proactive_mints_authority_after_reservation_before_claim_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationKind,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_proactive_authority",
                status="completed",
                summary="The proactive authority fixture is ready.",
            )
        ),
    )
    director.start()
    await director.next_decision()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=101, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=101,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=102, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=103, version=4)),
        logical_session_id=str(UUID(int=104, version=4)),
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    trace: list[str] = []
    original_mint = EvidenceLifecycleOwner.mint_proactive_turn
    original_create_task = asyncio.create_task

    def record_mint(
        self: EvidenceLifecycleOwner,
        reservation: Any,
    ) -> Any:
        assert reservation.kind is ConversationOperationKind.PROACTIVE
        assert scheduler.active_count == 1
        trace.append("mint")
        return original_mint(self, reservation)

    def record_spawn(coroutine: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("name") == "conversation-update-idle-mentions":
            assert not director._action_claims
            trace.append("spawn")
        return original_create_task(coroutine, *args, **kwargs)

    monkeypatch.setattr(EvidenceLifecycleOwner, "mint_proactive_turn", record_mint)
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_reserve_proactive_turn",
        lambda self, authority, reservation: EvidenceLeaseResultV1(
            lease=None,
            disposition=AppendDisposition.DISABLED,
        ),
    )
    monkeypatch.setattr(asyncio, "create_task", record_spawn)
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
    )

    executor._start_idle_mention_drain()

    assert trace == ["spawn"]
    operation = executor._mention_drain_operation
    assert operation is not None
    await operation
    assert trace == ["spawn", "mint"]
    await executor.close()


@pytest.mark.asyncio
async def test_idle_proactive_saturation_mints_claims_and_spawns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        ConversationOperationKind,
        ConversationOperationScheduler,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_saturated_proactive",
                status="completed",
                summary="This mention must remain pending.",
            )
        ),
    )
    director.start()
    await director.next_decision()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=111, max_operations=1)
    blocker = scheduler.try_reserve(ConversationOperationKind.RESPONSE)
    assert blocker is not None
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=111,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=112, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=113, version=4)),
        logical_session_id=str(UUID(int=114, version=4)),
    )

    def fail_mint(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("saturation must reject before authority mint")

    monkeypatch.setattr(lifecycle, "mint_proactive_turn", fail_mint)
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
    )

    executor._start_idle_mention_drain()

    assert executor._mention_drain_operation is None
    assert not director._action_claims
    assert len(director.pending_mentions()) == 1
    scheduler.release(blocker)
    await executor.close()


@pytest.mark.asyncio
async def test_idle_proactive_passes_minted_authority_into_speech_operation() -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
        EvidenceTurnLease,
        ProactiveTurnAuthorityV1,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_proactive_speech_authority",
                status="completed",
                summary="The authority must reach speech.",
            )
        ),
    )
    director.start()
    await director.next_decision()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=121, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=121,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=122, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=123, version=4)),
        logical_session_id=str(UUID(int=124, version=4)),
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    lease = object.__new__(EvidenceTurnLease)
    observed: list[ProactiveTurnAuthorityV1] = []
    admitted: list[tuple[ProactiveTurnAuthorityV1, object]] = []
    original_begin = speech._begin_update_operation

    def capture_begin(
        self: StreamingSpeechLoop,
        action_authority: Any,
        turn_id: str,
        evidence_authority: ProactiveTurnAuthorityV1,
    ) -> Any:
        del self
        observed.append(evidence_authority)
        return original_begin(action_authority, turn_id, evidence_authority)

    speech._begin_update_operation = MethodType(  # type: ignore[method-assign]
        capture_begin,
        speech,
    )
    original_reserve = EvidenceAdmissionControllerV1.try_reserve_proactive_turn

    def reserve(
        self: EvidenceAdmissionControllerV1,
        authority: ProactiveTurnAuthorityV1,
        operation: object,
    ) -> EvidenceLeaseResultV1:
        del self
        admitted.append((authority, operation))
        return EvidenceLeaseResultV1(lease=lease, disposition=AppendDisposition.ADMITTED)

    EvidenceAdmissionControllerV1.try_reserve_proactive_turn = reserve  # type: ignore[method-assign]
    published: list[
        tuple[EvidenceLifecycleOwner | None, EvidenceAdmissionControllerV1 | None]
    ] = [(None, None)]
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_resolver=lambda: published[0],
    )
    published[0] = (lifecycle.conversation_authority, admission)

    try:
        executor._start_idle_mention_drain()
        operation = executor._mention_drain_operation
        assert operation is not None
        await operation
    finally:
        EvidenceAdmissionControllerV1.try_reserve_proactive_turn = original_reserve  # type: ignore[method-assign]

    assert len(observed) == 1
    assert observed[0].owner_generation == 121
    assert len(admitted) == 1
    assert admitted[0][0] is observed[0]
    assert executor._proactive_authorities == {}
    await executor.close()


@pytest.mark.asyncio
async def test_interrupt_transfer_replaces_proactive_authority_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationScheduler,
        EvidenceAdmissionControllerV1,
        EvidenceLeaseResultV1,
    )
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    context = ConversationContextStore()
    source = TwoStageCompletionSource(
        TaskTerminalOutcome(
            task_id="task_idle_authority",
            status="completed",
            summary="The idle authority is active.",
        ),
        TaskTerminalOutcome(
            task_id="task_interrupt_authority",
            status="failed",
            reason="The interrupt needs replacement authority.",
        ),
    )
    director = director_for_test(
        context,
        source,
        policy=MentionThenInterruptPolicy(),
    )
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    scheduler = ConversationOperationScheduler(owner_generation=131, max_operations=1)
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=131,
        operation_scheduler=scheduler,
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=132, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=133, version=4)),
        logical_session_id=str(UUID(int=134, version=4)),
    )
    admission = object.__new__(EvidenceAdmissionControllerV1)
    object.__setattr__(admission, "_operation_scheduler", scheduler)
    minted_serials: list[int] = []
    original_mint = lifecycle.mint_proactive_turn

    def record_mint(reservation: Any) -> Any:
        minted_serials.append(reservation.operation_serial)
        return original_mint(reservation)

    monkeypatch.setattr(lifecycle, "mint_proactive_turn", record_mint)
    monkeypatch.setattr(
        EvidenceAdmissionControllerV1,
        "try_reserve_proactive_turn",
        lambda self, authority, reservation: EvidenceLeaseResultV1(
            lease=None,
            disposition=AppendDisposition.DISABLED,
        ),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
    )
    executor.start()
    await playback.first_started.wait()

    source.release_second.set()
    record = await asyncio.wait_for(executor.wait_for_action_record(2), timeout=1)

    assert record.disposition == "delivered"
    assert minted_serials[:2] == [1, 2]
    await executor.close()


@pytest.mark.asyncio
async def test_failed_interrupt_spawn_restores_the_transferred_idle_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ConversationContextStore()
    source = TwoStageCompletionSource(
        TaskTerminalOutcome(
            task_id="task_idle_before_spawn_failure",
            status="completed",
            summary="The idle update is still running.",
        ),
        TaskTerminalOutcome(
            task_id="task_interrupt_spawn_failure",
            status="failed",
            reason="The urgent update should fail to spawn.",
        ),
    )
    director = director_for_test(
        context,
        source,
        policy=MentionThenInterruptPolicy(),
    )
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
    )
    executor.start()
    await playback.first_started.wait()
    assert executor.reserved_operation_count == 1

    original_create_task = asyncio.create_task

    def fail_interrupt_spawn(coroutine: Any, *args: Any, **kwargs: Any) -> Any:
        name = kwargs.get("name")
        if type(name) is str and name.startswith("conversation-update-interrupt:"):
            raise RuntimeError("injected interrupt spawn failure")
        return original_create_task(coroutine, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_interrupt_spawn)
    source.release_second.set()
    consumer = executor._decision_consumer
    assert consumer is not None
    await asyncio.wait_for(asyncio.shield(consumer), timeout=1)

    assert executor.reserved_operation_count == 1
    assert len(director._action_claims) == 1
    remaining_decision = next(iter(director._action_claims.values()))[1]
    assert remaining_decision.kind == "mention_next"
    assert executor.action_records()[-1].disposition == "failed"
    monkeypatch.setattr(asyncio, "create_task", original_create_task)
    with pytest.raises(RuntimeError, match="injected interrupt spawn failure"):
        await executor.close()
    assert executor.reserved_operation_count == 0


@pytest.mark.asyncio
async def test_failed_interrupt_begin_restores_the_live_idle_reservation() -> None:
    from hermes_realtime.evidence import ConversationOperationKind

    context = ConversationContextStore()
    source = TwoStageCompletionSource(
        TaskTerminalOutcome(
            task_id="task_idle_before_interrupt_begin_failure",
            status="completed",
            summary="The idle update remains live.",
        ),
        TaskTerminalOutcome(
            task_id="task_interrupt_begin_failure",
            status="failed",
            reason="The urgent update cannot begin.",
        ),
    )
    director = director_for_test(
        context,
        source,
        policy=MentionThenInterruptPolicy(),
    )
    playback = BlockingPlayback()
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=playback,
        ledger=DeliveredSpeechLedger(),
    )
    original_begin = speech._begin_update_operation

    def fail_interrupt_begin(
        self: StreamingSpeechLoop,
        authority: Any,
        turn_id: str,
    ) -> Any:
        del self
        if turn_id.startswith("update_"):
            raise RuntimeError("injected interrupt begin failure")
        return original_begin(authority, turn_id)

    speech._begin_update_operation = MethodType(  # type: ignore[method-assign]
        fail_interrupt_begin,
        speech,
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
    )
    executor.start()
    await playback.first_started.wait()
    source.release_second.set()
    record = await asyncio.wait_for(executor.wait_for_action_record(2), timeout=1)

    assert record.disposition == "failed"
    assert executor.reserved_operation_count == 1
    assert (
        executor.operation_scheduler.try_reserve(ConversationOperationKind.PROACTIVE)
        is None
    )
    await executor.close()
    assert executor.reserved_operation_count == 0


@pytest.mark.asyncio
async def test_response_update_operation_failure_requeues_every_claim() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_response_begin_failure",
                status="completed",
                summary="This mention must remain available.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    director.start()
    await director.next_decision()
    executor = ConversationUpdateExecutor(director=director, speech=speech)
    executor.start()

    def fail_begin(
        self: StreamingSpeechLoop,
        authority: Any,
        turn_id: str,
    ) -> Any:
        del self, authority, turn_id
        raise RuntimeError("injected update operation failure")

    speech._begin_update_operation = MethodType(fail_begin, speech)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="injected update operation failure"):
        await executor.respond(
            "turn_response_begin_failure",
            Transcript(text="Keep the mention pending.", final=True),
        )

    assert not director._action_claims
    assert len(director.pending_mentions()) == 1
    assert executor.action_records()[-1].disposition == "requeued"
    assert executor.reserved_operation_count == 0
    await executor.close()


@pytest.mark.asyncio
async def test_rejected_idle_start_retires_real_unopened_proactive_lease() -> None:
    context = ConversationContextStore()
    director = director_for_test(
        context,
        CompletionSource(
            TaskTerminalOutcome(
                task_id="task_proactive_lease_race",
                status="completed",
                summary="This update loses the speech admission race.",
            )
        ),
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(),
        synthesizer=Synthesizer(),
        playback=Playback(),
        ledger=DeliveredSpeechLedger(),
    )
    lifecycle, admission, _, _ = _active_replay_evidence(owner_generation=426)

    async def reject_before_open(
        self: StreamingSpeechLoop,
        authority: Any,
        operation: Any,
        turn_id: str,
        decision: Any,
        evidence_lease: Any = None,
    ) -> bool:
        del self, authority, operation, turn_id, decision
        assert evidence_lease is not None
        return False

    speech._announce_idle_update = MethodType(  # type: ignore[method-assign]
        reject_before_open,
        speech,
    )
    executor = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        max_owned_operations=1,
        operation_scheduler=admission.operation_scheduler,
        evidence_resolver=lambda: (lifecycle.conversation_authority, admission),
    )
    executor.start()

    record = await asyncio.wait_for(executor.wait_for_action_record(1), timeout=1)
    assert record.disposition == "requeued"
    assert admission.diagnostics().active_lease_count == 0
    assert executor.reserved_operation_count == 0
    await executor.close()
