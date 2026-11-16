"""Typed-input lineage on assistant terminals is independent of evidence capture (#237)."""

import asyncio
from typing import Any
from uuid import UUID

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
    TaskTerminalOutcome,
)
from hermes_realtime.evidence import (
    BoundedEvidenceWriterQueueV1,
    ConversationOperationScheduler,
    EvidenceAdmissionControllerV1,
    InputSource,
)
from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_actions import (
    BlockingInference,
    BlockingPlayback,
    FailFirstPlayback,
    GatedCompletionSource,
    Playback,
    RecordingInference,
    Synthesizer,
    _active_replay_evidence,
    director_for_test,
)

_TERMINALS = ("assistant_turn_completed", "assistant_turn_interrupted")
_BINDING_GENERATION = 7


class _SilentSource:
    """No background completion ever arrives, so no update turn can interleave."""

    async def next_completion(self) -> TaskTerminalOutcome:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _CaptureOff:
    """A real lifecycle plus a real admission controller with capture disabled."""

    def __init__(
        self,
        *,
        inference: Any,
        playback: Any | None = None,
        source: Any | None = None,
    ) -> None:
        owner_generation = 237
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.replay_identities: list[object] = []
        self.context = ConversationContextStore()
        self.scheduler = ConversationOperationScheduler(
            owner_generation=owner_generation, max_operations=2,
        )
        self.lifecycle = EvidenceLifecycleOwner(
            owner_generation=owner_generation, operation_scheduler=self.scheduler,
        )
        self.lifecycle.activate_binding(
            binding_id=str(UUID(int=2371, version=4)),
            binding_generation=_BINDING_GENERATION,
            consent_epoch_id=str(UUID(int=2372, version=4)),
            logical_session_id=str(UUID(int=2373, version=4)),
        )
        self.admission = EvidenceAdmissionControllerV1(
            enabled=False,
            owner_generation=owner_generation,
            writer_sink=BoundedEvidenceWriterQueueV1(),
            operation_scheduler=self.scheduler,
            conversation_authority_is_current=self.lifecycle.conversation_authority_is_current,
        )
        self.director = director_for_test(
            self.context, _SilentSource() if source is None else source,
        )
        self.speech = StreamingSpeechLoop(
            context=self.context, foreground=ForegroundTurnCoordinator(),
            inference=inference, synthesizer=Synthesizer(),
            playback=Playback() if playback is None else playback,
            ledger=DeliveredSpeechLedger(),
            observer=lambda kind, data: self.events.append((kind, dict(data))),
        )
        original_run_turn = self.speech._run_turn

        async def recording_run_turn(*args: Any, **kwargs: Any) -> Any:
            self.replay_identities.append(kwargs.get("replay_identity"))
            return await original_run_turn(*args, **kwargs)

        object.__setattr__(self.speech, "_run_turn", recording_run_turn)
        self.executor = ConversationUpdateExecutor(
            director=self.director, speech=self.speech, max_owned_operations=2,
            operation_scheduler=self.scheduler,
            evidence_lifecycle=self.lifecycle.conversation_authority,
            evidence_admission=self.admission,
        )
        self.executor.start()

    def authority(self, source: InputSource, sequence: int) -> Any:
        return self.lifecycle.decline_to_user(
            self.lifecycle.mint_final_input(
                source=source,
                input_incarnation=sequence,
                media_incarnation=sequence if source is InputSource.MICROPHONE else None,
                typed_sequence=sequence if source is InputSource.TYPED else None,
            )
        )

    async def respond(self, turn_id: str, authority: Any) -> None:
        await self.executor.respond(
            turn_id, Transcript(text="Synthetic typed input.", final=True), authority,
            reservation=self.executor.reserve_response(),
        )

    def terminals(self, turn_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        return [
            (kind, data) for kind, data in self.events
            if kind in _TERMINALS and (turn_id is None or data["turnId"] == turn_id)
        ]


def _assert_no_evidence_claim(harness: _CaptureOff) -> None:
    assert harness.replay_identities and all(
        identity is None for identity in harness.replay_identities
    )
    assert harness.admission.diagnostics().active_lease_count == 0
    assert harness.speech._eligible_replay_identity() is None


@pytest.mark.asyncio
async def test_capture_off_typed_completion_carries_admitted_lineage() -> None:
    harness = _CaptureOff(inference=RecordingInference())
    try:
        await harness.respond("turn_typed_done", harness.authority(InputSource.TYPED, 3))
        assert harness.terminals() == [("assistant_turn_completed", {
            "turnId": "turn_typed_done", "turnGeneration": 1,
            "inputSequence": 3, "bindingGeneration": _BINDING_GENERATION,
        })]
        _assert_no_evidence_claim(harness)
    finally:
        await harness.executor.close()


@pytest.mark.asyncio
async def test_capture_off_typed_interruption_carries_admitted_lineage() -> None:
    inference = BlockingInference()
    harness = _CaptureOff(inference=inference)
    try:
        response = asyncio.create_task(
            harness.respond("turn_typed_cut", harness.authority(InputSource.TYPED, 4))
        )
        await asyncio.wait_for(inference.started.wait(), timeout=1)
        await harness.executor.cancel_foreground()
        with pytest.raises(asyncio.CancelledError):
            await response
        assert harness.terminals() == [("assistant_turn_interrupted", {
            "turnId": "turn_typed_cut", "turnGeneration": 1,
            "inputSequence": 4, "bindingGeneration": _BINDING_GENERATION,
        })]
        _assert_no_evidence_claim(harness)
    finally:
        await harness.executor.close()


@pytest.mark.asyncio
async def test_capture_off_lineage_is_this_turns_tuple_never_a_previous_one() -> None:
    harness = _CaptureOff(inference=RecordingInference())
    try:
        await harness.respond("turn_first", harness.authority(InputSource.TYPED, 1))
        await harness.respond("turn_mic", harness.authority(InputSource.MICROPHONE, 2))
        await harness.respond("turn_second", harness.authority(InputSource.TYPED, 3))
        lineage = [
            (data["turnId"], data.get("inputSequence"), data.get("bindingGeneration"))
            for _, data in harness.terminals()
        ]
        assert lineage == [
            ("turn_first", 1, _BINDING_GENERATION),
            ("turn_mic", None, None),
            ("turn_second", 3, _BINDING_GENERATION),
        ]
    finally:
        await harness.executor.close()


@pytest.mark.asyncio
async def test_capture_off_microphone_input_carries_no_typed_lineage() -> None:
    harness = _CaptureOff(inference=RecordingInference())
    try:
        await harness.respond("turn_mic_only", harness.authority(InputSource.MICROPHONE, 5))
        assert harness.terminals() == [("assistant_turn_completed", {
            "turnId": "turn_mic_only", "turnGeneration": 1,
        })]
    finally:
        await harness.executor.close()


@pytest.mark.asyncio
async def test_capture_off_non_integer_binding_generation_carries_no_typed_lineage() -> None:
    harness = _CaptureOff(inference=RecordingInference())
    try:
        authority = harness.authority(InputSource.TYPED, 6)
        # A forged non-int that the numeric range checks alone would accept as 1.
        object.__setattr__(authority, "binding_generation", True)
        await harness.respond("turn_forged_generation", authority)
        assert harness.terminals() == [("assistant_turn_completed", {
            "turnId": "turn_forged_generation", "turnGeneration": 1,
        })]
    finally:
        await harness.executor.close()


@pytest.mark.asyncio
async def test_response_without_authority_carries_no_typed_lineage() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    context = ConversationContextStore()
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(), synthesizer=Synthesizer(), playback=Playback(),
        ledger=DeliveredSpeechLedger(),
        observer=lambda kind, data: events.append((kind, dict(data))),
    )
    executor = ConversationUpdateExecutor(
        director=director_for_test(context, _SilentSource()), speech=speech,
    )
    executor.start()
    try:
        await executor.respond(
            "turn_no_authority", Transcript(text="Synthetic input.", final=True),
        )
        assert [(kind, data) for kind, data in events if kind in _TERMINALS] == [
            ("assistant_turn_completed", {"turnId": "turn_no_authority", "turnGeneration": 1}),
        ]
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_capture_off_update_turn_after_typed_turn_carries_no_typed_lineage() -> None:
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_lineage_update", status="completed", summary="The report is ready.",
        )
    )
    playback = BlockingPlayback()
    harness = _CaptureOff(inference=RecordingInference(), playback=playback, source=source)
    try:
        response = asyncio.create_task(
            harness.respond("turn_typed_before_update", harness.authority(InputSource.TYPED, 8))
        )
        await asyncio.wait_for(playback.first_started.wait(), timeout=1)
        source.release.set()
        for _ in range(100):
            if harness.director.pending_mentions():
                break
            await asyncio.sleep(0)
        assert harness.director.pending_mentions()
        playback.release_first.set()
        await asyncio.wait_for(response, timeout=1)
        await asyncio.wait_for(playback.update_played.wait(), timeout=1)
        await asyncio.wait_for(harness.executor.wait_for_action_record(1), timeout=1)
        terminals = harness.terminals()
        assert terminals[0] == ("assistant_turn_completed", {
            "turnId": "turn_typed_before_update", "turnGeneration": 1,
            "inputSequence": 8, "bindingGeneration": _BINDING_GENERATION,
        })
        assert len(terminals) == 2
        update_kind, update = terminals[1]
        assert update_kind == "assistant_turn_completed"
        assert update["turnId"] != "turn_typed_before_update"
        assert set(update) == {"turnId", "turnGeneration"}
    finally:
        await harness.executor.close()


@pytest.mark.asyncio
async def test_replay_of_typed_turn_carries_no_typed_lineage() -> None:
    lifecycle, admission, writer, _ = _active_replay_evidence(owner_generation=238)
    events: list[tuple[str, dict[str, Any]]] = []
    context = ConversationContextStore()
    speech = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(),
        inference=RecordingInference(), synthesizer=Synthesizer(),
        playback=FailFirstPlayback(), ledger=DeliveredSpeechLedger(),
        evidence_admission=admission,
        observer=lambda kind, data: events.append((kind, dict(data))),
    )
    executor = ConversationUpdateExecutor(
        director=director_for_test(context, _SilentSource()),
        speech=speech, max_owned_operations=1,
        operation_scheduler=lifecycle.operation_scheduler,
        evidence_lifecycle=lifecycle.conversation_authority,
        evidence_admission=admission,
    )
    executor.start()
    try:
        authority = lifecycle.decline_to_user(
            lifecycle.mint_final_input(
                source=InputSource.TYPED, input_incarnation=1,
                media_incarnation=None, typed_sequence=1,
            )
        )
        with pytest.raises(RuntimeError, match="injected transport failure"):
            await executor.respond(
                "turn_replay_lineage_source",
                Transcript(text="Synthetic typed input.", final=True),
                authority, reservation=executor.reserve_response(),
            )
        while writer.ordered_count:
            admission.complete_ordered_item(writer.get_nowait())
        assert await executor.resume_foreground() is True
        terminals = [(kind, data) for kind, data in events if kind in _TERMINALS]
        assert terminals[0] == ("assistant_turn_interrupted", {
            "turnId": "turn_replay_lineage_source", "turnGeneration": 1,
            "inputSequence": 1, "bindingGeneration": 1,
        })
        assert len(terminals) == 2
        assert terminals[1][0] == "assistant_turn_completed"
        assert set(terminals[1][1]) == {"turnId", "turnGeneration"}
        while writer.ordered_count:
            admission.complete_ordered_item(writer.get_nowait())
    finally:
        await executor.close()
