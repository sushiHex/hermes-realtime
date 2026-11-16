"""Typed lineage reaches assistant terminals with no evidence runtime at all (#237).

Composed as the host does with capture off: one shared executor and speech loop,
a command router, and per-binding session workers with no evidence lifecycle.
"""

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
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.evidence import ConversationOperationScheduler, InputSource
from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner
from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, Transcript
from tests.conversation.test_actions import (
    BlockingInference,
    BlockingPlayback,
    FailFirstPlayback,
    GatedCompletionSource,
    Playback,
    RecordingInference,
    Synthesizer,
    director_for_test,
)
from tests.conversation.test_worker import (
    CommandRouterProbe,
    PreliminaryAndEndpointFinalTranscriber,
    ScriptedVad,
    SilenceVad,
)

_TERMINALS = ("assistant_turn_completed", "assistant_turn_interrupted")
_IDENTITY = "browser_user"


class _SilentSource:
    async def next_completion(self) -> TaskTerminalOutcome:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _NoCaptureHost:
    def __init__(
        self,
        *,
        inference: Any | None = None,
        playback: Any | None = None,
        source: Any | None = None,
        handled: bool = False,
    ) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.context = ConversationContextStore()
        self.director = director_for_test(
            self.context, _SilentSource() if source is None else source,
        )
        self.speech = StreamingSpeechLoop(
            context=self.context, foreground=ForegroundTurnCoordinator(),
            inference=RecordingInference() if inference is None else inference,
            synthesizer=Synthesizer(),
            playback=Playback() if playback is None else playback,
            ledger=DeliveredSpeechLedger(),
            observer=lambda kind, data: self.events.append((kind, dict(data))),
        )
        self.executor = ConversationUpdateExecutor(director=self.director, speech=self.speech)
        self.router = CommandRouterProbe(handled=handled)

    def bind(self, generation: int, *, microphone: bool = False) -> ConversationSessionWorker:
        worker = ConversationSessionWorker(
            participant_identity=_IDENTITY,
            session_generation=generation,
            vad=ScriptedVad() if microphone else SilenceVad(),
            stt=PreliminaryAndEndpointFinalTranscriber(),
            actions=self.executor,
            command_router=self.router,
        )
        worker.start()
        return worker

    def lineage(self) -> list[tuple[str, str, int | None, int | None]]:
        return [
            (kind, data["turnId"], data.get("inputSequence"), data.get("bindingGeneration"))
            for kind, data in self.events if kind in _TERMINALS
        ]


async def _type(worker: ConversationSessionWorker, generation: int, sequence: int) -> None:
    await worker.submit_final_transcript(
        participant_identity=_IDENTITY, session_generation=generation,
        typed_sequence=sequence, text="Synthetic typed input.",
    )


@pytest.mark.asyncio
async def test_no_capture_typed_completion_carries_accepted_tuple() -> None:
    host = _NoCaptureHost()
    worker = host.bind(4)
    try:
        await _type(worker, 4, 1)
        await worker.wait_for_responses()
        assert host.lineage() == [("assistant_turn_completed", "session_4_turn_1", 1, 4)]
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_no_capture_typed_interruption_carries_accepted_tuple() -> None:
    inference = BlockingInference()
    host = _NoCaptureHost(inference=inference)
    worker = host.bind(5)
    try:
        await _type(worker, 5, 2)
        await asyncio.wait_for(inference.started.wait(), timeout=1)
        await host.executor.cancel_foreground()
        await worker.wait_for_responses()
        assert host.lineage() == [("assistant_turn_interrupted", "session_5_turn_1", 2, 5)]
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_no_capture_binding_rollover_carries_the_new_bindings_generation() -> None:
    host = _NoCaptureHost()
    first = host.bind(1)
    await _type(first, 1, 1)
    await first.wait_for_responses()
    await first.close_binding()
    second = host.bind(2)
    try:
        with pytest.raises(RuntimeError, match="stale session generation"):
            await _type(second, 1, 2)
        await _type(second, 2, 1)
        await second.wait_for_responses()
        assert host.lineage() == [
            ("assistant_turn_completed", "session_1_turn_1", 1, 1),
            ("assistant_turn_completed", "session_2_turn_1", 1, 2),
        ]
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_no_capture_stale_or_duplicate_sequence_carries_no_lineage() -> None:
    host = _NoCaptureHost()
    worker = host.bind(3)
    try:
        for sequence in (2, 2, 1, 3):
            await _type(worker, 3, sequence)
            await worker.wait_for_responses()
        assert host.lineage() == [
            ("assistant_turn_completed", "session_3_turn_1", 2, 3),
            ("assistant_turn_completed", "session_3_turn_2", None, None),
            ("assistant_turn_completed", "session_3_turn_3", None, None),
            ("assistant_turn_completed", "session_3_turn_4", 3, 3),
        ]
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_no_capture_handled_typed_command_mints_no_terminal_lineage() -> None:
    host = _NoCaptureHost(handled=True)
    worker = host.bind(6)
    try:
        await _type(worker, 6, 1)
        await worker.wait_for_responses()
        assert host.router.texts == ["Synthetic typed input."]
        assert host.lineage() == []
        assert not any("inputSequence" in data for _, data in host.events)
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_no_capture_microphone_turn_after_typed_turn_carries_no_lineage() -> None:
    host = _NoCaptureHost()
    worker = host.bind(7, microphone=True)
    try:
        await _type(worker, 7, 1)
        await worker.wait_for_responses()
        frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)
        await worker.receive_audio(_IDENTITY, 7, frame)
        await worker.receive_audio(_IDENTITY, 7, frame)
        await worker.wait_for_responses()
        assert host.lineage() == [
            ("assistant_turn_completed", "session_7_turn_1", 1, 7),
            ("assistant_turn_completed", "session_7_turn_2", None, None),
        ]
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_no_capture_update_turn_after_typed_turn_carries_no_lineage() -> None:
    source = GatedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_no_capture_update", status="completed", summary="Report ready.",
        )
    )
    playback = BlockingPlayback()
    host = _NoCaptureHost(playback=playback, source=source)
    worker = host.bind(8)
    try:
        await _type(worker, 8, 1)
        await asyncio.wait_for(playback.first_started.wait(), timeout=1)
        source.release.set()
        for _ in range(100):
            if host.director.pending_mentions():
                break
            await asyncio.sleep(0)
        assert host.director.pending_mentions()
        playback.release_first.set()
        await worker.wait_for_responses()
        await asyncio.wait_for(playback.update_played.wait(), timeout=1)
        await asyncio.wait_for(host.executor.wait_for_action_record(1), timeout=1)
        lineage = host.lineage()
        assert lineage[0] == ("assistant_turn_completed", "session_8_turn_1", 1, 8)
        assert len(lineage) == 2
        assert lineage[1][1] != "session_8_turn_1"
        assert lineage[1][2:] == (None, None)
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_no_capture_replay_of_typed_turn_carries_no_lineage() -> None:
    host = _NoCaptureHost(playback=FailFirstPlayback())
    worker = host.bind(9)
    try:
        await _type(worker, 9, 1)
        await asyncio.gather(*tuple(worker._response_operations), return_exceptions=True)
        assert host.speech._eligible_replay_identity() is None
        assert await host.executor.resume_foreground() is True
        lineage = host.lineage()
        assert lineage[0] == ("assistant_turn_interrupted", "session_9_turn_1", 1, 9)
        assert len(lineage) == 2
        assert lineage[1][0] == "assistant_turn_completed"
        assert lineage[1][2:] == (None, None)
    finally:
        with pytest.raises(RuntimeError, match="injected transport failure"):
            await worker.close()


def _typed_authority(sequence: int, binding_generation: int) -> Any:
    lifecycle = EvidenceLifecycleOwner(
        owner_generation=239,
        operation_scheduler=ConversationOperationScheduler(
            owner_generation=239, max_operations=1,
        ),
    )
    lifecycle.activate_binding(
        binding_id=str(UUID(int=2391, version=4)),
        binding_generation=binding_generation,
        consent_epoch_id=str(UUID(int=2392, version=4)),
        logical_session_id=str(UUID(int=2393, version=4)),
    )
    return lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=InputSource.TYPED, input_incarnation=sequence,
            media_incarnation=None, typed_sequence=sequence,
        )
    )


@pytest.mark.asyncio
async def test_explicit_lineage_must_equal_the_authority_derived_tuple() -> None:
    host = _NoCaptureHost()
    host.executor.start()
    try:
        authority = _typed_authority(3, 7)
        for mismatch in ((3, 8), (4, 7)):
            reservation = host.executor.reserve_response()
            with pytest.raises(ValueError, match="does not match"):
                await host.executor.respond(
                    "turn_mismatch", Transcript(text="Synthetic input.", final=True),
                    authority, reservation=reservation, typed_input=mismatch,
                )
            assert host.executor.reserved_operation_count == 0
        assert host.lineage() == []
        await host.executor.respond(
            "turn_agrees", Transcript(text="Synthetic input.", final=True),
            authority, typed_input=(3, 7),
        )
        assert host.lineage() == [("assistant_turn_completed", "turn_agrees", 3, 7)]
    finally:
        await host.executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("typed_input", [
    [3, 7], (3,), (3, 7, 1), (True, 7), (3, True), (0, 7), (3, 1 << 53), (3.0, 7),
])
async def test_explicit_lineage_is_an_exact_bounded_integer_pair(typed_input: Any) -> None:
    host = _NoCaptureHost()
    host.executor.start()
    try:
        reservation = host.executor.reserve_response()
        with pytest.raises(TypeError, match="exact pair"):
            await host.executor.respond(
                "turn_bad_lineage", Transcript(text="Synthetic input.", final=True),
                reservation=reservation, typed_input=typed_input,
            )
        assert host.executor.reserved_operation_count == 0
        assert host.lineage() == []
    finally:
        await host.executor.close()


@pytest.mark.asyncio
async def test_microphone_routing_refuses_typed_lineage() -> None:
    host = _NoCaptureHost()
    worker = host.bind(10)
    try:
        with pytest.raises(ValueError, match="only typed input"):
            await worker._route_final(
                None, Transcript(text="Synthetic input.", final=True), None,
                typed=False, typed_input=(1, 10),
            )
        assert host.lineage() == []
    finally:
        await worker.close()
