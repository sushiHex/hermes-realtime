"""Typed lineage reaches assistant terminals with no evidence runtime at all (#237).

Composed as the host does with capture off: one shared executor and speech loop,
a command router, and per-binding session workers with no evidence lifecycle. A few
capture-on cases use the real active lifecycle/admission fixture.
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
from hermes_realtime.evidence import InputSource
from hermes_realtime.evidence import models as m
from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, Transcript
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
from tests.conversation.test_worker import (
    CommandRouterProbe,
    EvidenceCommandRouterProbe,
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


_MISMATCH_MARKER = '[typed-lineage] {"version":1,"category":"authority_mismatch"}'


class _CaptureOnHost(_NoCaptureHost):
    """The same composition over the real active lifecycle/admission fixture."""

    def __init__(self, owner_generation: int) -> None:
        self.lifecycle, self.admission, self.writer, _ = _active_replay_evidence(
            owner_generation=owner_generation,
        )
        self.events = []
        self.replay_identities: list[object] = []
        self.context = ConversationContextStore()
        self.director = director_for_test(self.context, _SilentSource())
        self.speech = StreamingSpeechLoop(
            context=self.context, foreground=ForegroundTurnCoordinator(),
            inference=RecordingInference(), synthesizer=Synthesizer(),
            playback=Playback(), ledger=DeliveredSpeechLedger(),
            evidence_admission=self.admission,
            observer=lambda kind, data: self.events.append((kind, dict(data))),
        )
        original_run_turn = self.speech._run_turn

        async def recording_run_turn(*args: Any, **kwargs: Any) -> Any:
            self.replay_identities.append(kwargs.get("replay_identity"))
            return await original_run_turn(*args, **kwargs)

        object.__setattr__(self.speech, "_run_turn", recording_run_turn)
        self.executor = ConversationUpdateExecutor(
            director=self.director, speech=self.speech, max_owned_operations=1,
            operation_scheduler=self.lifecycle.operation_scheduler,
            evidence_lifecycle=self.lifecycle.conversation_authority,
            evidence_admission=self.admission,
        )
        self.evidence_router = EvidenceCommandRouterProbe(self.lifecycle)

    def bind(self, generation: int, *, microphone: bool = False) -> ConversationSessionWorker:
        del microphone
        worker = ConversationSessionWorker(
            participant_identity=_IDENTITY, session_generation=generation,
            vad=SilenceVad(), stt=PreliminaryAndEndpointFinalTranscriber(),
            actions=self.executor, command_router=self.evidence_router,
            evidence_lifecycle=self.lifecycle.conversation_authority,
        )
        worker.start()
        return worker

    def evidence_kinds(self) -> list[object]:
        kinds = []
        while self.writer.ordered_count:
            item = self.writer.get_nowait()
            kinds.append(item.payload.snapshot.event_kind)
            self.admission.complete_ordered_item(item)
        return kinds


def _markers(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[typed-lineage] ")
    ]


@pytest.mark.asyncio
async def test_capture_on_mismatched_lineage_is_withheld_and_the_response_continues(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _CaptureOnHost(240)
    host.executor.start()
    try:
        authority = host.lifecycle.decline_to_user(
            host.lifecycle.mint_final_input(
                source=InputSource.TYPED, input_incarnation=1,
                media_incarnation=None, typed_sequence=1,
            )
        )
        # The authority's tuple is (1, 1); neither it nor the explicit one is used.
        await host.executor.respond(
            "turn_mismatch", Transcript(text="Synthetic typed input.", final=True),
            authority, reservation=host.executor.reserve_response(), typed_input=(1, 2),
        )
        assert host.lineage() == [("assistant_turn_completed", "turn_mismatch", None, None)]
        assert _markers(capsys) == [_MISMATCH_MARKER]
        assert host.executor.reserved_operation_count == 0
        assert host.admission.diagnostics().active_lease_count == 0
        kinds = host.evidence_kinds()
        assert kinds[0] is m.EventKind.TURN_OPENED
        assert kinds[-1] is m.EventKind.TURN_SETTLED
        assert len(host.replay_identities) == 1 and host.replay_identities[0] is not None
    finally:
        await host.executor.close()


@pytest.mark.asyncio
async def test_capture_on_worker_and_authority_tuples_agree_and_publish_lineage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _CaptureOnHost(241)
    worker = host.bind(1)
    try:
        await _type(worker, 1, 1)
        await worker.wait_for_responses()
        [authority] = host.evidence_router.user_authorities
        assert (authority.typed_sequence, authority.binding_generation) == (1, 1)
        assert host.lineage() == [("assistant_turn_completed", "session_1_turn_1", 1, 1)]
        assert _markers(capsys) == []
        assert m.EventKind.TURN_OPENED in host.evidence_kinds()
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_capture_on_binding_rollover_refuses_stale_submit_and_rebinds_lineage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _CaptureOnHost(242)
    first = host.bind(1)
    await _type(first, 1, 1)
    await first.wait_for_responses()
    await first.close_binding()
    host.evidence_kinds()
    host.lifecycle.activate_binding(
        binding_id=str(UUID(int=905, version=4)), binding_generation=2,
        consent_epoch_id=str(UUID(int=903, version=4)),
        logical_session_id=str(UUID(int=904, version=4)),
    )
    second = host.bind(2)
    try:
        with pytest.raises(RuntimeError, match="stale session generation"):
            await _type(second, 1, 2)
        await _type(second, 2, 1)
        await second.wait_for_responses()
        assert [
            (a.typed_sequence, a.binding_generation)
            for a in host.evidence_router.user_authorities
        ] == [(1, 1), (1, 2)]
        assert host.lineage() == [
            ("assistant_turn_completed", "session_1_turn_1", 1, 1),
            ("assistant_turn_completed", "session_2_turn_1", 1, 2),
        ]
        assert _markers(capsys) == []
    finally:
        await second.close()


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
