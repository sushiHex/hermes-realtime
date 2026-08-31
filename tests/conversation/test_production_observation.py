import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import FrozenInstanceError, asdict, fields
from pathlib import Path
from threading import Event, Thread

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk, Transcript


def test_observations_are_exact_per_kind_immutable_dtos() -> None:
    from hermes_realtime.evidence import TerminalDisposition, TerminalReason
    from hermes_realtime.production_observation import (
        CloseResultV1,
        CloseStageObservationV1,
        CloseStageV1,
        ContextCommittedObservationV1,
        ObservationKindV1,
        TerminalSettledObservationV1,
    )

    context = ContextCommittedObservationV1(context_committed=True)
    terminal = TerminalSettledObservationV1(
        terminal_disposition=TerminalDisposition.COMPLETED,
        terminal_reason=TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
        context_committed=True,
    )
    closed = CloseStageObservationV1(
        stage=CloseStageV1.SPEECH_LOOP,
        result=CloseResultV1.SUCCEEDED,
    )

    assert context.kind is ObservationKindV1.CONTEXT_COMMITTED
    assert terminal.kind is ObservationKindV1.TERMINAL_SETTLED
    assert closed.kind is ObservationKindV1.CLOSE_STAGE
    assert tuple(field.name for field in fields(ContextCommittedObservationV1)) == (
        "context_committed",
    )
    assert tuple(field.name for field in fields(TerminalSettledObservationV1)) == (
        "terminal_disposition",
        "terminal_reason",
        "context_committed",
    )
    serialized = json.dumps((asdict(context), asdict(terminal)), default=str, sort_keys=True)
    for forbidden in (
        "context_digest",
        "context_bytes",
        "generated_segment_count",
        "queued_chunk_count",
        "started_chunk_count",
        "transport_confirmed_full_count",
    ):
        assert forbidden not in serialized
    with pytest.raises(FrozenInstanceError):
        terminal.context_committed = False  # type: ignore[misc]


def test_close_attempt_history_consumes_capacity_without_rewriting_failure() -> None:
    from hermes_realtime.production_observation import (
        CloseResultV1,
        CloseStageV1,
        _new_observation_channel,
    )

    observations, recorder = _new_observation_channel()
    for _ in range(254):
        recorder.record_context_committed()
    recorder.record_close_stage(stage=CloseStageV1.BROWSER_CLIENT, result=CloseResultV1.FAILED)
    recorder.record_close_stage(stage=CloseStageV1.BROWSER_CLIENT, result=CloseResultV1.SUCCEEDED)

    assert [(item.stage.value, item.result.value) for item in observations.records()[-2:]] == [
        ("browser_client", "failed"),
        ("browser_client", "succeeded"),
    ]
    assert observations.status().trace_complete is True

    recorder.record_close_stage(stage=CloseStageV1.BROWSER_CLIENT, result=CloseResultV1.SUCCEEDED)
    assert len(observations.records()) == 256
    assert observations.status().trace_complete is False


def test_raw_qualification_trace_is_internal_and_public_constructors_expose_no_injection() -> None:
    import hermes_realtime.production_observation as observations
    from hermes_realtime.host_launcher import build_local_host_launcher
    from hermes_realtime.launcher import LocalBrowserLauncher
    from hermes_realtime.testing import harness

    assert "qualification" not in str(inspect.signature(StreamingSpeechLoop))
    assert "qualification" not in str(inspect.signature(LocalBrowserLauncher))
    assert "writer_transport" not in str(inspect.signature(build_local_host_launcher))
    assert not hasattr(StreamingSpeechLoop, "deterministic_qualification_trace")
    assert not hasattr(LocalBrowserLauncher, "deterministic_qualification_trace")
    assert not hasattr(observations, "_new_qualification_trace")
    assert not hasattr(harness, "InProcessQualificationComposition")
    qualification_source = (
        Path(__file__).parents[2] / "src" / "hermes_realtime" / "_qualification.py"
    )
    source = qualification_source.read_text(encoding="utf-8")
    for forbidden in (
        "_QualificationTraceState",
        "_QualificationTraceV1",
        "_QualificationCompositionBundleV1",
        "_QualificationHostRegistrationV1",
        "committed_conversation_context_snapshot_bytes",
        "compose_host",
    ):
        assert forbidden not in source


def test_committed_conversation_context_snapshot_bytes_preserve_ordered_nonempty_updates() -> None:
    from tests.support.qualification import committed_conversation_context_snapshot_bytes

    assert committed_conversation_context_snapshot_bytes(
        revision=4,
        messages=(("user", "Give me an update."),),
        active_tasks=(("task_alpha", "Build alpha."),),
        terminal_task_count=2,
        updates=(
            (7, "task_beta", "running", "Beta is running."),
            (8, "task_alpha", "completed", "Alpha is complete."),
        ),
    ) == (
        b'{"activeTasks":[["task_alpha","Build alpha."]],'
        b'"messages":[["user","Give me an update."]],"revision":4,'
        b'"terminalTaskCount":2,"updates":[[7,"task_beta","running",'
        b'"Beta is running."],[8,"task_alpha","completed","Alpha is complete."]]}'
    )


@pytest.mark.asyncio
async def test_public_host_composition_exposes_only_metadata_and_unique_close_result(
    tmp_path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=71,
        retention_hours=24,
    )
    try:
        observations = runtime.production_observations

        assert not hasattr(observations, "_channel")
        assert observations.status().trace_complete is True

        await runtime.close()
        await runtime.close()

        assert [(item.stage.value, item.result.value) for item in observations.records()] == [
            ("evidence_runtime", "succeeded")
        ]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_qualification_host_return_is_emitted_once_at_the_top_level_boundary() -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    events: list[str] = []

    class Runtime:
        async def start(self) -> str:
            events.append("start")
            return "http://127.0.0.1:8765/#bootstrap=" + ("r" * 43)

        async def close(self) -> None:
            events.append("runtime_close")

    class Verifier:
        async def connect(self, _room_name: str, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds
            events.append("connect")

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds
            events.append("disconnect")

    class Provider:
        async def close(self) -> None:
            events.append("provider_close")

    qualification = InProcessQualificationComposition()
    host = qualification.compose_host(
        lambda: LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=lambda: _complete(),
            providers=(Provider(),),
        )
    )

    url = await qualification.run_host(host)

    assert url.startswith("http://127.0.0.1:8765/")
    assert events == ["connect", "start", "runtime_close", "disconnect", "provider_close"]
    assert [
        (record.kind.value, record.outcome.value) for record in qualification.trace.records()
    ] == [("host_return", "returned")]


@pytest.mark.asyncio
async def test_qualification_registered_host_can_run_inside_one_owner_bound_lease() -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    events: list[str] = []

    class Runtime:
        async def start(self) -> str:
            events.append("runtime_start")
            return "http://127.0.0.1:8765/#bootstrap=private"

        async def close(self) -> None:
            events.append("runtime_close")

    class Verifier:
        async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None:
            del room_name, timeout_seconds
            events.append("verifier_connect")

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds
            events.append("verifier_disconnect")

    qualification = InProcessQualificationComposition()
    registration = qualification.compose_host(
        lambda: LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=lambda: asyncio.sleep(0),
            providers=(),
        )
    )

    running = await qualification.start_host(registration)
    assert running.url.startswith("http://127.0.0.1:8765/")
    assert events == ["verifier_connect", "runtime_start"]

    await qualification.close_host(running)
    assert events == [
        "verifier_connect",
        "runtime_start",
        "runtime_close",
        "verifier_disconnect",
    ]
    assert [
        (record.kind.value, record.outcome.value) for record in qualification.trace.records()
    ] == [("host_return", "returned")]


@pytest.mark.asyncio
async def test_qualification_owner_bound_running_lease_exposes_only_its_immutable_observations(
) -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from hermes_realtime.production_observation import ProductionObservationViewV1
    from tests.support.qualification import InProcessQualificationComposition

    class Runtime:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=private"

        async def close(self) -> None:
            return None

    class Verifier:
        async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None:
            del room_name, timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    built: list[LocalBrowserLauncher] = []

    def build() -> LocalBrowserLauncher:
        launcher = LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=_complete,
            providers=(),
        )
        built.append(launcher)
        return launcher

    qualification = InProcessQualificationComposition()
    running = await qualification.start_host(qualification.compose_host(build))

    observations = qualification.production_observations(running)
    assert type(observations) is ProductionObservationViewV1
    assert observations is built[0].production_observations
    assert observations.status().trace_complete
    assert not hasattr(observations, "runtime")
    assert not hasattr(observations, "controller")
    assert not hasattr(observations, "queue")

    await qualification.close_host(running)
    assert qualification.production_observations(running) is observations


@pytest.mark.asyncio
async def test_observation_accessor_rejects_foreign_fabricated_and_registration_leases(
) -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    class Runtime:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=private"

        async def close(self) -> None:
            return None

    class Verifier:
        async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None:
            del room_name, timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    def build() -> LocalBrowserLauncher:
        return LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=_complete,
            providers=(),
        )

    first = InProcessQualificationComposition()
    second = InProcessQualificationComposition()
    foreign = await second.start_host(second.compose_host(build))
    registration = first.compose_host(build)
    try:
        with pytest.raises(TypeError, match="running host"):
            first.production_observations(object())
        with pytest.raises(TypeError, match="running host"):
            first.production_observations(registration)
        with pytest.raises(ValueError, match="same qualification bundle"):
            first.production_observations(foreign)
    finally:
        await second.close_host(foreign)


def test_qualification_full_host_composition_wires_closed_named_dependencies() -> None:
    from hermes_realtime._qualification import _current_qualification_full_host_dependencies
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    sentinels = {name: object() for name in ("inference", "speech", "synth", "stt", "vad")}

    class Runtime:
        async def start(self) -> str:
            raise AssertionError("dependency composition host must not start")

        async def close(self) -> None:
            return None

    class Verifier:
        async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None:
            del room_name, timeout_seconds
            raise AssertionError("dependency composition verifier must not connect")

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    def compose() -> LocalBrowserLauncher:
        dependencies = _current_qualification_full_host_dependencies()
        assert dependencies is not None
        assert dependencies.take() is dependencies
        assert dependencies.inference_factory() is sentinels["inference"]
        assert dependencies.speech_presence_factory() is sentinels["speech"]
        assert dependencies.synthesizer_factory() is sentinels["synth"]
        assert dependencies.transcriber_factory() is sentinels["stt"]
        assert dependencies.vad_factory() is sentinels["vad"]
        assert dependencies.identity_factory() == "qualification-identity"
        return LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=lambda: asyncio.sleep(0),
            providers=(),
        )

    qualification = InProcessQualificationComposition()
    registration = qualification.compose_full_host(
        compose,
        inference_factory=lambda: sentinels["inference"],
        speech_presence_factory=lambda: sentinels["speech"],
        synthesizer_factory=lambda: sentinels["synth"],
        transcriber_factory=lambda: sentinels["stt"],
        vad_factory=lambda: sentinels["vad"],
        identity_factory=lambda: "qualification-identity",
    )

    assert registration is not None
    assert _current_qualification_full_host_dependencies() is None


def test_qualification_capacity_probe_is_private_owner_bound_and_not_a_launcher_option() -> None:
    """Capacity facts are a closed qualification read surface, never host authority."""

    import inspect

    from hermes_realtime._qualification import _current_qualification_full_host_dependencies
    from hermes_realtime.evidence.admission import EvidenceAdmissionControllerV1
    from hermes_realtime.host_launcher import build_local_host_launcher
    from tests.support.qualification import InProcessQualificationComposition

    assert "capacity" not in inspect.signature(build_local_host_launcher).parameters
    assert "qualification_capacity_probe" not in inspect.signature(
        EvidenceAdmissionControllerV1
    ).parameters
    qualification = InProcessQualificationComposition()
    assert callable(qualification.capacity_observations)
    assert _current_qualification_full_host_dependencies() is None
    assert not hasattr(qualification, "capacity_probe")


@pytest.mark.asyncio
async def test_qualification_host_return_never_reports_returned_after_cleanup_failure() -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    class Runtime:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=" + ("r" * 43)

        async def close(self) -> None:
            raise RuntimeError("cleanup failed")

    class Verifier:
        async def connect(self, _room_name: str, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    qualification = InProcessQualificationComposition()
    host = qualification.compose_host(
        lambda: LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=_complete,
            providers=(),
        )
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await qualification.run_host(host)

    assert [
        (record.kind.value, record.outcome.value) for record in qualification.trace.records()
    ] == [("host_return", "failed")]


async def _complete() -> None:
    return None


class _Inference:
    async def _stream(self, *, turn_id: str) -> AsyncIterator[str]:
        yield f"authoritative answer for {turn_id}"

    def stream(self, _snapshot: object, *, turn_id: str) -> AsyncIterator[str]:
        return self._stream(turn_id=turn_id)

    async def cancel(self, _turn_id: str) -> None:
        return None


class _Synthesizer:
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id="chunk_001",
            text=text,
            audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=16_000, channels=1),
        )

    async def cancel(self, _turn_id: str) -> None:
        return None


class _Playback:
    async def play(
        self,
        _chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        assert is_valid()

    async def cancel(self, _turn_id: str) -> None:
        return None


class _ManySegmentInference(_Inference):
    async def _stream(self, *, turn_id: str) -> AsyncIterator[str]:
        del turn_id
        for index in range(129):
            yield f"segment {index}"


class _ManySegmentSynthesizer(_Synthesizer):
    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"chunk_{text.rsplit(' ', maxsplit=1)[1]}",
            text=text,
            audio=AudioFrame(pcm=b"\x00\x00", sample_rate_hz=16_000, channels=1),
        )


@pytest.mark.asyncio
async def test_disabled_foreground_keeps_private_exact_source_trace_without_terminal_evidence() -> (
    None
):
    from tests.support.qualification import (
        InProcessQualificationComposition,
        committed_conversation_context_snapshot_bytes,
    )

    qualification = InProcessQualificationComposition()
    loop = qualification.compose(
        lambda: StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(),
            inference=_Inference(),
            synthesizer=_Synthesizer(),
            playback=_Playback(),
            ledger=DeliveredSpeechLedger(),
        )
    )

    await loop.respond("turn_001", Transcript(text="What happened?", final=True))

    records = qualification.trace.records()
    assert qualification.trace.status().trace_complete is True
    assert not any(record.kind.value == "terminal_settled" for record in records)
    assert records[0].committed_conversation_context_snapshot == (
        committed_conversation_context_snapshot_bytes(
            revision=1,
            messages=(("user", "What happened?"),),
            active_tasks=(),
            terminal_task_count=0,
        )
    )
    assert records[1].generated_text == b"authoritative answer for turn_001"
    assert records[2].confirmed_text == b"authoritative answer for turn_001"

    await loop.close()


@pytest.mark.asyncio
async def test_bounded_private_trace_reports_incomplete_without_evicting_earlier_context() -> None:
    from tests.support.qualification import InProcessQualificationComposition

    qualification = InProcessQualificationComposition()
    loop = qualification.compose(
        lambda: StreamingSpeechLoop(
            context=ConversationContextStore(),
            foreground=ForegroundTurnCoordinator(output_capacity=129),
            inference=_ManySegmentInference(),
            synthesizer=_ManySegmentSynthesizer(),
            playback=_Playback(),
            ledger=DeliveredSpeechLedger(),
            max_segments=129,
        )
    )

    await loop.respond("turn_002", Transcript(text="Fill the trace.", final=True))

    records = qualification.trace.records()
    assert len(records) == 256
    assert qualification.trace.status().trace_complete is False
    assert records[0].kind.value == "committed_conversation_context_snapshot"

    await loop.close()


def test_trace_contention_latches_incomplete_without_a_later_writer_reverting_it() -> None:
    """Private trace mechanics retain a monotonic loss result under contention."""

    from hermes_realtime.production_observation import (
        _new_bounded_callbacks,
        _new_trace_state,
    )
    from tests.support.qualification import _new_trace_pair

    production_state = _new_trace_state()
    production_append, _production_records, production_status = _new_bounded_callbacks(
        production_state
    )
    qualification_trace, qualification_recorder = _new_trace_pair()

    production_state.lock.acquire()
    qualification_trace._state.lock.acquire()
    try:
        production_finished = Event()
        qualification_finished = Event()
        production_thread = Thread(
            target=lambda: (production_append(object()), production_finished.set()),
        )
        qualification_thread = Thread(
            target=lambda: (
                qualification_recorder.record_generated_text(b"lost"),
                qualification_finished.set(),
            ),
        )
        production_thread.start()
        qualification_thread.start()
        assert production_finished.wait(timeout=1)
        assert qualification_finished.wait(timeout=1)
    finally:
        production_state.lock.release()
        qualification_trace._state.lock.release()

    production_append(object())
    qualification_recorder.record_generated_text(b"retained")

    assert production_status().trace_complete is False
    assert qualification_trace.status().trace_complete is False
    assert qualification_trace.records()[-1].generated_text == b"retained"


@pytest.mark.asyncio
async def test_host_return_uses_an_independent_terminal_slot_at_trace_capacity() -> None:
    """HOST_RETURN is never lost with ordinary trace records or contention."""

    from tests.support.qualification import InProcessQualificationComposition

    qualification = InProcessQualificationComposition()
    for _ in range(256):
        qualification._collector.record_generated_text(b"ordinary")
    qualification._collector.record_generated_text(b"dropped")

    class Runtime:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=" + ("r" * 43)

        async def close(self) -> None:
            return None

    class Verifier:
        async def connect(self, _room_name: str, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    from hermes_realtime.launcher import LocalBrowserLauncher

    registration = qualification.compose_host(
        lambda: LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=_complete,
            providers=(),
        )
    )
    await qualification.run_host(registration)

    records = qualification.trace.records()
    assert len(records) == 257
    assert records[-1].kind.value == "host_return"
    assert records[-1].outcome.value == "returned"
    assert qualification.trace.status().trace_complete is False


@pytest.mark.asyncio
async def test_host_return_terminal_slot_records_cancellation_once() -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import (
        InProcessQualificationComposition,
        _HostReturnQualificationObservationV1,
    )

    started = asyncio.Event()
    closed = asyncio.Event()

    class Runtime:
        async def start(self) -> str:
            started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def close(self) -> None:
            closed.set()

    class Verifier:
        async def connect(self, _room_name: str, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    qualification = InProcessQualificationComposition()
    registration = qualification.compose_host(
        lambda: LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=_complete,
            providers=(),
        )
    )
    task = asyncio.create_task(qualification.run_host(registration))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()

    records = qualification.trace.records()
    assert len(records) == 1
    record = records[0]
    assert type(record) is _HostReturnQualificationObservationV1
    assert record.kind.value == "host_return"
    assert record.outcome.value == "cancelled"


@pytest.mark.asyncio
async def test_qualification_rejects_a_host_registration_from_another_bundle() -> None:
    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    class Host:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=" + ("r" * 43)

        async def close(self) -> None:
            return None

    first = InProcessQualificationComposition()
    second = InProcessQualificationComposition()
    with pytest.raises(TypeError, match="exact composed LocalBrowserLauncher"):
        first.compose_host(Host)

    class Runtime:
        async def start(self) -> str:
            return "http://127.0.0.1:8765/#bootstrap=" + ("r" * 43)

        async def close(self) -> None:
            return None

    class Verifier:
        async def connect(self, _room_name: str, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    registration = first.compose_host(
        lambda: LocalBrowserLauncher(
            runtime=Runtime(),
            verifier=Verifier(),
            room_name="qualification",
            preflight=_complete,
            providers=(),
        )
    )
    prebuilt = LocalBrowserLauncher(
        runtime=Runtime(),
        verifier=Verifier(),
        room_name="qualification",
        preflight=_complete,
        providers=(),
    )
    fabricated = object.__new__(LocalBrowserLauncher)

    with pytest.raises(TypeError, match="exact composed LocalBrowserLauncher"):
        first.compose_host(lambda: prebuilt)
    with pytest.raises(TypeError, match="exact composed LocalBrowserLauncher"):
        first.compose_host(lambda: fabricated)
    with pytest.raises(TypeError, match="exact composed LocalBrowserLauncher"):
        first.compose_host(lambda: registration._host)
    with pytest.raises(TypeError, match="exact composed LocalBrowserLauncher"):
        second.compose_host(lambda: registration._host)
    with pytest.raises(TypeError, match="test-owner host registration"):
        await first.run_host(lambda: Host())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same qualification bundle"):
        await second.run_host(registration)
    with pytest.raises(ValueError, match="different owner"), first._capability.wire(
        second._collector
    ):
        pass


def test_generic_qualification_composition_rejects_host_construction() -> None:
    """Only compose_host may construct a launcher that can later be run."""

    from hermes_realtime.launcher import LocalBrowserLauncher
    from tests.support.qualification import InProcessQualificationComposition

    calls: list[str] = []

    class Runtime:
        async def start(self) -> str:
            calls.append("start")
            return "http://127.0.0.1:8765/#bootstrap=" + ("r" * 43)

        async def close(self) -> None:
            calls.append("close")

    class Verifier:
        async def connect(self, _room_name: str, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

        async def disconnect(self, *, timeout_seconds: float = 10) -> None:
            del timeout_seconds

    qualification = InProcessQualificationComposition()
    assert qualification.compose(lambda: "ordinary") == "ordinary"
    with pytest.raises(RuntimeError, match="compose_host"):
        qualification.compose(
            lambda: LocalBrowserLauncher(
                runtime=Runtime(),
                verifier=Verifier(),
                room_name="qualification",
                preflight=_complete,
                providers=(),
            )
        )
    assert calls == []
def test_rollover_observation_is_closed_content_free_and_order_checked() -> None:
    from dataclasses import fields

    from hermes_realtime.production_observation import (
        ObservationKindV1,
        RolloverObservationV1,
        RolloverResultV1,
        RolloverStageV1,
        _new_observation_channel,
    )

    view, recorder = _new_observation_channel()
    for stage, result in (
        (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.QUEUED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.DURABLE, RolloverResultV1.COMMITTED),
        (RolloverStageV1.PUBLISHED, RolloverResultV1.COMMITTED),
        (RolloverStageV1.TERMINAL, RolloverResultV1.COMMITTED),
    ):
        recorder.record_rollover(stage=stage, result=result)
    records = view.records()
    assert all(type(record) is RolloverObservationV1 for record in records)
    assert all(record.kind is ObservationKindV1.ROLLOVER for record in records)
    assert tuple(field.name for field in fields(RolloverObservationV1)) == (
        "stage",
        "result",
    )
    assert view.status().trace_complete is True

    recorder.record_rollover(
        stage=RolloverStageV1.DURABLE,
        result=RolloverResultV1.IDEMPOTENT,
    )
    assert view.status().trace_complete is False
    assert view.records() == records
