import asyncio
import threading
from types import MethodType, SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from hermes_realtime.conversation import ConversationUpdateExecutor
from hermes_realtime.conversation.ingress import BoundedPcmIngress, IngressRecord
from hermes_realtime.conversation.telemetry import UtteranceTicket
from hermes_realtime.conversation.worker import (
    ConversationSessionWorker,
    ReconnectSafeConversationWorker,
)
from hermes_realtime.evidence import (
    CommandRoutingOutcome,
    CommandRoutingResultV1,
    ConversationOperationKind,
    ConversationOperationScheduler,
    FinalInputAuthorityV1,
    InputSource,
    UserTurnAuthorityV1,
)
from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner
from hermes_realtime.speech import (
    AudioFrame,
    ParticipantAudioFrame,
    SpeechPresence,
    SpeechPresenceVerifier,
    Transcript,
    VoiceActivity,
)


class ScriptedVad:
    def __init__(self, *activities: VoiceActivity) -> None:
        selected = activities or (
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_ENDED,
        )
        self._activities = iter(selected)

    def process(self, frame: AudioFrame) -> VoiceActivity:
        del frame
        return next(self._activities)


class FinalOnPushTranscriber:
    def __init__(self) -> None:
        self.push_calls = 0
        self.cancel_calls = 0

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        self.push_calls += 1
        return (Transcript(text="must-not-be-admitted", final=True),)

    async def finish_utterance(self) -> Transcript:
        raise AssertionError("finish should not be reached")

    async def cancel(self) -> None:
        self.cancel_calls += 1


class PreliminaryAndEndpointFinalTranscriber:
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        return (Transcript(text="preliminary final", final=True),)

    async def finish_utterance(self) -> Transcript:
        return Transcript(text="authoritative endpoint final", final=True)

    async def cancel(self) -> None:
        return None


class BlockingTranscriber:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_calls = 0

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        self.started.set()
        await self.release.wait()
        return ()

    async def finish_utterance(self) -> Transcript:
        raise AssertionError("utterance should not finish during disconnect")

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self.release.set()


class SilenceVad:
    def process(self, frame: AudioFrame) -> VoiceActivity:
        del frame
        return VoiceActivity.SILENCE


class ScriptedEchoGuard:
    def __init__(
        self,
        *decisions: bool,
        transcript_echoes: tuple[str, ...] = (),
        recent_playback: bool = False,
    ) -> None:
        self._decisions = iter(decisions)
        self._transcript_echoes = frozenset(transcript_echoes)
        self._recent_playback = recent_playback
        self.windows: list[tuple[AudioFrame, ...]] = []
        self.transcripts: list[str] = []

    def is_echo_dominated(self, frames: tuple[AudioFrame, ...]) -> bool:
        self.windows.append(frames)
        return next(self._decisions)

    def is_transcript_echo(self, text: str) -> bool:
        self.transcripts.append(text)
        return text in self._transcript_echoes

    def has_recent_playback(self) -> bool:
        return self._recent_playback


class ScriptedSpeechPresenceVerifier:
    def __init__(self, *decisions: SpeechPresence | BaseException) -> None:
        self._decisions = iter(decisions)
        self.windows: list[tuple[AudioFrame, ...]] = []

    def classify(self, frames: tuple[AudioFrame, ...]) -> SpeechPresence:
        self.windows.append(frames)
        decision = next(self._decisions)
        if isinstance(decision, BaseException):
            raise decision
        return decision


class BlockingSpeechPresenceVerifier(SpeechPresenceVerifier):
    def __init__(self, decision: SpeechPresence | BaseException) -> None:
        self._decision = decision
        self.entered = threading.Event()
        self.release = threading.Event()

    def classify(self, frames: tuple[AudioFrame, ...]) -> SpeechPresence:
        del frames
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release speech verifier")
        if isinstance(self._decision, BaseException):
            raise self._decision
        return self._decision


class OrderedTranscriber:
    def __init__(self, foreground_cancelled: asyncio.Event) -> None:
        self._foreground_cancelled = foreground_cancelled
        self.push_calls = 0
        self.cancel_calls = 0

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        assert self._foreground_cancelled.is_set()
        self.push_calls += 1
        return ()

    async def finish_utterance(self) -> Transcript:
        return Transcript(text="What is the status?", final=True)

    async def cancel(self) -> None:
        self.cancel_calls += 1


class NoSpeechTranscriber:
    def __init__(self) -> None:
        self.cancel_calls = 0

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        return ()

    async def finish_utterance(self) -> Transcript | None:
        return None

    async def cancel(self) -> None:
        self.cancel_calls += 1


class PartialThenFinalTranscriber(NoSpeechTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self.push_calls = 0

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        self.push_calls += 1
        if self.push_calls == 1:
            return (Transcript(text="Can you give", final=False),)
        return ()

    async def finish_utterance(self) -> Transcript:
        return Transcript(text="Can you give me a synopsis?", final=True)


class CapturingTranscriber(NoSpeechTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self.frames: list[AudioFrame] = []

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        self.frames.append(frame)
        return ()


class CapturingFinalTranscriber(CapturingTranscriber):
    async def finish_utterance(self) -> Transcript:
        return Transcript(text="Stop now please.", final=True)


class CapturingTextFinalTranscriber(CapturingTranscriber):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    async def finish_utterance(self) -> Transcript:
        return Transcript(text=self._text, final=True)


class CapturingPartialThenFinalTranscriber(CapturingFinalTranscriber):
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        self.frames.append(frame)
        if len(self.frames) == 1:
            return (Transcript(text="Stop now", final=False),)
        return ()


class CapturingPartialNoFinalTranscriber(CapturingTranscriber):
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        self.frames.append(frame)
        if len(self.frames) == 1:
            return (Transcript(text="Maybe", final=False),)
        return ()


class CapturingShortPartialNoFinalTranscriber(CapturingTranscriber):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        self.frames.append(frame)
        if len(self.frames) == 1:
            return (Transcript(text=self._text, final=False),)
        return ()


class CapturingPushFinalNoEndpointTranscriber(CapturingTranscriber):
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        self.frames.append(frame)
        if len(self.frames) == 1:
            return (Transcript(text="Actual interruption.", final=True),)
        return ()


class CommandRouterProbe:
    def __init__(self, *, handled: bool) -> None:
        self.handled = handled
        self.texts: list[str] = []

    async def route(self, text: str) -> bool:
        self.texts.append(text)
        return self.handled


class EvidenceCommandRouterProbe:
    def __init__(self, owner: EvidenceLifecycleOwner) -> None:
        self._owner = owner
        self.calls: list[tuple[str, FinalInputAuthorityV1]] = []
        self.user_authorities: list[UserTurnAuthorityV1] = []

    async def route(
        self,
        text: str,
        authority: FinalInputAuthorityV1,
    ) -> CommandRoutingResultV1:
        self.calls.append((text, authority))
        user_authority = self._owner.decline_to_user(authority)
        self.user_authorities.append(user_authority)
        return CommandRoutingResultV1(
            outcome=CommandRoutingOutcome.NOT_COMMAND,
            command_disposition=None,
            user_turn_authority=user_authority,
        )


def action_executor_probe() -> tuple[
    ConversationUpdateExecutor,
    asyncio.Event,
    list[tuple[str, Transcript]],
]:
    executor = object.__new__(ConversationUpdateExecutor)
    foreground = SimpleNamespace(
        foreground_active=True,
        active_turn_id="foreground_1",
        resume_count=0,
    )
    object.__setattr__(executor, "_speech", foreground)
    object.__setattr__(
        executor,
        "_operation_scheduler",
        ConversationOperationScheduler(owner_generation=1),
    )
    object.__setattr__(executor, "_started", True)
    object.__setattr__(executor, "_closed", False)
    object.__setattr__(executor, "_terminal_error", None)
    foreground_cancelled = asyncio.Event()
    responses: list[tuple[str, Transcript]] = []

    def start(self: ConversationUpdateExecutor) -> None:
        del self

    async def cancel_foreground(self: ConversationUpdateExecutor) -> None:
        del self
        foreground.foreground_active = False
        foreground.active_turn_id = None
        foreground_cancelled.set()

    async def cancel_foreground_if(
        self: ConversationUpdateExecutor,
        turn_id: str,
    ) -> bool:
        del self
        if foreground.active_turn_id != turn_id or not foreground.foreground_active:
            return False
        foreground.foreground_active = False
        foreground.active_turn_id = None
        foreground_cancelled.set()
        return True

    async def cancel_foreground_for_cleanup(
        self: ConversationUpdateExecutor,
        *,
        host_shutdown: bool = False,
        binding_closed: bool = False,
    ) -> None:
        del self, host_shutdown, binding_closed
        foreground_cancelled.set()

    async def resume_foreground(self: ConversationUpdateExecutor) -> bool:
        del self
        foreground.foreground_active = True
        foreground.resume_count += 1
        return True

    async def respond(
        self: ConversationUpdateExecutor,
        turn_id: str,
        transcript: Transcript,
        authority: UserTurnAuthorityV1 | None = None,
        *,
        reservation: Any | None = None,
    ) -> None:
        del authority
        responses.append((turn_id, transcript))
        if reservation is not None:
            self._operation_scheduler.release(reservation)

    async def close(self: ConversationUpdateExecutor) -> None:
        del self

    executor.start = MethodType(start, executor)  # type: ignore[method-assign]
    executor.cancel_foreground = MethodType(  # type: ignore[method-assign]
        cancel_foreground,
        executor,
    )
    executor.cancel_foreground_if = MethodType(  # type: ignore[method-assign]
        cancel_foreground_if,
        executor,
    )
    executor._cancel_foreground_for_cleanup = MethodType(  # type: ignore[method-assign]
        cancel_foreground_for_cleanup,
        executor,
    )
    executor.resume_foreground = MethodType(  # type: ignore[method-assign]
        resume_foreground,
        executor,
    )
    executor.respond = MethodType(respond, executor)  # type: ignore[method-assign]
    executor.close = MethodType(close, executor)  # type: ignore[method-assign]
    return executor, foreground_cancelled, responses


def evidence_action_executor_probe() -> tuple[
    ConversationUpdateExecutor,
    list[tuple[str, Transcript, UserTurnAuthorityV1]],
]:
    executor, _, _ = action_executor_probe()
    responses: list[tuple[str, Transcript, UserTurnAuthorityV1]] = []

    async def respond(
        self: ConversationUpdateExecutor,
        turn_id: str,
        transcript: Transcript,
        authority: UserTurnAuthorityV1,
        *,
        reservation: Any | None = None,
    ) -> None:
        responses.append((turn_id, transcript, authority))
        if reservation is not None:
            self._operation_scheduler.release(reservation)

    executor.respond = MethodType(respond, executor)  # type: ignore[method-assign]
    return executor, responses


@pytest.mark.asyncio
async def test_session_worker_confirms_server_audio_path_on_first_pcm_frame() -> None:
    executor, _, _ = action_executor_probe()
    ready: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        on_audio_ready=lambda: ready.append("ready"),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x00\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 7, frame)
    await worker.receive_audio("browser_user", 7, frame)

    assert ready == ["ready"]
    await worker.close()


@pytest.mark.asyncio
async def test_session_worker_reconfirms_server_audio_path_periodically() -> None:
    executor, _, _ = action_executor_probe()
    ready: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        on_audio_ready=lambda: ready.append("ready"),
        audio_ready_repeat_frames=2,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x00\x00" * 160, sample_rate_hz=16_000, channels=1)

    for _ in range(5):
        await worker.receive_audio("browser_user", 7, frame)

    assert ready == ["ready", "ready", "ready"]
    await worker.close()


@pytest.mark.asyncio
async def test_advisory_audio_callbacks_cannot_terminalize_pcm() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = CapturingTranscriber()

    def fail_ready() -> None:
        raise RuntimeError("ready projection failed")

    def fail_voice_activity(_active: bool) -> None:
        raise RuntimeError("voice projection failed")

    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        on_audio_ready=fail_ready,
        on_voice_activity=fail_voice_activity,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x00\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 7, frame)
    await worker.receive_audio("browser_user", 7, frame)

    assert transcriber.frames == [frame, frame]
    await worker.close()


@pytest.mark.asyncio
async def test_echo_only_observer_failure_cannot_terminalize_pcm() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    guard = ScriptedEchoGuard(True)

    def fail_echo_observation(kind: str, _data: dict[str, str | int | bool | None]) -> None:
        if kind == "echo_suppressed":
            raise RuntimeError("echo projection failed")

    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SILENCE),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        observer=fail_echo_observation,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x00\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 7, frame)
    await worker.receive_audio("browser_user", 7, frame)

    assert transcriber.frames == []
    await worker.close()


@pytest.mark.asyncio
async def test_session_worker_routes_endpointed_audio_to_foreground_response() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    voice_activity: list[bool] = []

    def observe(kind: str, data: dict[str, str | int | bool | None]) -> None:
        observed.append((kind, data))

    transcriber = OrderedTranscriber(foreground_cancelled)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        vad=ScriptedVad(),
        stt=transcriber,
        actions=executor,
        observer=observe,
        on_voice_activity=voice_activity.append,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 7, frame)
    await worker.receive_audio("browser_user", 7, frame)
    await worker.wait_for_responses()

    assert transcriber.push_calls == 2
    assert voice_activity == [True, False]
    assert responses == [
        (
            "session_7_turn_1",
            Transcript(text="What is the status?", final=True),
        )
    ]
    assert observed == [
        ("interrupt_requested", {"turnId": "foreground_1"}),
        ("playback_silenced", {}),
        ("speech_ended", {}),
        ("transcript_final", {"role": "user", "text": "What is the status?"}),
    ]
    await worker.close()
    assert transcriber.cancel_calls == 1


@pytest.mark.asyncio
async def test_only_endpoint_final_authorizes_one_response() -> None:
    executor, _, responses = action_executor_probe()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        vad=ScriptedVad(),
        stt=PreliminaryAndEndpointFinalTranscriber(),
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 7, frame)
    await worker.receive_audio("browser_user", 7, frame)
    await worker.wait_for_responses()

    assert responses == [
        (
            "session_7_turn_1",
            Transcript(text="authoritative endpoint final", final=True),
        )
    ]
    await worker.close()


@pytest.mark.asyncio
async def test_session_worker_projects_partial_without_admitting_it_as_a_turn() -> None:
    executor, _, responses = action_executor_probe()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    router = CommandRouterProbe(handled=False)
    transcriber = PartialThenFinalTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=3,
        vad=ScriptedVad(),
        stt=transcriber,
        actions=executor,
        command_router=router,
        observer=lambda kind, data: observed.append((kind, data)),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 3, frame)

    assert observed[-1] == (
        "transcript_partial",
        {"role": "user", "text": "Can you give"},
    )
    assert router.texts == []
    assert responses == []

    await worker.receive_audio("browser_user", 3, frame)
    await worker.wait_for_responses()

    assert router.texts == ["Can you give me a synopsis?"]
    assert responses == [
        (
            "session_3_turn_1",
            Transcript(text="Can you give me a synopsis?", final=True),
        )
    ]
    assert observed[-1] == (
        "transcript_final",
        {"role": "user", "text": "Can you give me a synopsis?"},
    )
    await worker.close()


@pytest.mark.asyncio
async def test_no_speech_result_does_not_terminalize_audio_or_typed_fallback() -> None:
    executor, _, responses = action_executor_probe()
    transcriber = NoSpeechTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 1, frame)
    await worker.receive_audio("browser_user", 1, frame)
    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=1,
        text="Typed fallback",
    )
    await worker.wait_for_responses()

    assert responses == [
        (
            "session_1_turn_1",
            Transcript(text="Typed fallback", final=True),
        )
    ]
    await worker.close()
    assert transcriber.cancel_calls == 1


@pytest.mark.asyncio
async def test_configured_pre_roll_preserves_every_vad_start_hysteresis_frame() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SILENCE,
            VoiceActivity.SILENCE,
            VoiceActivity.SILENCE,
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        pre_roll_frames=3,
    )
    worker.start()
    frames = tuple(
        AudioFrame(
            pcm=index.to_bytes(2, "little") * 160,
            sample_rate_hz=16_000,
            channels=1,
        )
        for index in range(1, 6)
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert transcriber.frames == list(frames)
    await worker.close()


@pytest.mark.asyncio
async def test_pre_roll_byte_budget_keeps_only_the_newest_complete_frames() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SILENCE,
            VoiceActivity.SILENCE,
            VoiceActivity.SILENCE,
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        pre_roll_frames=999,
        max_buffered_audio_bytes=640,
    )
    worker.start()
    frames = tuple(
        AudioFrame(
            pcm=index.to_bytes(2, "little") * 160,
            sample_rate_hz=16_000,
            channels=1,
        )
        for index in range(1, 6)
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert transcriber.frames == [frames[2], frames[3], frames[4]]
    await worker.close()


@pytest.mark.asyncio
async def test_playback_echo_never_cancels_foreground_or_reaches_stt() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    observed: list[str] = []
    guard = ScriptedEchoGuard(True, True, True, True)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=1,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(4):
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert transcriber.frames == []
    assert observed == ["echo_suppressed"]
    assert len(guard.windows) == 4
    await worker.close()


@pytest.mark.asyncio
async def test_single_non_echo_onset_during_playback_waits_for_confirmation() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    observed: list[str] = []
    guard = ScriptedEchoGuard(False, True, True)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=1,
        echo_promotion_checks=2,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(3):
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert transcriber.frames == []
    assert observed == ["echo_suppressed"]
    assert guard.windows == [(frame,), (frame, frame), (frame, frame, frame)]
    await worker.close()


@pytest.mark.asyncio
async def test_headset_cough_without_transcript_does_not_cancel_playback() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingTranscriber()
    observed: list[str] = []
    guard = ScriptedEchoGuard(False, False)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=1,
        echo_promotion_checks=2,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(3):
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert responses == []
    assert "interrupt_requested" not in observed
    assert "playback_silenced" not in observed
    await worker.close()


@pytest.mark.asyncio
async def test_partial_interruption_without_final_transcript_resumes_foreground() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingPartialNoFinalTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        echo_recheck_frames=1,
        echo_promotion_checks=2,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(3):
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is True
    assert cast(SimpleNamespace, executor._speech).resume_count == 1
    assert executor.foreground_active is True
    assert responses == []
    await worker.close()


@pytest.mark.parametrize("fragment", ["No.", "#."])
@pytest.mark.asyncio
async def test_short_partial_echo_fragment_does_not_interrupt_foreground(
    fragment: str,
) -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingShortPartialNoFinalTranscriber(fragment)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        echo_recheck_frames=1,
        echo_promotion_checks=2,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(3):
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert cast(SimpleNamespace, executor._speech).resume_count == 0
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_push_final_prevents_endpoint_from_resuming_interrupted_foreground() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingPushFinalNoEndpointTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 1, frame)
    await worker.receive_audio("browser_user", 1, frame)
    await asyncio.sleep(0)

    assert foreground_cancelled.is_set() is True
    assert len(responses) == 1
    assert responses[0][1].text == "Actual interruption."
    assert cast(SimpleNamespace, executor._speech).resume_count == 0
    await worker.close()


@pytest.mark.asyncio
async def test_outbound_text_echo_cannot_interrupt_or_create_a_user_turn() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingPushFinalNoEndpointTranscriber()
    guard = ScriptedEchoGuard(
        False,
        False,
        transcript_echoes=("Actual interruption.",),
    )
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 1, frame)
    await worker.receive_audio("browser_user", 1, frame)
    await asyncio.sleep(0)

    assert foreground_cancelled.is_set() is False
    assert responses == []
    assert guard.transcripts == ["Actual interruption."]
    assert "transcript_echo_suppressed" in observed
    await worker.close()


@pytest.mark.asyncio
async def test_retained_echo_tail_is_suppressed_after_foreground_becomes_inactive() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    executor._speech.foreground_active = False
    transcriber = CapturingTranscriber()
    guard = ScriptedEchoGuard(True, True, True, True)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=1,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x02\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(4):
        await worker.receive_audio("browser_user", 1, frame)

    assert guard.windows == [
        (frame,),
        (frame, frame),
        (frame, frame, frame),
        (frame, frame, frame, frame),
    ]
    assert foreground_cancelled.is_set() is False
    assert transcriber.frames == []
    await worker.close()


@pytest.mark.asyncio
async def test_short_echo_endpoint_final_without_streaming_evidence_is_not_admitted() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    transcriber = CapturingFinalTranscriber()
    guard = ScriptedEchoGuard(True, False)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=8,
        echo_promotion_checks=2,
        pre_roll_frames=3,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert transcriber.frames == list(frames)
    assert guard.windows == [(frames[0],), frames]
    await worker.close()


@pytest.mark.asyncio
async def test_recent_playback_tail_requires_speech_verification_after_foreground_settles() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    executor._speech.foreground_active = False  # type: ignore[attr-defined]
    transcriber = CapturingTextFinalTranscriber("Chapter 3.")
    verifier = ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_NON_SPEECH)
    guard = ScriptedEchoGuard(False, False, recent_playback=True)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        speech_presence_verifier=verifier,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert guard.windows == [(frames[0],), frames]
    assert verifier.windows == [frames]
    assert transcriber.frames == []
    assert foreground_cancelled.is_set() is False
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_endpointed_non_speech_candidate_cannot_reach_stt_or_cancel_foreground() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingTextFinalTranscriber("Oh. I'll try it.")
    verifier = ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_NON_SPEECH)
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x7f\x7f" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x20\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert verifier.windows == [frames]
    assert transcriber.frames == []
    assert foreground_cancelled.is_set() is False
    assert executor.foreground_active is True
    assert responses == []
    assert observed == ["echo_suppressed", "barge_in_non_speech_suppressed"]
    await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "presence",
    (
        SpeechPresence.CONFIRMED_SPEECH,
        SpeechPresence.UNCERTAIN,
    ),
)
async def test_playback_endpoint_final_without_streaming_evidence_is_not_authoritative(
    presence: SpeechPresence,
) -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingTextFinalTranscriber("Chapter 3.")
    verifier = ScriptedSpeechPresenceVerifier(presence)
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()

    assert verifier.windows == [frames]
    assert foreground_cancelled.is_set() is False
    assert executor.foreground_active is True
    assert responses == []
    assert "transcript_final" not in observed
    assert "interrupt_requested" not in observed
    assert "playback_silenced" not in observed
    await worker.close()


@pytest.mark.asyncio
async def test_playback_push_final_without_streaming_evidence_is_not_authoritative() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingPushFinalNoEndpointTranscriber()
    verifier = ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()

    assert verifier.windows == [frames]
    assert foreground_cancelled.is_set() is False
    assert executor.foreground_active is True
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_confirmed_speech_with_streaming_evidence_preserves_barge_in() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingPartialThenFinalTranscriber()
    verifier = ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False, False),
        speech_presence_verifier=verifier,
        echo_recheck_frames=1,
        echo_promotion_checks=2,
    )
    worker.start()
    frames = tuple(
        AudioFrame(pcm=bytes([index, 0]) * 480, sample_rate_hz=48_000, channels=1)
        for index in (1, 2, 3)
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()

    assert verifier.windows == [frames[:2]]
    assert foreground_cancelled.is_set() is True
    assert [transcript.text for _, transcript in responses] == ["Stop now please."]
    await worker.close()


@pytest.mark.asyncio
async def test_candidate_bound_to_replaced_turn_cannot_cancel_or_route_against_replacement(
) -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingPartialThenFinalTranscriber()
    verifier = BlockingSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    await worker.receive_audio("browser_user", 1, frames[0])
    endpoint = asyncio.create_task(worker.receive_audio("browser_user", 1, frames[1]))
    assert await asyncio.to_thread(verifier.entered.wait, 1.0)
    executor._speech.active_turn_id = "foreground_2"  # type: ignore[attr-defined]
    executor._speech.foreground_active = True  # type: ignore[attr-defined]
    verifier.release.set()
    await endpoint
    await worker.wait_for_responses()

    assert foreground_cancelled.is_set() is False
    assert executor.foreground_turn_id == "foreground_2"
    assert executor.foreground_active is True
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_non_speech_candidate_cannot_promote_mid_utterance_or_at_endpoint() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingTextFinalTranscriber("invented words")
    verifier = ScriptedSpeechPresenceVerifier(
        SpeechPresence.CONFIRMED_NON_SPEECH,
        SpeechPresence.CONFIRMED_NON_SPEECH,
    )
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False, False),
        speech_presence_verifier=verifier,
        echo_recheck_frames=1,
        echo_promotion_checks=2,
    )
    worker.start()
    frames = tuple(
        AudioFrame(pcm=bytes([index, 0]) * 480, sample_rate_hz=48_000, channels=1)
        for index in (1, 2, 3)
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert verifier.windows == [frames[:2], frames]
    assert transcriber.frames == []
    assert foreground_cancelled.is_set() is False
    assert executor.foreground_active is True
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    (
        SpeechPresence.CONFIRMED_SPEECH,
        SpeechPresence.CONFIRMED_NON_SPEECH,
        RuntimeError("verifier failed after close"),
    ),
)
async def test_binding_close_discards_speech_verification_completed_by_stale_generation(
    decision: SpeechPresence | BaseException,
) -> None:
    executor, _, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    verifier = BlockingSpeechPresenceVerifier(decision)
    voice_activity: list[bool] = []
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
        echo_recheck_frames=1,
        echo_promotion_checks=2,
        observer=lambda kind, _data: observed.append(kind),
        on_voice_activity=voice_activity.append,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 480, sample_rate_hz=48_000, channels=1)
    await worker.receive_audio("browser_user", 1, frame)
    receive = asyncio.create_task(worker.receive_audio("browser_user", 1, frame))
    entered = await asyncio.wait_for(asyncio.to_thread(verifier.entered.wait), timeout=1)
    assert entered is True

    try:
        await worker.close_binding()
    finally:
        verifier.release.set()

    with pytest.raises(RuntimeError, match="closed"):
        await receive
    assert transcriber.frames == []
    assert voice_activity == []
    assert observed == ["echo_suppressed"]
    await worker.close()


@pytest.mark.asyncio
async def test_endpointed_candidate_verifier_failure_retains_foreground() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingTextFinalTranscriber("invented words")
    verifier = ScriptedSpeechPresenceVerifier(RuntimeError("verifier failed"))
    observed: list[str] = []
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x10\x00" * 480, sample_rate_hz=48_000, channels=1)

    await worker.receive_audio("browser_user", 1, frame)
    await worker.receive_audio("browser_user", 1, frame)

    assert transcriber.frames == []
    assert foreground_cancelled.is_set() is False
    assert executor.foreground_active is True
    assert responses == []
    assert observed == ["echo_suppressed", "barge_in_verifier_unavailable"]
    await worker.close()


@pytest.mark.asyncio
async def test_short_echo_only_candidate_remains_suppressed_at_endpoint() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    transcriber = CapturingTranscriber()
    guard = ScriptedEchoGuard(True, True)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=8,
        echo_promotion_checks=2,
        pre_roll_frames=3,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert transcriber.frames == []
    assert guard.windows == [(frames[0],), frames]
    await worker.close()


@pytest.mark.asyncio
async def test_zero_pre_roll_endpoint_final_without_streaming_evidence_is_not_admitted() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    transcriber = CapturingFinalTranscriber()
    voice_activity: list[bool] = []
    guard = ScriptedEchoGuard(True, False)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=8,
        echo_promotion_checks=2,
        pre_roll_frames=0,
        on_voice_activity=voice_activity.append,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is False
    assert transcriber.frames == list(frames)
    assert guard.windows == [(frames[0],), frames]
    assert voice_activity == [True, False]
    await worker.close()


@pytest.mark.asyncio
async def test_uncorrelated_double_talk_promotes_to_real_barge_in() -> None:
    executor, foreground_cancelled, _ = action_executor_probe()
    transcriber = CapturingPartialThenFinalTranscriber()
    observed: list[str] = []
    guard = ScriptedEchoGuard(True, False, False)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_ENDED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=guard,
        echo_recheck_frames=1,
        echo_promotion_checks=2,
        pre_roll_frames=3,
        observer=lambda kind, _data: observed.append(kind),
    )
    worker.start()
    frames = tuple(
        AudioFrame(
            pcm=index.to_bytes(2, "little") * 480,
            sample_rate_hz=48_000,
            channels=1,
        )
        for index in range(1, 5)
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)

    assert foreground_cancelled.is_set() is True
    assert transcriber.frames == list(frames)
    assert observed == [
        "echo_suppressed",
        "interrupt_requested",
        "playback_silenced",
        "echo_barge_in_confirmed",
        "transcript_partial",
        "speech_ended",
        "transcript_final",
    ]
    await worker.close()


@pytest.mark.asyncio
async def test_idle_silence_never_reaches_stt_or_exhausts_utterance_capacity() -> None:
    executor, _, responses = action_executor_probe()
    transcriber = FinalOnPushTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x00\x00" * 480, sample_rate_hz=48_000, channels=1)

    for _ in range(20_000):
        await worker.receive_audio("browser_user", 1, frame)

    assert transcriber.push_calls == 0
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_reconnect_rejects_stale_binding_without_replacing_action_owner() -> None:
    executor, _, responses = action_executor_probe()
    transcribers: list[OrderedTranscriber] = []

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        cancelled = asyncio.Event()
        cancelled.set()
        transcriber = OrderedTranscriber(cancelled)
        transcribers.append(transcriber)
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=ScriptedVad(),
            stt=transcriber,
            actions=actions,
        )

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    first_generation = await worker.bind("browser_user")
    second_generation = await worker.bind("browser_user")
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    with pytest.raises(RuntimeError, match="stale"):
        await worker.receive_audio(
            "browser_user",
            first_generation,
            frame,
        )
    await worker.receive_audio("browser_user", second_generation, frame)

    assert transcribers[0].cancel_calls == 1
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_media_activation_admits_only_the_exact_livekit_publication() -> None:
    executor, _, _ = action_executor_probe()
    admitted = asyncio.Event()
    transcriber = CapturingTranscriber()

    async def push(frame: AudioFrame) -> tuple[Transcript, ...]:
        transcriber.frames.append(frame)
        admitted.set()
        return ()

    transcriber.push = push  # type: ignore[method-assign]

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=ScriptedVad(VoiceActivity.SPEECH_STARTED),
            stt=transcriber,
            actions=actions,
        )

    class PacketSource:
        def __init__(self) -> None:
            self.packets: asyncio.Queue[ParticipantAudioFrame] = asyncio.Queue()
            self.consumed: asyncio.Queue[None] = asyncio.Queue()

        async def receive_participant_audio(
            self,
            *,
            timeout_seconds: float,
        ) -> ParticipantAudioFrame:
            del timeout_seconds
            packet = await self.packets.get()
            self.consumed.put_nowait(None)
            return packet

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
        require_media_activation=True,
    )
    generation = await worker.bind("browser_user")
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)
    source = PacketSource()
    pump = asyncio.create_task(worker.run_source(source, generation))

    source.packets.put_nowait(
        ParticipantAudioFrame("browser_user", frame, track_name="microphone-1")
    )
    await source.consumed.get()
    assert transcriber.frames == []

    await worker.activate_media(
        participant_identity="browser_user",
        session_generation=generation,
        media_incarnation=2,
    )
    await worker.activate_media(
        participant_identity="browser_user",
        session_generation=generation,
        media_incarnation=2,
    )
    with pytest.raises(ValueError, match="monotonically"):
        await worker.activate_media(
            participant_identity="browser_user",
            session_generation=generation,
            media_incarnation=1,
        )

    source.packets.put_nowait(
        ParticipantAudioFrame("browser_user", frame, track_name="microphone-1")
    )
    await source.consumed.get()
    assert transcriber.frames == []
    source.packets.put_nowait(
        ParticipantAudioFrame("browser_user", frame, track_name="microphone-2")
    )
    await asyncio.wait_for(admitted.wait(), timeout=1)

    assert transcriber.frames == [frame]
    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump
    await worker.close()


@pytest.mark.asyncio
async def test_source_consumer_exit_after_rebind_is_not_reported_as_failure() -> None:
    executor, _, _ = action_executor_probe()

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=ScriptedVad(),
            stt=CapturingTranscriber(),
            actions=actions,
        )

    class BlockingSource:
        async def receive_participant_audio(
            self,
            *,
            timeout_seconds: float,
        ) -> ParticipantAudioFrame:
            del timeout_seconds
            await asyncio.Future()
            raise AssertionError("unreachable")

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    consumer_started = asyncio.Event()
    release_consumer = asyncio.Event()

    async def exiting_consumer(
        binding: ConversationSessionWorker,
        ingress: BoundedPcmIngress,
    ) -> None:
        del binding, ingress
        consumer_started.set()
        await release_consumer.wait()
        raise RuntimeError("conversation session worker is closed")

    worker._consume_source_ingress = exiting_consumer
    generation = await worker.bind("browser_user")
    pump = asyncio.create_task(worker.run_source(BlockingSource(), generation))
    await consumer_started.wait()

    await worker.bind("browser_user")
    release_consumer.set()

    await asyncio.wait_for(pump, timeout=1)
    await worker.close()


@pytest.mark.asyncio
async def test_source_keeps_pulling_while_first_admitted_frame_blocks_in_stt() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = BlockingTranscriber()

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=ScriptedVad(
                VoiceActivity.SPEECH_STARTED,
                VoiceActivity.SPEECH_CONTINUED,
            ),
            stt=transcriber,
            actions=actions,
        )

    class PacketSource:
        def __init__(self) -> None:
            self.packets: asyncio.Queue[ParticipantAudioFrame] = asyncio.Queue()
            self.consumed: asyncio.Queue[None] = asyncio.Queue()

        async def receive_participant_audio(
            self,
            *,
            timeout_seconds: float,
        ) -> ParticipantAudioFrame:
            del timeout_seconds
            packet = await self.packets.get()
            self.consumed.put_nowait(None)
            return packet

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    generation = await worker.bind("browser_user")
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)
    packet = ParticipantAudioFrame("browser_user", frame, track_name="microphone-1")
    source = PacketSource()
    pump = asyncio.create_task(worker.run_source(source, generation))
    try:
        source.packets.put_nowait(packet)
        await asyncio.wait_for(source.consumed.get(), timeout=1)
        await asyncio.wait_for(transcriber.started.wait(), timeout=1)

        source.packets.put_nowait(packet)
        await asyncio.wait_for(source.consumed.get(), timeout=0.1)
    finally:
        transcriber.release.set()
        pump.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pump
        await worker.close()


@pytest.mark.asyncio
async def test_media_activation_cannot_relabel_an_admitted_old_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _, _ = action_executor_probe()
    entered_delivery = asyncio.Event()
    release_delivery = asyncio.Event()
    second_admitted = asyncio.Event()
    transcriber = CapturingTranscriber()
    readiness: list[int | None] = []
    active_incarnation: int | None = None

    original_admit = BoundedPcmIngress.admit

    def record_admission(self: BoundedPcmIngress, record: IngressRecord) -> None:
        original_admit(self, record)
        if record.sequence == 2:
            second_admitted.set()

    monkeypatch.setattr(BoundedPcmIngress, "admit", record_admission)

    async def push(frame: AudioFrame) -> tuple[Transcript, ...]:
        transcriber.frames.append(frame)
        entered_delivery.set()
        await release_delivery.wait()
        return ()

    transcriber.push = push  # type: ignore[method-assign]

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=ScriptedVad(
                VoiceActivity.SPEECH_STARTED,
                VoiceActivity.SPEECH_CONTINUED,
            ),
            stt=transcriber,
            actions=actions,
            on_audio_ready=lambda: readiness.append(active_incarnation),
        )

    class PacketSource:
        def __init__(self) -> None:
            self.packets: asyncio.Queue[ParticipantAudioFrame] = asyncio.Queue()
            self.consumed: asyncio.Queue[None] = asyncio.Queue()

        async def receive_participant_audio(
            self,
            *,
            timeout_seconds: float,
        ) -> ParticipantAudioFrame:
            del timeout_seconds
            packet = await self.packets.get()
            self.consumed.put_nowait(None)
            return packet

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
        require_media_activation=True,
    )
    generation = await worker.bind("browser_user")
    await worker.activate_media(
        participant_identity="browser_user",
        session_generation=generation,
        media_incarnation=1,
    )
    active_incarnation = 1
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)
    source = PacketSource()
    pump = asyncio.create_task(worker.run_source(source, generation))
    source.packets.put_nowait(
        ParticipantAudioFrame("browser_user", frame, track_name="microphone-1")
    )
    await entered_delivery.wait()
    await source.consumed.get()
    source.packets.put_nowait(
        ParticipantAudioFrame("browser_user", frame, track_name="microphone-1")
    )
    await source.consumed.get()
    await second_admitted.wait()

    async def activate_replacement() -> None:
        nonlocal active_incarnation
        await worker.activate_media(
            participant_identity="browser_user",
            session_generation=generation,
            media_incarnation=2,
        )
        active_incarnation = 2

    replacement = asyncio.create_task(activate_replacement())
    await asyncio.wait_for(replacement, timeout=0.1)
    assert active_incarnation == 2
    assert readiness == [1]

    release_delivery.set()
    await asyncio.sleep(0.05)

    assert readiness == [1]
    assert transcriber.frames == [frame]
    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump
    await worker.close()


@pytest.mark.asyncio
async def test_readiness_cue_input_suppression_discards_pcm_until_exact_token_resumes() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = CapturingTranscriber()

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=ScriptedVad(VoiceActivity.SPEECH_STARTED),
            stt=transcriber,
            actions=actions,
        )

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    generation = await worker.bind("browser_user")
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    token = await worker.suppress_audio_input(
        participant_identity="browser_user",
        session_generation=generation,
    )
    await worker.receive_audio("browser_user", generation, frame)
    assert transcriber.frames == []
    assert await worker.resume_audio_input(object()) is False
    await worker.receive_audio("browser_user", generation, frame)
    assert transcriber.frames == []

    assert await worker.resume_audio_input(token) is True
    await worker.receive_audio("browser_user", generation, frame)
    assert transcriber.frames == [frame]
    await worker.close()


@pytest.mark.asyncio
async def test_reconnect_worker_can_close_only_media_binding_before_transport_unbind() -> None:
    executor, _, _ = action_executor_probe()
    transcribers: list[OrderedTranscriber] = []

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        cancelled = asyncio.Event()
        cancelled.set()
        transcriber = OrderedTranscriber(cancelled)
        transcribers.append(transcriber)
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=SilenceVad(),
            stt=transcriber,
            actions=actions,
        )

    worker = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    first_generation = await worker.bind("browser_user")

    await worker.close_binding()

    assert transcribers[0].cancel_calls == 1
    frame = AudioFrame(pcm=b"\x00\x00" * 160, sample_rate_hz=16_000, channels=1)
    with pytest.raises(RuntimeError, match="stale"):
        await worker.receive_audio("browser_user", first_generation, frame)
    assert await worker.bind("browser_user") == first_generation + 1
    await worker.close()


@pytest.mark.asyncio
async def test_binding_close_requests_binding_owned_foreground_reason_once() -> None:
    executor, _, _ = action_executor_probe()
    reasons: list[bool] = []

    async def record_cancel(
        self: ConversationUpdateExecutor,
        *,
        host_shutdown: bool = False,
        binding_closed: bool = False,
    ) -> None:
        del self, host_shutdown
        reasons.append(binding_closed)

    executor._cancel_foreground_for_cleanup = MethodType(  # type: ignore[method-assign]
        record_cancel,
        executor,
    )
    cancelled = asyncio.Event()
    cancelled.set()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=OrderedTranscriber(cancelled),
        actions=executor,
    )

    await worker.close_binding()
    await worker.close_binding()

    assert reasons == [True]


@pytest.mark.asyncio
async def test_binding_close_survives_caller_cancellation() -> None:
    executor, _, _ = action_executor_probe()
    cancellation_started = asyncio.Event()
    release_cancellation = asyncio.Event()

    async def blocked_cancel(
        self: ConversationUpdateExecutor,
        *,
        host_shutdown: bool = False,
        binding_closed: bool = False,
    ) -> None:
        del self, host_shutdown, binding_closed
        cancellation_started.set()
        await release_cancellation.wait()

    executor._cancel_foreground_for_cleanup = MethodType(  # type: ignore[method-assign]
        blocked_cancel,
        executor,
    )
    foreground_cancelled = asyncio.Event()
    foreground_cancelled.set()
    transcriber = OrderedTranscriber(foreground_cancelled)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    caller = asyncio.create_task(worker.close_binding())
    await cancellation_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release_cancellation.set()
    await worker.close_binding()

    assert transcriber.cancel_calls == 1


@pytest.mark.asyncio
async def test_cancelled_echo_promotion_resets_reusable_binding_state() -> None:
    executor, _, _ = action_executor_probe()
    cancellation_started = asyncio.Event()
    release_cancellation = asyncio.Event()

    async def blocked_cancel_if(
        self: ConversationUpdateExecutor,
        turn_id: str,
    ) -> bool:
        del self, turn_id
        cancellation_started.set()
        await release_cancellation.wait()
        return True

    executor.cancel_foreground_if = MethodType(  # type: ignore[method-assign]
        blocked_cancel_if,
        executor,
    )
    transcriber = CapturingPartialThenFinalTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(
            VoiceActivity.SPEECH_STARTED,
            VoiceActivity.SPEECH_CONTINUED,
            VoiceActivity.SPEECH_STARTED,
        ),
        stt=transcriber,
        actions=executor,
        echo_guard=ScriptedEchoGuard(True, False, False),
        echo_recheck_frames=1,
        echo_promotion_checks=1,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 1, frame)
    promotion = asyncio.create_task(worker.receive_audio("browser_user", 1, frame))
    await cancellation_started.wait()
    promotion.cancel()
    with pytest.raises(asyncio.CancelledError):
        await promotion

    executor._speech.foreground_active = False  # type: ignore[attr-defined]
    release_cancellation.set()
    await worker.receive_audio("browser_user", 1, frame)

    assert transcriber.frames == [frame, frame]
    await worker.close()


@pytest.mark.asyncio
async def test_binding_close_cancels_stt_without_waiting_for_audio_lock() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = BlockingTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)
    receive = asyncio.create_task(worker.receive_audio("browser_user", 1, frame))
    await transcriber.started.wait()

    await worker.close_binding()

    with pytest.raises(RuntimeError, match="closed"):
        await receive
    assert transcriber.cancel_calls == 1


@pytest.mark.asyncio
async def test_cancelled_speech_start_cannot_admit_stale_transcript() -> None:
    executor, _, responses = action_executor_probe()
    cancellation_started = asyncio.Event()
    release_cancellation = asyncio.Event()

    async def blocked_cancel_if(
        self: ConversationUpdateExecutor,
        turn_id: str,
    ) -> bool:
        del self, turn_id
        cancellation_started.set()
        await release_cancellation.wait()
        return True

    executor.cancel_foreground_if = MethodType(  # type: ignore[method-assign]
        blocked_cancel_if,
        executor,
    )
    transcriber = FinalOnPushTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)
    receive = asyncio.create_task(worker.receive_audio("browser_user", 1, frame))
    await cancellation_started.wait()

    receive.cancel()
    with pytest.raises(asyncio.CancelledError):
        await receive

    assert transcriber.push_calls == 0
    assert responses == []
    release_cancellation.set()
    await worker.close_binding()


@pytest.mark.asyncio
async def test_failed_barge_in_cancellation_cannot_admit_transcript() -> None:
    executor, _, responses = action_executor_probe()

    async def failing_cancel_if(
        self: ConversationUpdateExecutor,
        turn_id: str,
    ) -> bool:
        del self, turn_id
        raise RuntimeError("foreground cancellation failed")

    executor.cancel_foreground_if = MethodType(  # type: ignore[method-assign]
        failing_cancel_if,
        executor,
    )
    transcriber = FinalOnPushTranscriber()
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED),
        stt=transcriber,
        actions=executor,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    with pytest.raises(RuntimeError, match="foreground cancellation failed"):
        await worker.receive_audio("browser_user", 1, frame)

    assert transcriber.push_calls == 0
    assert responses == []
    with pytest.raises(RuntimeError, match="foreground cancellation failed"):
        await worker.close_binding()


@pytest.mark.asyncio
async def test_binding_cleanup_timeout_is_bounded_and_retryable() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = FinalOnPushTranscriber()
    cancel_started = asyncio.Event()
    release_cancel = asyncio.Event()

    async def stubborn_cancel(self: FinalOnPushTranscriber) -> None:
        del self
        cancel_started.set()
        await release_cancel.wait()

    transcriber.cancel = MethodType(stubborn_cancel, transcriber)  # type: ignore[method-assign]
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=transcriber,
        actions=executor,
        cleanup_timeout_seconds=0.01,
    )
    worker.start()

    with pytest.raises(TimeoutError, match="binding cleanup exceeded"):
        await worker.close_binding()
    assert cancel_started.is_set()

    release_cancel.set()
    await worker.close_binding()


@pytest.mark.asyncio
async def test_binding_cleanup_reinvokes_transiently_failed_dependency() -> None:
    executor, _, _ = action_executor_probe()
    transcriber = FinalOnPushTranscriber()
    cancel_attempts = 0

    async def flaky_cancel(self: FinalOnPushTranscriber) -> None:
        nonlocal cancel_attempts
        del self
        cancel_attempts += 1
        if cancel_attempts == 1:
            raise RuntimeError("temporary STT cleanup failure")

    transcriber.cancel = MethodType(flaky_cancel, transcriber)  # type: ignore[method-assign]
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=transcriber,
        actions=executor,
    )
    worker.start()

    with pytest.raises(RuntimeError, match="temporary STT cleanup failure"):
        await worker.close_binding()
    await worker.close_binding()

    assert cancel_attempts == 2


@pytest.mark.asyncio
async def test_full_close_attempts_action_authority_after_binding_failure() -> None:
    executor, _, _ = action_executor_probe()
    action_close_calls = 0

    async def record_action_close(self: ConversationUpdateExecutor) -> None:
        nonlocal action_close_calls
        del self
        action_close_calls += 1

    executor.close = MethodType(record_action_close, executor)  # type: ignore[method-assign]
    transcriber = FinalOnPushTranscriber()

    async def failed_stt_close(self: FinalOnPushTranscriber) -> None:
        del self
        raise RuntimeError("STT cleanup failed")

    transcriber.cancel = MethodType(failed_stt_close, transcriber)  # type: ignore[method-assign]
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=transcriber,
        actions=executor,
    )
    worker.start()

    with pytest.raises(RuntimeError, match="STT cleanup failed"):
        await worker.close()

    assert action_close_calls == 1


@pytest.mark.asyncio
async def test_supervisor_retries_binding_cleanup_after_action_authority_close() -> None:
    executor, _, _ = action_executor_probe()
    cleanup_attempts = 0
    action_close_calls = 0

    async def flaky_cleanup(
        self: ConversationUpdateExecutor,
        *,
        host_shutdown: bool = False,
        binding_closed: bool = False,
    ) -> None:
        nonlocal cleanup_attempts
        del self, host_shutdown, binding_closed
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise RuntimeError("transient foreground cleanup failure")

    async def record_action_close(self: ConversationUpdateExecutor) -> None:
        nonlocal action_close_calls
        del self
        action_close_calls += 1

    executor._cancel_foreground_for_cleanup = MethodType(  # type: ignore[method-assign]
        flaky_cleanup,
        executor,
    )
    executor.close = MethodType(record_action_close, executor)  # type: ignore[method-assign]

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=SilenceVad(),
            stt=FinalOnPushTranscriber(),
            actions=actions,
        )

    supervisor = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    await supervisor.bind("browser_user")

    with pytest.raises(RuntimeError, match="transient foreground cleanup failure"):
        await supervisor.close()
    await supervisor.close()

    assert cleanup_attempts == 2
    assert action_close_calls == 2
    assert supervisor.active_generation is None


@pytest.mark.asyncio
async def test_response_reservation_precedes_mutation_and_is_handed_to_executor() -> None:
    executor, _, _ = action_executor_probe()
    scheduler = ConversationOperationScheduler(owner_generation=77, max_operations=1)
    object.__setattr__(executor, "_operation_scheduler", scheduler)
    observed: list[tuple[str, int]] = []
    reservations: list[Any] = []

    async def respond(
        self: ConversationUpdateExecutor,
        turn_id: str,
        transcript: Transcript,
        authority: UserTurnAuthorityV1 | None = None,
        *,
        reservation: Any | None = None,
    ) -> None:
        del turn_id, transcript, authority
        assert reservation is not None
        observed.append(("respond", scheduler.active_count))
        reservations.append(reservation)
        self._operation_scheduler.release(reservation)

    executor.respond = MethodType(respond, executor)  # type: ignore[method-assign]
    owner = EvidenceLifecycleOwner(owner_generation=77)
    owner.activate_binding(
        binding_id=str(UUID(int=772, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=773, version=4)),
        logical_session_id=str(UUID(int=774, version=4)),
    )
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=EvidenceCommandRouterProbe(owner),
        evidence_lifecycle=owner.conversation_authority,
    )
    worker.start()

    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=1,
        text="reserve before any mutation",
    )
    await worker.wait_for_responses()

    assert observed == [("respond", 1)]
    assert len(reservations) == 1
    assert scheduler.active_count == 0
    await worker.close()


@pytest.mark.asyncio
async def test_response_saturation_rejects_before_turn_allocation_or_knowledge_mutation() -> None:
    executor, _, responses = action_executor_probe()
    scheduler = ConversationOperationScheduler(owner_generation=78, max_operations=1)
    blocker = scheduler.try_reserve(ConversationOperationKind.REPLAY)
    assert blocker is not None
    object.__setattr__(executor, "_operation_scheduler", scheduler)
    knowledge = SimpleNamespace(admit_calls=0)

    def active_ticket() -> UtteranceTicket:
        return UtteranceTicket(
            session_generation=1,
            media_incarnation=1,
            utterance_sequence=1,
        )

    def admit_final(*args: object, **kwargs: object) -> None:
        del args, kwargs
        knowledge.admit_calls += 1

    knowledge.admit_final = admit_final
    owner = EvidenceLifecycleOwner(owner_generation=78)
    owner.activate_binding(
        binding_id=str(UUID(int=782, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=783, version=4)),
        logical_session_id=str(UUID(int=784, version=4)),
    )
    router = EvidenceCommandRouterProbe(owner)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=router,
        evidence_lifecycle=owner.conversation_authority,
    )
    object.__setattr__(worker, "_knowledge_coordinator", knowledge)
    worker._active_ticket = active_ticket  # type: ignore[method-assign]
    worker.start()

    with pytest.raises(RuntimeError, match="operation capacity exhausted"):
        await worker.submit_final_transcript(
            participant_identity="browser_user",
            session_generation=1,
            typed_sequence=1,
            text="must reject before mutation",
        )

    assert worker._turn_sequence == 0
    assert knowledge.admit_calls == 0
    assert responses == []
    scheduler.release(blocker)
    object.__setattr__(worker, "_knowledge_coordinator", None)
    object.__setattr__(worker, "_terminal_error", None)
    await worker.close()


@pytest.mark.asyncio
async def test_natural_typed_final_passes_exact_user_authority_to_response() -> None:
    executor, responses = evidence_action_executor_probe()
    identities = iter((str(UUID(int=791, version=4)),))
    owner = EvidenceLifecycleOwner(owner_generation=79, uuid_factory=identities.__next__)
    owner.activate_binding(
        binding_id=str(UUID(int=792, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=793, version=4)),
        logical_session_id=str(UUID(int=794, version=4)),
    )
    router = EvidenceCommandRouterProbe(owner)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=router,
        evidence_lifecycle=owner.conversation_authority,
    )
    worker.start()

    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=1,
        text="natural evidence request",
    )
    await worker.wait_for_responses()

    assert len(responses) == 1
    turn_id, transcript, response_authority = responses[0]
    assert turn_id == "session_1_turn_1"
    assert transcript.text == "natural evidence request"
    assert response_authority is router.user_authorities[0]
    await worker.close()


@pytest.mark.asyncio
async def test_consent_published_lifecycle_resolver_injects_authority_into_normal_response(
) -> None:
    executor, responses = evidence_action_executor_probe()
    identities = iter((str(UUID(int=795, version=4)),))
    owner = EvidenceLifecycleOwner(owner_generation=79, uuid_factory=identities.__next__)
    owner.activate_binding(
        binding_id=str(UUID(int=796, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=797, version=4)),
        logical_session_id=str(UUID(int=798, version=4)),
    )
    published: list[EvidenceLifecycleOwner | None] = [None]
    router = EvidenceCommandRouterProbe(owner)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=router,
        evidence_lifecycle_resolver=lambda: published[0],
    )
    worker.start()

    published[0] = owner.conversation_authority
    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=1,
        text="consented normal response",
    )
    await worker.wait_for_responses()

    assert len(responses) == 1
    assert responses[0][2] is router.user_authorities[0]
    await worker.close()


@pytest.mark.asyncio
async def test_typed_final_mints_authority_before_closed_command_routing() -> None:
    executor, _, responses = action_executor_probe()
    identities = iter((str(UUID(int=801, version=4)),))
    owner = EvidenceLifecycleOwner(owner_generation=80, uuid_factory=identities.__next__)
    owner.activate_binding(
        binding_id=str(UUID(int=802, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=803, version=4)),
        logical_session_id=str(UUID(int=804, version=4)),
    )
    router = EvidenceCommandRouterProbe(owner)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=router,
        evidence_lifecycle=owner.conversation_authority,
    )
    worker.start()

    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=4_096,
        text="typed evidence request",
    )
    await worker.wait_for_responses()

    assert len(router.calls) == 1
    text, authority = router.calls[0]
    assert text == "typed evidence request"
    assert authority.source is InputSource.TYPED
    assert authority.typed_sequence == 4_096
    assert authority.media_incarnation is None
    assert responses == [
        ("session_1_turn_1", Transcript(text="typed evidence request", final=True))
    ]
    await worker.close()


@pytest.mark.asyncio
async def test_microphone_final_mints_authority_with_active_media_incarnation() -> None:
    executor, _, responses = action_executor_probe()
    owner = EvidenceLifecycleOwner(
        owner_generation=81,
        uuid_factory=lambda: str(UUID(int=811, version=4)),
    )
    owner.activate_binding(
        binding_id=str(UUID(int=812, version=4)),
        binding_generation=1,
        consent_epoch_id=str(UUID(int=813, version=4)),
        logical_session_id=str(UUID(int=814, version=4)),
    )
    router = EvidenceCommandRouterProbe(owner)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        media_incarnation_provider=lambda: 7,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=PreliminaryAndEndpointFinalTranscriber(),
        actions=executor,
        command_router=router,
        evidence_lifecycle=owner.conversation_authority,
    )
    worker.start()
    frame = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate_hz=16_000, channels=1)

    await worker.receive_audio("browser_user", 1, frame)
    await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()

    assert len(router.calls) == 1
    _, authority = router.calls[0]
    assert authority.source is InputSource.MICROPHONE
    assert authority.media_incarnation == 7
    assert authority.typed_sequence is None
    assert responses[0][1].text == "authoritative endpoint final"
    await worker.close()


@pytest.mark.asyncio
async def test_typed_final_transcript_admits_unhandled_natural_foreground_turn() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    observed: list[tuple[str, dict[str, str | int | bool | None]]] = []
    router = CommandRouterProbe(handled=False)

    def observe(kind: str, data: dict[str, str | int | bool | None]) -> None:
        observed.append((kind, data))

    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=router,
        observer=observe,
    )
    worker.start()

    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=1,
        text="typed fallback request",
    )
    await worker.wait_for_responses()

    assert foreground_cancelled.is_set()
    assert router.texts == ["typed fallback request"]
    assert responses == [
        ("session_1_turn_1", Transcript(text="typed fallback request", final=True))
    ]
    assert observed == [("transcript_final", {"role": "user", "text": "typed fallback request"})]
    await worker.close()


@pytest.mark.asyncio
async def test_explicit_task_command_bypasses_foreground_inference() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    router = CommandRouterProbe(handled=True)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=SilenceVad(),
        stt=FinalOnPushTranscriber(),
        actions=executor,
        command_router=router,
    )
    worker.start()

    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=1,
        typed_sequence=1,
        text="task: inspect release evidence",
    )
    await worker.wait_for_responses()

    assert foreground_cancelled.is_set()
    assert router.texts == ["task: inspect release evidence"]
    assert responses == []
    await worker.close()


@pytest.mark.asyncio
async def test_supervisor_rejects_stale_typed_transcript_before_admission() -> None:
    executor, _, responses = action_executor_probe()

    def binding_factory(
        participant_identity: str,
        generation: int,
        actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=SilenceVad(),
            stt=FinalOnPushTranscriber(),
            actions=actions,
        )

    supervisor = ReconnectSafeConversationWorker(
        actions=executor,
        binding_factory=binding_factory,
    )
    generation = await supervisor.bind("browser_user")

    with pytest.raises(RuntimeError, match="stale session generation"):
        await supervisor.submit_final_transcript(
            participant_identity="browser_user",
            session_generation=generation + 1,
            typed_sequence=1,
            text="must not be admitted",
        )

    assert responses == []
    await supervisor.close()


@pytest.mark.asyncio
async def test_utterance_ticket_uses_authoritative_media_incarnation_provider() -> None:
    executor, _cancelled, _responses = action_executor_probe()
    active_incarnation = 2
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=7,
        media_incarnation_provider=lambda: active_incarnation,
        vad=SilenceVad(),
        stt=NoSpeechTranscriber(),
        actions=executor,
    )

    first = worker._active_ticket()
    worker._active_utterance_ticket = None
    active_incarnation = 3
    replacement = worker._active_ticket()

    assert first.media_incarnation == 2
    assert replacement.media_incarnation == 3
    assert replacement.utterance_sequence == first.utterance_sequence + 1
    await worker.close()
@pytest.mark.asyncio
async def test_confirmed_playback_speech_admits_final_only_transcriber_final() -> None:
    executor, foreground_cancelled, responses = action_executor_probe()
    transcriber = CapturingTextFinalTranscriber("Chapter 3.")
    verifier = ScriptedSpeechPresenceVerifier(SpeechPresence.CONFIRMED_SPEECH)
    worker = ConversationSessionWorker(
        participant_identity="browser_user",
        session_generation=1,
        vad=ScriptedVad(VoiceActivity.SPEECH_STARTED, VoiceActivity.SPEECH_ENDED),
        stt=transcriber,
        stt_streams_partials=False,
        actions=executor,
        echo_guard=ScriptedEchoGuard(False, False),
        speech_presence_verifier=verifier,
    )
    worker.start()
    frames = (
        AudioFrame(pcm=b"\x03\x00" * 480, sample_rate_hz=48_000, channels=1),
        AudioFrame(pcm=b"\x04\x00" * 480, sample_rate_hz=48_000, channels=1),
    )

    for frame in frames:
        await worker.receive_audio("browser_user", 1, frame)
    await worker.wait_for_responses()

    assert verifier.windows == [frames]
    assert foreground_cancelled.is_set() is True
    assert len(responses) == 1
    assert responses[0][1] == Transcript(text="Chapter 3.", final=True)
    await worker.close()
