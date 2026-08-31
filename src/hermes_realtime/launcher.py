"""Runnable loopback browser composition for the explicit local provider profile."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from hermes_realtime._qualification import (
    _record_qualification_local_browser_launcher_construction,
)
from hermes_realtime.client import BrowserClientRuntime, BrowserEventProjection
from hermes_realtime.conversation import (
    DEFAULT_MAX_RESPONSE_SEGMENTS,
    ConversationContextStore,
    ConversationInferenceRequest,
    ConversationMessage,
    ConversationRole,
    ConversationSessionWorker,
    ConversationTaskController,
    ConversationUpdateDirector,
    ConversationUpdateExecutor,
    ForegroundTurnCoordinator,
    ReconnectSafeConversationWorker,
    StreamingSpeechLoop,
    UpdateDirective,
    UpdatePolicyInput,
)
from hermes_realtime.livekit import (
    LiveKitConnection,
    LiveKitConversationWorker,
    LiveKitPCMDeliveryConfirmation,
    LiveKitRoomPeer,
    LiveKitSpeechPlayback,
    ReconnectSafeLiveKitAudioPublisher,
)
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    ProductionObservationViewV1,
    _new_observation_channel,
    _ProductionObservationRecorderV1,
)
from hermes_realtime.protocol import (
    ControlCancelAcknowledgedEvent,
    ControlCancelEvent,
    WorkCompletedEvent,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchRequestedEvent,
)
from hermes_realtime.providers import (
    EdgeTtsSynthesizer,
    FasterWhisperTranscriber,
    OllamaStreamingInference,
    PlaybackEchoGuard,
    SileroSpeechPresenceVerifier,
    WebRtcVoiceActivityDetector,
)
from hermes_realtime.speech import DeliveredSpeechLedger

_ADVISORY_CONVERSATION_EVENTS = frozenset(
    {
        "echo_barge_in_confirmed",
        "echo_suppressed",
        "barge_in_non_speech_suppressed",
        "barge_in_verifier_unavailable",
        "transcript_echo_suppressed",
        "voice_activity_ended",
        "voice_activity_started",
    }
)


def _project_conversation_observation(
    projection: BrowserEventProjection,
    kind: str,
    data: dict[str, str | int | bool | None],
) -> None:
    if kind in _ADVISORY_CONVERSATION_EVENTS:
        projection.publish_advisory(kind, data)
        return
    projection.publish(kind, data)


class ConversationOnlyTaskSession:
    """Explicitly disable task authority while retaining cancellable controller lifecycle."""

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        del request
        raise PermissionError("task dispatch is disabled in the conversation-only profile")

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        del request
        raise PermissionError("task cancellation is disabled in the conversation-only profile")

    async def next_update(self) -> WorkCompletedEvent:
        await self._closed.wait()
        raise RuntimeError("conversation-only task session is closed")

    async def close(self) -> None:
        self._closed.set()


class ConversationOnlyUpdatePolicy:
    """Ignore impossible background completions in the no-task local profile."""

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        if type(policy_input) is not UpdatePolicyInput:
            raise TypeError("policy_input must be an exact UpdatePolicyInput")
        return UpdateDirective(kind="ignore")


class _BrowserRuntime(Protocol):
    async def start(self) -> str: ...

    async def close(self) -> None: ...


class _VerifierPeer(Protocol):
    async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None: ...

    async def disconnect(self, *, timeout_seconds: float = 10) -> None: ...


class _AsyncClosable(Protocol):
    async def close(self) -> None: ...


class LocalBrowserLauncher:
    """Own preflight, verifier, browser runtime, and provider cleanup as one lifecycle."""

    def __init__(
        self,
        *,
        runtime: _BrowserRuntime,
        verifier: _VerifierPeer,
        room_name: str,
        preflight: Callable[[], Awaitable[None]],
        providers: tuple[_AsyncClosable, ...],
        production_observations: ProductionObservationViewV1 | None = None,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
    ) -> None:
        for method in ("start", "close"):
            if not callable(getattr(runtime, method, None)):
                raise TypeError(f"runtime must provide {method}()")
        for method in ("connect", "disconnect"):
            if not callable(getattr(verifier, method, None)):
                raise TypeError(f"verifier must provide {method}()")
        if type(room_name) is not str:
            raise TypeError("room_name must be an exact built-in string")
        if not room_name.strip() or len(room_name) > 128:
            raise ValueError("room_name must contain 1 to 128 characters")
        if not callable(preflight):
            raise TypeError("preflight must be callable")
        if type(providers) is not tuple:
            raise TypeError("providers must be an exact tuple")
        if any(not callable(getattr(provider, "close", None)) for provider in providers):
            raise TypeError("every provider must provide close()")
        if (
            production_observations is not None
            and type(production_observations) is not ProductionObservationViewV1
        ):
            raise TypeError("production observations must be an exact view or None")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        if (production_observations is None) != (production_observation_recorder is None):
            raise ValueError("production observation view and recorder must be paired")
        if production_observations is None:
            production_observations, production_observation_recorder = _new_observation_channel()
        else:
            assert production_observation_recorder is not None
            if not production_observations._matches_recorder(production_observation_recorder):
                raise ValueError("production observation view and recorder must share one owner")
        assert production_observation_recorder is not None
        self._runtime = runtime
        self._verifier = verifier
        self._room_name = room_name
        self._preflight = preflight
        self._providers = providers
        self._start_attempted = False
        self._runtime_closed = False
        self._verifier_connected = False
        self._provider_closed = [False] * len(providers)
        self._started = False
        self._closed = False
        self._close_operation: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._production_observations = production_observations
        self._production_observation_recorder = production_observation_recorder
        _record_qualification_local_browser_launcher_construction(self)

    @property
    def production_observations(self) -> ProductionObservationViewV1:
        return self._production_observations

    async def start(self) -> str:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("local browser launcher is closed")
            if self._start_attempted:
                raise RuntimeError("local browser launcher is one-shot")
            self._start_attempted = True
            try:
                await self._preflight()
                await self._verifier.connect(self._room_name)
                self._verifier_connected = True
                url = await self._runtime.start()
                if type(url) is not str or not url.startswith(("http://", "https://")):
                    raise TypeError("runtime start must return an exact HTTP launch URL")
                self._started = True
                return url
            except BaseException as start_error:
                try:
                    await asyncio.shield(self._ensure_close_locked())
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "local launcher start and cleanup failed",
                        [start_error, cleanup_error],
                    ) from None
                raise

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            operation = self._ensure_close_locked()
        await asyncio.shield(operation)

    def _ensure_close_locked(self) -> asyncio.Task[None]:
        operation = self._close_operation
        if operation is None or (
            operation.done() and (operation.cancelled() or operation.exception() is not None)
        ):
            operation = asyncio.create_task(
                self._close_owned(),
                name="local-browser-launcher-close",
            )
            self._close_operation = operation
        return operation

    async def _close_owned(self) -> None:
        errors: list[BaseException] = []
        if self._start_attempted and not self._runtime_closed:
            try:
                await self._runtime.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._runtime_closed = True
        if self._verifier_connected:
            try:
                await self._verifier.disconnect()
            except BaseException as error:
                errors.append(error)
            else:
                self._verifier_connected = False
        for index, provider in enumerate(self._providers):
            if self._provider_closed[index]:
                continue
            try:
                await provider.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._provider_closed[index] = True
        self._started = False
        if len(errors) == 1:
            self._production_observation_recorder.record_close_stage(
                stage=CloseStageV1.LAUNCHER,
                result=CloseResultV1.FAILED,
            )
            raise errors[0]
        if errors:
            self._production_observation_recorder.record_close_stage(
                stage=CloseStageV1.LAUNCHER,
                result=CloseResultV1.FAILED,
            )
            raise BaseExceptionGroup("local browser launcher cleanup failed", errors)
        self._production_observation_recorder.record_close_stage(
            stage=CloseStageV1.LAUNCHER,
            result=CloseResultV1.SUCCEEDED,
        )
        self._closed = True

    @staticmethod
    def _raise_close_errors(errors: list[BaseException]) -> None:
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("local browser launcher cleanup failed", errors)


def build_local_conversation_launcher(
    *,
    livekit_url: str = "ws://127.0.0.1:7880",
    livekit_api_key: str = "devkey",
    livekit_api_secret: str = "local" + "-" + ("x" * 32),
    room_name: str = "hermes-realtime-local",
    worker_identity: str = "worker_local_realtime",
    browser_port: int = 8765,
    ollama_base_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "hermes-4.3-36b-iq4xs-16k:latest",
    whisper_model: str = "tiny.en",
    edge_voice: str = "en-US-AriaNeural",
) -> LocalBrowserLauncher:
    """Compose the explicit local conversation-only profile without hidden fallbacks."""

    if type(livekit_url) is not str:
        raise TypeError("livekit_url must be an exact built-in string")
    parsed_livekit = urlsplit(livekit_url)
    if (
        parsed_livekit.scheme != "ws"
        or parsed_livekit.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed_livekit.port is None
        or parsed_livekit.username is not None
        or parsed_livekit.password is not None
        or parsed_livekit.path not in {"", "/"}
        or parsed_livekit.query
        or parsed_livekit.fragment
    ):
        raise ValueError("local LiveKit URL must be an exact loopback WebSocket origin")
    connection = LiveKitConnection(
        url=livekit_url,
        api_key=livekit_api_key,
        api_secret=livekit_api_secret,
    )
    projection = BrowserEventProjection()
    production_observations, production_observation_recorder = _new_observation_channel()

    def observe(
        kind: str,
        data: dict[str, str | int | bool | None],
    ) -> None:
        _project_conversation_observation(projection, kind, data)

    context = ConversationContextStore()
    foreground = ForegroundTurnCoordinator(
        drain_timeout=10.0,
        output_capacity=DEFAULT_MAX_RESPONSE_SEGMENTS,
    )
    inference = OllamaStreamingInference(
        base_url=ollama_base_url,
        model=ollama_model,
        request_timeout_seconds=60.0,
    )
    synthesizer = EdgeTtsSynthesizer(voice=edge_voice)
    transcriber = FasterWhisperTranscriber(
        model_size_or_path=whisper_model,
        device="cpu",
        compute_type="int8",
    )
    publisher = ReconnectSafeLiveKitAudioPublisher()
    verifier = LiveKitRoomPeer(
        connection=connection,
        identity=f"verifier_{uuid.uuid4().hex}",
    )
    verifier.bind_remote_identity(worker_identity)
    confirmation = LiveKitPCMDeliveryConfirmation(
        verifier,
        timeout_seconds=30.0,
        max_frames=4096,
    )
    echo_guard = PlaybackEchoGuard()
    speech_presence_verifier = SileroSpeechPresenceVerifier()
    playback = LiveKitSpeechPlayback(
        publisher=publisher,
        confirmation=confirmation,
        publish_timeout_seconds=30.0,
        confirmation_timeout_seconds=30.0,
        echo_reference=echo_guard,
    )
    speech = StreamingSpeechLoop(
        context=context,
        foreground=foreground,
        inference=inference,
        synthesizer=synthesizer,
        playback=playback,
        ledger=DeliveredSpeechLedger(),
        observer=observe,
        max_segments=DEFAULT_MAX_RESPONSE_SEGMENTS,
        cleanup_timeout_seconds=10.0,
        production_observation_recorder=production_observation_recorder,
    )
    task_session = ConversationOnlyTaskSession()
    task_controller = ConversationTaskController(
        context=context,
        session=task_session,
        session_id="session_local_conversation",
        id_factory=lambda: uuid.uuid4().hex,
    )
    director = ConversationUpdateDirector(
        context=context,
        completions=task_controller,
        policy=ConversationOnlyUpdatePolicy(),
    )
    actions = ConversationUpdateExecutor(
        director=director,
        speech=speech,
        cleanup_timeout_seconds=10.0,
        production_observation_recorder=production_observation_recorder,
    )

    def binding_factory(
        participant_identity: str,
        generation: int,
        shared_actions: ConversationUpdateExecutor,
    ) -> ConversationSessionWorker:
        vad = WebRtcVoiceActivityDetector()
        return ConversationSessionWorker(
            participant_identity=participant_identity,
            session_generation=generation,
            vad=vad,
            stt=transcriber,
            actions=shared_actions,
            stt_streams_partials=False,
            production_observation_recorder=production_observation_recorder,
            observer=observe,
            cleanup_timeout_seconds=10.0,
            pre_roll_frames=vad.required_pre_roll_frames,
            echo_guard=echo_guard,
            speech_presence_verifier=speech_presence_verifier,
        )

    conversation = ReconnectSafeConversationWorker(
        actions=actions,
        binding_factory=binding_factory,
    )
    livekit_worker = LiveKitConversationWorker(
        runtime=conversation,
        publisher=publisher,
        production_observation_recorder=production_observation_recorder,
    )

    async def reject_approval(
        participant_identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        del participant_identity, generation, sequence, approval_id, decision
        raise PermissionError("approval is disabled in the conversation-only profile")

    runtime = BrowserClientRuntime(
        connection=connection,
        room_name=room_name,
        worker_identity=worker_identity,
        worker=livekit_worker,
        approval=reject_approval,
        static_root=Path(__file__).resolve().parent / "client" / "static",
        port=browser_port,
        projection=projection,
        production_observation_recorder=production_observation_recorder,
    )

    async def preflight() -> None:
        snapshot = ConversationInferenceRequest(
            revision=0,
            messages=(
                ConversationMessage(
                    role=ConversationRole.USER.value,
                    text="Reply with one short sentence confirming readiness.",
                ),
            ),
            active_tasks=(),
            updates=(),
        )
        segments = [
            segment async for segment in inference.stream(snapshot, turn_id="turn_preflight")
        ]
        if not segments:
            raise RuntimeError("local inference preflight produced no speakable output")

    return LocalBrowserLauncher(
        runtime=runtime,
        verifier=verifier,
        room_name=room_name,
        preflight=preflight,
        providers=(task_controller, synthesizer, inference),
        production_observations=production_observations,
        production_observation_recorder=production_observation_recorder,
    )


async def _run_local_cli(args: argparse.Namespace) -> None:
    launcher = build_local_conversation_launcher(
        room_name=args.room,
        worker_identity=args.worker_identity,
        browser_port=args.port,
        ollama_model=args.ollama_model,
        whisper_model=args.whisper_model,
        edge_voice=args.edge_voice,
    )
    try:
        launch_url = await launcher.start()
        print(
            "Conversation-only local profile: task dispatch and approvals are disabled.",
            flush=True,
        )
        print("Open this one-use URL in one local browser:", flush=True)
        print(launch_url, flush=True)
        await asyncio.Event().wait()
    finally:
        await launcher.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the explicit loopback Hermes Realtime conversation profile.",
    )
    parser.add_argument("--room", default="hermes-realtime-local")
    parser.add_argument("--worker-identity", default="worker_local_realtime")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--ollama-model",
        default="hermes-4.3-36b-iq4xs-16k:latest",
    )
    parser.add_argument("--whisper-model", default="tiny.en")
    parser.add_argument("--edge-voice", default="en-US-AriaNeural")
    args = parser.parse_args()
    try:
        asyncio.run(_run_local_cli(args))
    except KeyboardInterrupt:
        return 130
    return 0
