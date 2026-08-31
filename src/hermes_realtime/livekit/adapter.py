"""Official LiveKit SDK adapter for provider-neutral realtime audio."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol, cast

from livekit import api, rtc

from hermes_realtime.speech.types import (
    MAX_SPEECH_CHUNK_DURATION_SECONDS as MAX_SPEECH_CHUNK_DURATION_SECONDS,
)
from hermes_realtime.speech.types import (
    AudioFrame,
    ParticipantAudioFrame,
    SpeechChunk,
)


class _Publication(Protocol):
    sid: str


@dataclass(frozen=True, slots=True)
class LiveKitConnection:
    """Credentials and endpoint for a LiveKit server."""

    url: str
    api_key: str
    api_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("url", self.url),
            ("api_key", self.api_key),
            ("api_secret", self.api_secret),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be blank")

    def token(
        self,
        *,
        identity: str,
        room_name: str,
        ttl_seconds: int = 300,
    ) -> str:
        """Issue a room-scoped participant token."""
        if not identity.strip():
            raise ValueError("identity must not be blank")
        if not room_name.strip():
            raise ValueError("room_name must not be blank")
        if type(ttl_seconds) is not int:
            raise TypeError("ttl_seconds must be an exact integer")
        if ttl_seconds < 30 or ttl_seconds > 300:
            raise ValueError("ttl_seconds must be between 30 and 300")
        return cast(
            str,
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(identity)
            .with_ttl(timedelta(seconds=ttl_seconds))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_subscribe=True,
                    can_publish_data=False,
                    can_publish_sources=["microphone"],
                    can_update_own_metadata=False,
                )
            )
            .to_jwt(),
        )


class LiveKitRoomPeer:
    """One LiveKit room participant with raw PCM publish/receive boundaries."""

    _AUDIO_STREAM_CAPACITY = 16
    _SUBSCRIBED_AUDIO_CAPACITY = 8
    _DEFAULT_AUDIO_QUEUE_MS = 1_000
    _MAX_SPEECH_QUEUE_MS = int(MAX_SPEECH_CHUNK_DURATION_SECONDS * 1_000)

    def __init__(
        self,
        connection: LiveKitConnection,
        *,
        identity: str,
        sample_rate_hz: int = 48_000,
        channels: int = 1,
    ) -> None:
        if not identity.strip():
            raise ValueError("identity must not be blank")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._connection = connection
        self._identity = identity
        self._sample_rate_hz = sample_rate_hz
        self._channels = channels
        self._room = rtc.Room()
        self._subscribed_audio: asyncio.Queue[tuple[str, str, rtc.Track]] = (
            asyncio.Queue(maxsize=self._SUBSCRIBED_AUDIO_CAPACITY)
        )
        self._audio_source: rtc.AudioSource | None = None
        self._local_track: rtc.LocalAudioTrack | None = None
        self._publication_sid: str | None = None
        self._publication_task: asyncio.Task[_Publication] | None = None
        self._audio_stream: rtc.AudioStream | None = None
        self._audio_stream_name: str | None = None
        self._audio_stream_identity: str | None = None
        self._active_speech_track_name: str | None = None
        self._active_speech_chunk_key: tuple[str, str] | None = None
        self._speech_publication_ready = False
        self._publish_lock = asyncio.Lock()
        self._connected = False
        self._closed = False
        self._room_cleanup_required = False
        self._connection_attempted = False
        self._bound_remote_identity: str | None = None

        def on_track_subscribed(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            _participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            participant_identity = _participant.identity
            if type(participant_identity) is not str:
                return
            if (
                self._bound_remote_identity is not None
                and participant_identity != self._bound_remote_identity
            ):
                return
            if self._subscribed_audio.full():
                with suppress(asyncio.QueueEmpty):
                    self._subscribed_audio.get_nowait()
            self._subscribed_audio.put_nowait(
                (participant_identity, _publication.name, track)
            )

        def on_disconnected(_reason: rtc.DisconnectReason) -> None:
            self._connected = False
            self._room_cleanup_required = False
            self._closed = True
            self._clear_subscribed_audio()

        self._room.on("track_subscribed", on_track_subscribed)
        self._room.on("disconnected", on_disconnected)

    @property
    def remote_identities(self) -> frozenset[str]:
        """Current remote participant identities."""
        return frozenset(
            participant.identity for participant in self._room.remote_participants.values()
        )

    @property
    def connected(self) -> bool:
        """Whether the SDK room is currently considered connected."""
        return self._connected

    @staticmethod
    def _validate_timeout(timeout_seconds: float) -> None:
        if type(timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an exact number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    def bind_remote_identity(self, participant_identity: str) -> None:
        """Allow microphone tracks from one exact participant before connection."""

        if type(participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if not participant_identity.strip() or len(participant_identity) > 128:
            raise ValueError("participant_identity must contain 1 to 128 characters")
        if self._connection_attempted or self._connected or self._closed:
            raise RuntimeError("remote identity must be bound before connection")
        if (
            self._bound_remote_identity is not None
            and self._bound_remote_identity != participant_identity
        ):
            raise RuntimeError("peer is already bound to another remote identity")
        self._bound_remote_identity = participant_identity

    async def connect(self, room_name: str, *, timeout_seconds: float = 10) -> None:
        """Join one room with a narrowly scoped token."""
        self._validate_timeout(timeout_seconds)
        if self._closed:
            raise RuntimeError("peer is closed")
        if self._connected:
            raise RuntimeError("peer is already connected")
        self._connection_attempted = True
        token = self._connection.token(identity=self._identity, room_name=room_name)
        self._room_cleanup_required = True
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._room.connect(self._connection.url, token)
        except BaseException as connect_error:
            self._closed = True
            try:
                async with asyncio.timeout(timeout_seconds):
                    await self._room.disconnect()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "LiveKit connect and cleanup failed",
                    [connect_error, cleanup_error],
                ) from None
            self._room_cleanup_required = False
            raise
        self._connected = True

    async def publish_audio(self, frame: AudioFrame, *, timeout_seconds: float = 10) -> None:
        """Publish one provider-neutral PCM frame."""
        self._validate_audio_publish(frame, timeout_seconds)

        async with asyncio.timeout(timeout_seconds):
            async with self._publish_lock:
                await self._prepare_audio_locked("hermes-audio")
                if self._audio_source is None:
                    raise RuntimeError("audio source preparation failed")
                samples_per_channel = len(frame.pcm) // (2 * frame.channels)
                await self._audio_source.capture_frame(
                    rtc.AudioFrame(
                        data=frame.pcm,
                        sample_rate=frame.sample_rate_hz,
                        num_channels=frame.channels,
                        samples_per_channel=samples_per_channel,
                    )
                )

    async def prepare_audio(
        self,
        frame: AudioFrame,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        """Publish the audio track before the first latency-sensitive PCM frame."""

        self._validate_audio_publish(frame, timeout_seconds)
        async with asyncio.timeout(timeout_seconds):
            async with self._publish_lock:
                await self._prepare_audio_locked("hermes-audio")

    async def clear_audio_queue(self, *, timeout_seconds: float = 10) -> None:
        """Discard provider-buffered PCM so cancelled speech cannot resume."""

        if not self._connected:
            raise RuntimeError("peer is not connected")
        self._validate_timeout(timeout_seconds)
        async with asyncio.timeout(timeout_seconds):
            async with self._publish_lock:
                if self._audio_source is not None:
                    self._audio_source.clear_queue()

    async def prepare_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> str:
        if type(chunk) is not SpeechChunk:
            raise TypeError("chunk must be an exact SpeechChunk")
        self._validate_audio_publish(chunk.audio, timeout_seconds)
        self._validate_speech_chunk_duration(chunk.audio)
        chunk_key = (chunk.turn_id, chunk.chunk_id)
        async with asyncio.timeout(timeout_seconds):
            async with self._publish_lock:
                if self._active_speech_chunk_key is not None:
                    raise RuntimeError("another LiveKit speech chunk is active")
                if self._audio_source is not None and self._active_speech_track_name is None:
                    raise RuntimeError("generic LiveKit audio publication is active")
                track_name = self._active_speech_track_name
                if track_name is None:
                    track_name = f"hermes-speech-{uuid.uuid4().hex}"
                    self._active_speech_track_name = track_name
                self._active_speech_chunk_key = chunk_key
                try:
                    await self._prepare_audio_locked(
                        track_name,
                        queue_size_ms=self._MAX_SPEECH_QUEUE_MS,
                    )
                except BaseException:
                    if (
                        self._publication_task is None
                        and self._publication_sid is None
                        and self._audio_source is None
                    ):
                        self._active_speech_track_name = None
                        self._active_speech_chunk_key = None
                        self._speech_publication_ready = False
                    raise
                self._speech_publication_ready = True
                return track_name

    async def publish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        if type(chunk) is not SpeechChunk:
            raise TypeError("chunk must be an exact SpeechChunk")
        self._validate_audio_publish(chunk.audio, timeout_seconds)
        chunk_key = (chunk.turn_id, chunk.chunk_id)
        async with asyncio.timeout(timeout_seconds):
            async with self._publish_lock:
                if self._active_speech_chunk_key != chunk_key:
                    raise RuntimeError("LiveKit speech chunk is not prepared")
                source = self._audio_source
                if source is None:
                    raise RuntimeError("audio source preparation failed")
                samples_per_channel = len(chunk.audio.pcm) // (2 * chunk.audio.channels)
                await source.capture_frame(
                    rtc.AudioFrame(
                        data=chunk.audio.pcm,
                        sample_rate=chunk.audio.sample_rate_hz,
                        num_channels=chunk.audio.channels,
                        samples_per_channel=samples_per_channel,
                    )
                )

    async def finish_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        await self._release_speech_chunk(chunk, timeout_seconds, cancelled=False)

    async def cancel_speech_chunk(
        self,
        chunk: SpeechChunk,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        await self._release_speech_chunk(chunk, timeout_seconds, cancelled=True)

    async def _release_speech_chunk(
        self,
        chunk: SpeechChunk,
        timeout_seconds: float,
        *,
        cancelled: bool,
    ) -> None:
        if type(chunk) is not SpeechChunk:
            raise TypeError("chunk must be an exact SpeechChunk")
        self._validate_timeout(timeout_seconds)
        chunk_key = (chunk.turn_id, chunk.chunk_id)
        async with asyncio.timeout(timeout_seconds):
            async with self._publish_lock:
                if self._active_speech_chunk_key != chunk_key:
                    return
                source = self._audio_source
                if source is not None:
                    if cancelled:
                        source.clear_queue()
                    else:
                        await source.wait_for_playout()
                if not self._speech_publication_ready:
                    if self._publication_task is not None:
                        await self._settle_publication_for_publish()
                    if self._publication_sid is not None:
                        await self._room.local_participant.unpublish_track(
                            self._publication_sid
                        )
                        self._publication_sid = None
                    if source is not None:
                        await source.aclose()
                    self._audio_source = None
                    self._local_track = None
                    self._active_speech_track_name = None
                self._active_speech_chunk_key = None

    def _validate_audio_publish(self, frame: AudioFrame, timeout_seconds: float) -> None:
        if not self._connected:
            raise RuntimeError("peer is not connected")
        self._validate_timeout(timeout_seconds)
        if frame.sample_rate_hz != self._sample_rate_hz or frame.channels != self._channels:
            raise ValueError("audio frame format does not match the peer format")

    @classmethod
    def _validate_speech_chunk_duration(cls, frame: AudioFrame) -> None:
        samples_per_channel = len(frame.pcm) // (2 * frame.channels)
        duration_ms = math.ceil(samples_per_channel * 1_000 / frame.sample_rate_hz)
        if duration_ms > cls._MAX_SPEECH_QUEUE_MS:
            raise ValueError("speech chunk exceeds supported LiveKit queue duration")

    async def _prepare_audio_locked(
        self,
        track_name: str,
        *,
        queue_size_ms: int = _DEFAULT_AUDIO_QUEUE_MS,
    ) -> None:
        if not self._connected:
            raise RuntimeError("peer is not connected")
        if self._publication_task is not None:
            await self._settle_publication_for_publish()

        if self._audio_source is None:
            source = rtc.AudioSource(
                self._sample_rate_hz,
                self._channels,
                queue_size_ms=queue_size_ms,
            )
            track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            self._audio_source = source
            self._local_track = track
            self._publication_task = cast(
                asyncio.Task[_Publication],
                asyncio.create_task(
                    self._room.local_participant.publish_track(track, options)
                ),
            )
            await self._settle_publication_for_publish()

        if self._publication_sid is None:
            raise RuntimeError("audio publication cleanup is required")

    async def _settle_publication(self) -> None:
        task = self._publication_task
        if task is None:
            return
        try:
            publication = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise RuntimeError("LiveKit audio publication was cancelled") from None
            raise
        self._publication_sid = publication.sid
        self._publication_task = None

    async def _settle_publication_for_publish(self) -> None:
        try:
            await self._settle_publication()
        except asyncio.CancelledError:
            raise
        except BaseException as publication_error:
            self._publication_task = None
            source = self._audio_source
            if source is None:
                raise
            try:
                await source.aclose()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "LiveKit publication and source cleanup failed",
                    [publication_error, cleanup_error],
                ) from None
            self._audio_source = None
            self._local_track = None
            self._speech_publication_ready = False
            raise

    def _clear_subscribed_audio(self) -> None:
        while not self._subscribed_audio.empty():
            try:
                self._subscribed_audio.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def receive_audio(self, *, timeout_seconds: float) -> AudioFrame:
        """Receive one decoded PCM frame before one total deadline."""

        return (await self.receive_participant_audio(timeout_seconds=timeout_seconds)).frame

    async def receive_participant_audio(
        self,
        *,
        timeout_seconds: float,
    ) -> ParticipantAudioFrame:
        """Receive one decoded PCM frame with its remote participant identity."""

        if not self._connected:
            raise RuntimeError("peer is not connected")
        self._validate_timeout(timeout_seconds)

        async with asyncio.timeout(timeout_seconds):
            while True:
                await self._prepare_receive_audio_locked()
                stream = self._audio_stream
                participant_identity = self._audio_stream_identity
                track_name = self._audio_stream_name
                if stream is None or participant_identity is None or track_name is None:
                    raise RuntimeError("audio receive preparation failed")

                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    await stream.aclose()
                    self._audio_stream = None
                    self._audio_stream_name = None
                    self._audio_stream_identity = None
                    continue
                return ParticipantAudioFrame(
                    participant_identity=participant_identity,
                    track_name=track_name,
                    frame=AudioFrame(
                        pcm=bytes(event.frame.data.cast("B")),
                        sample_rate_hz=event.frame.sample_rate,
                        channels=event.frame.num_channels,
                    ),
                )

    async def prepare_receive_audio(self, *, timeout_seconds: float = 10) -> None:
        """Wait until the remote audio track is subscribed and ready to read."""

        if not self._connected:
            raise RuntimeError("peer is not connected")
        self._validate_timeout(timeout_seconds)
        async with asyncio.timeout(timeout_seconds):
            await self._prepare_receive_audio_locked()

    async def prepare_receive_speech_chunk(
        self,
        chunk: SpeechChunk,
        stream_identity: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        if type(chunk) is not SpeechChunk:
            raise TypeError("chunk must be an exact SpeechChunk")
        if type(stream_identity) is not str:
            raise TypeError("stream_identity must be an exact built-in string")
        if not stream_identity or len(stream_identity) > 128:
            raise ValueError("stream_identity must contain 1 to 128 characters")
        if not self._connected:
            raise RuntimeError("peer is not connected")
        self._validate_timeout(timeout_seconds)
        async with asyncio.timeout(timeout_seconds):
            await self._prepare_receive_audio_locked(stream_identity)

    async def _prepare_receive_audio_locked(
        self,
        expected_name: str | None = None,
    ) -> None:
        if self._audio_stream is not None and (
            expected_name is None or self._audio_stream_name == expected_name
        ):
            return
        if self._audio_stream is not None:
            await self._audio_stream.aclose()
            self._audio_stream = None
            self._audio_stream_name = None
            self._audio_stream_identity = None
        while self._audio_stream is None:
            participant_identity, track_name, track = await self._subscribed_audio.get()
            if expected_name is not None and track_name != expected_name:
                continue
            self._audio_stream = rtc.AudioStream(
                track,
                capacity=self._AUDIO_STREAM_CAPACITY,
                sample_rate=self._sample_rate_hz,
                num_channels=self._channels,
            )
            self._audio_stream_name = track_name
            self._audio_stream_identity = participant_identity

    async def disconnect(self, *, timeout_seconds: float = 10) -> None:
        """Release resources and leave the room, retaining failed cleanup for retry."""
        self._validate_timeout(timeout_seconds)

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        errors: list[Exception] = []

        async def attempt(operation: Callable[[], Awaitable[None]]) -> bool:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                errors.append(TimeoutError("disconnect timed out"))
                return False
            try:
                await asyncio.wait_for(operation(), remaining)
            except Exception as error:
                errors.append(error)
                return False
            return True

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            errors.append(TimeoutError("disconnect timed out"))
        else:
            try:
                await asyncio.wait_for(self._publish_lock.acquire(), remaining)
            except Exception as error:
                errors.append(error)
            else:
                try:
                    publication_pending = False
                    if self._publication_task is not None:
                        publication_task = self._publication_task
                        if not await attempt(self._settle_publication):
                            if publication_task.done():
                                self._publication_task = None
                            else:
                                publication_pending = True

                    if not publication_pending:
                        if self._audio_stream is not None:
                            stream = self._audio_stream
                            if await attempt(stream.aclose):
                                self._audio_stream = None
                                self._audio_stream_name = None
                                self._audio_stream_identity = None

                        if self._connected and self._publication_sid is not None:
                            publication_sid = self._publication_sid

                            async def unpublish_track() -> None:
                                await self._room.local_participant.unpublish_track(
                                    publication_sid
                                )

                            if await attempt(unpublish_track):
                                self._publication_sid = None
                                self._local_track = None

                        if self._audio_source is not None:
                            source = self._audio_source
                            if await attempt(source.aclose):
                                self._audio_source = None
                                self._active_speech_track_name = None
                                self._active_speech_chunk_key = None
                                self._speech_publication_ready = False

                        if (
                            self._connected or self._room_cleanup_required
                        ) and await attempt(self._room.disconnect):
                            self._connected = False
                            self._room_cleanup_required = False
                            self._publication_sid = None
                            self._local_track = None

                        if (
                            not self._connected
                            and not self._room_cleanup_required
                            and self._publication_task is None
                            and self._audio_source is None
                        ):
                            self._publication_sid = None
                            self._local_track = None
                finally:
                    self._publish_lock.release()

        if (
            not self._connected
            and not self._room_cleanup_required
            and self._publication_task is None
            and self._audio_stream is None
            and self._audio_source is None
        ):
            self._closed = True
            self._clear_subscribed_audio()

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("LiveKit disconnect cleanup failed", errors)
