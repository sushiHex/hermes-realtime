"""Reconnect-safe LiveKit transport binding for the production conversation worker."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import cast

from hermes_realtime.conversation.worker import ReconnectSafeConversationWorker
from hermes_realtime.production_observation import (
    CloseResultV1,
    CloseStageV1,
    _ProductionObservationRecorderV1,
)

from .adapter import LiveKitRoomPeer
from .playback import ReconnectSafeLiveKitAudioPublisher

_MAX_IDENTIFIER_CHARS = 128
_LOGGER = logging.getLogger(__name__)


class LiveKitConversationWorker:
    """Bind exactly one LiveKit peer generation to one conversation authority."""

    def __init__(
        self,
        *,
        runtime: ReconnectSafeConversationWorker,
        publisher: ReconnectSafeLiveKitAudioPublisher | None = None,
        production_observation_recorder: _ProductionObservationRecorderV1 | None = None,
    ) -> None:
        if type(runtime) is not ReconnectSafeConversationWorker:
            raise TypeError("runtime must be an exact reconnect-safe worker")
        if publisher is not None and type(publisher) is not ReconnectSafeLiveKitAudioPublisher:
            raise TypeError("publisher must be an exact reconnect-safe publisher")
        if (
            production_observation_recorder is not None
            and type(production_observation_recorder) is not _ProductionObservationRecorderV1
        ):
            raise TypeError("production observation recorder must be exact or None")
        self._runtime = runtime
        self._publisher = publisher
        self._peer: LiveKitRoomPeer | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._generation: int | None = None
        self._participant_identity: str | None = None
        self._receiver_error: BaseException | None = None
        self._receiver_failure_cleanup: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        self._peer_requires_disconnect = False
        self._runtime_binding_open = False
        self._publisher_bound = False
        self._runtime_closed = False
        self._closed = False
        self._production_observation_recorder = production_observation_recorder

    @property
    def active_generation(self) -> int | None:
        return self._generation

    @property
    def receiver_error(self) -> BaseException | None:
        return self._receiver_error

    async def connect(
        self,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float = 10,
    ) -> int:
        """Connect an initial peer and admit its authenticated media generation."""

        self._validate_binding(
            peer,
            room_name,
            participant_identity,
            timeout_seconds,
        )
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("LiveKit conversation worker is closed")
            if self._peer is not None:
                raise RuntimeError("LiveKit conversation worker is already connected")
            return await self._connect_locked(
                peer,
                room_name=room_name,
                participant_identity=participant_identity,
                timeout_seconds=timeout_seconds,
            )

    async def reconnect(
        self,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float = 10,
    ) -> int:
        """Settle the old receiver/peer before creating a fresh binding generation."""

        self._validate_binding(
            peer,
            room_name,
            participant_identity,
            timeout_seconds,
        )
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("LiveKit conversation worker is closed")
            await self._disconnect_locked(timeout_seconds=timeout_seconds)
            return await self._connect_locked(
                peer,
                room_name=room_name,
                participant_identity=participant_identity,
                timeout_seconds=timeout_seconds,
            )

    async def submit_final_transcript(
        self,
        *,
        participant_identity: str,
        session_generation: int,
        typed_sequence: int,
        text: str,
    ) -> None:
        """Route authenticated typed fallback through the active media generation."""

        if type(participant_identity) is not str or type(text) is not str:
            raise TypeError("typed transcript values must be exact built-in strings")
        if type(session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if type(typed_sequence) is not int:
            raise TypeError("typed_sequence must be an exact integer")
        if not 1 <= typed_sequence <= (1 << 63) - 1:
            raise ValueError("typed_sequence is outside the supported range")
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("LiveKit conversation worker is closed")
            if (
                self._generation != session_generation
                or self._participant_identity != participant_identity
            ):
                raise PermissionError("typed transcript does not match the active binding")
            await self._runtime.submit_final_transcript(
                participant_identity=participant_identity,
                session_generation=session_generation,
                typed_sequence=typed_sequence,
                text=text,
            )

    async def close(self) -> None:
        operation = self._close_operation
        if operation is None or self._close_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="livekit-conversation-worker-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    async def disconnect(self, *, timeout_seconds: float = 10) -> None:
        """Settle the active transport binding without closing shared authority."""

        if type(timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an exact number")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be between 0 and 60")
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("LiveKit conversation worker is closed")
            await self._disconnect_locked(timeout_seconds=float(timeout_seconds))

    @staticmethod
    def _close_failed(operation: asyncio.Task[None]) -> bool:
        if not operation.done():
            return False
        if operation.cancelled():
            return True
        return operation.exception() is not None

    async def _close_owned(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
            errors: list[BaseException] = []
            try:
                await self._disconnect_locked(timeout_seconds=10)
            except BaseException as error:
                errors.append(error)
            if not self._runtime_closed:
                try:
                    await self._runtime.close()
                except BaseException as error:
                    errors.append(error)
                else:
                    self._runtime_closed = True
            if len(errors) == 1:
                self._record_close_result(tuple(errors))
                raise errors[0]
            if errors:
                self._record_close_result(tuple(errors))
                raise BaseExceptionGroup("LiveKit conversation worker close failed", errors)
            self._record_close_result(())

    def _record_close_result(self, errors: tuple[BaseException, ...]) -> None:
        recorder = self._production_observation_recorder
        if recorder is None:
            return
        recorder.record_close_stage(
            stage=CloseStageV1.LIVEKIT_WORKER,
            result=(
                CloseResultV1.SUCCEEDED
                if not errors
                else CloseResultV1.CANCELLED
                if all(isinstance(error, asyncio.CancelledError) for error in errors)
                else CloseResultV1.FAILED
            ),
        )

    async def _connect_locked(
        self,
        peer: LiveKitRoomPeer,
        *,
        room_name: str,
        participant_identity: str,
        timeout_seconds: float,
    ) -> int:
        peer.bind_remote_identity(participant_identity)
        self._peer = peer
        self._peer_requires_disconnect = True
        try:
            await peer.connect(room_name, timeout_seconds=timeout_seconds)
            generation = await self._runtime.bind(participant_identity)
            self._runtime_binding_open = True
            if self._publisher is not None:
                await self._publisher.bind(peer)
                self._publisher_bound = True
        except BaseException as binding_error:
            cleanup_errors: list[BaseException] = []
            if self._publisher_bound:
                try:
                    assert self._publisher is not None
                    await self._publisher.unbind(peer)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                else:
                    self._publisher_bound = False
            if self._runtime_binding_open:
                try:
                    await self._runtime.close_binding()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                else:
                    self._runtime_binding_open = False
            if self._peer_requires_disconnect:
                try:
                    await peer.disconnect(timeout_seconds=timeout_seconds)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                else:
                    self._peer_requires_disconnect = False
            if not self._publisher_bound and not self._peer_requires_disconnect:
                self._peer = None
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "LiveKit binding and peer cleanup failed",
                    [binding_error, *cleanup_errors],
                ) from None
            raise
        self._generation = generation
        self._participant_identity = participant_identity
        self._receiver_error = None
        receiver = asyncio.create_task(
            self._runtime.run_source(peer, generation),
            name=f"livekit-conversation-receiver:{generation}",
        )
        self._receiver = receiver
        receiver.add_done_callback(self._receiver_done)
        return generation

    async def _disconnect_locked(self, *, timeout_seconds: float) -> None:
        receiver = self._receiver
        peer = self._peer
        self._generation = None
        self._participant_identity = None
        errors: list[BaseException] = []
        if receiver is not None:
            receiver.cancel()
            caller_cancellation: asyncio.CancelledError | None = None
            try:
                done, pending = await asyncio.wait(
                    {receiver},
                    timeout=min(timeout_seconds, 1.0),
                )
            except asyncio.CancelledError as error:
                caller_cancellation = error
                done, pending = await asyncio.wait(
                    {receiver},
                    timeout=min(timeout_seconds, 1.0),
                )
            if pending:
                errors.append(RuntimeError("LiveKit receiver resisted cancellation"))
            else:
                if self._receiver is receiver:
                    self._receiver = None
            if caller_cancellation is not None:
                errors.append(caller_cancellation)
        if self._runtime_binding_open:
            try:
                await self._runtime.close_binding()
            except BaseException as error:
                errors.append(error)
            else:
                self._runtime_binding_open = False
        if peer is not None and self._publisher is not None and self._publisher_bound:
            try:
                await self._publisher.unbind(peer)
            except BaseException as error:
                errors.append(error)
            else:
                self._publisher_bound = False
        if peer is not None and self._peer_requires_disconnect:
            try:
                await peer.disconnect(timeout_seconds=timeout_seconds)
            except BaseException as error:
                errors.append(error)
            else:
                self._peer_requires_disconnect = False
        if (
            peer is not None
            and not self._publisher_bound
            and not self._peer_requires_disconnect
            and self._peer is peer
        ):
            self._peer = None
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("LiveKit transport disconnect failed", errors)

    def _receiver_done(self, receiver: asyncio.Task[None]) -> None:
        if receiver.cancelled():
            return
        error = receiver.exception()
        if error is not None:
            self._receiver_error = error
            _LOGGER.error(
                "LiveKit conversation receiver failed",
                exc_info=(type(error), error, error.__traceback__),
            )
            if self._receiver is receiver and not self._closed:
                cleanup = asyncio.create_task(
                    self._settle_failed_receiver(receiver),
                    name="livekit-conversation-receiver-failure-cleanup",
                )
                self._receiver_failure_cleanup = cleanup
                cleanup.add_done_callback(self._receiver_failure_cleanup_done)

    async def _settle_failed_receiver(self, receiver: asyncio.Task[None]) -> None:
        async with self._lifecycle_lock:
            if self._closed or self._receiver is not receiver:
                return
            await self._disconnect_locked(timeout_seconds=10.0)

    def _receiver_failure_cleanup_done(self, cleanup: asyncio.Task[None]) -> None:
        if self._receiver_failure_cleanup is cleanup:
            self._receiver_failure_cleanup = None
        if cleanup.cancelled():
            return
        error = cleanup.exception()
        if error is not None:
            _LOGGER.error(
                "LiveKit failed receiver cleanup failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    @staticmethod
    def _validate_binding(
        peer: object,
        room_name: object,
        participant_identity: object,
        timeout_seconds: object,
    ) -> None:
        if type(peer) is not LiveKitRoomPeer:
            raise TypeError("peer must be an exact LiveKitRoomPeer")
        for name, value in (
            ("room_name", room_name),
            ("participant_identity", participant_identity),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact built-in string")
            if not value.strip() or len(value) > _MAX_IDENTIFIER_CHARS:
                raise ValueError(f"{name} must contain 1 to 128 characters")
        if type(timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an exact number")
        timeout = cast(int | float, timeout_seconds)
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("timeout_seconds must be between 0 and 60")
