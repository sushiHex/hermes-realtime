import asyncio
import logging
from types import MethodType

import pytest

from hermes_realtime.conversation.worker import ReconnectSafeConversationWorker
from hermes_realtime.livekit.adapter import LiveKitRoomPeer
from hermes_realtime.livekit.playback import ReconnectSafeLiveKitAudioPublisher
from hermes_realtime.livekit.worker import LiveKitConversationWorker


@pytest.mark.asyncio
async def test_receiver_failure_settles_binding_and_disconnects_peer_for_rebind() -> None:
    runtime, runtime_events, source_started = runtime_probe()
    peer, peer_events = peer_probe("failed-receiver")
    release_failure = asyncio.Event()
    peer_disconnected = asyncio.Event()

    async def fail_source(
        self: ReconnectSafeConversationWorker,
        source: object,
        session_generation: int,
        *,
        receive_timeout_seconds: float = 1.0,
    ) -> None:
        del self, source, receive_timeout_seconds
        runtime_events.append(f"source:{session_generation}")
        source_started.put_nowait(session_generation)
        await release_failure.wait()
        raise RuntimeError("synthetic connected receiver failure")

    async def observe_disconnect(
        self: LiveKitRoomPeer,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, timeout_seconds
        peer_events.append("disconnect:failed-receiver")
        peer_disconnected.set()

    runtime.run_source = MethodType(fail_source, runtime)  # type: ignore[method-assign]
    peer.disconnect = MethodType(observe_disconnect, peer)  # type: ignore[method-assign]
    worker = LiveKitConversationWorker(runtime=runtime)

    assert await worker.connect(
        peer,
        room_name="room",
        participant_identity="browser_user",
    ) == 1
    assert await source_started.get() == 1
    release_failure.set()
    await asyncio.wait_for(peer_disconnected.wait(), timeout=0.5)

    assert isinstance(worker.receiver_error, RuntimeError)
    assert runtime_events == [
        "bind:browser_user:1",
        "source:1",
        "runtime:close-binding",
    ]
    assert peer_events == [
        "bind:failed-receiver:browser_user",
        "connect:failed-receiver:room",
        "disconnect:failed-receiver",
    ]
    assert worker.active_generation is None

    await worker.close()


@pytest.mark.asyncio
async def test_receiver_failure_is_retained_and_logged_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime, _, _ = runtime_probe()
    worker = LiveKitConversationWorker(runtime=runtime)

    async def fail() -> None:
        raise RuntimeError("synthetic receiver failure")

    receiver = asyncio.create_task(fail())
    with pytest.raises(RuntimeError, match="synthetic receiver failure"):
        await receiver

    with caplog.at_level(logging.ERROR, logger="hermes_realtime.livekit.worker"):
        worker._receiver_done(receiver)

    assert worker.receiver_error is receiver.exception()
    assert "LiveKit conversation receiver failed" in caplog.text
    assert caplog.records[-1].exc_info is not None


def peer_probe(name: str) -> tuple[LiveKitRoomPeer, list[str]]:
    peer = object.__new__(LiveKitRoomPeer)
    events: list[str] = []

    def bind_remote_identity(
        self: LiveKitRoomPeer,
        participant_identity: str,
    ) -> None:
        del self
        events.append(f"bind:{name}:{participant_identity}")

    async def connect(
        self: LiveKitRoomPeer,
        room_name: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, timeout_seconds
        events.append(f"connect:{name}:{room_name}")

    async def disconnect(
        self: LiveKitRoomPeer,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, timeout_seconds
        events.append(f"disconnect:{name}")

    peer.bind_remote_identity = MethodType(  # type: ignore[method-assign]
        bind_remote_identity,
        peer,
    )
    peer.connect = MethodType(connect, peer)  # type: ignore[method-assign]
    peer.disconnect = MethodType(disconnect, peer)  # type: ignore[method-assign]
    return peer, events


def runtime_probe() -> tuple[
    ReconnectSafeConversationWorker,
    list[str],
    asyncio.Queue[int],
]:
    runtime = object.__new__(ReconnectSafeConversationWorker)
    events: list[str] = []
    source_started: asyncio.Queue[int] = asyncio.Queue()
    generation = 0

    async def bind(
        self: ReconnectSafeConversationWorker,
        participant_identity: str,
    ) -> int:
        nonlocal generation
        del self
        generation += 1
        events.append(f"bind:{participant_identity}:{generation}")
        return generation

    async def run_source(
        self: ReconnectSafeConversationWorker,
        source: object,
        session_generation: int,
        *,
        receive_timeout_seconds: float = 1.0,
    ) -> None:
        del self, source, receive_timeout_seconds
        events.append(f"source:{session_generation}")
        source_started.put_nowait(session_generation)
        await asyncio.Event().wait()

    async def close(self: ReconnectSafeConversationWorker) -> None:
        del self
        events.append("runtime:close")

    async def close_binding(self: ReconnectSafeConversationWorker) -> None:
        del self
        events.append("runtime:close-binding")

    async def submit_final_transcript(
        self: ReconnectSafeConversationWorker,
        *,
        participant_identity: str,
        session_generation: int,
        typed_sequence: int,
        text: str,
    ) -> None:
        del self
        events.append(
            f"typed:{participant_identity}:{session_generation}:{typed_sequence}:{text}"
        )

    runtime.bind = MethodType(bind, runtime)  # type: ignore[method-assign]
    runtime.run_source = MethodType(run_source, runtime)  # type: ignore[method-assign]
    runtime.submit_final_transcript = MethodType(  # type: ignore[method-assign]
        submit_final_transcript,
        runtime,
    )
    runtime.close_binding = MethodType(close_binding, runtime)  # type: ignore[method-assign]
    runtime.close = MethodType(close, runtime)  # type: ignore[method-assign]
    return runtime, events, source_started


@pytest.mark.asyncio
async def test_livekit_worker_reconnects_only_after_old_receiver_and_peer_close() -> None:
    runtime, runtime_events, source_started = runtime_probe()
    first, first_events = peer_probe("first")
    second, second_events = peer_probe("second")
    worker = LiveKitConversationWorker(runtime=runtime)

    first_generation = await worker.connect(
        first,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == 1
    second_generation = await worker.reconnect(
        second,
        room_name="room",
        participant_identity="browser_user",
    )

    assert first_generation == 1
    assert second_generation == 2
    assert await source_started.get() == 2
    assert first_events == [
        "bind:first:browser_user",
        "connect:first:room",
        "disconnect:first",
    ]
    assert second_events == ["bind:second:browser_user", "connect:second:room"]
    assert runtime_events[:5] == [
        "bind:browser_user:1",
        "source:1",
        "runtime:close-binding",
        "bind:browser_user:2",
        "source:2",
    ]

    await worker.close()
    assert second_events == [
        "bind:second:browser_user",
        "connect:second:room",
        "disconnect:second",
    ]
    assert runtime_events[-1] == "runtime:close"


@pytest.mark.asyncio
async def test_livekit_worker_recovers_after_peer_connect_failure() -> None:
    runtime, _runtime_events, source_started = runtime_probe()
    failed, failed_events = peer_probe("failed")
    replacement, replacement_events = peer_probe("replacement")

    async def fail_connect(
        self: LiveKitRoomPeer,
        room_name: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, room_name, timeout_seconds
        failed_events.append("connect-failed")
        raise TimeoutError("signal timed out")

    failed.connect = MethodType(fail_connect, failed)  # type: ignore[method-assign]
    worker = LiveKitConversationWorker(runtime=runtime)

    with pytest.raises(TimeoutError, match="signal timed out"):
        await worker.connect(
            failed,
            room_name="room",
            participant_identity="browser_user",
        )
    generation = await worker.connect(
        replacement,
        room_name="room",
        participant_identity="browser_user",
    )

    assert generation == 1
    assert await source_started.get() == 1
    assert failed_events == [
        "bind:failed:browser_user",
        "connect-failed",
        "disconnect:failed",
    ]
    assert replacement_events == [
        "bind:replacement:browser_user",
        "connect:replacement:room",
    ]
    await worker.close()


@pytest.mark.asyncio
async def test_public_disconnect_is_nonterminal_and_allows_fresh_generation() -> None:
    runtime, runtime_events, source_started = runtime_probe()
    first, first_events = peer_probe("first")
    second, second_events = peer_probe("second")
    worker = LiveKitConversationWorker(runtime=runtime)

    assert await worker.connect(
        first, room_name="room", participant_identity="browser_first"
    ) == 1
    assert await source_started.get() == 1
    await worker.disconnect()
    assert worker.active_generation is None
    assert await worker.connect(
        second, room_name="room", participant_identity="browser_second"
    ) == 2
    assert await source_started.get() == 2

    assert first_events[-1] == "disconnect:first"
    assert second_events[-1] == "connect:second:room"
    assert runtime_events.count("runtime:close-binding") == 1
    assert "runtime:close" not in runtime_events
    await worker.close()
    await worker.close()
    assert runtime_events.count("runtime:close") == 1
    with pytest.raises(RuntimeError, match="closed"):
        await worker.disconnect()


@pytest.mark.asyncio
async def test_livekit_worker_rebinds_reconnect_safe_audio_publisher_per_peer() -> None:
    runtime, _, source_started = runtime_probe()
    first, _ = peer_probe("publisher-first")
    second, _ = peer_probe("publisher-second")
    spare, _ = peer_probe("publisher-spare")
    publisher = ReconnectSafeLiveKitAudioPublisher()
    worker = LiveKitConversationWorker(runtime=runtime, publisher=publisher)

    await worker.connect(
        first,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == 1
    with pytest.raises(RuntimeError, match="already bound"):
        await publisher.bind(spare)

    await worker.reconnect(
        second,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == 2
    with pytest.raises(RuntimeError, match="already bound"):
        await publisher.bind(spare)

    await worker.close()
    await publisher.bind(spare)
    await publisher.unbind(spare)


@pytest.mark.asyncio
async def test_livekit_worker_routes_typed_fallback_to_active_generation() -> None:
    runtime, runtime_events, source_started = runtime_probe()
    peer, _ = peer_probe("typed")
    worker = LiveKitConversationWorker(runtime=runtime)
    generation = await worker.connect(
        peer,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == generation

    await worker.submit_final_transcript(
        participant_identity="browser_user",
        session_generation=generation,
        typed_sequence=1,
        text="typed request",
    )

    assert runtime_events[-1] == "typed:browser_user:1:1:typed request"
    await worker.close()


@pytest.mark.asyncio
async def test_failed_peer_disconnect_remains_retryable_for_reconnect() -> None:
    runtime, _, source_started = runtime_probe()
    first, first_events = peer_probe("first")
    second, _ = peer_probe("second")
    disconnect_attempts = 0

    async def flaky_disconnect(
        self: LiveKitRoomPeer,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        nonlocal disconnect_attempts
        del self, timeout_seconds
        disconnect_attempts += 1
        first_events.append(f"disconnect-attempt:{disconnect_attempts}")
        if disconnect_attempts == 1:
            raise RuntimeError("temporary disconnect failure")

    first.disconnect = MethodType(  # type: ignore[method-assign]
        flaky_disconnect,
        first,
    )
    worker = LiveKitConversationWorker(runtime=runtime)
    await worker.connect(
        first,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == 1

    with pytest.raises(RuntimeError, match="temporary disconnect failure"):
        await worker.reconnect(
            second,
            room_name="room",
            participant_identity="browser_user",
        )
    generation = await worker.reconnect(
        second,
        room_name="room",
        participant_identity="browser_user",
    )

    assert generation == 2
    assert disconnect_attempts == 2
    await worker.close()


@pytest.mark.asyncio
async def test_failed_final_close_retries_retained_peer_and_runtime() -> None:
    runtime, runtime_events, source_started = runtime_probe()
    peer, peer_events = peer_probe("close-retry")
    peer_attempts = 0
    runtime_attempts = 0

    async def flaky_disconnect(
        self: LiveKitRoomPeer,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        nonlocal peer_attempts
        del self, timeout_seconds
        peer_attempts += 1
        peer_events.append(f"disconnect-attempt:{peer_attempts}")
        if peer_attempts == 1:
            raise RuntimeError("temporary peer close failure")

    async def flaky_runtime_close(self: ReconnectSafeConversationWorker) -> None:
        nonlocal runtime_attempts
        del self
        runtime_attempts += 1
        runtime_events.append(f"runtime-close-attempt:{runtime_attempts}")
        if runtime_attempts == 1:
            raise RuntimeError("temporary runtime close failure")

    peer.disconnect = MethodType(  # type: ignore[method-assign]
        flaky_disconnect,
        peer,
    )
    runtime.close = MethodType(  # type: ignore[method-assign]
        flaky_runtime_close,
        runtime,
    )
    worker = LiveKitConversationWorker(runtime=runtime)
    await worker.connect(
        peer,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == 1

    with pytest.raises(BaseExceptionGroup, match="close failed"):
        await worker.close()
    await worker.close()

    assert peer_attempts == 2
    assert runtime_attempts == 2


@pytest.mark.asyncio
async def test_failed_connect_cleanup_remains_owned_for_final_close() -> None:
    runtime, _, _ = runtime_probe()
    peer, events = peer_probe("connect-failure")
    disconnect_attempts = 0

    async def failed_connect(
        self: LiveKitRoomPeer,
        room_name: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        del self, room_name, timeout_seconds
        raise BaseExceptionGroup(
            "LiveKit connect and cleanup failed",
            [
                RuntimeError("connect failed"),
                RuntimeError("nested cleanup failed"),
            ],
        )

    async def retry_disconnect(
        self: LiveKitRoomPeer,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        nonlocal disconnect_attempts
        del self, timeout_seconds
        disconnect_attempts += 1
        events.append(f"disconnect-retry:{disconnect_attempts}")

    peer.connect = MethodType(failed_connect, peer)  # type: ignore[method-assign]
    peer.disconnect = MethodType(  # type: ignore[method-assign]
        retry_disconnect,
        peer,
    )
    worker = LiveKitConversationWorker(runtime=runtime)

    with pytest.raises(BaseExceptionGroup, match="connect and cleanup failed"):
        await worker.connect(
            peer,
            room_name="room",
            participant_identity="browser_user",
        )
    await worker.close()

    assert disconnect_attempts == 1


@pytest.mark.asyncio
async def test_unbind_failure_still_disconnects_and_close_retry_converges_once() -> None:
    runtime, runtime_events, source_started = runtime_probe()
    peer, peer_events = peer_probe("unbind-failure")
    publisher = ReconnectSafeLiveKitAudioPublisher()
    original_unbind = publisher.unbind
    unbind_attempts = 0

    async def flaky_unbind(
        self: ReconnectSafeLiveKitAudioPublisher,
        bound_peer: LiveKitRoomPeer,
    ) -> None:
        nonlocal unbind_attempts
        del self
        unbind_attempts += 1
        if unbind_attempts == 1:
            raise RuntimeError("temporary publisher unbind failure")
        await original_unbind(bound_peer)

    publisher.unbind = MethodType(flaky_unbind, publisher)  # type: ignore[method-assign]
    worker = LiveKitConversationWorker(runtime=runtime, publisher=publisher)
    await worker.connect(
        peer,
        room_name="room",
        participant_identity="browser_user",
    )
    assert await source_started.get() == 1

    with pytest.raises(RuntimeError, match="temporary publisher unbind failure"):
        await worker.close()

    assert peer_events.count("disconnect:unbind-failure") == 1
    assert runtime_events.count("runtime:close") == 1

    await worker.close()

    assert unbind_attempts == 2
    assert peer_events.count("disconnect:unbind-failure") == 1
    assert runtime_events.count("runtime:close") == 1
