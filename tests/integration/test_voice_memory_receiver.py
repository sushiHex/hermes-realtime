import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest

from hermes_realtime.conversation import ConversationContextStore
from hermes_realtime.integration import voice_memory
from hermes_realtime.integration.voice_memory import VoiceMemoryReceiver
from hermes_realtime.memory import BuiltinMemorySnapshot
from hermes_realtime.protocol import (
    VOICE_MEMORY_CAPABILITY,
    VoiceMemoryEvent,
    VoiceMemoryRefusedEvent,
    VoiceMemorySnapshotEvent,
)


class MemoryLink:
    def __init__(self) -> None:
        self.capabilities = frozenset({VOICE_MEMORY_CAPABILITY})
        self.events: asyncio.Queue[object] = asyncio.Queue()
        self.started = asyncio.Event()
        self.request: VoiceMemoryEvent | None = None
        self.closed = False
        self.closed_event = asyncio.Event()

    async def memory(
        self, request: VoiceMemoryEvent
    ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]:
        self.request = request
        self.started.set()
        while True:
            event = await self.events.get()
            if event is None:
                return
            yield cast(VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent, event)

    async def close(self) -> None:
        self.closed = True
        self.closed_event.set()


class FailedAfterHandshakeLink(MemoryLink):
    def __init__(self, *, snapshot: bool) -> None:
        super().__init__()
        self.snapshot = snapshot

    async def memory(
        self, request: VoiceMemoryEvent,
    ) -> AsyncIterator[VoiceMemorySnapshotEvent | VoiceMemoryRefusedEvent]:
        self.request = request
        self.started.set()
        if self.snapshot:
            yield _snapshot(1, "Delivered preference.")
        raise ConnectionError("synthetic post-handshake loss")


async def _retry_delays(
    monkeypatch: pytest.MonkeyPatch, *, snapshot_connection: int | None,
) -> tuple[list[float], bool]:
    context = ConversationContextStore()
    changes = _observe(context)
    delays: list[float] = []
    reached = asyncio.Event()
    hold = asyncio.Event()
    original_sleep = asyncio.sleep
    connections = 0

    async def sleeping(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 4:
            reached.set()
            await hold.wait()
        else:
            await original_sleep(0)

    async def connect() -> FailedAfterHandshakeLink:
        nonlocal connections
        connections += 1
        return FailedAfterHandshakeLink(snapshot=connections == snapshot_connection)

    monkeypatch.setattr(voice_memory.asyncio, "sleep", sleeping)
    receiver = VoiceMemoryReceiver(
        context, connect, binding=lambda: ("conversation_1", 0),
        initial_backoff_seconds=0.01, max_backoff_seconds=0.04,
    )
    receiver.start()
    try:
        await asyncio.wait_for(reached.wait(), 1)
    finally:
        await receiver.close()
    observed = []
    while not changes.empty():
        observed.append(changes.get_nowait())
    return delays, BuiltinMemorySnapshot("Delivered preference.", "Ari") in observed


@pytest.mark.asyncio
async def test_post_handshake_failures_increase_backoff_to_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays, delivered = await _retry_delays(monkeypatch, snapshot_connection=None)
    assert delays == [0.01, 0.02, 0.04, 0.04]
    assert not delivered


@pytest.mark.asyncio
async def test_first_valid_snapshot_resets_retry_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays, delivered = await _retry_delays(monkeypatch, snapshot_connection=3)
    assert delivered
    assert delays == [0.01, 0.02, 0.01, 0.02]


def _observe(context: ConversationContextStore) -> asyncio.Queue[BuiltinMemorySnapshot | None]:
    changes: asyncio.Queue[BuiltinMemorySnapshot | None] = asyncio.Queue()
    original = context.set_memory

    def set_memory(value: BuiltinMemorySnapshot | None) -> None:
        before = context.snapshot().memory
        original(value)
        if context.snapshot().memory != before:
            changes.put_nowait(value)

    context.set_memory = set_memory  # type: ignore[method-assign]
    return changes


def _snapshot(revision: int, memory: str) -> VoiceMemorySnapshotEvent:
    return VoiceMemorySnapshotEvent(
        protocol_version="0.3",
        type="voice_memory_snapshot",
        conversation_id="conversation_1",
        generation=0,
        revision=revision,
        memory=memory,
        user="Ari",
        truncated=False,
    )


@pytest.mark.asyncio
async def test_receiver_applies_pushes_and_clears_on_refusal() -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    link = MemoryLink()

    async def connect() -> MemoryLink:
        return link

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    receiver.start()
    try:
        await asyncio.wait_for(link.started.wait(), 1)
        assert link.request == VoiceMemoryEvent(
            protocol_version="0.3", type="voice_memory",
            conversation_id="conversation_1", generation=0,
        )
        await link.events.put(_snapshot(1, "Prefers concise replies."))
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory="Prefers concise replies.", user="Ari"
        )
        await link.events.put(_snapshot(2, "Prefers detailed replies."))
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory="Prefers detailed replies.", user="Ari"
        )
        await link.events.put(
            VoiceMemoryRefusedEvent(
                protocol_version="0.3", type="voice_memory_refused",
                conversation_id="conversation_1", generation=0, category="quarantined",
            )
        )
        assert await asyncio.wait_for(changes.get(), 1) is None
        assert context.snapshot().memory is None
        await link.events.put(_snapshot(2, "Recovered preference."))
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory="Recovered preference.", user="Ari"
        )
    finally:
        await receiver.close()
    assert context.snapshot().memory is None


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ("pending", "capacity"))
async def test_transient_memory_refusal_retains_last_snapshot(
    category: str,
) -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    link = MemoryLink()

    async def connect() -> MemoryLink:
        return link

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    receiver.start()
    try:
        await asyncio.wait_for(link.started.wait(), 1)
        current = _snapshot(10, "Current preference.")
        await link.events.put(current)
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory=current.memory, user=current.user,
        )
        await link.events.put(VoiceMemoryRefusedEvent(
            protocol_version="0.3", type="voice_memory_refused",
            conversation_id="conversation_1", generation=0, category=category,
        ))
        await asyncio.sleep(0)
        assert context.snapshot().memory == BuiltinMemorySnapshot(
            memory=current.memory, user=current.user,
        )
        assert changes.empty()
        await link.events.put(_snapshot(10, "Recovered preference."))
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory="Recovered preference.", user="Ari",
        )
        await link.events.put(None)
        await asyncio.wait_for(link.closed_event.wait(), 1)
        assert context.snapshot().memory is None
    finally:
        await receiver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category", ("stale", "quarantined", "tombstoned", "not_ready", "invalid", "unknown")
)
async def test_nontransient_memory_refusal_clears_snapshot(category: str) -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    link = MemoryLink()

    async def connect() -> MemoryLink:
        return link

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    receiver.start()
    try:
        await asyncio.wait_for(link.started.wait(), 1)
        await link.events.put(_snapshot(1, "Current preference."))
        assert await asyncio.wait_for(changes.get(), 1) is not None
        await link.events.put(VoiceMemoryRefusedEvent(
            protocol_version="0.3", type="voice_memory_refused",
            conversation_id="conversation_1", generation=0, category=category,
        ))
        assert await asyncio.wait_for(changes.get(), 1) is None
        assert context.snapshot().memory is None
    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_receiver_clears_before_transport_reconnect_and_rebinds_once() -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    first, second = MemoryLink(), MemoryLink()
    links = iter((first, second))
    calls = 0

    async def connect() -> MemoryLink:
        nonlocal calls
        calls += 1
        return next(links)

    receiver = VoiceMemoryReceiver(
        context, connect, binding=lambda: ("conversation_1", 0),
        initial_backoff_seconds=0.01, max_backoff_seconds=0.02,
    )
    receiver.start()
    try:
        await asyncio.wait_for(first.started.wait(), 1)
        await first.events.put(_snapshot(1, "Old preference."))
        assert await asyncio.wait_for(changes.get(), 1) is not None
        await first.events.put(None)
        await asyncio.wait_for(first.closed_event.wait(), 1)
        assert context.snapshot().memory is None
        assert await asyncio.wait_for(changes.get(), 1) is None
        await asyncio.wait_for(second.started.wait(), 1)
        assert calls == 2
        assert context.snapshot().memory is None
        await second.events.put(_snapshot(1, "New preference."))
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory="New preference.", user="Ari"
        )
    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_receiver_rejects_wrong_identity_and_nonmonotonic_revision() -> None:
    for malformed in (
        VoiceMemorySnapshotEvent(
            protocol_version="0.3", type="voice_memory_snapshot",
            conversation_id="other", generation=0, revision=2,
            memory="Other profile.", user="Ari", truncated=False,
        ),
        _snapshot(0, "Regressed preference."),
    ):
        context = ConversationContextStore()
        changes = _observe(context)
        link = MemoryLink()

        async def connect(current: MemoryLink = link) -> MemoryLink:
            return current

        receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
        receiver.start()
        try:
            await asyncio.wait_for(link.started.wait(), 1)
            await link.events.put(_snapshot(1, "Valid preference."))
            assert await asyncio.wait_for(changes.get(), 1) is not None
            await link.events.put(malformed)
            assert await asyncio.wait_for(changes.get(), 1) is None
            assert context.snapshot().memory is None
        finally:
            await receiver.close()


@pytest.mark.asyncio
async def test_refusal_does_not_lower_connection_revision_high_watermark() -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    link = MemoryLink()

    async def connect() -> MemoryLink:
        return link

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    receiver.start()
    try:
        await asyncio.wait_for(link.started.wait(), 1)
        await link.events.put(_snapshot(10, "Current preference."))
        assert await asyncio.wait_for(changes.get(), 1) is not None
        await link.events.put(VoiceMemoryRefusedEvent(
            protocol_version="0.3", type="voice_memory_refused",
            conversation_id="conversation_1", generation=0, category="not_ready",
        ))
        assert await asyncio.wait_for(changes.get(), 1) is None
        await link.events.put(_snapshot(9, "Regressed preference."))
        await asyncio.wait_for(link.closed_event.wait(), 1)
        assert context.snapshot().memory is None
    finally:
        await receiver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refuse", (False, True))
async def test_equal_connection_revision_is_accepted(refuse: bool) -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    link = MemoryLink()

    async def connect() -> MemoryLink:
        return link

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    receiver.start()
    try:
        await asyncio.wait_for(link.started.wait(), 1)
        await link.events.put(_snapshot(10, "Current preference."))
        assert await asyncio.wait_for(changes.get(), 1) is not None
        if refuse:
            await link.events.put(VoiceMemoryRefusedEvent(
                protocol_version="0.3", type="voice_memory_refused",
                conversation_id="conversation_1", generation=0, category="not_ready",
            ))
            assert await asyncio.wait_for(changes.get(), 1) is None
        await link.events.put(_snapshot(10, "Recovered preference."))
        assert await asyncio.wait_for(changes.get(), 1) == BuiltinMemorySnapshot(
            memory="Recovered preference.", user="Ari"
        )
        assert not link.closed
    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_receiver_rejects_a_binding_generation_that_becomes_boolean() -> None:
    context = ConversationContextStore()
    changes = _observe(context)
    link = MemoryLink()
    generation: int | bool = 0

    async def connect() -> MemoryLink:
        return link

    receiver = VoiceMemoryReceiver(
        context, connect, binding=lambda: ("conversation_1", generation)
    )
    receiver.start()
    try:
        await asyncio.wait_for(link.started.wait(), 1)
        await link.events.put(_snapshot(1, "Valid preference."))
        assert await asyncio.wait_for(changes.get(), 1) is not None
        generation = False
        await link.events.put(_snapshot(2, "Must not enter context."))
        assert await asyncio.wait_for(changes.get(), 1) is None
        assert context.snapshot().memory is None
    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_receiver_refuses_missing_capability_without_retaining_memory() -> None:
    context = ConversationContextStore()
    context.set_memory(BuiltinMemorySnapshot(memory="Old preference.", user="Ari"))
    link = MemoryLink()
    link.capabilities = frozenset()
    calls = 0

    async def connect() -> MemoryLink:
        nonlocal calls
        calls += 1
        return link

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    receiver.start()
    try:
        await asyncio.wait_for(link.closed_event.wait(), 1)
        assert link.closed
        assert link.request is None
        assert context.snapshot().memory is None
        assert calls == 1
    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_receiver_close_clears_cache_even_before_start() -> None:
    context = ConversationContextStore()
    context.set_memory(BuiltinMemorySnapshot(memory="Old preference.", user="Ari"))

    async def connect() -> MemoryLink:
        raise AssertionError("not started")

    receiver = VoiceMemoryReceiver(context, connect, binding=lambda: ("conversation_1", 0))
    await receiver.close()
    assert context.snapshot().memory is None
