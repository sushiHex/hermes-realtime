from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from test_archive import FakeHermes, _rows

from hermes_realtime.companion.archive import VoiceArchive
from hermes_realtime.companion.host import VoiceCompanionService
from hermes_realtime.companion.integrity import VoiceBatch
from hermes_realtime.companion.store import CompanionStore
from hermes_realtime.memory import BuiltinMemorySnapshot
from hermes_realtime.protocol import VoiceMemoryEvent, VoiceMemoryRefusedEvent


class MemoryPort(FakeHermes):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0
        self.value = BuiltinMemorySnapshot("synthetic reference", "")

    def read_builtin_memory(self) -> BuiltinMemorySnapshot:
        self.reads += 1
        return self.value


def _request(generation: int = 0) -> VoiceMemoryEvent:
    return VoiceMemoryEvent(protocol_version="0.4", type="voice_memory",
                            conversation_id="conv", generation=generation)


@pytest.mark.asyncio
async def test_open_reads_without_creating_archive_then_finished_review_refreshes(
    tmp_path: Path,
) -> None:
    port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    stream = service.memory(_request())
    try:
        first = await anext(stream)
        assert first.memory == port.value.memory
        assert store.read("conv") is None
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.02)
        assert port.reads == 1 and not pending.done()
        port.value = BuiltinMemorySnapshot("synthetic correction", "")
        service._memory_finished()
        second = await asyncio.wait_for(pending, 2)
        assert second.memory == port.value.memory and second.revision > first.revision
        assert port.reads == 2
    finally:
        await stream.aclose()
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_quarantine_clears_subscribed_memory_without_another_native_read(
    tmp_path: Path,
) -> None:
    port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    await archive.open("conv")
    stream = service.memory(_request())
    try:
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        archive._quarantine("conv", "mismatch")
        reply = await asyncio.wait_for(pending, 2)
        assert type(reply) is VoiceMemoryRefusedEvent
        assert reply.category == "quarantined"
        assert port.reads == 1
    finally:
        await stream.aclose()
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state,category", [("closed", "not_ready"), ("fenced", "not_ready"),
                                           ("generation", "stale")])
async def test_unavailable_binding_never_reads(tmp_path: Path, state: str, category: str) -> None:
    port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    if state == "closed":
        service._memory_closed = True
    if state == "fenced":
        archive._fenced = True
    stream = service.memory(_request(1 if state == "generation" else 0))
    try:
        reply = await anext(stream)
        assert type(reply) is VoiceMemoryRefusedEvent and reply.category == category
        assert port.reads == 0
    finally:
        await stream.aclose()
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_inflight_native_read_is_owned_until_shutdown_quiesces(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()

    class HeldPort(MemoryPort):
        def read_builtin_memory(self) -> BuiltinMemorySnapshot:
            entered.set()
            assert release.wait(5)
            return super().read_builtin_memory()

    port = HeldPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    stream = service.memory(_request())
    pending = asyncio.create_task(anext(stream))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        closing = asyncio.create_task(service.close())
        await asyncio.sleep(0.02)
        assert not closing.done()
        assert service._memory_reads
        release.set()
        await asyncio.wait_for(closing, 2)
        assert not service._memory_reads
    finally:
        release.set()
        await stream.aclose()
        if closing is not None:
            await closing
        else:
            await service.close()
        store.close()


@pytest.mark.asyncio
async def test_remote_generation_change_invalidates_idle_subscription(tmp_path: Path) -> None:
    port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    await archive.open("conv")
    stream = service.memory(_request())
    pending = None
    try:
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await archive.archive("conv", VoiceBatch(1, 0, 1, _rows(0, 2, generation=1)))
        reply = await asyncio.wait_for(pending, 1)
        assert type(reply) is VoiceMemoryRefusedEvent and reply.category == "stale"
        assert port.reads == 1
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        await stream.aclose()
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_subscription_capacity_refuses_with_bounded_evidence(tmp_path: Path, capsys) -> None:
    port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    service._memory_events = {asyncio.Event() for _ in range(32)}
    stream = service.memory(_request())
    try:
        reply = await anext(stream)
        assert type(reply) is VoiceMemoryRefusedEvent and reply.category == "capacity"
        await stream.aclose()
        assert '[voice-memory] {"refusal": "capacity", "version": 1}' in capsys.readouterr().out
        assert port.reads == 0
    finally:
        await stream.aclose()
        await service.close()
        store.close()


@pytest.mark.asyncio
async def test_quarantine_during_native_read_discards_result(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()

    class HeldPort(MemoryPort):
        def read_builtin_memory(self) -> BuiltinMemorySnapshot:
            entered.set()
            assert release.wait(5)
            return super().read_builtin_memory()

    port = HeldPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    await archive.open("conv")
    stream = service.memory(_request())
    pending = asyncio.create_task(anext(stream))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        archive._quarantine("conv", "mismatch")
        release.set()
        reply = await asyncio.wait_for(pending, 2)
        assert type(reply) is VoiceMemoryRefusedEvent and reply.category == "quarantined"
    finally:
        release.set()
        await pending
        await stream.aclose()
        await service.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid", "capacity", "overflow"])
async def test_read_result_and_work_bounds_fail_closed(tmp_path: Path, failure: str) -> None:
    port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, port)
    service = VoiceCompanionService(archive, store, port)
    await service.start()
    tasks = set()
    if failure == "invalid":
        port.value = None  # type: ignore[assignment]
    elif failure == "capacity":
        tasks = {asyncio.create_task(asyncio.sleep(60)) for _ in range(32)}
        service._memory_reads.update(tasks)
    else:
        service._memory_revision = 2**53 - 1
        service._memory_finished()
    stream = service.memory(_request())
    try:
        reply = await anext(stream)
        assert type(reply) is VoiceMemoryRefusedEvent
        assert reply.category == ("not_ready" if failure == "overflow" else failure)
        assert port.reads == (1 if failure == "invalid" else 0)
    finally:
        await stream.aclose()
        await service.close()
        store.close()
