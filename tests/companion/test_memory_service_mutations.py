"""Focused witnesses for memory refresh and service ownership guards."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_memory_service import MemoryPort, _request
from test_review import FakeReviewPort, _batch

from hermes_realtime.companion.archive import VoiceArchive
from hermes_realtime.companion.host import VoiceCompanionService
from hermes_realtime.companion.review import ReviewRequest, VoiceReviewCoordinator
from hermes_realtime.companion.store import CompanionStore
from hermes_realtime.memory import BuiltinMemorySnapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_actual_review_terminal_refreshes_only_after_durable_success(
    tmp_path: Path, failed: bool,
) -> None:
    memory_port = MemoryPort()
    store = CompanionStore(tmp_path / "state.db")
    archive = VoiceArchive(store, memory_port)
    review_port = FakeReviewPort()
    review_port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    review_port.failed = failed
    review = VoiceReviewCoordinator(
        archive, store, review_port, lambda _: type("Parent", (), {})(),
    )
    service = VoiceCompanionService(archive, store, memory_port, review)
    observed: list[str] = []
    review_id = ""
    original_refresh = service._memory_finished

    def observed_refresh() -> None:
        observed.append(store.review_outcome("conv", review_id))
        original_refresh()

    review._on_finished = observed_refresh
    stream = service.memory(_request())
    pending: asyncio.Task[object] | None = None
    try:
        await service.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        first = await anext(stream)
        memory_port.value = BuiltinMemorySnapshot("corrected", "")
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        admitted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        assert admitted.status == "accepted"
        review_id = admitted.review_id
        await asyncio.wait_for(review_port.started.wait(), 5)
        review_port.release.set()
        assert await review.join("conv", 5)
        assert review.outcome("conv", review_id) == ("failed" if failed else "finished")
        if failed:
            assert observed == [] and not pending.done()
            assert service._memory_revision == first.revision
        else:
            refreshed = await asyncio.wait_for(pending, 2)
            assert refreshed.memory == "corrected"
            assert observed == ["finished"]
            assert refreshed.revision > first.revision
    finally:
        review_port.release.set()
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await stream.aclose()
        await service.close()
        store.close()
