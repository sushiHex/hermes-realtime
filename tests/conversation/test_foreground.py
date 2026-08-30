import asyncio
from typing import cast

import pytest

from hermes_realtime.conversation import (
    ForegroundOutputBackpressure,
    ForegroundPublication,
    ForegroundTurnClosed,
    ForegroundTurnCoordinator,
    ForegroundTurnDrainTimeout,
    ForegroundTurnLease,
)


@pytest.mark.asyncio
async def test_turn_id_requires_exact_builtin_string_without_subclass_code() -> None:
    strip_called = False

    class SideEffectString(str):
        def strip(self, chars: str | None = None) -> str:
            nonlocal strip_called
            strip_called = True
            return super().strip(chars)

    async def runner(_: ForegroundTurnLease) -> None:
        return None

    coordinator = ForegroundTurnCoordinator()
    with pytest.raises(TypeError, match="turn_id"):
        await coordinator.start(cast(str, 123), runner)
    with pytest.raises(TypeError, match="turn_id"):
        await coordinator.start(SideEffectString("turn_001"), runner)

    assert strip_called is False


@pytest.mark.asyncio
async def test_replacing_foreground_turn_cancels_and_drains_previous_owner() -> None:
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    release_second = asyncio.Event()

    async def first(_: ForegroundTurnLease) -> None:
        first_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def second(_: ForegroundTurnLease) -> None:
        await release_second.wait()

    coordinator = ForegroundTurnCoordinator()
    first_lease = await coordinator.start("turn_001", first)
    await asyncio.wait_for(first_started.wait(), timeout=1)

    second_lease = await coordinator.start("turn_002", second)

    assert first_cancelled.is_set()
    assert first_lease.is_current is False
    assert second_lease.is_current is True
    assert coordinator.active_turn_id == "turn_002"
    assert coordinator.active_task_count == 1

    release_second.set()
    await coordinator.cancel()
    assert coordinator.active_turn_id is None
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_stale_lease_cannot_cancel_newer_foreground_owner() -> None:
    release_first = asyncio.Event()
    release_second = asyncio.Event()

    async def first(_: ForegroundTurnLease) -> None:
        await release_first.wait()

    async def second(_: ForegroundTurnLease) -> None:
        await release_second.wait()

    coordinator = ForegroundTurnCoordinator()
    first_lease = await coordinator.start("turn_001", first)
    second_lease = await coordinator.start("turn_002", second)

    assert await coordinator.cancel_if_current(first_lease) is False
    assert second_lease.is_current is True
    assert coordinator.active_turn_id == "turn_002"

    release_second.set()
    await coordinator.cancel()


@pytest.mark.asyncio
async def test_stale_generation_cannot_publish_after_replacement() -> None:
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    allow_stale_attempt = asyncio.Event()
    release_second = asyncio.Event()
    stale_result: list[bool] = []

    async def first(lease: ForegroundTurnLease) -> None:
        first_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            first_cancelled.set()
        await allow_stale_attempt.wait()
        stale_result.append(await lease.publish("stale"))

    async def second(lease: ForegroundTurnLease) -> None:
        assert await lease.publish("current") is True
        await release_second.wait()

    coordinator = ForegroundTurnCoordinator()
    await coordinator.start("turn_001", first)
    await asyncio.wait_for(first_started.wait(), timeout=1)

    replacement = asyncio.create_task(coordinator.start("turn_002", second))
    await asyncio.wait_for(first_cancelled.wait(), timeout=1)
    allow_stale_attempt.set()
    second_lease = await asyncio.wait_for(replacement, timeout=1)
    await asyncio.sleep(0)

    assert stale_result == [False]
    publication = await asyncio.wait_for(coordinator.next_publication(), timeout=1)
    assert publication.text == "current"
    assert publication.turn_id == "turn_002"
    assert publication.generation == second_lease.generation
    assert coordinator.pending_publication_count == 0
    assert second_lease.is_current is True

    release_second.set()
    await coordinator.cancel()


@pytest.mark.asyncio
async def test_new_turn_invalidates_claimed_and_queued_completed_output() -> None:
    async def completed(lease: ForegroundTurnLease) -> None:
        assert await lease.publish("claimed-old") is True
        assert await lease.publish("queued-old") is True

    release = asyncio.Event()

    async def replacement(_: ForegroundTurnLease) -> None:
        await release.wait()

    coordinator = ForegroundTurnCoordinator()
    await coordinator.start("turn_001", completed)
    claimed = await asyncio.wait_for(coordinator.next_publication(), timeout=1)
    deadline = asyncio.get_running_loop().time() + 1
    while coordinator.active_task_count and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)

    await coordinator.start("turn_002", replacement)

    assert claimed.is_valid is False
    assert coordinator.pending_publication_count == 0
    release.set()
    await coordinator.close()


@pytest.mark.asyncio
async def test_current_lease_cannot_be_used_by_a_non_owner_task() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(_: ForegroundTurnLease) -> None:
        started.set()
        await release.wait()

    coordinator = ForegroundTurnCoordinator()
    lease = await coordinator.start("turn_001", runner)
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await lease.publish("not-owner") is False
    assert coordinator.pending_publication_count == 0

    release.set()
    await coordinator.cancel()
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_publication_is_generation_labelled_and_bounded() -> None:
    release = asyncio.Event()

    async def runner(lease: ForegroundTurnLease) -> None:
        assert await lease.publish("current") is True
        with pytest.raises(ForegroundOutputBackpressure):
            await lease.publish("overflow")
        await release.wait()

    coordinator = ForegroundTurnCoordinator(output_capacity=1)
    lease = await coordinator.start("turn_001", runner)
    publication = await asyncio.wait_for(coordinator.next_publication(), timeout=1)

    assert publication == ForegroundPublication(
        turn_id="turn_001",
        generation=lease.generation,
        text="current",
    )
    assert coordinator.pending_publication_count == 0
    release.set()
    await coordinator.cancel()


@pytest.mark.asyncio
async def test_owner_context_lifecycle_reentry_fails_without_deadlock() -> None:
    coordinator = ForegroundTurnCoordinator()
    observed: list[str] = []
    lifecycle_tasks: list[asyncio.Task[None]] = []

    async def runner(lease: ForegroundTurnLease) -> None:
        assert await lease.publish("current") is True
        lifecycle_tasks.append(asyncio.create_task(coordinator.cancel()))
        with pytest.raises(RuntimeError, match="owner task"):
            await asyncio.wait_for(lifecycle_tasks[0], timeout=0.1)
        observed.append("rejected")

    await coordinator.start("turn_001", runner)
    publication = await asyncio.wait_for(coordinator.next_publication(), timeout=1)
    assert publication.text == "current"
    deadline = asyncio.get_running_loop().time() + 1
    while coordinator.active_task_count and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)

    assert observed == ["rejected"]
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_cancel_invalidates_before_cancelled_runner_can_publish() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    allow_stale_attempt = asyncio.Event()

    async def runner(lease: ForegroundTurnLease) -> None:
        assert await lease.publish("claimed-before-cancel") is True
        assert await lease.publish("queued-before-cancel") is True
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
        await allow_stale_attempt.wait()
        assert await lease.publish("late") is False

    coordinator = ForegroundTurnCoordinator()
    lease = await coordinator.start("turn_001", runner)
    await asyncio.wait_for(started.wait(), timeout=1)
    claimed = await asyncio.wait_for(coordinator.next_publication(), timeout=1)
    assert claimed.is_valid is True

    cancellation = asyncio.create_task(coordinator.cancel())
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert lease.is_current is False
    allow_stale_attempt.set()
    await asyncio.wait_for(cancellation, timeout=1)

    assert coordinator.pending_publication_count == 0
    assert claimed.is_valid is False
    await asyncio.wait_for(claimed.wait_invalidated(), timeout=1)
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_caller_cancellation_is_not_swallowed_while_draining_owner() -> None:
    started = asyncio.Event()
    first_cancellation = asyncio.Event()

    async def runner(_: ForegroundTurnLease) -> None:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            first_cancellation.set()
        await asyncio.Future()

    coordinator = ForegroundTurnCoordinator()
    await coordinator.start("turn_001", runner)
    await asyncio.wait_for(started.wait(), timeout=1)

    cancellation = asyncio.create_task(coordinator.cancel())
    await asyncio.wait_for(first_cancellation.wait(), timeout=1)
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await cancellation

    await asyncio.sleep(0)
    assert coordinator.active_turn_id is None
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_caller_cancellation_wins_simultaneous_owner_settlement() -> None:
    started = asyncio.Event()
    drain_tasks: list[asyncio.Task[None]] = []

    async def runner(_: ForegroundTurnLease) -> None:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            current = asyncio.current_task()
            assert current is not None
            current.add_done_callback(lambda _: drain_tasks[0].cancel())

    coordinator = ForegroundTurnCoordinator()
    await coordinator.start("turn_001", runner)
    await asyncio.wait_for(started.wait(), timeout=1)
    drain_tasks.append(asyncio.create_task(coordinator.cancel()))

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(drain_tasks[0], timeout=1)

    assert drain_tasks[0].cancelling() > 0
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_stubborn_owner_times_out_without_admitting_replacement() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn(_: ForegroundTurnLease) -> None:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()

    coordinator = ForegroundTurnCoordinator(drain_timeout=0.02)
    lease = await coordinator.start("turn_001", stubborn)
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(
        ForegroundTurnDrainTimeout, match="turn_001"
    ) as first_timeout:
        await coordinator.cancel()
    assert first_timeout.value.turn_id == "turn_001"

    assert cancellation_seen.is_set()
    assert lease.is_current is False
    assert coordinator.active_turn_id is None
    assert coordinator.active_task_count == 1

    replacement_started = False

    async def replacement(_: ForegroundTurnLease) -> None:
        nonlocal replacement_started
        replacement_started = True

    with pytest.raises(
        ForegroundTurnDrainTimeout, match="turn_001"
    ) as replacement_timeout:
        await coordinator.start("turn_002", replacement)
    assert replacement_timeout.value.turn_id == "turn_001"
    assert replacement_started is False
    assert coordinator.active_task_count == 1

    release.set()
    deadline = asyncio.get_running_loop().time() + 1
    while coordinator.active_task_count and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_caller_cancellation_during_stubborn_drain_is_bounded() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn(_: ForegroundTurnLease) -> None:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()

    coordinator = ForegroundTurnCoordinator(drain_timeout=0.02)
    await coordinator.start("turn_001", stubborn)
    await asyncio.wait_for(started.wait(), timeout=1)
    cancellation = asyncio.create_task(coordinator.cancel())
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cancellation, timeout=1)

    assert coordinator.active_turn_id is None
    assert coordinator.active_task_count == 1
    release.set()
    deadline = asyncio.get_running_loop().time() + 1
    while coordinator.active_task_count and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)
    assert coordinator.active_task_count == 0


@pytest.mark.asyncio
async def test_close_wakes_consumers_and_invalidates_all_publications() -> None:
    empty = ForegroundTurnCoordinator()
    blocked_consumer = asyncio.create_task(empty.next_publication())
    await asyncio.sleep(0)
    await empty.close()
    with pytest.raises(ForegroundTurnClosed):
        await asyncio.wait_for(blocked_consumer, timeout=1)

    coordinator = ForegroundTurnCoordinator()

    async def runner(lease: ForegroundTurnLease) -> None:
        assert await lease.publish("claimed") is True
        assert await lease.publish("queued") is True

    await coordinator.start("turn_001", runner)
    claimed = await asyncio.wait_for(coordinator.next_publication(), timeout=1)
    deadline = asyncio.get_running_loop().time() + 1
    while coordinator.active_task_count and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)
    await coordinator.close()

    assert claimed.is_valid is False
    await asyncio.wait_for(claimed.wait_invalidated(), timeout=1)
    assert coordinator.pending_publication_count == 0
    with pytest.raises(ForegroundTurnClosed):
        await coordinator.next_publication()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_rejects_new_turns() -> None:
    coordinator = ForegroundTurnCoordinator()
    await coordinator.close()
    await coordinator.close()

    async def runner(_: ForegroundTurnLease) -> None:
        return None

    with pytest.raises(RuntimeError, match="closed"):
        await coordinator.start("turn_001", runner)
