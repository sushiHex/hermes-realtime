"""The review path may start only over the verified, bounded voice archive."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path
from typing import Any

import pytest
from test_companion_host import FakeHermes

from hermes_realtime.companion import hermes_compat
from hermes_realtime.companion.archive import VoiceArchive
from hermes_realtime.companion.integrity import ArchiveRefusal, Identity, VoiceBatch, VoiceRow
from hermes_realtime.companion.review import (
    ReviewQuiescenceError,
    ReviewRequest,
    VoiceReviewCoordinator,
)
from hermes_realtime.companion.store import CompanionStore


def _batch() -> VoiceBatch:
    return VoiceBatch(0, 0, 0, (VoiceRow(Identity(0, 0), "user", "review me", False, 1.0, None),))


class FakeReviewPort:
    def __init__(self) -> None:
        self.settings_calls = 0
        self.admissions: list[tuple[int, int, bool]] = []
        self.busy = False
        self.failed = False
        self.mismatch = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.logged = False
        self.admit_entered = threading.Event()
        self.admit_release = threading.Event()
        self.block_admit = False
        self.block_verify = False
        self.verify_entered = threading.Event()
        self.verify_release = threading.Event()
        self.block_settings = False
        self.settings_entered = threading.Event()
        self.settings_release = threading.Event()
        self.finish_calls = 0
        self.interval = 10
        self.enabled = True
        self.extra_tools: list[str] = []
        self.snapshot_rows = 1
        self.cancel_error = False
        self.cancel_entered = threading.Event()
        self.cancel_release = threading.Event()
        self.block_cancel = False
        self.cancel_calls = 0

    def verify_parent_binding(self, parent: Any) -> None:
        if self.block_verify:
            self.verify_entered.set()
            self.verify_release.wait(5)

    def bind_parent_callbacks(self, parent: Any, failed: Any) -> None:
        parent._safe_print = lambda *args, **kwargs: None
        parent.background_review_callback = None
        parent._emit_auxiliary_failure = lambda *args, **kwargs: failed()

    def close_parent(self, parent: Any) -> None:
        close = getattr(parent, "close", None)
        if callable(close):
            close()

    def drain_failed_parents(self, timeout: float) -> bool:
        return True

    def settings(self) -> tuple[int, dict[str, object]]:
        self.settings_calls += 1
        if self.block_settings:
            self.settings_entered.set()
            self.settings_release.wait(5)
        return self.interval, {"enabled": self.enabled, "extra_tools": self.extra_tools}

    def admit(
        self, parent: Any, record: Any, request: ReviewRequest,
        cap: int, lease_ttl_seconds: float,
    ) -> Any:
        if self.block_admit:
            self.admit_entered.set()
            self.admit_release.wait(5)
        if self.mismatch:
            raise ArchiveRefusal("mismatch")
        if self.busy:
            return None
        self.admissions.append((request.seq_from, request.seq_through, request.closing))
        return ([{"role": "user", "content": "review me"}] * self.snapshot_rows, object())

    def spawn(self, parent: Any, snapshot: Any, token: Any, task_cfg: Any) -> Any:
        def target() -> None:
            self.started._loop.call_soon_threadsafe(self.started.set)  # type: ignore[attr-defined]
            asyncio.run_coroutine_threadsafe(self.release.wait(), self.started._loop).result(5)
            if self.failed:
                if self.logged:
                    logging.getLogger("agent.background_review").warning("PRIVATE REVIEW TEXT")
                parent._emit_auxiliary_failure("review", RuntimeError("failed"))
            parent._safe_print("private review summary")

        return target

    def finish(self, parent: Any, token: Any) -> None:
        self.finish_calls += 1

    def cancel(self, parent: Any, token: Any) -> None:
        self.cancel_calls += 1
        if self.block_cancel:
            self.cancel_entered.set()
            self.cancel_release.wait(5)
        if self.cancel_error:
            raise RuntimeError("native cancel failed")


@pytest.mark.asyncio
async def test_cancelled_parent_construction_is_owned_and_closed(tmp_path: Path) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    entered = threading.Event()
    release = threading.Event()
    closed: list[bool] = []
    parent = type("Parent", (), {"close": lambda self: closed.append(True)})()

    def factory(session_id: str) -> Any:
        entered.set()
        if not release.wait(5):
            raise RuntimeError("parent construction did not release")
        return parent

    review = VoiceReviewCoordinator(archive, store, port, factory)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        request = ReviewRequest("conv", 0, 0, 0, True, True, False)
        pending = asyncio.create_task(review.review(request))
        assert await asyncio.to_thread(entered.wait, 5)
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert port.admissions == []
        assert closed == []
        await review.close()
        assert closed == [True]
    finally:
        release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_close_retains_profile_until_parent_build_quiesces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_realtime.companion import review as review_module

    monkeypatch.setattr(review_module, "_CLOSE_JOIN_SECONDS", 0.05)
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    entered = threading.Event()
    release = threading.Event()
    closed: list[bool] = []
    parent = type("Parent", (), {"close": lambda self: closed.append(True)})()

    def factory(session_id: str) -> Any:
        entered.set()
        if not release.wait(5):
            raise RuntimeError("parent construction did not release")
        return parent

    review = VoiceReviewCoordinator(archive, store, port, factory)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        pending = asyncio.create_task(
            review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        )
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert closed == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert port.admissions == []
        await review.close()
        assert closed == [True]
    finally:
        release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_cancelled_close_retains_constructing_parent(tmp_path: Path) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    entered = threading.Event()
    release = threading.Event()
    closed: list[bool] = []
    parent = type("Parent", (), {"close": lambda self: closed.append(True)})()

    def factory(session_id: str) -> Any:
        entered.set()
        if not release.wait(5):
            raise RuntimeError("parent construction did not release")
        return parent

    review = VoiceReviewCoordinator(archive, store, port, factory)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        pending = asyncio.create_task(
            review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        )
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(review.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(ReviewQuiescenceError):
            await closing
        assert closed == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await review.close()
        assert closed == [True]
    finally:
        release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["settings", "verify", "admit"])
async def test_close_owns_inflight_review_request_at_every_native_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from hermes_realtime.companion import review as review_module

    monkeypatch.setattr(review_module, "_CLOSE_JOIN_SECONDS", 0.05)
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    closed: list[bool] = []
    parent = type("Parent", (), {"close": lambda self: closed.append(True)})()
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    await review.start()
    await archive.open("conv")
    await archive.archive("conv", _batch())
    if stage == "settings":
        port.block_settings = True
        entered, release = port.settings_entered, port.settings_release
    elif stage == "verify":
        port.block_verify = True
        entered, release = port.verify_entered, port.verify_release
    else:
        port.block_admit = True
        entered, release = port.admit_entered, port.admit_release
    pending = asyncio.create_task(
        review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if stage != "verify":
            pending.cancel()  # The bridge has stopped waiting for this request.
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert closed == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert port.finish_calls == (1 if stage == "admit" else 0)
        await review.close()
        assert closed == ([] if stage == "settings" else [True])
    finally:
        release.set()
        pending.cancel()
        with contextlib.suppress(BaseException):
            await pending
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_cancelled_close_retains_inflight_admission(tmp_path: Path) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.block_admit = True
    closed: list[bool] = []
    parent = type("Parent", (), {"close": lambda self: closed.append(True)})()
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    await review.start()
    await archive.open("conv")
    await archive.archive("conv", _batch())
    pending = asyncio.create_task(
        review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
    )
    try:
        assert await asyncio.to_thread(port.admit_entered.wait, 5)
        closing = asyncio.create_task(review.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(ReviewQuiescenceError):
            await closing
        assert closed == []
        port.admit_release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert port.finish_calls == 1
        await review.close()
        assert closed == [True]
    finally:
        port.admit_release.set()
        pending.cancel()
        with contextlib.suppress(BaseException):
            await pending
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_failed_parent_close_is_retried(tmp_path: Path) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    attempts: list[bool] = []
    parent = type("Parent", (), {})()

    def close_parent(value: Any) -> None:
        assert value is parent
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("transient close failure")

    port.close_parent = close_parent  # type: ignore[method-assign]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        port.release.set()
        assert await review.join("conv", 5)
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert attempts == [True]
        await review.close()
        assert attempts == [True, True]
    finally:
        port.release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_live_parent_close_is_not_started_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_realtime.companion import review as review_module

    monkeypatch.setattr(review_module, "_CLOSE_JOIN_SECONDS", 0.05)
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    entered = threading.Event()
    release = threading.Event()
    attempts: list[bool] = []
    parent = type("Parent", (), {})()

    def close_parent(value: Any) -> None:
        assert value is parent
        attempts.append(True)
        entered.set()
        if not release.wait(5):
            raise RuntimeError("parent close did not release")

    port.close_parent = close_parent  # type: ignore[method-assign]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        port.release.set()
        assert await review.join("conv", 5)
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert entered.is_set()
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert attempts == [True]
        release.set()
        await review.close()
        assert attempts == [True]
    finally:
        port.release.set()
        release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_cancel_failure_retains_live_review_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_realtime.companion import review as review_module

    monkeypatch.setattr(review_module, "_CLOSE_JOIN_SECONDS", 0.05)
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.cancel_error = True
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: type("Parent", (), {})())
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        admitted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert review.active("conv") == admitted.review_id
        assert archive.ready("conv")
        port.release.set()
        assert await review.join("conv", 5)
        await review.close()
    finally:
        port.release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_completed_failed_cancel_helper_is_retried_while_review_runs(
    tmp_path: Path,
) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.cancel_error = True
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: type("Parent", (), {})())
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        admitted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        assert not await review.cancel_and_join("conv", 0.05)
        assert port.cancel_calls == 1
        assert review.active("conv") == admitted.review_id
        port.cancel_error = False
        assert not await review.cancel_and_join("conv", 0.05)
        assert port.cancel_calls == 2
        assert review.active("conv") == admitted.review_id
    finally:
        port.release.set()
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_native_cancel_helper_retains_parent_after_worker_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_realtime.companion import review as review_module

    monkeypatch.setattr(review_module, "_CLOSE_JOIN_SECONDS", 0.05)
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.block_cancel = True
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    closed: list[bool] = []
    parent = type("Parent", (), {"close": lambda self: closed.append(True)})()
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        admitted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        port.release.set()
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert port.cancel_entered.is_set()
        assert review.active("conv") == admitted.review_id
        assert closed == []
        with pytest.raises(ReviewQuiescenceError):
            await review.close()
        assert review.active("conv") == admitted.review_id
        assert port.cancel_calls == 1
        port.cancel_release.set()
        await review.close()
        assert closed == [True]
    finally:
        port.release.set()
        port.cancel_release.set()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_stale_watchdog_cannot_cancel_successor_review(tmp_path: Path) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: type("Parent", (), {})())
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        first = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        port.release.set()
        assert await review.join("conv", 5)
        port.started = asyncio.Event()
        port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
        port.release = asyncio.Event()
        second = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, True))
        await asyncio.wait_for(port.started.wait(), 5)
        await review._expire("conv", first.review_id)
        assert review.active("conv") == second.review_id
        assert review.outcome("conv", second.review_id) == "accepted"
    finally:
        port.release.set()
        await review.close()
        await archive.close()
        store.close()


def test_restart_review_scan_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hermes_realtime.companion import store as store_module

    store = CompanionStore(tmp_path / "companion.db")
    try:
        for index in range(2):
            store._connection.execute(
                "INSERT INTO voice_archive "
                "(conversation_id, session_id, pending_count, pending_chain) "
                "VALUES (?, ?, 0, ?)",
                (f"conv_{index}", f"session_{index}", "0" * 64),
            )
        monkeypatch.setattr(store_module, "MAX_BOUND_CONVERSATIONS", 1)
        with pytest.raises(RuntimeError, match="bound exceeded"):
            store.recover_reviews()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_review_is_acknowledged_only_after_owned_thread_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    parent = type("Parent", (), {})()
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        admitted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        assert admitted.status == "accepted"
        assert admitted.review_id
        await asyncio.wait_for(port.started.wait(), 5)
        port.release.set()
        assert await review.join("conv", 5) is True
        assert review.outcome("conv", admitted.review_id) == "finished"
        assert capsys.readouterr().out.splitlines() == [
            '[voice-review] {"outcome":"finished","version":1}'
        ]
    finally:
        port.release.set()
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", ["failed", "unknown"])
async def test_late_completion_cannot_promote_a_retained_admission_failure(
    tmp_path: Path, retained: str
) -> None:
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, FakeHermes())
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: type("Parent", (), {})())
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        admitted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        await asyncio.wait_for(port.started.wait(), 5)
        store.finish_review("conv", admitted.review_id, retained)
        port.release.set()
        assert await review.join("conv", 5)
        assert review.outcome("conv", admitted.review_id) == retained
    finally:
        port.release.set()
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_mismatch_quarantines_before_any_review(tmp_path: Path) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    port.mismatch = True
    review = VoiceReviewCoordinator(archive, store, port, lambda _: object())
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        with pytest.raises(ArchiveRefusal, match="mismatch"):
            await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        assert store.read("conv").quarantine == "mismatch"  # type: ignore[union-attr]
        assert port.admissions == []
    finally:
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_busy_keeps_coverage_and_replay_does_not_spawn_twice(tmp_path: Path) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    review = VoiceReviewCoordinator(archive, store, port, lambda _: type("Parent", (), {})())
    request = ReviewRequest("conv", 0, 0, 0, True, True, False)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        port.busy = True
        with pytest.raises(ArchiveRefusal, match="busy"):
            await review.review(request)
        assert store.find_review(request) is None
        port.busy = False
        accepted = await review.review(request)
        assert accepted.status == "accepted"
        replay = await review.review(request)
        assert replay.review_id == accepted.review_id
        assert port.admissions == [(0, 0, False)]
        port.release.set()
        assert await review.join("conv", 5)
    finally:
        port.release.set()
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_recovered_reservation_is_unknown_and_never_claimed_started(tmp_path: Path) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    await archive.open("conv")
    await archive.archive("conv", _batch())
    request = ReviewRequest("conv", 0, 0, 0, True, True, False)
    review_id = "vr_" + "1" * 32
    store.reserve_review(request, review_id)
    await archive.close()
    store.close()

    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    review = VoiceReviewCoordinator(archive, store, port, lambda _: object())
    try:
        assert review.outcome("conv", review_id) == "unknown"
        await archive.open("conv")
        with pytest.raises(ArchiveRefusal, match="unknown"):
            await review.review(request)
        assert port.admissions == []
    finally:
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_native_failure_and_summary_are_suppressed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    port.started._loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
    port.failed = True
    port.logged = True
    parent = type("Parent", (), {})()
    printed: list[str] = []
    parent._safe_print = printed.append
    review = VoiceReviewCoordinator(archive, store, port, lambda _: parent)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        accepted = await review.review(ReviewRequest("conv", 0, 0, 0, True, True, False))
        port.release.set()
        assert await review.join("conv", 5)
        assert review.outcome("conv", accepted.review_id) == "failed"
        assert printed == []
        assert "PRIVATE REVIEW TEXT" not in caplog.text
    finally:
        port.release.set()
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_cancelled_admission_releases_a_late_token_before_any_ack(tmp_path: Path) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    port.block_admit = True
    review = VoiceReviewCoordinator(archive, store, port, lambda _: object())
    request = ReviewRequest("conv", 0, 0, 0, True, True, False)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        task = asyncio.create_task(review.review(request))
        assert await asyncio.to_thread(port.admit_entered.wait, 5)
        task.cancel()
        port.admit_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert port.finish_calls == 1
        assert store.find_review(request) is None
    finally:
        port.admit_release.set()
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_native_commit_failure_returns_prepared_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        record = store.read("conv")
        assert record is not None
        token = object()
        finished: list[object] = []

        class Conn:
            def execute(self, sql: str, params: Any) -> Any:
                assert "SELECT role, content" in sql
                return type("Rows", (), {"fetchall": lambda self: [
                    ("user", "review me", "voice:conv:0:0")
                ]})()

        def execute_write(callback: Any) -> Any:
            callback(Conn())
            raise OSError("commit failed after callback")

        def method(db: Any, name: str) -> Any:
            if name == "SessionDB._execute_write":
                return execute_write
            if name == "SessionDB._check_transcript_write_guards":
                return lambda *args, **kwargs: None
            raise AssertionError(name)

        def resolve(name: str) -> Any:
            if name == "agent.background_review.prepare_background_review_run":
                return lambda parent: token
            if name == "agent.background_review.finish_background_review_run":
                return lambda parent, run: finished.append(run)
            if name in {"SessionTurnLeaseLostError", "CompressionSessionClosedError"}:
                return RuntimeError
            raise AssertionError(name)

        monkeypatch.setattr(hermes_compat, "_method", method)
        monkeypatch.setattr(hermes_compat, "resolve", resolve)
        monkeypatch.setattr(
            hermes_compat, "_read_projection",
            lambda conn, session_id, cap: hermes.read_projection(session_id, cap),
        )
        port = object.__new__(hermes_compat.HermesArchivePort)
        port._db = object()
        with pytest.raises(OSError, match="commit failed"):
            port.admit(
                object(), record, ReviewRequest("conv", 0, 0, 0, True, True, False),
                4096, 30.0,
            )
        assert finished == [token]
    finally:
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_native_admission_checks_the_current_lease_before_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        record = store.read("conv")
        assert record is not None
        prepares: list[object] = []

        class LostLease(RuntimeError):
            pass

        class Rotated(RuntimeError):
            pass

        def guard(*args: Any, **kwargs: Any) -> None:
            raise LostLease

        def method(db: Any, name: str) -> Any:
            if name == "SessionDB._execute_write":
                return lambda callback: callback(object())
            if name == "SessionDB._check_transcript_write_guards":
                return guard
            raise AssertionError(name)

        def resolve(name: str) -> Any:
            if name == "agent.background_review.prepare_background_review_run":
                return lambda parent: prepares.append(parent)
            if name == "SessionTurnLeaseLostError":
                return LostLease
            if name == "CompressionSessionClosedError":
                return Rotated
            raise AssertionError(name)

        monkeypatch.setattr(hermes_compat, "_method", method)
        monkeypatch.setattr(hermes_compat, "resolve", resolve)
        port = object.__new__(hermes_compat.HermesArchivePort)
        port._db = object()
        with pytest.raises(ArchiveRefusal, match="lease_lost"):
            port.admit(
                object(), record, ReviewRequest("conv", 0, 0, 0, True, True, False),
                4096, 30.0,
            )
        assert prepares == []
    finally:
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_native_admission_compares_the_full_chain_before_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        record = store.read("conv")
        assert record is not None
        prepares: list[object] = []

        class Changed:
            def fingerprint(self) -> object:
                return object()

        def method(db: Any, name: str) -> Any:
            if name == "SessionDB._execute_write":
                return lambda callback: callback(object())
            if name == "SessionDB._check_transcript_write_guards":
                return lambda *args, **kwargs: None
            raise AssertionError(name)

        def resolve(name: str) -> Any:
            if name == "agent.background_review.prepare_background_review_run":
                return lambda parent: prepares.append(parent)
            if name in {"SessionTurnLeaseLostError", "CompressionSessionClosedError"}:
                return RuntimeError
            raise AssertionError(name)

        monkeypatch.setattr(hermes_compat, "_method", method)
        monkeypatch.setattr(hermes_compat, "resolve", resolve)
        monkeypatch.setattr(hermes_compat, "_read_projection", lambda *args: Changed())
        port = object.__new__(hermes_compat.HermesArchivePort)
        port._db = object()
        with pytest.raises(ArchiveRefusal, match="mismatch"):
            port.admit(
                object(), record, ReviewRequest("conv", 0, 0, 0, True, True, False),
                4096, 30.0,
            )
        assert prepares == []
    finally:
        await archive.close()
        store.close()


@pytest.mark.parametrize(
    "fields",
    [
        ("conv", True, 0, True, True, False),
        ("conv", 0, -1, True, True, False),
        ("conv", 0, 0, 0, True, False),
        ("conv", 0, 0, True, False, False),
        ("conv", 0, 0, True, True, 0),
    ],
)
def test_review_request_refuses_coercion_and_bad_bounds(fields: tuple[Any, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReviewRequest(fields[0], fields[1], fields[2], 0, fields[3], fields[4], fields[5])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "category"),
    [("disabled", "disabled"), ("extras", "configuration"),
     ("interval", "configuration"), ("window", "window")],
)
async def test_spawn_policy_refuses_before_reserving_or_starting(
    tmp_path: Path, change: str, category: str
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    port = FakeReviewPort()
    review = VoiceReviewCoordinator(archive, store, port, lambda _: type("Parent", (), {})())
    request = ReviewRequest("conv", 0, 0, 0, True, True, False)
    try:
        await review.start()
        await archive.open("conv")
        await archive.archive("conv", _batch())
        if change == "disabled":
            port.enabled = False
        elif change == "extras":
            port.extra_tools = ["terminal"]
        elif change == "interval":
            port.interval = 11
        else:
            port.snapshot_rows = 25
        with pytest.raises(ArchiveRefusal, match=category):
            await review.review(request)
        assert store.find_review(request) is None
        assert review.active("conv") is None
    finally:
        await review.close()
        await archive.close()
        store.close()


@pytest.mark.parametrize(
    ("fence", "category"),
    [("quarantine", "quarantined"), ("tombstone", "tombstoned"),
     ("pending", "pending")],
)
@pytest.mark.asyncio
async def test_ledger_reservation_rechecks_fences_in_its_write_transaction(
    tmp_path: Path, fence: str, category: str
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        record = store.read("conv")
        assert record is not None and record.committed is not None
        request = ReviewRequest("conv", 0, 0, 0, True, True, False)
        if fence == "quarantine":
            store.quarantine("conv", "mismatch")
        elif fence == "tombstone":
            store._connection.execute(
                "UPDATE voice_archive SET tombstone = 0 WHERE conversation_id = 'conv'"
            )
        else:
            store.begin_pending("conv", record.committed, record.committed)
        with pytest.raises(ArchiveRefusal, match=category):
            store.reserve_review(request, "vr_" + "1" * 32)
        assert store.find_review(request) is None
    finally:
        await archive.close()
        store.close()


@pytest.mark.parametrize(
    "raw",
    [
        '{"0:0:0:0":{"id":"vr_11111111111111111111111111111111","outcome":"accepted"},'
        '"0:0:0:0":{"id":"vr_22222222222222222222222222222222","outcome":"accepted"}}',
        '{"x":{"id":"vr_11111111111111111111111111111111","outcome":"accepted"}}',
        '{"0:0:0:0":{"id":"vr_11111111111111111111111111111111","outcome":"accepted"},'
        '"0:1:1:0":{"id":"vr_11111111111111111111111111111111","outcome":"finished"}}',
    ],
)
@pytest.mark.asyncio
async def test_corrupt_ledger_is_refused_before_admission(tmp_path: Path, raw: str) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        store._connection.execute(
            "UPDATE voice_archive SET review_ledger = ? WHERE conversation_id = 'conv'",
            (raw,),
        )
        with pytest.raises(RuntimeError, match="review ledger is invalid"):
            store.find_review(ReviewRequest("conv", 0, 0, 0, True, True, False))
    finally:
        await archive.close()
        store.close()


@pytest.mark.asyncio
async def test_ledger_accept_rechecks_tombstone_but_outcome_can_settle_after_it(
    tmp_path: Path,
) -> None:
    hermes = FakeHermes()
    store = CompanionStore(tmp_path / "companion.db")
    archive = VoiceArchive(store, hermes)
    try:
        await archive.open("conv")
        await archive.archive("conv", _batch())
        request = ReviewRequest("conv", 0, 0, 0, True, True, False)
        review_id = "vr_" + "1" * 32
        store.reserve_review(request, review_id)
        store._connection.execute(
            "UPDATE voice_archive SET tombstone = 0 WHERE conversation_id = 'conv'"
        )
        with pytest.raises(ArchiveRefusal, match="tombstoned"):
            store.accept_review(request, review_id)
        assert store.review_outcome("conv", review_id) == "reserved"
        store.finish_review("conv", review_id, "unknown")
        assert store.review_outcome("conv", review_id) == "unknown"
    finally:
        await archive.close()
        store.close()
