"""Close observes terminal consent and retains one published lifecycle owner."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from hermes_realtime.evidence import models as m
from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
from hermes_realtime.production_observation import CloseResultV1, CloseStageV1
from tests.evidence.test_task11b_real_chain import _activate, _create_epoch


def _runtime(path: Path) -> HostEvidenceRuntimeV1:
    return HostEvidenceRuntimeV1(
        database=path / "capture-v1.sqlite3", owner_generation=42, retention_hours=24
    )


def _require_closed(runtime, admission) -> None:
    assert runtime._closed
    assert runtime._writer is runtime._transport is runtime._queue is None
    assert runtime._consent_settlement_operation is None
    assert runtime.operation_scheduler.active_count == 0
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == diagnostics.queue_canonical_bytes == 0
    assert diagnostics.active_lease_count == 0


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("closed_loop", [False, True])
def test_terminal_settlement_keeps_its_outcome_after_its_loop_stops(
    tmp_path, outcome, closed_loop
) -> None:
    runtime = _runtime(tmp_path)
    loop = asyncio.new_event_loop()
    failure = RuntimeError("synthetic consent settlement failure")

    async def settle():
        if outcome == "failure":
            raise failure
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return m.ConsentDisposition.CONSENT_ACTIVATED

    async def prepare():
        task = asyncio.create_task(settle())
        await asyncio.gather(task, return_exceptions=True)
        return task

    task = loop.run_until_complete(prepare())
    assert task.done() and task.get_loop() is loop
    runtime._consent_settlement_operation = task
    if closed_loop:
        loop.close()
    try:
        result = asyncio.run(runtime._cancel_and_observe_consent_settlement())
        assert result == {
            "success": (CloseResultV1.SUCCEEDED, None),
            "failure": (CloseResultV1.FAILED, failure),
            "cancelled": (CloseResultV1.CANCELLED, None),
        }[outcome]
    finally:
        if not loop.is_closed():
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()


@pytest.mark.asyncio
async def test_pending_settlement_is_still_cancelled_and_observed(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    entered, settled = asyncio.Event(), asyncio.Event()

    async def pending():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    operation = asyncio.create_task(pending())
    runtime._consent_settlement_operation = operation
    await entered.wait()
    assert await runtime._cancel_and_observe_consent_settlement() == (
        CloseResultV1.CANCELLED,
        None,
    )
    assert operation.cancelled() and settled.is_set()


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows evidence storage")
def test_public_close_observes_completed_real_consent_from_a_stopped_loop(tmp_path) -> None:
    from hermes_realtime.client import BrowserEventProjection

    runtime = _runtime(tmp_path)
    owner = asyncio.new_event_loop()
    release = threading.Event()

    async def activate_late():
        command, projection = _create_epoch(), BrowserEventProjection()
        authority = runtime.reserve_consent_authority(
            command,
            projection.reserve_capture_status(),
            projection.validate_capture_status_reservation,
        )
        delegate = runtime.create_sqlite_transport()

        class Delayed:
            def __getattr__(self, name):
                return getattr(delegate, name)

            def create_epoch(self, command):
                assert release.wait(5), "test did not release real consent creation"
                return delegate.create_epoch(command)

        try:
            assert await runtime.activate_consent(
                authority,
                transport=Delayed(),
                binding_is_current=lambda candidate: candidate is command,
                timeout_seconds=0.01,
            ) is m.ConsentDisposition.CONTROL_TIMED_OUT
            settlement = runtime.claim_consent_settlement_task(authority)
            release.set()
            assert await asyncio.wait_for(asyncio.shield(settlement), 5) is (
                m.ConsentDisposition.CONSENT_ACTIVATED
            )
            assert settlement.done() and settlement.get_loop() is owner
        finally:
            release.set()

    try:
        owner.run_until_complete(activate_late())
        assert not owner.is_running() and not owner.is_closed()
        assert runtime._retention_task is None
        admission = runtime.evidence_admission
        assert admission is not None
        asyncio.run(runtime.close())
        _require_closed(runtime, admission)
    finally:
        release.set()
        # Keep failure cleanup on the original loop; the public close above is
        # the behavior under test, including when its regression is RED.
        if not runtime._closed:
            owner.run_until_complete(runtime._close_owned())
        owner.close()


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows evidence storage")
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_published_close_join_retains_one_real_drain(tmp_path, cancel_first) -> None:
    runtime = _runtime(tmp_path)
    delegate = runtime.create_sqlite_transport()
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Blocking:
        def __getattr__(self, name):
            return getattr(delegate, name)

        def drain_and_close(self, command):
            calls.append("drain")
            entered.set()
            assert release.wait(5), "test did not release real drain"
            return delegate.drain_and_close(command)

        def close(self):
            calls.append("close")
            return delegate.close()

    await _activate(runtime, _create_epoch(), transport=Blocking())
    admission = runtime.evidence_admission
    assert admission is not None
    callers = []
    try:
        first = asyncio.create_task(runtime.close())
        callers.append(first)
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        operation = runtime._close_operation
        second = asyncio.create_task(runtime.close())
        callers.append(second)
        await asyncio.sleep(0)
        assert runtime._close_operation is operation
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not operation.cancelled()
        release.set()
        await second
        if not cancel_first:
            await first
        _require_closed(runtime, admission)
        assert calls == ["drain", "close"]
    finally:
        release.set()
        await asyncio.gather(*callers, return_exceptions=True)
        if not runtime._closed:
            await runtime._close_owned()


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows evidence storage")
@pytest.mark.asyncio
async def test_published_close_retries_an_early_owner_failure_once(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    await _activate(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    observe = runtime._cancel_and_observe_consent_settlement
    calls = 0

    async def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic early close failure")
        return await observe()

    monkeypatch.setattr(runtime, "_cancel_and_observe_consent_settlement", fail_once)
    try:
        with pytest.raises(RuntimeError, match="synthetic early close failure"):
            await runtime.close()
        first = runtime._close_operation
        assert first.done() and not runtime._closed
        await runtime.close()
        assert runtime._close_operation is not first and calls == 2
        _require_closed(runtime, admission)
        assert [
            item.result
            for item in runtime.production_observations.records()
            if getattr(item, "stage", None) is CloseStageV1.EVIDENCE_RUNTIME
        ] == [CloseResultV1.FAILED, CloseResultV1.SUCCEEDED]
    finally:
        monkeypatch.setattr(runtime, "_cancel_and_observe_consent_settlement", observe)
        if not runtime._closed:
            await runtime._close_owned()
