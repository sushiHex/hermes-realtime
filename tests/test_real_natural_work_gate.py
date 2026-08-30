from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from hermes_realtime.conversation import WorkCancelResult, WorkStartResult
from hermes_realtime.providers.codex_app_server import _CodexTransportClosed

_GATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "real_natural_work_gate.py"
sys.path.insert(0, str(_GATE_PATH.parent))
_GATE_SPEC = importlib.util.spec_from_file_location("real_natural_work_gate", _GATE_PATH)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
sys.modules[_GATE_SPEC.name] = _GATE
_GATE_SPEC.loader.exec_module(_GATE)

LatencySample = _GATE.LatencySample
TaskAcknowledgementSample = _GATE.TaskAcknowledgementSample
ObservedTransport = _GATE.ObservedTransport
RecordingWorkHandler = _GATE.RecordingWorkHandler
TurnTimingProbe = _GATE.TurnTimingProbe
WorkActivityObserver = _GATE.WorkActivityObserver
_NoDispatchHandler = _GATE._NoDispatchHandler
_ObservedTaskSession = _GATE._ObservedTaskSession
_codex_metadata = _GATE._codex_metadata
_failure_code = _GATE._failure_code
_finish_with_bounded_cleanup = _GATE._finish_with_bounded_cleanup
_validate_completion_task_evidence = _GATE._validate_completion_task_evidence
_timed_turn = _GATE._timed_turn
build_latency_report = _GATE.build_latency_report
paired_orders = _GATE.paired_orders
validate_configuration = _GATE.validate_configuration
validate_public_report = _GATE.validate_public_report


class _FakeTransport:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = iter(messages)
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send(self, message: Mapping[str, object]) -> None:
        self.sent.append(dict(message))

    async def receive(self) -> Mapping[str, object]:
        return next(self.messages)

    async def close(self) -> None:
        self.closed = True


class _FakeWorkHandler:
    max_objective_chars = 1024
    can_cancel_work = False

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        assert objective == "private objective text"
        assert invocation_id == "tool_0123456789abcdef0123456789abcdef"
        return WorkStartResult(accepted=True, state="active", task_id="task_public")

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        assert invocation_id == "tool_abcdef0123456789abcdef0123456789"
        return WorkCancelResult(accepted=True, state="cancelling", task_id="task_public")


class _FailingCloseSession:
    async def dispatch(self, request: object) -> object:
        return request

    async def cancel(self, request: object) -> object:
        return request

    async def next_update(self) -> None:
        return None

    async def close(self) -> None:
        raise RuntimeError("session close failed")


class _RejectedWorkHandler:
    max_objective_chars = 1024
    can_cancel_work = False

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        del objective, invocation_id
        return WorkStartResult(accepted=False, state="rejected", reason="not accepted")

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        del invocation_id
        return WorkCancelResult(accepted=False, state="rejected", reason="not accepted")


class _RaisingWorkHandler:
    max_objective_chars = 1024
    can_cancel_work = False

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        del objective, invocation_id
        raise RuntimeError("delegate failed")

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        del invocation_id
        raise RuntimeError("delegate failed")


def _sample(
    *,
    transcript_to_thread_ms: float,
    thread_to_delta_ms: float,
    delta_to_speakable_ms: float,
) -> LatencySample:
    return LatencySample(
        transcript_to_thread_ms=transcript_to_thread_ms,
        thread_to_first_delta_ms=thread_to_delta_ms,
        first_delta_to_speakable_ms=delta_to_speakable_ms,
    )


def test_task_acknowledgement_sample_separates_model_and_handler_time() -> None:
    sample = TaskAcknowledgementSample(
        request_to_accept_ms=9250.0,
        handler_to_accept_ms=25.0,
    )

    assert sample.decision_and_tool_call_ms == pytest.approx(9225.0)


def test_paired_orders_interleave_balanced_ab_ba_measurements() -> None:
    orders = paired_orders(30)

    assert len(orders) == 30
    assert orders[:4] == (
        ("absent", "present"),
        ("present", "absent"),
        ("absent", "present"),
        ("present", "absent"),
    )
    assert sum(order[0] == "absent" for order in orders) == 15
    assert sum(order[0] == "present" for order in orders) == 15


@pytest.mark.parametrize("pairs", [True, 0, 29, 1001])
def test_paired_orders_reject_invalid_release_sample_counts(pairs: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        paired_orders(pairs)  # type: ignore[arg-type]


def test_latency_report_accepts_within_paired_p95_and_ack_budgets() -> None:
    absent = tuple(
        _sample(
            transcript_to_thread_ms=10 + index,
            thread_to_delta_ms=100 + index,
            delta_to_speakable_ms=5 + index,
        )
        for index in range(30)
    )
    present = tuple(
        _sample(
            transcript_to_thread_ms=12 + index,
            thread_to_delta_ms=105 + index,
            delta_to_speakable_ms=7 + index,
        )
        for index in range(30)
    )

    report = build_latency_report(
        absent=absent,
        present=present,
        warmups_per_arm=5,
        task_acknowledgement_ms=(950.0, 1100.0),
        task_handler_acknowledgement_ms=(50.0, 60.0),
        task_ack_budget_ms=5000.0,
    )

    assert report["sample_pairs"] == 30
    assert report["warmups_per_arm"] == 5
    assert report["acceptance"] == {
        "present_unused_p95_within_budget": True,
        "task_acknowledgement_within_budget": True,
        "passed": True,
    }
    comparison = report["foreground_p95_comparison_ms"]
    paired = report["paired_foreground_delta_ms"]
    assert isinstance(comparison, dict)
    assert isinstance(paired, dict)
    assert comparison["regression_budget_ms"] == 100.0
    assert comparison["p95_delta_ms"] == pytest.approx(9.0)
    assert "absolute_distribution" in paired
    acknowledgements = report["task_acknowledgement"]
    assert acknowledgements == {
        "samples_ms": [950.0, 1100.0],
        "count": 2,
        "max_ms": 1100.0,
        "budget_ms": 5000.0,
    }
    assert "p50" not in acknowledgements
    assert "p95" not in acknowledgements
    assert report["task_acknowledgement_breakdown"] == {
        "decision_and_tool_call_ms": {
            "samples_ms": [900.0, 1040.0],
            "max_ms": 1040.0,
        },
        "handler_acceptance_ms": {
            "samples_ms": [50.0, 60.0],
            "max_ms": 60.0,
        },
    }


def test_latency_report_fails_when_present_unused_p95_exceeds_budget() -> None:
    absent = tuple(
        _sample(transcript_to_thread_ms=10, thread_to_delta_ms=90, delta_to_speakable_ms=0)
        for _ in range(30)
    )
    present = tuple(
        _sample(transcript_to_thread_ms=10, thread_to_delta_ms=191, delta_to_speakable_ms=0)
        for _ in range(30)
    )

    report = build_latency_report(
        absent=absent,
        present=present,
        warmups_per_arm=5,
        task_acknowledgement_ms=(1000.0,),
        task_handler_acknowledgement_ms=(40.0,),
        task_ack_budget_ms=5000.0,
    )

    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["present_unused_p95_within_budget"] is False
    assert acceptance["passed"] is False


def test_latency_report_fails_when_task_acknowledgement_exceeds_budget() -> None:
    samples = tuple(
        _sample(transcript_to_thread_ms=10, thread_to_delta_ms=90, delta_to_speakable_ms=0)
        for _ in range(30)
    )

    report = build_latency_report(
        absent=samples,
        present=samples,
        warmups_per_arm=5,
        task_acknowledgement_ms=(5100.0,),
        task_handler_acknowledgement_ms=(45.0,),
        task_ack_budget_ms=5000.0,
    )

    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["task_acknowledgement_within_budget"] is False
    assert acceptance["passed"] is False


def test_public_report_rejects_private_authority_tokens_recursively() -> None:
    validate_public_report({"gate": "passed", "counts": {"runs": 2}})

    with pytest.raises(ValueError, match="private authority"):
        validate_public_report({"gate": "failed", "nested": [{"detail": "run_0123456789abcdef"}]})
    with pytest.raises(ValueError, match="private authority"):
        validate_public_report({"gate": "failed", "detail": "deleg_private_token"})


@pytest.mark.asyncio
async def test_observed_transport_records_exact_foreground_timing_boundaries() -> None:
    times = iter((0.120, 0.160, 0.170))
    probe = TurnTimingProbe(clock=lambda: next(times))
    probe.begin(transcript_at=0.100)
    raw = _FakeTransport(
        [
            {"id": 7, "result": {"thread": {"id": "thread_public"}}},
            {"id": 8, "result": {"turn": {"id": "turn_public"}}},
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread_public",
                    "turnId": "turn_public",
                    "delta": "Ready.",
                },
            },
        ]
    )
    transport = ObservedTransport(raw, probe)

    await transport.send({"id": 7, "method": "thread/start", "params": {}})
    assert await transport.receive() == {
        "id": 7,
        "result": {"thread": {"id": "thread_public"}},
    }
    await transport.send(
        {
            "id": 8,
            "method": "turn/start",
            "params": {"threadId": "thread_public"},
        }
    )
    await transport.receive()
    await transport.receive()
    probe.mark_first_speakable()

    sample = probe.sample()
    assert sample.transcript_to_thread_ms == pytest.approx(20.0)
    assert sample.thread_to_first_delta_ms == pytest.approx(40.0)
    assert sample.first_delta_to_speakable_ms == pytest.approx(10.0)
    await transport.close()
    assert cast(_FakeTransport, raw).closed is True


def _established_probe() -> TurnTimingProbe:
    probe = TurnTimingProbe(clock=lambda: 1.0)
    probe.begin(transcript_at=0.5)
    probe.observe_send({"id": 7, "method": "thread/start", "params": {}})
    probe.observe_receive({"id": 7, "result": {"thread": {"id": "thread_current"}}})
    probe.observe_send(
        {
            "id": 8,
            "method": "turn/start",
            "params": {"threadId": "thread_current"},
        }
    )
    probe.observe_receive({"id": 8, "result": {"turn": {"id": "turn_current"}}})
    return probe


@pytest.mark.parametrize(
    "params",
    (
        {"threadId": "thread_stale", "turnId": "turn_current", "delta": "stale"},
        {"threadId": "thread_current", "turnId": "turn_stale", "delta": "stale"},
        {"threadId": "thread_current", "delta": "malformed"},
    ),
)
def test_turn_timing_probe_rejects_foreign_or_malformed_deltas(
    params: dict[str, object],
) -> None:
    probe = _established_probe()

    with pytest.raises(RuntimeError):
        probe.observe_receive({"method": "item/agentMessage/delta", "params": params})


def test_turn_timing_probe_rejects_delta_racing_turn_start_response() -> None:
    probe = TurnTimingProbe(clock=lambda: 1.0)
    probe.begin(transcript_at=0.5)
    probe.observe_send({"id": 7, "method": "thread/start", "params": {}})
    probe.observe_receive({"id": 7, "result": {"thread": {"id": "thread_current"}}})
    probe.observe_send(
        {
            "id": 8,
            "method": "turn/start",
            "params": {"threadId": "thread_current"},
        }
    )

    with pytest.raises(RuntimeError, match="preceded"):
        probe.observe_receive(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread_current",
                    "turnId": "turn_current",
                    "delta": "raced",
                },
            }
        )


def test_turn_timing_probe_fails_closed_when_a_boundary_is_missing() -> None:
    probe = TurnTimingProbe(clock=lambda: 1.0)
    probe.begin(transcript_at=0.5)

    with pytest.raises(RuntimeError, match="incomplete"):
        probe.sample()


def test_completion_task_evidence_allows_terminal_to_beat_active_snapshot() -> None:
    _validate_completion_task_evidence(
        accepted_task_id="task_gate1",
        active_task_ids=(),
        terminal_task_id="task_gate1",
    )


@pytest.mark.asyncio
async def test_observed_task_session_retains_close_failure_evidence() -> None:
    session = _ObservedTaskSession(_FailingCloseSession())

    with pytest.raises(RuntimeError, match="session close failed"):
        await session.close()

    assert session.close_started is True
    assert session.close_completed is True
    assert len(session.close_failures) == 1


@pytest.mark.asyncio
async def test_recording_work_handler_retains_only_counts_and_ack_latency() -> None:
    clock = iter((1.0, 1.25, 2.0, 2.1))
    handler = RecordingWorkHandler(_FakeWorkHandler(), clock=lambda: next(clock))

    start = await handler.start_work(
        objective="private objective text",
        invocation_id="tool_0123456789abcdef0123456789abcdef",
    )
    cancel = await handler.cancel_active_work(invocation_id="tool_abcdef0123456789abcdef0123456789")

    assert start.accepted is True
    assert cancel.accepted is True
    assert handler.max_objective_chars == 1024
    assert handler.start_attempts == 1
    assert handler.cancel_attempts == 1
    assert handler.accepted_start_calls == 1
    assert handler.accepted_cancel_calls == 1
    assert handler.last_cancel_was_exactly_accepted is True
    assert handler.start_acknowledgement_ms == pytest.approx((250.0,))
    assert handler.cancel_acknowledgement_ms == pytest.approx((100.0,))
    assert "private objective text" not in repr(handler)


@pytest.mark.asyncio
async def test_recording_work_handler_counts_attempts_before_failed_delegation() -> None:
    handler = RecordingWorkHandler(_RaisingWorkHandler(), clock=lambda: 1.0)

    with pytest.raises(RuntimeError, match="delegate failed"):
        await handler.start_work(objective="private", invocation_id="private-start")
    with pytest.raises(RuntimeError, match="delegate failed"):
        await handler.cancel_active_work(invocation_id="private-cancel")

    assert handler.start_attempts == 1
    assert handler.cancel_attempts == 1
    assert handler.accepted_start_calls == 0
    assert handler.accepted_cancel_calls == 0
    assert handler.start_acknowledgement_ms == ()
    assert handler.cancel_acknowledgement_ms == ()
    assert "private" not in repr(handler)


@pytest.mark.asyncio
async def test_recording_work_handler_does_not_acknowledge_rejected_results() -> None:
    handler = RecordingWorkHandler(_RejectedWorkHandler(), clock=lambda: 1.0)

    await handler.start_work(objective="private", invocation_id="private-start")
    await handler.cancel_active_work(invocation_id="private-cancel")

    assert handler.start_attempts == 1
    assert handler.cancel_attempts == 1
    assert handler.accepted_start_calls == 0
    assert handler.accepted_cancel_calls == 0
    assert handler.last_cancel_was_exactly_accepted is False
    assert handler.start_acknowledgement_ms == ()
    assert handler.cancel_acknowledgement_ms == ()


@pytest.mark.asyncio
async def test_present_unused_handler_counts_every_unexpected_invocation() -> None:
    handler = _NoDispatchHandler()

    with pytest.raises(RuntimeError, match="unexpectedly requested work"):
        await handler.start_work(objective="do not retain me", invocation_id="inv-1")
    with pytest.raises(RuntimeError, match="unexpectedly requested cancellation"):
        await handler.cancel_active_work(invocation_id="inv-2")

    assert handler.invocations == 2
    assert "do not retain me" not in repr(handler)


def test_present_unused_handler_satisfies_real_binding_contract() -> None:
    handler = _NoDispatchHandler()
    inference = _GATE.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: _FakeTransport([]),
    )

    inference.bind_work_tools(handler)
    assert handler.can_cancel_work is False


def test_transport_activity_observer_counts_malformed_pre_handler_attempts() -> None:
    observer = WorkActivityObserver()

    observer.observe_send(
        {
            "method": "thread/start",
            "params": {
                "dynamicTools": _GATE.CodexAppServerStreamingInference._dynamic_tools(
                    1024, include_cancel=False
                )
            },
        }
    )
    observer.observe_send({"method": "turn/start"})
    with pytest.raises(RuntimeError, match="private authority"):
        observer.observe_send({"id": 9, "result": {"run_id": "run_0123456789abcdef"}})
    observer.observe_receive({"id": "private", "method": "item/tool/call"})
    observer.observe_receive(
        {
            "method": "item/started",
            "params": {"item": {"type": "dynamicToolCall", "arguments": "private"}},
        }
    )
    with pytest.raises(RuntimeError, match="private authority"):
        observer.observe_receive(
            {
                "method": "item/tool/call",
                "params": {"arguments": {"objective": "run_0123456789abcdef"}},
            }
        )

    assert observer.request_boundaries == 1
    assert observer.lifecycle_boundaries == 1
    assert observer.total_boundaries == 2
    assert observer.thread_start_requests == 1
    assert observer.turn_start_requests == 1
    assert observer.thread_start_schemas == (("start_work",),)
    assert observer.thread_start_schema_hashes == (_GATE._idle_start_schema_sha256(),)
    assert observer.private_authority_messages == 2
    assert "private" not in repr(observer)
    observer.reset()
    assert observer.total_boundaries == 0
    assert observer.thread_start_requests == 0
    assert observer.turn_start_requests == 0
    assert observer.thread_start_schemas == ()
    assert observer.private_authority_messages == 0


def test_pinned_codex_metadata_matches_protocol_fixture() -> None:
    fixture = json.loads(_GATE._CODEX_PROTOCOL_FIXTURE.read_text(encoding="utf-8"))
    metadata = {
        "version": fixture["codexVersion"].removeprefix("codex-cli "),
        "binary_sha256": fixture["codexBinarySha256"],
    }

    _GATE._assert_pinned_codex(metadata)
    with pytest.raises(RuntimeError, match="pinned protocol fixture"):
        _GATE._assert_pinned_codex({**metadata, "binary_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="pinned protocol fixture"):
        _GATE._assert_pinned_codex({**metadata, "version": "0.0.0"})


@pytest.mark.asyncio
async def test_inference_cleanup_uses_one_deadline_and_forces_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTransport:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class HangingInference:
        def __init__(self) -> None:
            self._transport = FakeTransport()

        async def close(self) -> None:
            await asyncio.Event().wait()

    monkeypatch.setattr(_GATE, "_CLOSE_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(_GATE, "_CLOSE_WATCHDOG_GRACE_SECONDS", 0.2)
    inference = HangingInference()
    workspace = tmp_path / "arm"
    workspace.mkdir()
    arm = _GATE._InferenceArm(
        inference=inference,
        probe=SimpleNamespace(),
        activity=_GATE.WorkActivityObserver(),
        workspace=workspace,
    )
    started = asyncio.get_running_loop().time()

    with pytest.raises(_GATE._CleanupFailures):
        await _GATE._finish_with_bounded_cleanup(
            None,
            (arm.close_for_cleanup,),
            label="test cleanup",
            stop_after_failure=True,
        )

    assert asyncio.get_running_loop().time() - started < 0.8
    assert inference._transport.closed is True
    assert not workspace.exists()


@pytest.mark.parametrize(
    ("pairs", "warmups", "budget"),
    (
        (29, 5, 5000.0),
        (30, 4, 5000.0),
        (30, 101, 5000.0),
        (30, 5, 0.0),
        (30, 5, 60_001.0),
    ),
)
def test_configuration_rejects_invalid_release_gate_ranges_before_execution(
    pairs: int,
    warmups: int,
    budget: float,
) -> None:
    with pytest.raises(ValueError):
        validate_configuration(
            pairs=pairs,
            warmups_per_arm=warmups,
            task_ack_budget_ms=budget,
        )


@pytest.mark.asyncio
async def test_timed_turn_outer_deadline_bounds_a_wedged_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WedgedInference:
        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def stream(
            self,
            snapshot: object,
            *,
            turn_id: str,
        ) -> object:
            del snapshot, turn_id
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release.wait()
            if False:
                yield "unreachable"

        async def abort(self) -> None:
            self.release.set()

    inference = WedgedInference()
    arm = SimpleNamespace(
        inference=inference,
        probe=TurnTimingProbe(),
        activity=WorkActivityObserver(),
        abort=inference.abort,
    )
    context = _GATE.ConversationContextStore(max_item_chars=1024)

    monkeypatch.setattr(_GATE, "_TURN_CANCEL_DRAIN_SECONDS", 0.01)
    started_at = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await _timed_turn(
            arm=arm,
            context=context,
            text="bounded",
            turn_id="turn_bounded",
            timeout_seconds=0.01,
        )
    assert asyncio.get_running_loop().time() - started_at < 0.2
    assert not any(
        task.get_name() == "natural-gate-turn:turn_bounded"
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


@pytest.mark.asyncio
async def test_inference_close_drains_late_tool_activity_before_reader_cancel() -> None:
    closed = object()

    class LateActivityTransport:
        def __init__(self) -> None:
            self.messages: asyncio.Queue[object] = asyncio.Queue()

        async def send(self, message: Mapping[str, object]) -> None:
            if "id" in message:
                await self.messages.put({"id": message["id"], "result": {}})

        async def receive(self) -> Mapping[str, object]:
            message = await self.messages.get()
            if message is closed:
                raise _CodexTransportClosed("normal process EOF")
            return cast(Mapping[str, object], message)

        async def close(self) -> None:
            await self.messages.put({"id": 999, "result": {}})
            await self.messages.put({"id": "late", "method": "item/tool/call"})
            await self.messages.put(closed)

    raw = LateActivityTransport()
    probe = TurnTimingProbe()
    activity = WorkActivityObserver()
    observed = _GATE.ObservedTransport(raw, probe, activity)
    inference = _GATE.CodexAppServerStreamingInference(
        model="gpt-5.6-terra",
        effort="low",
        transport_factory=lambda: observed,
        request_timeout_seconds=1,
    )
    await inference._ensure_started()

    with pytest.raises(RuntimeError, match="server request shape is ambiguous"):
        await inference.close()

    assert activity.request_boundaries == 1


@pytest.mark.asyncio
async def test_cleanup_preserves_primary_and_attempts_every_dependency() -> None:
    calls: list[str] = []

    async def close_controller() -> None:
        calls.append("controller")
        raise ValueError("controller cleanup")

    async def close_session() -> None:
        calls.append("session")
        raise RuntimeError("session cleanup")

    primary = RuntimeError("primary")
    with pytest.raises(_GATE._PrimaryFailureGroup) as captured:
        await _finish_with_bounded_cleanup(
            primary,
            (close_controller, close_session),
            label="test cleanup",
            stop_after_failure=False,
        )

    assert calls == ["controller", "session"]
    assert len(captured.value.exceptions) == 3
    assert captured.value.exceptions[0] is primary
    assert isinstance(captured.value.exceptions[1], ValueError)
    assert isinstance(captured.value.exceptions[2], RuntimeError)
    assert _failure_code(captured.value) == "execution_failed"


@pytest.mark.asyncio
async def test_cleanup_deadlines_do_not_prevent_later_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def wedged_close() -> None:
        calls.append("wedged")
        await asyncio.Event().wait()

    async def later_close() -> None:
        calls.append("later")

    monkeypatch.setattr(_GATE, "_CLOSE_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(_GATE._CleanupFailures) as captured:
        await _finish_with_bounded_cleanup(
            None,
            (wedged_close, later_close),
            label="test cleanup",
            stop_after_failure=False,
        )

    assert calls == ["wedged", "later"]
    assert _failure_code(captured.value) == "cleanup_failed"


@pytest.mark.asyncio
async def test_dependent_cleanup_stops_after_owner_close_failure() -> None:
    calls: list[str] = []

    async def close_owner() -> None:
        calls.append("owner")
        raise RuntimeError("owner cleanup")

    async def close_dependency() -> None:
        calls.append("dependency")

    with pytest.raises(_GATE._CleanupFailures):
        await _finish_with_bounded_cleanup(
            None,
            (close_owner, close_dependency),
            label="dependent cleanup",
            stop_after_failure=True,
        )

    assert calls == ["owner"]


def test_codex_metadata_hashes_before_execution_in_fresh_minimal_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "codex-test.exe"
    executable.write_bytes(b"stable-binary")
    observed: dict[str, object] = {}

    def fake_version(
        selected: str,
        *,
        cwd: str,
        environment: Mapping[str, str],
    ) -> bytes:
        observed["selected"] = selected
        observed["cwd_exists"] = Path(cwd).is_dir()
        observed["cwd"] = cwd
        observed["environment"] = dict(environment)
        assert _GATE._codex_fingerprint(selected).sha256
        return b"codex-cli 1.2.3\n"

    monkeypatch.setenv("HERMES_PRIVATE_TEST_SECRET", "must-not-pass")
    monkeypatch.setattr(_GATE, "_run_bounded_version", fake_version)

    metadata = _codex_metadata(str(executable))

    assert metadata["version"] == "1.2.3"
    assert metadata["binary_sha256"] == _GATE.hashlib.sha256(b"stable-binary").hexdigest()
    assert observed["cwd_exists"] is True
    assert not Path(cast(str, observed["cwd"])).exists()
    environment = cast(dict[str, str], observed["environment"])
    assert "HERMES_PRIVATE_TEST_SECRET" not in environment
    assert set(environment) <= _GATE._VERSION_ENVIRONMENT_ALLOWLIST
    assert Path(cast(str, observed["selected"])) == executable.resolve()


def test_version_environment_canonicalizes_names_and_rejects_conflicts() -> None:
    assert _GATE._version_environment({"Path": "bin", "path": "bin", "PRIVATE": "secret"}) == {
        "PATH": "bin"
    }

    with pytest.raises(RuntimeError, match="case-conflicting"):
        _GATE._version_environment({"Path": "one", "PATH": "two"})


def test_codex_metadata_rejects_executable_replacement_during_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "codex-test.exe"
    executable.write_bytes(b"binary-one")

    def replacing_version(
        selected: str,
        *,
        cwd: str,
        environment: Mapping[str, str],
    ) -> bytes:
        del selected, cwd, environment
        executable.write_bytes(b"binary-two")
        return b"codex-cli 1.2.3\n"

    monkeypatch.setattr(_GATE, "_run_bounded_version", replacing_version)

    with pytest.raises(RuntimeError, match="changed"):
        _codex_metadata(str(executable))


def test_main_keeps_incidental_runtime_output_out_of_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    async def fake_gate(**kwargs: object) -> dict[str, object]:
        assert kwargs["pairs"] == 125
        assert kwargs["task_ack_budget_ms"] == 8000.0
        print("incidental adapter status")
        os.write(1, b"native adapter status\n")
        os.write(2, b"native adapter diagnostic\n")
        return {"gate": "passed", "boundary": {"passed": True}}

    monkeypatch.setattr(_GATE, "run_real_gate", fake_gate)
    monkeypatch.setattr(sys, "argv", ["real_natural_work_gate.py"])

    assert _GATE.main() == 0
    captured = capfd.readouterr()
    assert json.loads(captured.out) == {
        "boundary": {"passed": True},
        "discarded_python_output_bytes": len("incidental adapter status\n"),
        "gate": "passed",
    }
    assert captured.err == ""


def test_main_emits_bounded_failure_code_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    async def failing_gate(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise ValueError("secret diagnostic that must not be emitted")

    monkeypatch.setattr(_GATE, "run_real_gate", failing_gate)
    monkeypatch.setattr(sys, "argv", ["real_natural_work_gate.py"])

    assert _GATE.main() == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out) == {
        "discarded_python_output_bytes": 0,
        "error": {"code": "invalid_configuration"},
        "gate": "failed",
    }
    assert captured.err == "invalid_configuration\n"
    assert "secret diagnostic" not in captured.out + captured.err


def test_main_emits_static_phase_without_wrapped_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    async def failing_gate(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise _GATE._GatePhaseFailure(
            "latency",
            RuntimeError("private provider diagnostic that must not be emitted"),
        )

    monkeypatch.setattr(_GATE, "run_real_gate", failing_gate)
    monkeypatch.setattr(sys, "argv", ["real_natural_work_gate.py"])

    assert _GATE.main() == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out) == {
        "discarded_python_output_bytes": 0,
        "error": {"code": "execution_failed", "stage": "latency"},
        "gate": "failed",
    }
    assert captured.err == "execution_failed\n"
    assert "private provider diagnostic" not in captured.out + captured.err
