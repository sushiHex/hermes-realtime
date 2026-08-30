import asyncio
from datetime import UTC, datetime
from types import MethodType
from typing import Any

import pytest

from hermes_realtime.conversation import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationTaskController,
    ConversationUpdateDirector,
    PrivateRunDisclosureError,
    TaskTerminalOutcome,
    UnresolvedUpdate,
    UpdateAuditRecord,
    UpdateDecision,
    UpdateDecisionKind,
    UpdateDirective,
    UpdatePolicyInput,
)
from hermes_realtime.protocol import (
    ControlCancelAcknowledgedEvent,
    ControlCancelEvent,
    WorkCompletedEvent,
    WorkCompletedPayload,
    WorkDispatchAcknowledgedEvent,
    WorkDispatchAcknowledgedPayload,
    WorkDispatchRequestedEvent,
    WorkTerminalStatus,
)


class QueuedCompletionSource:
    def __init__(self, *updates: TaskTerminalOutcome) -> None:
        self._updates: asyncio.Queue[TaskTerminalOutcome] = asyncio.Queue()
        self.calls = 0
        for update in updates:
            self._updates.put_nowait(update)

    async def next_completion(self) -> TaskTerminalOutcome:
        self.calls += 1
        return await self._updates.get()


class ControllerSession:
    def __init__(self) -> None:
        self.request: WorkDispatchRequestedEvent | None = None
        self.updates: asyncio.Queue[WorkCompletedEvent] = asyncio.Queue()

    async def dispatch(
        self,
        request: WorkDispatchRequestedEvent,
    ) -> WorkDispatchAcknowledgedEvent:
        self.request = request
        return WorkDispatchAcknowledgedEvent(
            type="work.dispatch.acknowledged",
            event_id="evt_ack",
            session_id=request.session_id,
            sequence=request.sequence + 1,
            timestamp=request.timestamp,
            task_id=request.task_id,
            payload=WorkDispatchAcknowledgedPayload(
                accepted=True,
                run_id="deleg_private_update_director",
            ),
        )

    async def cancel(
        self,
        request: ControlCancelEvent,
    ) -> ControlCancelAcknowledgedEvent:
        raise AssertionError(request)

    async def next_update(self) -> WorkCompletedEvent:
        return await self.updates.get()

    async def close(self) -> None:
        return None


def test_director_rejects_non_authoritative_completion_source() -> None:
    with pytest.raises(TypeError, match="ConversationTaskController"):
        ConversationUpdateDirector(
            context=ConversationContextStore(),
            completions=QueuedCompletionSource(),  # type: ignore[arg-type]
            policy=CountingPolicy(),
        )


class CancellationResistantCompletionSource:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellations = 0

    async def next_completion(self) -> TaskTerminalOutcome:
        self.started.set()
        try:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
                    self.cancelled.set()
            raise RuntimeError("source released")
        finally:
            self.finished.set()


class ScriptedUpdatePolicy:
    def __init__(self, *directives: UpdateDirective) -> None:
        self._directives = iter(directives)
        self.inputs: list[UpdatePolicyInput] = []

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        self.inputs.append(policy_input)
        return next(self._directives)


class CountingPolicy:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        self.calls += 1
        return UpdateDirective(kind="ignore")


class FailingPolicy:
    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        raise LookupError("policy unavailable")


class DirectiveThenFailurePolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.failed = asyncio.Event()

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        self.calls += 1
        if self.calls == 1:
            return UpdateDirective(kind="ignore")
        self.failed.set()
        raise LookupError("second policy failed")


class CancellationResistantPolicy:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellations = 0

    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        del policy_input
        self.started.set()
        try:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
            return UpdateDirective(kind="ignore")
        finally:
            self.finished.set()


def _director_for_test(
    *,
    context: ConversationContextStore,
    completions: Any,
    policy: Any,
    **kwargs: Any,
) -> ConversationUpdateDirector:
    authority = ConversationTaskController(
        context=context,
        session=ControllerSession(),
        session_id="session_test_authority",
        id_factory=lambda: "unused",
    )
    result = ConversationUpdateDirector(
        context=context,
        completions=authority,
        policy=policy,
        **kwargs,
    )
    object.__setattr__(result, "_completions", completions)
    return result


@pytest.mark.asyncio
async def test_ordered_completions_produce_monotonic_update_decisions() -> None:
    context = ConversationContextStore()
    source = QueuedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_weather",
            status="completed",
            summary="Rain begins at six.",
        ),
        TaskTerminalOutcome(
            task_id="task_route",
            status="completed",
            summary="The selected bridge is closed.",
        ),
    )
    policy = ScriptedUpdatePolicy(
        UpdateDirective(kind="retain", text="Rain begins at six."),
        UpdateDirective(kind="interrupt", text="The selected bridge is closed."),
    )
    director = _director_for_test(
        context=context,
        completions=source,
        policy=policy,
    )

    first = await director.next_decision()
    second = await director.next_decision()
    urgent = await director.next_interrupt()
    await director.close()

    assert [first.sequence, second.sequence] == [1, 2]
    assert [first.kind, second.kind] == [
        UpdateDecisionKind.RETAIN,
        UpdateDecisionKind.INTERRUPT,
    ]
    assert [first.task_id, second.task_id] == ["task_weather", "task_route"]
    assert [item.context for item in policy.inputs] == [
        ConversationContextSnapshot(revision=0, messages=(), active_tasks=()),
        ConversationContextSnapshot(revision=0, messages=(), active_tasks=()),
    ]
    assert director.retained_updates() == (first,)
    assert urgent == second
    assert director.pending_interrupts() == ()


@pytest.mark.asyncio
async def test_mention_next_is_consumed_exactly_once() -> None:
    source = QueuedCompletionSource(
        TaskTerminalOutcome(
            task_id="task_notes",
            status="completed",
            summary="The notes are ready.",
        )
    )
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=source,
        policy=ScriptedUpdatePolicy(
            UpdateDirective(kind="mention_next", text="The notes are ready.")
        ),
    )

    decision = await director.next_decision()
    mention = await director.next_mention()

    assert mention == decision
    assert director.pending_mentions() == ()
    await director.close()


@pytest.mark.asyncio
async def test_settled_mention_claim_releases_publication_capacity() -> None:
    first_completion = TaskTerminalOutcome(
        task_id="task_first",
        status="completed",
        summary="The first report is ready.",
    )
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(first_completion),
        policy=ScriptedUpdatePolicy(
            UpdateDirective(kind="mention_next", text="The first report is ready.")
        ),
        max_pending_mentions=1,
    )
    owner = object()
    director._bind_action_executor(owner)
    director.start()
    await director._next_execution(owner)
    (claim,) = director._claim_pending_mentions(owner, max_count=1)

    second = UpdateDecision(
        sequence=2,
        completion=TaskTerminalOutcome(
            task_id="task_second",
            status="completed",
            summary="The second report is ready.",
        ),
        kind="mention_next",
        text="The second report is ready.",
    )
    publication = asyncio.create_task(director._publish_decision(second))
    done, _ = await asyncio.wait({publication}, timeout=0.01)
    assert not done

    director._settle_action_claim(owner, claim)
    await asyncio.wait_for(publication, timeout=0.1)
    assert director.pending_mentions() == (second,)
    await director.close()


@pytest.mark.asyncio
async def test_mutated_completion_is_rejected_before_policy_side_effects() -> None:
    class HostileTaskId(str):
        strip_calls = 0

        def strip(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            type(self).strip_calls += 1
            raise AssertionError("hostile task id executed")

    completion = TaskTerminalOutcome(
        task_id="task_safe",
        status="completed",
        summary="safe summary",
    )
    object.__setattr__(completion, "task_id", HostileTaskId("task_hostile"))
    policy = CountingPolicy()
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(completion),
        policy=policy,
    )

    with pytest.raises(TypeError, match="task_id"):
        await director.next_decision()

    assert HostileTaskId.strip_calls == 0
    assert policy.calls == 0
    await director.close()


def test_embedded_private_run_token_is_rejected_from_update_text() -> None:
    with pytest.raises(PrivateRunDisclosureError, match="private run token"):
        UpdateDirective(kind="retain", text="prefixdeleg_private_run suffix")


@pytest.mark.asyncio
async def test_close_wakes_every_blocked_update_consumer() -> None:
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(),
        policy=CountingPolicy(),
    )
    consumers = [
        asyncio.create_task(director.next_decision()),
        asyncio.create_task(director.next_mention()),
        asyncio.create_task(director.next_interrupt()),
    ]
    await asyncio.sleep(0)

    await director.close()
    results = await asyncio.wait_for(
        asyncio.gather(*consumers, return_exceptions=True),
        timeout=0.2,
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("closed" in str(result) for result in results)


@pytest.mark.asyncio
async def test_policy_timeout_is_finite_when_policy_resists_cancellation() -> None:
    policy = CancellationResistantPolicy()
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_slow",
                status="completed",
                summary="slow result",
            )
        ),
        policy=policy,
        policy_timeout_ms=5,
        cleanup_timeout_ms=5,
    )
    decision = asyncio.create_task(director.next_decision())
    await policy.started.wait()

    await asyncio.sleep(0.05)

    assert decision.done()
    with pytest.raises(TimeoutError, match="policy"):
        await decision
    assert policy.cancellations == 1
    close_task = asyncio.create_task(director.close())
    await asyncio.sleep(0.05)
    assert close_task.done()
    try:
        with pytest.raises(TimeoutError, match="cleanup"):
            await close_task
    finally:
        policy.release.set()
        await asyncio.wait_for(policy.finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_close_is_finite_when_completion_source_resists_cancellation() -> None:
    source = CancellationResistantCompletionSource()
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=source,
        policy=CountingPolicy(),
        cleanup_timeout_ms=5,
    )
    director.start()
    await source.started.wait()

    close_task = asyncio.create_task(director.close())
    await asyncio.sleep(0.05)

    assert close_task.done()
    try:
        with pytest.raises(TimeoutError, match="cleanup"):
            await close_task
        assert source.cancellations == 1
    finally:
        source.release.set()
        await asyncio.wait_for(source.finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_policy_failure_retains_ambiguous_authoritative_completion() -> None:
    completion = TaskTerminalOutcome(
        task_id="task_build",
        status="failed",
        reason="build worker disconnected",
    )
    source = QueuedCompletionSource(completion)
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=source,
        policy=FailingPolicy(),
    )

    with pytest.raises(LookupError, match="policy unavailable"):
        await director.next_decision()

    assert director.unresolved_updates() == (
        UnresolvedUpdate(
            sequence=1,
            completion=completion,
            context=ConversationContextSnapshot(
                revision=0,
                messages=(),
                active_tasks=(),
            ),
            phase="policy",
        ),
    )
    await asyncio.sleep(0)
    with pytest.raises(LookupError, match="policy unavailable"):
        await director.next_decision()
    await asyncio.sleep(0)
    assert source.calls == 1
    await director.close()


@pytest.mark.asyncio
async def test_retained_update_requires_exact_explicit_release() -> None:
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_weather",
                status="completed",
                summary="Rain begins at six.",
            )
        ),
        policy=ScriptedUpdatePolicy(
            UpdateDirective(kind="retain", text="Rain begins at six.")
        ),
    )

    decision = await director.next_decision()
    director.release_retained(decision)

    assert director.retained_updates() == ()
    with pytest.raises(KeyError, match="not retained"):
        director.release_retained(decision)
    await director.close()


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_cancel_owned_close_cleanup() -> None:
    source = CancellationResistantCompletionSource()
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=source,
        policy=CountingPolicy(),
        cleanup_timeout_ms=5,
    )
    director.start()
    await source.started.wait()
    close_task = asyncio.create_task(director.close())
    await source.cancelled.wait()

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    await asyncio.sleep(0.05)

    try:
        with pytest.raises(TimeoutError, match="cleanup"):
            await director.close()
    finally:
        source.release.set()
        await asyncio.wait_for(source.finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_replayed_completion_is_rejected_before_second_policy_call() -> None:
    completion = TaskTerminalOutcome(
        task_id="task_once",
        status="completed",
        summary="done",
    )
    policy = ScriptedUpdatePolicy(
        UpdateDirective(kind="ignore"),
        UpdateDirective(kind="ignore"),
    )
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(completion, completion),
        policy=policy,
    )

    first = await director.next_decision()
    assert first.task_id == "task_once"
    with pytest.raises(RuntimeError, match="replayed"):
        await director.next_decision()

    assert len(policy.inputs) == 1
    await director.close()


@pytest.mark.asyncio
async def test_ignore_is_auditable_without_retaining_conversational_text() -> None:
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_noise",
                status="completed",
                summary="verbose result that should not enter retained state",
            )
        ),
        policy=ScriptedUpdatePolicy(UpdateDirective(kind="ignore")),
    )

    decision = await director.next_decision()
    record = UpdateAuditRecord(
        sequence=1,
        task_id="task_noise",
        status="completed",
        kind="ignore",
    )

    assert director.audit_records() == (record,)
    assert not hasattr(record, "text")
    assert director.retained_updates() == ()
    director.release_audit(record)
    assert director.audit_records() == ()
    assert decision.text is None
    await director.close()


@pytest.mark.asyncio
async def test_retain_capacity_failure_preserves_second_update_as_unresolved() -> None:
    first_completion = TaskTerminalOutcome(
        task_id="task_first",
        status="completed",
        summary="first",
    )
    second_completion = TaskTerminalOutcome(
        task_id="task_second",
        status="completed",
        summary="second",
    )
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(first_completion, second_completion),
        policy=ScriptedUpdatePolicy(
            UpdateDirective(kind="retain", text="first"),
            UpdateDirective(kind="retain", text="second"),
        ),
        max_retained_updates=1,
    )

    first = await director.next_decision()
    with pytest.raises(RuntimeError, match="retained update capacity"):
        await director.next_decision()

    assert director.retained_updates() == (first,)
    assert director.unresolved_updates() == (
        UnresolvedUpdate(
            sequence=2,
            completion=second_completion,
            context=ConversationContextSnapshot(
                revision=0,
                messages=(),
                active_tasks=(),
            ),
            phase="publication",
        ),
    )
    await director.close()


@pytest.mark.asyncio
async def test_full_decision_queue_does_not_hide_later_policy_failure() -> None:
    second_completion = TaskTerminalOutcome(
        task_id="task_second",
        status="failed",
        reason="worker failed",
    )
    policy = DirectiveThenFailurePolicy()
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_first",
                status="completed",
                summary="first",
            ),
            second_completion,
        ),
        policy=policy,
        max_decisions=1,
    )
    director.start()
    await policy.failed.wait()

    first = await director.next_decision()
    with pytest.raises(LookupError, match="second policy failed"):
        await director.next_decision()

    assert first.task_id == "task_first"
    assert director.unresolved_updates()[0].completion == second_completion
    assert director.unresolved_updates()[0].phase == "policy"
    await director.close()


@pytest.mark.asyncio
async def test_competing_mention_consumers_cannot_claim_one_action_twice() -> None:
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(
            TaskTerminalOutcome(
                task_id="task_notes",
                status="completed",
                summary="notes ready",
            )
        ),
        policy=ScriptedUpdatePolicy(
            UpdateDirective(kind="mention_next", text="notes ready")
        ),
    )
    claims = [asyncio.create_task(director.next_mention()) for _ in range(2)]

    done, pending = await asyncio.wait(claims, return_when=asyncio.FIRST_COMPLETED)
    assert len(done) == 1
    assert len(pending) == 1
    claimed = next(iter(done)).result()
    await director.close()
    pending_result = (
        await asyncio.gather(next(iter(pending)), return_exceptions=True)
    )[0]

    assert claimed.kind == UpdateDecisionKind.MENTION_NEXT
    assert isinstance(pending_result, RuntimeError)


@pytest.mark.asyncio
async def test_director_consumes_authoritative_task_controller_completion() -> None:
    timestamp = datetime(2026, 7, 21, 21, 30, tzinfo=UTC)
    context = ConversationContextStore()
    session = ControllerSession()
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_update_director",
        id_factory=iter(("one", "dispatch")).__next__,
        clock=lambda: timestamp,
    )
    policy = ScriptedUpdatePolicy(
        UpdateDirective(kind="mention_next", text="result is ready")
    )
    director = ConversationUpdateDirector(
        context=context,
        completions=controller,
        policy=policy,
    )

    dispatch = await controller.dispatch(
        objective="produce the result",
        utterance_id="utterance_001",
    )
    assert dispatch.accepted is True
    assert session.request is not None
    await session.updates.put(
        WorkCompletedEvent(
            type="work.completed",
            event_id="evt_completed",
            session_id="session_update_director",
            sequence=session.request.sequence + 2,
            timestamp=timestamp,
            task_id=dispatch.task_id,
            run_id="deleg_private_update_director",
            payload=WorkCompletedPayload(
                status=WorkTerminalStatus.COMPLETED,
                summary="result produced",
            ),
        )
    )

    decision = await director.next_decision()

    assert decision.task_id == dispatch.task_id
    assert decision.completion.summary == "result produced"
    assert policy.inputs[0].context.revision == context.snapshot().revision
    assert "deleg_private_update_director" not in repr(decision)
    assert "deleg_private_update_director" not in repr(policy.inputs)
    await director.close()
    await controller.close()


@pytest.mark.asyncio
async def test_director_retries_transient_close_operation_failure() -> None:
    director = _director_for_test(
        context=ConversationContextStore(),
        completions=QueuedCompletionSource(),
        policy=ScriptedUpdatePolicy(UpdateDirective(kind="ignore")),
    )
    close_attempts = 0

    async def flaky_close(self: ConversationUpdateDirector) -> None:
        nonlocal close_attempts
        del self
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("temporary director close failure")

    director._close_owned = MethodType(  # type: ignore[method-assign]
        flaky_close,
        director,
    )

    with pytest.raises(RuntimeError, match="temporary director close failure"):
        await director.close()
    await director.close()

    assert close_attempts == 2
