"""Ordered provider-neutral routing for authoritative conversation updates."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .context import (
    ActiveTaskSummary,
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
    PrivateRunDisclosureError,
)
from .tasks import ConversationTaskController, TaskTerminalOutcome

_MAX_UPDATE_TEXT_CHARS = 4096
_MAX_DECISION_SEQUENCE = (1 << 63) - 1
_PRIVATE_RUN_TOKEN_PATTERN = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*")


class UpdateDecisionKind(StrEnum):
    """Closed set of model-safe Update Director outcomes."""

    IGNORE = "ignore"
    RETAIN = "retain"
    MENTION_NEXT = "mention_next"
    INTERRUPT = "interrupt"


def _validate_sequence(sequence: int) -> None:
    if type(sequence) is not int:
        raise TypeError("update sequence must be an exact built-in integer")
    if not 1 <= sequence <= _MAX_DECISION_SEQUENCE:
        raise ValueError("update sequence must be positive and bounded")


def _validate_kind(kind: str) -> None:
    if type(kind) is not str:
        raise TypeError("update decision kind must be an exact built-in string")
    if kind not in tuple(item.value for item in UpdateDecisionKind):
        raise ValueError("update decision kind is unsupported")


def _validate_text(text: str | None, *, kind: str) -> None:
    if kind == UpdateDecisionKind.IGNORE.value:
        if text is not None:
            raise ValueError("ignore decisions must not retain text")
        return
    if type(text) is not str:
        raise TypeError("non-ignore update text must be an exact built-in string")
    if not text.strip() or len(text) > _MAX_UPDATE_TEXT_CHARS:
        raise ValueError("update text must be nonblank and bounded")
    if _PRIVATE_RUN_TOKEN_PATTERN.search(text) is not None:
        raise PrivateRunDisclosureError("update text contains a private run token")


def _trusted_completion(update: TaskTerminalOutcome) -> TaskTerminalOutcome:
    if type(update) is not TaskTerminalOutcome:
        raise TypeError("completion must be an exact TaskTerminalOutcome")
    return TaskTerminalOutcome(
        task_id=update.task_id,
        status=update.status,
        summary=update.summary,
        reason=update.reason,
    )


def _trusted_snapshot(snapshot: ConversationContextSnapshot) -> ConversationContextSnapshot:
    if type(snapshot) is not ConversationContextSnapshot:
        raise TypeError("context must be an exact ConversationContextSnapshot")
    return ConversationContextSnapshot(
        revision=snapshot.revision,
        messages=tuple(
            ConversationMessage(role=message.role, text=message.text)
            for message in snapshot.messages
        ),
        active_tasks=tuple(
            ActiveTaskSummary(task_id=task.task_id, objective=task.objective)
            for task in snapshot.active_tasks
        ),
        terminal_task_count=snapshot.terminal_task_count,
    )


@dataclass(frozen=True, slots=True)
class UpdateDirective:
    """One provider-neutral policy result before director sequencing."""

    kind: str
    text: str | None = None

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        _validate_text(self.text, kind=self.kind)


@dataclass(frozen=True, slots=True)
class UpdatePolicyInput:
    """Immutable policy input with no private transport authority."""

    sequence: int
    completion: TaskTerminalOutcome
    context: ConversationContextSnapshot

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        object.__setattr__(self, "completion", _trusted_completion(self.completion))
        object.__setattr__(self, "context", _trusted_snapshot(self.context))


@dataclass(frozen=True, slots=True)
class UnresolvedUpdate:
    """Exact authoritative input retained when its decision cannot settle."""

    sequence: int
    completion: TaskTerminalOutcome
    context: ConversationContextSnapshot
    phase: str

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        object.__setattr__(self, "completion", _trusted_completion(self.completion))
        object.__setattr__(self, "context", _trusted_snapshot(self.context))
        if type(self.phase) is not str:
            raise TypeError("unresolved update phase must be an exact built-in string")
        if self.phase not in ("admission", "policy", "publication"):
            raise ValueError("unresolved update phase is unsupported")


def _trusted_unresolved(update: UnresolvedUpdate) -> UnresolvedUpdate:
    if type(update) is not UnresolvedUpdate:
        raise TypeError("unresolved update must be an exact UnresolvedUpdate")
    return UnresolvedUpdate(
        sequence=update.sequence,
        completion=update.completion,
        context=update.context,
        phase=update.phase,
    )


@dataclass(frozen=True, slots=True)
class UpdateAuditRecord:
    """Bounded decision metadata without conversational payload text."""

    sequence: int
    task_id: str
    status: str
    kind: str

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        if self.status == "completed":
            completion = TaskTerminalOutcome(
                task_id=self.task_id,
                status=self.status,
                summary="audit",
            )
        else:
            completion = TaskTerminalOutcome(
                task_id=self.task_id,
                status=self.status,
                reason="audit",
            )
        object.__setattr__(self, "task_id", completion.task_id)
        object.__setattr__(self, "status", completion.status)
        _validate_kind(self.kind)


def _trusted_audit(record: UpdateAuditRecord) -> UpdateAuditRecord:
    if type(record) is not UpdateAuditRecord:
        raise TypeError("audit record must be an exact UpdateAuditRecord")
    return UpdateAuditRecord(
        sequence=record.sequence,
        task_id=record.task_id,
        status=record.status,
        kind=record.kind,
    )


@dataclass(frozen=True, slots=True)
class UpdateDecision:
    """One ordered, bounded, model-safe Update Director decision."""

    sequence: int
    completion: TaskTerminalOutcome
    kind: str
    text: str | None = None

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        object.__setattr__(self, "completion", _trusted_completion(self.completion))
        _validate_kind(self.kind)
        _validate_text(self.text, kind=self.kind)

    @property
    def task_id(self) -> str:
        return self.completion.task_id

    @property
    def status(self) -> str:
        return self.completion.status


def _trusted_decision(decision: UpdateDecision) -> UpdateDecision:
    if type(decision) is not UpdateDecision:
        raise TypeError("decision must be an exact UpdateDecision")
    return UpdateDecision(
        sequence=decision.sequence,
        completion=decision.completion,
        kind=decision.kind,
        text=decision.text,
    )


@dataclass(frozen=True, slots=True, eq=False)
class _UpdateActionClaim:
    """Opaque one-use capability bound to one director and executor."""


class UpdatePolicy(Protocol):
    async def decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective: ...


class ConversationUpdateDirector:
    """Consume authoritative completions and emit decisions in input order."""

    def __init__(
        self,
        *,
        context: ConversationContextStore,
        completions: ConversationTaskController,
        policy: UpdatePolicy,
        max_decisions: int = 16,
        max_retained_updates: int = 16,
        max_pending_mentions: int = 16,
        max_pending_interrupts: int = 16,
        max_completion_ids: int = 4096,
        max_audit_records: int = 256,
        policy_timeout_ms: int = 1000,
        cleanup_timeout_ms: int = 1000,
    ) -> None:
        if type(context) is not ConversationContextStore:
            raise TypeError("context must be an exact ConversationContextStore")
        if type(completions) is not ConversationTaskController:
            raise TypeError(
                "completions must be an exact ConversationTaskController"
            )
        if type(max_decisions) is not int:
            raise TypeError("max_decisions must be an exact built-in integer")
        if not 1 <= max_decisions <= 256:
            raise ValueError("max_decisions must be between 1 and 256")
        if type(max_retained_updates) is not int:
            raise TypeError("max_retained_updates must be an exact built-in integer")
        if not 1 <= max_retained_updates <= 256:
            raise ValueError("max_retained_updates must be between 1 and 256")
        for value, name in (
            (max_pending_mentions, "max_pending_mentions"),
            (max_pending_interrupts, "max_pending_interrupts"),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact built-in integer")
            if not 1 <= value <= 256:
                raise ValueError(f"{name} must be between 1 and 256")
        if type(max_completion_ids) is not int:
            raise TypeError("max_completion_ids must be an exact built-in integer")
        if not 1 <= max_completion_ids <= 65_536:
            raise ValueError("max_completion_ids must be between 1 and 65536")
        if type(max_audit_records) is not int:
            raise TypeError("max_audit_records must be an exact built-in integer")
        if not 1 <= max_audit_records <= 4096:
            raise ValueError("max_audit_records must be between 1 and 4096")
        if type(policy_timeout_ms) is not int:
            raise TypeError("policy_timeout_ms must be an exact built-in integer")
        if not 1 <= policy_timeout_ms <= 60_000:
            raise ValueError("policy_timeout_ms must be between 1 and 60000")
        if type(cleanup_timeout_ms) is not int:
            raise TypeError("cleanup_timeout_ms must be an exact built-in integer")
        if not 1 <= cleanup_timeout_ms <= 60_000:
            raise ValueError("cleanup_timeout_ms must be between 1 and 60000")
        self._context = context
        self._completions = completions
        self._policy = policy
        self._policy_timeout = policy_timeout_ms / 1000
        self._cleanup_timeout = cleanup_timeout_ms / 1000
        self._max_decisions = max_decisions
        self._decisions: deque[UpdateDecision] = deque()
        self._max_retained_updates = max_retained_updates
        self._retained: list[UpdateDecision] = []
        self._unresolved: list[UnresolvedUpdate] = []
        self._max_pending_mentions = max_pending_mentions
        self._mention_slots = asyncio.BoundedSemaphore(max_pending_mentions)
        self._mentions: deque[UpdateDecision] = deque()

        self._max_pending_interrupts = max_pending_interrupts
        self._interrupts: deque[UpdateDecision] = deque()
        self._action_owner: object | None = None
        self._max_active_action_claims = (
            max_pending_mentions + max_pending_interrupts
        )
        self._action_claims: dict[
            int,
            tuple[_UpdateActionClaim, UpdateDecision],
        ] = {}
        self._max_completion_ids = max_completion_ids
        self._seen_task_ids: set[str] = set()
        self._max_audit_records = max_audit_records
        self._audit: list[UpdateAuditRecord] = []
        self._sequence = 0
        self._consumer: asyncio.Task[None] | None = None
        self._condition = asyncio.Condition()
        self._error: BaseException | None = None
        self._error_observed = False
        self._policy_tasks: set[asyncio.Task[UpdateDirective]] = set()
        self._close_operation: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._error is not None:
            self._error_observed = True
            raise self._error
        if self._closed:
            raise RuntimeError("conversation update director is closed")
        if self._consumer is None:
            self._consumer = asyncio.create_task(
                self._consume(),
                name="conversation-update-director",
            )
            self._consumer.add_done_callback(self._consumer_done)

    async def next_decision(self) -> UpdateDecision:
        if self._action_owner is not None:
            raise RuntimeError("update decisions are bound to an action executor")
        self._prepare_next(self._decisions)
        return await self._next_queued(self._decisions)

    def _bind_action_executor(self, owner: object) -> None:
        if owner is None:
            raise TypeError("action owner must not be None")
        if self._action_owner is None:
            self._action_owner = owner
            return
        if self._action_owner is not owner:
            raise RuntimeError("conversation update director already has an action executor")

    def _unbind_action_executor(self, owner: object) -> None:
        self._require_action_owner(owner)
        if self._action_claims:
            raise RuntimeError("cannot unbind an executor with active action claims")
        self._action_owner = None

    async def _next_execution(
        self,
        owner: object,
    ) -> tuple[UpdateDecision, _UpdateActionClaim | None]:
        self._require_action_owner(owner)
        self.start()
        async with self._condition:
            while True:
                if self._decisions:
                    decision = self._decisions[0]
                    claim: _UpdateActionClaim | None = None
                    if decision.kind == UpdateDecisionKind.INTERRUPT.value:
                        if not self._interrupts or self._interrupts[0] != decision:
                            raise RuntimeError(
                                "interrupt queue contradicts ordered decision"
                            )
                        claim = self._issue_action_claim(decision)
                        self._interrupts.popleft()
                    self._decisions.popleft()
                    self._condition.notify_all()
                    return _trusted_decision(decision), claim
                if self._error is not None:
                    self._error_observed = True
                    raise self._error
                if self._closed:
                    raise RuntimeError("conversation update director is closed")
                await self._condition.wait()

    def _claim_pending_mentions(
        self,
        owner: object,
        *,
        max_count: int,
    ) -> tuple[_UpdateActionClaim, ...]:
        self._require_action_owner(owner)
        if type(max_count) is not int or not 1 <= max_count <= 256:
            raise ValueError("max_count must be between 1 and 256")
        available = self._max_active_action_claims - len(self._action_claims)
        count = min(len(self._mentions), max_count, available)
        claims: list[_UpdateActionClaim] = []
        for _ in range(count):
            claims.append(self._issue_action_claim(self._mentions.popleft()))
        return tuple(claims)

    def _decision_for_action_claim(
        self,
        owner: object,
        claim: _UpdateActionClaim,
    ) -> UpdateDecision:
        self._require_action_owner(owner)
        if type(claim) is not _UpdateActionClaim:
            raise TypeError("action claim must be an exact _UpdateActionClaim")
        binding = self._action_claims.get(id(claim))
        if binding is None or binding[0] is not claim:
            raise KeyError("action claim is unknown or already settled")
        return _trusted_decision(binding[1])

    def _settle_action_claim(
        self,
        owner: object,
        claim: _UpdateActionClaim,
    ) -> None:
        decision = self._decision_for_action_claim(owner, claim)
        del self._action_claims[id(claim)]
        if decision.kind == UpdateDecisionKind.MENTION_NEXT.value:
            self._mention_slots.release()

    def _action_claim_is_active(
        self,
        owner: object,
        claim: _UpdateActionClaim,
    ) -> bool:
        self._require_action_owner(owner)
        if type(claim) is not _UpdateActionClaim:
            raise TypeError("action claim must be an exact claim")
        registered = self._action_claims.get(id(claim))
        return registered is not None and registered[0] is claim

    def _release_mention_action_claims(
        self,
        owner: object,
        claims: tuple[_UpdateActionClaim, ...],
    ) -> None:
        self._require_action_owner(owner)
        if type(claims) is not tuple:
            raise TypeError("mention claims must be an exact tuple")
        if len({id(claim) for claim in claims}) != len(claims):
            raise ValueError("mention claims must not contain duplicates")
        decisions = tuple(
            self._decision_for_action_claim(owner, claim) for claim in claims
        )
        if any(
            decision.kind != UpdateDecisionKind.MENTION_NEXT.value
            for decision in decisions
        ):
            raise ValueError("only mention claims can return to the mention queue")
        if len(self._mentions) + len(decisions) > self._max_pending_mentions:
            raise RuntimeError("pending mention capacity exhausted during claim release")
        for claim in claims:
            del self._action_claims[id(claim)]
        for decision in reversed(decisions):
            self._mentions.appendleft(decision)

    def _issue_action_claim(self, decision: UpdateDecision) -> _UpdateActionClaim:
        if len(self._action_claims) >= self._max_active_action_claims:
            raise RuntimeError("active update action claim capacity exhausted")
        claim = _UpdateActionClaim()
        self._action_claims[id(claim)] = (claim, _trusted_decision(decision))
        return claim

    def _require_action_owner(self, owner: object) -> None:
        if self._action_owner is not owner:
            raise RuntimeError("caller does not own update action execution")

    def retained_updates(self) -> tuple[UpdateDecision, ...]:
        return tuple(_trusted_decision(decision) for decision in self._retained)

    def release_retained(self, decision: UpdateDecision) -> None:
        trusted = _trusted_decision(decision)
        for index, retained in enumerate(self._retained):
            if retained == trusted:
                del self._retained[index]
                return
        raise KeyError("update decision is not retained")

    def unresolved_updates(self) -> tuple[UnresolvedUpdate, ...]:
        return tuple(_trusted_unresolved(update) for update in self._unresolved)

    def audit_records(self) -> tuple[UpdateAuditRecord, ...]:
        return tuple(_trusted_audit(record) for record in self._audit)

    def release_audit(self, record: UpdateAuditRecord) -> None:
        trusted = _trusted_audit(record)
        for index, existing in enumerate(self._audit):
            if existing == trusted:
                del self._audit[index]
                return
        raise KeyError("update audit record is not retained")

    def pending_mentions(self) -> tuple[UpdateDecision, ...]:
        return tuple(_trusted_decision(decision) for decision in self._mentions)

    def pending_interrupts(self) -> tuple[UpdateDecision, ...]:
        return tuple(_trusted_decision(decision) for decision in self._interrupts)

    async def next_mention(self) -> UpdateDecision:
        if self._action_owner is not None:
            raise RuntimeError("mention actions are bound to an action executor")
        self._prepare_next(self._mentions)
        decision = await self._next_queued(self._mentions)
        self._mention_slots.release()
        return decision

    async def next_interrupt(self) -> UpdateDecision:
        if self._action_owner is not None:
            raise RuntimeError("interrupt actions are bound to an action executor")
        self._prepare_next(self._interrupts)
        return await self._next_queued(self._interrupts)

    def _prepare_next(self, queue: deque[UpdateDecision]) -> None:
        if self._error is not None and queue:
            return
        self.start()

    async def close(self) -> None:
        operation = self._close_operation
        if operation is None or self._close_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="conversation-update-director-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    @staticmethod
    def _close_failed(operation: asyncio.Task[None]) -> bool:
        if not operation.done():
            return False
        if operation.cancelled():
            return True
        return operation.exception() is not None

    async def _close_owned(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        owned: set[asyncio.Task[object]] = set(self._policy_tasks)
        consumer = self._consumer
        if consumer is not None:
            owned.add(consumer)
        if owned:
            for task in owned:
                if not task.done() and task.cancelling() == 0:
                    task.cancel()
            done, pending = await asyncio.wait(owned, timeout=self._cleanup_timeout)
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    task.exception()
            if pending:
                cleanup_error = TimeoutError(
                    "update director cleanup did not settle before its deadline"
                )
                if self._error is not None and not self._error_observed:
                    self._error_observed = True
                    raise BaseExceptionGroup(
                        "update director operation and cleanup failed",
                        [self._error, cleanup_error],
                    )
                raise cleanup_error
        if self._action_claims:
            raise RuntimeError("update director closed with active action claims")
        if self._error is not None and not self._error_observed:
            self._error_observed = True
            raise self._error

    def _consumer_done(self, task: asyncio.Task[None]) -> None:
        if self._consumer is task:
            self._consumer = None
        if task.cancelled():
            return
        task.exception()

    async def _next_queued(
        self,
        queue: deque[UpdateDecision],
    ) -> UpdateDecision:
        async with self._condition:
            while True:
                if queue:
                    decision = queue.popleft()
                    self._condition.notify_all()
                    return _trusted_decision(decision)
                if self._error is not None:
                    self._error_observed = True
                    raise self._error
                if self._closed:
                    raise RuntimeError("conversation update director is closed")
                await self._condition.wait()

    async def _consume(self) -> None:
        unresolved: UnresolvedUpdate | None = None
        try:
            while True:
                completion = _trusted_completion(await self._completions.next_completion())
                if self._sequence >= _MAX_DECISION_SEQUENCE:
                    raise RuntimeError("update decision sequence capacity exhausted")
                sequence = self._sequence + 1
                snapshot = _trusted_snapshot(self._context.snapshot())
                unresolved = UnresolvedUpdate(
                    sequence=sequence,
                    completion=completion,
                    context=snapshot,
                    phase="admission",
                )
                if completion.task_id in self._seen_task_ids:
                    raise RuntimeError("completion task identity was replayed")
                if len(self._seen_task_ids) >= self._max_completion_ids:
                    raise RuntimeError("completion identity capacity exhausted")
                self._seen_task_ids.add(completion.task_id)
                policy_input = UpdatePolicyInput(
                    sequence=sequence,
                    completion=completion,
                    context=snapshot,
                )
                unresolved = UnresolvedUpdate(
                    sequence=sequence,
                    completion=completion,
                    context=snapshot,
                    phase="policy",
                )
                raw_directive = await self._decide(policy_input)
                if type(raw_directive) is not UpdateDirective:
                    raise TypeError("policy must return an exact UpdateDirective")
                directive = UpdateDirective(
                    kind=raw_directive.kind,
                    text=raw_directive.text,
                )
                decision = UpdateDecision(
                    sequence=sequence,
                    completion=completion,
                    kind=directive.kind,
                    text=directive.text,
                )
                unresolved = UnresolvedUpdate(
                    sequence=sequence,
                    completion=completion,
                    context=snapshot,
                    phase="publication",
                )
                await self._publish_decision(decision)
                unresolved = None
        except asyncio.CancelledError:
            if unresolved is not None:
                await self._retain_unresolved(unresolved)
            raise
        except BaseException as error:
            async with self._condition:
                if unresolved is not None:
                    self._unresolved.append(_trusted_unresolved(unresolved))
                self._error = error
                self._condition.notify_all()

    async def _retain_unresolved(self, unresolved: UnresolvedUpdate) -> None:
        trusted = _trusted_unresolved(unresolved)
        async with self._condition:
            self._unresolved.append(trusted)
            self._condition.notify_all()

    async def _decide(self, policy_input: UpdatePolicyInput) -> UpdateDirective:
        task = asyncio.create_task(
            self._policy.decide(policy_input),
            name=f"conversation-update-policy-{policy_input.sequence}",
        )
        self._policy_tasks.add(task)
        task.add_done_callback(self._policy_task_done)
        try:
            done, _ = await asyncio.wait({task}, timeout=self._policy_timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            raise TimeoutError("update policy did not settle before its deadline")
        return task.result()

    def _policy_task_done(self, task: asyncio.Task[UpdateDirective]) -> None:
        self._policy_tasks.discard(task)
        if task.cancelled():
            return
        task.exception()

    async def _publish_decision(self, decision: UpdateDecision) -> None:
        trusted = _trusted_decision(decision)
        mention_reserved = trusted.kind == UpdateDecisionKind.MENTION_NEXT.value
        if mention_reserved:
            await self._mention_slots.acquire()
        try:
            await self._publish_reserved_decision(trusted)
        except BaseException:
            if mention_reserved:
                self._mention_slots.release()
            raise

    async def _publish_reserved_decision(self, trusted: UpdateDecision) -> None:
        async with self._condition:
            while len(self._decisions) >= self._max_decisions:
                if self._closed:
                    raise RuntimeError("conversation update director is closed")
                await self._condition.wait()
            if self._closed:
                raise RuntimeError("conversation update director is closed")
            if len(self._audit) >= self._max_audit_records:
                raise RuntimeError("update audit capacity exhausted")
            if trusted.kind == UpdateDecisionKind.RETAIN.value:
                if len(self._retained) >= self._max_retained_updates:
                    raise RuntimeError("retained update capacity exhausted")
            elif (
                trusted.kind == UpdateDecisionKind.INTERRUPT.value
                and len(self._interrupts) >= self._max_pending_interrupts
            ):
                raise RuntimeError("pending interrupt capacity exhausted")

            if trusted.kind == UpdateDecisionKind.RETAIN.value:
                self._retained.append(_trusted_decision(trusted))
            elif trusted.kind == UpdateDecisionKind.MENTION_NEXT.value:
                self._mentions.append(_trusted_decision(trusted))
            elif trusted.kind == UpdateDecisionKind.INTERRUPT.value:
                self._interrupts.append(_trusted_decision(trusted))
            self._audit.append(
                UpdateAuditRecord(
                    sequence=trusted.sequence,
                    task_id=trusted.task_id,
                    status=trusted.status,
                    kind=trusted.kind,
                )
            )
            self._decisions.append(trusted)
            self._sequence = trusted.sequence
            self._condition.notify_all()
