"""One bounded, host-owned proposal ahead of the existing work admission surface.

The provider's selection is data. The host's configured start policy owns its
admission; confirmation and standing authorization share the same bound consume.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from .work_tools import (
    _UNAVAILABLE_REASON,
    WorkCommandAdmission,
    WorkStartResult,
    _validate_exact_text,
)

_AFFIRMATIVES = frozenset({"yes", "yeah", "yep", "sure", "okay", "go ahead", "do it", "please do"})
_PublicValue = str | int | bool | None
MAX_TASK_PROPOSAL_OBJECTIVE_CHARS = 1024
TaskStartPolicy = Literal["confirm", "automatic"]


def validate_task_start_policy(value: TaskStartPolicy) -> None:
    if type(value) is not str or value not in ("confirm", "automatic"):
        print('[task-start-policy] {"category":"configuration_refused","count":1}', flush=True)
        raise ValueError("task start policy must be exactly 'confirm' or 'automatic'")


@dataclass(frozen=True, slots=True)
class HostTaskProposalRequest:
    """An effect-free, terminal provider selection; not an admission receipt."""

    objective: str

    def __post_init__(self) -> None:
        _validate_exact_text(self.objective, "objective", maximum=MAX_TASK_PROPOSAL_OBJECTIVE_CHARS)


@dataclass(frozen=True, slots=True)
class TaskProposal:
    proposal_id: str
    objective: str
    participant: str
    generation: int
    expires_at: float

    @property
    def readback(self) -> str:
        # Never teach an assent phrase in outgoing audio: it can return through STT.
        return "Proposed background task. " + self.objective + " Shall I send this task?"


class _Surface(Protocol):
    @property
    def max_objective_chars(self) -> int: ...

    async def submit_start_command(
        self,
        *,
        objective: str,
        invocation_id: str,
    ) -> WorkCommandAdmission: ...

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult: ...


class ConversationTaskProposals:
    """Session-scoped proposal state, independent of foreground turn leases and tasks."""

    def __init__(
        self,
        *,
        surface: _Surface,
        observer: Callable[[str, dict[str, _PublicValue]], None],
        clock: Callable[[], float] = time.monotonic,
        speech_projection: Callable[[str], str | None] | None = None,
        start_policy: TaskStartPolicy = "confirm",
    ) -> None:
        validate_task_start_policy(start_policy)
        self._start_policy = start_policy
        self._surface = surface
        self._observer = observer
        self._clock = clock
        self._proposal_prefix = secrets.token_hex(8)
        self._proposal_serial = 0
        self._speech_projection = speech_projection
        self._binding: tuple[str, int] | None = None
        self._pending: TaskProposal | None = None
        self._pending_visible = False
        self._readback_complete: TaskProposal | None = None
        self._capture_binding: TaskProposal | None = None
        self._deadline: float | None = None
        self._readback_settled = False

    @property
    def max_objective_chars(self) -> int:
        return min(self._surface.max_objective_chars, MAX_TASK_PROPOSAL_OBJECTIVE_CHARS)

    def bind(self, participant: str, generation: int) -> None:
        if type(participant) is not str or not participant or len(participant) > 256:
            raise ValueError("proposal participant must be a bounded exact string")
        if type(generation) is not int or not 1 <= generation < 2**53:
            raise ValueError("proposal generation must be a positive browser-safe integer")
        binding = participant, generation
        if binding != self._binding:
            self.retire("binding_changed")
            self._binding = binding

    def close_binding(self) -> None:
        self.retire("binding_closed")
        self._binding = None

    @staticmethod
    def is_confirmation(text: str) -> bool:
        # Only case, outer whitespace and trailing .!? change; no substring matching,
        # stemming, synonym inference, or model-controlled intent classifier.
        return type(text) is str and text.strip().rstrip(".!?").casefold() in _AFFIRMATIVES

    def complete_readback(
        self,
        proposal: TaskProposal,
        confirmed_text: str,
        *,
        queued_chunks: int,
        confirmed_chunks: int,
    ) -> None:
        """Arm future clean captures only after every exact readback chunk settled."""
        if proposal is not self._pending or self._readback_settled:
            self._evidence("stale_readback_completion")
            return
        self._readback_settled = True
        project = self._speech_projection
        if (
            self._deadline is not None
            and self._clock() < self._deadline
            and confirmed_text == proposal.readback
            and type(queued_chunks) is int
            and type(confirmed_chunks) is int
            and queued_chunks > 0
            and queued_chunks == confirmed_chunks
            and project is not None
            and project(proposal.readback) == proposal.readback
        ):
            self._readback_complete = proposal
            self._deadline = self._clock() + 60.0
        else:
            self._readback_complete = None
            self._evidence("readback_not_exact")

    def speech_started(self, proposal: TaskProposal | None = None) -> None:
        """Any other host/assistant speech invalidates the spoken binding."""
        if self._pending is None:
            return
        self._readback_complete = None
        self._capture_binding = None
        if proposal is None or self._readback_settled:
            self._evidence("intervening_speech")

    def _matches_binding(self, participant: str, generation: int) -> bool:
        return (
            type(participant) is str
            and type(generation) is int
            and self._binding == (participant, generation)
        )

    def capture_started(self, *, participant: str, generation: int, clean: bool) -> None:
        """Latch at captured speech onset; delayed STT can never upgrade this fact."""
        self._capture_binding = (
            self._pending
            if clean is True
            and self._readback_complete is self._pending
            and self._matches_binding(participant, generation)
            else None
        )

    def typed_input_started(self, *, participant: str, generation: int) -> None:
        # The exact full-text card was published synchronously by offer().
        self._capture_binding = (
            self._pending if self._matches_binding(participant, generation) else None
        )

    def intervening_input(self) -> None:
        self.retire("intervening_input")

    def confirmation_target(self, text: str) -> TaskProposal | None:
        """Freeze the exact pending object before user-input recording can suspend."""
        return self._pending if self.is_confirmation(text) else None

    async def consume_input(
        self,
        text: str,
        *,
        expected_proposal: TaskProposal,
    ) -> WorkCommandAdmission:
        proposal = self._pending
        if proposal is None or expected_proposal is not proposal or not self.is_confirmation(text):
            self._evidence("stale_confirmation_input")
            return WorkCommandAdmission.REFUSED
        capture = self._capture_binding
        self._capture_binding = None
        if self._deadline is None or self._clock() >= self._deadline:
            self.retire("expired")
            return WorkCommandAdmission.REFUSED
        if capture is not proposal:
            reason = (
                "use_confirmation_card"
                if self._readback_settled and self._readback_complete is not proposal
                else "spoken_confirmation_not_ready"
            )
            try:
                self._publish(proposal, "pending", reason=reason)
            finally:
                self._evidence(reason)
            return WorkCommandAdmission.REFUSED
        return await self.decide(
            participant=proposal.participant,
            generation=proposal.generation,
            proposal_id=proposal.proposal_id,
            objective=proposal.objective,
            decision="confirm",
        )

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def automatic_start(self) -> bool:
        return self._start_policy == "automatic"

    async def submit(self, request: HostTaskProposalRequest) -> TaskProposal | None:
        """Apply host authorization to a clean, current user-turn selection."""
        if not self.automatic_start:
            return self.offer(request)
        proposal = self._offer(request, visible=False)
        admission = await self._consume(proposal, decision=None)
        assert isinstance(admission, WorkCommandAdmission)
        if admission is WorkCommandAdmission.UNAVAILABLE:
            try:
                self._observer("task_state", {
                    "status": "rejected", "taskId": None,
                    "reason": "Task control is unavailable. "
                    "Restart the session before trying again.",
                })
            finally:
                self._evidence("automatic_unavailable")
        return None

    async def submit_for_ack(self, request: HostTaskProposalRequest) -> WorkStartResult | None:
        """Wait for the independently owned dispatch, not merely command intake."""
        if not self.automatic_start:
            raise RuntimeError("automatic start policy required")
        proposal = self._offer(request, visible=False)
        result = await self._consume(proposal, decision=None, wait_for_ack=True)
        if isinstance(result, WorkCommandAdmission):
            return None
        if not result.accepted and result.task_id is None and result.reason == _UNAVAILABLE_REASON:
            try:
                self._observer("task_state", {"status": "rejected", "taskId": None,
                    "reason": "Task control is unavailable. "
                    "Restart the session before trying again."})
            finally:
                self._evidence("automatic_unavailable")
        return result

    def offer(self, request: HostTaskProposalRequest) -> TaskProposal:
        return self._offer(request, visible=True)

    def _offer(self, request: HostTaskProposalRequest, *, visible: bool) -> TaskProposal:
        category = "invalid_proposal"
        try:
            if type(request) is not HostTaskProposalRequest:
                raise TypeError("proposal request must be exact")
            _validate_exact_text(request.objective, "objective", maximum=self.max_objective_chars)
            if self._binding is None:
                category = "no_binding"
                raise PermissionError("proposal has no current session binding")
            if self._pending is not None:
                category = "pending_proposal"
                raise PermissionError("another proposal is pending")
            if self._proposal_serial >= (1 << 53) - 1:
                category = "identity_capacity"
                raise PermissionError("proposal identity capacity exhausted")
            self._proposal_serial += 1
            proposal_id = f"proposal_{self._proposal_prefix}_{self._proposal_serial:014x}"
            proposal = TaskProposal(
                proposal_id, request.objective, *self._binding, self._clock() + 60.0
            )
            # Confirmed mode requires successful card projection before it is actionable.
            if visible:
                self._publish(proposal, "pending")
            self._pending = proposal
            self._pending_visible = visible
            self._deadline = proposal.expires_at
            self._readback_settled = False
            category = "offered" if visible else "automatic_selected"
            return proposal
        finally:
            self._evidence(category)

    async def decide(
        self,
        *,
        participant: str,
        generation: int,
        proposal_id: str,
        objective: str,
        decision: str,
    ) -> WorkCommandAdmission:
        category: str | None = "binding_refused"
        try:
            if type(decision) is not str or decision not in {"confirm", "decline"}:
                raise ValueError("proposal decision must be confirm or decline")
            proposal = self._pending
            if (
                proposal is None
                or type(participant) is not str
                or type(generation) is not int
                or type(proposal_id) is not str
                or type(objective) is not str
                or self._binding != (participant, generation)
                or proposal.proposal_id != proposal_id
                or proposal.objective != objective
            ):
                return WorkCommandAdmission.REFUSED
            category = None
            result = await self._consume(proposal, decision=decision)
            assert isinstance(result, WorkCommandAdmission)
            return result
        finally:
            if category is not None:
                self._evidence(category)

    async def _consume(
        self, proposal: TaskProposal, *, decision: str | None, wait_for_ack: bool = False,
    ) -> WorkCommandAdmission | WorkStartResult:
        category = "expired"
        try:
            # Both callers synchronously validated/created this exact pending binding.
            if self._deadline is None or self._clock() >= self._deadline:
                self.retire("expired")
                return WorkCommandAdmission.REFUSED
            # The same one-use consume precedes all projections and admission awaits.
            self._pending = None
            self._pending_visible = False
            self._readback_complete = None
            self._capture_binding = None
            self._deadline = None
            if decision is not None:
                self._publish(proposal, "confirmed" if decision == "confirm" else "declined")
            category = "automatic_consumed" if decision is None else (
                "consumed" if decision == "confirm" else "declined")
            if decision == "decline":
                return WorkCommandAdmission.REFUSED
            # Intake is never a started claim; existing admission owns the accepted ack.
            if wait_for_ack:
                if decision is not None:
                    raise ValueError("acknowledgment applies only to automatic starts")
                result = await self._surface.start_work(
                    objective=proposal.objective, invocation_id="automatic_" + proposal.proposal_id,
                )
                if type(result) is not WorkStartResult:
                    raise TypeError("work start result must be exact")
                return result
            return await self._surface.submit_start_command(
                objective=proposal.objective,
                invocation_id=("automatic_" if decision is None else "confirmed_")
                + proposal.proposal_id,
            )
        finally:
            self._evidence(category)

    def retire(self, category: str) -> None:
        proposal, self._pending = self._pending, None
        visible, self._pending_visible = self._pending_visible, False
        self._readback_complete = None
        self._capture_binding = None
        self._deadline = None
        if proposal is not None and visible:
            try:
                self._publish(proposal, "retired")
            finally:
                self._evidence(category)

    def _publish(self, proposal: TaskProposal, status: str, *, reason: str | None = None) -> None:
        self._observer(
            "task_proposal",
            {
                "proposalId": proposal.proposal_id,
                "objective": proposal.objective,
                "generation": proposal.generation,
                "status": status,
                "reason": reason,
            },
        )

    @staticmethod
    def _evidence(category: str) -> None:
        print(
            "[task-proposal] "
            + json.dumps({"category": category, "count": 1}, separators=(",", ":")),
            flush=True,
        )
