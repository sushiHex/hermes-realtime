"""One bounded, host-owned proposal ahead of the existing work admission surface.

The provider's selection is data. Only an exact confirmation by the current
participant/session binding can consume it and call ordinary command admission.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .work_tools import WorkCommandAdmission, _validate_exact_text, _validate_identifier

_PROPOSAL_ID = re.compile(r"proposal_[A-Za-z0-9_-]{8,64}\Z")
_PublicValue = str | int | bool | None


@dataclass(frozen=True, slots=True)
class HostTaskProposalRequest:
    """An effect-free, terminal provider selection; not an admission receipt."""

    objective: str
    invocation_id: str

    def __post_init__(self) -> None:
        _validate_exact_text(self.objective, "objective", maximum=65_536)
        _validate_identifier(self.invocation_id, "invocation_id")


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
        return "Proposed background task. " + self.objective + " End of proposed task."


class _Surface(Protocol):
    @property
    def max_objective_chars(self) -> int: ...

    async def submit_start_command(
        self, *, objective: str, invocation_id: str,
    ) -> WorkCommandAdmission: ...


class ConversationTaskProposals:
    """Session-scoped proposal state, independent of foreground turn leases and tasks."""

    def __init__(
        self, *, surface: _Surface,
        observer: Callable[[str, dict[str, _PublicValue]], None],
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = lambda: "proposal_" + secrets.token_hex(12),
    ) -> None:
        self._surface = surface
        self._observer = observer
        self._clock = clock
        self._id_factory = id_factory
        self._binding: tuple[str, int] | None = None
        self._pending: TaskProposal | None = None
        self._seen_invocations: set[str] = set()
        self._seen_ids: set[str] = set()

    @property
    def max_objective_chars(self) -> int:
        return self._surface.max_objective_chars

    def bind(self, participant: str, generation: int) -> None:
        if type(participant) is not str or not participant or len(participant) > 256:
            raise ValueError("proposal participant must be a bounded exact string")
        if type(generation) is not int or not 1 <= generation < 2**53:
            raise ValueError("proposal generation must be a positive browser-safe integer")
        self.retire("binding_changed")
        self._binding = participant, generation

    def close_binding(self) -> None:
        self.retire("binding_closed")
        self._binding = None

    def offer(self, request: HostTaskProposalRequest) -> TaskProposal:
        category = "invalid_proposal"
        try:
            if type(request) is not HostTaskProposalRequest:
                raise TypeError("proposal request must be exact")
            _validate_exact_text(request.objective, "objective", maximum=self.max_objective_chars)
            if self._binding is None:
                category = "no_binding"
                raise PermissionError("proposal has no current session binding")
            if request.invocation_id in self._seen_invocations:
                category = "replay"
                raise PermissionError("proposal invocation was already consumed")
            if len(self._seen_invocations) >= 256:
                category = "capacity"
                raise PermissionError("proposal invocation capacity exhausted")
            # Reserving the invocation does not dispatch or grant any authority.
            self._seen_invocations.add(request.invocation_id)
            if self._pending is not None:
                category = "pending_proposal"
                raise PermissionError("another proposal is pending")
            proposal_id = self._id_factory()
            if (type(proposal_id) is not str or _PROPOSAL_ID.fullmatch(proposal_id) is None
                    or proposal_id in self._seen_ids):
                category = "identity"
                raise ValueError("proposal identity is invalid or reused")
            self._seen_ids.add(proposal_id)
            proposal = TaskProposal(proposal_id, request.objective, *self._binding,
                                    self._clock() + 60.0)
            # The card must be projected successfully before becoming actionable.
            self._publish(proposal, "pending")
            self._pending = proposal
            category = "offered"
            return proposal
        finally:
            self._evidence(category)

    async def decide(
        self, *, participant: str, generation: int, proposal_id: str,
        objective: str, decision: str,
    ) -> WorkCommandAdmission:
        category = "binding_refused"
        try:
            if type(decision) is not str or decision not in {"confirm", "decline"}:
                raise ValueError("proposal decision must be confirm or decline")
            proposal = self._pending
            if (proposal is None or type(participant) is not str or type(generation) is not int
                    or type(proposal_id) is not str or type(objective) is not str
                    or self._binding != (participant, generation)
                    or (proposal.participant, proposal.generation) != (participant, generation)
                    or proposal.proposal_id != proposal_id or proposal.objective != objective):
                raise PermissionError("confirmation does not match the exact current proposal")
            if self._clock() >= proposal.expires_at:
                category = "expired"
                self.retire("expired")
                raise PermissionError("proposal has expired")
            # Consume synchronously before projection or admission can suspend/fail.
            self._pending = None
            self._publish(proposal, "confirmed" if decision == "confirm" else "declined")
            category = "consumed" if decision == "confirm" else "declined"
            if decision == "decline":
                return WorkCommandAdmission.REFUSED
            # Admission owns dispatch and validated acknowledgment exactly as for
            # explicit commands. This return value never reports a task started.
            return await self._surface.submit_start_command(
                objective=proposal.objective, invocation_id="confirmed_" + proposal.proposal_id,
            )
        finally:
            self._evidence(category)

    def retire(self, category: str) -> None:
        proposal, self._pending = self._pending, None
        if proposal is not None:
            try:
                self._publish(proposal, "retired")
            finally:
                self._evidence(category)

    def _publish(self, proposal: TaskProposal, status: str) -> None:
        self._observer("task_proposal", {
            "proposalId": proposal.proposal_id, "objective": proposal.objective,
            "generation": proposal.generation, "status": status,
        })

    @staticmethod
    def _evidence(category: str) -> None:
        print("[task-proposal] " + json.dumps(
            {"category": category, "count": 1}, separators=(",", ":")), flush=True)
