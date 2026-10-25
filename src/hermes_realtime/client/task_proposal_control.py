"""Small exact public proposal contract shared by projection and control intake."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from hermes_realtime.conversation.task_proposals import MAX_TASK_PROPOSAL_OBJECTIVE_CHARS


def require_proposal_fields(proposal_id: object, objective: object, generation: object) -> None:
    if (
        type(proposal_id) is not str
        or re.fullmatch(r"proposal_[A-Za-z0-9_-]{8,64}", proposal_id) is None
    ):
        raise ValueError("proposal identity is invalid")
    if (
        type(objective) is not str
        or not objective.strip()
        or len(objective) > MAX_TASK_PROPOSAL_OBJECTIVE_CHARS
    ):
        raise ValueError("proposal objective is invalid")
    if type(generation) is not int or not 1 <= generation < 2**53:
        raise ValueError("proposal generation is invalid")


@dataclass(frozen=True, slots=True)
class TaskProposalDecision:
    proposal_id: str
    objective: str
    generation: int
    decision: str
    sequence: int

    def __post_init__(self) -> None:
        require_proposal_fields(self.proposal_id, self.objective, self.generation)
        if type(self.decision) is not str or self.decision not in {"confirm", "decline"}:
            raise ValueError("proposal decision is invalid")
        if type(self.sequence) is not int or not 1 <= self.sequence < 2**53:
            raise ValueError("proposal sequence is invalid")


def validate_proposal_event(data: Mapping[str, object]) -> None:
    if set(data) != {"proposalId", "objective", "generation", "status", "reason"}:
        raise ValueError("proposal event shape is invalid")
    require_proposal_fields(data["proposalId"], data["objective"], data["generation"])
    if type(data["status"]) is not str or data["status"] not in {
        "pending",
        "confirmed",
        "declined",
        "retired",
    }:
        raise ValueError("proposal status is invalid")
    if data["reason"] not in {None, "spoken_confirmation_not_ready", "use_confirmation_card"}:
        raise ValueError("proposal reason is invalid")
