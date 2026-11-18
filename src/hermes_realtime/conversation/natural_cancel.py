"""Host-targeted natural cancellation: the model selects, the host names the task (#237).

With exactly one running task, no start awaiting acknowledgment and nothing already
cancelling, the host freezes that task when the user's final input is admitted. A selection
carries nothing; the host cancels the frozen task exactly, or refuses it as stale.

With several running tasks (owner decision 2026-11-18), the host freezes the exact set
instead and offers each task as a turn-local choice label. The model may only nominate one
label. A nomination never cancels anything: the host speaks its own readback of that exact
task, and only a fresh spoken affirmative of that completed readback cancels it, through the
same exact-task command path as the task card. Whether the user meant to cancel at all, and
which listed task they meant, remain the model's judgment (#205); the readback and the
user's answer, not that judgment, authorize the mutation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .work_tools import (
    _UNAVAILABLE_REASON,
    INACTIVE_TASK_REASON,
    WorkCancelResult,
    _objective_identity,
)

# Fixed host sentences for a selected cancel that was not accepted. Each must be true for
# every result mapped to it; none says a task stopped.
CANCEL_UNAVAILABLE = "Background task control is unavailable, so nothing was cancelled."
CANCEL_TARGET_ENDED = "That task is no longer running, so there was nothing to cancel."
CANCEL_NOT_ACCEPTED = "The request to cancel that task was not accepted."

# Beyond this many running tasks, or when two cannot be told apart, the task cards remain
# the way to stop one.
MAX_CANCEL_CHOICES = 4
_CHOICE_LABEL = re.compile(r"choice_[1-9]")


def cancel_choice_label(index: int) -> str:
    """The label of the one-based ``index``-th choice."""
    if type(index) is not int or not 1 <= index <= MAX_CANCEL_CHOICES:
        raise ValueError("cancel choice index is out of bounds")
    return f"choice_{index}"


def _validate_choice_label(value: object) -> None:
    if type(value) is not str:
        raise TypeError("cancel choice label must be an exact built-in string")
    if _CHOICE_LABEL.fullmatch(value) is None or int(value[7:]) > MAX_CANCEL_CHOICES:
        raise ValueError("cancel choice label is not a bounded host label")


@dataclass(frozen=True, slots=True, kw_only=True)
class HostCancelRequest:
    """A terminal provider selection of a host-frozen cancel.

    ``target`` is ``None`` for the sole-task route and otherwise one offered choice label;
    it never carries a task or run identity, and is never positional.
    """

    target: str | None = None

    def __post_init__(self) -> None:
        if self.target is not None:
            _validate_choice_label(self.target)


@dataclass(frozen=True, slots=True)
class CancelChoice:
    """One model-visible choice: a turn-local label and the task's stored objective."""

    label: str
    objective: str

    def __post_init__(self) -> None:
        _validate_choice_label(self.label)
        if type(self.objective) is not str or not self.objective.strip():
            raise ValueError("cancel choice objective must be nonblank exact text")


def distinguishable(texts: tuple[str, ...]) -> bool:
    """Whether every text differs from the others once case and spacing are ignored."""
    identities = [_objective_identity(text) for text in texts]
    return len(set(identities)) == len(identities)


class NaturalCancellation(Protocol):
    """The work surface, as the speech loop needs it for a natural cancel."""

    def natural_cancel_target(self) -> str | None: ...

    async def cancel_work(
        self, *, task_id: str, invocation_id: str, still_current: Callable[[], bool],
    ) -> WorkCancelResult: ...


def unaccepted_cancel_speech(result: WorkCancelResult) -> str | None:
    """Select one fixed sentence by typed result category; never echo result text."""
    if type(result) is not WorkCancelResult:
        raise TypeError("work cancel result must be exact")
    if result.accepted:
        return None
    if result.task_id is None and result.reason == _UNAVAILABLE_REASON:
        # Only the host's own refusal before any cancel request has this shape.
        return CANCEL_UNAVAILABLE
    if result.reason == INACTIVE_TASK_REASON:
        return CANCEL_TARGET_ENDED
    return CANCEL_NOT_ACCEPTED


__all__ = [
    "CANCEL_NOT_ACCEPTED",
    "CANCEL_TARGET_ENDED",
    "CANCEL_UNAVAILABLE",
    "MAX_CANCEL_CHOICES",
    "CancelChoice",
    "HostCancelRequest",
    "NaturalCancellation",
    "cancel_choice_label",
    "distinguishable",
    "unaccepted_cancel_speech",
]
