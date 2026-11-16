"""Host-targeted natural cancellation: the model selects, the host names the task (#237).

The host offers the route only when exactly one task is running and no start awaits its
acknowledgment, and freezes that task when the user's final input is admitted. A selection
carries nothing; the host cancels the frozen task exactly, or refuses it as stale. The model
never names a task, so it cannot cancel the wrong one. Whether the user meant to cancel at
all remains the model's judgment (#205).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .work_tools import _UNAVAILABLE_REASON, INACTIVE_TASK_REASON, WorkCancelResult

# Fixed host sentences for a selected cancel that was not accepted. Each must be true for
# every result mapped to it; none says a task stopped.
CANCEL_UNAVAILABLE = "Background task control is unavailable, so nothing was cancelled."
CANCEL_TARGET_ENDED = "That task is no longer running, so there was nothing to cancel."
CANCEL_NOT_ACCEPTED = "The request to cancel that task was not accepted."


@dataclass(frozen=True, slots=True)
class HostCancelRequest:
    """A terminal provider selection of the host-frozen cancel; it names no task."""


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
    "HostCancelRequest",
    "NaturalCancellation",
    "unaccepted_cancel_speech",
]
