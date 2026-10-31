"""Internal host speech selection and read-only authoritative task projection."""

import json
from dataclasses import dataclass

from .context import _MAX_ACTIVE_TASKS_LIMIT


@dataclass(frozen=True, slots=True)
class HostSpeechRequest:
    """Select task status without carrying model text or effect arguments."""


@dataclass(frozen=True, slots=True)
class TaskStatusSnapshot:
    """Counts observed synchronously from the existing lifecycle owner."""

    available: bool = True
    pending: int = 0
    active: int = 0
    cancel_pending: int = 0
    cancelling: int = 0

    def __post_init__(self) -> None:
        try:
            if type(self.available) is not bool:
                raise TypeError("status availability must be exact boolean")
            counts = (self.pending, self.active, self.cancel_pending, self.cancelling)
            if any(type(count) is not int for count in counts):
                raise TypeError("status counts must be exact integers")
            if any(count < 0 for count in counts) or sum(counts) > _MAX_ACTIVE_TASKS_LIMIT:
                raise ValueError("status counts exceed the task capacity")
            if not self.available and any(counts):
                raise ValueError("unavailable status cannot assert task counts")
        except (TypeError, ValueError):
            try:
                raise
            finally:
                print(
                    "[host-status] "
                    + json.dumps({"category": "invalid_snapshot"}, separators=(",", ":")),
                    flush=True,
                )

    def render(self) -> str:
        if not self.available:
            return "I can't check background task status right now."
        parts = []
        if self.pending:
            if self.pending == 1:
                parts.append("One background task is awaiting acknowledgment.")
            else:
                parts.append(f"{self.pending} background tasks are awaiting acknowledgment.")
        if self.active:
            if self.active == 1:
                parts.append("One background task is running.")
            else:
                parts.append(f"{self.active} background tasks are running.")
        if self.cancel_pending:
            if self.cancel_pending == 1:
                parts.append("One task has a cancellation request awaiting acknowledgment.")
            else:
                parts.append(
                    f"{self.cancel_pending} tasks have cancellation requests "
                    "awaiting acknowledgment."
                )
        if self.cancelling:
            if self.cancelling == 1:
                parts.append(
                    "Cancellation has been acknowledged for one task; it is not finished yet."
                )
            else:
                parts.append(
                    "Cancellation has been acknowledged for "
                    f"{self.cancelling} tasks; they are not finished yet."
                )
        if not parts:
            return "No active background tasks in this session."
        return " ".join(parts)
