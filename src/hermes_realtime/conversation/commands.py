"""Deterministic explicit transcript commands for bounded task control."""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from typing import Protocol, overload

from hermes_realtime.evidence import (
    CommandAdmissionAuthorityV1,
    CommandDisposition,
    CommandRoutingOutcome,
    CommandRoutingResultV1,
    FinalInputAuthorityV1,
)
from hermes_realtime.evidence.lifecycle import EvidenceConversationAuthorityV1

from .context import ActiveTaskIdentityError
from .tasks import TaskCancelOutcome, TaskDispatchOutcome
from .work_tools import WorkCancelResult, WorkStartResult

_TASK_ID = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}\Z")
_COMMAND_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,110}\Z")
_PRIVATE_AUTHORITY = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*")
_MAX_OBJECTIVE_CHARS = 1024
_PublicValue = str | int | bool | None
_Observer = Callable[[str, dict[str, _PublicValue]], None]
_CommandAcceptedHook = Callable[[CommandAdmissionAuthorityV1], CommandDisposition]


class _TaskController(Protocol):
    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome: ...

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome: ...


class _WorkControlSurface(Protocol):
    @property
    def max_objective_chars(self) -> int: ...

    async def start_work(
        self,
        *,
        objective: str,
        invocation_id: str,
    ) -> WorkStartResult: ...

    async def cancel_work(
        self,
        *,
        task_id: str,
        invocation_id: str,
        reason: str = "user requested task cancellation",
    ) -> WorkCancelResult: ...


class ConversationTaskCommandRouter:
    """Consume only explicit task commands before foreground inference."""

    def __init__(
        self,
        *,
        observer: _Observer,
        reserve_observer_capacity: Callable[[], None],
        controller: _TaskController | None = None,
        utterance_id_factory: Callable[[], str] | None = None,
        surface: _WorkControlSurface | None = None,
        lifecycle_owner: EvidenceConversationAuthorityV1 | None = None,
        on_command_accepted: _CommandAcceptedHook | None = None,
    ) -> None:
        if surface is None and not callable(getattr(controller, "dispatch", None)):
            raise TypeError("controller must provide dispatch()")
        if surface is None and not callable(getattr(controller, "request_cancel", None)):
            raise TypeError("controller must provide request_cancel()")
        if surface is not None:
            if not callable(getattr(surface, "start_work", None)):
                raise TypeError("surface must provide start_work()")
            if not callable(getattr(surface, "cancel_work", None)):
                raise TypeError("surface must provide cancel_work()")
        if not callable(observer):
            raise TypeError("observer must be callable")
        if not callable(reserve_observer_capacity):
            raise TypeError("reserve_observer_capacity must be callable")
        if utterance_id_factory is not None and not callable(utterance_id_factory):
            raise TypeError("utterance_id_factory must be callable or None")
        if (
            lifecycle_owner is not None
            and type(lifecycle_owner) is not EvidenceConversationAuthorityV1
        ):
            raise TypeError("lifecycle_owner must be an exact conversation authority or None")
        if on_command_accepted is not None and not callable(on_command_accepted):
            raise TypeError("on_command_accepted must be callable or None")
        self._controller = controller
        self._observer = observer
        self._reserve = reserve_observer_capacity
        self._utterance_id_factory = utterance_id_factory or (lambda: secrets.token_hex(12))
        self._surface = surface
        self._lifecycle_owner = lifecycle_owner
        self._on_command_accepted = on_command_accepted

    @overload
    async def route(self, text: str) -> bool: ...

    @overload
    async def route(self, text: str, authority: None) -> bool: ...

    @overload
    async def route(
        self,
        text: str,
        authority: FinalInputAuthorityV1,
    ) -> CommandRoutingResultV1: ...

    async def route(
        self,
        text: str,
        authority: FinalInputAuthorityV1 | None = None,
    ) -> bool | CommandRoutingResultV1:
        """Route one explicit command and return whether it was consumed."""

        if type(text) is not str:
            raise TypeError("command text must be an exact built-in string")
        stripped = text.strip()
        folded = stripped.casefold()
        if folded.startswith("task:"):
            objective = stripped[len("task:") :].strip()
            if not self._valid_objective(objective):
                self._publish_invalid()
                return self._closed_nonaccepted_result(
                    authority,
                    CommandRoutingOutcome.INVALID,
                )
            token = self._utterance_id_factory()
            if type(token) is not str or _COMMAND_TOKEN.fullmatch(token) is None:
                raise RuntimeError("command utterance identifier factory returned an invalid value")
            if self._surface is not None:
                start_result = await self._surface.start_work(
                    objective=objective,
                    invocation_id=f"command_{token}",
                )
                if type(start_result) is not WorkStartResult:
                    raise TypeError("work surface returned the wrong start result")
                accepted_result = self._accepted_result(authority)
                return True if accepted_result is None else accepted_result
            self._reserve()
            controller = self._controller
            if controller is None:
                raise RuntimeError("task command router has no controller")
            dispatch_outcome = await controller.dispatch(
                objective=objective,
                utterance_id=f"utterance_{token}",
            )
            if type(dispatch_outcome) is not TaskDispatchOutcome:
                raise TypeError("task controller returned the wrong dispatch outcome")
            if dispatch_outcome.accepted:
                accepted_result = self._accepted_result(authority)
                try:
                    self._observer(
                        "task_state",
                        {"status": "active", "taskId": dispatch_outcome.task_id},
                    )
                except BaseException as projection_error:
                    try:
                        rollback = await controller.request_cancel(
                            dispatch_outcome.task_id,
                            reason="task state projection failed",
                        )
                        if not rollback.accepted:
                            raise RuntimeError("task projection rollback was rejected")
                    except BaseException as rollback_error:
                        raise BaseExceptionGroup(
                            "task state projection and rollback failed",
                            [projection_error, rollback_error],
                        ) from None
                    raise
            else:
                self._observer(
                    "task_state",
                    {
                        "reason": dispatch_outcome.reason or "Hermes rejected task dispatch",
                        "status": "rejected",
                        "taskId": dispatch_outcome.task_id,
                    },
                )
            if dispatch_outcome.accepted and accepted_result is not None:
                return accepted_result
            if not dispatch_outcome.accepted:
                return self._closed_nonaccepted_result(
                    authority,
                    CommandRoutingOutcome.REJECTED,
                )
            return True
        if folded.startswith("cancel task:"):
            task_id = stripped[len("cancel task:") :].strip()
            if _TASK_ID.fullmatch(task_id) is None or _PRIVATE_AUTHORITY.search(task_id):
                self._publish_invalid()
                return self._closed_nonaccepted_result(
                    authority,
                    CommandRoutingOutcome.INVALID,
                )
            if self._surface is not None:
                token = self._utterance_id_factory()
                if type(token) is not str or _COMMAND_TOKEN.fullmatch(token) is None:
                    raise RuntimeError(
                        "command invocation identifier factory returned an invalid value"
                    )
                cancel_result = await self._surface.cancel_work(
                    task_id=task_id,
                    invocation_id=f"command_{token}",
                    reason="user requested task cancellation",
                )
                if type(cancel_result) is not WorkCancelResult:
                    raise TypeError("work surface returned the wrong cancel result")
                if not cancel_result.accepted:
                    return self._closed_nonaccepted_result(
                        authority,
                        CommandRoutingOutcome.REJECTED,
                    )
                accepted_result = self._accepted_result(authority)
                return True if accepted_result is None else accepted_result
            self._reserve()
            controller = self._controller
            if controller is None:
                raise RuntimeError("task command router has no controller")
            try:
                cancel_outcome = await controller.request_cancel(
                    task_id,
                    reason="user requested task cancellation",
                )
            except ActiveTaskIdentityError:
                self._observer(
                    "task_state",
                    {"reason": "task is not active", "status": "rejected", "taskId": task_id},
                )
                return self._closed_nonaccepted_result(
                    authority,
                    CommandRoutingOutcome.REJECTED,
                )
            if type(cancel_outcome) is not TaskCancelOutcome:
                raise TypeError("task controller returned the wrong cancel outcome")
            if cancel_outcome.accepted:
                self._observer(
                    "task_state",
                    {"status": "cancelling", "taskId": cancel_outcome.task_id},
                )
                accepted_result = self._accepted_result(authority)
                return True if accepted_result is None else accepted_result
            self._observer(
                "task_state",
                {
                    "reason": cancel_outcome.reason or "Hermes rejected task cancellation",
                    "status": "rejected",
                    "taskId": cancel_outcome.task_id,
                },
            )
            return self._closed_nonaccepted_result(
                authority,
                CommandRoutingOutcome.REJECTED,
            )
        if authority is None:
            return False
        owner = self._lifecycle_owner
        if owner is None:
            raise RuntimeError("command router has no evidence lifecycle owner")
        return CommandRoutingResultV1(
            outcome=CommandRoutingOutcome.NOT_COMMAND,
            command_disposition=None,
            user_turn_authority=owner.decline_to_user(authority),
        )

    def _accepted_result(
        self,
        authority: FinalInputAuthorityV1 | None,
    ) -> CommandRoutingResultV1 | None:
        if authority is None:
            return None
        owner = self._lifecycle_owner
        hook = self._on_command_accepted
        if owner is None or hook is None:
            raise RuntimeError("accepted command evidence is not configured")
        disposition = hook(owner.accept_command(authority))
        if type(disposition) is not CommandDisposition:
            raise TypeError("command evidence hook returned the wrong disposition")
        return CommandRoutingResultV1(
            outcome=CommandRoutingOutcome.ACCEPTED,
            command_disposition=disposition,
            user_turn_authority=None,
        )

    def _closed_nonaccepted_result(
        self,
        authority: FinalInputAuthorityV1 | None,
        outcome: CommandRoutingOutcome,
    ) -> bool | CommandRoutingResultV1:
        if authority is None:
            return True
        owner = self._lifecycle_owner
        if owner is None:
            raise RuntimeError("command router has no evidence lifecycle owner")
        owner.retire_final_input(authority)
        return CommandRoutingResultV1(
            outcome=outcome,
            command_disposition=None,
            user_turn_authority=None,
        )

    def _valid_objective(self, objective: str) -> bool:
        maximum = (
            self._surface.max_objective_chars if self._surface is not None else _MAX_OBJECTIVE_CHARS
        )
        return bool(
            objective and len(objective) <= maximum and _PRIVATE_AUTHORITY.search(objective) is None
        )

    def _publish_invalid(self) -> None:
        self._reserve()
        self._observer(
            "task_state",
            {
                "reason": "invalid explicit task command",
                "status": "rejected",
                "taskId": None,
            },
        )


__all__ = ["ConversationTaskCommandRouter"]
