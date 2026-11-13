"""The one foreground turn a model adapter renders, and the routes it may offer.

Decided here, once, from the request and the host facts; a model adapter renders the
answer and never re-derives it from the request's type or its last row.
"""

from dataclasses import dataclass

from .context import ConversationContextSnapshot, ConversationRole
from .streaming import (
    AcceptedStartInferenceRequest,
    ConversationInferenceRequest,
    ConversationPromptUpdate,
)


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")


@dataclass(frozen=True, slots=True)
class ForegroundTurn:
    """One request for foreground inference, in the only form an adapter renders.

    Built from one of the three exact request types; anything else, a subclass included,
    is refused. ``request`` is rebuilt here, once, so what an adapter reads was validated
    by this type and not by the caller. Everything else is read from that one copy, so a
    purpose cannot be claimed apart from the request that carries it.
    """

    request: ConversationContextSnapshot

    def __post_init__(self) -> None:
        request = self.request
        rebuilt: ConversationContextSnapshot
        if type(request) is AcceptedStartInferenceRequest:
            rebuilt = AcceptedStartInferenceRequest(
                revision=request.revision, messages=request.messages,
                active_tasks=request.active_tasks,
                terminal_task_count=request.terminal_task_count, memory=request.memory,
                recent_task_results=request.recent_task_results, task_id=request.task_id,
            )
        elif type(request) is ConversationInferenceRequest:
            rebuilt = ConversationInferenceRequest(
                revision=request.revision, messages=request.messages,
                active_tasks=request.active_tasks,
                terminal_task_count=request.terminal_task_count, memory=request.memory,
                recent_task_results=request.recent_task_results, updates=request.updates,
            )
        elif type(request) is ConversationContextSnapshot:
            rebuilt = ConversationContextSnapshot(
                revision=request.revision, messages=request.messages,
                active_tasks=request.active_tasks,
                terminal_task_count=request.terminal_task_count, memory=request.memory,
                recent_task_results=request.recent_task_results,
            )
        else:
            raise TypeError(
                "request must be an exact ConversationContextSnapshot or inference request"
            )
        object.__setattr__(self, "request", rebuilt)

    @property
    def admitted(self) -> bool:
        """Whether this is a user turn the streaming loop admitted."""
        return type(self.request) is ConversationInferenceRequest

    @property
    def updates(self) -> tuple[ConversationPromptUpdate, ...]:
        """Proactive updates claimed for this turn; only an admitted user turn has any."""
        request = self.request
        return request.updates if type(request) is ConversationInferenceRequest else ()

    @property
    def accepted_objective(self) -> str | None:
        """The accepted objective, on the acknowledgment of an accepted start only."""
        request = self.request
        if type(request) is not AcceptedStartInferenceRequest:
            return None
        return next(
            task.objective for task in request.active_tasks if task.task_id == request.task_id
        )

    @property
    def awaiting_reply(self) -> bool:
        """Whether the last heard row is the user's."""
        messages = self.request.messages
        return bool(messages) and messages[-1].role == ConversationRole.USER.value


@dataclass(frozen=True, slots=True)
class TurnRoutes:
    """The work routes the host offers the model on one turn; an adapter advertises these."""

    start_work: bool
    task_status: bool

    def __post_init__(self) -> None:
        _require_bool("start_work", self.start_work)
        _require_bool("task_status", self.task_status)


def turn_routes(
    turn: ForegroundTurn,
    *,
    work_bound: bool,
    private_material: bool,
    lookup_attempted: bool,
    start_requires_admission: bool,
) -> TurnRoutes:
    """Decide the work routes for one turn.

    Both routes need a bound work handler, a user row to answer, a turn that is not an
    acknowledgment, and no private material. A lookup already attempted for the turn
    withdraws only the start. The status route also needs an admitted user turn.

    ``start_requires_admission`` records the one point on which the two adapters differ
    today: whether an unadmitted request may still be offered the start.
    """

    if type(turn) is not ForegroundTurn:
        raise TypeError("turn must be an exact ForegroundTurn")
    _require_bool("work_bound", work_bound)
    _require_bool("private_material", private_material)
    _require_bool("lookup_attempted", lookup_attempted)
    _require_bool("start_requires_admission", start_requires_admission)
    offered = (
        work_bound
        and turn.awaiting_reply
        and turn.accepted_objective is None
        and not private_material
    )
    return TurnRoutes(
        start_work=(
            offered
            and not lookup_attempted
            and (turn.admitted or not start_requires_admission)
        ),
        task_status=offered and turn.admitted,
    )
