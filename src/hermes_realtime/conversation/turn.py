"""The one foreground turn a model adapter renders, and the routes it may offer.

Decided here, once, from the request and the host facts; a model adapter renders the
answer and never re-derives it from the request's type or its last row.
"""

from dataclasses import dataclass

from .context import _MAX_ITEM_CHARS_LIMIT, ConversationContextSnapshot, ConversationRole
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

    ``context`` and ``updates`` are rebuilt here, so what an adapter reads was validated
    by this type and not by the caller. ``accepted_objective`` is set only on the
    acknowledgment of an accepted start. ``admitted`` is true only for a user turn the
    streaming loop admitted, and only such a turn carries updates.
    """

    context: ConversationContextSnapshot
    updates: tuple[ConversationPromptUpdate, ...]
    accepted_objective: str | None
    admitted: bool

    def __post_init__(self) -> None:
        if type(self.context) is not ConversationContextSnapshot:
            raise TypeError("context must be an exact ConversationContextSnapshot")
        _require_bool("admitted", self.admitted)
        request = ConversationInferenceRequest(
            revision=self.context.revision,
            messages=self.context.messages,
            active_tasks=self.context.active_tasks,
            terminal_task_count=self.context.terminal_task_count,
            memory=self.context.memory,
            recent_task_results=self.context.recent_task_results,
            updates=self.updates,
        )
        if request.updates and not self.admitted:
            raise ValueError("only an admitted user turn carries updates")
        object.__setattr__(self, "context", request.context)
        object.__setattr__(self, "updates", request.updates)
        if self.accepted_objective is None:
            return
        if type(self.accepted_objective) is not str:
            raise TypeError("accepted_objective must be an exact built-in string or None")
        if (
            not self.accepted_objective.strip()
            or len(self.accepted_objective) > _MAX_ITEM_CHARS_LIMIT
        ):
            raise ValueError("accepted_objective must be bounded nonblank text")
        if self.admitted:
            raise ValueError("an acknowledgment is not an admitted user turn")

    @property
    def awaiting_reply(self) -> bool:
        """Whether the last heard row is the user's."""
        messages = self.context.messages
        return bool(messages) and messages[-1].role == ConversationRole.USER.value


def foreground_turn(snapshot: ConversationContextSnapshot) -> ForegroundTurn:
    """Turn one of the three exact request types into the turn an adapter renders.

    The purpose comes from the exact type; anything else, a subclass included, is refused.
    """

    admitted = False
    accepted_objective: str | None = None
    updates: tuple[ConversationPromptUpdate, ...] = ()
    if type(snapshot) is AcceptedStartInferenceRequest:
        # Rebuilt so a request altered after construction cannot name another objective.
        accepted = AcceptedStartInferenceRequest(
            revision=snapshot.revision, messages=snapshot.messages,
            active_tasks=snapshot.active_tasks,
            terminal_task_count=snapshot.terminal_task_count, memory=snapshot.memory,
            task_id=snapshot.task_id, recent_task_results=snapshot.recent_task_results,
        )
        accepted_objective = next(
            task.objective for task in accepted.active_tasks if task.task_id == accepted.task_id
        )
    elif type(snapshot) is ConversationInferenceRequest:
        admitted = True
        updates = snapshot.updates
    elif type(snapshot) is not ConversationContextSnapshot:
        raise TypeError(
            "snapshot must be an exact ConversationContextSnapshot or inference request"
        )
    return ForegroundTurn(
        context=ConversationContextSnapshot(
            revision=snapshot.revision,
            messages=snapshot.messages,
            active_tasks=snapshot.active_tasks,
            terminal_task_count=snapshot.terminal_task_count,
            memory=snapshot.memory,
            recent_task_results=snapshot.recent_task_results,
        ),
        updates=updates,
        accepted_objective=accepted_objective,
        admitted=admitted,
    )


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
