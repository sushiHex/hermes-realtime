"""What one foreground turn is and which routes it may offer.

Decided here, once, from the request and the host facts; a model adapter renders the
answer and never re-derives it from the request's type or its last row.
"""

from dataclasses import dataclass

from .context import (
    _MAX_ITEM_CHARS_LIMIT,
    ConversationContextSnapshot,
    ConversationMessage,
    ConversationRole,
)
from .streaming import AcceptedStartInferenceRequest, ConversationInferenceRequest


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")


@dataclass(frozen=True, slots=True)
class ForegroundTurn:
    """What the request handed to a model adapter says about its turn.

    ``accepted_objective`` is set only on the acknowledgment of an accepted start.
    ``admitted`` is true only for a user turn the streaming loop admitted.
    ``awaiting_reply`` is true only when the last heard row is the user's.
    """

    accepted_objective: str | None
    admitted: bool
    awaiting_reply: bool

    def __post_init__(self) -> None:
        _require_bool("admitted", self.admitted)
        _require_bool("awaiting_reply", self.awaiting_reply)
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


def foreground_turn(snapshot: ConversationContextSnapshot) -> ForegroundTurn:
    """Classify one request by its exact type and its own rows."""

    # A subclass is deliberately not refused here: it is classified as a plain snapshot,
    # which earns neither admission nor an acknowledgment.
    if not isinstance(snapshot, ConversationContextSnapshot):
        raise TypeError("snapshot must be a ConversationContextSnapshot")
    accepted_objective: str | None = None
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
    messages = snapshot.messages
    if type(messages) is not tuple or (
        messages and type(messages[-1]) is not ConversationMessage
    ):
        raise TypeError("snapshot messages must be an exact tuple of ConversationMessage values")
    return ForegroundTurn(
        accepted_objective=accepted_objective,
        admitted=type(snapshot) is ConversationInferenceRequest,
        awaiting_reply=bool(messages) and messages[-1].role == ConversationRole.USER.value,
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
