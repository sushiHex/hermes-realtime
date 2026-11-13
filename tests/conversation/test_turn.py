from itertools import product
from typing import Any, cast

import pytest

from hermes_realtime.conversation.context import (
    ActiveTaskSummary,
    ConversationContextSnapshot,
    ConversationMessage,
    RecentTaskResult,
)
from hermes_realtime.conversation.streaming import (
    AcceptedStartInferenceRequest,
    ConversationInferenceRequest,
    ConversationPromptUpdate,
)
from hermes_realtime.conversation.turn import (
    ForegroundTurn,
    TurnRoutes,
    foreground_turn,
    turn_routes,
)
from hermes_realtime.memory import BuiltinMemorySnapshot

_USER = ConversationMessage("user", "Inspect the release.")
_REPLY = ConversationMessage("assistant", "On it.")
_TASKS = (
    ActiveTaskSummary("task_other", "Another objective."),
    ActiveTaskSummary("task_public", "Inspect the release."),
)
_UPDATE = ConversationPromptUpdate(
    sequence=1, task_id="task_done", status="completed", text="Synthetic finding.",
)
_REFERENCE: dict[str, Any] = {
    "revision": 7,
    "messages": (_USER,),
    "active_tasks": _TASKS,
    "terminal_task_count": 2,
    "memory": BuiltinMemorySnapshot(memory="Synthetic memory.", user="Synthetic user."),
    "recent_task_results": (
        RecentTaskResult("task_done", "Earlier objective.", "completed", "Earlier result."),
    ),
}


def _context(*messages: ConversationMessage) -> ConversationContextSnapshot:
    return ConversationContextSnapshot(revision=1, messages=messages, active_tasks=())


def _turn(
    *messages: ConversationMessage, objective: str | None = None, admitted: bool = False,
) -> ForegroundTurn:
    return ForegroundTurn(
        context=_context(*messages), updates=(), accepted_objective=objective, admitted=admitted,
    )


def test_each_exact_request_type_becomes_the_one_turn_with_its_purpose() -> None:
    context = ConversationContextSnapshot(**_REFERENCE)

    admitted = foreground_turn(ConversationInferenceRequest(**_REFERENCE, updates=(_UPDATE,)))
    plain = foreground_turn(context)
    acknowledgment = foreground_turn(
        AcceptedStartInferenceRequest(**_REFERENCE, task_id="task_public"),
    )

    assert admitted == ForegroundTurn(
        context=context, updates=(_UPDATE,), accepted_objective=None, admitted=True,
    )
    assert plain == ForegroundTurn(
        context=context, updates=(), accepted_objective=None, admitted=False,
    )
    assert acknowledgment == ForegroundTurn(
        context=context, updates=(), accepted_objective="Inspect the release.", admitted=False,
    )
    for turn in (admitted, plain, acknowledgment):
        assert type(turn.context) is ConversationContextSnapshot
        assert turn.context is not context


@pytest.mark.parametrize("base", [
    ConversationContextSnapshot, ConversationInferenceRequest, AcceptedStartInferenceRequest,
])
def test_a_request_subclass_is_refused(base: type[ConversationContextSnapshot]) -> None:
    class Forged(base):  # type: ignore[valid-type, misc]
        pass

    fields = {"revision": 1, "messages": (_USER,), "active_tasks": _TASKS}
    if base is AcceptedStartInferenceRequest:
        fields["task_id"] = "task_public"
    with pytest.raises(TypeError, match="exact ConversationContextSnapshot or inference request"):
        foreground_turn(Forged(**fields))


def test_a_request_that_is_not_a_snapshot_or_was_altered_is_refused() -> None:
    with pytest.raises(TypeError, match="exact ConversationContextSnapshot or inference request"):
        foreground_turn(cast(ConversationContextSnapshot, {"messages": (_USER,)}))

    altered_rows = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    object.__setattr__(altered_rows, "messages", [_USER])
    with pytest.raises(TypeError, match="messages"):
        foreground_turn(altered_rows)

    altered_updates = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    object.__setattr__(altered_updates, "updates", ({"sequence": 1},))
    with pytest.raises(TypeError, match="ConversationPromptUpdate"):
        foreground_turn(altered_updates)

    retargeted = AcceptedStartInferenceRequest(**_REFERENCE, task_id="task_public")
    object.__setattr__(retargeted, "task_id", "task_absent")
    with pytest.raises(ValueError, match="one active task"):
        foreground_turn(retargeted)


def test_the_turn_rebuilds_and_bounds_what_it_holds() -> None:
    context = _context(_USER)
    turn = ForegroundTurn(
        context=context, updates=(_UPDATE,), accepted_objective=None, admitted=True,
    )
    assert turn.context == context and turn.context is not context
    assert turn.updates == (_UPDATE,) and turn.updates[0] is not _UPDATE

    with pytest.raises(TypeError, match="exact ConversationContextSnapshot"):
        ForegroundTurn(
            context=ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=()),
            updates=(), accepted_objective=None, admitted=True,
        )
    with pytest.raises(TypeError, match="exact tuple"):
        ForegroundTurn(
            context=context, updates=cast(tuple[ConversationPromptUpdate, ...], [_UPDATE]),
            accepted_objective=None, admitted=True,
        )
    with pytest.raises(ValueError, match="capacity"):
        ForegroundTurn(
            context=context, updates=(_UPDATE,) * 4096, accepted_objective=None, admitted=True,
        )
    with pytest.raises(ValueError, match="admitted user turn"):
        ForegroundTurn(
            context=context, updates=(_UPDATE,), accepted_objective=None, admitted=False,
        )
    with pytest.raises(TypeError):
        ForegroundTurn(
            context=context, updates=(), accepted_objective=None, admitted=cast(bool, 1),
        )
    with pytest.raises(TypeError):
        ForegroundTurn(
            context=context, updates=(), accepted_objective=cast(str, b"x"), admitted=False,
        )
    with pytest.raises(ValueError):
        ForegroundTurn(context=context, updates=(), accepted_objective="   ", admitted=False)
    with pytest.raises(ValueError, match="acknowledgment"):
        ForegroundTurn(context=context, updates=(), accepted_objective="Inspect.", admitted=True)
    with pytest.raises(TypeError):
        TurnRoutes(start_work=cast(bool, 1), task_status=False)
    with pytest.raises(TypeError):
        TurnRoutes(start_work=False, task_status=cast(bool, None))


def test_awaiting_reply_is_true_only_when_the_last_heard_row_is_the_user() -> None:
    assert _turn(_USER).awaiting_reply is True
    assert _turn(_REPLY, _USER).awaiting_reply is True
    assert _turn().awaiting_reply is False
    assert _turn(_USER, _REPLY).awaiting_reply is False


_OPEN = _turn(_USER, admitted=True)
_FACTS = {
    "work_bound": True, "private_material": False, "lookup_attempted": False,
    "start_requires_admission": False,
}


def test_an_admitted_user_turn_with_bound_work_is_offered_both_routes() -> None:
    assert turn_routes(_OPEN, **_FACTS) == TurnRoutes(start_work=True, task_status=True)


@pytest.mark.parametrize(
    ("turn", "facts", "expected"),
    [
        pytest.param(
            _OPEN, {"work_bound": False}, TurnRoutes(start_work=False, task_status=False),
            id="no work handler",
        ),
        pytest.param(
            _turn(_USER, _REPLY, admitted=True), {},
            TurnRoutes(start_work=False, task_status=False), id="last row is not the user",
        ),
        pytest.param(
            _turn(_USER, objective="Inspect."), {},
            TurnRoutes(start_work=False, task_status=False), id="acknowledgment",
        ),
        pytest.param(
            _OPEN, {"private_material": True}, TurnRoutes(start_work=False, task_status=False),
            id="private material",
        ),
        pytest.param(
            _OPEN, {"lookup_attempted": True}, TurnRoutes(start_work=False, task_status=True),
            id="a lookup withdraws only the start",
        ),
        pytest.param(
            _turn(_USER), {}, TurnRoutes(start_work=True, task_status=False),
            id="unadmitted request keeps the start where admission is not required",
        ),
        pytest.param(
            _turn(_USER), {"start_requires_admission": True},
            TurnRoutes(start_work=False, task_status=False),
            id="unadmitted request loses the start where admission is required",
        ),
    ],
)
def test_each_withdrawal_closes_exactly_its_routes(
    turn: ForegroundTurn, facts: dict[str, bool], expected: TurnRoutes,
) -> None:
    assert turn_routes(turn, **{**_FACTS, **facts}) == expected


def test_no_route_is_offered_without_bound_work_or_a_user_row_to_answer() -> None:
    for objective, admitted, awaiting, bound, private, attempted, required in product(
        (None, "Inspect."), (False, True), (False, True), (False, True), (False, True),
        (False, True), (False, True),
    ):
        if objective is not None and admitted:
            continue
        rows = (_USER,) if awaiting else (_USER, _REPLY)
        routes = turn_routes(
            _turn(*rows, objective=objective, admitted=admitted),
            work_bound=bound, private_material=private, lookup_attempted=attempted,
            start_requires_admission=required,
        )
        if not bound or not awaiting or objective is not None or private:
            assert routes == TurnRoutes(start_work=False, task_status=False)
        if routes.task_status:
            assert admitted


@pytest.mark.parametrize(
    "name", ["work_bound", "private_material", "lookup_attempted", "start_requires_admission"],
)
def test_route_facts_must_be_exact_booleans(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        turn_routes(_OPEN, **{**_FACTS, name: cast(bool, 1)})
    with pytest.raises(TypeError, match="ForegroundTurn"):
        turn_routes(cast(ForegroundTurn, object()), **_FACTS)
