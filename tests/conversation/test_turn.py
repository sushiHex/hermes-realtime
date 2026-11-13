from itertools import product
from typing import Any, cast

import pytest

from hermes_realtime.conversation.context import (
    ActiveTaskSummary,
    ConversationContextSnapshot,
    ConversationMessage,
)
from hermes_realtime.conversation.streaming import (
    AcceptedStartInferenceRequest,
    ConversationInferenceRequest,
)
from hermes_realtime.conversation.turn import (
    ForegroundTurn,
    TurnRoutes,
    foreground_turn,
    turn_routes,
)

_USER = ConversationMessage("user", "Inspect the release.")
_REPLY = ConversationMessage("assistant", "On it.")
_TASKS = (
    ActiveTaskSummary("task_other", "Another objective."),
    ActiveTaskSummary("task_public", "Inspect the release."),
)


def _acknowledgment(**changes: Any) -> AcceptedStartInferenceRequest:
    fields: dict[str, Any] = {
        "revision": 1, "messages": (_USER,), "active_tasks": _TASKS, "task_id": "task_public",
    }
    return AcceptedStartInferenceRequest(**{**fields, **changes})


def test_purpose_comes_from_the_exact_request_type() -> None:
    admitted = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    plain = ConversationContextSnapshot(revision=1, messages=(_USER,), active_tasks=())

    assert foreground_turn(admitted) == ForegroundTurn(
        accepted_objective=None, admitted=True, awaiting_reply=True,
    )
    assert foreground_turn(plain) == ForegroundTurn(
        accepted_objective=None, admitted=False, awaiting_reply=True,
    )
    assert foreground_turn(_acknowledgment()) == ForegroundTurn(
        accepted_objective="Inspect the release.", admitted=False, awaiting_reply=True,
    )


def test_a_subclass_earns_neither_admission_nor_acknowledgment() -> None:
    class ForgedRequest(ConversationInferenceRequest):
        pass

    class ForgedAcknowledgment(AcceptedStartInferenceRequest):
        pass

    least = ForegroundTurn(accepted_objective=None, admitted=False, awaiting_reply=True)

    assert foreground_turn(
        ForgedRequest(revision=1, messages=(_USER,), active_tasks=()),
    ) == least
    assert foreground_turn(ForgedAcknowledgment(
        revision=1, messages=(_USER,), active_tasks=_TASKS, task_id="task_public",
    )) == least


def test_awaiting_reply_is_true_only_when_the_last_heard_row_is_the_user() -> None:
    def awaiting(*messages: ConversationMessage) -> bool:
        return foreground_turn(ConversationInferenceRequest(
            revision=1, messages=messages, active_tasks=(),
        )).awaiting_reply

    assert awaiting(_USER) is True
    assert awaiting(_REPLY, _USER) is True
    assert awaiting() is False
    assert awaiting(_USER, _REPLY) is False


def test_request_that_is_not_a_snapshot_or_was_altered_is_refused() -> None:
    with pytest.raises(TypeError, match="snapshot"):
        foreground_turn(cast(ConversationContextSnapshot, {"messages": (_USER,)}))

    altered_rows = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    object.__setattr__(altered_rows, "messages", [_USER])
    with pytest.raises(TypeError, match="messages"):
        foreground_turn(altered_rows)

    forged_row = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    object.__setattr__(forged_row, "messages", ({"role": "user"},))
    with pytest.raises(TypeError, match="messages"):
        foreground_turn(forged_row)

    retargeted = _acknowledgment()
    object.__setattr__(retargeted, "task_id", "task_absent")
    with pytest.raises(ValueError, match="one active task"):
        foreground_turn(retargeted)


def test_turn_facts_are_exact_and_consistent() -> None:
    with pytest.raises(TypeError):
        ForegroundTurn(accepted_objective=None, admitted=cast(bool, 1), awaiting_reply=True)
    with pytest.raises(TypeError):
        ForegroundTurn(accepted_objective=None, admitted=True, awaiting_reply=cast(bool, "yes"))
    with pytest.raises(TypeError):
        ForegroundTurn(accepted_objective=cast(str, b"x"), admitted=False, awaiting_reply=True)
    with pytest.raises(ValueError):
        ForegroundTurn(accepted_objective="   ", admitted=False, awaiting_reply=True)
    with pytest.raises(ValueError, match="acknowledgment"):
        ForegroundTurn(accepted_objective="Inspect.", admitted=True, awaiting_reply=True)
    with pytest.raises(TypeError):
        TurnRoutes(start_work=cast(bool, 1), task_status=False)
    with pytest.raises(TypeError):
        TurnRoutes(start_work=False, task_status=cast(bool, None))


_OPEN = ForegroundTurn(accepted_objective=None, admitted=True, awaiting_reply=True)
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
            ForegroundTurn(accepted_objective=None, admitted=True, awaiting_reply=False), {},
            TurnRoutes(start_work=False, task_status=False), id="last row is not the user",
        ),
        pytest.param(
            ForegroundTurn(accepted_objective="Inspect.", admitted=False, awaiting_reply=True), {},
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
            ForegroundTurn(accepted_objective=None, admitted=False, awaiting_reply=True), {},
            TurnRoutes(start_work=True, task_status=False),
            id="unadmitted request keeps the start where admission is not required",
        ),
        pytest.param(
            ForegroundTurn(accepted_objective=None, admitted=False, awaiting_reply=True),
            {"start_requires_admission": True},
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
        routes = turn_routes(
            ForegroundTurn(
                accepted_objective=objective, admitted=admitted, awaiting_reply=awaiting,
            ),
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
