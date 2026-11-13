from itertools import product
from typing import Any, cast

import pytest

import hermes_realtime.conversation.context as context_module
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
from hermes_realtime.conversation.turn import ForegroundTurn, TurnRoutes, turn_routes
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


def _plain(*messages: ConversationMessage) -> ForegroundTurn:
    return ForegroundTurn(
        ConversationContextSnapshot(revision=1, messages=messages, active_tasks=()),
    )


def _admitted(*messages: ConversationMessage) -> ForegroundTurn:
    return ForegroundTurn(
        ConversationInferenceRequest(revision=1, messages=messages, active_tasks=()),
    )


def _acknowledgment(*messages: ConversationMessage) -> ForegroundTurn:
    return ForegroundTurn(AcceptedStartInferenceRequest(
        revision=1, messages=messages, active_tasks=_TASKS, task_id="task_public",
    ))


def test_each_exact_request_type_becomes_a_turn_with_its_purpose() -> None:
    request = ConversationInferenceRequest(**_REFERENCE, updates=(_UPDATE,))
    plain = ConversationContextSnapshot(**_REFERENCE)
    accepted = AcceptedStartInferenceRequest(**_REFERENCE, task_id="task_public")

    admitted_turn = ForegroundTurn(request)
    assert admitted_turn.request == request and type(admitted_turn.request) is type(request)
    assert admitted_turn.admitted is True
    assert admitted_turn.updates == (_UPDATE,)
    assert admitted_turn.accepted_objective is None

    plain_turn = ForegroundTurn(plain)
    assert plain_turn.request == plain and type(plain_turn.request) is type(plain)
    assert plain_turn.admitted is False
    assert plain_turn.updates == ()
    assert plain_turn.accepted_objective is None

    accepted_turn = ForegroundTurn(accepted)
    assert accepted_turn.request == accepted and type(accepted_turn.request) is type(accepted)
    assert accepted_turn.admitted is False
    assert accepted_turn.updates == ()
    assert accepted_turn.accepted_objective == "Inspect the release."


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
        ForegroundTurn(Forged(**fields))


def test_a_request_that_is_not_a_snapshot_or_was_altered_is_refused() -> None:
    with pytest.raises(TypeError, match="exact ConversationContextSnapshot or inference request"):
        ForegroundTurn(cast(ConversationContextSnapshot, {"messages": (_USER,)}))

    for kind in (ConversationContextSnapshot, ConversationInferenceRequest):
        altered_rows = kind(revision=1, messages=(_USER,), active_tasks=())
        object.__setattr__(altered_rows, "messages", [_USER])
        with pytest.raises(TypeError, match="messages"):
            ForegroundTurn(altered_rows)

    altered_updates = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    object.__setattr__(altered_updates, "updates", ({"sequence": 1},))
    with pytest.raises(TypeError, match="ConversationPromptUpdate"):
        ForegroundTurn(altered_updates)

    overfull = ConversationInferenceRequest(revision=1, messages=(_USER,), active_tasks=())
    object.__setattr__(overfull, "updates", (_UPDATE,) * 4096)
    with pytest.raises(ValueError, match="capacity"):
        ForegroundTurn(overfull)

    retargeted = AcceptedStartInferenceRequest(**_REFERENCE, task_id="task_public")
    object.__setattr__(retargeted, "task_id", "task_absent")
    with pytest.raises(ValueError, match="one active task"):
        ForegroundTurn(retargeted)


@pytest.mark.parametrize("request_value", [
    ConversationContextSnapshot(**_REFERENCE),
    ConversationInferenceRequest(**_REFERENCE, updates=(_UPDATE,)),
    AcceptedStartInferenceRequest(**_REFERENCE, task_id="task_public"),
])
def test_the_request_is_rebuilt_exactly_once(
    request_value: ConversationContextSnapshot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebuilds: list[type] = []
    validate = ConversationContextSnapshot.__post_init__

    def counted(self: ConversationContextSnapshot) -> None:
        rebuilds.append(type(self))
        validate(self)

    monkeypatch.setattr(context_module.ConversationContextSnapshot, "__post_init__", counted)
    turn = ForegroundTurn(request_value)
    monkeypatch.undo()

    assert rebuilds == [type(request_value)]
    assert turn.request == request_value and turn.request is not request_value
    assert turn.request.messages[0] is not request_value.messages[0]
    if type(request_value) is ConversationInferenceRequest:
        assert turn.updates[0] is not _UPDATE


def test_awaiting_reply_is_true_only_when_the_last_heard_row_is_the_user() -> None:
    assert _plain(_USER).awaiting_reply is True
    assert _plain(_REPLY, _USER).awaiting_reply is True
    assert _plain().awaiting_reply is False
    assert _plain(_USER, _REPLY).awaiting_reply is False


def test_route_values_are_exact_booleans() -> None:
    with pytest.raises(TypeError):
        TurnRoutes(start_work=cast(bool, 1), task_status=False)
    with pytest.raises(TypeError):
        TurnRoutes(start_work=False, task_status=cast(bool, None))


_OPEN = _admitted(_USER)
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
            _admitted(_USER, _REPLY), {},
            TurnRoutes(start_work=False, task_status=False), id="last row is not the user",
        ),
        pytest.param(
            _acknowledgment(_USER), {},
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
            _plain(_USER), {}, TurnRoutes(start_work=True, task_status=False),
            id="unadmitted request keeps the start where admission is not required",
        ),
        pytest.param(
            _plain(_USER), {"start_requires_admission": True},
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
    for build, awaiting, bound, private, attempted, required in product(
        (_plain, _admitted, _acknowledgment), (False, True), (False, True), (False, True),
        (False, True), (False, True),
    ):
        turn = build(*((_USER,) if awaiting else (_USER, _REPLY)))
        routes = turn_routes(
            turn, work_bound=bound, private_material=private, lookup_attempted=attempted,
            start_requires_admission=required,
        )
        if not bound or not awaiting or build is _acknowledgment or private:
            assert routes == TurnRoutes(start_work=False, task_status=False)
        if routes.task_status:
            assert build is _admitted


@pytest.mark.parametrize(
    "name", ["work_bound", "private_material", "lookup_attempted", "start_requires_admission"],
)
def test_route_facts_must_be_exact_booleans(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        turn_routes(_OPEN, **{**_FACTS, name: cast(bool, 1)})
    with pytest.raises(TypeError, match="ForegroundTurn"):
        turn_routes(cast(ForegroundTurn, object()), **_FACTS)
