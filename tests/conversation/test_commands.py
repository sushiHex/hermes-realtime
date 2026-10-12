from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationTaskCommandRouter,
    ConversationWorkControlSurface,
    TaskCancelOutcome,
    TaskDispatchOutcome,
)
from hermes_realtime.evidence import (
    CommandDisposition,
    CommandRoutingOutcome,
    InputSource,
    ReservationError,
)
from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner


@dataclass
class TaskControllerProbe:
    dispatch_outcome: TaskDispatchOutcome
    cancel_outcome: TaskCancelOutcome
    context: ConversationContextStore | None = None

    def __post_init__(self) -> None:
        self.dispatches: list[tuple[str, str]] = []
        self.cancellations: list[tuple[str, str | None]] = []

    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        self.dispatches.append((objective, utterance_id))
        if self.dispatch_outcome.accepted and self.context is not None:
            self.context.record_task_accepted(
                task_id=self.dispatch_outcome.task_id,
                run_id="deleg_probe_private",
                objective=objective,
            )
        return self.dispatch_outcome

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        self.cancellations.append((task_id, reason))
        return self.cancel_outcome


def _probe(context: ConversationContextStore | None = None) -> TaskControllerProbe:
    return TaskControllerProbe(
        dispatch_outcome=TaskDispatchOutcome(
            task_id="task_release_check",
            accepted=True,
        ),
        cancel_outcome=TaskCancelOutcome(
            task_id="task_release_check",
            accepted=True,
        ),
        context=context,
    )


def _spoken_router(context: ConversationContextStore, *, record=None):
    controller = _probe(context)
    events = []
    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
    )
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        validate_user_input=context.validate_user_text if record is not None else None,
        record_user_input=record,
    )
    return router, controller, events


async def _settle_surface(router: ConversationTaskCommandRouter) -> None:
    surface = router._surface
    assert isinstance(surface, ConversationWorkControlSurface)
    await asyncio.gather(*(binding.operation for binding in surface._bindings.values()))


@pytest.mark.asyncio
async def test_spoken_start_uses_existing_acknowledged_work_surface() -> None:
    router, controller, events = _spoken_router(ConversationContextStore())
    assert await router.route("Start task Inspect release evidence.") is True
    await _settle_surface(router)
    assert controller.dispatches[0][0] == "Inspect release evidence."
    assert events == [("task_state", {"status": "active", "taskId": "task_release_check"})]


@pytest.mark.asyncio
async def test_spoken_start_emits_no_active_state_before_accepted_ack() -> None:
    router, controller, events = _spoken_router(ConversationContextStore())
    controller.dispatch_outcome = TaskDispatchOutcome(
        task_id="task_release_check", accepted=False, reason="dispatch refused"
    )
    assert await router.route("start task Inspect release evidence") is True
    await _settle_surface(router)
    assert events == [
        (
            "task_state",
            {
                "status": "rejected",
                "taskId": "task_release_check",
                "reason": "dispatch refused",
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ("start task", "Start task.", "start task!", "start task?"))
async def test_spoken_start_without_objective_returns_fixed_guidance(text: str) -> None:
    router, controller, events = _spoken_router(ConversationContextStore())
    assert await router.route(text) is True
    assert controller.dispatches == []
    assert events == [
        (
            "task_state",
            {
                "status": "rejected",
                "taskId": None,
                "reason": "Provide an objective after Start task.",
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ("cancel task", "Cancel task.", "cancel task!", "cancel task?"))
async def test_spoken_cancel_freezes_the_sole_public_task(text: str) -> None:
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_release_check", run_id="deleg_fixture", objective="Inspect release evidence"
    )
    router, controller, events = _spoken_router(context)
    assert await router.route(text) is True
    await _settle_surface(router)
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]
    assert events == [("task_state", {"status": "cancelling", "taskId": "task_release_check"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (0, 2))
async def test_spoken_cancel_refuses_missing_or_ambiguous_identity(count: int, capsys) -> None:
    context = ConversationContextStore()
    for index in range(count):
        context.record_task_accepted(
            task_id=f"task_fixture_{index}",
            run_id=f"deleg_fixture_{index}",
            objective=f"Work {index}",
        )
    router, controller, events = _spoken_router(context)
    assert await router.route("cancel task") is True
    assert controller.cancellations == []
    assert events == [
        (
            "task_state",
            {
                "status": "rejected",
                "taskId": None,
                "reason": "No active task to cancel."
                if count == 0
                else "Several tasks are active. Use Cancel on the task you want to stop.",
            },
        )
    ]
    marker = capsys.readouterr().out
    assert marker.startswith("[task-command-refusal] ")
    assert json.loads(marker.removeprefix("[task-command-refusal] ")) == {
        "kind": "explicit-command",
        "category": "no-active-task" if count == 0 else "ambiguous-task",
        "candidate_count": count,
    }


@pytest.mark.asyncio
async def test_spoken_cancel_never_retargets_after_recording_await() -> None:
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_release_check", run_id="deleg_fixture", objective="Original work"
    )

    async def record(_text: str) -> None:
        await asyncio.sleep(0)
        context.record_task_completed(task_id="task_release_check", run_id="deleg_fixture")
        context.record_task_accepted(
            task_id="task_replacement", run_id="deleg_replacement", objective="Replacement work"
        )

    router, controller, events = _spoken_router(context, record=record)
    assert await router.route("cancel task") is True
    await _settle_surface(router)
    assert controller.cancellations == []
    assert events == [
        (
            "task_state",
            {
                "status": "rejected",
                "taskId": "task_release_check",
                "reason": "task is not active",
            },
        )
    ]


@pytest.mark.asyncio
async def test_spoken_exact_identity_cancels_only_named_task_among_several() -> None:
    context = ConversationContextStore()
    for task_id in ("task_release_check", "task_other"):
        context.record_task_accepted(
            task_id=task_id, run_id=f"deleg_{task_id}", objective=f"Work for {task_id}"
        )
    router, controller, _events = _spoken_router(context)
    assert await router.route("cancel task task_release_check") is True
    await _settle_surface(router)
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]
    assert tuple(task.task_id for task in context.snapshot().active_tasks) == (
        "task_release_check",
        "task_other",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    (
        "stop",
        "do not cancel task",
        '"cancel task"',
        '"start task Inspect release evidence"',
        "please start task Inspect release evidence",
        "start tasks Inspect release evidence",
        "cancel tasks",
        "start tasking Inspect release evidence",
    ),
)
async def test_spoken_command_grammar_does_not_infer_authority(text: str) -> None:
    router, controller, _events = _spoken_router(ConversationContextStore())
    assert await router.route(text) is False
    assert controller.dispatches == []
    assert controller.cancellations == []


def _lifecycle_owner() -> tuple[EvidenceLifecycleOwner, object]:
    from uuid import UUID

    owner = EvidenceLifecycleOwner(
        owner_generation=71,
        uuid_factory=lambda: str(UUID(int=701, version=4)),
    )
    owner.activate_binding(
        binding_id=str(UUID(int=702, version=4)),
        binding_generation=3,
        consent_epoch_id=str(UUID(int=703, version=4)),
        logical_session_id=str(UUID(int=704, version=4)),
    )
    authority = owner.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=4,
        media_incarnation=None,
        typed_sequence=5,
    )
    return owner, authority


@pytest.mark.asyncio
async def test_non_command_returns_closed_user_turn_result_with_copied_lineage() -> None:
    owner, authority = _lifecycle_owner()
    router = ConversationTaskCommandRouter(
        controller=_probe(),
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        lifecycle_owner=owner.conversation_authority,
    )

    result = await router.route("Please inspect the release evidence", authority)

    assert result.outcome is CommandRoutingOutcome.NOT_COMMAND
    assert result.command_disposition is None
    assert result.user_turn_authority is not None
    assert result.user_turn_authority.utterance_id == authority.utterance_id
    with pytest.raises(TypeError, match="boolean"):
        bool(result)


@pytest.mark.asyncio
async def test_accepted_command_invokes_evidence_hook_once_and_returns_its_disposition() -> None:
    owner, authority = _lifecycle_owner()
    accepted = []

    def on_command_accepted(command_authority: object) -> CommandDisposition:
        accepted.append(command_authority)
        return CommandDisposition.ADMITTED

    router = ConversationTaskCommandRouter(
        controller=_probe(),
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "accepted_1",
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=on_command_accepted,
    )

    result = await router.route("task: Inspect release evidence", authority)

    assert result.outcome is CommandRoutingOutcome.ACCEPTED
    assert result.command_disposition is CommandDisposition.ADMITTED
    assert result.user_turn_authority is None
    assert len(accepted) == 1
    assert accepted[0].utterance_id == authority.utterance_id


@pytest.mark.asyncio
async def test_invalid_command_retires_final_input_without_evidence_hook() -> None:
    owner, authority = _lifecycle_owner()
    accepted = []
    router = ConversationTaskCommandRouter(
        controller=_probe(),
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=lambda command: accepted.append(command) or CommandDisposition.ADMITTED,
    )

    result = await router.route("task:", authority)

    assert result.outcome is CommandRoutingOutcome.INVALID
    assert result.command_disposition is None
    assert result.user_turn_authority is None
    assert accepted == []
    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(authority)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "accepted", "expected_outcome"),
    (
        ("cancel task:", True, CommandRoutingOutcome.INVALID),
        ("cancel task: task_release_check", False, CommandRoutingOutcome.REJECTED),
        ("cancel task: task_release_check", True, CommandRoutingOutcome.ACCEPTED),
    ),
)
async def test_cancel_command_returns_exact_evidence_result_and_consumes_final_input(
    text: str,
    accepted: bool,
    expected_outcome: CommandRoutingOutcome,
) -> None:
    owner, authority = _lifecycle_owner()
    accepted_authorities = []
    controller = TaskControllerProbe(
        dispatch_outcome=TaskDispatchOutcome(task_id="task_release_check", accepted=True),
        cancel_outcome=TaskCancelOutcome(
            task_id="task_release_check",
            accepted=accepted,
            reason=None if accepted else "cancellation rejected",
        ),
    )
    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "cancel_evidence_1",
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=lambda command: (
            accepted_authorities.append(command) or CommandDisposition.ADMITTED
        ),
    )

    result = await router.route(text, authority)

    assert result.outcome is expected_outcome
    assert result.user_turn_authority is None
    assert result.command_disposition is (
        CommandDisposition.ADMITTED if expected_outcome is CommandRoutingOutcome.ACCEPTED else None
    )
    assert len(accepted_authorities) == (
        1 if expected_outcome is CommandRoutingOutcome.ACCEPTED else 0
    )
    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(authority)


@pytest.mark.asyncio
async def test_rejected_command_ack_retires_final_input_without_evidence_hook() -> None:
    owner, authority = _lifecycle_owner()
    accepted = []
    controller = TaskControllerProbe(
        dispatch_outcome=TaskDispatchOutcome(
            task_id="task_rejected",
            accepted=False,
            reason="dispatch unavailable",
        ),
        cancel_outcome=TaskCancelOutcome(
            task_id="task_rejected",
            accepted=False,
        ),
    )
    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "rejected_1",
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=lambda command: accepted.append(command) or CommandDisposition.ADMITTED,
    )

    result = await router.route("task: Inspect release evidence", authority)

    assert result.outcome is CommandRoutingOutcome.REJECTED
    assert result.command_disposition is None
    assert result.user_turn_authority is None
    assert accepted == []
    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(authority)


@pytest.mark.asyncio
async def test_explicit_start_routes_through_shared_work_control_surface() -> None:
    context = ConversationContextStore()
    controller = _probe(context)
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    capacity_checks = 0

    def reserve() -> None:
        nonlocal capacity_checks
        capacity_checks += 1

    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=reserve,
        utterance_id_factory=lambda: "surface_1",
    )
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=reserve,
        utterance_id_factory=lambda: "route_1",
    )

    assert await router.route("task: Inspect the release evidence") is True
    await surface.close()
    assert capacity_checks == 1
    assert controller.dispatches == [("Inspect the release evidence", "utterance_surface_1")]
    assert events == [("task_state", {"status": "active", "taskId": "task_release_check"})]


@pytest.mark.asyncio
async def test_surface_accepted_command_with_authority_returns_typed_evidence_result() -> None:
    owner, authority = _lifecycle_owner()
    context = ConversationContextStore()
    controller = _probe(context)
    accepted: list[object] = []
    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "surface_typed",
    )
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "route_typed",
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=lambda command: accepted.append(command) or CommandDisposition.ADMITTED,
    )

    result = await router.route("task: Inspect release evidence", authority)

    assert result.outcome is CommandRoutingOutcome.ACCEPTED
    assert result.command_disposition is CommandDisposition.ADMITTED
    assert result.user_turn_authority is None
    assert len(accepted) == 1
    await surface.close()


@pytest.mark.asyncio
async def test_explicit_cancel_routes_through_shared_work_control_surface() -> None:
    controller = _probe()
    context = ConversationContextStore()
    context.record_task_accepted(
        task_id="task_release_check",
        run_id="deleg_private_run",
        objective="Inspect the release evidence",
    )
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "unused",
    )
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "route_cancel",
    )

    assert await router.route("cancel task: task_release_check") is True
    await surface.close()
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]
    assert events == [("task_state", {"status": "cancelling", "taskId": "task_release_check"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("task_active", (False, True))
async def test_shared_work_surface_cancel_preserves_evidence_outcome(
    task_active: bool,
) -> None:
    controller = _probe()
    context = ConversationContextStore()
    if task_active:
        context.record_task_accepted(
            task_id="task_release_check",
            run_id="deleg_private_run",
            objective="Inspect release evidence",
        )
    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "surface_cancel",
    )
    owner, authority = _lifecycle_owner()
    accepted = []
    router = ConversationTaskCommandRouter(
        surface=surface,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "surface_route",
        lifecycle_owner=owner.conversation_authority,
        on_command_accepted=lambda command: accepted.append(command) or CommandDisposition.ADMITTED,
    )

    result = await router.route("cancel task: task_release_check", authority)

    assert result.outcome is (
        CommandRoutingOutcome.ACCEPTED if task_active else CommandRoutingOutcome.REJECTED
    )
    assert result.command_disposition is (CommandDisposition.ADMITTED if task_active else None)
    assert result.user_turn_authority is None
    assert len(accepted) == (1 if task_active else 0)
    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(authority)


@pytest.mark.asyncio
async def test_closed_surface_command_is_rejected_with_fixed_guidance_and_no_turn() -> None:
    owner, authority = _lifecycle_owner()
    context = ConversationContextStore()
    router, controller, events = _spoken_router(context)
    surface = router._surface
    assert isinstance(surface, ConversationWorkControlSurface)
    await surface.close()
    router._lifecycle_owner = owner.conversation_authority
    router._on_command_accepted = lambda _command: CommandDisposition.ADMITTED
    result = await router.route("start task Synthetic work", authority)
    assert result.outcome is CommandRoutingOutcome.REJECTED
    assert result.user_turn_authority is None
    assert controller.dispatches == []
    assert events == [("task_state", {
        "status": "rejected", "taskId": None,
        "reason": "Task control is unavailable. Restart the session before trying again.",
    })]
    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(authority)


@pytest.mark.asyncio
async def test_surface_command_admission_rejects_wrong_runtime_type() -> None:
    router, _, _ = _spoken_router(ConversationContextStore())
    async def wrong(**_kwargs):
        return "admitted"
    router._surface.submit_start_command = wrong
    with pytest.raises(TypeError, match="wrong command admission"):
        await router.route("start task Synthetic work")


@pytest.mark.parametrize("missing", ("submit_start_command", "submit_cancel_command"))
def test_surface_requires_both_bounded_admission_methods(missing: str) -> None:
    from types import SimpleNamespace
    methods = {"submit_start_command": lambda **_kwargs: None,
               "submit_cancel_command": lambda **_kwargs: None}
    methods[missing] = None
    with pytest.raises(TypeError, match=missing):
        ConversationTaskCommandRouter(
            surface=SimpleNamespace(**methods), observer=lambda *_args: None,
            reserve_observer_capacity=lambda: None,
        )


@pytest.mark.asyncio
async def test_task_command_routes_only_explicit_prefix_and_projects_after_ack() -> None:
    controller = _probe()
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    capacity_checks = 0

    def reserve() -> None:
        nonlocal capacity_checks
        capacity_checks += 1

    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=reserve,
        utterance_id_factory=lambda: "command_1",
    )

    assert await router.route("Please inspect the release evidence") is False
    assert controller.dispatches == []
    assert await router.route("task: Inspect the release evidence") is True

    assert capacity_checks == 1
    assert controller.dispatches == [("Inspect the release evidence", "utterance_command_1")]
    assert events == [
        (
            "task_state",
            {"status": "active", "taskId": "task_release_check"},
        )
    ]


@pytest.mark.asyncio
async def test_cancel_command_targets_only_public_task_identifier() -> None:
    controller = _probe()
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "unused",
    )

    assert await router.route("cancel task: task_release_check") is True
    assert controller.cancellations == [("task_release_check", "user requested task cancellation")]
    assert events == [
        (
            "task_state",
            {"status": "cancelling", "taskId": "task_release_check"},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    (
        "task:",
        "task: deleg_private_123",
        "cancel task:",
        "cancel task: run_private_12345678",
        "cancel task: release_check",
    ),
)
async def test_malformed_or_private_command_is_consumed_and_fails_before_transport(
    text: str,
) -> None:
    controller = _probe()
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "unused",
    )

    assert await router.route(text) is True
    assert controller.dispatches == []
    assert controller.cancellations == []
    assert events == [
        (
            "task_state",
            {"reason": "invalid explicit task command", "status": "rejected", "taskId": None},
        )
    ]


@pytest.mark.asyncio
async def test_projection_capacity_is_reserved_before_task_side_effect() -> None:
    controller = _probe()

    def reserve() -> None:
        raise RuntimeError("public event projection capacity was exceeded")

    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=reserve,
        utterance_id_factory=lambda: "unused",
    )

    with pytest.raises(RuntimeError, match="capacity"):
        await router.route("task: must not dispatch")
    assert controller.dispatches == []


@pytest.mark.asyncio
async def test_post_ack_projection_failure_rolls_back_new_task() -> None:
    controller = _probe()

    def observe(_kind: str, _data: dict[str, str | int | bool | None]) -> None:
        raise RuntimeError("projection failed")

    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=observe,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "rollback",
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        await router.route("task: dispatch then roll back")
    assert controller.cancellations == [("task_release_check", "task state projection failed")]


class _OrderedControllerProbe(TaskControllerProbe):
    order: list[str]

    async def dispatch(self, *, objective: str, utterance_id: str) -> TaskDispatchOutcome:
        self.order.append("dispatch")
        return await super().dispatch(objective=objective, utterance_id=utterance_id)

    async def request_cancel(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskCancelOutcome:
        self.order.append("cancel")
        return await super().request_cancel(task_id, reason=reason)


def _ordered_probe(order: list[str]) -> _OrderedControllerProbe:
    probe = _OrderedControllerProbe(
        dispatch_outcome=TaskDispatchOutcome(task_id="task_release_check", accepted=True),
        cancel_outcome=TaskCancelOutcome(task_id="task_release_check", accepted=True),
    )
    probe.order = order
    return probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "action"),
    [
        pytest.param("task: Inspect the release evidence", "dispatch", id="task"),
        pytest.param("start task Inspect the release evidence", "dispatch", id="spoken-start"),
        pytest.param("cancel task", None, id="spoken-no-context"),
        pytest.param("  Task: padded utterance  ", "dispatch", id="exact-utterance"),
        pytest.param("cancel task: task_release_check", "cancel", id="cancel"),
        pytest.param("task:", None, id="invalid-task"),
        pytest.param("cancel task: release_check", None, id="invalid-cancel"),
    ],
)
async def test_every_command_becomes_one_user_row_before_the_router_acts(
    text: str,
    action: str | None,
) -> None:
    order: list[str] = []

    async def record_user_input(recorded: str) -> None:
        order.append(f"record:{recorded}")

    router = ConversationTaskCommandRouter(
        controller=_ordered_probe(order),
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "command_1",
        validate_user_input=lambda checked: order.append(f"validate:{checked}"),
        record_user_input=record_user_input,
    )

    assert await router.route("Please inspect the release evidence") is False
    assert order == []
    assert await router.route(text) is True

    assert order == [
        f"validate:{text}",
        f"record:{text}",
        *([action] if action is not None else []),
    ]


@pytest.mark.asyncio
async def test_surface_commands_are_recorded_before_the_work_surface_acts() -> None:
    order: list[str] = []

    class SurfaceProbe:
        max_objective_chars = 1024

        async def submit_start_command(self, *, objective: str, invocation_id: str) -> object:
            del objective, invocation_id
            order.append("start")
            from hermes_realtime.conversation.work_tools import WorkCommandAdmission

            return WorkCommandAdmission.REFUSED

        async def submit_cancel_command(self, **_kwargs: object) -> object:
            order.append("cancel")
            from hermes_realtime.conversation.work_tools import WorkCommandAdmission

            return WorkCommandAdmission.REFUSED

    async def record_user_input(recorded: str) -> None:
        order.append(f"record:{recorded}")

    router = ConversationTaskCommandRouter(
        surface=SurfaceProbe(),  # type: ignore[arg-type]
        observer=lambda _kind, _data: None,
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "command_1",
        validate_user_input=lambda _text: None,
        record_user_input=record_user_input,
    )

    assert await router.route("task: Inspect it") is True
    assert await router.route("cancel task: task_release_check") is True

    assert order == [
        "record:task: Inspect it",
        "start",
        "record:cancel task: task_release_check",
        "cancel",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("task: " + "x" * 12, id="over-the-context-item-bound"),
        pytest.param("start task " + "x" * 12, id="spoken-context-item-bound"),
        pytest.param("cancel task: task_deleg_private", id="private-run-token"),
        pytest.param("task: lone \ud800", id="not-utf8-encodable"),
    ],
)
async def test_a_command_the_context_cannot_hold_is_invalid_before_transport(text: str) -> None:
    context = ConversationContextStore(max_item_chars=16)
    controller = _probe()
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    recorded: list[str] = []

    async def record_user_input(text: str) -> None:
        recorded.append(text)

    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "unused",
        validate_user_input=context.validate_user_text,
        record_user_input=record_user_input,
    )

    assert await router.route(text) is True

    assert recorded == []
    assert controller.dispatches == []
    assert controller.cancellations == []
    assert events == [
        (
            "task_state",
            {"reason": "invalid explicit task command", "status": "rejected", "taskId": None},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(RuntimeError("prior turn cleanup failed"), id="cleanup-failure"),
        pytest.param(ValueError("not a validation refusal"), id="value-error"),
    ],
)
async def test_a_recording_failure_propagates_and_is_never_reported_invalid(
    failure: Exception,
) -> None:
    controller = _probe()
    events: list[tuple[str, dict[str, str | int | bool | None]]] = []

    async def record_user_input(_text: str) -> None:
        raise failure

    router = ConversationTaskCommandRouter(
        controller=controller,
        observer=lambda kind, data: events.append((kind, data)),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "unused",
        validate_user_input=ConversationContextStore().validate_user_text,
        record_user_input=record_user_input,
    )

    with pytest.raises(type(failure), match=str(failure)):
        await router.route("task: Inspect it")

    assert controller.dispatches == []
    assert events == []


def test_the_user_input_hooks_must_be_callable_and_paired() -> None:
    async def record(_text: str) -> None:
        return None

    for kwargs in (
        {"record_user_input": "not callable", "validate_user_input": lambda _text: None},
        {"record_user_input": record, "validate_user_input": "not callable"},
    ):
        with pytest.raises(TypeError, match="user_input"):
            ConversationTaskCommandRouter(
                controller=_probe(),
                observer=lambda _kind, _data: None,
                reserve_observer_capacity=lambda: None,
                **kwargs,  # type: ignore[arg-type]
            )
    for kwargs in ({"record_user_input": record}, {"validate_user_input": lambda _text: None}):
        with pytest.raises(ValueError, match="together"):
            ConversationTaskCommandRouter(
                controller=_probe(),
                observer=lambda _kind, _data: None,
                reserve_observer_capacity=lambda: None,
                **kwargs,  # type: ignore[arg-type]
            )
