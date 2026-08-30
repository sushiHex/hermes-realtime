from __future__ import annotations

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
        on_command_accepted=lambda command: accepted.append(command)
        or CommandDisposition.ADMITTED,
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
        on_command_accepted=lambda command: accepted_authorities.append(command)
        or CommandDisposition.ADMITTED,
    )

    result = await router.route(text, authority)

    assert result.outcome is expected_outcome
    assert result.user_turn_authority is None
    assert result.command_disposition is (
        CommandDisposition.ADMITTED
        if expected_outcome is CommandRoutingOutcome.ACCEPTED
        else None
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
        on_command_accepted=lambda command: accepted.append(command)
        or CommandDisposition.ADMITTED,
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
        on_command_accepted=lambda command: accepted.append(command)
        or CommandDisposition.ADMITTED,
    )

    result = await router.route("task: Inspect release evidence", authority)

    assert result.outcome is CommandRoutingOutcome.ACCEPTED
    assert result.command_disposition is CommandDisposition.ADMITTED
    assert result.user_turn_authority is None
    assert len(accepted) == 1


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
    assert controller.cancellations == [
        ("task_release_check", "user requested task cancellation")
    ]
    assert events == [
        ("task_state", {"status": "cancelling", "taskId": "task_release_check"})
    ]


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
        on_command_accepted=lambda command: accepted.append(command)
        or CommandDisposition.ADMITTED,
    )

    result = await router.route("cancel task: task_release_check", authority)

    assert result.outcome is (
        CommandRoutingOutcome.ACCEPTED if task_active else CommandRoutingOutcome.REJECTED
    )
    assert result.command_disposition is (
        CommandDisposition.ADMITTED if task_active else None
    )
    assert result.user_turn_authority is None
    assert len(accepted) == (1 if task_active else 0)
    with pytest.raises(ReservationError, match="consumed"):
        owner.decline_to_user(authority)


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
    assert controller.dispatches == [
        ("Inspect the release evidence", "utterance_command_1")
    ]
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
    assert controller.cancellations == [
        ("task_release_check", "user requested task cancellation")
    ]
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
    assert controller.cancellations == [
        ("task_release_check", "task state projection failed")
    ]
