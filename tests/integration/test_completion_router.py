import asyncio
from datetime import UTC, datetime

import pytest

from hermes_realtime.integration import (
    EventSequencer,
    HermesCompletionRouter,
    HermesDispatchCommand,
    HermesIntegrationService,
    SessionBindings,
)
from hermes_realtime.protocol import (
    WorkDispatchAcknowledgedEvent,
    WorkDispatchRequestedEvent,
)


class ImmediateDispatcher:
    def __init__(self) -> None:
        self.commands: list[HermesDispatchCommand] = []

    async def dispatch(self, command: HermesDispatchCommand) -> str:
        self.commands.append(command)
        return f"deleg_{len(self.commands):03d}"


def request(task_id: str) -> WorkDispatchRequestedEvent:
    return WorkDispatchRequestedEvent(
        type="work.dispatch.requested",
        event_id=f"evt_{task_id}",
        session_id="session_001",
        sequence=7,
        timestamp=datetime(2026, 7, 19, 3, 59, tzinfo=UTC),
        task_id=task_id,
        utterance_id=f"utterance_{task_id}",
        payload={"objective": f"Complete {task_id}"},
    )


@pytest.mark.asyncio
async def test_completion_router_emits_ordered_event_for_acknowledged_run() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    event_ids = iter(("evt_ack_001", "evt_completed_001"))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        sequencer=sequencer,
    )
    router = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: next(event_ids),
        clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
    )

    acknowledgment = await service.dispatch("participant_001", request("task_001"))
    await router.track(acknowledgment)
    completed = await router.complete(
        "deleg_001",
        status="completed",
        summary="The delegated comparison is complete.",
    )

    assert acknowledgment.sequence == 8
    assert completed.sequence == 9
    assert completed.session_id == "session_001"
    assert completed.task_id == "task_001"
    assert completed.run_id == "deleg_001"
    assert completed.payload.status == "completed"
    assert completed.payload.summary == "The delegated comparison is complete."


@pytest.mark.asyncio
async def test_shared_sequencer_prevents_acknowledgment_completion_collisions() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    counter = iter(range(1, 20))
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=lambda: f"evt_{next(counter)}",
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        sequencer=sequencer,
    )
    router = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=lambda: f"evt_{next(counter)}",
        clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
    )

    acknowledgments = await asyncio.gather(
        service.dispatch("participant_001", request("task_001")),
        service.dispatch("participant_001", request("task_002")),
    )
    await asyncio.gather(*(router.track(ack) for ack in acknowledgments))
    completions = await asyncio.gather(
        router.complete("deleg_001", status="completed", summary="First complete"),
        router.complete("deleg_002", status="completed", summary="Second complete"),
    )

    sequences = [event.sequence for event in [*acknowledgments, *completions]]
    assert sorted(sequences) == [8, 9, 10, 11]
    assert len(set(sequences)) == 4


@pytest.mark.asyncio
async def test_completion_router_rejects_unknown_run() -> None:
    router = HermesCompletionRouter(
        sequencer=EventSequencer(),
        event_id_factory=lambda: "evt_completed_001",
        clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
    )

    with pytest.raises(KeyError, match="deleg_unknown"):
        await router.complete(
            "deleg_unknown",
            status="completed",
            summary="Must not be attributed",
        )


@pytest.mark.asyncio
async def test_completion_router_bounds_terminal_retry_retention() -> None:
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")
    sequencer = EventSequencer()
    service = HermesIntegrationService(
        bindings=bindings,
        dispatcher=ImmediateDispatcher(),
        event_id_factory=iter(("evt_ack_001", "evt_ack_002")).__next__,
        clock=lambda: datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        sequencer=sequencer,
    )
    router = HermesCompletionRouter(
        sequencer=sequencer,
        event_id_factory=iter(("evt_completed_001", "evt_completed_002")).__next__,
        clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
        max_retained_completions=1,
    )
    first = await service.dispatch("participant_001", request("task_001"))
    second = await service.dispatch("participant_001", request("task_002"))
    await router.track(first)
    await router.complete("deleg_001", status="completed", summary="First done")
    await router.track(second)
    await router.complete("deleg_002", status="completed", summary="Second done")

    with pytest.raises(KeyError, match="unknown Hermes run"):
        await router.complete("deleg_001", status="completed", summary="First done")


def acknowledgment(run_id: str, sequence: int) -> WorkDispatchAcknowledgedEvent:
    return WorkDispatchAcknowledgedEvent(
        type="work.dispatch.acknowledged",
        event_id=f"evt_ack_{run_id}",
        session_id="session_001",
        sequence=sequence,
        timestamp=datetime(2026, 7, 19, 4, 0, tzinfo=UTC),
        task_id=f"task_{run_id}",
        payload={"accepted": True, "run_id": run_id},
    )


@pytest.mark.asyncio
async def test_router_bounds_acknowledgments_that_never_report_a_terminal_result() -> None:
    # Regression: acknowledgments were released only when their *completion* was
    # evicted, so a run that is cancelled or interrupted without a terminal
    # callback was retained for the process lifetime.
    router = HermesCompletionRouter(
        sequencer=EventSequencer(),
        event_id_factory=lambda: "evt_completed_001",
        clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
        max_retained_completions=2,
        max_tracked_acknowledgments=2,
    )

    await router.track(acknowledgment("deleg_001", 10))
    await router.track(acknowledgment("deleg_002", 11))

    with pytest.raises(RuntimeError, match="acknowledgment retention capacity"):
        await router.track(acknowledgment("deleg_003", 12))


@pytest.mark.asyncio
async def test_router_retracking_a_known_run_does_not_consume_capacity() -> None:
    router = HermesCompletionRouter(
        sequencer=EventSequencer(),
        event_id_factory=lambda: "evt_completed_001",
        clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
        max_retained_completions=1,
        max_tracked_acknowledgments=1,
    )

    await router.track(acknowledgment("deleg_001", 10))
    await router.track(acknowledgment("deleg_001", 10))

    assert await router.is_tracked("deleg_001")


def test_router_rejects_an_acknowledgment_bound_below_its_completion_bound() -> None:
    with pytest.raises(ValueError, match="max_tracked_acknowledgments"):
        HermesCompletionRouter(
            sequencer=EventSequencer(),
            event_id_factory=lambda: "evt_completed_001",
            clock=lambda: datetime(2026, 7, 19, 4, 1, tzinfo=UTC),
            max_retained_completions=8,
            max_tracked_acknowledgments=4,
        )
