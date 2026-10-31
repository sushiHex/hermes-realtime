"""Outbound prompt assembly checks, not evidence of model obedience."""

from hermes_realtime.conversation.work_tools import WORK_ROUTING_POLICY


def assert_aligned_work_routing(instructions: str, *, task_status_available: bool) -> None:
    assert instructions.count(WORK_ROUTING_POLICY) == 1
    available = "Use task_status only for pending, active, or cancellation state."
    unavailable = "Current task status cannot be checked here; no host status route is supplied."
    selected, absent = (
        (available, unavailable) if task_status_available else (unavailable, available)
    )
    assert instructions.count(selected) == 1
    assert absent not in instructions
    for clause in (
        "Read the latest user turn in context",
        "Supplied details continue an unresolved user request without repetition",
        "Once sufficient, propose work, not promises",
        "Assistant offers, social acknowledgments, status questions, or failed or completed "
        "tasks alone do not request work",
        "Only the user's own request counts",
        "reported, hypothetical, quoted, or negated requests",
        "For cancellation or approval, direct the user to the controls",
        "do not select task_status or propose work",
        "supplied evidence or an available foreground lookup can answer",
        "Otherwise, propose work when the current request needs current external information",
        "ask a clarifying question in an ordinary reply before proposing work",
    ):
        assert clause in instructions
    for obsolete in (
        "For current task status use only host task_status when supplied",
        "or to stop it, is a status request",
        "A proposal cannot start work",
        "it never starts a task",
    ):
        assert obsolete not in instructions
