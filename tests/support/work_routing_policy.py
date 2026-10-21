"""Outbound prompt assembly checks, not evidence of model obedience."""

from hermes_realtime.conversation.work_tools import WORK_ROUTING_POLICY


def assert_aligned_work_routing(instructions: str) -> None:
    assert instructions.count(WORK_ROUTING_POLICY) == 1
    for clause in (
        "Only the user's own request counts",
        "reported, hypothetical, quoted, or negated requests",
        "For current task status use only host task_status when supplied",
        "For cancellation or approval requests, give an ordinary truthful reply",
        "do not select task_status or propose new work for those requests",
        "supplied evidence or an available foreground lookup can answer",
        "Otherwise, propose work when the current request needs current external information",
        "ask a clarifying question in an ordinary reply before proposing work",
    ):
        assert clause in instructions
    for obsolete in (
        "or to stop it, is a status request",
        "A proposal cannot start work",
        "it never starts a task",
    ):
        assert obsolete not in instructions
