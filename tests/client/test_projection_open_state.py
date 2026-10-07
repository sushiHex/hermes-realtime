from __future__ import annotations

import pytest

from hermes_realtime.client import BrowserEventProjection


def test_open_tasks_and_approvals_outlive_a_lease_reset_and_stay_bounded() -> None:
    projection = BrowserEventProjection(capacity=4)
    for index in range(4):
        projection.publish("task_state", {"taskId": f"task_{index}", "status": "active"})
    projection.acknowledge_through(4)
    # Every event slot is free again, but four tasks are still open: a fifth is refused.
    with pytest.raises(RuntimeError, match="open task and approval state"):
        projection.publish("task_state", {"taskId": "task_4", "status": "active"})
    # Settling one makes room, and an update to an open one is never refused.
    projection.publish("task_state", {"taskId": "task_0", "status": "completed"})
    projection.acknowledge_through(5)
    projection.publish("task_state", {"taskId": "task_1", "status": "cancelling"})
    projection.reset()
    assert projection.open_state_count == 3

    projection.republish_open_state()

    assert [dict(event.data) for event in projection.events_after(0)] == [
        {"taskId": "task_1", "status": "cancelling"},
        {"taskId": "task_2", "status": "active"},
        {"taskId": "task_3", "status": "active"},
    ]


def test_an_approval_is_open_while_actionable_and_settles_when_decided() -> None:
    projection = BrowserEventProjection()
    pending = {
        "actionable": True,
        "approvalId": "approval_0123456789abcdef",
        "command": "chmod 777 /tmp/x",
        "description": "change permissions",
        "state": "pending",
        "taskId": "task_1",
    }
    projection.publish("approval_state", dict(pending))
    assert projection.open_state_count == 1
    projection.publish(
        "approval_state",
        {"actionable": False, "approvalId": "approval_0123456789abcdef", "state": "approve"},
    )
    assert projection.open_state_count == 0
