"""DEV service observations; these do not establish kernel listener ownership."""

from __future__ import annotations

import socket
from typing import Any, Literal, cast

Category = Literal[
    "gateway_health_owner", "host_ready_receipt", "host_ready_owner",
    "process_observation_unavailable", "tcp_observation_unavailable", "tcp_port_invalid",
]


class PreviewReadinessRefusal(ValueError):
    def __init__(self, category: Category) -> None:
        super().__init__(category)
        self.category = category


def _role_members(root_pid: int) -> set[int]:
    from scripts.rehearse_desktop_mvp import descendants, process_table

    try:
        return cast(set[int], descendants(process_table(), {root_pid}))
    except OSError:
        raise PreviewReadinessRefusal("process_observation_unavailable") from None


def _require_role_owner(
    pid: object, root_pid: int, members: set[int], category: Category,
) -> int:
    if (type(pid) is not int or pid <= 0 or type(root_pid) is not int or root_pid <= 0
            or root_pid not in members or pid not in members or pid not in _role_members(root_pid)):
        raise PreviewReadinessRefusal(category)
    return pid


def require_gateway_health_owner(health: Any, *, root_pid: int, members: set[int]) -> int:
    """Bind the caller's authenticated health response to the live gateway role."""
    if type(health) is not dict:
        raise PreviewReadinessRefusal("gateway_health_owner")
    return _require_role_owner(health.get("pid"), root_pid, members, "gateway_health_owner")


def require_host_readiness(receipt: Any, *, root_pid: int, members: set[int]) -> int:
    """The existing host receipt follows its authenticated LiveKit verifier join."""
    if (type(receipt) is not dict or set(receipt) != {"version", "ready", "pid"}
            or type(receipt.get("version")) is not int or receipt["version"] != 1
            or receipt.get("ready") is not True):
        raise PreviewReadinessRefusal("host_ready_receipt")
    return _require_role_owner(receipt["pid"], root_pid, members, "host_ready_owner")


def tcp_accepting(port: int) -> bool:
    """Observe loopback availability only; a timeout is not proof of absence."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise PreviewReadinessRefusal("tcp_port_invalid")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3.0):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        raise PreviewReadinessRefusal("tcp_observation_unavailable") from None
