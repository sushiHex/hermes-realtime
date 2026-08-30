from importlib.metadata import EntryPoint, entry_points

import pytest

from hermes_realtime.integration import (
    HermesDispatchCommand,
    LocalHermesBridgeServer,
    SessionBindings,
)
from hermes_realtime.protocol import Durability


def hermes_realtime_entry_point() -> EntryPoint:
    matches = [
        entry_point
        for entry_point in entry_points(group="hermes_agent.plugins")
        if entry_point.name == "hermes-realtime"
    ]
    assert len(matches) == 1
    return matches[0]


def test_package_exposes_hermes_plugin_entry_point() -> None:
    entry_point = hermes_realtime_entry_point()

    assert entry_point.value == "hermes_realtime.hermes_plugin"


class PluginContextStub:
    def __init__(self, delegation_id: str) -> None:
        self.delegation_id = delegation_id
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.hooks: dict[str, object] = {}

    def dispatch_tool(self, name: str, args: dict[str, object]) -> str:
        self.calls.append((name, args))
        return '{"status":"rejected","error":"ordinary dispatch not expected"}'

    def dispatch_reserved_delegation(
        self, args: dict[str, object], delegation_id: str
    ) -> str:
        self.calls.append(("delegate_task", args))
        return (
            '{"status":"dispatched","delegation_id":"'
            f'{delegation_id}"}}'
        )

    def register_hook(self, name: str, callback: object) -> None:
        self.hooks[name] = callback

    def interrupt_delegation(self, delegation_id: str) -> bool:
        del delegation_id
        return True


@pytest.mark.asyncio
async def test_entry_point_registration_can_be_refreshed() -> None:
    plugin = hermes_realtime_entry_point().load()
    first_context = PluginContextStub("deleg_old")
    refreshed_context = PluginContextStub("deleg_refreshed")

    plugin.register(first_context)
    plugin.register(refreshed_context)

    run_id = await plugin.get_dispatcher().dispatch(
        HermesDispatchCommand(
            session_id="session_001",
            task_id="task_001",
            objective="Verify plugin registration",
            durability=Durability.EPHEMERAL,
        )
    )
    assert run_id.startswith("deleg_")
    assert len(run_id) == 38
    assert first_context.calls == []
    assert len(refreshed_context.calls) == 1


def test_entry_point_routes_exact_async_delegation_completion() -> None:
    plugin = hermes_realtime_entry_point().load()
    context = PluginContextStub("deleg_completed")
    plugin.register(context)
    completions = []
    unsubscribe = plugin.get_runtime().subscribe_completion(completions.append)

    callback = context.hooks["subagent_stop"]
    assert callable(callback)
    callback(
        delegation_id="deleg_completed",
        child_status="completed",
        child_summary="The delegated work completed successfully.",
    )
    unsubscribe()

    assert len(completions) == 1
    assert completions[0].run_id == "deleg_completed"
    assert completions[0].status == "completed"
    assert completions[0].summary.startswith("The delegated work")


def test_entry_point_ignores_synchronous_subagent_completion_without_handle() -> None:
    plugin = hermes_realtime_entry_point().load()
    context = PluginContextStub("deleg_unused")
    plugin.register(context)
    completions = []
    unsubscribe = plugin.get_runtime().subscribe_completion(completions.append)

    callback = context.hooks["subagent_stop"]
    assert callable(callback)
    callback(
        delegation_id=None,
        child_status="completed",
        child_summary="Synchronous child with no async handle.",
    )
    unsubscribe()

    assert completions == []


def test_create_local_bridge_uses_registered_plugin_runtime() -> None:
    plugin = hermes_realtime_entry_point().load()
    context = PluginContextStub("deleg_factory")
    plugin.register(context)
    bindings = SessionBindings()
    bindings.bind("participant_001", "session_001")

    bridge = plugin.create_local_bridge(
        bindings=bindings,
        token="test-token-with-sufficient-entropy",
    )

    assert isinstance(bridge, LocalHermesBridgeServer)
