"""The preview owner replaces a host that ended itself, within a fixed budget (#531).

A host that exits with the reserved status ended itself after a host-wide fault and saved
the conversation first. The owner replaces it in place, against the same run directory,
gateway and LiveKit, so the successor settles the same run record and can tell the user what
was stopped. Any other exit ends the run as before.

These tests drive the real ``serve`` with the process owner, the Job and the probes replaced,
as ``test_preview_dev_services`` does. One test at the end uses real processes on Windows.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.host_launcher import HOST_FATAL_EXIT_STATUS
from scripts import preview_cutover
from tests.test_preview_dev_services import (  # noqa: F401 -- the fixture is used by name
    _FakeProcess,
    _read_receipt,
    dev_service_case,
)

_HOST_PIDS = (41003, 41103, 41203, 41303, 41403)


class _Run:
    """Script the fake world: what each host incarnation does, and what the owner did."""

    def __init__(self, case: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.lookup_backend_logs as lookup_backend_logs
        import scripts.rehearse_desktop_mvp as owner

        self.case = case
        self.plan = case.plan
        self.state = case.state
        self.exits: list[int] = []  # the status each successive host exits with
        self.children: dict[int, int] = {}  # pid -> start time, below the live host
        self.immortal: set[int] = set()
        self.outside_job: set[int] = set()
        self.respawn_fault: str | None = None
        self.also_exits: str | None = None
        self.after_exit: Any = lambda: None
        self.serving = False  # true once the owner has said ready
        self.roots: dict[str, _FakeProcess] = {}
        self.port_stays_open = False
        self.snapshots = 0
        self.snapshots_at_ready = 0
        self.snapshots_while_first_host_lived = 0
        self.idle_ticks = 0  # owner polls the first host lives through before it exits
        self.late_children: dict[int, int] = {}  # started after the last tree sample
        self.grandchildren: dict[int, tuple[int, int]] = {}  # pid -> (parent, start)
        self.ends_early: set[int] = set()  # ends just before the host does
        self.unobservable: set[int] = set()  # cannot be opened or timed
        self.actions: list[str] = []
        self.lines: list[dict[str, Any]] = []
        self.hosts: list[_FakeProcess] = []
        self.started = {41001: 1, 41002: 1}
        self.ready_pids: list[int] = []
        run = self

        class Capped:
            def __init__(self, owned: Any, _directory: Any) -> None:
                self.owned = owned
                self.roots = run.roots

            def spawn(self, name: str, _argv: Any, *, env: Any, cwd: Any = None,
                      log: Any = None) -> _FakeProcess:
                assert name not in self.roots
                return self._launch(name)

            def respawn(self, name: str, _argv: Any, *, env: Any, cwd: Any = None,
                        log: Any = None) -> _FakeProcess:
                assert name == "host" and self.roots[name].poll() is not None
                return self._launch(name)

            def _launch(self, name: str) -> _FakeProcess:
                run.state["spawned"].append(name)
                run.actions.append("launch:" + name)
                if name != "host":
                    pid = run.case.pids[name]
                    process = _FakeProcess(pid)
                    run.state["table"][pid] = (1, name)
                    ports = ("api", "companion") if name == "gateway" else ("livekit",)
                    for port in ports:
                        run.state["tcp"][run.plan.ports[port]] = True
                    self.roots[name] = process
                    return process
                pid = _HOST_PIDS[len(run.hosts)]
                process = _FakeProcess(pid)
                run.hosts.append(process)
                self.roots[name] = process
                run.state["table"][pid] = (1, "host")
                run.started[pid] = 1
                if run.respawn_fault == "exits_during_start" and len(run.hosts) > 1:
                    process.returncode = 1
                    return process
                ready = run.plan.output / "host-ready.private.json"
                # The owner removed the exited host's receipt before starting this one.
                assert not ready.exists()
                ready.write_text(
                    json.dumps({"version": 1, "ready": True, "pid": pid}), encoding="utf-8"
                )
                run.ready_pids.append(pid)
                run.state["tcp"][run.plan.ports["host"]] = True
                for child, when in run.children.items():
                    run.state["table"][child] = (pid, "child")
                    run.started[child] = when
                for grandchild, (parent, when) in run.grandchildren.items():
                    run.state["table"][grandchild] = (parent, "grandchild")
                    run.started[grandchild] = when
                return process

            def stop_all(self, _containment: Any) -> None:
                run.state["processes_stopped"] = True
                for process in self.roots.values():
                    process.returncode = 0 if process.returncode is None else process.returncode
                for port in run.state["tcp"]:
                    run.state["tcp"][port] = False

            def finish(self) -> dict[str, Any]:
                return {}

        class Containment:
            def members(self) -> set[int]:
                return set(run.state["table"]) - run.outside_job

        def process_table() -> dict[int, tuple[int, str]]:
            run.snapshots += 1
            return dict(run.state["table"])

        def tick(_seconds: float) -> None:
            # Every owner poll sleeps once: that is the fake world's turn to act.
            run._host_acts()

        def kill_tree(pid: int) -> None:
            run.actions.append(f"kill:{pid}")
            if pid not in run.immortal:
                run.state["table"].pop(pid, None)
                run.started.pop(pid, None)

        def emit(event: dict[str, Any]) -> None:
            run.lines.append(dict(event))
            if event.get("category") == "ready":
                run.serving = True
                run.snapshots_at_ready = run.snapshots
            if event.get("category") == "host_restart":
                run.actions.append("line:" + event["outcome"])

        monkeypatch.setattr(lookup_backend_logs, "CappedProcesses", Capped)
        monkeypatch.setattr(owner, "Containment", Containment)
        monkeypatch.setattr(owner, "process_table", process_table)
        def started_at(pid: int) -> int | None:
            if pid in run.unobservable:
                raise owner.ProcessUnobservable(pid)
            return run.started.get(pid)

        monkeypatch.setattr(owner, "started_at", started_at)
        monkeypatch.setattr(owner, "_kill_tree", kill_tree)
        monkeypatch.setattr(preview_cutover, "emit", emit)
        monkeypatch.setattr(
            preview_cutover, "time", types.SimpleNamespace(sleep=tick, monotonic=time.monotonic)
        )
        monkeypatch.setattr(preview_cutover, "_STOP_POLL_SECONDS", 0)
        monkeypatch.setattr(preview_cutover, "_HOST_RESTART_SETTLE_SECONDS", 0.05)
        monkeypatch.setattr(
            preview_cutover, "_runtime_modules",
            lambda _plan: (None, types.SimpleNamespace(
                _load_api_bearer=lambda _env: "synthetic-bearer",
                HOST_FATAL_EXIT_STATUS=HOST_FATAL_EXIT_STATUS,
            )),
        )

    def _host_acts(self) -> None:
        if not self.hosts or not self.serving:
            return
        host = self.hosts[-1]
        if host.returncode is not None:
            return
        if self.idle_ticks > 0:
            self.idle_ticks -= 1
            return
        if len(self.hosts) <= len(self.exits):
            if len(self.hosts) == 1:
                self.snapshots_while_first_host_lived = self.snapshots - self.snapshots_at_ready
            for child, when in self.late_children.items():
                self.state["table"][child] = (host.pid, "child")
                self.started[child] = when
            for gone in self.ends_early:
                self.state["table"].pop(gone, None)
                self.started.pop(gone, None)
            # It ends itself: the process and its listener go, its children stay.
            host.returncode = host.exit_code = self.exits[len(self.hosts) - 1]
            self.state["table"].pop(host.pid, None)
            self.started.pop(host.pid, None)
            if not self.port_stays_open:
                self.state["tcp"][self.plan.ports["host"]] = False
            if self.also_exits is not None:
                self.roots[self.also_exits].returncode = 1
            self.after_exit()
        else:
            (self.plan.output / "stop.request").write_text("", encoding="utf-8")

    def serve(self) -> int:
        return preview_cutover.serve(self.plan.output / "active.private.json")

    @property
    def restart_lines(self) -> list[dict[str, Any]]:
        return [line for line in self.lines if line.get("category") == "host_restart"]

    @property
    def receipt(self) -> dict[str, Any]:
        receipt = _read_receipt(self.plan.output, "result-receipt.private.json")
        assert isinstance(receipt, dict)
        return receipt


@pytest.fixture
def run(dev_service_case: Any, monkeypatch: pytest.MonkeyPatch) -> _Run:  # noqa: F811
    return _Run(dev_service_case, monkeypatch)


def test_a_host_that_ended_itself_is_replaced_in_place_and_the_run_can_stop_cleanly(
    run: _Run,
) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.children = {41900: 7, 41901: 8}

    assert run.serve() == 0

    # Its leftover children are ended by process ID before the successor starts, and the
    # one line follows the outcome.
    assert run.actions == [
        "launch:gateway", "launch:livekit", "launch:host",
        "kill:41900", "kill:41901", "launch:host", "line:restarted",
    ]
    assert run.restart_lines == [
        {"category": "host_restart", "outcome": "restarted", "restarts": 1, "remaining": 2}
    ]
    # Same run directory, gateway and LiveKit: only the host was launched again.
    assert run.state["spawned"] == ["gateway", "livekit", "host", "host"]
    assert run.ready_pids == [41003, 41103]
    assert run.receipt["category"] == "stopped"
    assert run.receipt["host_restarts"] == 1
    assert run.receipt["failure_category"] is None
    # Readiness was claimed once; a restart is not a second ready receipt.
    assert [line["category"] for line in run.lines].count("ready") == 1


def test_the_budget_is_three_per_owner_run_and_its_end_is_a_distinct_final_stop(
    run: _Run,
) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS] * 4

    assert run.serve() == 1

    seen = [(line["outcome"], line["restarts"], line["remaining"]) for line in run.restart_lines]
    assert seen == [
        ("restarted", 1, 2), ("restarted", 2, 1), ("restarted", 3, 0), ("budget_spent", 3, 0),
    ]
    assert run.state["spawned"].count("host") == 4
    assert run.receipt["category"] == "refused"
    assert run.receipt["failure_category"] == "host_restart_budget_spent"
    assert run.receipt["host_restarts"] == 3
    assert run.receipt["host_exit_code"] == HOST_FATAL_EXIT_STATUS


@pytest.mark.parametrize("status", [0, 1, 2, 130, -1, 3221225477])
def test_a_host_that_exits_with_any_other_status_ends_the_run_as_before(
    run: _Run, status: int,
) -> None:
    run.exits = [status]

    assert run.serve() == 1

    assert run.restart_lines == []
    assert run.state["spawned"] == ["gateway", "livekit", "host"]
    assert run.receipt["failure_category"] == "owned_child_exited"
    assert run.receipt["host_restarts"] == 0


def test_a_runtime_without_the_reserved_status_is_never_restarted(
    run: _Run, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preview_cutover, "_runtime_modules",
        lambda _plan: (None, types.SimpleNamespace(
            _load_api_bearer=lambda _env: "synthetic-bearer"
        )),
    )
    run.exits = [HOST_FATAL_EXIT_STATUS]

    assert run.serve() == 1

    assert run.restart_lines == []
    assert run.state["spawned"] == ["gateway", "livekit", "host"]
    assert run.receipt["failure_category"] == "owned_child_exited"


@pytest.mark.parametrize("also", ["gateway", "livekit"])
def test_another_child_exiting_is_never_a_reason_to_restart(run: _Run, also: str) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.also_exits = also

    assert run.serve() == 1

    assert run.restart_lines == []
    assert run.state["spawned"].count("host") == 1
    assert run.receipt["failure_category"] == "owned_child_exited"


def test_a_leftover_child_that_will_not_end_refuses_the_restart(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.children = {41900: 7}
    run.immortal = {41900}

    assert run.serve() == 1

    assert run.actions[-2:] == ["kill:41900", "line:refused"]
    assert run.state["spawned"].count("host") == 1
    assert run.restart_lines == [
        {"category": "host_restart", "outcome": "refused", "restarts": 0, "remaining": 3}
    ]
    assert run.receipt["failure_category"] == "host_restart_survivors"
    assert run.receipt["host_restarts"] == 0


def test_only_the_same_process_inside_the_job_is_ever_ended(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.children = {41900: 7, 41901: 8, 41902: 9}
    run.outside_job = {41901}

    def reuse_a_pid() -> None:
        # 41902 ended by itself and its number now names an unrelated process.
        run.started[41902] = 999
        run.state["table"][41902] = (1, "unrelated")

    run.after_exit = reuse_a_pid

    assert run.serve() == 0

    assert [action for action in run.actions if action.startswith("kill:")] == ["kill:41900"]
    assert run.receipt["host_restarts"] == 1


def test_a_host_port_still_held_refuses_the_restart(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.port_stays_open = True

    assert run.serve() == 1

    assert run.state["spawned"].count("host") == 1
    assert run.restart_lines[0]["outcome"] == "refused"
    assert run.receipt["failure_category"] == "host_restart_port_held"


@pytest.mark.parametrize(
    ("where", "category"),
    [("store", "host_restart_store_held"), ("record", "host_restart_record_held")],
)
def test_a_lock_the_exited_host_still_holds_refuses_the_restart(
    run: _Run, monkeypatch: pytest.MonkeyPatch, where: str, category: str,
) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.plan.state_root.mkdir(parents=True, exist_ok=True)
    store_lock = run.plan.state_root / "voice-tail-v1.json.lock"
    record_lock = run.plan.output / "state" / "hermes-runs-v1.json.lock"
    store_lock.write_bytes(b"")
    record_lock.write_bytes(b"")
    held = store_lock if where == "store" else record_lock
    armed = {"on": False}
    run.after_exit = lambda: armed.update(on=True)
    monkeypatch.setattr(
        preview_cutover, "_host_lock_held", lambda path: armed["on"] and path == held
    )

    assert run.serve() == 1

    assert run.state["spawned"].count("host") == 1
    assert run.restart_lines[0]["outcome"] == "refused"
    assert run.receipt["failure_category"] == category


def test_a_successor_that_fails_to_become_ready_ends_the_run(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.respawn_fault = "exits_during_start"

    assert run.serve() == 1

    assert run.state["spawned"].count("host") == 2
    assert run.restart_lines == [
        {"category": "host_restart", "outcome": "refused", "restarts": 0, "remaining": 3}
    ]
    assert run.receipt["failure_category"] == "host_exited_during_start"
    assert run.receipt["host_restarts"] == 0


def test_a_long_session_samples_the_process_table_on_a_slow_cadence_not_every_poll(
    run: _Run,
) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.idle_ticks = 400  # two hundred seconds of owner polls at the real interval

    assert run.serve() == 0

    # Four hundred polls with the host alive took one sample between them.
    assert run.snapshots_while_first_host_lived == 1
    assert run.receipt["host_restarts"] == 1


def test_a_child_started_after_the_last_sample_is_still_ended(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.idle_ticks = 5
    run.late_children = {41950: 5}

    assert run.serve() == 0

    assert [action for action in run.actions if action.startswith("kill:")] == ["kill:41950"]
    assert run.receipt["host_restarts"] == 1


def test_a_new_child_that_could_never_be_identified_refuses_the_restart(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.idle_ticks = 2
    run.late_children = {41950: 5}
    run.unobservable = {41950}

    assert run.serve() == 1

    # It was never recorded, so settling could not have known to wait for it.
    assert run.state["spawned"].count("host") == 1
    assert run.receipt["failure_category"] == "host_restart_process_unobservable"



def test_a_grandchild_whose_parent_ended_before_the_host_is_still_ended(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.idle_ticks = 3
    run.children = {41900: 7}
    run.grandchildren = {41901: (41900, 8)}
    run.ends_early = {41900}

    assert run.serve() == 0

    # Nothing leads from the exited host to 41901 any more; only the earlier sample does.
    assert [action for action in run.actions if action.startswith("kill:")] == ["kill:41901"]


def test_a_host_with_more_children_than_the_bound_is_not_restarted(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.children = {42000 + index: 1 for index in range(preview_cutover._HOST_TREE_LIMIT + 1)}

    assert run.serve() == 1

    assert not any(action.startswith("kill:") for action in run.actions)
    assert run.state["spawned"].count("host") == 1
    assert run.restart_lines == [
        {"category": "host_restart", "outcome": "refused", "restarts": 0, "remaining": 3}
    ]
    assert run.receipt["failure_category"] == "host_restart_tree_unbounded"


class _World:
    """A process table and a clock for the tree tracker alone."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.rehearse_desktop_mvp as owner

        self.table: dict[int, tuple[int, str]] = {}
        self.started: dict[int, int] = {}
        self.snapshots = 0
        self.now = 100.0

        def process_table() -> dict[int, tuple[int, str]]:
            self.snapshots += 1
            return dict(self.table)

        monkeypatch.setattr(owner, "process_table", process_table)
        self.unobservable: set[int] = set()

        def started_at(pid: int) -> int | None:
            if pid in self.unobservable:
                raise owner.ProcessUnobservable(pid)
            return self.started.get(pid)

        monkeypatch.setattr(owner, "started_at", started_at)
        monkeypatch.setattr(
            preview_cutover, "time", types.SimpleNamespace(monotonic=lambda: self.now)
        )

    def run(self, pid: int, parent: int, started: int = 1) -> None:
        self.table[pid] = (parent, "image")
        self.started[pid] = started

    def end(self, pid: int) -> None:
        del self.table[pid]
        del self.started[pid]


def test_the_host_tree_is_sampled_at_most_once_per_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _World(monkeypatch)
    world.run(10, 1)
    world.run(11, 10)
    tree = preview_cutover._HostTree()
    interval = preview_cutover._HOST_TREE_SAMPLE_SECONDS

    for _ in range(100):
        tree.sample(10)
        world.now += interval / 101
    assert world.snapshots == 1 and tree.seen == {11: 1}

    world.now += interval
    tree.sample(10)
    assert world.snapshots == 2
    # An exit is sampled at once, whatever the cadence says.
    tree.sample(10, force=True)
    assert world.snapshots == 3
    assert interval >= 5.0


def test_the_host_tree_forgets_what_ended_and_keeps_what_lost_its_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _World(monkeypatch)
    world.run(10, 1)
    world.run(11, 10)
    world.run(12, 11)
    world.run(13, 10)
    world.run(14, 10)
    tree = preview_cutover._HostTree()
    tree.sample(10, force=True)
    assert tree.seen == {11: 1, 12: 1, 13: 1, 14: 1}

    world.end(13)  # ended: forgotten
    world.end(11)  # ended, leaving 12 alive with no path back to the host: kept
    world.started[14] = 99  # ended, and its number now names another process: forgotten
    world.table[14] = (1, "unrelated")
    tree.sample(10, force=True)

    assert tree.seen == {12: 1}


def test_the_host_tree_keeps_a_process_it_can_no_longer_observe_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _World(monkeypatch)
    world.run(10, 1)
    world.run(11, 10)
    world.run(12, 10)
    tree = preview_cutover._HostTree()
    tree.sample(10, force=True)
    assert tree.unobservable is False

    world.unobservable = {11}
    world.run(13, 10)
    world.unobservable.add(13)  # a new child that could never be identified
    tree.sample(10, force=True)

    # Not known to be gone, so not forgotten; and the sample says it was incomplete.
    assert tree.seen == {11: 1, 12: 1}
    assert tree.unobservable is True

    world.unobservable = set()
    tree.sample(10, force=True)
    assert tree.seen == {11: 1, 12: 1, 13: 1}
    assert tree.unobservable is False


def test_a_leftover_child_that_cannot_be_observed_refuses_the_restart(run: _Run) -> None:
    run.exits = [HOST_FATAL_EXIT_STATUS]
    run.idle_ticks = 2
    run.children = {41900: 7}
    run.after_exit = lambda: run.unobservable.add(41900)

    assert run.serve() == 1

    assert not any(action.startswith("kill:") for action in run.actions)
    assert run.state["spawned"].count("host") == 1
    assert run.restart_lines == [
        {"category": "host_restart", "outcome": "refused", "restarts": 0, "remaining": 3}
    ]
    assert run.receipt["failure_category"] == "host_restart_process_unobservable"



def test_the_final_sample_finds_children_of_a_host_that_has_already_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _World(monkeypatch)
    world.run(21, 10)
    world.run(22, 21)
    world.run(30, 1)
    tree = preview_cutover._HostTree()

    tree.sample(10, force=True)

    assert tree.seen == {21: 1, 22: 1}


def test_the_host_tree_is_bounded_and_says_so_instead_of_growing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _World(monkeypatch)
    world.run(10, 1)
    world.run(11, 10)
    tree = preview_cutover._HostTree()
    tree.sample(10, force=True)
    for index in range(preview_cutover._HOST_TREE_LIMIT):
        world.run(1000 + index, 10)

    tree.sample(10, force=True)

    assert tree.unbounded is True
    assert tree.seen == {11: 1}
    assert preview_cutover._HOST_TREE_LIMIT == 64


def test_settling_ends_nothing_whose_number_was_reused_after_the_last_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import scripts.rehearse_desktop_mvp as owner

    killed: list[int] = []
    started = {50: 1, 51: 2, 52: 3}
    unobservable: set[int] = set()

    def started_at(pid: int) -> int | None:
        if pid in unobservable:
            raise owner.ProcessUnobservable(pid)
        return started.get(pid)

    monkeypatch.setattr(owner, "started_at", started_at)
    monkeypatch.setattr(owner, "_kill_tree", lambda pid: (killed.append(pid), started.pop(pid)))
    monkeypatch.setattr(preview_cutover, "_endpoint_snapshot", lambda _plan, ports: {})
    plan = types.SimpleNamespace(
        ports={"host": 1}, state_root=tmp_path / "store", output=tmp_path,
    )
    # 51 is not in this Job; 52 was that process when sampled and is another one now.
    started[52] = 99

    preview_cutover._settle_exited_host(
        plan, {50: 1, 51: 2, 52: 3}, lambda: {50, 52},  # type: ignore[arg-type]
    )

    assert killed == [50]

    # A process of this Job that cannot be observed is not known to be gone: refuse.
    started.update({60: 4, 61: 5})
    unobservable.update({60, 61})
    preview_cutover._settle_exited_host(
        plan, {61: 5}, lambda: {60},  # type: ignore[arg-type]
    )
    with pytest.raises(preview_cutover.PreviewRefusal, match="host_restart_process_unobservable"):
        preview_cutover._settle_exited_host(
            plan, {60: 4}, lambda: {60},  # type: ignore[arg-type]
        )
    assert killed == [50]


def test_capped_processes_respawn_only_replaces_a_root_that_has_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.lookup_backend_logs as lookup_backend_logs

    class Pipe:
        def read1(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            return None

    class Launched:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.stdout = Pipe()
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    class Owner:
        def __init__(self) -> None:
            self.roots: dict[str, Any] = {}

        def _record(self) -> None:
            return None

    monkeypatch.setattr(lookup_backend_logs.subprocess, "Popen", Launched)
    processes = lookup_backend_logs.CappedProcesses(Owner(), tmp_path)

    with pytest.raises(RuntimeError, match="has not exited"):
        processes.respawn("host", [], env={}, log=None)
    first = processes.spawn("host", [], env={}, log=None)
    with pytest.raises(RuntimeError, match="has not exited"):
        processes.respawn("host", [], env={}, log=None)

    first.returncode = HOST_FATAL_EXIT_STATUS
    second = processes.respawn("host", [], env={}, log=None)

    assert processes.roots == {"host": second} and second is not first
    assert sorted(processes.captures) == ["host", "host.exited1"]
    assert (tmp_path / "host.private.log").exists()
    assert (tmp_path / "host.restart1.private.log").exists()
    for thread in processes._drainers:
        thread.join(timeout=5)
    facts = processes.finish()
    # The earlier launch keeps its own exit status; the root's is the successor's.
    assert facts["host.exited1"]["exit_code"] == HOST_FATAL_EXIT_STATUS
    assert facts["host"]["exit_code"] is None


@pytest.mark.skipif(sys.platform != "win32", reason="process identity and trees are Windows")
def test_settling_ends_a_real_process_an_exited_host_left_behind(tmp_path: Path) -> None:
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys, time, pathlib\n"
        "child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'],\n"
        "                         stdin=subprocess.DEVNULL)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')\n"
        "while not pathlib.Path(sys.argv[2]).exists():\n"
        "    time.sleep(0.05)\n"
        f"raise SystemExit({HOST_FATAL_EXIT_STATUS})\n",
        encoding="utf-8",
    )
    child_file, go = tmp_path / "child.pid", tmp_path / "go"
    parent = subprocess.Popen(
        [sys.executable, "-I", str(script), str(child_file), str(go)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 20
        while not child_file.exists() or not child_file.read_text(encoding="ascii"):
            assert time.monotonic() < deadline and parent.poll() is None
            time.sleep(0.05)
        child_pid = int(child_file.read_text(encoding="ascii"))
        watching = preview_cutover._HostTree()
        watching.sample(parent.pid, force=True)
        tree = dict(watching.seen)
        assert child_pid in tree

        go.write_text("", encoding="ascii")
        assert parent.wait(timeout=20) == HOST_FATAL_EXIT_STATUS
        # The host is gone and its child is not: exactly what a host-fatal exit leaves.
        from scripts.rehearse_desktop_mvp import started_at

        assert started_at(child_pid) == tree[child_pid]

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        plan = types.SimpleNamespace(
            ports={"host": free_port}, readiness_policy="dev_owned_services",
            state_root=tmp_path / "store", output=tmp_path,
        )
        preview_cutover._settle_exited_host(plan, tree, lambda: set(tree))  # type: ignore[arg-type]

        assert started_at(child_pid) is None
    finally:
        go.write_text("", encoding="ascii")
        if parent.poll() is None:
            parent.kill()
        if child_pid is not None:
            subprocess.run(
                ("taskkill", "/PID", str(child_pid), "/T", "/F"),
                stdin=subprocess.DEVNULL, capture_output=True, check=False,
            )
