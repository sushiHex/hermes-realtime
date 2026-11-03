from __future__ import annotations

import json
import types
from urllib.error import HTTPError

import pytest

from scripts import preview_cutover
from tests.test_preview_cutover import _plan


class _FakeProcess:
    def __init__(self, pid: int, returncode: int | None = None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def _read_receipt(output, name: str):
    return json.loads((output / name).read_text(encoding="utf-8"))


@pytest.fixture
def dev_service_case(tmp_path, monkeypatch):
    plan_path, profile = _plan(tmp_path)
    raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_plan["readiness_policy"] = "dev_owned_services"
    plan_path.write_text(json.dumps(raw_plan), encoding="utf-8")
    plan = preview_cutover.load_plan(plan_path)
    plan.output.mkdir()
    (plan.output / "state").mkdir()
    (plan.output / "state" / "hermes-runs-v1.json").write_text(
        json.dumps({"version": 1, "pending": [], "admitted": []}), encoding="utf-8"
    )

    watched_ports = [
        plan.ports[name] for name in ("api", "companion", "livekit", "host")
    ] + list(plan.ports["protected"])
    state = {
        "fault": None,
        "tcp": {port: False for port in watched_ports},
        "pids": {},
        "table": {},
        "listener_calls": 0,
        "health_calls": 0,
        "host_http_calls": 0,
        "spawned": [],
    }
    pids = {"gateway": 41001, "livekit": 41002, "host": 41003}
    for port in plan.ports["protected"]:
        state["tcp"][port] = False

    monkeypatch.setattr(preview_cutover.sys, "platform", "win32")
    monkeypatch.setattr(preview_cutover, "_active_plan", lambda _path: (plan, None, profile))
    monkeypatch.setattr(preview_cutover, "preflight", lambda *_args: None)

    def tcp_accepting(port: int) -> bool:
        if port in plan.ports["protected"] and state["fault"] == "protected_tcp_unknown":
            raise OSError("synthetic protected probe unavailable")
        if (state["fault"] == "owned_tcp_unknown_after_stop"
                and state.get("processes_stopped") is True
                and port in [plan.ports[name] for name in ("api", "companion", "livekit", "host")]):
            raise OSError("synthetic owned probe unavailable")
        return state["tcp"].get(port, False)

    monkeypatch.setattr(preview_cutover, "_tcp_accepting", tcp_accepting)

    def kernel_listener_reader(_port):
        state["listener_calls"] += 1
        raise OSError("synthetic kernel listener reader unavailable")

    import scripts.local_livekit as local_livekit
    import scripts.lookup_backend_logs as lookup_backend_logs
    import scripts.lookup_backend_task as lookup_backend_task
    import scripts.preview_owner_launch as owner_launch
    import scripts.rehearse_desktop_mvp as owner

    monkeypatch.setattr(lookup_backend_task, "listener_rows", kernel_listener_reader)
    monkeypatch.setattr(preview_cutover, "_listener_rows", kernel_listener_reader)
    monkeypatch.setattr(owner_launch, "assert_independent_owner", lambda **_kwargs: None)

    class FakeOwnerRefusal(RuntimeError):
        pass

    monkeypatch.setattr(owner_launch, "PreviewOwnerLaunchRefusal", FakeOwnerRefusal)
    monkeypatch.setattr(local_livekit, "verified_server", lambda: plan.executables["livekit"])
    monkeypatch.setattr(local_livekit, "server_command", lambda executable: [str(executable)])

    def livekit_command(executable, port):
        return [str(executable), str(port)]

    monkeypatch.setattr(preview_cutover, "_livekit_command", livekit_command)
    monkeypatch.setattr(preview_cutover, "_gateway_environment", lambda *_args: {})
    monkeypatch.setattr(preview_cutover, "_host_environment", lambda *_args: {})
    monkeypatch.setattr(
        preview_cutover,
        "_runtime_modules",
        lambda _plan: (None, types.SimpleNamespace(
            _load_api_bearer=lambda _env: "synthetic-bearer"
        )),
    )

    class FakeContainment:
        def members(self):
            members = set(state["table"])
            if state["fault"] == "livekit_outside_job":
                members.discard(pids["livekit"])
            return members

    class FakeOwned:
        def __init__(self, _directory):
            self.processes = {}

        def survivors(self, _containment):
            return {}

    class FakeCappedProcesses:
        def __init__(self, owned, _directory):
            self.owned = owned
            self.roots = {}

        def spawn(self, name, _argv, *, env, cwd=None, log=None):
            assert name in pids
            assert type(env) is dict
            pid = pids[name]
            process = _FakeProcess(pid)
            self.roots[name] = process
            self.owned.processes[name] = process
            state["pids"][name] = pid
            state["table"][pid] = (1, name)
            state["spawned"].append(name)
            if name == "gateway":
                state["tcp"][plan.ports["api"]] = True
                state["tcp"][plan.ports["companion"]] = True
            elif name == "livekit":
                state["tcp"][plan.ports["livekit"]] = True
            elif name == "host":
                if state["fault"] == "host_auth_failure":
                    process.returncode = 1
                else:
                    state["tcp"][plan.ports["host"]] = True
                    receipt = {"version": 1, "ready": True, "pid": pid}
                    if state["fault"] == "malformed_host_receipt":
                        receipt["ready"] = "true"
                    elif state["fault"] == "wrong_host_pid":
                        receipt["pid"] = 41999
                    (plan.output / "host-ready.private.json").write_text(
                        json.dumps(receipt), encoding="utf-8"
                    )
            return process

        def stop_all(self, _containment):
            state["processes_stopped"] = True
            for process in self.roots.values():
                process.returncode = 0
            for port in state["tcp"]:
                state["tcp"][port] = False

        def finish(self):
            return {}

    monkeypatch.setattr(owner, "Containment", FakeContainment)
    monkeypatch.setattr(owner, "Processes", FakeOwned)
    monkeypatch.setattr(owner, "process_table", lambda: state["table"])
    monkeypatch.setattr(lookup_backend_logs, "CappedProcesses", FakeCappedProcesses)

    def request_json(url, token, timeout=5):
        state["health_calls"] += 1
        assert url == f"http://127.0.0.1:{plan.ports['api']}/health/detailed"
        assert token == "synthetic-bearer"
        assert timeout == 3
        if state["fault"] == "gateway_auth_failure":
            raise HTTPError(url, 401, "Unauthorized", {}, None)
        return {"pid": 41999 if state["fault"] == "wrong_gateway_pid" else pids["gateway"]}

    monkeypatch.setattr(lookup_backend_task, "request_json", request_json)

    def host_http_ready(port):
        state["host_http_calls"] += 1
        assert port == plan.ports["host"]
        assert state["tcp"][port] is True
        if state["fault"] == "host_http_failure":
            raise preview_cutover.PreviewRefusal("host_http_not_ready")

    monkeypatch.setattr(preview_cutover, "_require_host_http_ready", host_http_ready)

    original_emit = preview_cutover.emit

    def emit_and_stop(event):
        original_emit(event)
        if event.get("category") == "ready":
            (plan.output / "stop.request").write_text("", encoding="utf-8")

    monkeypatch.setattr(preview_cutover, "emit", emit_and_stop)
    return types.SimpleNamespace(plan=plan, profile=profile, state=state, pids=pids)


def test_dev_serve_starts_ready_and_settles_without_kernel_listener_reads(dev_service_case):
    case = dev_service_case
    assert preview_cutover.serve(case.plan.output / "active.private.json") == 0

    ready = _read_receipt(case.plan.output, "ready-receipt.private.json")
    stopped = _read_receipt(case.plan.output, "result-receipt.private.json")
    assert ready["ready"] is True
    assert ready["readiness_policy"] == "dev_owned_services"
    assert ready["listener_ownership"] == "unavailable"
    assert stopped["category"] == "stopped"
    assert stopped["cleanup_settled"] is True
    assert stopped["remaining_accepting_endpoints"] == 0
    assert stopped["listener_ownership"] == "unavailable"
    assert case.state["listener_calls"] == 0
    assert case.state["spawned"] == ["gateway", "livekit", "host"]
    assert case.state["health_calls"] == 1
    assert case.state["host_http_calls"] == 1


@pytest.mark.parametrize(
    ("fault", "failure_category", "spawned"),
    [
        ("wrong_gateway_pid", "gateway_health_owner", ["gateway", "livekit"]),
        ("gateway_auth_failure", "gateway_health_refused", ["gateway", "livekit"]),
        ("livekit_outside_job", "livekit_process_owner", ["gateway", "livekit"]),
        ("malformed_host_receipt", "host_ready_receipt", ["gateway", "livekit", "host"]),
        ("wrong_host_pid", "host_ready_owner", ["gateway", "livekit", "host"]),
        ("host_auth_failure", "host_exited_during_start", ["gateway", "livekit", "host"]),
        ("host_http_failure", "host_http_not_ready", ["gateway", "livekit", "host"]),
    ],
)
def test_dev_serve_refuses_unowned_or_unauthenticated_readiness_before_ready(
    dev_service_case, fault, failure_category, spawned
):
    case = dev_service_case
    case.state["fault"] = fault
    assert preview_cutover.serve(case.plan.output / "active.private.json") == 1

    stopped = _read_receipt(case.plan.output, "result-receipt.private.json")
    assert stopped["category"] == "refused"
    assert stopped["failure_category"] == failure_category
    assert stopped["cleanup_settled"] is True
    assert case.state["spawned"] == spawned
    assert not (case.plan.output / "ready-receipt.private.json").exists()
    assert case.state["listener_calls"] == 0


def test_dev_cleanup_tcp_unknown_is_unsettled_but_not_misreported_as_absent(dev_service_case):
    case = dev_service_case
    case.state["fault"] = "owned_tcp_unknown_after_stop"
    assert preview_cutover.serve(case.plan.output / "active.private.json") == 1

    stopped = _read_receipt(case.plan.output, "result-receipt.private.json")
    assert stopped["category"] == "refused"
    assert stopped["cleanup_failed"] is True
    assert stopped["cleanup_settled"] is False
    assert stopped["remaining_accepting_endpoints"] is None
    assert stopped["owned_survivor_count"] == 0
    assert case.state["listener_calls"] == 0


def test_protected_tcp_unknown_is_optional_metadata_and_does_not_block_dev_readiness(
    dev_service_case,
):
    case = dev_service_case
    case.state["fault"] = "protected_tcp_unknown"
    assert preview_cutover.serve(case.plan.output / "active.private.json") == 0

    ready = _read_receipt(case.plan.output, "ready-receipt.private.json")
    stopped = _read_receipt(case.plan.output, "result-receipt.private.json")
    assert ready["ready"] is True
    assert stopped["category"] == "stopped"
    assert stopped["cleanup_settled"] is True
    assert stopped["protected_endpoints_unchanged"] is None
    assert stopped["remaining_accepting_endpoints"] == 0
    assert case.state["listener_calls"] == 0
