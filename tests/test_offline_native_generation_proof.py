"""Deterministic owner sequencing checks; no native child or provider is launched."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "native_owner_proof", Path(__file__).with_name("offline_native_generation_proof.py")
)
proof = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(proof)


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "metadata",
        "unknown",
        "incomplete",
        "stale",
        "overflow",
        "missing",
        "malformed",
        "network",
    ],
)
def test_lifetime_audit_uses_shared_classification(tmp_path, fault):
    shared = proof.load_source("offline_plugin_generation_proof", proof.ROOT / "tests")
    fixtures = proof.load_source("test_generation_runner_imports", proof.ROOT / "tests")
    event = fixtures.passive_metadata_event()
    audit = dict(
        worker_nonce="expected",
        blocked_network_attempts=6,
        unclassified_network_attempts=0,
        blocked_process_attempts=0 if fault == "none" else 1,
        process_events=[] if fault == "none" else [event],
        process_events_truncated=False,
    )
    if fault == "unknown":
        event["frames"][0]["function"] = "unclassified"
    elif fault == "incomplete":
        event["incomplete"] = True
    elif fault == "stale":
        audit["worker_nonce"] = "stale"
    elif fault == "overflow":
        audit["process_events_truncated"] = True
    elif fault == "malformed":
        audit["blocked_network_attempts"] = False
    elif fault == "network":
        audit["unclassified_network_attempts"] = 1
    path = tmp_path / "audit.json"
    if fault != "missing":
        path.write_text(json.dumps(audit), encoding="utf-8")
    if fault in {"none", "metadata"}:
        _, classified = proof.inspect_native_audit(path, "expected", shared)
        assert sum(classified.values()) == (fault == "metadata")
    else:
        with pytest.raises((RuntimeError, OSError)):
            proof.inspect_native_audit(path, "expected", shared)


@pytest.fixture
def scenario():
    events = []
    state = {"ready": True}
    old = SimpleNamespace(poll=lambda: 0)
    containment = SimpleNamespace(members=lambda: {os.getpid()})

    def stop(_containment):
        assert state["ready"] is False
        events.append("stop")
        return 10.0

    def survivors(_containment):
        events.append("survivors")
        return {}

    owner = SimpleNamespace(
        stop_all=stop,
        survivors=survivors,
        cleanup_stop_failed=False,
        first_cleanup_refusal=None,
    )
    previous = {"generation": "1" * 32, "policy": "ordinary-lookup/v1", "ready": True}
    expected = {
        "generation": "2" * 32,
        "policy": "ordinary-lookup/v1",
        "ready": True,
        "nonce": "fresh-child",
    }
    child = object()

    def publish(ready):
        state["ready"] = ready
        events.append("open" if ready else "close")

    def start():
        assert state["ready"] is False
        events.append("start")
        return child

    def readiness(actual):
        assert actual is child and state["ready"] is False
        events.append("readiness")
        return expected

    def listeners(port, *, deadline):
        assert port == 12345 and deadline == 10.0
        events.append("listeners")
        return []

    return SimpleNamespace(
        owner=owner,
        containment=containment,
        old=old,
        previous=previous,
        expected=expected,
        child=child,
        publish=publish,
        start=start,
        readiness=readiness,
        listeners=listeners,
        events=events,
        state=state,
    )


def replace(scenario, **overrides):
    values = dict(
        owner=scenario.owner,
        containment=scenario.containment,
        old=scenario.old,
        port=12345,
        previous=scenario.previous,
        nonce="fresh-child",
        publish=scenario.publish,
        start=scenario.start,
        readiness=scenario.readiness,
    )
    values.update(overrides)
    with patch.object(proof, "listeners", scenario.listeners):
        return proof.replace_generation(**values)


def test_healthy_replacement_orders_settlement_before_start(scenario):
    assert replace(scenario) is scenario.child
    assert scenario.events == [
        "close",
        "stop",
        "survivors",
        "listeners",
        "start",
        "readiness",
        "open",
    ]
    assert scenario.state["ready"] is True


@pytest.mark.parametrize("boundary", ["replacement", "final"])
def test_exited_job_member_uses_existing_survivor_verdict(scenario, boundary):
    # A retained exited process object may remain listed; survivors owns liveness.
    scenario.containment.members = lambda: {os.getpid(), 999999}
    if boundary == "replacement":
        assert replace(scenario) is scenario.child
    else:
        scenario.state["ready"] = False
        with patch.object(proof, "listeners", scenario.listeners):
            proof.settle(scenario.owner, scenario.containment, [scenario.old], 12345)
    assert "survivors" in scenario.events and "listeners" in scenario.events


@pytest.mark.parametrize(
    "fault",
    [
        "containment",
        "prior-refusal",
        "stop-failed",
        "root",
        "survivors",
        "cleanup-refusal",
        "listeners",
    ],
)
def test_unsafe_old_generation_cannot_start_replacement(scenario, fault):
    values = {}
    if fault == "containment":
        values["containment"] = None
    elif fault == "prior-refusal":
        scenario.owner.first_cleanup_refusal = {"unknown": 1}
    elif fault == "stop-failed":
        scenario.owner.cleanup_stop_failed = True
    elif fault == "root":
        scenario.old.poll = lambda: None
    elif fault == "survivors":
        scenario.owner.survivors = lambda _: {"unknown": 1}
    elif fault == "cleanup-refusal":

        def survivors(_):
            scenario.owner.first_cleanup_refusal = {"unknown": 1}
            return {}

        scenario.owner.survivors = survivors
    else:
        scenario.listeners = lambda *_args, **_kwargs: ["occupied"]
    with pytest.raises(proof.Refusal):
        replace(scenario, **values)
    assert "start" not in scenario.events and scenario.state["ready"] is False
    if fault in {"containment", "prior-refusal", "stop-failed"}:
        assert "stop" not in scenario.events


def test_listener_failure_keeps_intake_closed(scenario):
    def failed(*_args, **_kwargs):
        raise OSError("synthetic")

    scenario.listeners = failed
    with pytest.raises(OSError):
        replace(scenario)
    assert "start" not in scenario.events and scenario.state["ready"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", "1" * 32),
        ("generation", "bad"),
        ("policy", "other"),
        ("ready", False),
        ("nonce", "old-child"),
    ],
)
def test_stale_or_unbound_readiness_never_reopens_intake(scenario, field, value):
    scenario.expected[field] = value
    with pytest.raises(proof.Refusal):
        replace(scenario)
    assert "start" in scenario.events and scenario.state["ready"] is False


def test_stop_exception_keeps_intake_closed(scenario):
    def failed(_):
        raise OSError("synthetic")

    scenario.owner.stop_all = failed
    with pytest.raises(OSError):
        replace(scenario)
    assert "start" not in scenario.events and scenario.state["ready"] is False


@pytest.mark.parametrize("settled,refusal", [(False, False), (True, True)])
def test_unsettled_profile_is_preserved(tmp_path, settled, refusal):
    home = tmp_path / "native-owned-generation-test"
    home.mkdir()
    (home / "evidence").write_bytes(b"synthetic")
    outcome = {"ok": True, "owned_process_cleanup": settled}
    if refusal:
        outcome["cleanup_category"] = "synthetic"
    with patch.object(proof.tempfile, "gettempdir", return_value=str(tmp_path)):
        proof.finish_profile(home, outcome)
    assert home.is_dir() and (home / "evidence").read_bytes() == b"synthetic"
    assert outcome["ok"] is False and outcome["profile_cleanup"] == "preserved-unsettled"


@pytest.mark.parametrize("semantic_ok", [False, True])
def test_settled_profile_cleanup_control(tmp_path, semantic_ok):
    home = tmp_path / "native-owned-generation-test"
    home.mkdir()
    outcome = {"ok": semantic_ok, "owned_process_cleanup": True}
    with patch.object(proof.tempfile, "gettempdir", return_value=str(tmp_path)):
        proof.finish_profile(home, outcome)
    assert not home.exists() and outcome["profile_cleanup"] == "completed"
    assert outcome["ok"] is semantic_ok


@pytest.mark.parametrize("invalid", ["parent", "prefix"])
def test_profile_deletion_scope_refuses_invalid_target(tmp_path, invalid):
    home = (
        tmp_path / "nested" / "native-owned-generation-test"
        if invalid == "parent"
        else tmp_path / "unrelated"
    )
    home.mkdir(parents=True)
    outcome = {"ok": True, "owned_process_cleanup": True}
    with patch.object(proof.tempfile, "gettempdir", return_value=str(tmp_path)):
        proof.finish_profile(home, outcome)
    assert home.is_dir() and outcome["ok"] is False
    assert outcome["profile_cleanup"] == "preserved-path-refused"


def native_audit_hook(tmp_path):
    args = SimpleNamespace(
        home=tmp_path,
        exported=tmp_path,
        hermes_source=tmp_path,
        child_receipt=tmp_path / "child.json",
        port=12345,
        nonce="expected",
    )
    helper = SimpleNamespace(configure_imports=lambda *_args: None)
    shared = proof.load_source("offline_plugin_generation_proof", proof.ROOT / "tests")
    hooks = []

    def do_not_run_native(coroutine):
        coroutine.close()

    with (
        patch.object(
            proof,
            "load_source",
            side_effect=lambda name, _directory: (
                shared if name == "offline_plugin_generation_proof" else helper
            ),
        ),
        patch.object(proof.sys, "addaudithook", side_effect=hooks.append),
        patch.object(proof.asyncio, "run", side_effect=do_not_run_native),
        patch.dict(proof.os.environ, {"HERMES_NATIVE_PROOF_KEY": "synthetic"}),
    ):
        assert proof.native_child(args) == 0
    return hooks[0]


def test_native_os_system_attempt_is_denied_and_retained(tmp_path):
    hook = native_audit_hook(tmp_path)
    with pytest.raises(OSError):
        hook("os.system", (b"synthetic-never-executed",))
    audit = json.loads((tmp_path / "child.audit.json").read_bytes())
    assert audit["blocked_process_attempts"] == 1
    assert audit["worker_nonce"] == "expected"
    assert audit["process_events"][0]["event"] == "os.system"


def test_version_probe_handles_retained_denials(tmp_path):
    import platform
    import subprocess

    hook = native_audit_hook(tmp_path)

    def denied(*_args, **_kwargs):
        hook("subprocess.Popen", ())
        raise AssertionError("launch-barrier-returned")

    with patch.object(subprocess, "check_output", side_effect=denied) as blocked:
        assert platform._syscmd_ver("Windows", "unknown", "unknown") == (
            "Windows",
            "unknown",
            "unknown",
        )
    audit = json.loads((tmp_path / "child.audit.json").read_bytes())
    assert blocked.call_count == 3 and audit["blocked_process_attempts"] == 3


def test_early_child_failure_preserves_primary_error(tmp_path):
    args = SimpleNamespace(
        home=tmp_path,
        receipt=tmp_path / "controller.json",
        expected_head="reviewed",
        hermes_python=Path("python.exe"),
        hermes_source=tmp_path,
        port=12345,
    )
    shared = proof.load_source("offline_plugin_generation_proof", proof.ROOT / "tests")
    root = SimpleNamespace(poll=lambda: 1)

    def spawn(_name, command, **_kwargs):
        receipt = Path(command[command.index("--child-receipt") + 1])
        nonce = command[command.index("--nonce") + 1]
        receipt.with_suffix(".audit.json").write_text(
            json.dumps(
                dict(
                    worker_nonce=nonce,
                    blocked_network_attempts=0,
                    unclassified_network_attempts=0,
                    blocked_process_attempts=0,
                    process_events=[],
                    process_events_truncated=False,
                )
            ),
            encoding="utf-8",
        )
        return root

    owner = SimpleNamespace(
        spawn=spawn,
        stop_all=lambda _: 10.0,
        survivors=lambda _: {},
        cleanup_stop_failed=False,
        first_cleanup_refusal=None,
    )
    module = SimpleNamespace(
        Processes=lambda _: owner,
        Containment=lambda: object(),
        listeners=lambda *_a, **_k: [],
    )
    with (
        patch.object(proof, "verify_repository", return_value={"source_commit": "reviewed"}),
        patch.object(
            proof,
            "load_source",
            side_effect=lambda name, _: (
                shared if name == "offline_plugin_generation_proof" else module
            ),
        ),
        patch.object(proof, "prepare_profile", return_value=(tmp_path, {})),
        patch.object(proof, "wait_ready", side_effect=proof.Refusal("native-child-exited")),
        patch.dict(proof.os.environ, {"HERMES_NATIVE_PROOF_KEY": "synthetic"}),
    ):
        assert proof.controller(args) == 1
    outcome = json.loads(args.receipt.read_bytes())
    assert outcome["ok"] is False and outcome["category"] == "native-child-exited"
    assert outcome["owned_process_cleanup"] is True and "cleanup_category" not in outcome
