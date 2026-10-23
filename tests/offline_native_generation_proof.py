"""Explicit model-free native-Hermes generation/owner witness, Windows only.

Run only from a reviewed clean commit after the owner admits one attempt. This
composes the existing owner and frozen synthetic plugin; it is not product setup,
a restart service, or a substitute for #306 usefulness and #159 voice acceptance.
The controller wait is 120s. The launch mode MUST itself be run by the existing
hidden, DEVNULL stdlib subprocess boundary with timeout=180 and check=True. For
example, subprocess.run([python, "-I", "-S", "-B", script, *args], timeout=180,
stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
check=True, creationflags=subprocess.CREATE_NO_WINDOW) on Windows. That caller
is the actual outer deadline, including preparation and temporary-file cleanup;
the 120s controller wait alone is not an overall wall-clock bound. A timeout is
a failed attempt with unknown unfinished cleanup, never permission to retry.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = "ordinary-lookup/v1"
HERMES = "29112bef099274229cadff79cdff7bf7b99c4b77"
EXECUTED_SOURCES = {}


class Refusal(RuntimeError):
    pass


def listeners(*_args, **_kwargs):
    raise Refusal("owner-not-loaded")


def replace_generation(
    *, owner, containment, old, port, previous, nonce, publish, start, readiness
):
    """Retain the controller Job and close intake through actual old-root settlement."""
    publish(False)
    if containment is None:
        raise Refusal("containment-required")
    if owner.cleanup_stop_failed or owner.first_cleanup_refusal is not None:
        raise Refusal("prior-cleanup-refusal")
    deadline = owner.stop_all(containment)
    if old.poll() is None:
        raise Refusal("old-root-not-signaled")
    survivors = owner.survivors(containment)
    if survivors or owner.cleanup_stop_failed or owner.first_cleanup_refusal is not None:
        raise Refusal("old-generation-unsettled")
    if listeners(port, deadline=deadline):
        raise Refusal("old-listener-present")
    child = start()
    actual = readiness(child)
    if (
        type(actual) is not dict
        or actual.get("ready") is not True
        or actual.get("policy") != "ordinary-lookup/v1"
        or type(actual.get("generation")) is not str
        or re.fullmatch(r"[0-9a-f]{32}", actual["generation"]) is None
        or actual["generation"] == previous["generation"]
        or actual.get("nonce") != nonce
    ):
        raise Refusal("fresh-policy-readiness-required")
    publish(True)
    return child


def load_source(name, directory):
    path = directory / (name + ".py")
    payload = path.read_bytes()
    EXECUTED_SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def audit_record(path, counts):
    """Retain a fixed category/count snapshot even when this child is stopped."""
    temporary = path.with_suffix(".pending")
    try:
        temporary.write_text(json.dumps(counts, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        os._exit(2)  # Missing refusal evidence cannot leave this child running.


def verify_repository(expected):
    def git(*argv):
        return (
            subprocess.run(
                ["git", *argv],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            .stdout.decode()
            .strip()
        )

    if git("remote", "get-url", "origin") != "https://github.com/sushiHex/hermes-realtime.git":
        raise Refusal("source-origin")
    if not re.fullmatch(r"[0-9a-f]{40}", expected) or git("rev-parse", "HEAD") != expected:
        raise Refusal("source-head")
    git("diff", "--quiet", "HEAD")
    return {"source_commit": expected, "origin_verified": True}


def request(port, key, path, *, body=None, headers=None, timeout=1.0):
    supplied = {"Authorization": "Bearer " + key, **(headers or {})}
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        supplied["Content-Type"] = "application/json"
    query = urllib.request.Request(
        f"http://127.0.0.1:{port}" + path,
        data=data,
        headers=supplied,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(query, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        payload = response.read(65537)
        if len(payload) > 65536:
            raise Refusal("response-bound")
        return response.status, json.loads(payload), response.headers


def available_ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def wait_ready(process, port, key, receipt, nonce):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise Refusal("native-child-exited")
        if receipt.is_file():
            binding = json.loads(receipt.read_bytes())
            status, ready, _ = request(port, key, "/lookup-policy/ready")
            if status != 200 or binding["nonce"] != nonce or ready != binding["ready"]:
                raise Refusal("native-readiness-binding")
            return {**ready, "nonce": nonce}
        time.sleep(0.05)
    raise Refusal("native-readiness-timeout")


def sentinel(port, nonce):
    def deny(event, _args):
        if event in {"subprocess.Popen", "socket.connect", "os.system"}:
            raise Refusal("sentinel-effect-denied")

    sys.addaudithook(deny)
    payload = json.dumps({"nonce": nonce}).encode()
    with socket.socket() as server:
        server.bind(("127.0.0.1", port))
        server.listen(4)
        while True:
            connection, _ = server.accept()
            with connection:
                connection.sendall(payload)


def sentinel_matches(process, port, nonce):
    if process is not None and process.poll() is not None:
        return False
    with socket.create_connection(("127.0.0.1", port), timeout=1) as connection:
        payload = connection.recv(1025)
    return len(payload) <= 1024 and json.loads(payload) == {"nonce": nonce}


def native_child(args):
    """Serve actual pinned Runs with only the existing synthetic plugin effects."""
    helper = load_source("generation_imports", ROOT / "tests")
    helper.configure_imports(args.exported, args.hermes_source, ROOT / "tests")
    os.environ.update(
        HERMES_HOME=str(args.home),
        HERMES_SKIP_UPDATE_CHECK="1",
        HERMES_TERMINAL_SCRIPT_SYNC="0",
        HERMES_QUIET="1",
    )
    key = os.environ["HERMES_NATIVE_PROOF_KEY"]
    audit_path = args.child_receipt.with_suffix(".audit.json")
    blocked = {"metadata": 0, "unclassified": 0, "subprocess": 0}
    audit_lock = threading.Lock()
    audit_record(audit_path, blocked)

    def deny(event, values):
        if event in {"subprocess.Popen", "os.system"}:
            with audit_lock:
                blocked["subprocess"] += 1
                audit_record(audit_path, blocked)
            raise Refusal("native-process-denied")
        if event == "socket.connect":
            address = values[1]
            frame = sys._getframe(1)
            while frame is not None:
                if (
                    frame.f_code.co_filename == socket.__file__
                    and frame.f_code.co_name == "_fallback_socketpair"
                    and isinstance(address, tuple)
                    and address[0] == "127.0.0.1"
                ):
                    return
                frame = frame.f_back
            if not (isinstance(address, tuple) and address == ("127.0.0.1", args.port)):
                names = {frame.name for frame in traceback.extract_stack(limit=18)}
                category = (
                    "metadata"
                    if names & {"query_ollama_num_ctx", "get_pricing_entry"}
                    else "unclassified"
                )
                with audit_lock:
                    blocked[category] += 1
                    audit_record(audit_path, blocked)
                raise Refusal("native-network-denied")

    sys.addaudithook(deny)

    async def serve():
        from aiohttp import ClientSession
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().discover_and_load()
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": args.port,
                    "key": key,
                    "model_name": "synthetic-proof",
                },
            )
        )
        try:
            if not await adapter.connect():
                raise Refusal("native-connect")
            async with (
                ClientSession(headers={"Authorization": "Bearer " + key}) as client,
                client.get(f"http://127.0.0.1:{args.port}/lookup-policy/ready") as reply,
            ):
                ready = await reply.json()
                if reply.status != 200 or ready.get("ready") is not True:
                    raise Refusal("native-plugin-not-ready")
            origins = helper.verify_import_origins(args.exported)
            with args.child_receipt.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "nonce": args.nonce,
                        "ready": ready,
                        "executed_sources": EXECUTED_SOURCES,
                        **origins,
                    },
                    stream,
                )
            await asyncio.sleep(120)
        finally:
            await adapter.disconnect()

    asyncio.run(serve())
    return int(bool(blocked["unclassified"] or blocked["subprocess"]))


def prepare_profile(args, home, key):
    frozen = load_source("offline_plugin_generation_proof", ROOT / "tests")
    binding = frozen.verify_source(args.hermes_source)
    exporter = load_source("generation_source", ROOT / "tests")
    exported = home / "committed-source"
    binding.update(exporter.export_tree(args.hermes_source, HERMES, exported))
    import shutil

    plugin = home / "plugins/plugin-generation-candidate"
    plugin.mkdir(parents=True)
    fixture = ROOT / "tests/fixtures/plugin_generation_candidate"
    for name in ("__init__.py", "policy.py", "entrypoint.py", "plugin.yaml"):
        shutil.copyfile(fixture / name, plugin / name)
    binding["fixture_sha256"] = {
        name: hashlib.sha256((plugin / name).read_bytes()).hexdigest()
        for name in ("__init__.py", "policy.py", "entrypoint.py", "plugin.yaml")
    }
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  default: synthetic-proof\n"
        "  base_url: http://127.0.0.1:9/v1\n  api_key: synthetic-only\n"
        "  context_length: 131072\nplatform_toolsets:\n  api_server: [proof_budget]\n"
        "plugins:\n  enabled: [plugin-generation-candidate]\n  entries:\n"
        "    plugin-generation-candidate:\n      allow_tool_override: true\n"
        "      settings:\n        proof_key: " + key + "\n"
        "agent:\n  max_turns: 3\nmemory:\n  memory_enabled: false\n"
        "  user_profile_enabled: false\n",
        encoding="utf-8",
    )
    return exported, binding


def settle(owner, containment, roots, port):
    deadline = owner.stop_all(containment)
    if any(root.poll() is None for root in roots):
        raise Refusal("final-root-not-signaled")
    if (
        owner.survivors(containment)
        or owner.cleanup_stop_failed
        or owner.first_cleanup_refusal is not None
    ):
        raise Refusal("final-generation-unsettled")
    if listeners(port, deadline=deadline):
        raise Refusal("final-job-or-listener")


def controller(args):
    global listeners
    binding = verify_repository(args.expected_head)
    for name in ("local_livekit", "real_gate_support", "rehearse_desktop_mvp"):
        owner_module = load_source(name, ROOT / "scripts")
    listeners = owner_module.listeners
    owner = owner_module.Processes(args.home)
    containment = None
    roots = []
    outcome = {"ok": False, "category": "setup", **binding}
    key = os.environ["HERMES_NATIVE_PROOF_KEY"]
    try:
        containment = owner_module.Containment()
        exported, source_binding = prepare_profile(args, args.home, key)
        outcome.update(source_binding)
        env = {
            name: value
            for name, value in os.environ.items()
            if not any(part in name.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))
        }
        env["HERMES_NATIVE_PROOF_KEY"] = key

        def start(nonce):
            receipt = args.home / ("child-" + nonce + ".json")
            process = owner.spawn(
                "native-generation",
                [
                    str(args.hermes_python),
                    "-I",
                    "-S",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--mode",
                    "child",
                    "--home",
                    str(args.home),
                    "--exported",
                    str(exported),
                    "--hermes-source",
                    str(args.hermes_source),
                    "--port",
                    str(args.port),
                    "--nonce",
                    nonce,
                    "--child-receipt",
                    str(receipt),
                ],
                env=env,
                cwd=ROOT,
                log=Path(os.devnull),
            )
            roots.append(process)
            return process, receipt

        first_nonce = secrets.token_hex(16)
        first, first_receipt = start(first_nonce)
        previous = wait_ready(first, args.port, key, first_receipt, first_nonce)
        status, capabilities, _ = request(args.port, key, "/v1/capabilities")
        if (
            status != 200
            or capabilities.get("features", {}).get("runs_idempotency", {}).get("durable")
            is not True
        ):
            raise Refusal("durable-idempotency-unavailable")
        if not sentinel_matches(None, args.control_port, args.control_nonce):
            raise Refusal("outside-sentinel-lost")
        objective = {
            "input": "Synthetic lifecycle request.",
            "session_id": "synthetic-owner-session",
        }
        replay_key = secrets.token_hex(16)
        started = time.monotonic()
        status, accepted, headers = request(
            args.port,
            key,
            "/v1/runs",
            body=objective,
            headers={
                "X-Lookup-Policy": POLICY,
                "Idempotency-Key": replay_key,
                "X-Proof-Case": "cancel-result",
            },
        )
        if (
            status != 202
            or headers.get("X-Lookup-Generation") != previous["generation"]
            or headers.get("X-Lookup-Policy") != POLICY
            or headers.get("X-Lookup-Run") != accepted["run_id"]
        ):
            raise Refusal("native-admission-binding")
        while time.monotonic() - started < 2:
            _, snapshot, _ = request(args.port, key, "/generation-proof/state")
            if (
                snapshot["waiting"]
                and snapshot["counts"].get(accepted["run_id"], {}).get("search") == 1
            ):
                break
            time.sleep(0.02)
        else:
            raise Refusal("pending-native-worker-unproven")
        request(args.port, key, "/generation-proof/control", body={"action": "lose-authority"})
        closed, _, _ = request(args.port, key, "/lookup-policy/ready")
        if closed != 503:
            raise Refusal("old-policy-intake-not-closed")
        intake = {"ready": True}
        second_nonce = secrets.token_hex(16)
        second_receipt = args.home / ("child-" + second_nonce + ".json")

        def next_start():
            if time.monotonic() - started >= 12:
                raise Refusal("pending-worker-window-exceeded")
            return start(second_nonce)[0]

        fresh_ready = {}

        def next_ready(child):
            observed = wait_ready(child, args.port, key, second_receipt, second_nonce)
            fresh_ready.update(observed)
            return observed

        second = replace_generation(
            owner=owner,
            containment=containment,
            old=first,
            port=args.port,
            previous=previous,
            nonce=second_nonce,
            publish=lambda value: intake.update(ready=value),
            start=next_start,
            readiness=next_ready,
        )
        status, old_replay, _ = request(
            args.port,
            key,
            "/v1/runs",
            body=objective,
            headers={"X-Lookup-Policy": POLICY, "Idempotency-Key": replay_key},
        )
        if status != 409 or old_replay != {"error": "policy_old_generation"}:
            raise Refusal("old-generation-replay-not-refused")
        _, untouched, _ = request(args.port, key, "/generation-proof/state")
        if untouched["counts"] or untouched["budgets"]:
            raise Refusal("old-replay-created-authority")
        status, fresh, fresh_headers = request(
            args.port, key, "/v1/runs", body=objective, headers={"X-Lookup-Policy": POLICY}
        )
        if (
            status != 202
            or fresh["run_id"] == accepted["run_id"]
            or fresh_headers.get("X-Lookup-Generation") != fresh_ready["generation"]
            or fresh_headers.get("X-Lookup-Policy") != POLICY
            or fresh_headers.get("X-Lookup-Run") != fresh["run_id"]
        ):
            raise Refusal("fresh-generation-dispatch")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            _, finished, final_headers = request(args.port, key, "/v1/runs/" + fresh["run_id"])
            if finished["status"] == "completed":
                break
            time.sleep(0.05)
        else:
            raise Refusal("fresh-synthetic-completion")
        _, snapshot, _ = request(args.port, key, "/generation-proof/state")
        counts = snapshot["counts"].get(fresh["run_id"], {})
        if [counts.get(name) for name in ("provider", "search", "extract")] != [
            3,
            1,
            1,
        ] or final_headers.get("X-Lookup-Policy-Refused") != "false":
            raise Refusal("fresh-positive-control")
        if not sentinel_matches(None, args.control_port, args.control_nonce):
            raise Refusal("outside-sentinel-lost")
        if first.poll() is None or second.poll() is not None:
            raise Refusal("native-process-order")
        outcome.update(
            ok=True,
            category="semantic-passed",
            generations=2,
            fresh_policy=True,
            old_replay_refused=True,
            outside_control_preserved=True,
        )
    except BaseException as error:
        outcome.update(
            ok=False, category=str(error) if isinstance(error, Refusal) else type(error).__name__
        )
    finally:
        try:
            if containment is None:
                raise Refusal("containment-required")
            settle(owner, containment, roots, args.port)
            outcome["owned_process_cleanup"] = True
            audits = [
                json.loads(path.with_suffix(".audit.json").read_bytes())
                for path in (first_receipt, second_receipt)
            ]
            if any(counts["unclassified"] or counts["subprocess"] for counts in audits):
                raise Refusal("native-audit-refusal")
            outcome["denied_metadata_attempts"] = sum(counts["metadata"] for counts in audits)
            outcome["retained_unclassified_attempts"] = 0
            outcome["executed_sources"] = EXECUTED_SOURCES
        except BaseException as error:
            outcome.update(ok=False, cleanup_category=type(error).__name__)
        args.receipt.write_text(json.dumps(outcome, sort_keys=True), encoding="utf-8")
    return int(not outcome["ok"])


def launch(args):
    if os.name != "nt":
        raise Refusal("windows-required")
    verify_repository(args.expected_head)
    outcome = {"ok": False, "category": "setup", "full_acceptance": False}
    outside = None
    child = None
    home = Path(tempfile.mkdtemp(prefix="native-owned-generation-"))
    try:
        try:
            port, control_port = available_ports()
            nonce = secrets.token_hex(16)
            outside = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--mode",
                    "sentinel",
                    "--port",
                    str(control_port),
                    "--nonce",
                    nonce,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    if sentinel_matches(outside, control_port, nonce):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise Refusal("outside-sentinel-not-ready")
            env = {
                name: value
                for name, value in os.environ.items()
                if not any(
                    part in name.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")
                )
            }
            env["HERMES_NATIVE_PROOF_KEY"] = secrets.token_hex(32)
            receipt = home / "controller.json"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--mode",
                    "controller",
                    "--home",
                    str(home),
                    "--port",
                    str(port),
                    "--hermes-python",
                    str(args.hermes_python),
                    "--hermes-source",
                    str(args.hermes_source),
                    "--expected-head",
                    args.expected_head,
                    "--receipt",
                    str(receipt),
                    "--control-port",
                    str(control_port),
                    "--control-nonce",
                    nonce,
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            child.wait(timeout=120)
            outcome = json.loads(receipt.read_bytes())
            outcome["controller_exit_ok"] = child.returncode == 0
            outcome["outside_sentinel_preserved"] = sentinel_matches(outside, control_port, nonce)
            outcome["ok"] = (
                outcome["ok"]
                and outcome["controller_exit_ok"]
                and outcome["outside_sentinel_preserved"]
            )
        finally:
            # Stop owned processes before removing their files, including on timeout.
            for process in (child, outside):
                try:
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait(timeout=15)
                except BaseException as error:
                    outcome.update(ok=False, cleanup_category=type(error).__name__)
    except BaseException as error:
        outcome.update(
            ok=False, category=str(error) if isinstance(error, Refusal) else type(error).__name__
        )
    finally:
        finish_profile(home, outcome)
        outcome["full_acceptance"] = False
        args.receipt.write_text(json.dumps(outcome, sort_keys=True), encoding="utf-8")
        print(
            "[native-generation-proof] "
            + json.dumps({"ok": outcome["ok"], "category": outcome["category"]})
        )
    return int(not outcome["ok"])


def finish_profile(home, outcome):
    """Retain files whenever native ownership settlement is unproven."""
    if outcome.get("owned_process_cleanup") is not True or "cleanup_category" in outcome:
        outcome.update(ok=False, profile_cleanup="preserved-unsettled")
        return
    import shutil

    target = home.resolve()
    parent = Path(tempfile.gettempdir()).resolve()
    if target.parent != parent or not target.name.startswith("native-owned-generation-"):
        outcome.update(ok=False, profile_cleanup="preserved-path-refused")
        return
    try:
        shutil.rmtree(target)
        outcome["profile_cleanup"] = "completed"
    except OSError as error:
        outcome.update(ok=False, profile_cleanup="refused", profile_error_kind=type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("launch", "controller", "child", "sentinel"), default="launch"
    )
    for name in ("receipt", "home", "exported", "hermes-python", "hermes-source", "child-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--nonce")
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--control-nonce")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    if not __debug__:
        raise Refusal("optimized-mode-unsupported")
    if args.mode == "child":
        return native_child(args)
    if args.mode == "sentinel":
        sentinel(args.port, args.nonce)
        return 0
    if args.mode == "controller":
        return controller(args)
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
