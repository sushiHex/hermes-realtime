"""Explicit Windows load-only observation for the staged native plugin.

Use a reviewed clean candidate and prepared pinned Hermes interpreter. Launch the
controller hidden with -I -S -B, DEVNULL streams, timeout=70 and check=True. Both
worker waits share an absolute deadline 60 seconds after controller preparation
starts. Settlement and profile deletion use the caller's 70-second outer bound;
total elapsed time may exceed 60 seconds. Existing Processes/Containment ownership
is reused, and the controller never closes its own Job.
No native Runs, model, retrieval, service metadata or user-profile action exists.
This is not a live-task, transport or usefulness qualification.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERMES = "29112bef099274229cadff79cdff7bf7b99c4b77"
FILES = (
    "tests/offline_lookup_deployment_proof.py",
    "scripts/stage_lookup_policy.py",
    "tests/offline_plugin_generation_proof.py",
    "tests/generation_source.py",
    "tests/generation_imports.py",
    "scripts/local_livekit.py",
    "scripts/real_gate_support.py",
    "scripts/rehearse_desktop_mvp.py",
    "tests/fixtures/plugin_generation_candidate/entrypoint.py",
    "tests/fixtures/plugin_generation_candidate/policy.py",
)


def load(relative):
    path = ROOT / relative
    payload = path.read_bytes()
    module = types.ModuleType("_load_only_" + path.stem)
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def bindings(head, git):
    if git(ROOT, "rev-parse", "HEAD").strip().decode() != head:
        raise RuntimeError("candidate-head-refused")
    if git(ROOT, "remote", "get-url", "origin").strip() != (
        b"https://github.com/sushiHex/hermes-realtime.git"
    ):
        raise RuntimeError("candidate-origin-refused")
    result = {}
    for name in FILES:
        payload = (ROOT / name).read_bytes()
        if payload != git(ROOT, "show", head + ":" + name):
            raise RuntimeError("candidate-source-refused")
        result[name] = hashlib.sha256(payload).hexdigest()
    return result


async def observe(stage, profile, consent):
    from aiohttp import web
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager.discover_and_load()
    rows = [row for row in manager.list_plugins() if row["name"] == stage.PLUGIN]
    if len(rows) != 1 or rows[0]["enabled"] is not consent:
        raise RuntimeError("plugin-consent-refused")
    factories = [
        factory
        for factory, name in manager.get_platform_handler_factories("api_server")
        if name == stage.PLUGIN
    ]
    if not consent:
        if factories or rows[0]["error"] is None:
            raise RuntimeError("missing-consent-failed-open")
        return {"consent_refused": True}
    if len(factories) != 1 or (rows[0]["tools"], rows[0]["middleware"]) != (2, 2):
        raise RuntimeError("native-registration-refused")
    factory = factories[0]
    generation = factory.__self__
    if not generation.available() or len(generation.handles) != 4:
        raise RuntimeError("native-handles-refused")
    package = profile / "plugins" / stage.PLUGIN
    loaded = manager._plugins[stage.PLUGIN].module
    for module, name in (
        (loaded, "__init__.py"),
        (sys.modules[loaded.__name__ + ".entrypoint"], "entrypoint.py"),
        (sys.modules[loaded.__name__ + ".policy"], "policy.py"),
    ):
        if Path(module.__file__).resolve() != (package / name).resolve():
            raise RuntimeError("plugin-import-origin-refused")
    app = web.Application()
    factory(app, None)
    if {route.resource.canonical for route in app.router.routes()} != {"/lookup-policy/ready"}:
        raise RuntimeError("plugin-routes-refused")
    route = next(route for route in app.router.routes() if route.method == "GET")
    response = await route.handler(None)
    ready = json.loads(response.body)
    if response.status != 200 or ready != generation.readiness() or ready["ready"] is not True:
        raise RuntimeError("plugin-readiness-refused")
    if ready["policy"] != stage.POLICY or len(ready["generation"]) != 32:
        raise RuntimeError("plugin-readiness-binding-refused")
    return {"native_registration": True, "native_readiness": True}


def worker(args, shared, exporter, expected):
    result = dict(
        ok=False,
        semantic_ok=False,
        nonce=args.nonce,
        consent=args.consent,
        bindings=expected,
        stage="artifact",
    )
    try:
        return worker_body(args, shared, exporter, result)
    except BaseException as error:
        result.update(semantic_ok=False, error_kind=type(error).__name__)
        shared.write_receipt(args.receipt, result)
        return 1


def worker_body(args, shared, exporter, result):
    stage = load("scripts/stage_lookup_policy.py")
    profile = args.profile
    result["artifact"] = stage.stage_lookup_policy(
        profile, ROOT / "tests/fixtures/plugin_generation_candidate"
    )
    result["stage"] = "export"
    shared.write_receipt(args.receipt, result)
    exported = profile / "committed-source"
    result.update(exporter.export_tree(args.hermes_source, HERMES, exported))
    result["stage"] = "import-bootstrap"
    shared.write_receipt(args.receipt, result)
    imports = load("tests/generation_imports.py")
    imports.configure_imports(exported, args.hermes_source, ROOT / "tests")
    # Only load/override consent is selected. No real model, web or credential settings.
    config = {
        "plugins": {
            "enabled": [stage.PLUGIN],
            "entries": {stage.PLUGIN: {"allow_tool_override": args.consent}},
        }
    }
    (profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    os.environ["HERMES_HOME"] = str(profile)
    os.environ["HERMES_SKIP_UPDATE_CHECK"] = "1"
    os.environ["HERMES_TERMINAL_SCRIPT_SYNC"] = "0"
    os.environ.pop("HERMES_ENABLE_PROJECT_PLUGINS", None)
    os.environ.pop("HERMES_BUNDLED_PLUGINS", None)
    roots = {
        "hermes": exported.resolve(),
        "fixture": (profile / "plugins" / stage.PLUGIN).resolve(),
        "proof": ROOT.resolve(),
        "dependency": (
            Path(sys.executable).absolute().parent.parent / "Lib/site-packages"
        ).resolve(),
        "stdlib": (Path(sys.base_prefix) / "Lib").resolve(),
    }
    audit = dict(
        worker_nonce=args.nonce,
        process_events=[],
        process_events_truncated=False,
        blocked_process_attempts=0,
        blocked_network_attempts=0,
        unclassified_network_attempts=0,
    )
    hashes, lock = {}, threading.RLock()
    audit_path = args.receipt.with_name(args.receipt.name + ".audit.json")

    def save():
        shared.audit_record(audit_path, audit)

    def no_external(event, values):
        if event in {"subprocess.Popen", "os.system"}:
            with lock:
                audit["blocked_process_attempts"] += 1
                if len(audit["process_events"]) < 8:
                    audit["process_events"].append(
                        {"event": event, **shared.process_provenance(roots, hashes)}
                    )
                audit["process_events_truncated"] = audit["blocked_process_attempts"] > len(
                    audit["process_events"]
                )
                save()
            raise OSError("load-only-process-denied")
        if event in {"socket.connect", "socket.getaddrinfo"}:
            with lock:
                audit["blocked_network_attempts"] += 1
                audit["unclassified_network_attempts"] += 1
                save()
            raise OSError("load-only-network-denied")

    # Construct only the stdlib event loop/self-pipe before the external-effect barrier.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    save()
    sys.addaudithook(no_external)
    try:
        result["stage"] = "native-plugin-load"
        shared.write_receipt(args.receipt, result)
        result.update(loop.run_until_complete(observe(stage, profile, args.consent)))
        result["stage"] = "loaded-source-verification"
        result.update(imports.verify_import_origins(exported))
        expected_files = result["artifact"]["files_sha256"]
        package = profile / "plugins" / stage.PLUGIN
        if {path.name for path in package.iterdir()} != set(expected_files) or any(
            hashlib.sha256((package / name).read_bytes()).hexdigest() != digest
            for name, digest in expected_files.items()
        ):
            raise RuntimeError("loaded-artifact-integrity-refused")
        result["semantic_ok"] = True
    except BaseException as error:
        result["error_kind"] = type(error).__name__
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
            logging.shutdown()
            save()
            shared.write_receipt(args.receipt, result)
    return int(not result["semantic_ok"])


def validate_worker(observed, expected, nonce, consent):
    if (
        type(observed) is not dict
        or observed.get("ok") is not False
        or observed.get("semantic_ok") is not True
        or observed.get("nonce") != nonce
        or observed.get("consent") is not consent
        or observed.get("bindings") != expected
    ):
        raise RuntimeError("worker-evidence-refused")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--nonce")
    parser.add_argument("--consent", action="store_true")
    args = parser.parse_args()
    if (
        not __debug__
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.dont_write_bytecode
    ):
        raise RuntimeError("isolated-no-site-no-bytecode-required")
    if os.name != "nt" or args.receipt.exists():
        raise RuntimeError("owned-fresh-windows-observation-required")
    shared, exporter = (
        load("tests/offline_plugin_generation_proof.py"),
        load("tests/generation_source.py"),
    )
    started = time.monotonic()
    if args.profile is None:
        for name in ("local_livekit", "real_gate_support", "rehearse_desktop_mvp"):
            module = load("scripts/" + name + ".py")
            sys.modules[name] = module
        owner = module.Containment()
        for suffix in (".granted", ".denied"):
            receipt = args.receipt.with_name(args.receipt.name + suffix)
            if any(
                path.exists()
                for path in (
                    receipt,
                    receipt.with_name(receipt.name + ".worker"),
                    receipt.with_name(receipt.name + ".worker.audit.json"),
                )
            ):
                raise RuntimeError("observation-receipt-not-fresh")
    expected = bindings(args.expected_head, exporter._git)
    if exporter._git(args.hermes_source, "rev-parse", "HEAD").strip().decode() != HERMES:
        raise RuntimeError("hermes-pin-refused")
    if exporter._git(args.hermes_source, "remote", "get-url", "origin").strip() != (
        b"https://github.com/NousResearch/hermes-agent.git"
    ):
        raise RuntimeError("hermes-origin-refused")
    if args.profile is not None:
        return worker(args, shared, exporter, expected)
    cases = []
    for consent in (True, False):
        receipt = args.receipt.with_name(args.receipt.name + (".granted" if consent else ".denied"))
        case = dict(
            ok=False, child_settled=False, stage="loader", case="load-only", mutation="none"
        )
        with shared.proof_profile(receipt, case, started) as home:
            profile = Path(home) / "profile"
            profile.mkdir()
            processes = module.Processes(Path(home))
            nonce = secrets.token_hex(16)
            child_receipt = receipt.with_name(receipt.name + ".worker")
            command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).resolve()),
                "--hermes-source",
                str(args.hermes_source),
                "--expected-head",
                args.expected_head,
                "--receipt",
                str(child_receipt),
                "--profile",
                str(profile),
                "--nonce",
                nonce,
            ]
            if consent:
                command.append("--consent")
            environment = {
                key: value
                for key, value in os.environ.items()
                if not any(
                    part in key.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")
                )
            }
            try:
                child = processes.spawn(
                    "loader", command, env=environment, log=Path(home) / "private.log"
                )
                exit_code = child.wait(timeout=max(0.001, 60 - (time.monotonic() - started)))
                observed = json.loads(child_receipt.read_bytes())
                case["worker"] = observed
                validate_worker(observed, expected, nonce, consent)
                audit = json.loads(
                    child_receipt.with_name(child_receipt.name + ".audit.json").read_bytes()
                )
                if (
                    exit_code != 0
                    or audit.get("worker_nonce") != nonce
                    or audit.get("blocked_network_attempts") != 0
                ):
                    raise RuntimeError("worker-lifetime-refused")
                case["classified_process_denials"] = shared.classify_denied_processes(audit)
                case["worker"] = observed
                case["ok"] = True
            except BaseException as error:
                case.update(ok=False, error_kind=type(error).__name__)
            finally:
                processes.stop_all(owner)
                left = processes.survivors(owner)
                case["child_settled"] = not (
                    left
                    or processes.cleanup_stop_failed
                    or processes.first_cleanup_refusal is not None
                )
                case["ok"] = case["ok"] and case["child_settled"]
        cases.append(case)
        if not case["ok"]:
            break
    result = {
        "ok": len(cases) == 2 and all(case["ok"] for case in cases),
        "cases": cases,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    shared.write_receipt(args.receipt, result)
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
