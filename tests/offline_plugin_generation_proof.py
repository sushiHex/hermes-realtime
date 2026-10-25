"""Explicit offline proof runner; requires an owner-supplied pinned Hermes checkout.

Not part of ordinary pytest collection. No core edits, method replacement or
monkeypatches. The fixture is copied only into a disposable test profile.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import linecache
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from pathlib import Path

PROOF_ROOT = Path(__file__).resolve().parent


def load_reviewed(name, root=PROOF_ROOT):
    """Execute the exact reviewed bytes, bypassing finder and bytecode overlays."""
    path = (root / (name + ".py")).resolve()
    payload = path.read_bytes()
    module = types.ModuleType("_proof_" + name)
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(payload).hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--case",
        choices=(
            "healthy",
            "host-initial",
            "five",
            "six",
            "mixed",
            "malformed",
            "bad-query",
            "extra",
            "duplicate-urls",
            "absent-url",
            "four-urls",
            "rewrite",
            "lost-row",
            "overlap",
            "cancel-dispatch",
            "cancel-result",
            "replay",
            "authority",
            "unscoped",
            "readiness",
            "replace-row",
            "reset-refused",
            "builtins",
            "no-opt-in",
            "unsupported-mode",
            "middleware-error",
            "handler-error",
            "unscoped-error",
            "extract-error",
            "mode-guard",
            "http-error",
            "alias-admission",
            "alias-unscoped",
            "alias-cancel-dispatch",
            "alias-cancel-result",
            "alias-status",
            "guard-matrix",
        ),
        default="healthy",
    )
    parser.add_argument("--mutation", default="none")
    parser.add_argument("--private-cleanup-evidence", type=Path)
    parser.add_argument("--worker-profile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-nonce", help=argparse.SUPPRESS)
    return parser.parse_args()


def verify_source(source):
    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=source,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        ).stdout

    pin = "29112bef099274229cadff79cdff7bf7b99c4b77"
    origin = "https://github.com/NousResearch/hermes-agent.git"
    assert git("rev-parse", "HEAD").decode().strip() == pin, "source-head"
    assert git("remote", "get-url", "origin").decode().strip() == origin, "source-origin"
    git("diff", "--quiet", "HEAD", "--", "*.py")
    return {"hermes_commit": pin, "hermes_origin": origin}


def python_manifest(source):
    files = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*.py")
        if not path.relative_to(source).as_posix().startswith("website/")
    ]
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256((source / name).read_bytes()).digest())
    return {
        "hermes_python_files": len(files),
        "hermes_python_digest": digest.hexdigest(),
        "excluded_nonruntime_prefix": "website/",
    }


def write_receipt(receipt, result):
    """Replace evidence only after a complete new snapshot has been written."""
    pending = receipt.with_name(receipt.name + ".pending")
    pending.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, receipt)


def checkpoint(receipt, result, stage, started):
    result["stage"] = stage
    result.setdefault("phase_timings", []).append(
        {"stage": stage, "elapsed_seconds": round(time.monotonic() - started, 3)}
    )
    try:
        write_receipt(receipt, {**result, "ok": False})
    except BaseException as exc:
        result.update(
            receipt_category="proof-receipt-refused", receipt_error_kind=type(exc).__name__
        )
        raise


def audit_record(path, record):
    """A failed protected-effect snapshot cannot leave an acceptable stale zero."""
    try:
        write_receipt(path, record)
    except BaseException:
        os._exit(2)


def process_provenance(roots, hashes):
    """Bounded source identities only; prepared interpreter/dependencies stay trusted."""
    frames = []
    frame = sys._getframe(1)
    incomplete = False
    for _ in range(64):
        if frame is None:
            break
        location = {
            "origin": "unknown-outside",
            "module": "unknown",
            "function": "unknown",
            "line": 0,
            "source_sha256": None,
        }
        try:
            filename = frame.f_code.co_filename
            if filename.startswith("<frozen ") and filename.endswith(">"):
                # Interpreter infrastructure is recorded, never sufficient caller evidence.
                location.update(
                    origin="runtime-internal",
                    module="frozen-interpreter",
                    function=frame.f_code.co_name[:80],
                    line=min(max(frame.f_lineno, 0), 2**31 - 1),
                )
            else:
                module_origin = getattr(frame.f_globals.get("__spec__"), "origin", None)
                path = Path(
                    module_origin
                    if type(module_origin) is str and Path(module_origin).is_absolute()
                    else frame.f_globals.get("__file__", filename)
                ).resolve()
                for origin, root in roots.items():
                    if not path.is_relative_to(root):
                        continue
                    relative = path.relative_to(root).as_posix()
                    if (
                        path.suffix != ".py"
                        or len(relative) > 256
                        or (
                            origin == "stdlib"
                            and {"site-packages", "dist-packages"}.intersection(
                                Path(relative).parts
                            )
                        )
                    ):
                        break
                    if path not in hashes:
                        if len(hashes) >= 128:
                            break
                        with path.open("rb") as stream:
                            payload = stream.read(8 * 1024 * 1024 + 1)
                        if len(payload) > 8 * 1024 * 1024:
                            break
                        hashes[path] = hashlib.sha256(payload).hexdigest()
                    location.update(
                        origin=origin,
                        module=relative,
                        function=frame.f_code.co_name[:80],
                        line=min(max(frame.f_lineno, 0), 2**31 - 1),
                        source_sha256=hashes[path],
                    )
                    break
        except Exception:
            pass
        incomplete |= location["origin"] == "unknown-outside"
        frames.append(location)
        frame = frame.f_back
    return {
        "frames": frames,
        "truncated": frame is not None,
        "incomplete": incomplete or frame is not None,
    }


def cleanup_failure_details(exc, profile):
    """Classify actual OS exception evidence without publishing target paths."""
    detail = {
        "cleanup_target_scope": "unknown",
        "cleanup_target_kind": "unknown",
        "cleanup_operation": "unknown",
        "cleanup_operation_frames": [],
    }
    relative = None
    filename = getattr(exc, "filename", None)
    if type(filename) is str:
        try:
            relative = Path(filename).resolve().relative_to(Path(profile).resolve())
            detail["cleanup_target_scope"] = "owned"
            name = relative.as_posix()
            detail["cleanup_target_kind"] = (
                "source"
                if relative.parts[:1] == ("committed-source",)
                else "log"
                if relative.parts[:1] == ("logs",)
                else "lock"
                if name.endswith(".lock")
                else "database"
                if name.endswith((".db", ".db-wal", ".db-shm"))
                else "other"
            )
        except ValueError:
            detail["cleanup_target_scope"] = "outside"
        except OSError:
            pass
    for key in ("errno", "winerror"):
        value = getattr(exc, key, None)
        detail["cleanup_" + key] = value if type(value) is int and 0 <= value < 2**31 else None
    trusted = {str(Path(tempfile.__file__).resolve()), str(Path(shutil.__file__).resolve())}
    operations = {
        name: getattr(os, name) for name in ("unlink", "remove", "rmdir", "chmod", "scandir")
    }
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        if str(Path(frame.f_code.co_filename).resolve()) not in trusted:
            continue
        detail["cleanup_operation_frames"].append(frame.f_code.co_name)
        line = linecache.getline(frame.f_code.co_filename, lineno)
        for name, function in operations.items():
            if frame.f_locals.get("func") is function or f".{name}(" in line:
                detail["cleanup_operation"] = name
    detail["cleanup_operation_frames"] = detail["cleanup_operation_frames"][-8:]
    target = relative.as_posix() if relative is not None else None
    return detail, {
        "target_relative": target if target is not None and len(target) <= 512 else None
    }


def shutdown_logging():
    """Stop the public queue owner before closing its handlers.

    The drain is best effort; only successful profile deletion qualifies cleanup.
    """
    from hermes_logging import drain_log_queue

    try:
        drain_log_queue(timeout=1.0)
    finally:
        logging.shutdown()


def new_profile():
    # No automatic finalizer: abnormal worker exit must preserve the owned profile.
    name = tempfile.mkdtemp(prefix="hr-plugin-proof-")
    return types.SimpleNamespace(name=name, cleanup=lambda: shutil.rmtree(name))


@contextlib.contextmanager
def proof_profile(receipt, result, started=None, private_cleanup_evidence=None):
    """Keep semantic evidence pending until the disposable profile is removed."""
    if (
        private_cleanup_evidence is not None
        and private_cleanup_evidence.resolve() == receipt.resolve()
    ):
        raise ValueError("cleanup-evidence-path-conflict")
    directory = new_profile()
    profile_started = time.monotonic()
    started = profile_started if started is None else started
    try:
        yield directory.name
    except BaseException as exc:
        result["ok"] = False
        result.setdefault("error_kind", type(exc).__name__)
        raise
    finally:
        result["semantic_ok"] = result["ok"]
        result["proof_stage"] = result["stage"]
        result.update(ok=False, stage="profile-cleanup", profile_cleanup="pending")
        pending_recorded = False
        try:
            checkpoint(receipt, result, "profile-cleanup", started)
            pending_recorded = True
        except BaseException as exc:
            result.update(
                receipt_category="proof-receipt-refused", receipt_error_kind=type(exc).__name__
            )
        cleanup_started = time.monotonic()
        try:
            if result.get("child_settled") is not True:
                result.update(profile_cleanup="preserved", cleanup_category="worker-not-settled")
            else:
                directory.cleanup()
                result.update(
                    ok=result["semantic_ok"] and pending_recorded, profile_cleanup="completed"
                )
        except BaseException as exc:
            result.update(
                ok=False,
                profile_cleanup="refused",
                cleanup_category="profile-cleanup-refused",
                cleanup_error_kind=type(exc).__name__,
            )
            try:
                detail, private_detail = cleanup_failure_details(exc, directory.name)
            except Exception:
                detail, private_detail = {"cleanup_evidence_category": "classification-refused"}, {}
            result.update(detail)
            if private_cleanup_evidence is not None:
                try:
                    write_receipt(
                        private_cleanup_evidence, {**detail, **private_detail, "ok": False}
                    )
                except BaseException as evidence_error:
                    result["private_cleanup_evidence_error_kind"] = type(evidence_error).__name__
        result["profile_elapsed_seconds"] = round(time.monotonic() - profile_started, 3)
        result["profile_cleanup_seconds"] = round(time.monotonic() - cleanup_started, 3)
        result["stage"] = "finished"
        if pending_recorded:
            try:
                write_receipt(receipt, result)
            except BaseException as exc:
                result.update(
                    ok=False,
                    receipt_category="proof-receipt-refused",
                    receipt_error_kind=type(exc).__name__,
                )
        print(
            "[plugin-seam-proof] "
            + json.dumps(
                {
                    "ok": result["ok"],
                    "case": result["case"],
                    "mutation": result["mutation"],
                    "profile_cleanup": result["profile_cleanup"],
                    "category": result.get(
                        "receipt_category",
                        result.get(
                            "cleanup_category",
                            result.get(
                                "category",
                                result.get("assertion", result.get("error_kind", "passed")),
                            ),
                        ),
                    ),
                },
                sort_keys=True,
            )
        )


def worker(args):
    started = time.monotonic()
    args.hermes_source = args.hermes_source.resolve()
    args.receipt = args.receipt.resolve()
    result = {
        "ok": False,
        "stage": "setup",
        "blocked_network_attempts": 0,
        "network_categories": [],
        "stdlib_self_pipes": 0,
        "unclassified_network_attempts": 0,
        "blocked_process_attempts": 0,
        "process_events": [],
        "case": args.case,
        "mutation": args.mutation,
        "complete_enforcement_qualified": False,
        "native_completed": 0,
        "stand_in_calls": 0,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reviewed_source_bindings": reviewed_bindings(),
        "worker_nonce": args.worker_nonce,
    }
    try:
        if not __debug__:
            raise RuntimeError("optimized-mode-unsupported")
        if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
            raise RuntimeError("isolated-no-site-no-bytecode-required")
        checkpoint(args.receipt, result, "source-verification", started)
        result.update(verify_source(args.hermes_source))
        result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except BaseException as exc:
        result.update({"stage": "source-refused", "error_kind": type(exc).__name__})
        if str(exc) in {"optimized-mode-unsupported", "isolated-no-site-no-bytecode-required"}:
            result["category"] = str(exc)
        result["frames"] = [frame.name for frame in traceback.extract_tb(exc.__traceback__)[-6:]]
        write_receipt(args.receipt, result)
        return 1
    with contextlib.ExitStack() as restore:
        restore.callback(os.chdir, Path.cwd())
        home = args.worker_profile
        profile = str(home)
        exported = home / "committed-source"
        preparation_started = time.monotonic()
        try:
            source_helper, result["source_export_helper_sha256"] = load_reviewed(
                "generation_source"
            )
            imports_helper, result["source_import_helper_sha256"] = load_reviewed(
                "generation_imports"
            )
            cases_helper, result["cases_sha256"] = load_reviewed("generation_native_cases")
            result["reviewed_helpers_loading"] = "exact-source-bytes"
            checkpoint(args.receipt, result, "export", started)
            result.update(
                source_helper.export_tree(args.hermes_source, result["hermes_commit"], exported)
            )
            checkpoint(args.receipt, result, "import-bootstrap", started)
            result.update(
                imports_helper.configure_imports(exported, args.hermes_source, PROOF_ROOT)
            )
            checkpoint(args.receipt, result, "python-manifest", started)
            result.update(python_manifest(exported))
            result["source_preparation_seconds"] = round(time.monotonic() - preparation_started, 3)
        except BaseException as exc:
            result.update(stage="source-import-refused", error_kind=type(exc).__name__)
            if "source_helper" in locals() and isinstance(exc, source_helper.SourceExportError):
                result["category"] = exc.category
            return 1
        fixture = PROOF_ROOT / "fixtures" / "plugin_generation_candidate"
        plugin_home = home / "plugins" / "plugin-generation-candidate"
        plugin_home.mkdir(parents=True)
        for filename in ("__init__.py", "policy.py", "entrypoint.py", "plugin.yaml"):
            shutil.copyfile(fixture / filename, plugin_home / filename)
        checks = Path(__file__).with_name("generation_policy_checks.py")
        shutil.copyfile(
            checks, home / "plugins/plugin-generation-candidate/generation_policy_checks.py"
        )
        result["guard_checks_sha256"] = hashlib.sha256(checks.read_bytes()).hexdigest()
        key = "synthetic-test-" + "a7" * 24
        (home / "config.yaml").write_text(
            "model:\n  provider: custom\n  default: synthetic-proof\n"
            "  base_url: http://127.0.0.1:9/v1\n  api_key: synthetic-only\n"
            "  context_length: 131072\n"
            + ("  api_mode: anthropic_messages\n" if args.case == "unsupported-mode" else "")
            + "platform_toolsets:\n  api_server: [proof_budget]\n"
            "plugins:\n  enabled: [plugin-generation-candidate]\n"
            "  entries:\n    plugin-generation-candidate:\n      allow_tool_override: "
            + ("false" if args.case == "no-opt-in" else "true")
            + "\n"
            "      settings:\n        proof_key: " + key + "\n"
            "        host_initial_proof: "
            + ("true" if args.case == "host-initial" else "false")
            + "\n"
            "        real_originals: "
            + ("true" if args.case in {"builtins", "no-opt-in"} else "false")
            + "\n"
            "        terminal_error_probe: "
            + ("true" if args.case == "middleware-error" else "false")
            + "\n"
            "agent:\n  max_turns: 3\n"
            # This fixture stands in primary calls only; disable optional auxiliary inference.
            "auxiliary:\n  title_generation:\n    enabled: false\n"
            "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n",
            encoding="utf-8",
        )
        os.environ["HERMES_HOME"] = profile
        os.environ["HERMES_SKIP_UPDATE_CHECK"] = "1"
        os.environ["HERMES_TERMINAL_SCRIPT_SYNC"] = "0"
        os.environ["HERMES_QUIET"] = "1"
        plugin_source = home / "plugins" / "plugin-generation-candidate" / "__init__.py"
        result["fixture_sha256"] = hashlib.sha256(plugin_source.read_bytes()).hexdigest()
        result["fixture_manifest_sha256"] = hashlib.sha256(
            (plugin_source.parent / "plugin.yaml").read_bytes()
        ).hexdigest()
        policy_source = plugin_source.with_name("policy.py")
        result["policy_sha256"] = hashlib.sha256(policy_source.read_bytes()).hexdigest()
        result["entrypoint_sha256"] = hashlib.sha256(
            plugin_source.with_name("entrypoint.py").read_bytes()
        ).hexdigest()
        if args.mutation != "none":
            mutations, result["mutations_sha256"] = load_reviewed("generation_policy_mutations")

            text = policy_source.read_text(encoding="utf-8")
            old, new = mutations.MUTATIONS[args.mutation]
            assert text.count(old) == 1, "mutation-anchor"
            policy_source.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
        result["effective_policy_sha256"] = hashlib.sha256(policy_source.read_bytes()).hexdigest()
        result["effective_fixture_sha256"] = hashlib.sha256(plugin_source.read_bytes()).hexdigest()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audit_lock = threading.Lock()
        provenance_roots = {
            "hermes": exported.resolve(),
            "fixture": plugin_home.resolve(),
            "proof": PROOF_ROOT.resolve(),
            "dependency": (
                Path(sys.executable).absolute().parent.parent / "Lib/site-packages"
            ).resolve(),
            "stdlib": (Path(sys.base_prefix) / "Lib").resolve(),
        }
        provenance_hashes = {}
        audit_path = args.receipt.with_name(args.receipt.name + ".audit.json")

        def save_audit():
            audit_record(
                audit_path,
                {
                    "worker_nonce": args.worker_nonce,
                    "process_events": result["process_events"],
                    "process_events_truncated": result["blocked_process_attempts"]
                    > len(result["process_events"]),
                    **{
                        name: result[name]
                        for name in (
                            "unclassified_network_attempts",
                            "blocked_network_attempts",
                            "blocked_process_attempts",
                        )
                    },
                },
            )

        save_audit()

        def no_external(event, values):
            if event in {"subprocess.Popen", "os.system"}:
                with audit_lock:
                    result["blocked_process_attempts"] += 1
                    if len(result["process_events"]) < 8:
                        result["process_events"].append(
                            {
                                "event": event,
                                **process_provenance(provenance_roots, provenance_hashes),
                            }
                        )
                    save_audit()
                raise OSError("offline-process-block")
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
                        result["stdlib_self_pipes"] += 1
                        return
                    frame = frame.f_back
                if not (
                    isinstance(address, tuple) and address[0] == "127.0.0.1" and address[1] == port
                ):
                    names = [f.name for f in traceback.extract_stack(limit=18)[:-1]]
                    with audit_lock:
                        result["blocked_network_attempts"] += 1
                        if not any(
                            name in names for name in ("query_ollama_num_ctx", "get_pricing_entry")
                        ):
                            result["unclassified_network_attempts"] += 1
                        if len(result["network_categories"]) < 8:
                            result["network_categories"].append(
                                {
                                    "loopback": isinstance(address, tuple)
                                    and address[0] in {"127.0.0.1", "::1"},
                                    "frames": names,
                                }
                            )
                        save_audit()
                    raise RuntimeError("offline-network-block")

        sys.addaudithook(no_external)
        proof_started = time.monotonic()
        try:
            checkpoint(args.receipt, result, "test", started)
            result.update(loop.run_until_complete(cases_helper.prove(port, key, result, args.case)))
            result["ok"] = True
        except BaseException as exc:
            result["error_kind"] = type(exc).__name__
            result["frames"] = [
                frame.name for frame in traceback.extract_tb(exc.__traceback__)[-6:]
            ]
            if str(exc) in {"offline-process-block", "offline-network-block"}:
                result["category"] = str(exc)
            if isinstance(exc, AssertionError):
                result["assertion"] = str(exc)
        finally:
            try:
                try:
                    checkpoint(args.receipt, result, "finalization", started)
                finally:
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    finally:
                        try:
                            loop.run_until_complete(loop.shutdown_default_executor())
                        finally:
                            shutdown_logging()
            except BaseException as exc:
                result.update(
                    ok=False, category="proof-finalization-refused", error_kind=type(exc).__name__
                )
            loop.close()
            try:
                checkpoint(args.receipt, result, "origin-verification", started)
                result.update(imports_helper.verify_import_origins(exported))
            except BaseException as exc:
                result.update(
                    ok=False, category="source-import-refused", error_kind=type(exc).__name__
                )
            result["native_proof_seconds"] = round(time.monotonic() - proof_started, 3)
    result.update(semantic_ok=result["ok"], ok=False, profile_cleanup="parent-owned-pending")
    write_receipt(args.receipt, result)
    return 0 if result["semantic_ok"] else 1


def reviewed_bindings():
    names = (
        "offline_plugin_generation_proof.py",
        "generation_native_cases.py",
        "generation_policy_mutations.py",
        "generation_source.py",
        "generation_imports.py",
        "generation_policy_checks.py",
        *(
            "fixtures/plugin_generation_candidate/" + name
            for name in ("__init__.py", "policy.py", "entrypoint.py", "plugin.yaml")
        ),
    )
    return {name: hashlib.sha256((PROOF_ROOT / name).read_bytes()).hexdigest() for name in names}


def classify_denied_processes(audit):
    """Shared #306 fixture rule: classify denials, never authorize a launch."""

    def refuse():
        raise RuntimeError("worker-audit-refused")

    events = audit.get("process_events")
    count = audit.get("blocked_process_attempts")
    if (
        type(count) is not int
        or type(events) is not list
        or not 0 <= count <= 8
        or len(events) != count
        or audit.get("process_events_truncated") is not False
    ):
        refuse()
    platform_hash = "4a3cc68ba7354f8d5cc28c55315a7d77b77dcdbbaa3a092572a636b5fb5b6938"
    aiohttp_hash = "c46842a498a119baed4422d4f0ca12792e48a9fddf269fd5beffcd252805f815"
    probe_hash = "d46c016348a493cf26e1b0ed82c3b935672f5a9d27493d5cab2e04ecf37c34df"
    windows = [
        ("stdlib", "platform.py", name, line, platform_hash)
        for name, line in (
            ("_syscmd_ver", 284),
            ("win32_ver", 371),
            ("uname", 856),
            ("system", 921),
        )
    ]
    windows.append(("dependency", "aiohttp/helpers.py", "<module>", 57, aiohttp_hash))
    chains = {"windows-version": windows}
    for kind, function, line, caller_line in (
        ("python-version", "_python_version_of", 145, 214),
        ("pip-version", "_pip_python_version", 188, 216),
    ):
        chains[kind] = [
            ("hermes", "tools/env_probe.py", name, number, probe_hash)
            for name, number in (
                ("_run", 115),
                (function, line),
                ("_build_probe_line", caller_line),
                ("_probe_worker", 332),
            )
        ]
    classified = dict.fromkeys(chains, 0)
    for event in events:
        if (
            type(event) is not dict
            or event.get("event") != "subprocess.Popen"
            or event.get("incomplete") is not False
            or event.get("truncated") is not False
            or type(event.get("frames")) is not list
            or not 1 <= len(event["frames"]) <= 64
        ):
            refuse()
        sites = []
        for frame in event["frames"]:
            if type(frame) is not dict:
                refuse()
            origin, module = frame.get("origin"), frame.get("module")
            function, line, digest = (
                frame.get("function"),
                frame.get("line"),
                frame.get("source_sha256"),
            )
            if (
                type(module) is not str
                or not 1 <= len(module) <= 256
                or type(function) is not str
                or not 1 <= len(function) <= 80
                or type(line) is not int
                or not 0 <= line < 2**31
            ):
                refuse()
            if origin == "runtime-internal":
                if module != "frozen-interpreter" or digest is not None:
                    refuse()
            elif (
                origin not in {"hermes", "fixture", "proof", "dependency", "stdlib"}
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                refuse()
            sites.append((origin, module, function, line, digest))
        for kind, chain in chains.items():
            if any(sites[index : index + len(chain)] == chain for index in range(len(sites))):
                classified[kind] += 1
                break
        else:
            refuse()
    return classified


def run_owned_worker(args, home, result, owner, processes, started):
    """Keep the exact child handle through exit, then validate its evidence and Job."""
    receipt = args.receipt.with_name(args.receipt.name + ".worker.json")
    if receipt.exists():
        raise RuntimeError("worker-receipt-exists")
    nonce = secrets.token_hex(16)
    expected = reviewed_bindings()
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(Path(__file__).resolve()),
        "--hermes-source",
        str(args.hermes_source),
        "--receipt",
        str(receipt),
        "--case",
        args.case,
        "--mutation",
        args.mutation,
        "--worker-profile",
        str(home),
        "--worker-nonce",
        nonce,
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))
    }
    with subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ) as child:
        try:
            remaining = 60 - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 60)
            exit_code = child.wait(timeout=remaining)
        except BaseException:
            child.kill()
            child.wait(timeout=max(0.001, 60 - (time.monotonic() - started)))
            raise
        result["worker_exit_code"] = exit_code
        observed = json.loads(receipt.read_bytes())
        result["worker_nonce_match"] = (
            type(observed) is dict and observed.get("worker_nonce") == nonce
        )
        # The worker is never an overall-success authority, even on normal exit.
        if (
            type(observed) is not dict
            or observed.get("ok") is not False
            or type(observed.get("semantic_ok")) is not bool
            or observed.get("worker_nonce") != nonce
            or observed.get("case") != args.case
            or observed.get("mutation") != args.mutation
            or observed.get("reviewed_source_bindings") != expected
        ):
            raise RuntimeError("worker-evidence-refused")
        loaded = {
            "runner_sha256": "offline_plugin_generation_proof.py",
            "source_export_helper_sha256": "generation_source.py",
            "source_import_helper_sha256": "generation_imports.py",
            "cases_sha256": "generation_native_cases.py",
            "guard_checks_sha256": "generation_policy_checks.py",
            "fixture_sha256": "fixtures/plugin_generation_candidate/__init__.py",
            "policy_sha256": "fixtures/plugin_generation_candidate/policy.py",
            "entrypoint_sha256": "fixtures/plugin_generation_candidate/entrypoint.py",
            "fixture_manifest_sha256": "fixtures/plugin_generation_candidate/plugin.yaml",
        }
        if args.mutation != "none":
            loaded["mutations_sha256"] = "generation_policy_mutations.py"
        if any(observed.get(field) != expected.get(name) for field, name in loaded.items()):
            raise RuntimeError("worker-loaded-source-refused")
        members = owner.members()
        result.update(
            worker_job_member_count=len(members), worker_job_self_present=os.getpid() in members
        )
        left = processes.survivors(owner)
        result["worker_survivor_count"] = sum(left.values())
        if left or processes.cleanup_stop_failed or processes.first_cleanup_refusal is not None:
            raise RuntimeError("worker-job-not-settled")
        # Abnormal exits preserve the profile even when the process has settled.
        if exit_code not in (0, 1) or (exit_code == 0) is not observed["semantic_ok"]:
            raise RuntimeError("worker-exit-refused")
        if exit_code != 0:
            result.update(observed, ok=False)
            raise RuntimeError("worker-semantic-refused")
        try:
            audit = json.loads(receipt.with_name(receipt.name + ".audit.json").read_bytes())
        except (OSError, ValueError):
            raise RuntimeError("worker-audit-refused") from None
        count_names = (
            "unclassified_network_attempts",
            "blocked_network_attempts",
            "blocked_process_attempts",
        )
        if (
            type(audit) is not dict
            or audit.get("worker_nonce") != nonce
            or any(
                type(audit.get(name)) is not int or not 0 <= audit[name] < 2**31
                for name in count_names
            )
        ):
            raise RuntimeError("worker-audit-refused")
        result["worker_audit_counts"] = {name: audit[name] for name in count_names}
        if audit["unclassified_network_attempts"]:
            raise RuntimeError("worker-audit-refused")
        result["classified_process_denials"] = classify_denied_processes(audit)
        result.update(
            observed,
            ok=True,
            child_settled=True,
            worker_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )


def worker_refusal_category(exc):
    named = {
        "worker-receipt-exists",
        "worker-evidence-refused",
        "worker-loaded-source-refused",
        "worker-job-not-settled",
        "worker-exit-refused",
        "worker-semantic-refused",
        "worker-audit-refused",
    }
    return str(exc) if type(exc) is RuntimeError and str(exc) in named else "worker-refused"


def main():
    args = arguments()
    args.hermes_source = args.hermes_source.resolve()
    args.receipt = args.receipt.resolve()
    if args.worker_profile is not None:
        return worker(args)
    started = time.monotonic()
    result = dict(
        ok=False,
        child_settled=False,
        stage="controller",
        case=args.case,
        mutation=args.mutation,
        complete_enforcement_qualified=False,
    )
    try:
        if not __debug__:
            raise RuntimeError("optimized-mode-unsupported")
        if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
            raise RuntimeError("isolated-no-site-no-bytecode-required")
        if os.name != "nt":
            raise RuntimeError("windows-controller-required")
        scripts = PROOF_ROOT.parent / "scripts"
        owner_bindings = {}
        for name in ("local_livekit", "real_gate_support", "rehearse_desktop_mvp"):
            module, owner_bindings[name] = load_reviewed(name, scripts)
            sys.modules[name] = module
        owner = module.Containment()  # This hidden controller joins; never close its Job here.
        result["owner_source_bindings"] = owner_bindings
        with proof_profile(args.receipt, result, started, args.private_cleanup_evidence) as home:
            processes = module.Processes(Path(home))
            try:
                checkpoint(args.receipt, result, "worker", started)
                run_owned_worker(args, home, result, owner, processes, started)
            except BaseException as exc:
                result.update(
                    ok=False,
                    child_settled=False,
                    category=worker_refusal_category(exc),
                    error_kind=type(exc).__name__,
                )
    except BaseException as exc:
        result.update(ok=False, category="controller-refused", error_kind=type(exc).__name__)
        if str(exc) in {
            "optimized-mode-unsupported",
            "isolated-no-site-no-bytecode-required",
            "windows-controller-required",
        }:
            result["category"] = str(exc)
        write_receipt(args.receipt, result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
