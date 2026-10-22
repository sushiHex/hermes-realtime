"""Explicit offline proof runner; requires an owner-supplied pinned Hermes checkout.

Not part of ordinary pytest collection. No core edits, method replacement or
monkeypatches. The fixture is copied only into a disposable test profile.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from generation_native_cases import prove


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--case",
        choices=(
            "healthy",
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
        ),
        default="healthy",
    )
    parser.add_argument("--mutation", default="none")
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
    assert not git("ls-files", "--others", "--exclude-standard", "--", "*.py"), "untracked-source"
    files = [
        name
        for name in git("ls-files", "-z", "--", "*.py").decode().split("\0")
        if name and not name.startswith("website/")
    ]
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256((source / name).read_bytes()).digest())
    return {
        "hermes_commit": pin,
        "hermes_origin": origin,
        "hermes_python_files": len(files),
        "hermes_python_digest": digest.hexdigest(),
        "excluded_nonruntime_prefix": "website/",
    }


def main():
    args = arguments()
    result = {
        "ok": False,
        "stage": "setup",
        "blocked_network_attempts": 0,
        "network_categories": [],
        "stdlib_self_pipes": 0,
        "unclassified_network_attempts": 0,
        "case": args.case,
        "mutation": args.mutation,
        "complete_enforcement_qualified": False,
        "native_completed": 0,
        "stand_in_calls": 0,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        if not __debug__:
            raise RuntimeError("optimized-mode-unsupported")
        result.update(verify_source(args.hermes_source))
        result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except BaseException as exc:
        result.update({"stage": "source-refused", "error_kind": type(exc).__name__})
        if str(exc) == "optimized-mode-unsupported":
            result["category"] = "optimized-mode-unsupported"
        result["frames"] = [frame.name for frame in traceback.extract_tb(exc.__traceback__)[-6:]]
        args.receipt.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    with tempfile.TemporaryDirectory(prefix="hr-plugin-proof-") as profile:
        home = Path(profile)
        fixture = Path(__file__).parent / "fixtures" / "plugin_generation_candidate"
        shutil.copytree(fixture, home / "plugins" / "plugin-generation-candidate")
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
            "        real_originals: "
            + ("true" if args.case in {"builtins", "no-opt-in"} else "false")
            + "\n"
            "        terminal_error_probe: "
            + ("true" if args.case == "middleware-error" else "false")
            + "\n"
            "agent:\n  max_turns: 3\n"
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
            from generation_policy_mutations import MUTATIONS

            text = policy_source.read_text(encoding="utf-8")
            old, new = MUTATIONS[args.mutation]
            assert text.count(old) == 1, "mutation-anchor"
            policy_source.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
        result["effective_policy_sha256"] = hashlib.sha256(policy_source.read_bytes()).hexdigest()
        result["cases_sha256"] = hashlib.sha256(
            Path(__file__).with_name("generation_native_cases.py").read_bytes()
        ).hexdigest()
        result["effective_fixture_sha256"] = hashlib.sha256(plugin_source.read_bytes()).hexdigest()
        sys.path.insert(0, str(args.hermes_source))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def no_external(event, values):
            if event == "subprocess.Popen":
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
                    result["blocked_network_attempts"] += 1
                    names = [f.name for f in traceback.extract_stack(limit=18)[:-1]]
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
                    raise RuntimeError("offline-network-block")

        sys.addaudithook(no_external)
        try:
            result["stage"] = "native-proof"
            result.update(loop.run_until_complete(prove(port, key, result, args.case)))
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
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()
            logging.shutdown()
            args.receipt.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            print(
                "[plugin-seam-proof] "
                + json.dumps(
                    {
                        "ok": result["ok"],
                        "case": args.case,
                        "mutation": args.mutation,
                        "category": result.get("assertion", result.get("error_kind", "passed")),
                        "native_completed": result.get("native_completed", 0),
                    },
                    sort_keys=True,
                )
            )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
