"""One explicit task through the production adapter; private bounded evidence only."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import itertools
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import MethodType
from typing import Any
from urllib.error import URLError
from urllib.parse import quote

from scripts.setup_lookup_backend import ROOT, digest, read_json, require


def select_runtime(m: Any) -> Any:
    package = Path(m["packageRoot"]).resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "hermes_realtime" or name.startswith("hermes_realtime."):
            relative = name.split(".")[1:]
            expected = (
                {package / "__init__.py"}
                if not relative
                else {
                    package.joinpath(*relative).with_suffix(".py"),
                    package.joinpath(*relative) / "__init__.py",
                }
            )
            filename = getattr(module, "__file__", None)
            require(
                type(filename) is str and Path(filename).resolve() in expected,
                "foreign_cached_namespace",
            )
    sys.path.insert(0, str(package.parent))
    runtime = importlib.import_module("hermes_realtime")
    require(Path(str(runtime.__file__)).resolve() == package / "__init__.py", "runtime_origin")


async def execute(m: Any, directory: Any) -> Any:
    import aiohttp

    from hermes_realtime.conversation import (
        ConversationContextStore,
        ConversationTaskController,
        ConversationWorkControlSurface,
    )
    from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession

    directory.mkdir(exist_ok=False)
    token = next(
        line.partition("=")[2]
        for line in (Path(m["profile"]) / ".env").read_text().splitlines()
        if line.startswith("API_SERVER_KEY=")
    )
    stream = {"completedEvents": 0, "fallbackEntries": 0}
    observed = {"posts": 0, "accepted": 0}

    def event(kind: Any, _tool: Any) -> Any:
        if kind == "run.completed":
            stream["completedEvents"] = min(2, stream["completedEvents"] + 1)

    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=m["baseUrl"],
            bearer=token,
            request_timeout_seconds=30,
            settlement_timeout_seconds=60,
            lookup_policy="ordinary-lookup/v1",
        ),
        event_observer=event,
        session_id="session_lookup_setup",
        run_record_path=directory / "run-record.private.json",
    )
    original_settle = session._settle_authority

    async def settle(self: Any, authority: Any) -> Any:
        stream["fallbackEntries"] = min(2, stream["fallbackEntries"] + 1)
        return await original_settle(authority)

    session._settle_authority = MethodType(settle, session)  # type: ignore[method-assign]
    original_request = session._request_json

    async def request(self: Any, method: Any, path: Any, **kwargs: Any) -> Any:
        if method == "POST" and path == "/v1/runs":
            observed["posts"] += 1
            require(
                observed["posts"] == 1 and kwargs.get("body", {}).get("input") == m["objective"],
                "single_exact_dispatch",
            )
        response = await original_request(method, path, **kwargs)
        if method == "POST" and path == "/v1/runs" and response[0] == 202:
            observed["accepted"] += 1
        return response

    session._request_json = MethodType(request, session)  # type: ignore[method-assign]
    context = ConversationContextStore(max_item_chars=1024)
    ids = itertools.count(1)
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_lookup_setup",
        id_factory=lambda: f"lookup{next(ids)}",
    )
    surface = ConversationWorkControlSurface(
        controller=controller,
        context=context,
        observer=lambda *_: None,
        reserve_observer_capacity=lambda: None,
    )

    async def snapshot(run_id: Any, name: Any) -> Any:
        async with asyncio.timeout(20):
            async with aiohttp.ClientSession(
                headers={"Authorization": "Bearer " + token}, trust_env=False
            ) as client:

                async def get(path: Any) -> Any:
                    async with client.get(
                        m["baseUrl"] + path,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        require(response.status == 200, "snapshot_status")
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(8192):
                            raw.extend(chunk)
                            require(len(raw) <= 65536, "snapshot_bound")
                        return json.loads(raw)

                run = await get("/v1/runs/" + quote(run_id, safe=""))
                session_id = run.get("session_id")
                require(type(session_id) is str and session_id, "snapshot_session")
                messages = await get(
                    "/api/sessions/"
                    + quote(session_id, safe="")
                    + "/messages?limit=64&offset=0&order=oldest"
                )
            raw = json.dumps({"run": run, "messages": messages}, ensure_ascii=False).encode()
            require(len(raw) <= 65536, "snapshot_combined_bound")
            (directory / name).write_bytes(raw)
            return hashlib.sha256(raw).hexdigest()

    result: dict[str, Any] = {"category": "refused", "semantic_review": "pending"}
    try:
        await session.start()
        started = time.monotonic()
        start = await asyncio.wait_for(
            surface.start_work(objective=m["objective"], invocation_id="explicit_lookup_setup"), 90
        )
        require(start.accepted and observed == {"posts": 1, "accepted": 1}, "accepted_ack")
        require(len(session.admitted_run_ids) == 1, "owned_run_count")
        run_id = next(iter(session.admitted_run_ids))
        controller.start()
        try:
            terminal = await asyncio.wait_for(controller.next_completion(), 120)
        except TimeoutError:
            try:
                result["evidence_sha256"] = await snapshot(run_id, "timeout-evidence.private.json")
            except Exception as error:
                result["evidence_failure_kind"] = type(error).__name__
            raise
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["adapter_completed"] = (
            terminal.task_id == start.task_id and terminal.status == "completed"
        )
        result["sse_completion_proven"] = stream == {"completedEvents": 1, "fallbackEntries": 0}
        primary = (
            "adapter_completion"
            if not result["adapter_completed"]
            else "sse_completion"
            if not result["sse_completion_proven"]
            else None
        )
        # Retain terminal evidence before any conformance refusal.
        try:
            result["evidence_sha256"] = await snapshot(run_id, "terminal-evidence.private.json")
        except Exception as error:
            result["evidence_failure_kind"] = type(error).__name__
            if primary is not None:
                raise ValueError(primary) from None
            raise
        require(primary is None, primary or "completion_refused")
        result["category"] = "completed"
    except Exception as error:
        result["failure_kind"] = type(error).__name__
        if type(error) is ValueError:
            result["failure_category"] = "validation_refused"
    finally:
        errors = []
        for close in (surface.close, controller.close, session.close):
            try:
                await close()
            except BaseException as error:
                errors.append(type(error).__name__)
        result.update(observed, stream=stream, cleanup_failure_kinds=errors)
        if errors:
            result["category"] = "cleanup_refused"
        (directory / "receipt.private.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def listener_rows(port: Any) -> Any:
    rows = []
    for protocol in ("tcp", "tcpv6"):
        output = subprocess.run(
            ["netstat", "-ano", "-p", protocol],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
        for line in output.splitlines():
            fields = line.split()
            if (
                len(fields) == 5
                and fields[0] == "TCP"
                and fields[3] == "LISTENING"
                and fields[1].rpartition(":")[2] == str(port)
            ):
                rows.append((fields[1].rpartition(":")[0], int(fields[4])))
    return sorted(rows)


def request_json(url: Any, token: Any, timeout: Any = 5) -> Any:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("readiness_redirect")

    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        require(response.status == 200, "readiness_status")
        raw = response.read(65537)
    require(len(raw) <= 65536, "readiness_bound")
    return json.loads(raw)


def require_owned_listeners(rows: Any, members: Any) -> None:
    # Authenticate only after OS listener ownership is known.
    require(
        len(rows) == 2 and all(address == "127.0.0.1" and pid in members for address, pid in rows),
        "gateway_ownership",
    )


def run_owned(m: Any) -> Any:
    require(os.name == "nt", "windows_required")
    require(
        Path(sys.executable).resolve() == Path(m["hermes_python"]).resolve(),
        "prepared_interpreter_required",
    )
    select_runtime(m)  # Before importing the owner and its runtime dependencies.
    from scripts import rehearse_desktop_mvp as owner
    from scripts.lookup_backend_logs import CappedProcesses
    from scripts.lookup_backend_model import start_model

    output = Path(m["output"])
    with (output / "attempted.json").open("x", encoding="utf-8") as stream:
        json.dump({"candidate": m["candidate"], "single_task": True}, stream)
    before = listener_rows(11434)
    require(
        not listener_rows(m["api_port"]) and not listener_rows(m["companion_port"]),
        "gateway_ports_occupied",
    )
    containment = owner.Containment()
    processes = CappedProcesses(owner.Processes(output), output)
    watchdog = threading.Timer(360, os._exit, args=(124,))
    watchdog.daemon = True
    watchdog.start()
    result: dict[str, Any] = {"category": "refused"}
    try:
        start_model(
            m,
            processes,
            containment,
            output,
            time.monotonic() + 60,
            verified=m["_verified_artifacts"],
        )
        env = owner._isolated(Path(m["profile"]))
        env.update(
            HERMES_HOME=m["profile"],
            API_SERVER_HOST="127.0.0.1",
            API_SERVER_PORT=str(m["api_port"]),
            PYTHONDONTWRITEBYTECODE="1",
        )
        gateway = processes.spawn(
            "gateway",
            [
                m["hermes_python"],
                "-I",
                "-B",
                str(ROOT / "scripts/lookup_backend_gateway.py"),
                str(output / "prepared.json"),
            ],
            env=env,
            cwd=Path(m["hermes_source"]),
            log=Path(os.devnull),
        )
        token = next(
            line.partition("=")[2]
            for line in (Path(m["profile"]) / ".env").read_text().splitlines()
            if line.startswith("API_SERVER_KEY=")
        )
        deadline = time.monotonic() + 30
        while True:
            require(gateway.poll() is None, "gateway_exited")
            rows = listener_rows(m["api_port"]) + listener_rows(m["companion_port"])
            if len(rows) == 2:
                require_owned_listeners(rows, containment.members())
                try:
                    health = request_json(
                        m["baseUrl"] + "/health/detailed",
                        token,
                        min(5, max(0.01, deadline - time.monotonic())),
                    )
                    break
                except (URLError, TimeoutError):
                    pass
            require(time.monotonic() < deadline, "gateway_startup_timeout")
            time.sleep(0.25)
        require(
            type(health.get("pid")) is int
            and health["pid"] in containment.members()
            and all(address == "127.0.0.1" and pid == health["pid"] for address, pid in rows),
            "gateway_ownership",
        )
        witness = read_json(output / "gateway-witness.private.json")
        require(
            witness.get("interpreterExact")
            and witness.get("packageExact")
            and witness.get("recordSha256") == m["recordSha256"]
            and witness.get("configSha256") == digest(Path(m["profile"]) / "config.yaml"),
            "gateway_runtime_binding",
        )
        result = asyncio.run(execute(m, output / "result"))
        witness = read_json(output / "gateway-witness.private.json")
        require(witness.get("transportRefused") is False, "transport_refused")
    except Exception as error:
        result["category"] = "refused"
        result["failure_kind"] = type(error).__name__
        if type(error) is ValueError:
            result["failure_category"] = "validation_refused"
    finally:
        result["cleanup_begin_ns"] = time.monotonic_ns()
        cleanup_errors = []
        try:
            processes.stop_all(containment)
        except BaseException as error:
            cleanup_errors.append(type(error).__name__)
        result["logs"] = processes.finish()
        result["cleanup_settled_ns"] = time.monotonic_ns()
        try:
            result["owned_listeners_gone"] = all(
                not listener_rows(m[p]) for p in ("api_port", "companion_port", "model_port")
            )
            result["user_listener_unchanged"] = before == listener_rows(11434)
            result["owned_processes_settled"] = not processes.survivors(containment)
        except BaseException as error:
            cleanup_errors.append(type(error).__name__)
        result["cleanup_failure_kinds"] = cleanup_errors
        if (
            cleanup_errors
            or processes.first_cleanup_refusal
            or not result.get("owned_listeners_gone")
            or not result.get("user_listener_unchanged")
            or not result.get("owned_processes_settled")
            or any(item["read_failed"] for item in result["logs"].values())
        ):
            result["category"] = "cleanup_refused"
        watchdog.cancel()
        (output / "owner-receipt.private.json").write_text(json.dumps(result), encoding="utf-8")
    # Private bounded evidence includes source/answer. stdout receives scalars only.
    return {
        name: result[name]
        for name in (
            "category",
            "posts",
            "accepted",
            "elapsed_seconds",
            "evidence_sha256",
            "semantic_review",
            "owned_listeners_gone",
            "user_listener_unchanged",
            "owned_processes_settled",
            "failure_kind",
            "failure_category",
        )
        if name in result
    }
