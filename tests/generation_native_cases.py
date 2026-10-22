"""Bounded synthetic client cases against the unchanged native Runs adapter."""

import asyncio
import inspect
import sys
from pathlib import Path


async def prove(port, key, witness, case):
    from aiohttp import ClientSession
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_cli.plugins import get_plugin_manager

    assert (
        Path(inspect.getfile(APIServerAdapter)).resolve()
        == (Path(sys.path[0]) / "gateway/platforms/api_server.py").resolve()
    ), "native-import-origin"
    witness["native_import_origin_verified"] = True
    get_plugin_manager().discover_and_load()
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": port,
                "key": key,
                "model_name": "synthetic-proof",
            },
        )
    )
    base = f"http://127.0.0.1:{port}"
    aliased = base + "/p/default"
    status_base = aliased if case.startswith("alias-") else base
    cancellation = case in {
        "cancel-dispatch",
        "cancel-result",
        "alias-cancel-dispatch",
        "alias-cancel-result",
    }
    cancel_scenario = case.removeprefix("alias-")
    run_ids = []
    scoped_ids = set()
    statuses = []
    outcomes = []
    try:
        assert await adapter.connect(), "native-connect"
        async with ClientSession(headers={"Authorization": "Bearer " + key}) as http:

            async def state():
                async with http.get(base + "/generation-proof/state") as reply:
                    assert reply.status == 200, "state-authentication"
                    return await reply.json()

            async def control(action, **extra):
                async with http.post(
                    base + "/generation-proof/control", json={"action": action, **extra}
                ) as reply:
                    assert reply.status == 200, "fixture-control"

            async with http.get(base + "/lookup-policy/ready") as reply:
                if case == "no-opt-in":
                    assert reply.status == 404, "override-opt-in-required"
                    witness["native_posts"] = 0
                    return {"cases": 1}
                ready = await reply.json()
                assert reply.status == 200 and ready["ready"] is True, "generation-readiness"
                assert ready["policy"] == "ordinary-lookup/v1" and len(ready["generation"]) == 32, (
                    "readiness-binding"
                )
            if case in {"readiness", "builtins"}:
                if case == "builtins":
                    __import__("model_tools")  # Public import triggers normal builtin discovery.
                    from hermes_cli.middleware import run_tool_execution_middleware

                    terminal_attempts = []

                    def no_original(args):
                        terminal_attempts.append(True)
                        raise RuntimeError("offline-before-original-barrier")

                    run_tool_execution_middleware("web_search", {"query": "synthetic"}, no_original)
                    snapshot = await state()
                    assert snapshot["original_modules"] == {
                        "web_search": "tools.web_tools",
                        "web_extract": "tools.web_tools",
                    }, "real-originals-captured"
                    assert snapshot["current_wrappers"] is True, "lazy-registration-wrapper-binding"
                    assert snapshot["ready"] == ready, "real-originals-readiness"
                    assert snapshot["native_budget_before"] == snapshot["native_budget_after"], (
                        "native-result-budget-equivalence"
                    )
                    assert (
                        not terminal_attempts and snapshot["safety"]["blocked_original_calls"] == 1
                    ), "before-original-barrier"
                    witness["real_originals_wrapped"] = True
                    witness["original_handler_invocations"] = 0
                return {"cases": 1}
            if case == "middleware-error":
                from hermes_cli.middleware import (
                    run_llm_execution_middleware,
                    run_tool_execution_middleware,
                )

                for operation in ("model", "tool"):
                    calls = []
                    original = ValueError("synthetic-downstream-error")

                    def terminal(payload, calls=calls, original=original):
                        calls.append(True)
                        raise original

                    try:
                        if operation == "model":
                            run_llm_execution_middleware({}, terminal, api_mode="chat_completions")
                        else:
                            run_tool_execution_middleware("web_search", {}, terminal)
                    except ValueError as exc:
                        assert exc is original and calls == [True], "downstream-error-identity-once"
                    else:
                        raise AssertionError("downstream-error-preserved")
                return {"cases": 1, "middleware_errors_preserved": 2}
            if case == "http-error":
                async with http.get(base + "/generation-proof/error") as reply:
                    assert reply.status == 500, "native-http-error-preserved"
                async with http.get(base + "/lookup-policy/ready") as reply:
                    assert reply.status == 200, "unscoped-http-error-no-epoch-close"
                return {"cases": 1}

            if case == "alias-admission":
                async with http.post(
                    aliased + "/v1/runs",
                    headers={"X-Lookup-Policy": "ordinary-lookup/v1"},
                    json={"input": "Synthetic request.", "session_id": "shared-synthetic-session"},
                ) as reply:
                    witness["alias_native_admission_ack"] = reply.status == 202
                    assert reply.status == 400, "alias-policy-rejected-before-dispatch"
                snapshot = await state()
                assert snapshot["observed"] == [], "alias-policy-zero-provider"
                return {"cases": 1, "stand_in_calls": 0}

            async def start(scenario, designated=True, replay_key=None, admission_base=None):
                headers = {"X-Proof-Case": scenario}
                if designated:
                    headers["X-Lookup-Policy"] = "ordinary-lookup/v1"
                if replay_key:
                    headers["Idempotency-Key"] = replay_key
                async with http.post(
                    (admission_base or base) + "/v1/runs",
                    headers=headers,
                    json={"input": "Synthetic request.", "session_id": "shared-synthetic-session"},
                ) as reply:
                    accepted = await reply.json()
                    assert reply.status == 202, "native-admission"
                    run_id = accepted["run_id"]
                    if designated:
                        scoped_ids.add(run_id)
                        assert (
                            reply.headers.get("X-Lookup-Generation") == ready["generation"]
                            and reply.headers.get("X-Lookup-Policy") == ready["policy"]
                            and reply.headers.get("X-Lookup-Run") == run_id
                        ), "policy-ack-binding"
                    if run_id not in run_ids:
                        run_ids.append(run_id)
                    return run_id

            async def until(predicate, assertion):
                for _ in range(240):
                    snapshot = await state()
                    if predicate(snapshot):
                        return snapshot
                    if assertion == "late-worker-witness":
                        if (
                            cancel_scenario == "cancel-dispatch"
                            and snapshot["counts"][run_ids[0]]["search"]
                        ):
                            witness["post_stop_search_effects"] = snapshot["counts"][run_ids[0]][
                                "search"
                            ]
                            raise AssertionError("post-stop-new-effect")
                        if (
                            cancel_scenario == "cancel-result"
                            and snapshot["budgets"][run_ids[0]]["urls"]
                        ):
                            witness["late_result_url_count"] = snapshot["budgets"][run_ids[0]][
                                "urls"
                            ]
                            raise AssertionError("late-result-not-authority")
                    await asyncio.sleep(0.05)
                raise AssertionError(assertion)

            if case == "overlap":
                await start("overlap-healthy")
                await start("overlap-refuse")
            elif case == "lost-row":
                await start("lost-before")
                await start("unscoped", designated=False)
            elif case == "replace-row":
                await start("replace-before")
                await start("unscoped", designated=False)
            elif case == "reset-refused":
                await start("reset-refused")
                await start("unscoped", designated=False)
            elif case == "unscoped":
                await start("unscoped", designated=False)
            elif case == "unscoped-error":
                await start("unscoped-error", designated=False)
            elif case == "alias-unscoped":
                await start("unscoped", designated=False, admission_base=aliased)
            elif cancellation:
                await start(cancel_scenario)
            elif case == "alias-status":
                await start("healthy")
            else:
                await start(
                    "healthy" if case == "replay" else case,
                    replay_key="synthetic-replay" if case == "replay" else None,
                )

            if case in {"healthy", "six", "alias-status"}:
                async with http.get(
                    status_base + "/v1/runs/" + run_ids[0],
                    headers={"Authorization": "Bearer wrong-synthetic-key"},
                ) as reply:
                    assert reply.status in {401, 403}, "unauthorized-status-rejected"
                    assert not any(
                        name.lower().startswith("x-lookup-") for name in reply.headers
                    ), "unauthorized-no-policy-disclosure"
                async with http.get(status_base + "/v1/runs/unknown-synthetic-run") as reply:
                    assert reply.status == 404 and not any(
                        name.lower().startswith("x-lookup-") for name in reply.headers
                    ), "non200-no-policy-outcome"

            if cancellation:
                await until(lambda snapshot: snapshot["waiting"], "cancellation-barrier")
                async with http.post(
                    status_base + "/v1/runs/" + run_ids[0] + "/stop",
                    headers={"Authorization": "Bearer wrong-synthetic-key"},
                ) as reply:
                    assert reply.status in {401, 403}, "unauthorized-stop-rejected"
                before = await state()
                assert before["budgets"][run_ids[0]]["closed"] is False, (
                    "unauthorized-stop-no-policy-change"
                )
                async with http.post(
                    status_base + "/v1/runs/" + run_ids[0] + "/stop",
                    headers={"X-Lookup-Policy": "ordinary-lookup/v1"},
                ) as reply:
                    assert reply.status == 200, "native-stop-accepted"
                    stop_reply = await reply.json()
                    assert stop_reply["status"] == "stopping", "truthful-stop-ack"
                closed = await state()
                try:
                    assert closed["budgets"][run_ids[0]]["closed"] is True, (
                        "cancellation-policy-cut"
                    )
                finally:
                    await control("release")
                marker = "late_denied" if cancel_scenario == "cancel-dispatch" else "late_results"
                await until(lambda snapshot: snapshot[marker] >= 1, "late-worker-witness")

            for _ in range(240):
                statuses = []
                outcomes = []
                for run_id in run_ids:
                    async with http.get(status_base + "/v1/runs/" + run_id) as reply:
                        assert reply.status == 200, "native-status"
                        statuses.append((await reply.json())["status"])
                        outcomes.append(
                            (
                                reply.headers.get("X-Lookup-Generation"),
                                reply.headers.get("X-Lookup-Policy"),
                                reply.headers.get("X-Lookup-Policy-Refused"),
                            )
                        )
                if all(
                    status in {"completed", "failed", "cancelled", "interrupted"}
                    for status in statuses
                ):
                    break
                await asyncio.sleep(0.05)
            snapshot = await state()
            for run_id, outcome in zip(run_ids, outcomes, strict=True):
                if run_id in scoped_ids:
                    assert outcome == (
                        ready["generation"],
                        ready["policy"],
                        "true" if run_id in snapshot["refused"] else "false",
                    ), "authenticated-refusal-outcome"
                else:
                    assert outcome == (None, None, None), "unscoped-no-policy-outcome"
            if not cancellation and case != "unsupported-mode":
                assert statuses == ["completed"] * len(run_ids), "native-completion"
            else:
                assert all(
                    status in {"completed", "failed", "cancelled", "interrupted"}
                    for status in statuses
                ), "native-cancel-terminal"
            if case == "replay":
                before = snapshot["counts"].copy()
                assert await start("healthy", replay_key="synthetic-replay") == run_ids[0], (
                    "same-policy-replay-identity"
                )
                await asyncio.sleep(0.1)
                snapshot = await state()
                assert snapshot["counts"] == before, "replay-no-budget-reset"
                async with http.post(
                    base + "/v1/runs",
                    headers={"Idempotency-Key": "synthetic-replay"},
                    json={"input": "Synthetic request.", "session_id": "shared-synthetic-session"},
                ) as reply:
                    assert reply.status == 409, "replay-policy-conflict"
            if case == "authority":
                await control("lose-authority")
                async with http.get(base + "/lookup-policy/ready") as reply:
                    assert reply.status == 503 and (await reply.json())["ready"] is False, (
                        "authority-loss-readiness"
                    )
                async with http.get(base + "/v1/runs/" + run_ids[0]) as reply:
                    assert (
                        reply.status == 200
                        and reply.headers.get("X-Lookup-Policy-Outcome") == "unavailable"
                    ), "authority-loss-outcome"
                async with http.post(
                    base + "/v1/runs",
                    headers={"X-Lookup-Policy": "ordinary-lookup/v1"},
                    json={"input": "Synthetic request."},
                ) as reply:
                    assert reply.status == 503, "authority-loss-admission"

            counts = [
                snapshot["counts"].get(run_id, {"provider": 0, "search": 0, "extract": 0})
                for run_id in run_ids
            ]
            refused = [run_id in snapshot["refused"] for run_id in run_ids]
            expected = {
                "healthy": [(3, 1, 1, False)],
                "five": [(3, 5, 1, False)],
                "six": [(1, 0, 0, True)],
                "mixed": [(1, 0, 0, True)],
                "malformed": [(1, 0, 0, True)],
                "bad-query": [(1, 0, 0, True)],
                "extra": [(2, 1, 0, True)],
                "duplicate-urls": [(2, 1, 0, True)],
                "absent-url": [(2, 1, 0, True)],
                "four-urls": [(2, 1, 0, True)],
                "rewrite": [(2, 1, 0, True)],
                "replay": [(3, 1, 1, False)],
                "authority": [(2, 1, 0, False)],
                "unscoped": [(2, 1, 0, False)],
                "lost-row": [(0, 0, 0, True), (2, 1, 0, False)],
                "replace-row": [(0, 0, 0, True), (2, 1, 0, False)],
                "reset-refused": [(0, 0, 0, True), (2, 1, 0, False)],
                "overlap": [(2, 5, 0, False), (1, 0, 0, True)],
                "cancel-dispatch": [(1, 0, 0, True)],
                "cancel-result": [(1, 1, 0, False)],
                "unsupported-mode": [(0, 0, 0, True)],
                "handler-error": [(2, 1, 0, False)],
                "unscoped-error": [(2, 1, 0, False)],
                "extract-error": [(3, 1, 1, False)],
                "mode-guard": [(1, 0, 0, True)],
                "alias-unscoped": [(2, 1, 0, False)],
                "alias-status": [(3, 1, 1, False)],
                "alias-cancel-dispatch": [(1, 0, 0, True)],
                "alias-cancel-result": [(1, 1, 0, False)],
            }[case]
            actual = [
                (row["provider"], row["search"], row["extract"], denied)
                for row, denied in zip(counts, refused, strict=True)
            ]
            witness["effect_counts"] = [list(row) for row in actual]
            witness["native_statuses"] = statuses
            witness["late_denied"] = snapshot["late_denied"]
            witness["late_results"] = snapshot["late_results"]
            assert actual == expected, "exact-policy-effects"
            if case == "mode-guard":
                assert snapshot["safety"]["mode_probe_continuations"] == 0, (
                    "unsupported-mode-no-continuation"
                )
            assert snapshot["safety"]["fixture_failures"] == 0, "synthetic-provider-no-fallback"
            if case in {"handler-error", "unscoped-error", "extract-error"}:
                assert counts[0]["original_error"] is True, "original-handler-error-preserved"
            if case == "overlap":
                assert snapshot["overlapped"] == 2, "actual-model-overlap"
                assert {item["run"] for item in snapshot["observed"]} == set(run_ids), (
                    "true-run-isolation"
                )
            if cancellation:
                assert snapshot["budgets"][run_ids[0]]["urls"] == 0, "late-result-not-authority"
            assert all(
                item["session"] == "shared-synthetic-session" for item in snapshot["observed"]
            ), "shared-session"
            assert witness["unclassified_network_attempts"] == 0, "no-provider-fallback"
            return {
                "cases": 1,
                "native_completed": statuses.count("completed"),
                "distinct_run_scopes": len(run_ids),
                "stand_in_calls": sum(row["provider"] for row in counts),
            }
    finally:
        await adapter.disconnect()
