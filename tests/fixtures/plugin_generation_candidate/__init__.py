"""Synthetic native-Hermes fixture around the separate generation policy candidate.

Never install in a user profile. Only this fixture knows test scenario names.
"""

import json
import threading

from aiohttp import web
from openai.types.chat import ChatCompletion
from tools.approval import get_current_session_key
from tools.registry import registry

from .entrypoint import install
from .policy import Generation, Presentation


def completion(calls=(), refused=False):
    message = {
        "role": "assistant",
        "content": "Policy refusal." if refused else "Synthetic completion.",
    }
    if calls:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": "synthetic-" + str(index),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
            for index, (name, args) in enumerate(calls)
        ]
    return ChatCompletion.model_validate(
        {
            "id": "synthetic-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic-proof",
            "choices": [
                {"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def register(ctx):
    refusal = completion(refused=True)
    host_initial_proof = ctx.get_config("host_initial_proof", False)
    presentation = Presentation({
        "provider": "custom", "base_url": "http://127.0.0.1:9/v1",
        "model": "synthetic-proof", "api_key_env": "SYNTHETIC_PRESENTATION_KEY",
    }) if host_initial_proof else None
    generation = Generation(lambda: refusal, presentation=presentation)
    cases = {}
    counts = {}
    host_results = {}
    observed = []
    release = threading.Event()
    waiting = threading.Event()
    overlap = threading.Barrier(2, timeout=12)
    overlapped = set()
    real_originals = ctx.get_config("real_originals", False)
    safety = {"blocked_original_calls": 0, "fixture_failures": 0, "mode_probe_continuations": 0}
    budget_before = {}

    def native_budgets():
        from tools.budget_config import BudgetConfig, budget_for_context_window

        return {
            str(context): [
                budget_for_context_window(context).resolve_threshold(name)
                for name in ("web_search", "web_extract")
            ]
            for context in (None, 4096, 131072, 1000000)
        } | {
            "override": [
                BudgetConfig(tool_overrides={name: 1234}).resolve_threshold(name)
                for name in ("web_search", "web_extract")
            ]
        }

    def row_for(run_id):
        return counts.setdefault(
            run_id, {"provider": 0, "search": 0, "extract": 0, "original_error": False}
        )

    def wire(app, adapter):
        # Fixture metadata is captured before the native202 wrapper binds policy.
        @web.middleware
        async def scenarios(request, handler):
            response = await handler(request)
            if (
                request.method == "POST"
                and request.path.endswith("/v1/runs")
                and response.status == 202
            ):
                run_id = json.loads(response.body)["run_id"]
                with generation.lock:
                    cases.setdefault(run_id, request.headers.get("X-Proof-Case", "healthy"))
                    if cases[run_id] == "lost-before":
                        generation.budgets.pop(run_id, None)
                    if cases[run_id] == "replace-before":
                        generation.budgets[run_id] = dict(generation.budgets[run_id])
                    if cases[run_id] == "reset-refused":
                        generation.refuse(run_id)
                        generation.budgets[run_id]["closed"] = False
            return response

        app.middlewares.append(scenarios)
        generation.wire(app, adapter)

        def authorized(request):
            return request.headers.get("Authorization") == "Bearer " + ctx.get_config("proof_key")

        async def state(request):
            if not authorized(request):
                return web.json_response({}, status=401)
            with generation.lock:
                return web.json_response(
                    {
                        "ready": generation.readiness(),
                        "counts": counts,
                        "observed": observed,
                        "refused": sorted(generation.refused),
                        "overlapped": len(overlapped),
                        "waiting": waiting.is_set(),
                        "late_denied": generation.late_denied,
                        "late_results": generation.late_results,
                        "original_modules": {
                            name: handler.__module__
                            for name, handler in generation.originals.items()
                        },
                        "current_wrappers": all(
                            registry.get_entry(name).handler is handler
                            for name, handler, _ in generation.registrations
                        ),
                        "safety": safety,
                        "native_budget_before": budget_before,
                        "native_budget_after": native_budgets() if real_originals else {},
                        "budgets": {
                            key: {
                                "phase": row["phase"],
                                "closed": row["closed"],
                                "urls": len(row["urls"]),
                                "cancelled": row["cancelled"],
                            }
                            for key, row in generation.budgets.items()
                        },
                    }
                )

        async def control(request):
            if not authorized(request):
                return web.json_response({}, status=401)
            body = await request.json()
            with generation.lock:
                action = body["action"]
                if action == "release":
                    release.set()
                elif action == "lose-budget":
                    generation.budgets.pop(body["run_id"])
                elif action == "lose-authority":
                    generation.members = None
                else:
                    return web.json_response({}, status=400)
            return web.json_response({"accepted": True})

        app.router.add_get("/generation-proof/state", state)
        app.router.add_post("/generation-proof/control", control)

        async def native_error(request):
            if not authorized(request):
                return web.json_response({}, status=401)
            raise ValueError("synthetic-http-error")

        app.router.add_get("/generation-proof/error", native_error)

        async def guard_checks(request):
            if not authorized(request):
                return web.json_response({}, status=401)
            from pathlib import Path

            from .generation_policy_checks import run_matrix

            return web.json_response(
                await run_matrix(
                    generation, completion, Path(__file__).with_name("policy.py"), install, ctx
                )
            )

        app.router.add_get("/generation-proof/guard-checks", guard_checks)

    def provider_response(**kw):
        run_id = get_current_session_key(default="")
        with generation.lock:
            observed.append(
                {
                    "run": run_id,
                    "session": kw["session_id"],
                    "designated": run_id in generation.members,
                }
            )
            count = row_for(run_id)
            count["provider"] += 1
            number = count["provider"]
            scenario = cases.get(run_id, "healthy")
            if number > 1 and scenario in {"handler-error", "unscoped-error", "extract-error"}:
                count["original_error"] = any(
                    "synthetic-handler-error" in str(message.get("content", ""))
                    for message in kw["request"].get("messages", [])
                    if type(message) is dict
                )
            budget = generation.budgets.get(run_id)
            urls = sorted(budget["urls"]) if budget else []
        if scenario == "host-initial":
            from agent.tool_dispatch_helpers import _maybe_wrap_untrusted

            # Only the actual native executor may create the tool-result message.
            assert number == 1 and count["search"] == 1 and count["extract"] == 0
            assert budget is not None and budget["phase"] == "search" and len(urls) == 4
            tool_results = [message["content"] for message in kw["request"].get("messages", [])
                            if message.get("role") == "tool"]
            with generation.lock:
                original_result = host_results[run_id]
            assert tool_results == [_maybe_wrap_untrusted("web_search", original_result)]
            original_data = json.loads(original_result)
            assert original_data["success"] is True
            assert {item["url"] for item in original_data["data"]["web"]} == set(urls)
            safety["host_initial_native_result_seen"] = True
            waiting.set()
            if not release.wait(timeout=12):
                raise TimeoutError("fixture-release")
            return completion()
        if scenario == "mode-guard":

            def continuation(request):
                safety["mode_probe_continuations"] += 1
                return completion()

            return generation.model_gate(
                **{
                    **kw,
                    "api_mode": "anthropic_messages",
                    "next_call": continuation,
                }
            )
        if scenario.startswith("overlap-") and number == 1:
            with generation.lock:
                overlapped.add(run_id)
            overlap.wait()
        if scenario == "lost-row" and number == 1:
            waiting.set()
            if not release.wait(timeout=12):
                raise TimeoutError("fixture-release")
        if number == 1:
            size = (
                6
                if scenario in {"six", "overlap-refuse"}
                else (5 if scenario in {"five", "overlap-healthy"} else 1)
            )
            calls = [("web_search", {"query": "synthetic-" + str(index)}) for index in range(size)]
            if scenario == "mixed":
                calls = [
                    ("web_search", {"query": "synthetic"}),
                    ("web_extract", {"urls": ["https://example.test/absent"]}),
                ]
            elif scenario == "malformed":
                calls = [("web_search", {"query": "synthetic"}), ("web_search", "{")]
            elif scenario == "bad-query":
                calls = [("web_search", {"query": "synthetic"}), ("web_search", {})]
            return completion(calls)
        if number == 2 and scenario in {
            "healthy",
            "five",
            "rewrite",
            "extra",
            "duplicate-urls",
            "absent-url",
            "four-urls",
            "extract-error",
        }:
            if scenario == "extra":
                return completion([("web_search", {"query": "extra"})])
            if scenario == "duplicate-urls":
                urls = urls[:1] * 2
            if scenario == "absent-url":
                urls = ["https://example.test/absent"]
            if scenario == "four-urls":
                urls = urls[:4]
            return completion(
                [
                    (
                        "web_extract",
                        {
                            "urls": urls[:3]
                            if scenario in {"healthy", "five", "rewrite", "extract-error"}
                            else urls
                        },
                    )
                ]
            )
        return completion()

    def provider(**kw):
        try:
            return provider_response(**kw)
        except BaseException:
            safety["fixture_failures"] += 1
            return refusal

    def search(args, **kw):
        run_id = get_current_session_key(default="")
        with generation.lock:
            row_for(run_id)["search"] += 1
            scenario = cases[run_id]
            if scenario == "host-initial":
                assert row_for(run_id)["provider"] == 0, "host-initial-no-model-before-effect"
                assert args == {"query": "Synthetic request."}, "host-initial-admitted-objective"
                safety["host_initial_model_before_effect"] = 0
        if scenario == "cancel-result":
            waiting.set()
            if not release.wait(timeout=12):
                raise TimeoutError("fixture-release")
        if scenario in {"handler-error", "unscoped-error"}:
            raise ValueError("synthetic-handler-error")
        result = json.dumps(
            {
                "success": True,
                "data": {
                    "web": [
                        {"url": "https://example.test/" + args["query"] + "/" + str(i)}
                        for i in range(4)
                    ]
                },
            }
        )
        if scenario == "host-initial":
            with generation.lock:
                host_results[run_id] = result
        return result

    async def extract(args, **kw):
        with generation.lock:
            run_id = get_current_session_key(default="")
            row_for(run_id)["extract"] += 1
            scenario = cases[run_id]
        if scenario == "extract-error":
            raise ValueError("synthetic-handler-error")
        return json.dumps({"success": True, "data": "Synthetic page."})

    def late_hook(**kw):
        with generation.lock:
            scenario = cases.get(get_current_session_key(default=""))
        if scenario == "rewrite" and kw["tool_name"] == "web_extract":
            return {"action": "modify", "args": {"urls": ["https://example.test/absent"]}}
        if scenario == "cancel-dispatch" and kw["tool_name"] == "web_search":
            waiting.set()
            if not release.wait(timeout=12):
                raise TimeoutError("fixture-release")
        return None

    if real_originals:
        import tools.web_tools  # noqa: F401 -- shipped self-registration only.

        budget_before.update(native_budgets())

        install(ctx, generation, wire=wire)

        def deny_original(**kw):
            safety["blocked_original_calls"] += 1
            return json.dumps({"error": "offline-original-invocation-denied"})

        generation.handles.append(ctx.register_middleware("tool_execution", deny_original))
        return

    schemas = {}
    for name, handler in (("web_search", search), ("web_extract", extract)):
        schemas[name] = {
            "name": name,
            "description": "Synthetic fixture only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "urls": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
        generation.handles.append(
            ctx.register_tool(
                name,
                "proof_budget",
                schemas[name],
                handler,
                is_async=name == "web_extract",
                override=True,
            )
        )
    install(ctx, generation, wire=wire)
    if not ctx.get_config("terminal_error_probe", False):
        generation.handles.append(ctx.register_middleware("llm_execution", provider))
    generation.handles.append(ctx.register_hook("pre_tool_call", late_hook))
