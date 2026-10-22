"""Finite direct checks of the candidate's remaining defensive predicates.

These are policy unit checks in the prepared Hermes environment, not Runs,
provider, or lifecycle qualification. Each mutant is a separate source copy.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

# Each anchor is one predicate; names are also the single failing check category.
TARGETS = {
    "refusal-authority": (
        "self.refused is not self._refusal_authority",
        "False",
    ),
    "budget-container": (
        "type(self.budgets) is not dict",
        "False",
    ),
    "handles-present": (
        "or not self.handles",
        "or False",
    ),
    "handle-present": (
        "handle is not None and handle.active",
        "handle.active",
    ),
    "handle-active": (
        "handle is not None and handle.active",
        "handle is not None",
    ),
    "registry-present": (
        "(entry := registry.get_entry(name)) is not None",
        "(entry := registry.get_entry(name)) is not None or True",
    ),
    "registry-handler": (
        "entry.handler is handler",
        "True",
    ),
    "registry-async": (
        "entry.is_async is asynchronous",
        "True",
    ),
    "scope-ordinary": (
        "if designated and (",
        "if (",
    ),
    "policy-version": (
        'if selected not in {"", VERSION}:',
        "if False:",
    ),
    "capacity": (
        "len(self.members) + self.pending >= 128",
        "False",
    ),
    "admission-before": (
        "                if starting:\n"
        "                    with self.lock:\n"
        "                        if not self.available():",
        "                if starting:\n"
        "                    with self.lock:\n"
        "                        if False:",
    ),
    "admission-after": (
        "                    if starting and response.status == 202:\n"
        "                        if not self.available():",
        "                    if starting and response.status == 202:\n"
        "                        if False:",
    ),
    "replay-conflict": (
        "if not selected or known[:2] != (VERSION, identity):",
        "if False:",
    ),
    "replay-old": (
        'if accepted.get("replayed") is True:',
        "if False:",
    ),
    "stop-status": (
        'and path.endswith("/stop")\n                        and response.status == 200',
        'and path.endswith("/stop")\n                        and True',
    ),
    "status-ordinary": (
        "elif run_id in self.members:",
        "else:",
    ),
    "status-unavailable": (
        "                        if not self.available():\n"
        "                            response.headers",
        "                        if False:\n                            response.headers",
    ),
    "model-closed": (
        'if row is not None and row["closed"]:',
        "if False:",
    ),
    "choices": (
        "len(response.choices) != 1",
        "False",
    ),
    "call-id-empty": (
        "not call.id or",
        "False or",
    ),
    "call-id-repeat": (
        "call.id in plan or",
        "False or",
    ),
    "call-args": (
        "type(args) is not dict",
        "False",
    ),
    "extract-single": (
        "and len(calls) == 1",
        "and True",
    ),
    "search-settled": (
        'row["settled"] == row["reserved"]',
        "True",
    ),
    "urls-list": (
        "type(urls) is not list",
        "False",
    ),
    "urls-nonempty": (
        "not 1 <= len(urls) <= EXTRACT_LIMIT",
        "not 0 <= len(urls) <= EXTRACT_LIMIT",
    ),
    "urls-string": (
        "any(type(url) is not str for url in urls)",
        "False",
    ),
    "tool-closed": (
        '                        row["closed"]\n                        or row["request"]',
        '                        False\n                        or row["request"]',
    ),
    "tool-request": (
        'row["request"] != kw["api_request_id"]',
        "False",
    ),
    "tool-spent": (
        'call in row["spent"]',
        "False",
    ),
    "tool-plan": (
        'row["plan"].get(call) != expected',
        "False",
    ),
    "effect-ticket": (
        "or ticket is None",
        "or False",
    ),
    "effect-request": (
        'ticket[:2] != (run_id, row["request"])',
        "False",
    ),
    "effect-spent": (
        'ticket[2] not in row["spent"]',
        "False",
    ),
    "effect-repeat": (
        'ticket[2] in row["admitted"]',
        "False",
    ),
    "result-type": (
        "type(result) is not str or",
        "False or",
    ),
    "result-bytes": (
        'len(result.encode("utf-8")) > 262144',
        "False",
    ),
    "result-list": (
        "type(web_results) is not list",
        "False",
    ),
    "result-count": (
        "len(web_results) > 256",
        "False",
    ),
    "result-url": (
        'type(url) is not str or len(url.encode("utf-8")) > 8192',
        "False",
    ),
    "result-url-bytes": (
        'len(url.encode("utf-8")) > 8192',
        "False",
    ),
    "result-union": (
        'len(urls | row["urls"]) > 256',
        "False",
    ),
    "model-return-identity": (
        "                if current is not row:\n                    raise ValueError",
        "                if False:\n                    raise ValueError",
    ),
    "model-return-closed": (
        '                if row["closed"]:\n                    return self.refusal()',
        "                if False:\n                    return self.refusal()",
    ),
    "search-return-identity": (
        "                        if current is not row:\n"
        '                            raise ValueError("budget-replaced")\n'
        '                        row["settled"]',
        "                        if False:\n"
        '                            raise ValueError("budget-replaced")\n'
        '                        row["settled"]',
    ),
    "extract-return-identity": (
        "                        if current is not row:\n"
        '                            raise ValueError("budget-replaced")\n'
        '                        if row["closed"]:',
        "                        if False:\n"
        '                            raise ValueError("budget-replaced")\n'
        '                        if row["closed"]:',
    ),
    "extract-return-closed": (
        '                            raise ValueError("budget-replaced")\n'
        '                        if row["closed"]:\n'
        "                            self.late_results += 1\n"
        "                            return REFUSAL\n"
        "                return result\n"
        "            except BaseException:",
        '                            raise ValueError("budget-replaced")\n'
        "                        if False:\n"
        "                            self.late_results += 1\n"
        "                            return REFUSAL\n"
        "                return result\n"
        "            except BaseException:",
    ),
}


REDUNDANT = {
    "call-args": "non-object JSON also fails argument access",
    "effect-ticket": "absent ticket also fails subsequent ticket access",
    "urls-string": "non-string URL also fails returned-string membership",
}


async def evaluate(name, cls, installed, completion):
    from aiohttp import web
    from tools.approval import reset_current_session_key, set_current_session_key

    refused = object()
    g = cls(lambda: refused)
    g.handles = list(installed.handles)
    g.registrations = list(installed.registrations)
    tokens = []
    native_calls = []

    async def request(
        method="POST",
        path="/v1/runs",
        selected="ordinary-lookup/v1",
        status=202,
        run="unit-run",
        replayed=False,
        before=None,
        objective="Synthetic input.",
    ):
        app = web.Application()
        g.wire(app, None)

        async def body():
            return {"input": objective}

        req = SimpleNamespace(
            method=method,
            path=path,
            headers={"X-Lookup-Policy": selected},
            match_info={},
            json=body,
        )

        async def native(_):
            native_calls.append(1)
            if before:
                before()
            return web.json_response({"run_id": run, "replayed": replayed}, status=status)

        return await app.middlewares[-1](req, native)

    def call(response, callback=None):
        return g.model_gate(
            api_mode="chat_completions",
            api_request_id="unit-request",
            request={},
            next_call=callback or (lambda _: response),
        )

    def switch():
        tokens.append(set_current_session_key("unit-other"))

    def extraction(urls):
        return completion([("web_extract", {"urls": urls})])

    try:
        await request()
        row = g.budgets["unit-run"]
        tokens.append(set_current_session_key("unit-run"))
        response = completion([("web_search", {"query": "q"})])
        if name == "healthy-readiness":
            return g.available()
        if name == "healthy-search":
            return call(response) is response and row["reserved"] == 1
        if name == "refusal-authority":
            g.refused = frozenset({"unrelated"})
            return g.available() is False
        if name == "budget-container":
            g.budgets = []
            return g.available() is False
        if name.startswith("handle"):
            g.handles = (
                []
                if name == "handles-present"
                else [None if name == "handle-present" else SimpleNamespace(active=False)]
            )
            return g.available() is False
        if name.startswith("registry-"):
            entry = g.registrations[0]
            g.registrations = [
                ("not-a-registered-tool", entry[1], entry[2])
                if name == "registry-present"
                else (entry[0], object(), entry[2])
                if name == "registry-handler"
                else (entry[0], entry[1], not entry[2])
            ]
            return g.available() is False
        if name == "scope-ordinary":
            tokens.append(set_current_session_key("unit-unscoped"))
            return g.scope() == ("unit-unscoped", None)
        if name == "policy-version":
            return (await request(selected="unknown-policy")).status == 400
        if name == "capacity":
            g.pending = 127  # One retained member makes the exact capacity 128.
            return (await request(run="unit-next")).status == 503
        if name == "admission-before":
            g.closed = True
            previous = len(native_calls)
            return (await request()).status == 503 and len(native_calls) == previous
        if name == "admission-after":
            return (
                await request(run="unit-next", before=lambda: setattr(g, "closed", True))
            ).status == 503
        if name == "replay-conflict":
            same = await request(replayed=True)
            changed = await request(replayed=True, objective="Different synthetic input.")
            return (
                same.status == 202
                and g.budgets["unit-run"] is row
                and changed.status == 409
                and g.budgets["unit-run"] is row
                and len(g.members) == 1
                and row["reserved"] == 0
            )
        if name == "replay-old":
            return (await request(run="unit-old", replayed=True)).status == 409
        if name == "stop-status":
            await request(path="/v1/runs/unit-run/stop", status=403)
            return row["closed"] is False
        if name == "status-ordinary":
            reply = await request(method="GET", path="/v1/runs/unit-ordinary", status=200)
            return not any(key.lower().startswith("x-lookup-") for key in reply.headers)
        if name == "status-unavailable":
            g.closed = True
            reply = await request(method="GET", path="/v1/runs/unit-run", status=200)
            return (
                reply.status == 200
                and reply.headers.get("X-Lookup-Policy-Outcome") == "unavailable"
            )
        if name == "model-closed":
            row["closed"] = True
            count = []
            result = call(response, lambda _: (count.append(1), response)[1])
            return result is refused and not count
        if name == "choices":
            response.choices.append(response.choices[0].model_copy(deep=True))
            return call(response) is refused
        if name in {"call-id-empty", "call-id-repeat", "call-args"}:
            if name == "call-id-empty":
                response.choices[0].message.tool_calls[0].id = ""
            elif name == "call-id-repeat":
                response.choices[0].message.tool_calls *= 2
            else:
                response.choices[0].message.tool_calls[0].function.arguments = "[]"
            return call(response) is refused
        if name.startswith("model-return"):
            await request(run="unit-other")

            def provider(_):
                switch() if name.endswith("identity") else row.update(closed=True)
                return response

            return call(response, provider) is refused
        if name in {
            "extract-single",
            "search-settled",
            "urls-list",
            "urls-nonempty",
            "urls-string",
        }:
            call(response)
            row["settled"] = 0 if name == "search-settled" else row["reserved"]
            row["urls"] = {"u"}
            urls = (
                {"u": 1}
                if name == "urls-list"
                else []
                if name == "urls-nonempty"
                else [1]
                if name == "urls-string"
                else ["u"]
            )
            response = extraction(urls)
            if name == "extract-single":
                response.choices[0].message.tool_calls.append(
                    response.choices[0].message.tool_calls[0].model_copy(deep=True)
                )
                response.choices[0].message.tool_calls[1].id = "second-extraction"
            return call(response) is refused
        call(response)
        if name.startswith("tool-"):
            row["closed"] = name == "tool-closed"
            row["spent"] = {"synthetic-0"} if name == "tool-spent" else set()
            count = []
            g.tool_gate(
                api_request_id="wrong" if name == "tool-request" else "unit-request",
                tool_call_id="synthetic-0",
                tool_name="web_search",
                args={"query": "other" if name == "tool-plan" else "q"},
                next_call=lambda _: count.append(1),
            )
            return not count
        if name.startswith("effect-"):
            row["spent"] = set() if name == "effect-spent" else {"synthetic-0"}
            row["admitted"] = {"synthetic-0"} if name == "effect-repeat" else set()
            ticket = (
                None
                if name == "effect-ticket"
                else (
                    "unit-run",
                    "wrong" if name == "effect-request" else "unit-request",
                    "synthetic-0",
                    "web_search",
                    '{"query":"q"}',
                )
            )
            token = g.ticket.set(ticket)
            try:
                try:
                    g.admit_effect("web_search", {"query": "q"})
                except (ValueError, TypeError):
                    return True
                return False
            finally:
                g.ticket.reset(token)
        values = [{"url": "u"}]
        if name == "result-list":
            values = {}
        elif name == "result-count":
            values *= 257  # A cumulative-set check cannot mask the list-count guard.
        elif name == "result-url":
            values = [{"url": 1}]
        elif name == "result-url-bytes":
            values = [{"url": "u" * 8193}]
        elif name == "result-union":
            row["urls"] = {str(i) for i in range(256)}
        data = json.dumps({"success": True, "data": {"web": values}})
        if name == "result-bytes":
            data += " " * 262144
        elif name == "result-type":

            class Text(str):
                pass

            data = Text(data)
        await request(run="unit-other")

        def search_original(*_args, **_kw):
            if name == "search-return-identity":
                switch()
            return data

        if name.startswith("extract-return"):
            row.update(phase="search", settled=1, reserved=1, urls={"u"})
            call(extraction(["u"]))

            async def extract_original(*_args, **_kw):
                switch() if name.endswith("identity") else row.update(closed=True)
                return "synthetic-extraction-result"

            original = g.wrap_extract(extract_original)
            args = {"urls": ["u"]}
            token = g.ticket.set(
                ("unit-run", "unit-request", "synthetic-0", "web_extract", '{"urls":["u"]}')
            )
            row["spent"].add("synthetic-0")
            try:
                result = await original(args)
            finally:
                g.ticket.reset(token)
        else:
            result = g.tool_gate(
                api_request_id="unit-request",
                tool_call_id="synthetic-0",
                tool_name="web_search",
                args={"query": "q"},
                next_call=lambda args: g.wrap_search(search_original)(args),
            )
        return result == json.dumps({"error": "policy_refused"})
    finally:
        for token in reversed(tokens):
            reset_current_session_key(token)


async def installer_shape(installer, installed, ctx):
    from tools.registry import registry

    original = registry.get_entry("web_search")

    async def unsupported(_args, **_kw):
        raise RuntimeError("never-invoke-installation-probe")

    replacement = ctx.register_tool(
        "web_search",
        original.toolset,
        original.schema,
        unsupported,
        is_async=True,
        override=True,
    )
    candidate = type(installed)(lambda: None)
    rejected = False
    try:
        try:
            installer(ctx, candidate)
        except ValueError as error:
            rejected = str(error) == "unsupported-web-registration"
        return rejected and not candidate.handles
    finally:
        for handle in reversed(candidate.handles):
            handle.dispose()
        replacement.dispose()


async def run_matrix(installed, completion, source, installer, ctx):
    import hashlib
    import tempfile

    text = Path(source).read_text(encoding="utf-8")
    rows = []
    with tempfile.TemporaryDirectory(prefix="policy-guard-checks-") as directory:
        for name, (old, new) in TARGETS.items():
            if text.count(old) != 1:
                raise ValueError("guard-anchor:" + name)
            path = Path(directory) / (name.replace("-", "_") + ".py")
            path.write_text(text.replace(old, new), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("guard_variant", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            outcomes = []
            for case, cls in (
                (name, type(installed)),
                (name, module.Generation),
                ("healthy-readiness", module.Generation),
                ("healthy-search", module.Generation),
            ):
                try:
                    passed = bool(await evaluate(case, cls, installed, completion))
                except Exception:
                    passed = False
                outcomes.append(passed)
            rows.append(
                {
                    "guard": name,
                    "outcomes": outcomes,
                    "expected": [True, name in REDUNDANT, True, True],
                    "independent_mutation_proven": name not in REDUNDANT,
                    "mutation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        entrypoint = Path(source).with_name("entrypoint.py").read_text(encoding="utf-8")
        anchor = "if original is None or original.is_async is not asynchronous:"
        if entrypoint.count(anchor) != 1:
            raise ValueError("guard-anchor:installer-shape")
        path = Path(directory) / "entrypoint_variant.py"
        path.write_text(entrypoint.replace(anchor, "if False:"), encoding="utf-8")
        package = type(installed).__module__.rsplit(".", 1)[0]
        spec = importlib.util.spec_from_file_location(package + ".entrypoint_variant", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outcomes = [
            await installer_shape(installer, installed, ctx),
            await installer_shape(module.install, installed, ctx),
        ]
        for case in ("healthy-readiness", "healthy-search"):
            outcomes.append(bool(await evaluate(case, type(installed), installed, completion)))
        rows.append(
            {
                "guard": "installer-shape",
                "outcomes": outcomes,
                "expected": [True, False, True, True],
                "independent_mutation_proven": True,
                "mutation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "ok": all(row["outcomes"] == row["expected"] for row in rows),
        "guards": len(rows),
        "rows": rows,
    }
