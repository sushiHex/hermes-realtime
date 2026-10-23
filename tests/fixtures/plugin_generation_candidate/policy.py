"""Candidate process-lifetime policy for an explicitly designated native Runs route.

Offline fixture only. Active plugin removal is outside its selected lifecycle.
No prompt classification, model names or synthetic scenario names occur here.
"""

import contextvars
import hashlib
import json
import secrets
import threading
from types import MappingProxyType

from aiohttp import web
from tools.approval import get_current_session_key
from tools.registry import registry

VERSION = "ordinary-lookup/v1"
SEARCH_LIMIT = 5
EXTRACT_LIMIT = 3
REFUSAL = json.dumps({"error": "policy_refused"})


def frozen(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Generation:
    def __init__(self, refusal):
        self.epoch = secrets.token_hex(16)
        self.lock = threading.RLock()
        self.members = MappingProxyType({})
        self._membership_authority = self.members  # Retained until actual process exit.
        self.budgets = {}  # Deliberately separate mutable rows.
        self.refused = frozenset()
        self._refusal_authority = self.refused
        self.handles = []
        self.registrations = []
        self.originals = {}
        self.closed = False
        self.pending = 0
        self.refusal = refusal
        self.ticket = contextvars.ContextVar("generation-effect-ticket", default=None)
        self.late_denied = 0
        self.late_results = 0

    def available(self):
        if (
            self.members is not self._membership_authority
            or self.refused is not self._refusal_authority
            or type(self.budgets) is not dict
            or not self.handles
            or not all(handle is not None and handle.active for handle in self.handles)
            or not all(
                (entry := registry.get_entry(name)) is not None
                and entry.handler is handler
                and entry.is_async is asynchronous
                for name, handler, asynchronous in self.registrations
            )
        ):
            self.closed = True
        return not self.closed

    def refuse(self, run_id):
        if self.members is self._membership_authority and run_id in self.members:
            self.refused = self.refused | {run_id}
            self._refusal_authority = self.refused
        row = self.budgets.get(run_id) if type(self.budgets) is dict else None
        if type(row) is dict:
            row["closed"] = True

    def scope(self):
        run_id = get_current_session_key(default="")
        if not self.available():
            raise ValueError("generation-unavailable")
        designated = run_id in self.members
        row = self.budgets.get(run_id)
        if designated and (run_id in self.refused or row is not self.members[run_id][2]):
            self.refuse(run_id)
            raise ValueError("budget-unavailable")
        return run_id, row if designated else None

    def readiness(self):
        with self.lock:
            return {"ready": self.available(), "generation": self.epoch, "policy": VERSION}

    def wire(self, app, adapter):
        @web.middleware
        async def admission(request, handler):
            starting = request.method == "POST" and request.path == "/v1/runs"
            selected = request.headers.get("X-Lookup-Policy", "")
            reserved = False
            native_error = False
            path = request.path
            profile = request.match_info.get("profile")
            if profile and path.startswith("/p/" + profile + "/"):
                path = path[len("/p/" + profile) :]

            async def call_native():
                nonlocal native_error
                try:
                    return await handler(request)
                except BaseException:
                    native_error = True
                    if starting and selected:
                        with self.lock:
                            self.closed = True
                    raise

            try:
                if request.method == "POST" and path == "/v1/runs" and not starting and selected:
                    return web.json_response({"error": "policy_route_unsupported"}, status=400)
                if starting:
                    with self.lock:
                        if not self.available():
                            return web.json_response({"error": "policy_unavailable"}, status=503)
                        if selected not in {"", VERSION}:
                            return web.json_response({"error": "policy_unknown"}, status=400)
                        if selected:
                            if len(self.members) + self.pending >= 128:
                                return web.json_response({"error": "policy_capacity"}, status=503)
                            self.pending += 1
                            reserved = True
                    if selected:
                        try:
                            body = await request.json()
                        except Exception:
                            return await call_native()
                        identity = hashlib.sha256(frozen(body).encode()).hexdigest()
                response = await call_native()
                with self.lock:
                    if starting and response.status == 202:
                        if not self.available():
                            return web.json_response({"error": "policy_unavailable"}, status=503)
                        accepted = json.loads(response.body)
                        run_id = accepted["run_id"]
                        known = self.members.get(run_id)
                        if known is not None:
                            if not selected or known[:2] != (VERSION, identity):
                                return web.json_response({"error": "policy_conflict"}, status=409)
                        elif selected:
                            if accepted.get("replayed") is True:
                                return web.json_response(
                                    {"error": "policy_old_generation"}, status=409
                                )
                            row = {
                                "phase": "new",
                                "closed": False,
                                "cancelled": False,
                                "plan": {},
                                "request": "",
                                "spent": set(),
                                "admitted": set(),
                                "reserved": 0,
                                "settled": 0,
                                "urls": set(),
                            }
                            self.budgets[run_id] = row
                            members = dict(self.members)
                            members[run_id] = (VERSION, identity, row)
                            self.members = MappingProxyType(members)
                            self._membership_authority = self.members
                        if selected:
                            response.headers["X-Lookup-Generation"] = self.epoch
                            response.headers["X-Lookup-Policy"] = VERSION
                            response.headers["X-Lookup-Run"] = run_id
                    if (
                        request.method == "POST"
                        and path.startswith("/v1/runs/")
                        and path.endswith("/stop")
                        and response.status == 200
                    ):
                        # Native auth/ownership and stop execute first, without an await.
                        # This lock closes new effect admission before the ack is returned.
                        run_id = path.split("/")[3]
                        row = self.budgets.get(run_id)
                        if (
                            self.members is self._membership_authority
                            and run_id in self.members
                            and row
                        ):
                            row["closed"] = row["cancelled"] = True
                    if (
                        request.method == "GET"
                        and response.status == 200
                        and path.startswith("/v1/runs/")
                        and len(path.split("/")) == 4
                    ):
                        # Native GET already authenticated and checked run ownership.
                        run_id = path.split("/")[3]
                        if not self.available():
                            response.headers["X-Lookup-Policy-Outcome"] = "unavailable"
                        elif run_id in self.members:
                            response.headers["X-Lookup-Generation"] = self.epoch
                            response.headers["X-Lookup-Policy"] = VERSION
                            response.headers["X-Lookup-Policy-Refused"] = (
                                "true" if run_id in self.refused else "false"
                            )
                return response
            except BaseException:
                if native_error:
                    raise
                with self.lock:
                    self.closed = True
                return web.json_response({"error": "policy_unavailable"}, status=503)
            finally:
                if reserved:
                    with self.lock:
                        self.pending -= 1

        app.middlewares.append(admission)

        async def ready(request):
            body = self.readiness()
            return web.json_response(body, status=200 if body["ready"] else 503)

        app.router.add_get("/lookup-policy/ready", ready)

    def model_gate(self, **kw):
        run_id = ""
        try:
            with self.lock:
                run_id, row = self.scope()
                if row is not None and row["closed"]:
                    return self.refusal()
                if row is not None and kw.get("api_mode") != "chat_completions":
                    raise ValueError("unsupported-response-mode")
        except BaseException:
            with self.lock:
                self.refuse(run_id)
            return self.refusal()
        response = kw["next_call"](kw["request"])
        if row is None:
            return response
        try:
            with self.lock:
                _, current = self.scope()  # Replaced rows are never adopted.
                if current is not row:
                    raise ValueError("budget-replaced")
                if row["closed"]:
                    return self.refusal()
                calls = response.choices[0].message.tool_calls or []
                if len(response.choices) != 1 or len(calls) > SEARCH_LIMIT:
                    raise ValueError("batch-envelope")
                plan = {}
                for call in calls:
                    args = json.loads(call.function.arguments)
                    if not call.id or call.id in plan or type(args) is not dict:
                        raise ValueError("call-envelope")
                    if call.function.name == "web_search" and not (
                        type(args.get("query")) is str and args["query"]
                    ):
                        raise ValueError("search-arguments")
                    plan[call.id] = (call.function.name, frozen(args))
                names = {value[0] for value in plan.values()}
                if not calls:
                    row["closed"] = True
                elif (
                    row["phase"] == "new"
                    and names == {"web_search"}
                    and 1 <= len(calls) <= SEARCH_LIMIT
                ):
                    row["phase"], row["reserved"] = "search", len(calls)
                elif (
                    row["phase"] == "search"
                    and names == {"web_extract"}
                    and len(calls) == 1
                    and row["settled"] == row["reserved"]
                ):
                    urls = json.loads(next(iter(plan.values()))[1]).get("urls")
                    if (
                        type(urls) is not list
                        or not 1 <= len(urls) <= EXTRACT_LIMIT
                        or any(type(url) is not str for url in urls)
                        or len(set(urls)) != len(urls)
                        or any(url not in row["urls"] for url in urls)
                    ):
                        raise ValueError("url-membership")
                    row["phase"] = "extract"
                else:
                    raise ValueError("batch-budget")
                row["plan"] = plan
                row["request"] = kw["api_request_id"]
                row["spent"], row["admitted"] = set(), set()
            return response
        except BaseException:
            with self.lock:
                self.refuse(run_id)
            return self.refusal()

    def tool_gate(self, **kw):
        run_id = ""
        token = None
        try:
            with self.lock:
                run_id, row = self.scope()
                if row is not None:
                    call = kw["tool_call_id"]
                    expected = (kw["tool_name"], frozen(kw["args"]))
                    if (
                        row["closed"]
                        or row["request"] != kw["api_request_id"]
                        or call in row["spent"]
                        or row["plan"].get(call) != expected
                    ):
                        raise ValueError("call-ticket")
                    row["spent"].add(call)
                    token = self.ticket.set((run_id, row["request"], call, *expected))
        except BaseException:
            with self.lock:
                self.refuse(run_id)
            return REFUSAL
        try:
            return kw["next_call"](kw["args"])
        finally:
            if token is not None:
                self.ticket.reset(token)

    def admit_effect(self, name, args):
        with self.lock:
            run_id, row = self.scope()
            if row is None:
                return run_id, None
            ticket = self.ticket.get()
            if (
                row["closed"]
                or ticket is None
                or ticket[:2] != (run_id, row["request"])
                or ticket[2] not in row["spent"]
                or ticket[2] in row["admitted"]
                or ticket[3:] != (name, frozen(args))
            ):
                self.late_denied += int(row["cancelled"])
                self.refuse(run_id)
                raise ValueError("final-effect-ticket")
            row["admitted"].add(ticket[2])
            return run_id, row

    def wrap_search(self, original):
        def guarded(args, **kw):
            run_id = ""
            try:
                run_id, row = self.admit_effect("web_search", args)
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL
            result = original(args, **kw)
            try:
                if row is not None:
                    with self.lock:
                        _, current = self.scope()
                        if current is not row:
                            raise ValueError("budget-replaced")
                        row["settled"] += 1
                        if row["closed"]:
                            self.late_results += 1
                            return REFUSAL
                        if type(result) is not str or len(result.encode("utf-8")) > 262144:
                            raise ValueError("result-bound")
                        data = json.loads(result)
                        if data.get("success") is True:
                            web_results = data["data"]["web"]
                            if type(web_results) is not list or len(web_results) > 256:
                                raise ValueError("returned-url-bound")
                            urls = set()
                            for item in web_results:
                                url = item["url"]
                                if type(url) is not str or len(url.encode("utf-8")) > 8192:
                                    raise ValueError("returned-url-type-bound")
                                urls.add(url)
                            if len(urls | row["urls"]) > 256:
                                raise ValueError("returned-url-bound")
                            row["urls"].update(urls)
                return result
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL

        return guarded

    def wrap_extract(self, original):
        async def guarded(args, **kw):
            run_id = ""
            try:
                run_id, row = self.admit_effect("web_extract", args)
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL
            result = await original(args, **kw)
            try:
                if row is not None:
                    with self.lock:
                        _, current = self.scope()
                        if current is not row:
                            raise ValueError("budget-replaced")
                        if row["closed"]:
                            self.late_results += 1
                            return REFUSAL
                return result
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL

        return guarded
