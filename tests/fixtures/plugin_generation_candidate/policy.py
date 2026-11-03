"""Candidate process-lifetime policy for an explicitly designated native Runs route.

Offline fixture only. Active plugin removal is outside its selected lifecycle.
No prompt classification, model names or synthetic scenario names occur here.
"""

import asyncio
import contextlib
import contextvars
import copy
import hashlib
import json
import re
import secrets
import threading
import time
import weakref
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from aiohttp import web
from tools.approval import get_current_session_key
from tools.registry import registry

from hermes_realtime.integration.evidence_presentation import (
    CODEX_PROVIDER_REJECTIONS as CODEX_PROVIDER_REJECTIONS,
)
from hermes_realtime.integration.evidence_presentation import (
    CODEX_REJECTED_EVENT_KINDS as CODEX_REJECTED_EVENT_KINDS,
)
from hermes_realtime.integration.evidence_presentation import (
    MAX_CODEX_STREAM_EVENTS as MAX_CODEX_STREAM_EVENTS,
)
from hermes_realtime.integration.evidence_presentation import (
    MAX_CODEX_STREAM_TEXT_CHARS as MAX_CODEX_STREAM_TEXT_CHARS,
)
from hermes_realtime.integration.evidence_presentation import (
    PRESENTATION_INSTRUCTIONS as PRESENTATION_INSTRUCTIONS,
)
from hermes_realtime.integration.evidence_presentation import (
    PRESENTATION_MODES as PRESENTATION_MODES,
)
from hermes_realtime.integration.evidence_presentation import (
    PRESENTATION_SCHEMA as PRESENTATION_SCHEMA,
)
from hermes_realtime.integration.evidence_presentation import (
    REFUSAL as REFUSAL,
)
from hermes_realtime.integration.evidence_presentation import (
    _codex_refusal_response as _codex_refusal_response,
)
from hermes_realtime.integration.evidence_presentation import (
    _codex_usage as _codex_usage,
)
from hermes_realtime.integration.evidence_presentation import (
    _raw_output as _raw_output,
)
from hermes_realtime.integration.evidence_presentation import (
    _raw_responses_result as _raw_responses_result,
)
from hermes_realtime.integration.evidence_presentation import (
    _replace_codex_final_text as _replace_codex_final_text,
)
from hermes_realtime.integration.evidence_presentation import (
    captured_passages as captured_passages,
)
from hermes_realtime.integration.evidence_presentation import (
    frozen as frozen,
)
from hermes_realtime.integration.evidence_presentation import (
    presentation_input as presentation_input,
)
from hermes_realtime.integration.evidence_presentation import (
    render_presentation_response as render_presentation_response,
)
from hermes_realtime.integration.evidence_presentation import (
    response_facts as response_facts,
)
from hermes_realtime.integration.presentation_transport import (
    Presentation as Presentation,
)
from hermes_realtime.integration.presentation_transport import (
    _consume_codex_stream as _consume_codex_stream,
)
from hermes_realtime.integration.presentation_transport import (
    cancel_presentation as cancel_presentation,
)

VERSION = "ordinary-lookup/v1"
SEARCH_LIMIT = 5
EXTRACT_LIMIT = 3
EXTRACT_TOOL_DESCRIPTION = (
    "Optionally extract supporting passages from one or more listed URLs once; "
    "otherwise give the final answer from available search evidence. Choose a suitable "
    "per-page character budget for the needed inline context."
)
EXTRACT_URLS_DESCRIPTION = "Select one to three distinct URLs from the admitted search results."
EXTRACT_CHAR_LIMIT_DESCRIPTION = (
    "Optional per-page character budget for the inline content returned by this extraction."
)


def lookup_query(value):
    if type(value) is not str or not value.strip() or len(value) > 1024:
        raise ValueError("lookup-objective")
    return value


def host_search_request(query, api_mode="chat_completions"):
    """A policy-authored request, never a provider answer or fabricated tool result."""
    if api_mode == "codex_responses":
        call_id = "host-search-" + secrets.token_hex(12)
        return SimpleNamespace(
            id="host-response-" + secrets.token_hex(12), object="response",
            status="completed", output=[SimpleNamespace(
                id="fc_" + secrets.token_hex(12), type="function_call", call_id=call_id,
                name="web_search", arguments=frozen({"query": query}), status="completed",
            )], output_text=None, usage=None,
        )
    call = SimpleNamespace(id="host-search-" + secrets.token_hex(12), type="function",
        function=SimpleNamespace(name="web_search", arguments=frozen({"query": query})))
    return SimpleNamespace(id="host-request-" + secrets.token_hex(12), object="chat.completion",
        created=int(time.time()), model="host-policy/" + VERSION, usage=None,
        policy_origin="host-initial-retrieval", choices=[SimpleNamespace(index=0,
            finish_reason="tool_calls", message=SimpleNamespace(role="assistant", content=None,
                tool_calls=[call]))])


def _successful_empty_search(result):
    """Recognize only the normalized successful empty search shape."""
    if type(result) is not str:
        return False
    try:
        if len(result.encode("utf-8")) > 262144:
            return False
        value = json.loads(result)
    except (TypeError, UnicodeError, ValueError):
        return False
    if type(value) is not dict or value.get("success") is not True:
        return False
    if "error" in value and value["error"] is not None and value["error"] is not False:
        return False
    data = value.get("data")
    return type(data) is dict and type(data.get("web")) is list and not data["web"]


def _search_recovery_event(category, attempts_reserved):
    categories = {
        "attempt_state_refused", "empty_retry_reserved",
        "empty_retry_budget_full", "empty_retry_already_used",
        "empty_retry_state_refused", "empty_retry_binding_refused",
    }
    if category not in categories:
        category = "attempt_state_refused"
    count = min(max(attempts_reserved, 0), SEARCH_LIMIT) if type(attempts_reserved) is int else 0
    with contextlib.suppress(OSError, TypeError, ValueError):
        print("[lookup-search-recovery] " + frozen({
            "attempts_reserved": count, "category": category, "version": 1,
        }), flush=True)


def _extract_url_diagnostics(args, admitted_urls):
    """Return bounded counts describing a failed URL membership guard."""
    present = "urls" in args
    urls = args.get("urls")
    if not present:
        kind = "missing"
    elif type(urls) is list:
        kind = "list"
    elif urls is None:
        kind = "null"
    elif type(urls) is str:
        kind = "string"
    elif type(urls) is dict:
        kind = "mapping"
    elif type(urls) is bool:
        kind = "boolean"
    elif type(urls) in (int, float):
        kind = "number"
    else:
        kind = "other"

    count = min(len(urls), EXTRACT_LIMIT + 1) if type(urls) is list else 0
    overflow = type(urls) is list and len(urls) > EXTRACT_LIMIT + 1
    facts = {
        "extract_urls_kind": kind,
        "extract_urls_count": count,
        "extract_urls_count_overflow": overflow,
        "extract_urls_rejection_cause": "not_list" if type(urls) is not list else "count",
        "extract_urls_non_string_count": 0,
        "extract_urls_duplicate_count": 0,
        "extract_urls_unadmitted_count": 0,
    }
    if type(urls) is not list or not 1 <= len(urls) <= EXTRACT_LIMIT:
        return facts

    non_strings = sum(type(url) is not str for url in urls)
    facts["extract_urls_non_string_count"] = non_strings
    if non_strings:
        facts["extract_urls_rejection_cause"] = "non_string"
        return facts

    seen = set()
    duplicates = 0
    for url in urls:
        if url in seen:
            duplicates += 1
        else:
            seen.add(url)
    facts["extract_urls_duplicate_count"] = duplicates
    if duplicates:
        facts["extract_urls_rejection_cause"] = "duplicate"
        return facts

    facts["extract_urls_unadmitted_count"] = sum(url not in admitted_urls for url in urls)
    facts["extract_urls_rejection_cause"] = "not_admitted"
    return facts


def _project_selected_request(request, api_mode, row):
    """Narrow only the selected run's tool surface for its current evidence phase."""
    if type(request) is not dict:
        raise ValueError("lookup-request-shape")
    phase = row["phase"]
    if phase == "extract":
        projected = copy.deepcopy(request)
        for key in ("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"):
            projected.pop(key, None)
        return projected
    if phase != "search" or row["settled"] != row["reserved"]:
        return request

    projected = copy.deepcopy(request)
    tools = projected.get("tools")
    eligible = []
    if type(tools) is list:
        for tool in tools:
            if type(tool) is not dict or tool.get("type") != "function":
                continue
            if api_mode == "chat_completions":
                function = tool.get("function")
                if type(function) is not dict or function.get("name") != "web_extract":
                    continue
            else:
                function = tool
                if function.get("name") != "web_extract":
                    continue
            parameters = function.get("parameters")
            properties = parameters.get("properties") if type(parameters) is dict else None
            urls = properties.get("urls") if type(properties) is dict else None
            items = urls.get("items") if type(urls) is dict else None
            if (type(parameters) is not dict or type(properties) is not dict
                    or type(urls) is not dict or urls.get("type") != "array"
                    or type(items) is not dict or items.get("type") != "string"):
                raise ValueError("lookup-extract-schema")
            function["description"] = EXTRACT_TOOL_DESCRIPTION
            urls["description"] = EXTRACT_URLS_DESCRIPTION
            urls["minItems"] = 1
            urls["maxItems"] = EXTRACT_LIMIT
            urls["uniqueItems"] = True
            items["enum"] = sorted(row["urls"])
            char_limit = properties.get("char_limit")
            if char_limit is not None:
                if type(char_limit) is not dict or char_limit.get("type") != "integer":
                    raise ValueError("lookup-extract-schema")
                char_limit["description"] = EXTRACT_CHAR_LIMIT_DESCRIPTION
            eligible.append(tool)
            break
    projected["tools"] = eligible
    projected.pop("functions", None)
    projected.pop("function_call", None)
    if eligible:
        projected["tool_choice"] = "auto"
        projected["parallel_tool_calls"] = False
    else:
        projected.pop("tool_choice", None)
        projected.pop("parallel_tool_calls", None)
    return projected


class Generation:
    def __init__(self, refusal, presentation=None):
        self.epoch = secrets.token_hex(16)
        self.lock = threading.RLock()
        self.members = MappingProxyType({})
        self._membership_authority = self.members
        self.terminal = MappingProxyType({})
        self._terminal_authority = self.terminal
        self.ordinary = weakref.WeakKeyDictionary()
        self._ordinary_authority = self.ordinary
        self.adapter = None
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
        self.presentation = presentation

    def bind_native(self, adapter):
        from gateway.platforms import api_server, api_server_runs
        for module, digest in (
            (api_server, "6aec6687d47c81b3567e930882a499c5c5c041ecbe40e1d08f93038212401962"),
            (api_server_runs, "048ae843592d701ff47437bd8edd47cd64ca6c0fdf88a71045bdb6fb337fbc63"),
        ):
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
                raise ValueError("native-lifecycle-source")
        if type(adapter) is not api_server.APIServerAdapter:
            raise ValueError("native-lifecycle-adapter")
        self.adapter = adapter
        self.native_tasks = adapter._active_run_tasks
        self.native_statuses = adapter._run_statuses
        self.native_loop = asyncio.get_running_loop()
        if (type(self.native_tasks) is not dict or type(self.native_statuses) is not dict
                or self.native_loop.get_task_factory() is not None):
            raise ValueError("native-lifecycle-composition")

    def forget_ordinary(self, task):
        with self.lock:
            if self.ordinary is self._ordinary_authority:
                self.ordinary.pop(task, None)

    def captured_task(self, member, row):
        # Adapter-free unit fixtures do not claim native lifecycle authority.
        if self.adapter is None:
            return row.get("native_task")
        task = member[3]  # Immutable admission authority, never recovered from the row.
        if row.get("native_task") is not task:
            raise ValueError("native-task-authority")
        return task

    def retire(self):
        """Only settled native tasks lose heavy state; old effects never become ordinary."""
        with self.lock:
            if not self.available():
                return
            members = dict(self.members)
            terminal = dict(self.terminal)
            for run_id, member in tuple(members.items()):
                row = member[2]
                try:
                    task = self.captured_task(member, row)
                except ValueError:
                    self.refuse(run_id)
                    self.work_refused()
                    continue
                if (task is not None and self.native_tasks.get(run_id) is not None
                        and self.native_tasks.get(run_id) is not task):
                    self.refuse(run_id)
                    self.work_refused()
                    continue
                status = self.native_statuses.get(run_id) if self.adapter is not None else None
                if (task is None or not task.done() or row.get("owned_work", 0) != 0
                        or row.get("presentation_active") is not None
                        or type(status) is not dict
                        or status.get("status") not in {"completed", "failed", "cancelled"}):
                    continue
                if self.budgets.get(run_id) is not row:
                    self.refuse(run_id)
                    self.work_refused()
                    continue
                row["closed"] = True
                terminal[run_id] = (member[0], member[1], run_id in self.refused)
                while len(terminal) > 128:
                    del terminal[next(iter(terminal))]
                del members[run_id]
                del self.budgets[run_id]
                self.refused = self.refused - {run_id}
            self.members = MappingProxyType(members)
            self._membership_authority = self.members
            self.terminal = MappingProxyType(terminal)
            self._terminal_authority = self.terminal
            self._refusal_authority = self.refused

    def begin_work(self):
        with self.lock:
            _, row = self.scope()
            if row is not None:
                row["owned_work"] = row.get("owned_work", 0) + 1
            return row

    def end_work(self, row):
        if row is not None:
            with self.lock:
                row["owned_work"] -= 1
                self.retire()

    def work_refused(self):
        with contextlib.suppress(OSError, ValueError):
            print('[lookup-lifecycle] {"category":"effect_refused"}', flush=True)

    def refusal_for(self, api_mode):
        if api_mode == "codex_responses":
            return _codex_refusal_response()
        return self.refusal()

    def cancel_presentation(self, row):
        cancel_presentation(self, row)

    def close(self):
        with self.lock:
            self.closed = True
            for member in self._membership_authority.values():
                row = member[2]
                row["closed"] = row["cancelled"] = True
                self.cancel_presentation(row)

    def available(self):
        lifecycle_invalid = (
            self.terminal is not self._terminal_authority
            or self.ordinary is not self._ordinary_authority
            or (self.adapter is not None and (
                self.adapter._active_run_tasks is not self.native_tasks
                or self.adapter._run_statuses is not self.native_statuses
                or self.native_loop.get_task_factory() is not None))
        )
        if lifecycle_invalid:
            self.work_refused()
        if (
            self.members is not self._membership_authority
            or self.refused is not self._refusal_authority
            or lifecycle_invalid
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
        if not designated and (
            run_id in self.terminal or re.fullmatch(r"run_[0-9a-f]{32}", run_id)
        ):
            task = self.native_tasks.get(run_id) if self.adapter is not None else None
            if (task is None or task.done() or self.ordinary.get(task) != run_id):
                raise ValueError("retired-or-unbound-run")
        row = self.budgets.get(run_id)
        if designated:
            member = self.members[run_id]
            if run_id in self.refused or row is not member[2]:
                self.refuse(run_id)
                raise ValueError("budget-unavailable")
            try:
                task = self.captured_task(member, row)
            except ValueError:
                self.refuse(run_id)
                raise
            if task is not None and (task.done() or (
                self.native_tasks.get(run_id) is not None
                and self.native_tasks.get(run_id) is not task
            )):
                self.refuse(run_id)
                raise ValueError("budget-unavailable")
        return run_id, row if designated else None

    def readiness(self):
        with self.lock:
            return {"ready": self.available(), "generation": self.epoch, "policy": VERSION}

    def wire(self, app, adapter):
        try:
            self.bind_native(adapter)
        except BaseException:
            self.closed = True
            self.work_refused()
            raise
        @web.middleware
        async def admission(request, handler):
            admitted_at = time.monotonic()
            starting = request.method == "POST" and request.path == "/v1/runs"
            selected = request.headers.get("X-Lookup-Policy", "")
            reserved = False
            native_error = False
            path = request.path
            profile = request.match_info.get("profile")
            if profile and path.startswith("/p/" + profile + "/"):
                path = path[len("/p/" + profile) :]

            native_start = request.method == "POST" and path == "/v1/runs"

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
                        self.retire()
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
                            if self.presentation is not None:
                                return web.json_response({"error": "lookup_objective"}, status=400)
                            return await call_native()
                        if self.presentation is not None:
                            try:
                                objective = lookup_query(body["input"])
                            except (KeyError, TypeError, ValueError):
                                return web.json_response({"error": "lookup_objective"}, status=400)
                        identity = hashlib.sha256(frozen(body).encode()).hexdigest()
                response = await call_native()
                with self.lock:
                    if native_start and response.status == 202:
                        if not self.available():
                            return web.json_response({"error": "policy_unavailable"}, status=503)
                        accepted = json.loads(response.body)
                        run_id = accepted["run_id"]
                        known = self.members.get(run_id) or self.terminal.get(run_id)
                        if known is not None:
                            if not selected or known[:2] != (VERSION, identity):
                                return web.json_response({"error": "policy_conflict"}, status=409)
                        elif selected:
                            if accepted.get("replayed") is True:
                                return web.json_response(
                                    {"error": "policy_old_generation"}, status=409
                                )
                            task = self.native_tasks.get(run_id)
                            if type(task) is not asyncio.Task or task.done():
                                raise ValueError("native-task-binding")
                            row = {
                                "native_task": task,
                                "owned_work": 0,
                                "deadline": admitted_at + 120,
                                "phase": "new",
                                "closed": False,
                                "cancelled": False,
                                "plan": {},
                                "request": "",
                                "spent": set(),
                                "admitted": set(),
                                "reserved": 0,
                                "settled": 0,
                                "search_attempts_reserved": 0,
                                "empty_retry_reserved": False,
                                "search_primary_started": set(),
                                "urls": set(),
                                "passages": MappingProxyType({}),
                                "objective": objective if self.presentation is not None else None,
                            }
                            self.budgets[run_id] = row
                            members = dict(self.members)
                            members[run_id] = (VERSION, identity, row, task)
                            self.members = MappingProxyType(members)
                            self._membership_authority = self.members
                            task.add_done_callback(lambda _: self.retire())
                        elif accepted.get("replayed") is not True:
                            task = self.native_tasks.get(run_id)
                            if type(task) is not asyncio.Task or task.done():
                                raise ValueError("native-task-binding")
                            self.ordinary[task] = run_id
                            task.add_done_callback(self.forget_ordinary)
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
                            self.cancel_presentation(row)
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
                        elif run_id in self.members or run_id in self.terminal:
                            response.headers["X-Lookup-Generation"] = self.epoch
                            response.headers["X-Lookup-Policy"] = VERSION
                            response.headers["X-Lookup-Policy-Refused"] = (
                                "true" if (run_id in self.refused or
                                           self.terminal.get(run_id, (None, None, False))[2])
                                else "false"
                            )
                        elif re.fullmatch(r"run_[0-9a-f]{32}", run_id):
                            response.headers["X-Lookup-Policy-Outcome"] = "unavailable"
                            self.work_refused()
                return response
            except BaseException:
                if native_error:
                    raise
                with self.lock:
                    self.closed = True
                self.work_refused()
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
        try:
            row = self.begin_work()
        except BaseException:
            self.work_refused()
            return self.refusal_for(kw.get("api_mode", "chat_completions"))
        try:
            return self._model_gate(**kw)
        finally:
            self.end_work(row)

    def _model_gate(self, **kw):
        run_id = ""
        row = None
        response = None
        provider_request = kw["request"]
        native_diagnostics = {}
        api_mode = kw.get("api_mode", "")
        try:
            with self.lock:
                run_id, row = self.scope()
                if row is not None and row["closed"]:
                    return self.refusal_for(api_mode)
                if row is not None and api_mode not in PRESENTATION_MODES:
                    raise ValueError("unsupported-response-mode")
                if (row is not None and self.presentation is not None
                        and self.presentation.config["api_mode"] != api_mode):
                    raise ValueError("presentation-route-mode")
                if row is not None and self.presentation is not None:
                    if row["cancelled"] or time.monotonic() >= row["deadline"]:
                        raise ValueError("lookup-closed")
                    if row["phase"] == "new":
                        query = lookup_query(row["objective"])
                        tools = kw["request"].get("tools")
                        if api_mode == "codex_responses":
                            advertised = [tool for tool in tools if type(tool) is dict
                                and tool.get("type") == "function"
                                and tool.get("name") == "web_search"
                            ] if type(tools) is list else []
                        else:
                            advertised = [tool for tool in tools if type(tool) is dict
                                and type(tool.get("function")) is dict
                                and tool["function"].get("name") == "web_search"
                            ] if type(tools) is list else []
                        if (len(advertised) != 1
                                or advertised[0].get("type") != "function"):
                            raise ValueError("lookup-search-unavailable")
                        response = host_search_request(query, api_mode)
                    elif not row["urls"]:
                        row["presentation_evidence"] = {
                            "category": "refused", "stage": "retrieval",
                            "rejection": "retrieval-required",
                            "failure_kind": "ValueError",
                            "calls": 0, "wire_requests": 0}
                        raise ValueError("retrieval-required")
                    if row["phase"] in {"search", "extract"}:
                        provider_request = _project_selected_request(
                            kw["request"], api_mode, row)
        except BaseException:
            with self.lock:
                self.refuse(run_id)
            evidence = row.get("presentation_evidence") if type(row) is dict else None
            if type(evidence) is dict and evidence.get("stage") == "retrieval":
                with contextlib.suppress(OSError, ValueError):
                    print("[lookup-presentation] " + frozen(evidence), flush=True)
            return self.refusal_for(api_mode)
        if response is None:
            response = kw["next_call"](provider_request)
        if row is None:
            return response
        native_facts = (response_facts(response, "source", api_mode)
                        if self.presentation is not None else None)
        try:
            with self.lock:
                _, current = self.scope()  # Replaced rows are never adopted.
                if current is not row:
                    raise ValueError("budget-replaced")
                if row["closed"]:
                    return self.refusal_for(api_mode)
                if api_mode == "codex_responses":
                    raw_calls, _final_text, _final_item = _raw_responses_result(response)
                    calls = [SimpleNamespace(
                        id=call["call_id"], type="function",
                        function=SimpleNamespace(name=call["name"],
                                                 arguments=call["arguments"]),
                    ) for call in raw_calls]
                else:
                    calls = response.choices[0].message.tool_calls or []
                    if len(response.choices) != 1:
                        raise ValueError("batch-envelope")
                if len(calls) > SEARCH_LIMIT:
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
                    row["search_attempts_reserved"] = len(calls)
                    row["search_primary_started"] = set()
                elif (
                    row["phase"] == "search"
                    and names == {"web_extract"}
                    and len(calls) == 1
                    and row["settled"] == row["reserved"]
                ):
                    extract_args = json.loads(next(iter(plan.values()))[1])
                    urls = extract_args.get("urls")
                    if (
                        type(urls) is not list
                        or not 1 <= len(urls) <= EXTRACT_LIMIT
                        or any(type(url) is not str for url in urls)
                        or len(set(urls)) != len(urls)
                        or any(url not in row["urls"] for url in urls)
                    ):
                        native_diagnostics = _extract_url_diagnostics(extract_args, row["urls"])
                        raise ValueError("url-membership")
                    row["phase"] = "extract"
                else:
                    raise ValueError("batch-budget")
                row["plan"] = plan
                row["request"] = kw["api_request_id"]
                row["spent"], row["admitted"] = set(), set()
            if not calls and self.presentation is not None:
                return self.present(run_id, row, response, kw)
            if getattr(response, "policy_origin", None) == "host-initial-retrieval":
                with contextlib.suppress(OSError, ValueError):
                    print("[lookup-host-request] " + frozen({"origin": "host-policy",
                        "tool_requests": 1, "provider_calls": 0}), flush=True)
            return response
        except BaseException as error:
            with self.lock:
                self.refuse(run_id)
            if native_facts is not None:
                rejection = "native-policy-error"
                if (type(error) is ValueError and len(error.args) == 1
                        and type(error.args[0]) is str and error.args[0] in {
                            "generation-unavailable", "budget-unavailable", "budget-replaced",
                            "batch-envelope", "call-envelope", "search-arguments",
                            "url-membership", "batch-budget",
                        }):
                    rejection = error.args[0]
                elif type(error) is json.JSONDecodeError:
                    rejection = "call-json"
                evidence = {"category": "refused", "stage": "native-response", "calls": 0,
                            "wire_requests": 0, "failure_kind": type(error).__name__,
                            "rejection": rejection,
                            **native_facts, **native_diagnostics}
                row["presentation_evidence"] = evidence
                with contextlib.suppress(OSError, ValueError):
                    print("[lookup-presentation] " + frozen(evidence), flush=True)
            return self.refusal_for(api_mode)

    def present(self, run_id, row, response, kw):
        evidence = {"category": "refused", "stage": "source", "calls": 0, "wire_requests": 0}
        row["presentation_evidence"] = evidence
        api_mode = kw.get("api_mode", "chat_completions")
        try:
            evidence.update(response_facts(response, "source", api_mode))
            if api_mode == "codex_responses":
                calls, source, _final_source = _raw_responses_result(response)
                if calls or type(source) is not str or not source.strip() or len(source) > 4096:
                    raise ValueError("presentation-source")
                message = None
            else:
                message = response.choices[0].message
                source = message.content
                if (response.choices[0].finish_reason != "stop"
                        or getattr(message, "refusal", None)
                        or type(source) is not str or not source.strip() or len(source) > 4096):
                    raise ValueError("presentation-source")
            row["original_response"] = response
            evidence["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            evidence["stage"] = "retrieval"
            if not row["urls"]:
                evidence["rejection"] = "retrieval-required"
                raise ValueError("retrieval-required")
            evidence["stage"] = "catalog"
            with self.lock:
                catalog = tuple(sorted(row["urls"]))
                passages = row.get("passages")
            catalog_data = [{"id": index, "url": url} for index, url in enumerate(catalog)]
            source_input = presentation_input(source, row["objective"], catalog, passages)
            evidence["input_sha256"] = hashlib.sha256(source_input.encode()).hexdigest()
            evidence["catalog_sha256"] = hashlib.sha256(frozen(catalog_data).encode()).hexdigest()
            evidence["catalog_count"] = len(catalog)
            evidence["stage"] = "route"
            self.presentation.validate_route(kw)
            evidence["stage"] = "worker-loop"
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise ValueError("presentation-event-loop")
            evidence["calls"] = 1
            evidence["stage"] = "provider"
            formatted = asyncio.run(self.presentation.complete(
                self, run_id, row, source_input, evidence))
            content = render_presentation_response(formatted, catalog, api_mode, evidence)
            evidence["stage"] = "publication"
            with self.lock:
                _, current = self.scope()
                if (current is not row or row["cancelled"] or time.monotonic() >= row["deadline"]):
                    raise ValueError("presentation-late-result")
                evidence["stage"] = "catalog-publication"
                if set(catalog) != row["urls"]:
                    raise ValueError("presentation-source-catalog-changed")
                if passages is not row.get("passages"):
                    raise ValueError("presentation-evidence-changed")
                # Retain native identity, usage and reasoning; replace only final answer text.
                if api_mode == "codex_responses":
                    result = _replace_codex_final_text(response, content)
                else:
                    result = copy.copy(response)
                    published_choice = copy.copy(response.choices[0])
                    published_message = copy.copy(message)
                    published_message.content = content
                    published_choice.message = published_message
                    result.choices = [published_choice]
                evidence["category"] = "presented"
                evidence["stage"] = "complete"
                return result
        except BaseException as error:
            evidence["failure_kind"] = type(error).__name__
            if (api_mode == "codex_responses" and evidence.get("stage") == "provider"
                    and type(error) is ValueError and len(error.args) == 1
                    and type(error.args[0]) is str
                    and error.args[0] in CODEX_PROVIDER_REJECTIONS):
                evidence["provider_rejection"] = error.args[0]
            with self.lock:
                self.refuse(run_id)
            return self.refusal_for(api_mode)
        finally:
            print("[lookup-presentation] " + frozen(evidence), flush=True)

    def tool_gate(self, **kw):
        try:
            row = self.begin_work()
        except BaseException:
            self.work_refused()
            return REFUSAL
        try:
            return self._tool_gate(**kw)
        finally:
            self.end_work(row)

    def _tool_gate(self, **kw):
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

    def _reserve_search_attempt(self, run_id, row, *, ticket, args, retry=False):
        category = "attempt_state_refused"
        allowed = False
        ordinary_denial = False
        with self.lock:
            try:
                current_run, current = self.scope()
                if current_run != run_id or current is not row or row["closed"]:
                    category = "empty_retry_state_refused" if retry else "attempt_state_refused"
                    raise ValueError("search-attempt-state")
                if row["cancelled"] or time.monotonic() >= row["deadline"]:
                    category = "empty_retry_state_refused" if retry else "attempt_state_refused"
                    raise ValueError("search-attempt-deadline")
                category = "empty_retry_binding_refused" if retry else "attempt_state_refused"
                snapshot = frozen(args)
                plan = row.get("plan")
                planned = row.get("reserved")
                primary_started = row.get("search_primary_started")
                if (
                    type(plan) is not dict
                    or type(planned) is not int or not 1 <= planned <= SEARCH_LIMIT
                    or type(ticket) is not tuple or len(ticket) != 5
                    or self.ticket.get() is not ticket
                    or ticket[0] != run_id
                    or ticket[1] != row["request"]
                    or ticket[2] not in plan
                    or ticket[3:] != ("web_search", snapshot)
                    or row["phase"] != "search"
                    or plan[ticket[2]] != ("web_search", snapshot)
                    or type(row.get("empty_retry_reserved")) is not bool
                    or type(primary_started) is not set
                ):
                    raise ValueError("search-attempt-binding")
                attempts = row.get("search_attempts_reserved")
                if type(attempts) is not int or not 0 <= attempts <= SEARCH_LIMIT:
                    category = "empty_retry_state_refused" if retry else "attempt_state_refused"
                    raise ValueError("search-attempt-state")
                if retry:
                    if (
                        ticket[2] not in primary_started
                        or planned != len(plan)
                        or attempts != planned + int(row["empty_retry_reserved"])
                    ):
                        category = "empty_retry_state_refused"
                        raise ValueError("search-retry-state")
                    if row["empty_retry_reserved"]:
                        category = "empty_retry_already_used"
                        ordinary_denial = True
                    elif attempts >= SEARCH_LIMIT:
                        category = "empty_retry_budget_full"
                        ordinary_denial = True
                    else:
                        row["empty_retry_reserved"] = True
                        row["search_attempts_reserved"] = attempts + 1
                        allowed = True
                        category = "empty_retry_reserved"
                else:
                    expected_attempts = planned + int(row["empty_retry_reserved"])
                    if (
                        ticket[2] in primary_started
                        or attempts != expected_attempts
                        or planned != len(plan)
                    ):
                        raise ValueError("search-primary-state")
                    primary_started.add(ticket[2])
                    allowed = True
                    category = "attempt_reserved"
            except BaseException:
                if not ordinary_denial:
                    self.refuse(run_id)
        if retry or not allowed:
            _search_recovery_event(category, row.get("search_attempts_reserved", 0))
        return allowed

    def wrap_search(self, original):
        inner = self._wrap_search(original)
        def guarded(args, **kw):
            try:
                row = self.begin_work()
            except BaseException:
                self.work_refused()
                return REFUSAL
            try:
                return inner(args, **kw)
            finally:
                self.end_work(row)
        return guarded

    def _wrap_search(self, original):
        def guarded(args, **kw):
            run_id = ""
            try:
                run_id, row = self.admit_effect("web_search", args)
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL
            ticket = self.ticket.get()
            if row is not None and not self._reserve_search_attempt(
                run_id, row, ticket=ticket, args=args,
            ):
                return REFUSAL
            result = original(args, **kw)
            if (
                row is not None
                and _successful_empty_search(result)
                and self._reserve_search_attempt(
                    run_id, row, ticket=ticket, args=args, retry=True,
                )
            ):
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
                        if data.get("success") is True and not data.get("error"):
                            web_results = data["data"]["web"]
                            if type(web_results) is not list or len(web_results) > 256:
                                raise ValueError("returned-url-bound")
                            urls = set()
                            for item in web_results:
                                url = item["url"]
                                if type(url) is not str or len(url.encode("utf-8")) > 8192:
                                    raise ValueError("returned-url-type-bound")
                                if url.strip():
                                    urls.add(url)
                            if len(urls | row["urls"]) > 256:
                                raise ValueError("returned-url-bound")
                            row["urls"].update(urls)
                            if self.presentation is not None:
                                self.capture_passages(row, web_results, "search", urls)
                return result
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL

        return guarded

    def wrap_extract(self, original):
        inner = self._wrap_extract(original)
        async def guarded(args, **kw):
            try:
                row = self.begin_work()
            except BaseException:
                self.work_refused()
                return REFUSAL
            try:
                return await inner(args, **kw)
            finally:
                self.end_work(row)
        return guarded

    def _wrap_extract(self, original):
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
                        if self.presentation is not None:
                            category = "bound"
                            try:
                                if type(result) is not str or len(result) > 262144:
                                    raise ValueError("extract-result-bound")
                                if len(result.encode("utf-8")) > 262144:
                                    raise ValueError("extract-result-bound")
                                category = "shape"
                                data = json.loads(result)
                                items = data["results"] if not data.get("error") else None
                                category = None
                                if items is not None:
                                    self.capture_passages(row, items, "extract", set(args["urls"]))
                            finally:
                                if category is not None:
                                    print("[lookup-evidence] " + frozen({
                                        "category": category, "kind": "extract"}), flush=True)
                return result
            except BaseException:
                with self.lock:
                    self.refuse(run_id)
                return REFUSAL

        return guarded

    def capture_passages(self, row, items, kind, admitted):
        row["passages"] = captured_passages(
            row.get("passages", MappingProxyType({})), items, kind, admitted, row["urls"])
