"""Observe native web results and present bounded evidence for opted-in Runs.

This module is observational: failures never suppress or repeat a native tool call.
It owns only per-run snapshots and formatting; Hermes retains tool admission and
execution authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

from aiohttp import web

from .evidence_presentation import (
    _codex_refusal_response,
    _raw_responses_result,
    _replace_codex_final_text,
    captured_passages,
    frozen,
    presentation_input,
    render_presentation_response,
)
from .presentation_transport import Presentation, cancel_presentation
from .web_evidence_contract import (
    ATTESTATION_HEADER,
    CONTRACT,
    CONTRACT_HEADER,
    GENERATION_HEADER,
    READY_PATH,
    REPORT_MAX_BYTES,
    RUN_HEADER,
    valid_generation,
)

_RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")
_MAX_ROWS = 128
_MAX_RESULT_BYTES = 262_144
_MAX_ITEMS = 256
_MAX_URL_BYTES = 8192
_MAX_INPUT_BYTES = 16_384
_METADATA_TOOLS = frozenset({"tool_search", "tool_describe"})
_WEB_TOOLS = frozenset({"web_search", "web_extract"})


@dataclass(slots=True)
class _RequestBinding:
    contract: str | None
    generation: str | None
    objective: str | None
    deadline: float = 0.0


def _text_content(value: object) -> str | None:
    if type(value) is str:
        return value
    if type(value) is list:
        parts: list[str] = []
        for item in value:
            if type(item) is str:
                parts.append(item)
            elif type(item) is dict and item.get("type") in {"input_text", "text"}:
                text = item.get("text")
                if type(text) is not str:
                    return None
                parts.append(text)
            else:
                return None
        return "".join(parts)
    return None


def _request_objective(body: object) -> str | None:
    if type(body) is not dict:
        return None
    raw = body.get("input")
    if type(raw) is str:
        objective = raw
    elif type(raw) is list:
        objective = None
        for item in reversed(raw):
            if type(item) is dict and item.get("role") == "user":
                objective = _text_content(item.get("content"))
                break
    else:
        objective = None
    if (
        type(objective) is not str
        or not objective.strip()
        or len(objective.encode("utf-8")) > _MAX_INPUT_BYTES
    ):
        return None
    return objective


def _request_binding_for(
    contract: object,
    generation: object,
    body: object,
    *,
    expected_generation: str,
    admitted_at: float,
) -> _RequestBinding:
    if contract != CONTRACT:
        raise ValueError("web-evidence-contract")
    if (
        type(generation) is not str
        or not valid_generation(generation)
        or generation != expected_generation
    ):
        raise ValueError("web-evidence-generation")
    objective = _request_objective(body)
    if objective is None:
        raise ValueError("web-evidence-objective")
    return _RequestBinding(CONTRACT, generation, objective, admitted_at + 120.0)


def _extract_text_result(
    result: object,
    *,
    tool: str,
) -> tuple[str, list[dict[str, Any]]]:
    if (
        type(result) is not str
        or len(result) > _MAX_RESULT_BYTES
        or len(result.encode("utf-8")) > _MAX_RESULT_BYTES
    ):
        raise ValueError("native-result-bound")
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("native-result-json") from exc
    if type(payload) is not dict:
        raise ValueError("native-result-unsuccessful")
    error = payload.get("error")
    if payload.get("success") is False:
        if type(error) is str and error:
            # Known operational failure: its error and any attached data are not evidence.
            return result, []
        raise ValueError("native-result-unsuccessful")
    if error:
        raise ValueError("native-result-unsuccessful")
    if tool == "web_search":
        if payload.get("success") is not True:
            raise ValueError("native-result-unsuccessful")
        data = payload.get("data")
        items = data.get("web") if type(data) is dict else None
    else:
        items = payload.get("results")
    if type(items) is not list or len(items) > _MAX_ITEMS:
        raise ValueError("native-result-shape")
    for item in items:
        if type(item) is not dict:
            raise ValueError("native-result-item")
        url = item.get("url")
        if (
            type(url) is not str
            or len(url) > _MAX_URL_BYTES
            or len(url.encode("utf-8")) > _MAX_URL_BYTES
        ):
            raise ValueError("native-result-url")
    return result, cast(list[dict[str, Any]], items)


def _terminal_text(response: Any, api_mode: str) -> str:
    if api_mode == "codex_responses":
        calls, text, _final = _raw_responses_result(response)
        if calls or type(text) is not str or not text.strip():
            raise ValueError("native-final-response")
        return text
    choices = getattr(response, "choices", None)
    if type(choices) is not list or len(choices) != 1:
        raise ValueError("native-final-response")
    choice = choices[0]
    message = getattr(choice, "message", None)
    text = getattr(message, "content", None)
    if (
        getattr(choice, "finish_reason", None) != "stop"
        or getattr(message, "refusal", None)
        or type(text) is not str
        or not text.strip()
    ):
        raise ValueError("native-final-response")
    return text


def _has_tool_calls(response: Any, api_mode: str) -> bool:
    if api_mode == "codex_responses":
        calls, _text, _final = _raw_responses_result(response)
        return bool(calls)
    choices = getattr(response, "choices", None)
    return bool(
        type(choices) is list
        and choices
        and getattr(getattr(choices[0], "message", None), "tool_calls", None)
    )


class NativeWebEvidence:
    """Native plugin observer bound to one API-server generation."""

    def __init__(self, presentation: Presentation, *, generation: str | None = None) -> None:
        self.generation = generation or uuid4().hex
        if not valid_generation(self.generation):
            raise ValueError("web-evidence-generation")
        self.presentation = presentation
        self.lock = threading.RLock()
        self.rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.reports: OrderedDict[str, str] = OrderedDict()
        self._pending = 0
        self.adapter: Any = None
        self._get_session_key: Callable[[], str] | None = None
        self._closed = False

    def scope(self) -> tuple[str, dict[str, Any] | None]:
        if self._get_session_key is None:
            return "", None
        run_id = self._get_session_key()
        row = self.rows.get(run_id)
        if row is None:
            return run_id, None
        if row.get("native_task") is not self._task_for_run(run_id):
            raise ValueError("web-evidence-task-binding")
        return run_id, row

    def _get_run_id(self) -> str:
        if self._get_session_key is None:
            raise ValueError("web-evidence-session-unavailable")
        run_id = self._get_session_key()
        if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("web-evidence-run-id")
        return run_id

    def _task_for_run(self, run_id: str) -> asyncio.Task[Any]:
        tasks = getattr(self.adapter, "_active_run_tasks", None)
        if type(tasks) is not dict:
            raise ValueError("web-evidence-task-map")
        task = tasks.get(run_id)
        if type(task) is not asyncio.Task or task.done():
            raise ValueError("web-evidence-task-missing")
        return task

    def _make_row(
        self,
        binding: _RequestBinding,
        run_id: str,
        task: asyncio.Task[Any],
    ) -> dict[str, Any]:
        return {
            "contract": binding.contract,
            "generation": binding.generation,
            "objective": binding.objective,
            "native_task": task,
            "deadline": binding.deadline,
            "cancelled": False,
            "closed": False,
            "invalid": False,
            "web_seen": False,
            "other_tool_seen": False,
            "urls": set(),
            "passages": MappingProxyType({}),
            "catalog_count": 0,
            "passage_count": 0,
            "truncated_count": 0,
            "missing_count": 0,
            "disposition": None,
            "input_sha256": None,
            "input_bytes": 0,
            "catalog_sha256": None,
            "output_sha256": None,
            "presentation_active": None,
        }

    def _retire_locked(self) -> None:
        for run_id, row in tuple(self.rows.items()):
            task = row.get("native_task")
            if type(task) is asyncio.Task and task.done() and not row.get("presentation_active"):
                row["closed"] = True
                self._drop_source_locked(row)
                del self.rows[run_id]

    def _row_for_effect(self) -> tuple[str, dict[str, Any] | None]:
        run_id = self._get_run_id()
        with self.lock:
            if self._closed:
                raise ValueError("web-evidence-closed")
            self._retire_locked()
            row = self.rows.get(run_id)
            if row is None:
                # Ordinary Runs are deliberately outside this observer.
                return run_id, None
            task = self._task_for_run(run_id)
            if row["native_task"] is not task:
                row["invalid"] = True
                row["disposition"] = "refused"
                raise ValueError("web-evidence-binding")
            if row["cancelled"] or row["closed"]:
                row["invalid"] = True
                row["disposition"] = "refused"
                raise ValueError("web-evidence-closed")
            return run_id, row

    def _accepted_row(
        self,
        binding: _RequestBinding,
        run_id: str,
        replayed: bool,
    ) -> dict[str, Any] | None:
        with self.lock:
            existing = self.rows.get(run_id)
            if existing is not None:
                if replayed:
                    raw = self.reports.get(run_id)
                    try:
                        report = json.loads(raw) if type(raw) is str else None
                    except (ValueError, UnicodeError):
                        report = None
                    if (
                        type(report) is dict
                        and report.get("contract") == binding.contract
                        and report.get("generation") == binding.generation
                        and report.get("run_id") == run_id
                        and existing["generation"] == binding.generation
                        and existing["contract"] == binding.contract
                    ):
                        return existing
                try:
                    task = self._task_for_run(run_id)
                except ValueError:
                    task = None
                if (
                    existing["native_task"] is task
                    and existing["generation"] == binding.generation
                    and existing["contract"] == binding.contract
                    and existing["objective"] == binding.objective
                ):
                    return existing
                existing["invalid"] = True
                existing["disposition"] = "refused"
                return None
            if replayed:
                raw = self.reports.get(run_id)
                try:
                    report = json.loads(raw) if type(raw) is str else None
                except (ValueError, UnicodeError):
                    report = None
                if (
                    type(report) is dict
                    and report.get("contract") == binding.contract
                    and report.get("generation") == binding.generation
                    and report.get("run_id") == run_id
                ):
                    return cast(dict[str, Any], report)
                return None
            try:
                task = self._task_for_run(run_id)
            except ValueError:
                return None
            if len(self.rows) >= _MAX_ROWS:
                return None
            row = self._make_row(binding, run_id, task)
            self.rows[run_id] = row
            return row

    def _refuse(self, run_id: str, row: dict[str, Any] | None) -> None:
        emit = False
        try:
            with self.lock:
                if row is not None and self.rows.get(run_id) is row:
                    emit = not row.get("refusal_logged", False)
                    row["refusal_logged"] = True
                    row["invalid"] = True
                    row["disposition"] = "refused"
        finally:
            if emit:
                print('[web-evidence] {"category":"refused","count":1}', flush=True)

    def _observe_result(
        self, run_id: str, row: dict[str, Any], tool: str, args: object, result: object
    ) -> None:
        if tool not in _WEB_TOOLS:
            if tool not in _METADATA_TOOLS:
                with self.lock:
                    if self.rows.get(run_id) is row:
                        row["other_tool_seen"] = True
            return
        try:
            _raw, items = _extract_text_result(result, tool=tool)
            with self.lock:
                if self.rows.get(run_id) is not row or row["native_task"] is not self._task_for_run(
                    run_id
                ):
                    raise ValueError("web-evidence-late-result")
                row["web_seen"] = True
                if tool == "web_search":
                    returned = {item["url"] for item in items if item["url"].strip()}
                    if len(returned | row["urls"]) > 256:
                        raise ValueError("web-evidence-catalog-bound")
                    row["urls"].update(returned)
                    row["passages"] = captured_passages(
                        row["passages"],
                        items,
                        "search",
                        returned,
                        returned,
                    )
                else:
                    requested = args.get("urls") if type(args) is dict else None
                    if (
                        type(requested) is not list
                        or not 1 <= len(requested) <= 5
                        or any(type(url) is not str for url in requested)
                        or any(
                            not url or len(url.encode("utf-8")) > _MAX_URL_BYTES
                            for url in requested
                        )
                    ):
                        raise ValueError("web-evidence-extract-request")
                    requested_urls = set(requested)
                    if len(row["urls"] | requested_urls) > 256:
                        raise ValueError("web-evidence-catalog-bound")
                    row["urls"].update(requested_urls)
                    passages = dict(row["passages"])
                    for url in requested_urls:
                        passages.setdefault(url, ())
                    row["passages"] = MappingProxyType(passages)
                    row["passages"] = captured_passages(
                        row["passages"],
                        items,
                        "extract",
                        requested_urls,
                        row["urls"],
                    )
                row["catalog_count"] = len(row["urls"])
                row["passage_count"] = sum(len(values) for values in row["passages"].values())
                if row["passage_count"] > 2048:
                    raise ValueError("web-evidence-passage-count")
        except BaseException:
            self._refuse(run_id, row)

    def tool_execution(self, **kw: Any) -> Any:
        """Wrap the native effect exactly once and retain its original result."""
        tool = kw.get("tool_name")
        if type(tool) is not str:
            tool = kw.get("function_name")
        next_call = kw.get("next_call")
        args = kw.get("args")
        if type(tool) is not str or not callable(next_call):
            # Native middleware failure is fail-open; never skip native next_call.
            raise RuntimeError("native tool execution middleware payload invalid")
        if tool in _METADATA_TOOLS:
            return next_call(args)
        try:
            run_id, row = self._row_for_effect()
            if row is None:
                return next_call(args)
            if row is not None and tool in _WEB_TOOLS:
                with self.lock:
                    row["web_seen"] = True
        except BaseException:
            run_id, row = "", None
            try:
                run_id = self._get_run_id()
                refused_row = self.rows.get(run_id)
            except BaseException:
                refused_row = None
            self._refuse(run_id, refused_row)
        try:
            result = next_call(args)
        except BaseException:
            raise
        if row is None:
            return result
        if inspect.isawaitable(result):

            async def observe_async() -> Any:
                try:
                    resolved = await result
                except BaseException:
                    raise
                self._observe_result(run_id, row, tool, args, resolved)
                return resolved

            return observe_async()
        self._observe_result(run_id, row, tool, args, result)
        return result

    def llm_execution(self, **kw: Any) -> Any:
        request = kw.get("request")
        next_call = kw.get("next_call")
        if not callable(next_call):
            raise RuntimeError("native LLM execution middleware payload invalid")
        try:
            run_id, row = self.scope()
        except BaseException:
            try:
                run_id = self._get_run_id()
                row = self.rows.get(run_id)
            except BaseException:
                row = None
            if row is None:
                return next_call(request)
            self._refuse(run_id, row)
        if row is None:
            return next_call(request)
        try:
            self.presentation.validate_route(kw)
        except BaseException:
            self._refuse(run_id, row)
        result = next_call(request)
        if inspect.isawaitable(result):
            return self._finish_after_await(run_id, row, result, kw)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._finish_response(run_id, row, result, kw))
        return self._finish_response(run_id, row, result, kw)

    async def _finish_after_await(
        self, run_id: str, row: dict[str, Any], result: Awaitable[Any], kw: dict[str, Any]
    ) -> Any:
        response = await result
        return await self._finish_response(run_id, row, response, kw)

    async def _finish_response(
        self, run_id: str, row: dict[str, Any], response: Any, kw: dict[str, Any]
    ) -> Any:
        try:
            api_mode = kw.get("api_mode", "chat_completions")
            if _has_tool_calls(response, api_mode):
                return response
            text = _terminal_text(response, api_mode)
            with self.lock:
                self._assert_current(run_id, row)
                if row["invalid"]:
                    raise ValueError("web-evidence-capture-refused")
                if row["other_tool_seen"] and row["web_seen"]:
                    row["disposition"] = "mixed_ungrounded"
                    row["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    row["closed"] = True
                    self._store_report_locked(run_id, row)
                    self._drop_source_locked(row)
                    return response
                if not row["web_seen"]:
                    row["disposition"] = "no_web"
                    row["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    row["closed"] = True
                    self._store_report_locked(run_id, row)
                    self._drop_source_locked(row)
                    return response
                if not row["urls"] or not row["objective"]:
                    raise ValueError("web-evidence-insufficient")
                if time.monotonic() >= row["deadline"]:
                    raise ValueError("web-evidence-deadline")
                catalog = tuple(sorted(row["urls"]))
                passages = row["passages"]
                source_input = presentation_input(row["objective"], catalog, passages)
                if len(source_input.encode("utf-8")) > _MAX_INPUT_BYTES:
                    raise ValueError("web-evidence-input-bound")
                source_payload = json.loads(source_input)
                missing = len(source_payload["missing_source_ids"])
                evidence_items = source_payload["evidence"]
                passage_count = sum(
                    item["kind"] in {"search", "extract"} and bool(item["text"])
                    for item in evidence_items
                )
                if not catalog or not passage_count or missing >= len(catalog):
                    raise ValueError("web-evidence-insufficient")
                truncated_count = sum(item["truncated"] is True for item in evidence_items)
                row["input_sha256"] = hashlib.sha256(source_input.encode()).hexdigest()
                row["input_bytes"] = len(source_input.encode())
                row["passage_count"] = passage_count
                row["missing_count"] = missing
                row["truncated_count"] = truncated_count
                row["catalog_sha256"] = hashlib.sha256(
                    frozen(
                        [{"id": index, "url": url} for index, url in enumerate(catalog)]
                    ).encode()
                ).hexdigest()
            evidence: dict[str, Any] = {
                "wire_requests": 0,
                "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            formatted = await self.presentation.complete(self, run_id, row, source_input, evidence)
            rendered = render_presentation_response(
                formatted, catalog, api_mode, evidence, passages=passages)
            with self.lock:
                self._assert_current(run_id, row, enforce_deadline=True)
                if row["urls"] != set(catalog) or row["passages"] is not passages:
                    raise ValueError("web-evidence-source-changed")
                if api_mode == "codex_responses":
                    published = _replace_codex_final_text(response, rendered)
                else:
                    import copy

                    published = copy.copy(response)
                    choice = copy.copy(response.choices[0])
                    message = copy.copy(response.choices[0].message)
                    message.content = rendered
                    choice.message = message
                    published.choices = [choice]
                row["disposition"] = "presented"
                row["output_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
                row["closed"] = True
                self._store_report_locked(run_id, row)
                self._drop_source_locked(row)
                return published
        except asyncio.CancelledError:
            self._refuse(run_id, row)
            raise
        except BaseException:
            self._refuse(run_id, row)
            refusal = self._refusal_response(response, kw.get("api_mode", "chat_completions"))
            try:
                text = _terminal_text(refusal, kw.get("api_mode", "chat_completions"))
                with self.lock:
                    row["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    row["closed"] = True
                    self._store_report_locked(run_id, row)
                    self._drop_source_locked(row)
            except BaseException:
                pass
            return refusal

    @staticmethod
    def _refusal_response(response: Any, api_mode: str) -> Any:
        if api_mode == "codex_responses":
            return _codex_refusal_response()
        from openai.types.chat import ChatCompletion  # type: ignore[import-not-found]

        from .evidence_presentation import REFUSAL

        return ChatCompletion.model_validate(
            {
                "id": "web-evidence-refusal",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": REFUSAL, "role": "assistant"},
                    }
                ],
                "created": 0,
                "model": "web-evidence-refusal",
                "object": "chat.completion",
            }
        )

    def _assert_current(
        self, run_id: str, row: dict[str, Any], *, enforce_deadline: bool = False
    ) -> None:
        if (
            self.rows.get(run_id) is not row
            or row["native_task"] is not self._task_for_run(run_id)
            or row["cancelled"]
            or row["closed"]
            or row["invalid"]
            or (enforce_deadline and time.monotonic() >= row["deadline"])
        ):
            raise ValueError("web-evidence-publish-closed")

    def _report(self, run_id: str, row: dict[str, Any]) -> str:
        disposition = row.get("disposition")
        if disposition not in {"presented", "no_web", "mixed_ungrounded"}:
            disposition = "refused"
        presented = disposition == "presented"
        body = {
            "version": 1,
            "contract": CONTRACT,
            "generation": row["generation"],
            "run_id": run_id,
            "disposition": disposition,
            "output_sha256": row.get("output_sha256"),
            "input_sha256": row.get("input_sha256") if presented else None,
            "input_bytes": row.get("input_bytes", 0) if presented else 0,
            "catalog_sha256": row.get("catalog_sha256") if presented else None,
            "catalog_count": row.get("catalog_count", 0) if presented else 0,
            "passage_count": row.get("passage_count", 0) if presented else 0,
            "truncated_count": row.get("truncated_count", 0) if presented else 0,
            "missing_count": row.get("missing_count", 0) if presented else 0,
        }
        raw = frozen(body)
        if len(raw.encode("utf-8")) > REPORT_MAX_BYTES:
            raise ValueError("web-evidence-report-bound")
        return raw

    def _finish_report(self, run_id: str, row: dict[str, Any]) -> str:
        with self.lock:
            if self.rows.get(run_id) is not row:
                raise ValueError("web-evidence-row-retired")
            return self._store_report_locked(run_id, row)

    def _store_report_locked(self, run_id: str, row: dict[str, Any]) -> str:
        report = self._report(run_id, row)
        self.reports[run_id] = report
        self.reports.move_to_end(run_id)
        while len(self.reports) > _MAX_ROWS:
            self.reports.popitem(last=False)
        return report

    @staticmethod
    def _drop_source_locked(row: dict[str, Any]) -> None:
        row["objective"] = None
        row["passages"] = MappingProxyType({})
        row["urls"] = set()

    def stop(self, run_id: str) -> None:
        with self.lock:
            row = self.rows.get(run_id)
            if row is not None:
                row["cancelled"] = row["closed"] = True
                row["disposition"] = "refused"
                cancel_presentation(self, row)

    def close(self) -> None:
        """Close new observer work and cancel only owned presentation tasks."""
        with self.lock:
            self._closed = True
            for row in self.rows.values():
                row["cancelled"] = row["closed"] = True
                row["disposition"] = "refused"
                cancel_presentation(self, row)

    def install(self, app: web.Application, adapter: Any) -> None:
        authorize = getattr(adapter, "_check_auth", None)
        if not callable(authorize):
            raise ValueError("web-evidence-native-auth-required")
        self.adapter = adapter

        if self._get_session_key is None:
            try:
                from tools.approval import get_current_session_key  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("web-evidence-native-session-unavailable") from exc
            self._get_session_key = lambda: get_current_session_key(default="")

        @web.middleware
        async def request_binding(
            request: web.Request, handler: Callable[..., Awaitable[web.StreamResponse]]
        ) -> web.StreamResponse:
            selected = request.method == "POST" and request.path == "/v1/runs"
            binding = None
            if selected and (
                CONTRACT_HEADER in request.headers or GENERATION_HEADER in request.headers
            ):
                with self.lock:
                    if self._closed:
                        return web.json_response({"error": "web_evidence_closed"}, status=503)
                if asyncio.get_running_loop().get_task_factory() is not None:
                    return web.json_response({"error": "web_evidence_task_factory"}, status=503)
                admitted_at = time.monotonic()
                try:
                    binding = _request_binding_for(
                        request.headers.get(CONTRACT_HEADER),
                        request.headers.get(GENERATION_HEADER),
                        await request.json(),
                        expected_generation=self.generation,
                        admitted_at=admitted_at,
                    )
                except ValueError as error:
                    category = error.args[0] if error.args else "web-evidence-contract"
                    if category == "web-evidence-generation":
                        return web.json_response({"error": category}, status=409)
                    return web.json_response({"error": category}, status=400)
                except BaseException:
                    return web.json_response({"error": "web-evidence-objective"}, status=400)
                with self.lock:
                    self._retire_locked()
                    if len(self.rows) + self._pending >= _MAX_ROWS:
                        return web.json_response({"error": "web_evidence_capacity"}, status=503)
                    self._pending += 1
            try:
                response = await handler(request)
            finally:
                if selected and binding is not None:
                    with self.lock:
                        self._pending -= 1
            if selected and binding is not None and response.status == 202:
                try:
                    accepted = json.loads(getattr(response, "body", b""))
                    run_id = accepted.get("run_id")
                    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
                        return response
                    row = self._accepted_row(
                        binding,
                        run_id,
                        accepted.get("replayed") is True,
                    )
                    if row is not None:
                        response.headers[CONTRACT_HEADER] = CONTRACT
                        response.headers[GENERATION_HEADER] = self.generation
                        response.headers[RUN_HEADER] = run_id
                except BaseException:
                    pass
            if (
                request.method == "POST"
                and request.path.endswith("/stop")
                and response.status == 200
            ):
                parts = request.path.split("/")
                if len(parts) == 5 and parts[1:3] == ["v1", "runs"]:
                    self.stop(parts[3])
            if (
                request.method == "GET"
                and request.path.startswith("/v1/runs/")
                and response.status == 200
            ):
                run_id = request.path.rsplit("/", 1)[-1]
                try:
                    status = json.loads(getattr(response, "body", b""))
                    if status.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
                        with self.lock:
                            row = self.rows.get(run_id)
                            report = self.reports.get(run_id)
                        if row is not None:
                            report = self._finish_report(run_id, row)
                        if report is not None:
                            response.headers[ATTESTATION_HEADER] = report
                        if row is not None:
                            with self.lock:
                                task = row.get("native_task")
                                if (
                                    type(task) is asyncio.Task
                                    and task.done()
                                    and not row.get("presentation_active")
                                ):
                                    row["closed"] = True
                                    self._drop_source_locked(row)
                                    self.rows.pop(run_id, None)
                except BaseException:
                    pass
            return response

        app.middlewares.append(request_binding)

        async def ready(request: web.Request) -> web.Response:
            refused = authorize(request)
            if refused is not None:
                return cast(web.Response, refused)
            return web.json_response(
                {
                    "contract": CONTRACT,
                    "generation": self.generation,
                    "ready": not self._closed,
                },
                status=200 if not self._closed else 503,
            )

        app.router.add_get(READY_PATH, ready)

    def register(self, context: object) -> None:
        register_middleware = getattr(context, "register_middleware", None)
        register_platform = getattr(context, "register_platform_handler", None)
        if not callable(register_middleware) or not callable(register_platform):
            raise RuntimeError("web-evidence-plugin-api-unavailable")
        register_middleware("tool_execution", self.tool_execution)
        register_middleware("llm_execution", self.llm_execution)
        register_platform("api_server", self.install)


def _verify_live_credential_runtime(presentation_config: dict[str, Any]) -> None:
    """Prove the selected native source after the gateway's real env loading."""
    accepted = False
    try:
        import importlib
        import os
        import sys
        import sysconfig
        from pathlib import Path

        from .codex_access_credentials import validate_native_binding
        from .native_credential_probe import verify_native_credential_resolution

        native_config = importlib.import_module("hermes_cli.config").load_config()
        expected_presentation = native_config["plugins"]["entries"]["hermes-realtime"][
            "settings"]["presentation"]
        if expected_presentation != presentation_config:
            raise ValueError("credential_config_drift")
        helper = Path(__file__).with_name("codex_access_credentials.py")
        executable = Path(sys._base_executable)  # type: ignore[attr-defined]
        validate_native_binding(native_config, executable, helper)
        selected = native_config["model"]["provider"]
        provider = native_config["providers"][selected.split(":", 1)[1]]
        native_package = importlib.import_module("hermes_cli")
        if native_package.__file__ is None:
            raise ValueError("credential_runtime_origin")
        verify_native_credential_resolution(
            executable=executable,
            hermes_source=Path(native_package.__file__).resolve().parent.parent,
            native_site_packages=Path(sysconfig.get_path("purelib")),
            home=Path(os.environ["HERMES_HOME"]), env_file=None,
            provider=selected, base_url=provider["base_url"], model=provider["default_model"],
            api_mode=provider["api_mode"], command=provider["key_cmd"],
            environment=dict(os.environ),
        )
        accepted = True
    except Exception:
        raise RuntimeError("native_credential_binding") from None
    finally:
        if not accepted:
            import sys

            print('[native-credential-refused] {"version":1,"category":"runtime_refused"}',
                  file=sys.stderr, flush=True)


def register_web_evidence(context: object) -> NativeWebEvidence | None:
    """Install the explicitly enabled native evidence observer; default is off."""
    get_config = getattr(context, "get_config", None)
    if not callable(get_config):
        return None
    enabled = get_config("web_evidence", False)
    if type(enabled) is not bool:
        raise RuntimeError("web-evidence-config-invalid")
    if not enabled:
        return None
    presentation_config = get_config("presentation", None)
    if type(presentation_config) is not dict:
        raise RuntimeError("web-evidence-presentation-config-required")
    on_unload = getattr(context, "on_unload", None)
    if not callable(on_unload):
        raise RuntimeError("web-evidence-unload-hook-required")
    presentation = Presentation(presentation_config)
    if "key_cmd_argv" in presentation_config:
        _verify_live_credential_runtime(presentation_config)
    observer = NativeWebEvidence(presentation)
    observer.register(context)
    on_unload(observer.close)
    return observer
