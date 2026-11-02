"""Bounded read-only native Run/session evidence; never consumes the SSE queue."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")
KNOWN = {"web_search", "web_extract", "mcp__vram__vram_status", "tool_search", "tool_describe"}
ROW_CAP = 512
TEXT_CAP = 256 * 1024
TOTAL_CAP = 2 * 1024 * 1024


def require(ok: bool, category: str) -> None:
    if not ok:
        try:
            raise ValueError(category)
        finally:
            print("[native-journey-refusal] " + json.dumps({"version": 1, "category": category}))


def validate_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port is not None
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment,
        "witness_origin",
    )
    return origin.rstrip("/")


def object_json(raw: object) -> dict:
    require(type(raw) is str and len(raw) <= TEXT_CAP, "witness_json_bound")
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("witness_json_shape") from None
    require(type(value) is dict, "witness_json_object")
    return value


def result_object(name: str, raw: str) -> dict:
    require(type(raw) is str and len(raw) <= TEXT_CAP, "witness_result_bound")
    opening = '<untrusted_tool_result source="' + name + '">\n'
    closing = "\n</untrusted_tool_result>"
    if raw.startswith(opening):
        require(raw.endswith(closing) and "\n\n" in raw, "witness_result_envelope")
        raw = raw.split("\n\n", 1)[1][: -len(closing)]
    return object_json(raw)


def status_payload(value: dict) -> dict:
    # Native MCP handler puts structuredContent beside its text result.
    if "structuredContent" in value:
        value = value["structuredContent"]
    elif "result" in value:
        value = value["result"]
        if type(value) is str:
            value = object_json(value)
    require(type(value) is dict, "witness_status_object")
    return value


@contextmanager
def connect_readonly(path: Path):
    require(path.is_file(), "witness_database_missing")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
    db.execute("PRAGMA query_only=ON")
    deadline = time.monotonic() + 3
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    try:
        yield db
    finally:
        db.close()


class NativeRunWitness:
    def __init__(self, api_bearer: str, run_id: str, *, profile: Path, api_origin: str) -> None:
        require(
            type(run_id) is str and RUN_ID.fullmatch(run_id) is not None, "witness_run_identity"
        )
        validate_origin(api_origin)
        self.api_bearer, self.run_id = api_bearer, run_id
        self.profile, self.api_origin = profile, api_origin.rstrip("/")

    def _status(self) -> dict:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        request = urllib.request.Request(
            self.api_origin + "/v1/runs/" + quote(self.run_id, safe=""),
            headers={"Authorization": "Bearer " + self.api_bearer, "Accept": "application/json"},
            method="GET",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=5) as response:
                require(
                    response.status == 200
                    and response.headers.get_content_type() == "application/json",
                    "witness_run_response",
                )
                raw = response.read(TEXT_CAP + 1)
        except (OSError, urllib.error.URLError):
            raise ValueError("witness_run_http") from None
        require(len(raw) <= TEXT_CAP, "witness_run_bound")
        return object_json(raw.decode("utf-8"))

    def read(self) -> dict:
        status = self._status()
        require(
            status.get("object") == "hermes.run" and status.get("run_id") == self.run_id,
            "witness_run_binding",
        )
        terminal = status.get("status")
        require(
            terminal in {"completed", "failed", "cancelled", "interrupted"}, "witness_not_terminal"
        )
        # Production adapter sends no session_id: pinned Runs binds it to run_id.
        require(status.get("session_id") == self.run_id, "witness_session_binding")
        summary = {
            "terminal": terminal,
            "toolCalls": [],
            "toolEventCount": 0,
            "toolRecordCount": 0,
            "evidenceSource": "native_session_db",
            "pairSuccess": False,
            "toolTraceAvailable": False,
        }
        with connect_readonly(self.profile / "state.db") as db:
            session = db.execute(
                "SELECT source FROM sessions WHERE id=?", (self.run_id,)
            ).fetchall()
            if not session and terminal in {"cancelled", "interrupted"}:
                return summary  # Cancellation before agent/session creation; no tool-use claim.
            require(session == [("api_server",)], "witness_session_source")
            sizes = db.execute(
                "SELECT length(content),length(tool_calls) FROM messages "
                "WHERE session_id=? AND active=1 AND (role='tool' OR tool_calls IS NOT NULL) "
                "ORDER BY id LIMIT ?",
                (self.run_id, ROW_CAP + 1),
            ).fetchall()
            require(
                len(sizes) <= ROW_CAP
                and all((a or 0) <= TEXT_CAP and (b or 0) <= TEXT_CAP for a, b in sizes)
                and sum((a or 0) + (b or 0) for a, b in sizes) <= TOTAL_CAP,
                "witness_rows_bound",
            )
            rows = db.execute(
                "SELECT role,CASE WHEN role='tool' THEN content ELSE NULL END,"
                "tool_calls,tool_call_id,tool_name FROM messages "
                "WHERE session_id=? AND active=1 AND (role='tool' OR tool_calls IS NOT NULL) "
                "ORDER BY id LIMIT ?",
                (self.run_id, ROW_CAP + 1),
            ).fetchall()
        pending, by_name, seen = {}, {}, set()
        record_count = 0
        for role, content, calls, call_id, tool_name in rows:
            if calls is not None:
                require(role == "assistant", "witness_request_role")
                try:
                    proposals = json.loads(calls)
                except (ValueError, RecursionError):
                    raise ValueError("witness_calls_json") from None
                require(type(proposals) is list and len(proposals) <= 64, "witness_calls_bound")
                for call in proposals:
                    require(
                        type(call) is dict and type(call.get("function")) is dict,
                        "witness_call_shape",
                    )
                    identity, function = call.get("id"), call["function"]
                    require(
                        type(identity) is str
                        and IDENTITY.fullmatch(identity) is not None
                        and identity not in seen,
                        "witness_call_identity",
                    )
                    seen.add(identity)
                    require(len(seen) <= 128, "witness_call_count")
                    name = function.get("name")
                    arguments = object_json(function.get("arguments"))
                    if name == "tool_call":
                        name = arguments.get("name")
                        arguments = arguments.get("arguments", {})
                        if type(arguments) is str:
                            arguments = object_json(arguments)
                    require(name in KNOWN and type(arguments) is dict, "witness_tool_identity")
                    pending[identity] = (name, arguments)
                    require(len(pending) <= 128, "witness_pending_bound")
                    record_count += 1
            if role != "tool":
                continue
            require(call_id in pending, "witness_unpaired_result")
            name, arguments = pending.pop(call_id)
            require(tool_name == name, "witness_result_tool_binding")
            value = result_object(name, content)
            error = bool(value.get("error")) or value.get("success") is False
            valid = not error
            entry = by_name.setdefault(
                name,
                {
                    "tool": name,
                    "started": 0,
                    "completed": 0,
                    "errorPresent": 0,
                    "durationTotal": None,
                    "pairedRequests": 0,
                    "pairedResults": 0,
                    "successfulResults": 0,
                },
            )
            if name == "web_search" and not error:
                data = value.get("data")
                valid = (
                    value.get("success") is True
                    and type(data) is dict
                    and type(data.get("web")) is list
                )
                require(valid, "witness_search_shape")
                entry["resultRows"] = entry.get("resultRows", 0) + len(data["web"])
            elif name == "web_extract" and not error:
                # Pinned native web_tools returns top-level results, without a success flag.
                rows = value.get("results")
                shaped = (
                    type(rows) is list and bool(rows) and all(type(row) is dict for row in rows)
                )
                requested = arguments.get("urls")
                requested_urls = []
                if type(requested) is list:
                    for item in requested:
                        url = (item.get("url") or item.get("href")) if type(item) is dict else item
                        requested_urls.append(url.strip() if type(url) is str else None)
                source_bound = "urls" not in arguments or (
                    shaped
                    and type(requested) is list
                    and bool(requested_urls)
                    and all(type(url) is str and bool(url) for url in requested_urls)
                    and all(type(row.get("url")) is str for row in rows)
                    and {row["url"] for row in rows} == set(requested_urls)
                )
                valid = (
                    shaped
                    and source_bound
                    and all(
                        not row.get("error")
                        and type(row.get("content")) is str
                        and bool(row["content"].strip())
                        for row in rows
                    )
                )
                error = not valid
                entry["resultRows"] = len(rows) if type(rows) is list else 0
            elif name == "mcp__vram__vram_status" and not error:
                value = status_payload(value)
                require(
                    type(arguments.get("cmdline_chars")) is int and arguments["cmdline_chars"] == 0,
                    "witness_status_commandlines",
                )
                require(
                    type(value.get("gpus")) is list
                    and type(value.get("observations")) is dict
                    and "gpu" in value["observations"]
                    and "ollama" in value["observations"]
                    and not value.get("error")
                    and value.get("ok") is not False,
                    "witness_status_shape",
                )
                require(
                    all(
                        type(row) is dict and "cmdline" not in row
                        for row in value.get("other_processes") or []
                    ),
                    "witness_status_privacy",
                )
                entry["zeroCommandlineRequest"] = True
                entry["statusPayloadStructured"] = True
            for field in ("started", "completed", "pairedRequests", "pairedResults"):
                entry[field] += 1
            entry["errorPresent"] += int(error)
            entry["successfulResults"] += int(valid)
            record_count += 1
        require(not pending or terminal in {"cancelled", "interrupted"}, "witness_unpaired_request")
        summary.update(
            toolCalls=list(by_name.values()),
            toolRecordCount=record_count,
            unpairedRequestCount=len(pending),
            pairSuccess=not pending
            and bool(by_name)
            and all(x["successfulResults"] == x["pairedResults"] for x in by_name.values()),
            toolTraceAvailable=True,
        )
        return summary
