"""Bounded read-only native Run/session evidence; never consumes the SSE queue."""

from __future__ import annotations

import http.client
import json
import re
import socket
import sqlite3
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib.parse import quote, urlsplit

IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")
KNOWN = {"web_search", "web_extract", "mcp__vram__vram_status", "tool_search", "tool_describe"}
ROW_CAP = 512
TEXT_CAP = 256 * 1024
TOTAL_CAP = 2 * 1024 * 1024
# One deadline for the whole read: status request, response and database. Progress never
# extends it, and at expiry the status socket is shut down so a blocked read returns.
READ_DEADLINE_SECONDS = 15.0
STATUS_IDLE_SECONDS = 5.0
# Pinned Runs statuses: live (integration/api.py _LIVE_RUN_STATUSES) and terminal.
KNOWN_RUN_STATUSES = frozenset({
    "queued", "running", "waiting_for_approval", "stopping",
    "completed", "failed", "cancelled", "interrupted",
})


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


def web_row_counts(rows: object, text_key: str, *, source_bound: bool, call_error: bool) -> dict:
    from hermes_realtime.providers.current_facts import _public_http_target

    counts = {"resultRows": 0, "usableRows": 0, "errorRows": 0, "emptyRows": 0, "unboundRows": 0}
    if type(rows) is not list:
        return counts
    counts["resultRows"] = len(rows)
    for row in rows:
        item = row if type(row) is dict else {}
        url, text = item.get("url"), item.get(text_key)
        try:
            target = type(url) is str and bool(url.strip()) and _public_http_target(url)
        except ValueError:
            target = False
        bound = source_bound and target
        has_text = type(text) is str and bool(text.strip())
        error = (bool(item.get("error") or item.get("blocked_by_policy"))
                 or item.get("success") is False)
        counts["usableRows"] += int(bound and has_text and not error and not call_error)
        counts["errorRows"] += int(error)
        counts["emptyRows"] += int(not has_text)
        counts["unboundRows"] += int(not bound)
    return counts


@contextmanager
def connect_readonly(path: Path, overall_deadline: float | None = None):
    require(path.is_file(), "witness_database_missing")
    deadline = time.monotonic() + 3
    if overall_deadline is not None:
        deadline = min(deadline, overall_deadline)
    remaining = deadline - time.monotonic()
    require(remaining > 0, "witness_deadline")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=min(2, remaining))
    try:
        db.execute("PRAGMA query_only=ON")
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        yield db
    except sqlite3.OperationalError:
        # Interrupted by the deadline, or still locked when it came: a deadline refusal.
        require(overall_deadline is None or time.monotonic() < overall_deadline,
                "witness_deadline")
        raise
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
        self._deadline = time.monotonic() + READ_DEADLINE_SECONDS

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        require(remaining > 0, "witness_deadline")
        return remaining

    def _status(self) -> dict:
        # A plain loopback connection: no proxy, and a redirect is not followed (status != 200).
        origin = urlsplit(self.api_origin)
        # validate_origin admitted only the literal loopback host with an explicit port.
        connection = http.client.HTTPConnection(
            "127.0.0.1", origin.port, timeout=min(STATUS_IDLE_SECONDS, self._remaining())
        )
        expired = threading.Event()
        # The socket itself, held here: http.client detaches it from the connection when a
        # response will close (Connection: close, HTTP/1.0), while the response still reads it.
        owned: list[socket.socket] = []

        def expire() -> None:
            expired.set()
            for sock in owned:
                with suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

        timer: threading.Timer | None = None
        response: http.client.HTTPResponse | None = None
        failed = False
        try:
            try:
                connection.connect()
                owned.append(connection.sock)
                timer = threading.Timer(self._remaining(), expire)
                timer.daemon = True
                timer.start()
                connection.request(
                    "GET",
                    "/v1/runs/" + quote(self.run_id, safe=""),
                    headers={
                        "Authorization": "Bearer " + self.api_bearer,
                        "Accept": "application/json",
                    },
                )
                response = connection.getresponse()
                status, content_type = response.status, response.headers.get_content_type()
                raw = response.read(TEXT_CAP + 1)
            except (OSError, http.client.HTTPException):
                failed = True
            # A read cut short by the deadline is never taken for a short response.
            require(not expired.is_set() and time.monotonic() < self._deadline,
                    "witness_deadline")
            if failed:
                raise ValueError("witness_run_http")
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            if response is not None:
                response.close()
            connection.close()
            for sock in owned:
                sock.close()
        require(status == 200 and content_type == "application/json", "witness_run_response")
        require(len(raw) <= TEXT_CAP, "witness_run_bound")
        return object_json(raw.decode("utf-8"))

    def live_status(self) -> str:
        """The bound run's current status, which need not be terminal yet.

        Only one known status category leaves: no session, tool or message evidence is read.
        The same one deadline, run binding and session binding as ``read`` apply.
        """
        self._deadline = time.monotonic() + READ_DEADLINE_SECONDS
        status = self._status()
        require(
            status.get("object") == "hermes.run" and status.get("run_id") == self.run_id,
            "witness_run_binding",
        )
        require(status.get("session_id") == self.run_id, "witness_session_binding")
        value = status.get("status")
        require(type(value) is str and value in KNOWN_RUN_STATUSES, "witness_status_unknown")
        return str(value)

    def read(self) -> dict:
        self._deadline = time.monotonic() + READ_DEADLINE_SECONDS
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
        with connect_readonly(self.profile / "state.db", self._deadline) as db:
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
                    require(
                        type(name) is str and name in KNOWN and type(arguments) is dict,
                        "witness_tool_identity",
                    )
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
            call_error = error
            source_bound = name == "web_search"
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
            if name in {"web_search", "web_extract"}:
                data = value.get("data")
                web_rows = (data.get("web") if type(data) is dict else None)
                counts = web_row_counts(
                    web_rows if name == "web_search" else value.get("results"),
                    "description" if name == "web_search" else "content",
                    source_bound=source_bound and (name == "web_search" or "urls" in arguments),
                    call_error=call_error,
                )
                for field, count in counts.items():
                    entry[field] = entry.get(field, 0) + count
                entry["usableResults"] = (
                    entry.get("usableResults", 0) + int(valid and counts["usableRows"] > 0)
                )
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
