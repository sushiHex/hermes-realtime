"""Bounded private child output; delegates all process ownership to Processes."""

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate diagnostic field")
    return value


def _foreground_failure_category(raw: bytes) -> str:
    value = json.loads(raw, object_pairs_hook=_unique_fields)
    if (
        type(value) is not dict
        or set(value) != {"version", "stage", "failure_kind", "text_phase"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["stage"] != "producer"
        or type(value["failure_kind"]) is not str
        or value["failure_kind"] not in {"value", "type", "runtime", "timeout", "group", "other"}
        or type(value["text_phase"]) is not str
        or value["text_phase"] not in {"before_text", "after_text"}
    ):
        raise ValueError("invalid foreground diagnostic")
    return f"foreground_producer_{value['failure_kind']}_{value['text_phase']}"


_HOST_STAGES = {"configuration", "host_construction", "host_start", "serving", "host_close"}
_FAILURE_KINDS = {"ValueError", "RuntimeError", "OSError", "TimeoutError", "TypeError",
                  "ImportError", "PermissionError", "other"}
_MCP_CATEGORIES = {
    "cleanup_unknown", "client_binding", "closed", "configuration_invalid",
    "interpreter_binding", "launch_attempted", "load_refused", "load_unknown",
    "profile_binding", "same_thread", "source_binding", "status_unknown",
    "transport_unavailable", "unsupported_force", "unsupported_platform",
}
_CREDENTIAL_MARKERS = {
    b"[native-credential-refused] ":
        ("native_credential_", {"binding_refused", "runtime_refused"}),
    b"[presentation-credential-refused] ":
        ("presentation_credential_", {"credential", "cleanup_unknown"}),
}


def _startup_failure_category(raw: bytes, host: bool) -> str:
    value = json.loads(raw, object_pairs_hook=_unique_fields)
    fields = {"version", "stage", "failure_kind"} if host else {"version", "category"}
    if (type(value) is not dict or set(value) != fields
            or type(value["version"]) is not int or value["version"] != 1):
        raise ValueError("invalid startup diagnostic")
    if host:
        if (type(value["stage"]) is not str or value["stage"] not in _HOST_STAGES
                or type(value["failure_kind"]) is not str
                or value["failure_kind"] not in _FAILURE_KINDS):
            raise ValueError("invalid host diagnostic")
        return f"{value['stage']}_{value['failure_kind']}"
    if type(value["category"]) is not str or value["category"] not in _MCP_CATEGORIES:
        raise ValueError("invalid managed speech diagnostic")
    return "kokoro_mcp_" + value["category"]


class CappedCapture:
    def __init__(self, path: Any, limit: Any = 1048576) -> None:
        self.stream = Path(path).open("xb")  # noqa: SIM115 -- retained through owned settlement.
        self.limit, self.stored, self.seen = limit, 0, 0
        self.overflow = self.read_failed = self.line_overflow = False
        self._pending = bytearray()
        self._dropping = False
        self.categories: dict[str, Any] = {}
        self.first_bytes_ns: int | None = None
        self.last_bytes_ns: int | None = None

    def feed(self, chunk: Any) -> Any:
        now = time.monotonic_ns()
        self.first_bytes_ns = self.first_bytes_ns or now
        self.last_bytes_ns = now
        self.seen = min(self.seen + len(chunk), self.limit + 1)
        kept = chunk[: max(0, self.limit - self.stored)]
        self.overflow |= len(kept) != len(chunk)
        try:
            self.stream.write(kept)
            self.stored += len(kept)
        except OSError:
            self.read_failed = True
        # Scalar observations remain useful after the raw-storage cap. Line
        # buffering and category cardinality stay bounded independently.
        for byte in chunk:
            if byte == 10:
                if not self._dropping:
                    self._line(bytes(self._pending), now)
                self._pending.clear()
                self._dropping = False
            elif not self._dropping:
                if len(self._pending) < 8192:
                    self._pending.append(byte)
                else:
                    self.line_overflow = self._dropping = True
                    self._pending.clear()

    def _line(self, raw: Any, now: Any) -> Any:
        for prefix, (label, allowed) in _CREDENTIAL_MARKERS.items():
            if raw.startswith(prefix):
                name = "credential_failure_invalid"
                try:
                    value = json.loads(raw[len(prefix):], object_pairs_hook=_unique_fields)
                    if (type(value) is dict and set(value) == {"version", "category"}
                            and type(value["version"]) is int and value["version"] == 1
                            and type(value["category"]) is str and value["category"] in allowed):
                        name = label + value["category"]
                except (ValueError, TypeError, RecursionError):
                    pass
                item = self.categories.setdefault(
                    name, {"count": 0, "first_ns": now, "last_ns": now},
                )
                item["count"] = min(1000000, item["count"] + 1)
                item["last_ns"] = now
                return
        for prefix, host, invalid in (
            (b"[voice-preview-host-failure] ", True, "host_failure_invalid"),
            (b"[kokoro-mcp-refused] ", False, "kokoro_mcp_failure_invalid"),
        ):
            if raw.startswith(prefix):
                try:
                    name = _startup_failure_category(raw[len(prefix):], host)
                except (ValueError, TypeError, RecursionError):
                    name = invalid
                item = self.categories.setdefault(
                    name, {"count": 0, "first_ns": now, "last_ns": now},
                )
                item["count"] = min(1000000, item["count"] + 1)
                item["last_ns"] = now
                return
        prefix = b"[foreground-failure] "
        if raw.startswith(prefix):
            try:
                name = _foreground_failure_category(raw[len(prefix):])
            except (ValueError, TypeError, RecursionError):
                name = "foreground_failure_invalid"
            item = self.categories.setdefault(name, {"count": 0, "first_ns": now, "last_ns": now})
            item["count"] = min(1000000, item["count"] + 1)
            item["last_ns"] = now
            return
        for name, pattern in (
            ("listening", rb"listening on"),
            ("load_start", rb"loading model|starting llama server|starting runner|load request"),
            ("load_ready", rb"runner started in|model loaded|load model complete"),
            ("out_of_memory", rb"out of memory|memory allocation failed"),
            ("load_failed", rb"failed to load model|error loading model"),
            ("http_chat_completed", rb"\[GIN\].*POST.*(?:/v1/chat/completions|/api/chat)"),
            ("cancelled", rb"context cancel|request cancel"),
            ("shutdown", rb"shutting down|server closed"),
        ):
            if re.search(pattern, raw, re.I):
                item = self.categories.setdefault(
                    name, {"count": 0, "first_ns": now, "last_ns": now}
                )
                item["count"] = min(1000000, item["count"] + 1)
                item["last_ns"] = now

    def finish(self) -> Any:
        if self._pending and not self._dropping:
            self._line(bytes(self._pending), self.last_bytes_ns or time.monotonic_ns())
        self._pending.clear()
        try:
            self.stream.close()
        except OSError:
            self.read_failed = True
        return {
            "limit_bytes": self.limit,
            "stored_bytes": self.stored,
            "seen_capped": self.seen,
            "overflow": self.overflow,
            "read_failed": self.read_failed,
            "line_overflow": self.line_overflow,
            "categories": self.categories,
            "first_bytes_ns": self.first_bytes_ns,
            "last_bytes_ns": self.last_bytes_ns,
            "timestamps_are_drain_observations": True,
        }


class CappedProcesses:
    """Delegate ownership to Processes; each owned child gets its own bounded pipe."""

    def __init__(self, owner: Any, directory: Any) -> None:
        self.owner = owner
        self.directory = Path(directory)
        self.captures: dict[str, Any] = {}
        self.roots = owner.roots
        self._drainers: list[Any] = []
        self._pipes: list[Any] = []

    def __getattr__(self, name: Any) -> Any:
        return getattr(self.owner, name)

    def spawn(self, name: Any, argv: Any, *, env: Any, cwd: Any = None, log: Any) -> Any:
        witness = CappedCapture(self.directory / (name + ".private.log"))
        self.captures[name] = witness
        if name in self.owner.roots:
            raise RuntimeError("owned launch already attempted")
        process = subprocess.Popen(
            argv,
            env=dict(env),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self.owner.roots[name] = process
        self._pipes.append(process)
        self.owner._record()

        def drain() -> Any:
            try:
                while True:
                    pipe: Any = process.stdout
                    chunk = pipe.read1(4096)
                    if not chunk:
                        break
                    witness.feed(chunk)
            except BaseException:
                witness.read_failed = True

        thread = threading.Thread(target=drain, daemon=True, name="lookup-output-drain")
        self._drainers.append(thread)
        thread.start()
        return process

    def stop_all(self, containment: Any = None) -> Any:
        try:
            self.owner.stop_all(containment)
        finally:
            for thread in self._drainers:
                thread.join(timeout=5)
                if thread.is_alive():
                    for capture in self.captures.values():
                        capture.read_failed = True
                    raise RuntimeError("numeric log drain did not settle")
            for process in self._pipes:
                process.stdout.close()

    def finish(self) -> Any:
        facts = {}
        for name, capture in self.captures.items():
            item = capture.finish()
            process = self.roots.get(name)
            try:
                code = process.poll() if process is not None else None
            except Exception:
                code = None  # Diagnostic uncertainty cannot change settlement authority.
            item["exit_code"] = (code if type(code) is int and -(2**31) <= code <= 2**32 - 1
                                 else None)
            facts[name] = item
        return facts
