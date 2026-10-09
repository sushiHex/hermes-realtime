"""Rehearse the desktop MVP diagnostic session end to end, unattended (#159).

    uv run --frozen --extra local --extra browser-acceptance \\
      python scripts/rehearse_desktop_mvp.py --ollama-model <name from ollama list>

It composes the stack the operator runs in ``docs/desktop-mvp-diagnostic.md`` and walks that
runbook's session steps in order:

- a real ``hermes gateway run`` from a throwaway home laid out the way the upstream installer
  lays it out: the pinned checkout at ``home/hermes-agent``, its locked environment at
  ``home/hermes-agent/venv``, this checkout's built wheel installed there and the plugin enabled
  with the runbook's own ``hermes`` commands. Hermes's model is a stand-in served by this process;
- the companion inside that gateway;
- the full host (``hermes-realtime-host --hermes-env-file ...``) with the runbook's flags;
- the shared pinned LiveKit server (``scripts/local_livekit.py``);
- a headless system Chrome. Its microphone is a synthetic track: spoken steps play Kokoro-
  synthesized clips into it, so speech crosses WebRTC, LiveKit, VAD and Moonshine for real.

Each step prints one ``[desktop-mvp-rehearsal]`` line in the record sheet's categories:
outcome, timings, the bounded markers that appeared, and notes. Like the runbook, a failed step
is a finding and the rehearsal continues where it can. Records carry categories, counts and
timings only: never transcripts, audio, objectives, model output, keys, URLs or paths.

Every long-lived process is started detached, with its log and PID kept in the run directory,
and stopped at the end even on failure; the final record counts what is left. The run directory
is left in place under the system temporary directory (``--run-dir`` chooses another). When a
required local piece is missing, it prints one ``not_run`` preflight record and exits 0.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import ctypes
import importlib
import json
import os
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

import local_livekit
from real_gate_support import (
    HERMES_BASELINE,
    PINNED_HERMES,
    available_port,
    provision_pinned_hermes,
)

_PREFIX = "[desktop-mvp-rehearsal] "
_REPOSITORY = Path(__file__).resolve().parents[1]
_UPSTREAM = "https://github.com/NousResearch/hermes-agent.git"
_OLLAMA = "http://127.0.0.1:11434"
_DETACHED = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
# A bounded marker other code printed: `[name] {json}`.
_MARKER_LINE = re.compile(r"\[[a-z][a-z0-9-]{0,47}\] \{.*\}")
# The markers this session's components print (the runbook's list, plus memory readback and
# Hermes's identity check); any other name is counted, never recorded.
_KNOWN_MARKERS = frozenset(
    {
        "voice-tail", "voice-tail-lock", "voice-tail-outbox", "voice-archive-send",
        "voice-review-send", "voice-review-close", "hermes-run-record-lock",
        "hermes-restart-settlement", "hermes-dispatch-recovery", "codex-session-auth",
        "codex-tool-refusal", "voice-companion", "voice-archive-open", "voice-archive",
        "voice-archive-lease", "voice-review", "hermes-bridge-hello", "hermes-bridge-welcome",
        "voice-memory", "voice-memory-receive", "voice-memory-stream", "voice-forget",
        "voice-forget-send",
        "hermes-identity", "real-hermes-gate", "consent-activation", "qualification-checkpoint",
        "speech-stop", "ollama-prompt", "rehearsal-prompt", "rehearsal-recall",
    }
)  # fmt: skip
_MARKER_CATEGORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
_EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,79}")
# Tracebacks known to be upstream noise; any other one is a finding.
_KNOWN_TRACEBACKS = {
    # LiveKit's Python SDK disposes FFI handles after its server at interpreter exit.
    "livekit_ffi_handle_dispose": (("FfiHandle", "_ffi_client.py"), "AssertionError"),
    # Hermes's gateway watchdog uses a Unix socket API that Windows lacks.
    "hermes_watchdog_unix_server": (
        ("shutdown_watchdog.py", "start_unix_server"),
        "AttributeError",
    ),
}
# Where the conversation's phrase must be found before deletion.
_SEEDED_SOURCES = ("voice_tail", "hermes_database")
# One 50 ms window louder than this is sound (about -47 dBFS).
_FRAME_ENERGY = 1e-6
# The readiness cue is one short spoken chunk.
_CUE_MAX_SECONDS = 2.0
# How far from the page's voice-input confirmation the cue may start or end.
_CUE_WINDOW_MS = 3000.0
# A cue delivered as a reconnect starts can stall once mid-word (about 350 ms observed).
_CUE_GAP_MS = 500.0
_MAX_MARKER_CHARS = 512
_MAX_MARKERS = 32
_NOTICE = re.compile(
    r"NOTICE: a previous session left Hermes background work behind: (\d+) run\(s\) were "
    r"stopped or had already ended, and (\d+) dispatch\(es\) have an unknown outcome"
)
_HOST_STOP_FILE = "stop-host"
_LAUNCH_LINE = "Open this stable loopback URL in a local browser:"
# The page's named latency markers, as `name: 12.3 ms`; names are a closed set.
_LATENCY = re.compile(r"([a-z][a-z0-9_]{0,63}): (\d+(?:\.\d+)?) ms")
_MARKER_NAME = re.compile(r"[A-Za-z0-9_]{1,64}: (?:server monotonic )?\d+(?:\.\d+)? ms\Z")
# The page's fixed connection labels and session-toggle labels.
_LABELS = frozenset(
    {
        "Preparing microphone…",
        "Connecting speech path…",
        "Microphone not reaching server",
        "Listening",
        "Ready to type — microphone unavailable",
        "Idle",
        "Bootstrapping",
        "Connecting",
        "Connected",
        "Reconnecting",
        "Stopping",
        "Stopped",
        "Error",
    }
)
_TOGGLES = frozenset(
    {"Connect", "Connecting…", "Stopping…", "Stop session", "Fresh launch required"}
)
_RESTARTED = "I restarted, so background work from before will not resume."
# The context checks' question; the host child recognizes its prompt by it.
_BIRD_QUESTION = "What is my favorite bird? Answer with one word."
# How long a context check waits for the host child's recall trials.
_RECALL_SECONDS = 300.0
_PRIVATE_ID = re.compile(r"\b(?:run|deleg)_[A-Za-z0-9]")
_DELETE_STATES = {
    "Starting deletion…": "starting",
    "Deletion pending. Hermes is still verifying the archive.": "pending",
    "Voice conversation deleted.": "complete",
    "Deletion could not be confirmed. Checking status.": "unconfirmed",
    "Deletion status unavailable. Checking again.": "status_unavailable",
    "Ready to delete this voice conversation.": "idle",
    "Voice conversation deletion is unavailable on this host.": "unavailable",
    "Connect to delete this voice conversation.": "disconnected",
}
_TERMINAL_TASK = frozenset({"completed", "failed", "interrupted", "rejected"})
_ANSWERED_PATHS = frozenset({"/api/v1/input", "/api/v1/approval"})
_REPLY_SECONDS = 180.0
_APPROVAL_CARD_SECONDS = 120.0
# An abandoned browser session holds the persistent front door until the host reaps it.
_RECONNECT_SECONDS = 420.0
# Decoded remote audio energy above which a reply was audible.
_AUDIBLE_ENERGY = 1e-4
_VIEWPORT = {"width": 1366, "height": 768}
# How long a page is watched after Connect for audio no one asked for.
_QUIET_SECONDS = 7.0
# How long a stop is given to reach the stand-in model's stream.
_SETTLE_SECONDS = 15.0
# The stand-in model's answer to a dispatched task, as Hermes returns it.
_STAND_IN_RESULT = "Rehearsal step complete."
# Rare enough that no other turn in the session, or a guess, says one.
_BIRDS = ("hoopoe", "cassowary", "quetzal", "lyrebird", "kingfisher")
# The operator's spoken lines, synthesized once per run.
_CLIPS = {
    "question": "What is the capital of France?",
    "story": "Please tell me a long story about a lighthouse keeper, in about ten sentences.",
    "interruption": "Wait, stop. Let me ask something else.",
}


def _emit(record: Mapping[str, object]) -> None:
    print(_PREFIX + json.dumps(record, separators=(",", ":"), sort_keys=True), flush=True)


# --- The stand-in model -------------------------------------------------------------------

_CHMOD = re.compile(r"chmod 777 (\S+)")
# What Hermes writes when the user refused a command it asked to approve: "Command denied by
# user" on the API approval path the host uses, "User denied" on the interactive ones.
_DENIED = re.compile(r"BLOCKED: (?:Command denied by user|User denied)")
_TIMED = re.compile(r"\babout (\d{1,3}) seconds\b")
_LABEL = re.compile(r"\brehearsal ([a-z]+) task\b")
_ENDLESS = ("until you are stopped", "think carefully for several minutes")


def _text(message: object) -> str:
    content = message.get("content") if type(message) is dict else None
    if type(content) is str:
        return content
    if type(content) is list:
        return " ".join(
            part["text"]
            for part in content
            if type(part) is dict and type(part.get("text")) is str
        )
    return ""


def route(body: object) -> tuple[str, str]:
    """What the stand-in does for one chat completion request, from its last user turn.

    Only a run with the terminal tool is dispatched work; anything else (a review, a summary)
    gets one short sentence, so it can never keep a review or a deletion waiting. A tool
    result after the last user turn ends the run.
    """

    messages = body.get("messages") if type(body) is dict else None
    if type(messages) is not list or not messages:
        return ("text", "")
    tools = body.get("tools") if type(body) is dict else None
    names = {
        tool["function"].get("name")
        for tool in (tools if type(tools) is list else [])
        if type(tool) is dict and type(tool.get("function")) is dict
    }
    users = [index for index, message in enumerate(messages) if _role(message) == "user"]
    if "terminal" not in names or not users:
        return ("text", "")
    if any(_role(message) == "tool" for message in messages[users[-1] + 1 :]):
        return ("text", "")
    objective = _text(messages[users[-1]])
    chmod = _CHMOD.search(objective)
    if chmod is not None:
        return ("tool", f"chmod 777 {chmod.group(1).rstrip('.')}")
    label = _LABEL.search(objective)
    if any(phrase in objective for phrase in _ENDLESS):
        return ("endless", label.group(1) if label is not None else "other")
    timed = _TIMED.search(objective)
    if timed is not None:
        return ("timed", timed.group(1))
    return ("text", "")


def tool_outcome(body: object) -> str | None:
    """What Hermes reported for the tool call this request answers, as a category.

    Hermes itself writes the result of a command it was asked to approve: "denied" when the
    user refused it, "blocked" when anything else stopped it, "ran" otherwise. None when the
    request answers no tool call.
    """

    messages = body.get("messages") if type(body) is dict else None
    if type(messages) is not list:
        return None
    users = [index for index, message in enumerate(messages) if _role(message) == "user"]
    results = [
        _text(message)
        for message in messages[(users[-1] + 1 if users else 0) :]
        if _role(message) == "tool"
    ]
    if not results:
        return None
    if _DENIED.search(results[-1]) is not None:
        return "denied"
    return "blocked" if "BLOCKED" in results[-1] else "ran"


def _role(message: object) -> object:
    return message.get("role") if type(message) is dict else None


class StandInModel:
    """An OpenAI-compatible model: instant sentences, timed and endless work, one tool call.

    It counts what it was asked for by kind, and the endless streams still open by label,
    so a step can witness that a crash left work running and that a restart stopped it.
    """

    def __init__(self) -> None:
        self.requests: Counter[str] = Counter()
        self.open: Counter[str] = Counter()
        self.started: Counter[str] = Counter()
        self.tool_outcomes: Counter[str] = Counter()
        self._runner: Any = None

    async def start(self) -> str:
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._complete)
        app.router.add_get("/v1/models", self._models)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = available_port()
        await web.TCPSite(self._runner, "127.0.0.1", port).start()
        return f"http://127.0.0.1:{port}/v1"

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _models(self, request: Any) -> Any:
        from aiohttp import web

        del request
        model = {"id": "stand-in", "object": "model", "context_length": 131072}
        return web.json_response({"object": "list", "data": [model]})

    async def _complete(self, request: Any) -> Any:
        from aiohttp import web

        try:
            body = json.loads(await request.text())
        except ValueError:
            return web.Response(status=400)
        kind, argument = route(body)
        self.requests[kind] += 1
        outcome = tool_outcome(body)
        if outcome is not None:
            self.tool_outcomes[outcome] += 1
        stream = type(body) is dict and body.get("stream") is True
        if kind == "tool":
            return await self._answer(request, stream, tool=argument)
        if kind == "endless":
            return await self._endless(request, argument)
        if kind == "timed":
            await asyncio.sleep(int(argument))
            return await self._answer(request, stream, text="The long rehearsal task finished.")
        return await self._answer(request, stream, text=_STAND_IN_RESULT)

    async def _answer(
        self, request: Any, stream: bool, *, text: str | None = None, tool: str | None = None
    ) -> Any:
        from aiohttp import web

        calls = (
            [
                {
                    "id": "call_rehearsal",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": json.dumps({"command": tool})},
                }
            ]
            if tool is not None
            else None
        )
        finish = "tool_calls" if calls is not None else "stop"
        if not stream:
            message: dict[str, object] = {"role": "assistant", "content": text}
            if calls is not None:
                message["tool_calls"] = calls
            return web.json_response(
                {
                    "id": "stand-in",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "stand-in",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        delta: dict[str, object] = {"role": "assistant"}
        if calls is not None:
            delta["tool_calls"] = [call | {"index": 0} for call in calls]
        else:
            delta["content"] = text
        with contextlib.suppress(ConnectionResetError):
            await response.write(_chunk(delta, None))
            await response.write(_chunk({}, finish))
            await response.write(b"data: [DONE]\n\n")
        return response

    async def _endless(self, request: Any, label: str) -> Any:
        """Stream forever; the caller hanging up is the only way it ends."""

        from aiohttp import web

        self.started[label] += 1
        self.open[label] += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            await response.prepare(request)
            while True:
                await response.write(_chunk({"content": "."}, None))
                await asyncio.sleep(0.5)
        except ConnectionResetError:
            return response
        finally:
            self.open[label] -= 1


def _chunk(delta: Mapping[str, object], finish: str | None) -> bytes:
    payload = {
        "id": "stand-in",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "stand-in",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


# --- Detached processes -------------------------------------------------------------------


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def process_table() -> dict[int, tuple[int, str]]:
    """Every process now: pid to (parent pid, image name)."""

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        raise OSError("process snapshot failed")
    table: dict[int, tuple[int, str]] = {}
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(_ProcessEntry)
        more = kernel.Process32FirstW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
        while more:
            table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile)
            more = kernel.Process32NextW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
    finally:
        kernel.CloseHandle(ctypes.c_void_p(snapshot))
    return table


def started_at(pid: int) -> int | None:
    """When a running process started (a PID-reuse guard), or None if it is not running."""

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        code = ctypes.c_uint32()
        times = [ctypes.c_uint64() for _ in range(4)]
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
            return None
        if code.value != 259:  # STILL_ACTIVE: an exited process can still have handles.
            return None
        if not kernel.GetProcessTimes(
            ctypes.c_void_p(handle), *(ctypes.byref(time) for time in times)
        ):
            return None
        return int(times[0].value)
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class Containment:
    """A kill-on-close Job Object holding this process and everything it starts.

    Every child inherits the job, so even a hard-killed rehearsal leaves no orphan: the
    system closes the job with the last handle and terminates what is in it. The job is
    also the authority on what is still running.
    """

    def __init__(self) -> None:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = ctypes.c_void_p
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        self._kernel = kernel
        self._job = kernel.CreateJobObjectW(None, None)
        if not self._job:
            raise OSError("the job object could not be created")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            ctypes.c_void_p(self._job), 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ) or not kernel.AssignProcessToJobObject(
            ctypes.c_void_p(self._job), ctypes.c_void_p(kernel.GetCurrentProcess())
        ):
            raise OSError("this process could not join its job object")

    def members(self) -> set[int]:
        """Every process in the job, this one included."""

        size = 4096
        buffer = (ctypes.c_uint8 * (8 + 8 * size))()
        if not self._kernel.QueryInformationJobObject(
            ctypes.c_void_p(self._job), 3, buffer, ctypes.sizeof(buffer), None
        ):
            raise OSError("the job object could not be read")
        count = int.from_bytes(bytes(buffer[4:8]), "little")
        ids = (ctypes.c_size_t * count).from_buffer(buffer, 8)
        return {int(pid) for pid in ids}


def descendants(table: Mapping[int, tuple[int, str]], roots: set[int]) -> set[int]:
    """The roots that are alive and every process below them."""

    found = {pid for pid in roots if pid in table}
    frontier = set(found)
    while frontier:
        frontier = {pid for pid, (parent, _) in table.items() if parent in frontier} - found
        found |= frontier
    return found


class Processes:
    """Owns every long-lived process: detached, logged, recorded, and stopped at the end."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.roots: dict[str, subprocess.Popen[bytes]] = {}
        # Every process ever seen in an owned tree, by pid: its image and start time, so a
        # survivor whose root already exited is still found, and a reused pid is not.
        self.seen: dict[int, tuple[str, int]] = {}
        self._logs: list[TextIO | Any] = []

    def spawn(
        self,
        name: str,
        argv: list[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        log: Path,
    ) -> subprocess.Popen[bytes]:
        if name in self.roots and self.roots[name].poll() is None:
            raise RuntimeError(f"{name} is already running")
        stream = log.open("ab")
        self._logs.append(stream)
        process = subprocess.Popen(
            argv,
            env=dict(env),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=_DETACHED,
        )
        self.roots[name] = process
        self._record()
        return process

    def observe(self) -> None:
        table = process_table()
        roots = {process.pid for process in self.roots.values()}
        roots |= {pid for pid, _ in self.alive()}
        for pid in descendants(table, roots):
            started = started_at(pid)
            if started is not None and pid not in self.seen:
                self.seen[pid] = (table[pid][1], started)

    def alive(self) -> list[tuple[int, str]]:
        """Every process seen in an owned tree that is still the same running process."""

        return [
            (pid, image)
            for pid, (image, started) in sorted(self.seen.items())
            if started_at(pid) == started
        ]

    def _record(self) -> None:
        self.observe()
        pids = {name: process.pid for name, process in self.roots.items()}
        (self._directory / "pids.json").write_text(json.dumps(pids), encoding="utf-8")

    def kill(self, name: str) -> float:
        """End a tree the way a crash does; seconds until its root exited."""

        process = self.roots.get(name)
        if process is None:
            return 0.0
        self.observe()
        started = time.monotonic()
        if process.poll() is None:
            _kill_tree(process.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
        return time.monotonic() - started

    def wait(self, name: str, timeout: float) -> int | None:
        """The root's exit code once it exits, or None if it is still running."""

        process = self.roots.get(name)
        if process is None:
            return None
        self.observe()
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop_all(self) -> None:
        """Kill every owned tree, then any process from one that outlived its root."""

        for name in reversed(list(self.roots)):
            self.kill(name)
        for pid, _ in self.alive():
            subprocess.run(
                ("taskkill", "/PID", str(pid), "/F"),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                creationflags=0x08000000,
            )
        for stream in self._logs:
            with contextlib.suppress(OSError):
                stream.close()

    def survivors(self, containment: Containment | None) -> dict[str, int]:
        """What is left running, by image name: every process still in the job, and any
        process seen in an owned tree, so nothing that escaped either view is missed."""

        table = process_table()
        left = {pid for pid, _ in self.alive()}
        if containment is not None:
            left |= containment.members() - {os.getpid()}
        return dict(Counter(table[pid][1] for pid in left if pid in table))


def listeners(port: int) -> list[str]:
    """Local addresses listening on a TCP port."""

    output = subprocess.run(
        ("netstat", "-ano", "-p", "tcp"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        creationflags=0x08000000,
    ).stdout + subprocess.run(
        ("netstat", "-ano", "-p", "tcpv6"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        creationflags=0x08000000,
    ).stdout
    found: list[str] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0] == "TCP" and fields[3] == "LISTENING":
            address, _, local_port = fields[1].rpartition(":")
            if local_port == str(port):
                found.append(address)
    return found


def _loopback_only(addresses: list[str]) -> bool:
    return bool(addresses) and all(address in {"127.0.0.1", "[::1]"} for address in addresses)


def listener_verdict(name: str, addresses: list[str]) -> list[Finding]:
    """A listener is part of the security boundary: anything but loopback fails the step."""

    return [] if _loopback_only(addresses) else [("fail", f"{name}_not_loopback")]


def _run(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run one command to completion; on a timeout, end its whole tree, not just its root."""

    process = subprocess.Popen(
        argv,
        env=None if env is None else dict(env),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=0x08000000,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process.pid)
        process.communicate()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _kill_tree(pid: int) -> None:
    subprocess.run(
        ("taskkill", "/PID", str(pid), "/T", "/F"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        creationflags=0x08000000,
    )


# --- Logs and records ---------------------------------------------------------------------


class LogTail:
    """Complete lines appended to one log since it was opened."""

    def __init__(self, source: str, path: Path) -> None:
        self.source = source
        self._path = path
        self._offset = 0
        self._partial = b""
        self.lines: list[str] = []

    def poll(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("rb") as stream:
            stream.seek(self._offset)
            data = stream.read()
        self._offset += len(data)
        *complete, self._partial = (self._partial + data).split(b"\n")
        self.lines.extend(line.decode("utf-8", "replace").rstrip("\r") for line in complete)


def markers(lines: list[str]) -> tuple[list[str], int]:
    """The known markers among ``lines``, re-rendered from their values, and the rest counted.

    A marker is kept by name, from the components this session runs, and only its scalar
    values are rendered: a category, a count or a flag. Any other line, or a value that could
    carry text, is never recorded, so an upstream line holding a path cannot leak.
    """

    found: list[str] = []
    dropped = 0
    for line in lines:
        if _MARKER_LINE.fullmatch(line) is None:
            continue
        if len(line) > _MAX_MARKER_CHARS:
            dropped += 1
            continue
        name = line[1 : line.index("]")]
        try:
            value = json.loads(line.split("] ", 1)[1])
        except ValueError:
            value = None
        if name not in _KNOWN_MARKERS or type(value) is not dict or not _is_category(value):
            dropped += 1
            continue
        found.append(f"[{name}] " + json.dumps(value, separators=(",", ":"), sort_keys=True))
    return found[:_MAX_MARKERS], dropped + max(len(found) - _MAX_MARKERS, 0)


def _is_category(value: object, depth: int = 0) -> bool:
    """A count, a flag, a short category, or a list or object of those, at most 3 deep."""

    if value is None or type(value) in (bool, int):
        return True
    if type(value) is str:
        return _MARKER_CATEGORY.fullmatch(value) is not None
    if type(value) is list and depth < 3:
        return len(value) <= 16 and all(_is_category(item, depth + 1) for item in value)
    if type(value) is dict and depth < 3:
        return all(
            type(key) is str
            and _MARKER_CATEGORY.fullmatch(key) is not None
            and _is_category(field, depth + 1)
            for key, field in value.items()
        )
    return False


def tracebacks(lines: list[str]) -> tuple[Counter[str], Counter[str]]:
    """Tracebacks in a log: (known, by allowlist entry; unexpected, by exception type).

    Only the exception's type name leaves the log: never its message or its frames.
    """

    known: Counter[str] = Counter()
    unexpected: Counter[str] = Counter()
    index = 0
    while index < len(lines):
        line = lines[index]
        if "Traceback (most recent call last):" not in line:
            index += 1
            continue
        start = max(index - 1, 0)  # "Exception ignored in: ..." names the owner.
        index += 1
        while index < len(lines) and lines[index][:1] in (" ", "|", "+", "\t"):
            index += 1
        final = lines[index] if index < len(lines) else ""
        block = "\n".join(lines[start : index + 1])
        exception = final.split(":", 1)[0].strip()
        if _EXCEPTION_NAME.fullmatch(exception) is None:
            exception = "unparsed"
        for name, (needles, expected) in _KNOWN_TRACEBACKS.items():
            if exception == expected and all(needle in block for needle in needles):
                known[name] += 1
                break
        else:
            unexpected[exception] += 1
        index += 1
    return known, unexpected


Finding = tuple[str, str]  # ("differ" | "fail", category)


def transcript_matches(expected: str, heard: str) -> bool:
    """Every word of four letters or more in the spoken line is in the final transcript."""

    words = set(re.findall(r"[a-z]+", heard.casefold()))
    return all(word in words for word in re.findall(r"[a-z]{4,}", expected.casefold()))


def deletion_verdict(
    before: Mapping[str, int],
    after: Mapping[str, int],
    follow_up: Mapping[str, int],
    next_reply_has_phrase: bool,
) -> list[Finding]:
    """The phrase must have been everywhere it is looked for, and then nowhere."""

    findings: list[Finding] = []
    if any(before.get(source, 0) <= 0 for source in _SEEDED_SOURCES):
        findings.append(("fail", "phrase_not_seeded"))
    findings += retained_verdict(after, "phrase_retained")
    findings += retained_verdict(follow_up, "phrase_returned")
    if next_reply_has_phrase:
        findings.append(("fail", "phrase_in_next_reply"))
    return findings


def retained_verdict(scan: Mapping[str, int], category: str) -> list[Finding]:
    """A deleted phrase found anywhere fails; in built-in memory it is the documented limit."""

    retained = {key for key, count in scan.items() if count}
    if retained - {"memories"}:
        return [("fail", category)]
    if retained:
        # Built-in memory is not erased by design: there is no unlearning in the MVP.
        return [("differ", "memory_retains_phrase")]
    return []


def tail_rows(raw: bytes) -> list[tuple[str, str]] | None:
    """The (role, text) rows of a voice tail document, or None when it is not one."""

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    rows = document.get("messages") if type(document) is dict else None
    if type(rows) is not list:
        return None
    parsed: list[tuple[str, str]] = []
    for row in rows:
        if type(row) is not dict or type(row.get("role")) is not str:
            return None
        if type(row.get("text")) is not str:
            return None
        parsed.append((row["role"], row["text"]))
    return parsed


def stated_by_user(rows: list[tuple[str, str]], phrase: str) -> bool:
    """Is the fact in a row the user said? The model's own echo of it is not the fact."""

    folded = phrase.casefold()
    return any(role == "user" and folded in text.casefold() for role, text in rows)


def restored_fact(loaded: list[tuple[str, str]] | None, restored: object, phrase: str) -> bool:
    """Did a restarted host load the user's statement of the fact?

    The host restores a tail whole or not at all, so its restore count must equal the rows
    of the file it read, and the user's statement must be among them.
    """

    return (
        loaded is not None
        and type(restored) is int
        and restored == len(loaded)
        and stated_by_user(loaded, phrase)
    )


def context_verdict(in_kept_context: bool, in_prompt: bool | None) -> list[Finding]:
    """Was the earlier fact kept, and did the question's prompt carry it?

    Losing it is a host finding. Whether the model then uses a fact its prompt carries is
    model quality: the answer and the measured recall rate are notes, never a finding.
    """

    if not in_kept_context:
        return [("differ", "context_lost")]
    if in_prompt is None:
        return [("differ", "no_prompt_observation")]
    return [] if in_prompt else [("differ", "context_not_in_prompt")]


def prompt_observation(messages: list[dict[str, str]], phrase: str) -> dict[str, int]:
    """Counts only, at the adapter boundary: how many user rows of the prompt state the fact."""

    folded = phrase.casefold()
    return {
        "chars": sum(len(message["content"]) for message in messages),
        "fact_rows": sum(
            message["role"] == "user" and folded in message["content"].casefold()
            for message in messages
        ),
        "messages": len(messages),
        "rows": sum(message["role"] in ("user", "assistant") for message in messages),
        "version": 1,
    }


def recall_trials(
    messages: list[dict[str, str]],
    phrase: str,
    trials: int,
    send: Callable[[list[dict[str, str]]], str],
) -> dict[str, int]:
    """Send one exact prompt ``trials`` times; count the replies that name the fact."""

    hits = errors = 0
    for _ in range(trials):
        try:
            hits += phrase.casefold() in send(messages).casefold()
        except (OSError, ValueError, KeyError, TypeError):
            errors += 1
    return {"errors": errors, "hits": hits, "trials": trials, "version": 1}


def restart_work_verdict(
    before_crash: int, after_crash: int, after_settlement: int
) -> list[Finding]:
    """The crash must leave real work running, and the restart must stop it."""

    findings: list[Finding] = []
    if before_crash < 1 or after_crash < 1:
        findings.append(("differ", "no_work_left_by_crash"))
    if after_settlement:
        findings.append(("fail", "work_left_running"))
    return findings


def cancel_work_verdict(open_after_cancel: int) -> list[Finding]:
    return [("fail", "work_left_running")] if open_after_cancel else []


def page_input_verdict(name: str, status: int | None, witnessed: bool) -> list[Finding]:
    """A typed turn or an approval decision, judged twice: by the server's status for it on
    the wire, and by what came of it (a reply, or Hermes's own report of the refusal)."""

    if status is None or not 200 <= status < 300:
        return [("fail", f"{name}_refused")]
    return [] if witnessed else [("fail", f"{name}_without_effect")]


def audio_without_speech(
    timeline: list[tuple[float, float]],
    cue_at: float | None,
    new_assistant_rows: int,
) -> tuple[list[Finding], dict[str, object]]:
    """Audio no one asked for, with only the host's identified readiness cue excused.

    ``timeline`` holds (page time, decoded energy) per 50 ms window. The cue is excused only
    when it is one burst of at most ``_CUE_MAX_SECONDS`` around the page's own
    voice-input confirmation; any other energy, or any assistant row, is stale.
    """

    loud = [(at, energy) for at, energy in timeline if energy > _FRAME_ENERGY]
    cue: list[tuple[float, float]] = []
    if cue_at is not None:
        # The host plays the cue when it confirms voice input, and the page learns of that
        # confirmation at its next event poll, so the cue may start before the page's marker.
        window = [
            (at, energy)
            for at, energy in loud
            if cue_at - _CUE_WINDOW_MS <= at <= cue_at + _CUE_WINDOW_MS
        ]
        gaps = [b[0] - a[0] for a, b in zip(window, window[1:], strict=False)]
        contiguous = all(gap <= _CUE_GAP_MS for gap in gaps)
        if window and contiguous and window[-1][0] - window[0][0] <= _CUE_MAX_SECONDS * 1000:
            cue = window
    unexcused = sum(energy for at, energy in loud if (at, energy) not in cue)
    notes: dict[str, object] = {
        "cue_seconds": round((cue[-1][0] - cue[0][0]) / 1000 + 0.05, 2) if cue else 0.0,
        "first_sound_ms": round(loud[0][0] - cue_at) if loud and cue_at is not None else None,
        "cue_max_gap_ms": round(
            max((b[0] - a[0] for a, b in zip(cue, cue[1:], strict=False)), default=0)
        ),
        "unexcused_energy": round(unexcused, 6),
        "new_assistant_rows": new_assistant_rows,
    }
    findings: list[Finding] = []
    if unexcused > _AUDIBLE_ENERGY or new_assistant_rows:
        findings.append(("differ", "stale_audio"))
    return findings, notes


def harness_matches_head(repository: Path, path: str) -> bool:
    """Is the running harness exactly the file HEAD tracks? An untracked copy is not."""

    tracked = _run(["git", "ls-files", "--error-unmatch", "--", path], cwd=repository)
    unchanged = _run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=repository)
    return tracked.returncode == 0 and unchanged.returncode == 0


def phrase_in_database(database: Path, phrase: str) -> int:
    """Rows holding ``phrase`` in any column of any table, or matching it in any FTS index.

    A missing database is an error, never a zero.
    """

    if not database.is_file():
        raise FileNotFoundError("the Hermes state database is missing")
    pattern = f"%{phrase.casefold()}%"
    total = 0
    with contextlib.closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        tables = db.execute(
            "SELECT name, coalesce(sql, '') FROM sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for table, sql in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            if re.search(r"USING\s+fts", sql, re.IGNORECASE):
                with contextlib.suppress(sqlite3.Error):
                    query = f"SELECT count(*) FROM {quoted} WHERE {quoted} MATCH ?"
                    total += db.execute(query, ('"' + phrase + '"',)).fetchone()[0]
            for column in db.execute(f"PRAGMA table_info({quoted})").fetchall():
                name = '"' + str(column[1]).replace('"', '""') + '"'
                with contextlib.suppress(sqlite3.Error):
                    total += db.execute(
                        f"SELECT count(*) FROM {quoted} WHERE lower(CAST({name} AS TEXT)) LIKE ?",
                        (pattern,),
                    ).fetchone()[0]
    return int(total)


def phrase_in_files(directory: Path, phrase: str) -> int:
    """Occurrences of ``phrase`` in every file under ``directory``; a missing one is an error."""

    if not directory.is_dir():
        raise FileNotFoundError("a Hermes directory to scan is missing")
    needle = phrase.casefold().encode()
    return sum(
        path.read_bytes().lower().count(needle) for path in directory.rglob("*") if path.is_file()
    )


def notice(lines: list[str]) -> dict[str, int] | None:
    """The counts of the host's restart NOTICE line, if it printed one."""

    for line in lines:
        match = _NOTICE.search(line)
        if match is not None:
            return {"stopped": int(match.group(1)), "unknown": int(match.group(2))}
    return None


def latencies(entries: list[str]) -> dict[str, float]:
    """The page's latest named latency values, from its diagnostics marker list."""

    values: dict[str, float] = {}
    for entry in entries:
        match = _LATENCY.fullmatch(entry)
        if match is not None:
            values[match.group(1)] = float(match.group(2))
    return values


class Step:
    """One record-sheet row: outcome, timings, markers and notes, emitted exactly once."""

    def __init__(self, number: str, name: str, logs: list[LogTail]) -> None:
        self.number = number
        self.name = name
        self.outcome = "as_expected"
        self.category: str | None = None
        self.timings: dict[str, float] = {}
        self.notes: dict[str, object] = {}
        self.findings: list[str] = []
        # Where the step is, so a failure names the wait that expired.
        self.stage = "start"
        self._logs = logs
        self._start = {id(log): len(log.lines) for log in logs}

    def differ(self, category: str) -> None:
        if self.outcome == "as_expected":
            self.outcome, self.category = "different", category
        self.findings.append(category)

    def fail(self, category: str) -> None:
        if self.outcome != "failed":
            self.outcome, self.category = "failed", category
        self.findings.append(category)

    def skip(self, category: str) -> None:
        self.outcome, self.category = "not_run", category

    def apply(self, findings: list[Finding]) -> None:
        for level, category in findings:
            (self.fail if level == "fail" else self.differ)(category)

    def lines(self, source: str | None = None) -> list[str]:
        found: list[str] = []
        for log in self._logs:
            log.poll()
            if source is None or log.source == source:
                found.extend(log.lines[self._start.get(id(log), 0) :])
        return found

    def time(self, name: str, started: float) -> None:
        self.timings[name] = round(time.monotonic() - started, 2)

    def record(self) -> dict[str, object]:
        found: list[str] = []
        dropped = 0
        known: Counter[str] = Counter()
        unexpected: Counter[str] = Counter()
        for log in self._logs:
            log.poll()
            new = log.lines[self._start.get(id(log), 0) :]
            lines, extra = markers(new)
            found.extend(f"{log.source}: {line}" for line in lines)
            dropped += extra
            allowed, other = tracebacks(new)
            known += allowed
            unexpected += Counter({f"{log.source}:{name}": n for name, n in other.items()})
        if unexpected and "unexpected_traceback" not in self.findings:
            self.differ("unexpected_traceback")
        record: dict[str, object] = {
            "markers": found[:_MAX_MARKERS],
            "name": self.name,
            "notes": self.notes,
            "outcome": self.outcome,
            "step": self.number,
            "timings": self.timings,
            "version": 1,
        }
        if self.category is not None:
            record["category"] = self.category
        if self.findings:
            record["findings"] = self.findings
        if dropped or len(found) > _MAX_MARKERS:
            record["markers_dropped"] = dropped + max(len(found) - _MAX_MARKERS, 0)
        if known or unexpected:
            record["tracebacks"] = {"known": dict(known), "unexpected": dict(unexpected)}
        return record


# --- The rehearsal ------------------------------------------------------------------------

_INIT_SCRIPT = r"""
(() => {
  let context = null;
  let destination = null;
  const ensure = async () => {
    if (context === null) {
      context = new AudioContext({ sampleRate: 48000 });
      destination = context.createMediaStreamDestination();
    }
    await context.resume();
    return destination;
  };
  // A synthetic track runs no audio processing, and with no speaker-to-microphone path it needs
  // none; it reports the AEC-only processing the page requires, which is declared, not run.
  const synthetic = new WeakSet();
  const settings = MediaStreamTrack.prototype.getSettings;
  MediaStreamTrack.prototype.getSettings = function () {
    const value = settings.call(this);
    return synthetic.has(this)
      ? { ...value, echoCancellation: true, autoGainControl: false, noiseSuppression: false,
          voiceIsolation: false }
      : value;
  };
  const clone = MediaStreamTrack.prototype.clone;
  MediaStreamTrack.prototype.clone = function () {
    const copy = clone.call(this);
    if (synthetic.has(this)) synthetic.add(copy);
    return copy;
  };
  const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (constraints && constraints.audio && !constraints.video) {
      const target = await ensure();
      const tracks = target.stream.getAudioTracks().map((track) => clone.call(track));
      for (const track of tracks) synthetic.add(track);
      return new MediaStream(tracks);
    }
    return original(constraints);
  };
  window.__rehearsalSpeak = async (encoded) => {
    const target = await ensure();
    const bytes = Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    const count = bytes.length / 2;
    const buffer = context.createBuffer(1, count, 48000);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < count; i += 1) samples[i] = view.getInt16(i * 2, true) / 32768;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(target);
    source.start();
    return count / 48000;
  };
  // The energy of the remote audio the page attached for playback, measured on its track: a
  // media element's clock advances while any stream is attached, silent or not.
  let meter = null;
  let analyser = null;
  let measured = null;
  let energy = 0;
  const timeline = [];
  const window_ = new Float32Array(1024);
  setInterval(() => {
    const stream = document.querySelector("#remote-audio")?.srcObject;
    const track = stream instanceof MediaStream ? stream.getAudioTracks()[0] ?? null : null;
    if (track !== measured) {
      measured = track;
      analyser = null;
      if (track !== null) {
        if (meter === null) meter = new AudioContext();
        analyser = meter.createAnalyser();
        analyser.fftSize = window_.length;
        meter.createMediaStreamSource(new MediaStream([track])).connect(analyser);
      }
    }
    if (analyser === null || track.readyState !== "live") return;
    analyser.getFloatTimeDomainData(window_);
    let sum = 0;
    for (const sample of window_) sum += sample * sample;
    const increment = (sum / window_.length) * 0.05;
    energy += increment;
    timeline.push([performance.now(), increment]);
    if (timeline.length > 4000) timeline.shift();
  }, 50);
  window.__rehearsalEnergy = async () => {
    if (meter !== null) await meter.resume();
    return energy;
  };
  window.__rehearsalTimeline = (since) => timeline.filter(([at]) => at >= since);
  window.__rehearsalNow = () => performance.now();
  // Page markers by name and time, so the readiness cue can be identified, not assumed.
  window.__rehearsalMarkers = [];
  // Rows present before a turn are marked, so a check reads only the turn's own rows.
  window.__rehearsalMark = () => {
    for (const item of document.querySelectorAll("#transcript li")) {
      item.dataset.rehearsalSeen = "1";
    }
  };
  window.__rehearsalUnseen = (role) =>
    [...document.querySelectorAll(`#transcript li[data-role=${role}]`)]
      .filter((item) => item.dataset.rehearsalSeen !== "1"
        && item.dataset.partialTranscript !== "true")
      .map((item) => item.textContent ?? "");
  window.__rehearsalTasks = [];
  const record = (item) => {
    if (item instanceof HTMLElement && item.dataset.operation === "task" && item.dataset.status) {
      window.__rehearsalTasks.push([item.dataset.taskId, item.dataset.status, performance.now()]);
    }
    if (item instanceof HTMLLIElement && item.parentElement?.id === "markers") {
      const name = (item.textContent ?? "").split(":", 1)[0];
      window.__rehearsalMarkers.push([name, performance.now()]);
    }
  };
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "attributes") record(mutation.target);
      else for (const node of mutation.addedNodes) record(node);
    }
  }).observe(document, { subtree: true, childList: true, attributes: true,
                         attributeFilter: ["data-status"] });
})();
"""

_SNAPSHOT = r"""
() => {
  const final = (role) => [...document.querySelectorAll(`#transcript li[data-role=${role}]`)]
    .filter((item) => item.dataset.partialTranscript !== "true");
  return {
    state: document.querySelector("#connection-status")?.dataset.state ?? null,
    label: document.querySelector("#connection-state")?.textContent ?? "",
    toggle: (document.querySelector("#session-toggle")?.textContent ?? "").trim(),
    typed: !(document.querySelector("#typed-input")?.disabled ?? true),
    users: final("user").length,
    assistants: final("assistant").length,
    live: [...document.querySelectorAll('#transcript li[data-role=assistant]')]
      .filter((item) => item.dataset.partialTranscript === "true").length,
    interrupted: [...document.querySelectorAll('#transcript li[data-role=assistant]')]
      .filter((item) => item.dataset.interrupted === "true").length,
    results: document.querySelectorAll('#transcript li[data-role=task-result]').length,
    tasks: [...document.querySelectorAll('#transcript li[data-operation=task]')]
      .map((item) => [item.dataset.taskId, item.dataset.status]),
    latency: (document.querySelector("#latency-last")?.textContent ?? "").trim(),
    markers: [...document.querySelectorAll("#markers li")].map((item) => item.textContent),
    deletion: (document.querySelector("#voice-delete-status")?.textContent ?? "").trim(),
    deletable: !(document.querySelector("#delete-voice-conversation")?.disabled ?? true),
  };
}
"""


_HIT_TEST = r"""
(control) => {
  control.scrollIntoView({ block: "center" });
  const box = control.getBoundingClientRect();
  const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
  const name = (element) => element === null ? null
    : [element.tagName.toLowerCase(), element.id, element.className].join("|");
  return {
    self: hit === control,
    inside: hit !== null && control.contains(hit),
    element: name(hit),
    box: [Math.round(box.left), Math.round(box.top), Math.round(box.width),
          Math.round(box.height), window.innerWidth, window.innerHeight],
  };
}
"""


def _isolated(home: Path) -> dict[str, str]:
    """Hermes's environment: its homes replaced, so no credential store is reachable."""

    from hermes_realtime.providers.codex_app_server import _subscription_environment

    homes = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HERMES_HOME", "CODEX_HOME")
    environment: dict[str, str] = dict(_subscription_environment(os.environ))
    return environment | dict.fromkeys(homes, str(home)) | {"PYTHONIOENCODING": "utf-8"}


def _chrome() -> Path | None:
    for base in (
        os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"),
        os.environ.get("PROGRAMFILES", "C:/Program Files"),
    ):
        candidate = Path(base) / "Google/Chrome/Application/chrome.exe"
        if candidate.is_file():
            return candidate
    return None


def _http(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read(1 << 20)
    except urllib.error.HTTPError as error:
        return error.code, b""
    except (urllib.error.URLError, OSError):
        return 0, b""


def preflight(model: str | None) -> list[str]:
    """The local pieces this rehearsal needs and does not have."""

    missing: list[str] = []
    if sys.platform != "win32":
        return ["windows"]
    if model is None:
        missing.append("ollama_model")
    try:
        local_livekit.verified_server()
    except local_livekit.LiveKitUnavailable:
        missing.append("livekit")
    if _chrome() is None:
        missing.append("chrome")
    for module in ("playwright", "kokoro_onnx", "aiohttp"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    for tool in ("git", "uv"):
        if shutil.which(tool) is None:
            missing.append(tool)
    status, body = _http(f"{_OLLAMA}/api/tags")
    if status != 200:
        missing.append("ollama")
    elif model is not None:
        try:
            names = {entry["name"] for entry in json.loads(body)["models"]}
        except (ValueError, KeyError, TypeError):
            names = set()
        if model not in names:
            missing.append("ollama_model")
    if listeners(7880):
        missing.append("livekit_port_in_use")
    return missing


class Rehearsal:
    def __init__(self, run_dir: Path, model: str, recall_trials: int = 0) -> None:
        self.run_dir = run_dir
        self.model = model
        self.recall_trials = recall_trials
        self.home = run_dir / "home"
        self.logs_dir = run_dir / "logs"
        self.state_dir = run_dir / "state"
        self.candidate = run_dir / "candidate"
        self.host_env = run_dir / "host-env"
        self.wheel = Path()
        self.processes = Processes(run_dir)
        self.stand_in = StandInModel()
        self.logs: list[LogTail] = []
        self.records: list[dict[str, object]] = []
        self.api_port = available_port()
        self.companion_port = available_port()
        self.host_port = available_port()
        self.phrase = secrets.choice(_BIRDS)
        self.clips: dict[str, str] = {}
        self.url: str | None = None
        self.host_incarnation = 0
        self.page: Any = None
        self.browser: Any = None
        self.playwright: Any = None
        self.ready = {"gateway": False, "host": False, "browser": False}
        self.dialogs: Counter[str] = Counter()
        # The server's status for every typed turn and approval decision the page sent, as
        # the browser's network layer saw it, never as the page reported it.
        self.answers: list[tuple[str, int]] = []
        self.deleted = False
        self.containment: Containment | None = None
        self._step: Step | None = None

    # -- helpers --

    @contextlib.asynccontextmanager
    async def step(self, number: str, name: str) -> AsyncIterator[Step]:
        step = self._step = Step(number, name, self.logs)
        try:
            yield step
        except Exception as error:  # A failed step is a finding; the rehearsal continues.
            step.fail(type(error).__name__)
            step.notes["failed_at"] = step.stage
            if self.page is not None:
                with contextlib.suppress(Exception):
                    step.notes["page"] = _page_categories(await self.snapshot())
        finally:
            record = step.record()
            self.records.append(record)
            _emit(record)

    def _tail(self, source: str, path: Path) -> LogTail:
        tail = LogTail(source, path)
        self.logs.append(tail)
        return tail

    @property
    def hermes_python(self) -> Path:
        return self.home / "hermes-agent" / "venv" / "Scripts" / "python.exe"

    @property
    def hermes_cli(self) -> Path:
        return self.home / "hermes-agent" / "venv" / "Scripts" / "hermes.exe"

    @property
    def host_python(self) -> Path:
        return self.host_env / "Scripts" / "python.exe"

    async def snapshot(self) -> dict[str, Any]:
        value = await self.page.evaluate(_SNAPSHOT)
        assert type(value) is dict
        return value

    async def wait(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float, stage: str
    ) -> dict[str, Any]:
        if self._step is not None:
            self._step.stage = stage
        async with asyncio.timeout(timeout):
            while True:
                snapshot = await self.snapshot()
                if predicate(snapshot):
                    return snapshot
                await asyncio.sleep(0.1)

    async def type(self, text: str) -> None:
        await self.page.locator("#typed-input").fill(text)
        await self.page.locator("#send").click()

    async def speak(self, clip: str) -> float:
        duration = await self.page.evaluate(
            "(encoded) => window.__rehearsalSpeak(encoded)", self.clips[clip]
        )
        return float(duration)

    async def reply(
        self, before: dict[str, Any], timeout: float = _REPLY_SECONDS
    ) -> dict[str, Any]:
        return await self.wait(
            lambda now: now["assistants"] > before["assistants"] and now["live"] == 0,
            timeout,
            "reply",
        )

    async def mark(self) -> float:
        """Mark every row on the page as seen; the page time to observe from."""

        await self.page.evaluate("() => window.__rehearsalMark()")
        return float(await self.page.evaluate("() => window.__rehearsalNow()"))

    async def unseen(self, role: str) -> list[str]:
        """The final rows of ``role`` added since the last mark (held here, never recorded)."""

        rows = await self.page.evaluate("(role) => window.__rehearsalUnseen(role)", role)
        return [str(row) for row in rows]

    async def unseen_assistant_rows(self) -> int:
        """Assistant rows added since the last mark, live ones included."""

        return int(
            await self.page.evaluate(
                "() => [...document.querySelectorAll('#transcript li[data-role=assistant]')]"
                ".filter((item) => item.dataset.rehearsalSeen !== '1').length"
            )
        )

    async def turn_has_phrase(self) -> bool:
        """Does any assistant row of this turn say the phrase? Replies span one row per
        sentence, so every row since the mark is read, not only the last."""

        return any(self.phrase in row.casefold() for row in await self.unseen("assistant"))

    async def quiet(self, step: Step, name: str, since: float) -> None:
        """Was anything heard after ``since`` that no one asked for, beyond the readiness cue?"""

        timeline = await self.page.evaluate("(since) => window.__rehearsalTimeline(since)", since)
        confirmations = [
            float(at)
            for marker, at in await self.page.evaluate("() => window.__rehearsalMarkers")
            if marker == "voice_input_server_confirmed" and float(at) >= since
        ]
        findings, notes = audio_without_speech(
            [(float(at), float(energy)) for at, energy in timeline],
            confirmations[0] if confirmations else None,
            await self.unseen_assistant_rows(),
        )
        step.notes[f"{name}_audio"] = notes
        step.apply(findings)

    async def energy(self) -> float:
        return float(await self.page.evaluate("() => window.__rehearsalEnergy()"))

    async def ensure_connected(self, step: Step) -> None:
        """Reconnect a page an earlier step left disconnected, so this step still runs."""

        now = await self.snapshot()
        if now["state"] == "connected" and now["label"] == "Listening":
            return
        step.notes["reconnected_first"] = True
        step.timings["reconnect_first_s"] = await self.connect_until(_RECONNECT_SECONDS)

    async def connect_until(self, budget: float) -> float:
        """Press Connect until it reaches Listening; seconds taken. Refusals are retried."""

        started = time.monotonic()
        while True:
            try:
                await self.connect()
                return round(time.monotonic() - started, 2)
            except TimeoutError:
                if time.monotonic() - started > budget:
                    raise
                await asyncio.sleep(5)

    async def connect(self) -> float:
        started = time.monotonic()
        await self.wait(lambda now: now["toggle"] == "Connect", 30, "connect_offered")
        await self.page.locator("#session-toggle").click()
        await self.wait(
            lambda now: now["state"] == "connected" and now["label"] == "Listening",
            90,
            "listening",
        )
        return round(time.monotonic() - started, 2)

    async def task_log(self) -> list[tuple[str, str, float]]:
        value = await self.page.evaluate("() => window.__rehearsalTasks")
        return [(str(a), str(b), float(c)) for a, b, c in value]

    async def track_task(
        self, before_ids: set[str], started: float, step: Step, timeout: float
    ) -> tuple[str | None, list[str]]:
        """The new task's id and its state sequence, with seconds to each first state."""

        task_id: str | None = None
        sequence: list[str] = []
        async with asyncio.timeout(timeout):
            while True:
                for identity, status in (await self.snapshot())["tasks"]:
                    if identity not in before_ids and task_id is None:
                        task_id = identity
                    if identity == task_id and (not sequence or sequence[-1] != status):
                        sequence.append(status)
                        step.timings.setdefault(f"{status}_s", round(time.monotonic() - started, 2))
                if sequence and sequence[-1] in _TERMINAL_TASK:
                    return task_id, sequence
                await asyncio.sleep(0.05)

    async def observed_sequence(self, task_id: str) -> list[str]:
        """Every state the page rendered for one task, from its mutation log."""

        sequence: list[str] = []
        for identity, status, _ in await self.task_log():
            if identity == task_id and (not sequence or sequence[-1] != status):
                sequence.append(status)
        return sequence

    # -- setup: the runbook's setup steps 1 to 7 --

    async def setup(self) -> None:
        async with self.step("setup", "setup") as step:
            started = time.monotonic()
            step.notes["candidate"] = await asyncio.to_thread(self._candidate)
            step.time("candidate_s", started)
            cache_python = await asyncio.to_thread(provision_pinned_hermes)
            del cache_python
            checkout = self.home / "hermes-agent"
            await asyncio.to_thread(self._clone, checkout)
            step.time("install_s", started)
            synced = await asyncio.to_thread(
                _run,
                ["uv", "sync", "--extra", "all", "--locked", "--python", "3.11", "--quiet"],
                env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(checkout / "venv")},
                cwd=checkout,
                timeout=1800,
            )
            if synced.returncode != 0:
                raise RuntimeError("the installer-layout environment did not sync")
            step.time("environment_s", started)
            base_url = await self.stand_in.start()
            config = {
                "model": {"provider": "custom", "base_url": base_url, "default": "stand-in"},
                "auxiliary": {"title_generation": {"enabled": False}},
            }
            (self.home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
            cli = self._hermes_command
            step.notes["profile_use_default"] = await cli(step, "profile", "use", "default")
            step.notes["memory_off"] = await cli(step, "memory", "off")
            memory = await asyncio.to_thread(self._hermes, "memory", "status")
            step.notes["memory_status_builtin"] = "built-in" in memory.stdout.casefold()
            step.notes.update(await self._install_wheel(step))
            step.notes["plugin_enable"] = await cli(
                step, "plugins", "enable", "hermes-realtime", "--no-allow-tool-override"
            )
            listed = await asyncio.to_thread(self._hermes, "plugins", "list", "--enabled")
            step.notes["plugin_listed"] = "hermes-realtime" in listed.stdout
            self._write_env()
            step.time("configured_s", started)
            await self._start_gateway(step)
            step.time("gateway_ready_s", started)

    def _clone(self, checkout: Path) -> None:
        source = PINNED_HERMES / "source"
        commit = HERMES_BASELINE["commit"]
        checkout.parent.mkdir(parents=True, exist_ok=True)
        for argv in (
            ["git", "clone", "-q", "--no-checkout", str(source), str(checkout)],
            ["git", "-C", str(checkout), "config", "core.autocrlf", "false"],
            ["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", "/*", "!/website/"],
            ["git", "-C", str(checkout), "checkout", "-q", "--detach", commit],
            ["git", "-C", str(checkout), "remote", "set-url", "origin", _UPSTREAM],
        ):
            if _run(argv).returncode != 0:
                raise RuntimeError("the installer-layout checkout failed")

    def _hermes(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        completed = _run(
            [str(self.hermes_cli), *arguments], env=_isolated(self.home), timeout=300
        )
        with (self.logs_dir / "hermes-cli.log").open("a", encoding="utf-8") as log:
            log.write(f"$ hermes {' '.join(arguments)} -> {completed.returncode}\n")
            log.write(completed.stdout + completed.stderr)
        return completed

    async def _hermes_command(self, step: Step, *arguments: str) -> str:
        completed = await asyncio.to_thread(self._hermes, *arguments)
        found, _ = markers((completed.stdout + completed.stderr).splitlines())
        if found:
            step.notes[f"markers_{arguments[0]}_{arguments[1]}"] = found
        if completed.returncode != 0:
            step.differ(f"hermes_{arguments[0]}_{arguments[1]}")
            return f"exit_{completed.returncode}"
        return "ok"

    def _candidate(self) -> dict[str, object]:
        """Bind the run to a commit: a clone of HEAD, its wheel, and a host env from both.

        The working tree may hold anything; what runs is exactly HEAD, and the record says
        which commit that is and whether the tree it was taken from was clean.
        """

        commit = _run(["git", "rev-parse", "HEAD"], cwd=_REPOSITORY).stdout.strip()
        tracked = _run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPOSITORY
        ).stdout
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RuntimeError("the candidate commit could not be read")
        for argv in (
            ["git", "clone", "-q", "--no-checkout", "--no-hardlinks", str(_REPOSITORY),
             str(self.candidate)],
            ["git", "-C", str(self.candidate), "checkout", "-q", "--detach", commit],
        ):  # fmt: skip
            if _run(argv).returncode != 0:
                raise RuntimeError("the candidate clone failed")
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
        }
        built = _run(
            ["uv", "build", "--wheel", "--out-dir", str(self.run_dir / "dist")],
            env=clean_env,
            cwd=self.candidate,
        )
        wheels = sorted((self.run_dir / "dist").glob("hermes_realtime-*-py3-none-any.whl"))
        if built.returncode != 0 or len(wheels) != 1:
            raise RuntimeError("the candidate wheel did not build")
        self.wheel = wheels[0]
        requirements = self.run_dir / "host-requirements.txt"
        steps = (
            ["uv", "export", "--frozen", "--no-dev", "--extra", "local", "--no-emit-project",
             "--no-hashes", "--quiet", "-o", str(requirements)],
            ["uv", "venv", "--quiet", "--python", "3.11", str(self.host_env)],
            ["uv", "pip", "install", "--quiet", "--python", str(self.host_python), "-r",
             str(requirements)],
            ["uv", "pip", "install", "--quiet", "--no-deps", "--python", str(self.host_python),
             str(self.wheel)],
        )  # fmt: skip
        for argv in steps:
            if _run(argv, env=clean_env, cwd=self.candidate, timeout=1800).returncode != 0:
                raise RuntimeError("the candidate host environment did not install")
        located = _run(
            [str(self.host_python), "-I", "-c",
             "import hermes_realtime; print(hermes_realtime.__file__)"],
            env=clean_env,
        ).stdout.strip()  # fmt: skip
        return {
            "commit": commit,
            "clean": tracked == "",
            "harness_at_head": harness_matches_head(_REPOSITORY, "scripts/rehearse_desktop_mvp.py"),
            "host_from_wheel": Path(located).resolve().is_relative_to(self.host_env.resolve()),
        }

    async def _install_wheel(self, step: Step) -> dict[str, object]:
        python = str(self.hermes_python)
        before = await asyncio.to_thread(_run, ["uv", "pip", "freeze", "--python", python])
        installed = await asyncio.to_thread(
            _run,
            [
                "uv", "pip", "install", "--reinstall-package", "hermes-realtime",
                "--python", python, str(self.wheel),
            ],
        )
        if installed.returncode != 0:
            raise RuntimeError("the candidate wheel did not install")
        after = await asyncio.to_thread(_run, ["uv", "pip", "freeze", "--python", python])
        old = {line for line in before.stdout.splitlines() if "hermes-realtime" not in line}
        new = {line for line in after.stdout.splitlines() if "hermes-realtime" not in line}
        return {
            "wheel_version": self.wheel.name.split("-")[1],
            "shared_packages_changed": len(old ^ new),
        }

    def _write_env(self) -> None:
        lines = (
            "API_SERVER_ENABLED=true",
            "API_SERVER_HOST=127.0.0.1",
            f"API_SERVER_PORT={self.api_port}",
            f"API_SERVER_KEY={secrets.token_urlsafe(32)}",
            f"HERMES_REALTIME_COMPANION_PORT={self.companion_port}",
            f"HERMES_REALTIME_COMPANION_TOKEN={secrets.token_urlsafe(32)}",
        )
        (self.home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def _start_gateway(self, step: Step) -> None:
        log = self.logs_dir / "gateway.log"
        self._tail("gateway", log)
        self.processes.spawn(
            "gateway",
            [str(self.hermes_cli), "gateway", "run"],
            env=_isolated(self.home),
            cwd=self.home,
            log=log,
        )
        async with asyncio.timeout(240):
            while _http(f"http://127.0.0.1:{self.api_port}/health")[0] != 200:
                if self.processes.roots["gateway"].poll() is not None:
                    raise RuntimeError("the gateway exited before it was healthy")
                await asyncio.sleep(0.5)
            while not listeners(self.companion_port):
                await asyncio.sleep(0.5)
        exposed = listener_verdict("api", listeners(self.api_port)) + listener_verdict(
            "companion", listeners(self.companion_port)
        )
        step.apply(exposed)
        self.processes.observe()
        # Nothing runs behind a listener that is open beyond loopback.
        self.ready["gateway"] = not exposed

    # -- session step 1 --

    async def gate(self) -> None:
        async with self.step("1", "installed_runtime_gate") as step:
            if not self.ready["gateway"]:
                step.skip("no_gateway")
                return
            started = time.monotonic()
            completed = await asyncio.to_thread(
                _run,
                [
                    str(self.hermes_python),
                    str(self.candidate / "scripts" / "real_hermes_api_gate.py"),
                    "--hermes-home",
                    str(self.home),
                    "--hermes-api-url",
                    f"http://127.0.0.1:{self.api_port}",
                ],
                env=_isolated(self.home),
                cwd=self.candidate,
                timeout=1200,
            )
            step.time("gate_s", started)
            lines = completed.stdout.splitlines()
            step.notes["stdout_lines"] = len(lines)
            step.notes["stderr_bytes"] = len(completed.stderr.encode())
            line = lines[-1] if lines else ""
            if line.startswith("[real-hermes-gate] "):
                refusal, _ = markers([line])
                step.notes["gate"] = refusal[0] if refusal else "unrecognised"
                step.fail("gate_refused")
            else:
                try:
                    record = json.loads(line)
                except ValueError:
                    record = None
                if (
                    type(record) is not dict
                    or record.get("gate") != "passed"
                    or not _is_category(record)
                ):
                    step.fail("gate_unreadable")
                else:
                    step.notes["gate"] = record
            if len(lines) != 1 or completed.stderr:
                step.differ("gate_output_shape")

    # -- session step 2 --

    async def start_and_talk(self) -> None:
        async with self.step("2", "start_host_open_browser_talk") as step:
            if not self.ready["gateway"]:
                step.skip("no_gateway")
                return
            started = time.monotonic()
            step.notes["prior_voice_tail"] = (self.state_dir / "voice-tail-v1.json").exists()
            if not await self._start_livekit(step):
                return
            step.time("livekit_ready_s", started)
            await self.start_host()
            step.time("host_ready_s", started)
            start_markers, _ = markers(step.lines("host"))
            if any(line.startswith("[voice-tail] ") for line in start_markers):
                step.differ("voice_tail_at_first_start")
            await self._start_browser()
            step.time("browser_ready_s", started)
            since = await self.mark()
            step.timings["connect_to_listening_s"] = await self.connect()
            # Nothing should be heard but the identified readiness cue.
            await asyncio.sleep(_QUIET_SECONDS)
            await self.quiet(step, "connect", since)
            step.notes["session_model"] = await self.page.evaluate(
                "() => Object.fromEntries(['model-provider','model-name','stt-provider',"
                "'stt-model','tts-provider','tts-model'].map((id) => "
                "[id, (document.getElementById(id)?.textContent ?? '').trim()]))"
            )
            before = await self.snapshot()
            await self.mark()
            energy = await self.energy()
            spoken = time.monotonic()
            await self.speak("question")
            try:
                await self.wait(lambda now: now["users"] > before["users"], 60, "final_transcript")
                step.notes["final_transcript"] = True
                step.time("final_transcript_s", spoken)
            except TimeoutError:
                step.notes["final_transcript"] = False
                step.differ("no_final_transcript")
                return
            # The transcript must say what was spoken, not merely exist.
            heard = " ".join(await self.unseen("user"))
            step.notes["transcript_matches"] = transcript_matches(_CLIPS["question"], heard)
            if not step.notes["transcript_matches"]:
                step.differ("transcript_mismatch")
            after = await self.reply(before)
            step.time("reply_delivered_s", spoken)
            await self._reply_observations(step, after, await self.energy() - energy)

    async def _reply_observations(self, step: Step, after: dict[str, Any], energy: float) -> None:
        step.notes["audible"] = energy > _AUDIBLE_ENERGY
        step.notes["audio_energy"] = round(energy, 6)
        if not step.notes["audible"]:
            step.differ("not_audible")
        step.notes["response_last"] = after["latency"]
        named = latencies(after["markers"])
        for name in ("transcript_to_first_token", "first_token_to_audio"):
            if name in named:
                step.timings[f"{name}_ms"] = named[name]

    async def _start_livekit(self, step: Step) -> bool:
        """Start LiveKit; False, with the step failed, when signaling is open beyond loopback."""

        log = self.logs_dir / "livekit.log"
        self.processes.spawn(
            "livekit",
            local_livekit.server_command(local_livekit.verified_server()),
            env=os.environ | {"LIVEKIT_KEYS": local_livekit.DEVELOPMENT_KEYS},
            log=log,
        )
        async with asyncio.timeout(30):
            while _http("http://127.0.0.1:7880/") != (200, b"OK"):
                await asyncio.sleep(0.2)
        exposed = listener_verdict("livekit_signaling", listeners(7880))
        step.notes["livekit_signaling_loopback_only"] = not exposed
        step.apply(exposed)
        return not exposed

    def _host_argv(self) -> list[str]:
        return [
            str(self.host_python),
            str(self.candidate / "scripts" / "rehearse_desktop_mvp.py"),
            "--host-child", str(self.run_dir / _HOST_STOP_FILE), "--",
            "--hermes-env-file", str(self.home / ".env"),
            "--hermes-api-url", f"http://127.0.0.1:{self.api_port}",
            "--inference-provider", "ollama", "--ollama-model", self.model,
            "--persistent-loopback-launch",
            "--allow-unsandboxed-hermes-tasks",
            "--port", str(self.host_port),
            "--voice-tail", str(self.state_dir / "voice-tail-v1.json"),
            "--hermes-run-record", str(self.state_dir / "hermes-runs-v1.json"),
        ]  # fmt: skip

    async def start_host(self) -> None:
        self.host_incarnation += 1
        log = self.logs_dir / f"host-{self.host_incarnation}.log"
        tail = self._tail("host", log)
        self.processes.spawn(
            "host",
            self._host_argv(),
            env={
                key: value
                for key, value in os.environ.items()
                if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
            }
            | {
                # For the host child's content-free observation of each prompt.
                "HERMES_REHEARSAL_PHRASE": self.phrase,
                "HERMES_REHEARSAL_MODEL": self.model,
                "HERMES_REHEARSAL_RECALL_TRIALS": str(self.recall_trials),
            },
            cwd=self.candidate,
            log=log,
        )
        async with asyncio.timeout(900):
            while True:
                tail.poll()
                if _LAUNCH_LINE in tail.lines:
                    index = tail.lines.index(_LAUNCH_LINE)
                    if len(tail.lines) > index + 1:
                        self.url = tail.lines[index + 1].strip()
                        break
                if self.processes.roots["host"].poll() is not None:
                    raise RuntimeError("the host exited before it offered its URL")
                await asyncio.sleep(0.5)
        self.processes.observe()
        self.ready["host"] = True

    async def _start_browser(self) -> None:
        # Imported by name: the hermetic release gate type-checks this script without it.
        playwright = importlib.import_module("playwright.async_api")

        chrome = _chrome()
        assert chrome is not None
        port = available_port()
        self.processes.spawn(
            "chrome",
            [
                str(chrome),
                "--headless=new",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={self.run_dir / 'chrome-profile'}",
                "--no-first-run",
                "--no-default-browser-check",
                "--autoplay-policy=no-user-gesture-required",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "about:blank",
            ],
            env=os.environ,
            log=self.logs_dir / "chrome.log",
        )
        async with asyncio.timeout(60):
            while _http(f"http://127.0.0.1:{port}/json/version")[0] != 200:
                await asyncio.sleep(0.2)
        self.playwright = await playwright.async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        context = self.browser.contexts[0]
        await context.add_init_script(_INIT_SCRIPT)
        self.page = await context.new_page()
        # A common laptop desktop window, not the headless default.
        await self.page.set_viewport_size(_VIEWPORT)

        async def accept_delete(dialog: Any) -> None:
            if dialog.message.startswith("Delete this voice conversation?"):
                self.dialogs["accepted"] += 1
                await dialog.accept()
            else:
                self.dialogs["dismissed"] += 1
                await dialog.dismiss()

        self.page.on("dialog", accept_delete)

        def record_answer(response: Any) -> None:
            path = urlsplit(response.url).path
            if response.request.method == "POST" and path in _ANSWERED_PATHS:
                self.answers.append((path, int(response.status)))

        self.page.on("response", record_answer)
        await self.open_page()
        self.processes.observe()
        self.ready["browser"] = True

    async def open_page(self) -> None:
        assert self.url is not None
        await self.page.goto(self.url)
        await self.page.wait_for_load_state("networkidle")

    # -- session step 3 --

    async def typed(self) -> None:
        async with self.step("3", "typed_response") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            before = await self.snapshot()
            energy = await self.energy()
            sent = time.monotonic()
            await self.type(
                f"Remember this for later: my favorite bird is the {self.phrase}. "
                "What is the capital of Italy?"
            )
            after = await self.reply(before)
            step.time("reply_delivered_s", sent)
            await self._reply_observations(step, after, await self.energy() - energy)

    # -- session step 4 --

    async def delegate(self) -> None:
        async with self.step("4", "delegate_task") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            before = await self.snapshot()
            ids = {identity for identity, _ in before["tasks"]}
            await self.mark()
            sent = time.monotonic()
            await self.type("task: Reply with one sentence confirming the rehearsal delegation.")
            task_id, sequence = await self.track_task(ids, sent, step, 300)
            if task_id is not None:
                sequence = await self.observed_sequence(task_id) or sequence
            step.notes["states"] = sequence
            if sequence[:1] != ["active"] or sequence[-1:] != ["completed"]:
                step.differ("state_sequence")
            # The result Hermes produced must reach the page: the stand-in's own sentence in a
            # background-result row. A spoken reply alone is recorded, not counted.
            step.stage = "result_reported"
            reported = False
            deadline = time.monotonic() + 60
            while not reported and time.monotonic() < deadline:
                rows = await self.unseen("task-result")
                reported = any(_STAND_IN_RESULT in row for row in rows)
                if not reported:
                    await asyncio.sleep(0.2)
            step.notes["result_reported"] = reported
            if not reported:
                step.differ("result_not_reported")
            step.notes["assistant_rows_after_result"] = len(await self.unseen("assistant"))
            body = await self.page.evaluate("() => document.body.innerText")
            step.notes["private_ids"] = len(_PRIVATE_ID.findall(body))
            if step.notes["private_ids"]:
                step.fail("private_id_visible")

    # -- session step 5 --

    async def interrupt(self) -> None:
        async with self.step("5", "interrupt_speech_while_task_runs") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            before = await self.snapshot()
            ids = {identity for identity, _ in before["tasks"]}
            sent = time.monotonic()
            await self.type(
                "task: Wait about 60 seconds, then reply with one sentence confirming the "
                "long rehearsal task."
            )
            now = await self.wait(
                lambda now: any(i not in ids and s == "active" for i, s in now["tasks"]),
                120,
                "task_active",
            )
            step.time("active_s", sent)
            task_id = next(i for i, s in now["tasks"] if i not in ids)
            asked = await self.snapshot()
            await self.speak("story")
            await self.wait(lambda now: now["users"] > asked["users"], 60, "story_transcript")
            await self.wait(lambda now: now["live"] > 0, _REPLY_SECONDS, "reply_live")
            # Speak over the reply only once it is audibly playing.
            step.stage = "playback"
            energy = await self.energy()
            async with asyncio.timeout(60):
                while await self.energy() - energy <= 10 * _AUDIBLE_ENERGY:
                    await asyncio.sleep(0.1)
            states = dict((await self.snapshot())["tasks"])
            step.notes["task_before_interruption"] = states.get(task_id)
            interrupted = await self.snapshot()
            spoken = time.monotonic()
            await self.speak("interruption")
            try:
                await self.wait(
                    lambda now: now["interrupted"] > interrupted["interrupted"], 20, "yield"
                )
                step.notes["playback_yielded"] = True
                step.time("speech_to_yield_s", spoken)
            except TimeoutError:
                step.notes["playback_yielded"] = False
                step.differ("playback_did_not_yield")
            step.notes["task_after_interruption"] = dict((await self.snapshot())["tasks"]).get(
                task_id
            )
            if step.notes["task_after_interruption"] != "active":
                step.fail("task_changed_by_interruption")
            final = await self.wait(
                lambda now: dict(now["tasks"]).get(task_id) in _TERMINAL_TASK, 240, "task_terminal"
            )
            step.time("task_terminal_s", sent)
            step.notes["task_final"] = dict(final["tasks"]).get(task_id)
            if step.notes["task_final"] != "completed":
                step.differ("task_not_completed")
            # The page keeps a bounded marker list, so new entries are found by content.
            new = list((Counter(final["markers"]) - Counter(before["markers"])).elements())
            step.notes["latency_markers_ms"] = latencies(new)

    # -- session step 6 --

    async def cancel(self) -> None:
        async with self.step("6", "cancel_task") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            before = await self.snapshot()
            ids = {identity for identity, _ in before["tasks"]}
            await self.type(
                "task: Keep working on the rehearsal cancellation task until you are stopped."
            )
            now = await self.wait(
                lambda now: any(i not in ids and s == "active" for i, s in now["tasks"]),
                120,
                "task_active",
            )
            task_id = next(i for i, s in now["tasks"] if i not in ids)
            others = {i: s for i, s in now["tasks"] if i != task_id}
            await self.wait(
                lambda now: self.stand_in.open["cancellation"] > 0, 60, "work_started"
            )
            sent = time.monotonic()
            await self.type(f"cancel task: {task_id}")
            final = await self.wait(
                lambda now: dict(now["tasks"]).get(task_id) in _TERMINAL_TASK,
                120,
                "cancel_terminal",
            )
            step.time("cancel_to_terminal_s", sent)
            sequence = await self.observed_sequence(task_id)
            step.notes["states"] = sequence
            if sequence[-2:] != ["cancelling", "interrupted"]:
                step.differ("state_sequence")
            changed = sum(1 for i, s in final["tasks"] if i in others and others[i] != s)
            step.notes["other_tasks_changed"] = changed
            if changed:
                step.fail("other_task_changed")
            # The stop must reach the work itself: the stand-in's stream must be hung up.
            deadline = time.monotonic() + _SETTLE_SECONDS
            while self.stand_in.open["cancellation"] and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            step.notes["model_streams_open"] = self.stand_in.open["cancellation"]
            step.apply(cancel_work_verdict(self.stand_in.open["cancellation"]))

    # -- session step 7 --

    async def reconnect(self) -> None:
        async with self.step("7", "reconnect_browser") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            await self.page.locator("#session-toggle").click()
            await self.wait(
                lambda now: now["toggle"] == "Connect" and not now["typed"], 60, "stopped"
            )
            step.stage = "stop_connect"
            since = await self.mark()
            step.timings["stop_connect_to_listening_s"] = await self.connect()
            await self._after_reconnect(step, "stop_connect", since)
            step.stage = "reload"
            await self.page.reload()
            await self.page.wait_for_load_state("networkidle")
            since = await self.mark()
            try:
                step.timings["reload_connect_to_listening_s"] = await self.connect()
            except TimeoutError:
                # The page left without stopping, so its session may still hold the door.
                step.notes["reload_connect_first"] = _page_categories(await self.snapshot())
                step.differ("reload_connect_refused")
                step.timings["reload_connect_recovered_s"] = await self.connect_until(
                    _RECONNECT_SECONDS
                )
            await self._after_reconnect(step, "reload_connect", since)
            # The same host kept running: the tail mirrors the context it holds.
            rows = tail_rows(self._tail_bytes())
            step.notes["context_rows"] = None if rows is None else len(rows)
            await self._context_check(
                step, "context_carried_over", rows is not None and stated_by_user(rows, self.phrase)
            )

    # -- after deletion: a reloaded page's input counters, which the runbook has no step for --

    async def reload_inputs(self) -> None:
        """A reloaded page restarts its typed-input and approval counters, so the session
        behind it must restart them too. Its own step, after the context checks and the
        deletion: its turns would otherwise push the step-3 fact out of the bounded tail."""

        async with self.step("reload_inputs", "reload_input_counters") as step:
            step.notes["runbook_step"] = None
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            # Spend both counters in this session, then reload and use them again.
            await self._typed_round(step, "typed_before_reload")
            await self._approval_round(step, "approval_before_reload")
            step.stage = "reload"
            await self.page.reload()
            await self.page.wait_for_load_state("networkidle")
            step.timings["reload_connect_to_listening_s"] = await self.connect()
            await self._typed_round(step, "typed_after_reload")
            await self._approval_round(step, "approval_after_reload")

    def _answer_since(self, mark: int, path: str) -> int | None:
        statuses = [status for seen, status in self.answers[mark:] if seen == path]
        return statuses[-1] if statuses else None

    async def _typed_round(self, step: Step, name: str) -> None:
        """One typed turn: the server's status for it on the wire, and the reply it drew."""

        step.stage = name
        before = await self.snapshot()
        await self.mark()
        mark = len(self.answers)
        await self.type("Please answer in one short sentence: what is two plus two?")
        replied = False
        with contextlib.suppress(TimeoutError):
            await self.reply(before)
            replied = True
        status = self._answer_since(mark, "/api/v1/input")
        step.notes[f"{name}_status"] = status
        step.notes[f"{name}_replied"] = replied
        step.apply(page_input_verdict(name, status, replied))

    async def _approval_round(self, step: Step, name: str) -> None:
        """One task that needs approval, rejected from its card: the server's status for the
        decision on the wire, and Hermes's own report to the model that the user denied it."""

        step.stage = name
        denied = self.stand_in.tool_outcomes["denied"]
        before = await self.snapshot()
        ids = {identity for identity, _ in before["tasks"]}
        sent = time.monotonic()
        await self.type(f"task: run chmod 777 rehearsal-{name.replace('_', '-')}-target")
        card = self.page.locator(
            "#transcript li[data-operation=approval][data-status=pending]"
        ).last
        try:
            await card.wait_for(state="visible", timeout=_APPROVAL_CARD_SECONDS * 1000)
        except Exception:
            step.notes[f"{name}_card"] = False
            step.fail(f"{name}_no_card")
            return
        step.notes[f"{name}_card"] = True
        mark = len(self.answers)
        await card.get_by_role("button", name="Reject").click()
        task_id: str | None = None
        sequence: list[str] = []
        with contextlib.suppress(TimeoutError):
            task_id, sequence = await self.track_task(ids, sent, step, _REPLY_SECONDS)
        deadline = time.monotonic() + 30
        while self.stand_in.tool_outcomes["denied"] == denied and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        witnessed = self.stand_in.tool_outcomes["denied"] > denied
        status = self._answer_since(mark, "/api/v1/approval")
        step.notes[f"{name}_status"] = status
        step.notes[f"{name}_denied_by_hermes"] = witnessed
        step.notes[f"{name}_task_states"] = sequence
        step.apply(page_input_verdict(name, status, witnessed))

    async def _after_reconnect(self, step: Step, name: str, since: float) -> None:
        """No stale audio and no duplicated transcript entries after a reconnect."""

        await asyncio.sleep(_QUIET_SECONDS)
        await self.quiet(step, name, since)
        entries = await self.page.evaluate(
            "() => [...document.querySelectorAll('#transcript li[data-role]')]"
            ".map((item) => item.dataset.role + ':' + item.textContent)"
        )
        duplicates = len(entries) - len(set(entries))
        step.notes[f"{name}_duplicate_entries"] = duplicates
        if duplicates:
            step.differ("duplicate_transcript")

    async def _ask_phrase(self) -> bool:
        before = await self.snapshot()
        await self.mark()
        await self.type(_BIRD_QUESTION)
        await self.reply(before)
        return await self.turn_has_phrase()

    async def _context_check(self, step: Step, name: str, carried: bool) -> None:
        """Is the step-3 fact still in the context the host keeps, and in the question's prompt?

        Nothing restates it: when the host's bounded window has dropped it, the verdict says
        so. The caller judges ``carried`` from the rows the host keeps, never from a raw count;
        the host child counts the prompt's fact rows at the adapter boundary. The model's
        answer, and with ``--recall-trials`` its rate on that exact prompt, are notes.
        """

        step.stage = name
        answered = await self._ask_phrase()
        step.notes[name] = carried
        step.notes[f"{name}_answered"] = answered
        # The question's own prompt: what the adapter sent beside what Ollama evaluated.
        step.notes[f"{name}_prompt"] = self._last_marker(step, "rehearsal-prompt")
        step.notes[f"{name}_ollama"] = self._last_marker(step, "ollama-prompt")
        if self.recall_trials:
            # A measurement, never a verdict: a missing one is recorded as null.
            step.stage = f"{name}_recall"
            deadline = time.monotonic() + _RECALL_SECONDS
            while (
                recall := self._last_marker(step, "rehearsal-recall")
            ) is None and time.monotonic() < deadline:
                await asyncio.sleep(1)
            step.notes[f"{name}_recall"] = recall
        prompt = step.notes[f"{name}_prompt"]
        in_prompt = None if type(prompt) is not dict else prompt.get("fact_rows", 0) > 0
        step.apply(context_verdict(carried, in_prompt))

    @staticmethod
    def _last_marker(step: Step, name: str) -> dict[str, Any] | None:
        sent = [line for line in step.lines("host") if line.startswith(f"[{name}] ")]
        found, _ = markers(sent[-1:])
        return json.loads(found[0].split("] ", 1)[1]) if found else None

    def _tail_bytes(self) -> bytes:
        tail = self.state_dir / "voice-tail-v1.json"
        return tail.read_bytes() if tail.exists() else b""

    def _tail_count(self) -> int:
        """Every occurrence of the phrase in the tail file: what a deletion must remove."""

        tail = self.state_dir / "voice-tail-v1.json"
        if not tail.exists():
            return 0
        return tail.read_text(encoding="utf-8").casefold().count(self.phrase)

    # -- session step 8 --

    async def restart(self) -> None:
        async with self.step("8", "restart_host") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            before = await self.snapshot()
            ids = {identity for identity, _ in before["tasks"]}
            await self.type(
                "task: Keep working on the rehearsal restart task until you are stopped."
            )
            await self.wait(
                lambda now: any(i not in ids and s == "active" for i, s in now["tasks"]),
                120,
                "task_active",
            )
            await self.wait(
                lambda now: self.stand_in.open["restart"] > 0, 60, "work_started"
            )
            running_before_crash = self.stand_in.open["restart"]
            step.timings["kill_s"] = round(self.processes.kill("host"), 2)
            self.ready["host"] = False
            await asyncio.sleep(2)
            running_after_crash = self.stand_in.open["restart"]
            step.notes["work_running_before_crash"] = running_before_crash
            step.notes["work_running_after_crash"] = running_after_crash
            # Exactly what the new host will read, taken before it can write anything.
            loaded = tail_rows(self._tail_bytes())
            restarted = time.monotonic()
            await self.start_host()
            step.time("restart_to_url_s", restarted)
            host_lines = self.logs[-1].lines
            ordered, _ = markers(host_lines)
            step.notes["start_marker_order"] = [line.split("]", 1)[0] + "]" for line in ordered]
            step.notes["notice"] = notice(host_lines)
            restored = _marker_value(ordered, "voice-tail", "restored")
            stopped = _marker_value(ordered, "hermes-restart-settlement", "stopped")
            if not (type(restored) is int and restored > 0):
                step.differ("no_restored_tail")
            if not (type(stopped) is int and stopped >= 1):
                step.differ("no_settlement")
            if step.notes["notice"] is None:
                step.differ("no_notice")
            deadline = time.monotonic() + _SETTLE_SECONDS
            while self.stand_in.open["restart"] and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            step.notes["work_running_after_settlement"] = self.stand_in.open["restart"]
            step.apply(
                restart_work_verdict(
                    running_before_crash, running_after_crash, self.stand_in.open["restart"]
                )
            )
            await self.open_page()
            step.timings["connect_to_listening_s"] = await self.connect()
            await asyncio.sleep(15)
            announcements = await self.page.evaluate(
                "(text) => [...document.querySelectorAll('#transcript li[data-role=assistant]')]"
                ".filter((item) => item.textContent.includes(text)).length",
                _RESTARTED,
            )
            step.notes["announcements"] = announcements
            if announcements != 1:
                step.differ("announcement_count")
            step.notes["context_rows"] = None if loaded is None else len(loaded)
            await self._context_check(
                step, "context_resumed", restored_fact(loaded, restored, self.phrase)
            )

    # -- M3: deletion, which the runbook has no session step for --

    async def delete(self) -> None:
        async with self.step("delete", "delete_voice_conversation") as step:
            step.notes["runbook_step"] = None
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            seeded = await self._phrase_counts()
            step.notes["phrase_before"] = seeded
            initial = await self.snapshot()
            step.notes["initial_state"] = _DELETE_STATES.get(initial["deletion"], "other")
            if not initial["deletable"]:
                step.fail("delete_unavailable")
                return
            control = self.page.locator("#delete-voice-conversation")
            step.stage = "delete_click"
            clicked = time.monotonic()
            try:
                await control.click(timeout=15_000)
            except Exception:
                step.timings["click_attempt_s"] = round(time.monotonic() - clicked, 2)
                step.notes["control"] = {
                    "visible": await control.is_visible(),
                    "enabled": await control.is_enabled(),
                    # What receives a click at the control's centre once it is scrolled into
                    # view: element kind, id and classes only.
                    "hit": await control.evaluate(_HIT_TEST),
                }
                if self.dialogs["accepted"]:
                    # The click landed and its confirmation was accepted; only the driver's
                    # wait for the click to settle expired.
                    step.notes["click"] = "landed_driver_wait_expired"
                else:
                    # Record why the operator's click would not land, then press it directly
                    # so the rest of the deletion is still rehearsed.
                    step.notes["click"] = "not_landed"
                    step.differ("delete_control_not_actionable")
                    await control.dispatch_event("click")
            step.notes["dialogs"] = dict(self.dialogs)
            states: list[str] = []
            async with asyncio.timeout(300):
                while not states or states[-1] != "complete":
                    state = _DELETE_STATES.get((await self.snapshot())["deletion"], "other")
                    if not states or states[-1] != state:
                        states.append(state)
                        step.timings.setdefault(f"{state}_s", round(time.monotonic() - clicked, 2))
                    await asyncio.sleep(0.1)
            step.notes["states"] = states
            after = await self._phrase_counts()
            step.notes["phrase_after"] = after
            asked = await self._ask_phrase()
            step.notes["phrase_in_next_reply"] = asked
            follow_up = await self._phrase_counts()
            step.notes["phrase_after_follow_up"] = follow_up
            step.apply(deletion_verdict(seeded, after, follow_up, asked))
            self.deleted = True

    async def _phrase_counts(self) -> dict[str, int]:
        """Where the conversation's unique phrase still is, as counts: the page, the voice
        tail, every table and full-text index of Hermes's database, and its session and
        memory files."""

        page = await self.page.evaluate(
            "(word) => (document.querySelector('#transcript')?.textContent ?? '')"
            ".toLowerCase().split(word).length - 1",
            self.phrase,
        )
        return {
            "page": int(page),
            "voice_tail": self._tail_count(),
            "hermes_database": phrase_in_database(self.home / "state.db", self.phrase),
            "sessions": phrase_in_files(self.home / "sessions", self.phrase),
            "memories": phrase_in_files(self.home / "memories", self.phrase),
        }

    # -- session step 9 --

    async def cleanup(self) -> None:
        async with self.step("9", "cleanup") as step:
            if self.ready["browser"]:
                await self.page.locator("#session-toggle").click()
                try:
                    await self.wait(
                        lambda now: now["toggle"] == "Connect" and not now["typed"], 60, "stopped"
                    )
                    step.notes["browser_stopped"] = True
                except TimeoutError:
                    step.notes["browser_stopped"] = False
                    step.differ("browser_not_stopped")
            # Ctrl-C cannot reach a detached process. The host child takes the same path on
            # its stop file; the gateway takes Hermes's own Windows stop path, the planned-stop
            # marker `hermes gateway stop` writes; LiveKit has no clean path and is killed.
            if self.ready["host"]:
                step.stage = "host_stop"
                (self.run_dir / _HOST_STOP_FILE).write_text("stop", encoding="utf-8")
                code = await asyncio.to_thread(self.processes.wait, "host", 60)
                step.notes["host_exit"] = "timeout" if code is None else code
                if code != 130:
                    step.differ("host_not_stopped_cleanly")
            if self.ready["gateway"]:
                step.stage = "gateway_stop"
                written = self._planned_stop()
                code = await asyncio.to_thread(self.processes.wait, "gateway", 60)
                step.notes["gateway_exit"] = (
                    "no_marker" if not written else "timeout" if code is None else code
                )
                if code != 0:
                    step.differ("gateway_not_stopped_cleanly")
            if "livekit" in self.processes.roots:
                self.processes.kill("livekit")
                step.notes["livekit_stop"] = "killed"
            if self.deleted:
                # With every writer stopped, the deleted phrase must still be gone.
                step.stage = "phrase_rescan"
                final = await self._phrase_counts()
                step.notes["phrase_at_cleanup"] = final
                step.apply(retained_verdict(final, "phrase_returned"))
        self.processes.observe()

    def _planned_stop(self) -> bool:
        """Ask the gateway to stop the way `hermes gateway stop` does on Windows."""

        try:
            raw = (self.home / "gateway.pid").read_text(encoding="utf-8").strip()
            value = json.loads(raw)
            pid = value["pid"] if type(value) is dict else value
        except (OSError, ValueError, KeyError):
            return False
        if type(pid) is not int:
            return False
        record = {
            "target_pid": pid,
            "target_start_time": None,
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(UTC).isoformat(),
        }
        marker = self.home / ".gateway-planned-stop.json"
        marker.write_text(json.dumps(record), encoding="utf-8")
        return True

    async def close(self) -> dict[str, object]:
        with contextlib.suppress(Exception):
            if self.browser is not None:
                await self.browser.close()
        with contextlib.suppress(Exception):
            if self.playwright is not None:
                await self.playwright.stop()
        self.processes.stop_all()
        await self.stand_in.close()
        ports = [7880, self.api_port, self.companion_port, self.host_port]
        return {
            "processes_left": self.processes.survivors(self.containment),
            "contained": self.containment is not None,
            "ports_listening": sum(1 for port in ports if listeners(port)),
        }


def _page_categories(snapshot: Mapping[str, Any]) -> dict[str, object]:
    """What the page showed, as UI states, counts and marker names: never text it holds."""

    names = sorted(
        {entry.split(":", 1)[0] for entry in snapshot["markers"] if _MARKER_NAME.match(entry)}
    )
    return {
        "connection": snapshot["state"],
        "label": snapshot["label"] if snapshot["label"] in _LABELS else "other",
        "toggle": snapshot["toggle"] if snapshot["toggle"] in _TOGGLES else "other",
        "typed_enabled": snapshot["typed"],
        "users": snapshot["users"],
        "assistants": snapshot["assistants"],
        "live": snapshot["live"],
        "tasks": sorted(status for _, status in snapshot["tasks"]),
        "marker_names": names,
    }


def _marker_value(lines: list[str], name: str, field: str) -> object:
    for line in lines:
        if line.startswith(f"[{name}] "):
            value = json.loads(line.split("] ", 1)[1])
            if field in value:
                return value[field]
    return None


async def _synthesize_clips() -> dict[str, str]:
    from hermes_realtime.providers.kokoro import KokoroSynthesizer

    synthesizer = KokoroSynthesizer(voice="am_michael", language="en-us")
    clips: dict[str, str] = {}
    try:
        for name, text in _CLIPS.items():
            pcm = b"".join(
                [chunk.audio.pcm async for chunk in synthesizer.synthesize(text, f"clip_{name}")]
            )
            silence = b"\x00\x00" * (48_000 // 2)
            clips[name] = base64.b64encode(silence + pcm + silence).decode("ascii")
    finally:
        await synthesizer.close()
    return clips


async def _rehearse(run_dir: Path, model: str, recall_trials: int = 0) -> int:
    rehearsal = Rehearsal(run_dir, model, recall_trials)
    for directory in (rehearsal.home, rehearsal.logs_dir, rehearsal.state_dir):
        directory.mkdir(parents=True, exist_ok=True)
    left: dict[str, object] = {}
    try:
        try:
            rehearsal.containment = Containment()
        except OSError:
            rehearsal.containment = None  # Reported in the summary; survivors still counted.
        rehearsal.clips = await _synthesize_clips()
        await rehearsal.setup()
        for step in (
            rehearsal.gate,
            rehearsal.start_and_talk,
            rehearsal.typed,
            rehearsal.delegate,
            rehearsal.interrupt,
            rehearsal.cancel,
            rehearsal.reconnect,
            rehearsal.restart,
            rehearsal.delete,
            rehearsal.reload_inputs,
            rehearsal.cleanup,
        ):
            await step()
    finally:
        left = await rehearsal.close()
        outcomes = {str(record["step"]): record["outcome"] for record in rehearsal.records}
        _emit(
            {
                "name": "summary",
                "outcomes": outcomes,
                "run_dir": run_dir.name,
                "stand_in_requests": dict(rehearsal.stand_in.requests),
                "version": 1,
            }
            | left
        )
    clean = all(outcome == "as_expected" for outcome in outcomes.values())
    return 0 if clean and left.get("processes_left") == {} else 1


def _host_child(stop_file: Path, arguments: list[str]) -> int:
    """The full host, stopped through the same path as Ctrl-C when ``stop_file`` appears.

    Ctrl-C cannot reach a detached process, and a console break ends the host without
    settling anything; so a watcher interrupts the main thread exactly as Ctrl-C does.
    """

    import _thread
    import threading

    from hermes_realtime.host_launcher import main as host_main

    stop_file.unlink(missing_ok=True)

    def watch() -> None:
        while not stop_file.exists():
            time.sleep(0.2)
        _thread.interrupt_main(signal.SIGINT)

    threading.Thread(target=watch, name="rehearsal-stop", daemon=True).start()
    _observe_prompts()
    sys.argv = ["hermes-realtime-host", *arguments]
    return int(host_main())


def _observe_prompts() -> None:
    """Report, without content, whether each Ollama prompt carries the step-3 fact.

    It wraps the adapter's own rendering, so the counts are of exactly what is sent. With
    recall trials, the context question's exact prompt is sent again that many times once
    its own reply has finished, and only the count of replies naming the fact is printed.
    """

    import threading

    from hermes_realtime.providers.ollama import DEFAULT_OLLAMA_NUM_CTX, OllamaStreamingInference

    phrase = os.environ.get("HERMES_REHEARSAL_PHRASE", "")
    if not phrase:
        return
    trials = int(os.environ.get("HERMES_REHEARSAL_RECALL_TRIALS", "0"))
    model = os.environ.get("HERMES_REHEARSAL_MODEL", "")
    render = OllamaStreamingInference._messages
    stream = OllamaStreamingInference.stream
    question: list[list[dict[str, str]]] = []

    def marker(name: str, value: Mapping[str, object]) -> None:
        print(f"[{name}] " + json.dumps(value, separators=(",", ":"), sort_keys=True), flush=True)

    def send(messages: list[dict[str, str]]) -> str:
        body = {
            "model": model,
            "messages": messages,
            "options": {"num_ctx": DEFAULT_OLLAMA_NUM_CTX},
            "stream": False,
        }
        request = urllib.request.Request(
            f"{_OLLAMA}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=120) as response:
            return str(json.loads(response.read())["message"]["content"])

    def observed(snapshot: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = render(snapshot)
        marker("rehearsal-prompt", prompt_observation(messages, phrase))
        # System notes (background work, updates) may follow the question.
        users = [message["content"] for message in messages if message["role"] == "user"]
        question[:] = [messages] if trials and users[-1:] == [_BIRD_QUESTION] else []
        return messages

    async def observed_stream(self: Any, snapshot: Any, *, turn_id: str) -> AsyncIterator[str]:
        inner = stream(self, snapshot, turn_id=turn_id)
        try:
            async for segment in inner:
                yield segment
        finally:
            # Closing this stream closes the adapter's, exactly as without the wrapper.
            await inner.aclose()
        if question:
            messages = question.pop()
            threading.Thread(
                target=lambda: marker(
                    "rehearsal-recall", recall_trials(messages, phrase, trials, send)
                ),
                name="rehearsal-recall",
                daemon=True,
            ).start()

    OllamaStreamingInference._messages = staticmethod(observed)
    OllamaStreamingInference.stream = observed_stream


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["--host-child"] and arguments[2:3] == ["--"]:
        return _host_child(Path(arguments[1]), arguments[3:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ollama-model", help="the host's foreground model, from ollama list")
    parser.add_argument("--run-dir", type=Path, help="where to keep the run (default: temp)")
    parser.add_argument(
        "--recall-trials",
        type=int,
        choices=range(0, 101),
        default=0,
        metavar="N",
        help="resend each context question's exact prompt N times and count recall (0-100)",
    )
    args = parser.parse_args(arguments)
    missing = preflight(args.ollama_model)
    if missing:
        # A listener already on 7880, such as an orphaned server, is a failure, not a skip.
        occupied = "livekit_port_in_use" in missing
        record: dict[str, object] = {
            "name": "preflight",
            "notes": {"missing": missing},
            "outcome": "failed" if occupied else "not_run",
            "step": "preflight",
            "version": 1,
        }
        if occupied:
            record["category"] = "livekit_port_in_use"
        _emit(record)
        return 1 if occupied else 0
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="hermes-mvp-rehearsal-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_rehearse(run_dir.resolve(), args.ollama_model, args.recall_trials))


if __name__ == "__main__":
    sys.exit(main())
