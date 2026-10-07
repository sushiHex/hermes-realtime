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
- the pinned local LiveKit server;
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
import hashlib
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

from real_gate_support import (
    HERMES_BASELINE,
    PINNED_HERMES,
    available_port,
    provision_pinned_hermes,
)

_PREFIX = "[desktop-mvp-rehearsal] "
_REPOSITORY = Path(__file__).resolve().parents[1]
_UPSTREAM = "https://github.com/NousResearch/hermes-agent.git"
_LIVEKIT = _REPOSITORY / ".tools" / "livekit" / "livekit-server.exe"
_LIVEKIT_SHA256 = "4d60c4043c8c6ff34845727587c7a7f86946d92c390b879ea35ad3793fcbd916"
_LIVEKIT_KEYS = "devkey: local-" + "x" * 32 + "\n"
_OLLAMA = "http://127.0.0.1:11434"
_DETACHED = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
# A bounded marker other code printed: `[name] {json}`, recorded verbatim.
_MARKER_LINE = re.compile(r"\[[a-z][a-z0-9-]{0,47}\] \{.*\}")
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
_REPLY_SECONDS = 180.0
# An abandoned browser session holds the persistent front door until the host reaps it.
_RECONNECT_SECONDS = 420.0
# Decoded remote audio energy above which a reply was audible.
_AUDIBLE_ENERGY = 1e-4
_VIEWPORT = {"width": 1366, "height": 768}
# Long enough for the readiness cue the host plays when voice input becomes ready.
_CUE_SECONDS = 4.0
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
        stream = type(body) is dict and body.get("stream") is True
        if kind == "tool":
            return await self._answer(request, stream, tool=argument)
        if kind == "endless":
            return await self._endless(request, argument)
        if kind == "timed":
            await asyncio.sleep(int(argument))
            return await self._answer(request, stream, text="The long rehearsal task finished.")
        return await self._answer(request, stream, text="Rehearsal step complete.")

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
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                creationflags=0x08000000,
            )
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

    def survivors(self) -> dict[str, int]:
        """What is left running, by image name."""

        return dict(Counter(image for _, image in self.alive()))


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


def _run(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        env=None if env is None else dict(env),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
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
    """The bounded markers among ``lines``, verbatim and in order, and how many were dropped."""

    found: list[str] = []
    for line in lines:
        if len(line) > _MAX_MARKER_CHARS or _MARKER_LINE.fullmatch(line) is None:
            continue
        try:
            json.loads(line.split("] ", 1)[1])  # The pattern already requires an object.
        except ValueError:
            continue
        found.append(line)
    return found[:_MAX_MARKERS], max(len(found) - _MAX_MARKERS, 0)


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
        errors: dict[str, int] = {}
        for log in self._logs:
            log.poll()
            new = log.lines[self._start.get(id(log), 0) :]
            lines, extra = markers(new)
            found.extend(f"{log.source}: {line}" for line in lines)
            dropped += extra
            tracebacks = sum("Traceback (most recent call last)" in line for line in new)
            if tracebacks:
                errors[log.source] = errors.get(log.source, 0) + tracebacks
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
        if errors:
            record["tracebacks"] = errors
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
    energy += (sum / window_.length) * 0.05;
  }, 50);
  window.__rehearsalEnergy = async () => {
    if (meter !== null) await meter.resume();
    return energy;
  };
  window.__rehearsalTasks = [];
  const record = (item) => {
    if (item instanceof HTMLElement && item.dataset.operation === "task" && item.dataset.status) {
      window.__rehearsalTasks.push([item.dataset.taskId, item.dataset.status, performance.now()]);
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
    if not _LIVEKIT.is_file() or hashlib.sha256(_LIVEKIT.read_bytes()).hexdigest() != (
        _LIVEKIT_SHA256
    ):
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
        missing.append("livekit_port_free")
    return missing


class Rehearsal:
    def __init__(self, run_dir: Path, model: str) -> None:
        self.run_dir = run_dir
        self.model = model
        self.home = run_dir / "home"
        self.logs_dir = run_dir / "logs"
        self.state_dir = run_dir / "state"
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

    async def last_assistant_has(self, word: str) -> bool:
        text = await self.page.evaluate(
            "() => { const items = [...document.querySelectorAll("
            "'#transcript li[data-role=assistant]')].filter("
            "(item) => item.dataset.partialTranscript !== 'true');"
            " return items.length ? items[items.length - 1].textContent : ''; }"
        )
        return type(text) is str and word.casefold() in text.casefold()

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

    async def _install_wheel(self, step: Step) -> dict[str, object]:
        out = self.run_dir / "dist"
        built = await asyncio.to_thread(
            _run,
            ["uv", "build", "--wheel", "--out-dir", str(out)],
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            cwd=_REPOSITORY,
        )
        wheels = sorted(out.glob("hermes_realtime-*-py3-none-any.whl"))
        if built.returncode != 0 or len(wheels) != 1:
            raise RuntimeError("the candidate wheel did not build")
        python = str(self.hermes_python)
        before = await asyncio.to_thread(_run, ["uv", "pip", "freeze", "--python", python])
        installed = await asyncio.to_thread(
            _run,
            [
                "uv", "pip", "install", "--reinstall-package", "hermes-realtime",
                "--python", python, str(wheels[0]),
            ],
        )
        if installed.returncode != 0:
            raise RuntimeError("the candidate wheel did not install")
        after = await asyncio.to_thread(_run, ["uv", "pip", "freeze", "--python", python])
        old = {line for line in before.stdout.splitlines() if "hermes-realtime" not in line}
        new = {line for line in after.stdout.splitlines() if "hermes-realtime" not in line}
        return {
            "wheel_version": wheels[0].name.split("-")[1],
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
        step.notes["api_loopback_only"] = _loopback_only(listeners(self.api_port))
        step.notes["companion_loopback_only"] = _loopback_only(listeners(self.companion_port))
        if not (step.notes["api_loopback_only"] and step.notes["companion_loopback_only"]):
            step.differ("not_loopback")
        self.processes.observe()
        self.ready["gateway"] = True

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
                    str(_REPOSITORY / "scripts" / "real_hermes_api_gate.py"),
                    "--hermes-home",
                    str(self.home),
                    "--hermes-api-url",
                    f"http://127.0.0.1:{self.api_port}",
                ],
                env=_isolated(self.home),
                cwd=_REPOSITORY,
                timeout=1200,
            )
            step.time("gate_s", started)
            lines = completed.stdout.splitlines()
            step.notes["stdout_lines"] = len(lines)
            step.notes["stderr_bytes"] = len(completed.stderr.encode())
            line = lines[-1] if lines else ""
            if line.startswith("[real-hermes-gate] "):
                step.notes["gate"] = line
                step.fail("gate_refused")
            else:
                try:
                    record = json.loads(line)
                except ValueError:
                    record = None
                if type(record) is not dict or record.get("gate") != "passed":
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
            await self._start_livekit(step)
            step.time("livekit_ready_s", started)
            await self.start_host()
            step.time("host_ready_s", started)
            start_markers, _ = markers(step.lines("host"))
            if any(line.startswith("[voice-tail] ") for line in start_markers):
                step.differ("voice_tail_at_first_start")
            await self._start_browser()
            step.time("browser_ready_s", started)
            step.timings["connect_to_listening_s"] = await self.connect()
            await self._readiness_cue(step, "connect")
            # What the page plays while nothing is said: the floor for "audible" and "stale".
            idle = await self.energy()
            await asyncio.sleep(3)
            step.notes["idle_audio_energy"] = round(await self.energy() - idle, 6)
            step.notes["session_model"] = await self.page.evaluate(
                "() => Object.fromEntries(['model-provider','model-name','stt-provider',"
                "'stt-model','tts-provider','tts-model'].map((id) => "
                "[id, (document.getElementById(id)?.textContent ?? '').trim()]))"
            )
            before = await self.snapshot()
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

    async def _start_livekit(self, step: Step) -> None:
        log = self.logs_dir / "livekit.log"
        self.processes.spawn(
            "livekit",
            [str(_LIVEKIT), "--dev", "--bind", "127.0.0.1"],
            env=os.environ | {"LIVEKIT_KEYS": _LIVEKIT_KEYS},
            log=log,
        )
        async with asyncio.timeout(30):
            while _http("http://127.0.0.1:7880/") != (200, b"OK"):
                await asyncio.sleep(0.2)
        step.notes["livekit_signaling_loopback_only"] = listeners(7880) == ["127.0.0.1"]

    def _host_argv(self) -> list[str]:
        return [
            "uv", "run", "--frozen", "--extra", "local", "python",
            str(Path(__file__).resolve()),
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
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            cwd=_REPOSITORY,
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
            sent = time.monotonic()
            await self.type("task: Reply with one sentence confirming the rehearsal delegation.")
            task_id, sequence = await self.track_task(ids, sent, step, 300)
            if task_id is not None:
                sequence = await self.observed_sequence(task_id) or sequence
            step.notes["states"] = sequence
            if sequence[:1] != ["active"] or sequence[-1:] != ["completed"]:
                step.differ("state_sequence")
            try:
                await self.wait(
                    lambda now: now["results"] > before["results"]
                    or now["assistants"] > before["assistants"] + 0,
                    60,
                    "result_reported",
                )
                step.notes["result_reported"] = True
            except TimeoutError:
                step.notes["result_reported"] = False
                step.differ("result_not_reported")
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
            await asyncio.sleep(2)
            step.notes["model_streams_open"] = self.stand_in.open["cancellation"]

    # -- session step 7 --

    async def reconnect(self) -> None:
        async with self.step("7", "reconnect_browser") as step:
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            # Whether the fact from step 3 is still among the heard messages the host keeps.
            step.notes["earlier_fact_in_window"] = self._tail_count() > 0
            await self._restate(step)
            await self.page.locator("#session-toggle").click()
            await self.wait(
                lambda now: now["toggle"] == "Connect" and not now["typed"], 60, "stopped"
            )
            step.stage = "stop_connect"
            step.timings["stop_connect_to_listening_s"] = await self.connect()
            await self._after_reconnect(step, "stop_connect")
            step.stage = "reload"
            await self.page.reload()
            await self.page.wait_for_load_state("networkidle")
            try:
                step.timings["reload_connect_to_listening_s"] = await self.connect()
            except TimeoutError:
                # The page left without stopping, so its session may still hold the door.
                step.notes["reload_connect_first"] = _page_categories(await self.snapshot())
                step.differ("reload_connect_refused")
                step.timings["reload_connect_recovered_s"] = await self.connect_until(
                    _RECONNECT_SECONDS
                )
            await self._after_reconnect(step, "reload_connect")
            await self._context_check(step, "context_carried_over")

    async def _readiness_cue(self, step: Step, name: str) -> None:
        """Let the host's readiness cue, played once voice input is ready, finish; record it."""

        energy = await self.energy()
        await asyncio.sleep(_CUE_SECONDS)
        step.notes[f"{name}_readiness_cue"] = (await self.energy() - energy) > _AUDIBLE_ENERGY

    async def _after_reconnect(self, step: Step, name: str) -> None:
        """No stale audio and no duplicated transcript entries after a reconnect."""

        await self._readiness_cue(step, name)
        before = await self.snapshot()
        energy = await self.energy()
        await asyncio.sleep(3)
        after = await self.snapshot()
        heard = await self.energy() - energy
        stale = (
            heard > _AUDIBLE_ENERGY
            and after["assistants"] == before["assistants"]
            and after["live"] == before["live"] == 0
        )
        step.notes[f"{name}_stale_audio"] = stale
        step.notes[f"{name}_audio_energy"] = round(heard, 6)
        if stale:
            step.differ("stale_audio")
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
        await self.type("What is my favorite bird? Answer with one word.")
        await self.reply(before)
        return await self.last_assistant_has(self.phrase)

    async def _restate(self, step: Step) -> None:
        """Restate the fact a later context question depends on.

        The host keeps the last 16 heard messages as context, one per spoken sentence, so a
        long reply pushes an older fact out; restating it here tests carry-over, not that bound.
        """

        step.stage = "restate"
        before = await self.snapshot()
        await self.type(f"Please keep in mind that my favorite bird is the {self.phrase}.")
        await self.reply(before)

    async def _context_check(self, step: Step, name: str) -> None:
        """Is the fact in the context the host carried over, and did the model use it?

        The voice tail holds exactly the context the host keeps and restores, so the first
        answer is mechanical. The second depends on the foreground model, and is recorded.
        """

        carried = self._tail_count() > 0
        step.notes[name] = carried
        if not carried:
            step.differ("context_lost")
        step.notes[f"{name}_answered"] = await self._ask_phrase()

    def _tail_count(self) -> int:
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
            await self._restate(step)
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
            step.timings["kill_s"] = round(self.processes.kill("host"), 2)
            self.ready["host"] = False
            await asyncio.sleep(2)
            step.notes["work_running_after_crash"] = self.stand_in.open["restart"]
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
            await asyncio.sleep(2)
            step.notes["work_running_after_settlement"] = self.stand_in.open["restart"]
            if self.stand_in.open["restart"]:
                step.fail("work_left_running")
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
            await self._context_check(step, "context_resumed")

    # -- M3: deletion, which the runbook has no session step for --

    async def delete(self) -> None:
        async with self.step("delete", "delete_voice_conversation") as step:
            step.notes["runbook_step"] = None
            if not self.ready["browser"]:
                step.skip("no_browser")
                return
            await self.ensure_connected(step)
            step.notes["phrase_before"] = await self._phrase_counts()
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
            if any(after.values()):
                step.fail("phrase_retained")
            before = await self.snapshot()
            await self.type("What is my favorite bird? Answer with one word.")
            await self.reply(before)
            step.notes["phrase_in_next_reply"] = await self.last_assistant_has(self.phrase)
            if step.notes["phrase_in_next_reply"]:
                step.fail("phrase_in_next_reply")

    async def _phrase_counts(self) -> dict[str, int]:
        """Where the conversation's unique phrase still is, as counts."""

        page = await self.page.evaluate(
            "(word) => (document.querySelector('#transcript')?.textContent ?? '')"
            ".toLowerCase().split(word).length - 1",
            self.phrase,
        )
        return {
            "page": int(page),
            "voice_tail": self._tail_count(),
            "hermes_messages": self._db_count(),
        }

    def _db_count(self) -> int:
        database = self.home / "state.db"
        if not database.exists():
            return 0
        uri = f"{database.as_uri()}?mode=ro"
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
            row = db.execute(
                "SELECT count(*) FROM messages WHERE lower(content) LIKE ?",
                (f"%{self.phrase}%",),
            ).fetchone()
        return int(row[0])

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
            "processes_left": self.processes.survivors(),
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


async def _rehearse(run_dir: Path, model: str) -> int:
    rehearsal = Rehearsal(run_dir, model)
    for directory in (rehearsal.home, rehearsal.logs_dir, rehearsal.state_dir):
        directory.mkdir(parents=True, exist_ok=True)
    left: dict[str, object] = {}
    try:
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
    sys.argv = ["hermes-realtime-host", *arguments]
    return int(host_main())


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["--host-child"] and arguments[2:3] == ["--"]:
        return _host_child(Path(arguments[1]), arguments[3:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ollama-model", help="the host's foreground model, from ollama list")
    parser.add_argument("--run-dir", type=Path, help="where to keep the run (default: temp)")
    args = parser.parse_args(arguments)
    missing = preflight(args.ollama_model)
    if missing:
        _emit(
            {
                "name": "preflight",
                "notes": {"missing": missing},
                "outcome": "not_run",
                "step": "preflight",
                "version": 1,
            }
        )
        return 0
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="hermes-mvp-rehearsal-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_rehearse(run_dir.resolve(), args.ollama_model))


if __name__ == "__main__":
    sys.exit(main())
