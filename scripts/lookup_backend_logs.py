"""Bounded private child output; delegates all process ownership to Processes."""

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


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
        for byte in kept:
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
            finally:
                witness._pending.clear()

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
        return {name: capture.finish() for name, capture in self.captures.items()}
