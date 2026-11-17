"""The native run witness read ends at one overall deadline, whatever the server does (#571)."""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from scripts import native_run_witness
from scripts.native_run_witness import NativeRunWitness

RUN = "run_" + "a" * 32
_DEADLINE = 1.0
_STATUS = json.dumps(
    {"object": "hermes.run", "run_id": RUN, "session_id": RUN, "status": "cancelled"}
).encode()


class _Server:
    """One loopback HTTP peer that answers in a chosen, possibly hostile, way."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.stop = threading.Event()
        self.client_gone = threading.Event()
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        connection, _ = self._listener.accept()
        with connection:
            connection.settimeout(0.05)
            request = b""
            while b"\r\n\r\n" not in request and not self.stop.is_set():
                try:
                    request += connection.recv(4096)
                except TimeoutError:
                    continue
            length = b"Content-Length: " + str(len(_STATUS)).encode() + b"\r\n"
            if self.mode == "trickle_body_connection_close":
                # http.client detaches a will-close response's socket from its connection.
                headers = (
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + length
                    + b"Connection: close\r\n\r\n"
                )
            elif self.mode == "trickle_body_http10":
                # No length at all: the body ends only when the server closes.
                headers = b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n"
            else:
                headers = (
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" + length + b"\r\n"
                )
            if self.mode == "complete":
                connection.sendall(headers + _STATUS)
                return
            if self.mode == "stall":
                self._wait_for_client_or_stop(connection)
                return
            body_only = self.mode.startswith("trickle_body")
            prefix = headers if body_only else b""
            trickled = _STATUS if body_only else headers + _STATUS
            try:
                connection.sendall(prefix)
                # One byte every 0.1 s: never idle long enough for a per-socket timeout.
                for index in range(len(trickled)):
                    if self.stop.wait(0.1):
                        return
                    connection.sendall(trickled[index : index + 1])
            except OSError:
                self.client_gone.set()

    def _wait_for_client_or_stop(self, connection: socket.socket) -> None:
        while not self.stop.is_set():
            try:
                if connection.recv(1) == b"":
                    self.client_gone.set()
                    return
            except TimeoutError:
                continue
            except OSError:
                self.client_gone.set()
                return

    def close(self) -> None:
        self.stop.set()
        self._thread.join(timeout=10)
        self._listener.close()


def _profile(tmp_path: Path) -> Path:
    db = sqlite3.connect(tmp_path / "state.db")
    db.executescript(
        "CREATE TABLE sessions(id TEXT,source TEXT);"
        "CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,"
        "content TEXT,tool_calls TEXT,tool_call_id TEXT,tool_name TEXT,active INTEGER);"
    )
    db.commit()
    db.close()
    return tmp_path


def _read_in_thread(subject: NativeRunWitness) -> tuple[threading.Thread, list[object]]:
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(subject.read())
        except BaseException as error:  # the outcome is asserted by the test
            outcome.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


@pytest.mark.parametrize(
    "mode",
    [
        "trickle_headers",
        "trickle_body",
        "trickle_body_connection_close",
        "trickle_body_http10",
        "stall",
    ],
)
def test_a_slow_status_response_ends_the_read_itself_at_the_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    monkeypatch.setattr(native_run_witness, "READ_DEADLINE_SECONDS", _DEADLINE, raising=False)
    server = _Server(mode)
    subject = NativeRunWitness(
        "secret", RUN, profile=_profile(tmp_path), api_origin=f"http://127.0.0.1:{server.port}"
    )
    started = time.monotonic()
    thread, outcome = _read_in_thread(subject)
    try:
        # The reading thread itself settles near the deadline, not after the server relents.
        thread.join(timeout=_DEADLINE + 2)
        settled_after = time.monotonic() - started
        assert not thread.is_alive(), f"witness read still running after {settled_after:.1f}s"
    finally:
        server.close()
        thread.join(timeout=15)
    assert len(outcome) == 1
    error = outcome[0]
    assert type(error) is ValueError and str(error) == "witness_deadline"
    assert '"category": "witness_deadline"' in capsys.readouterr().out


def test_the_database_read_shares_the_one_deadline_and_is_not_restarted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(native_run_witness, "READ_DEADLINE_SECONDS", _DEADLINE, raising=False)
    profile = _profile(tmp_path)
    subject = NativeRunWitness("secret", RUN, profile=profile, api_origin="http://127.0.0.1:1")

    def slow_status() -> dict:
        time.sleep(_DEADLINE * 0.7)  # most of the budget goes on the status request
        return {"object": "hermes.run", "run_id": RUN, "session_id": RUN, "status": "completed"}

    subject._status = slow_status  # type: ignore[method-assign]
    holder = sqlite3.connect(profile / "state.db")
    holder.execute("BEGIN EXCLUSIVE")  # the database stays locked past the deadline
    try:
        started = time.monotonic()
        with pytest.raises(ValueError, match="witness_deadline"):
            subject.read()
        elapsed = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()
    # The lock wait is cut at the overall deadline, not given a fresh two-second wait (which
    # would end near 2.7 s). SQLite's busy wait overshoots its timeout by up to about 0.5 s
    # on Windows, so the margin allows that and no more.
    assert elapsed < _DEADLINE + 1.0
    assert '"category": "witness_deadline"' in capsys.readouterr().out


def test_a_prompt_status_response_still_reads_within_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native_run_witness, "READ_DEADLINE_SECONDS", _DEADLINE, raising=False)
    server = _Server("complete")
    subject = NativeRunWitness(
        "secret", RUN, profile=_profile(tmp_path), api_origin=f"http://127.0.0.1:{server.port}"
    )
    try:
        summary = subject.read()
    finally:
        server.close()
    # Cancelled before any session existed: a verified result with no tool-use claim.
    assert summary["terminal"] == "cancelled"
    assert summary["toolTraceAvailable"] is False
