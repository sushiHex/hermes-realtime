"""Qualify ADR 0003 steps 1-3 against the pinned Hermes: work never starts twice, and a
restart stops whatever a crash left running.

One unattended command:

    uv run python scripts/qualify_hermes_continuity.py

It provisions the baseline Hermes, runs it with a throwaway home and key, and points it at a
stand-in model served by this process. The stand-in streams forever, so a run keeps working
until something stops it. Each scenario crashes a real host process, restarts it on the same
run record, and compares what the record held and what the restart settled with two
independent witnesses:

- Hermes's own durable admissions: how many runs it admitted, and how each one ended;
- the work itself: every dispatch's objective carries a nonce, and the stand-in counts the
  model requests started for it and those still open.

The scenarios are:

- ``crash_before_acknowledgment``: Hermes admitted the run, the host died before recording it;
- ``crash_after_acknowledgment``: the host died with the run recorded as admitted;
- ``hermes_restart``: as the first, and Hermes died too, so the replay meets a restarted Hermes.

Hermes gets a throwaway home in place of the user's, so it can find no credential, and the
output is counts only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import secrets
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from real_gate_support import (
    PINNED_HERMES,
    available_port,
    installed_hermes_identity,
    provision_pinned_hermes,
)

_PREFIX = "[hermes-continuity] "
_READY_PREFIX = "[hermes-continuity-ready] "
_NONCE = re.compile(r"continuity-[0-9a-f]{16}")
_CRASHED = 75
_SETTLE_SECONDS = 60
# How long a late second admission or model request is given to appear before the census.
_LATE_SECONDS = 5
_CHUNK = (
    b'data: {"id":"stand-in","object":"chat.completion.chunk","created":0,"model":"stand-in",'
    b'"choices":[{"index":0,"delta":{"content":"."},"finish_reason":null}]}\n\n'
)

_SETTLED = {
    "settlement": {"stopped": 1, "unknown": 0},
    "work": {"started": 1, "running": 0},
    "remaining": {"admitted": 0, "pending": 0},
}
# One admission each, ended by the restart's stop, or by Hermes's own restart.
_EXPECTED: dict[str, dict[str, object]] = {
    "crash_before_acknowledgment": _SETTLED
    | {"recorded": {"admitted": 0, "pending": 1}, "admissions": ["cancelled"]},
    "crash_after_acknowledgment": _SETTLED
    | {"recorded": {"admitted": 1, "pending": 0}, "admissions": ["cancelled"]},
    "hermes_restart": _SETTLED
    | {"recorded": {"admitted": 0, "pending": 1}, "admissions": ["interrupted"]},
}


class _StandInModel:
    """An OpenAI-compatible model that never finishes, and counts the work it is asked for."""

    def __init__(self) -> None:
        self._started: Counter[str] = Counter()
        self._running: Counter[str] = Counter()
        # Requests naming no single dispatch: refused, and counted so they cannot hide work.
        self.unattributed = 0
        self._runner: Any = None

    async def start(self) -> str:
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._complete)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = available_port()
        await web.TCPSite(self._runner, "127.0.0.1", port).start()
        return f"http://127.0.0.1:{port}/v1"

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def work(self, nonce: str) -> dict[str, int]:
        return {"started": self._started[nonce], "running": self._running[nonce]}

    async def wait_until(self, nonce: str, *, timeout: float, **counts: int) -> None:
        async with asyncio.timeout(timeout):
            while any(self.work(nonce)[name] != count for name, count in counts.items()):
                await asyncio.sleep(0.05)

    async def _complete(self, request: Any) -> Any:
        from aiohttp import web

        nonces = set(_NONCE.findall(await request.text()))
        if len(nonces) != 1:
            self.unattributed += 1
            return web.Response(status=400)
        (nonce,) = nonces
        self._started[nonce] += 1
        self._running[nonce] += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            await response.prepare(request)
            while True:
                await response.write(_CHUNK)
                await asyncio.sleep(0.2)
        except ConnectionResetError:
            # The caller hung up: whatever was doing this work has stopped. Returning, rather
            # than raising, keeps the expected hang-up out of the server's error log.
            return response
        finally:
            self._running[nonce] -= 1


def _census(record: Path) -> dict[str, int]:
    """How many runs a record holds, by kind; never what they are."""

    if not record.exists():
        return {"admitted": 0, "pending": 0}
    document = json.loads(record.read_text(encoding="utf-8"))
    return {"admitted": len(document["admitted"]), "pending": len(document["pending"])}


def _admissions(home: Path) -> list[str]:
    """How each run Hermes admitted under a key ended, in admission order, from its own store.

    Every dispatch carries a key when Hermes retains keys durably, so this is every run.
    """

    database = home / "runs_idempotency.db"
    if not database.exists():
        return []
    with contextlib.closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT status_json FROM run_idempotency ORDER BY created_at").fetchall()
    return [json.loads(status)["status"] for (status,) in rows]


def _passed(
    observed: dict[str, dict[str, object]], unattributed: int, hermes: dict[str, object]
) -> bool:
    """Only the qualified baseline can pass: evidence from another Hermes qualifies nothing."""
    return hermes["baseline"] is True and observed == _EXPECTED and unattributed == 0


class _Hermes:
    """The pinned Hermes API server in its own process, which a scenario may kill."""

    def __init__(self, python: Path, home: Path, key: str) -> None:
        self._python = python
        self.home = home
        self.key = key
        self._process: asyncio.subprocess.Process | None = None
        self._drain: asyncio.Task[None] | None = None
        self.url = ""
        self.version = ""

    async def start(self) -> None:
        from hermes_realtime.providers.codex_app_server import _subscription_environment

        port = available_port()
        # The user's homes are replaced, not inherited, so no credential store is reachable.
        homes = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HERMES_HOME", "CODEX_HOME")
        environment = (
            _subscription_environment(os.environ)
            | dict.fromkeys(homes, str(self.home))
            | {"HERMES_CONTINUITY_KEY": self.key}
        )
        log = (self.home / "hermes-stderr.log").open("ab")
        with log:
            self._process = await asyncio.create_subprocess_exec(
                str(self._python),
                __file__,
                "--hermes",
                str(port),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
            )
        assert self._process.stdout is not None
        async with asyncio.timeout(180):
            while True:
                line = (await self._process.stdout.readline()).decode("utf-8", "replace")
                if not line:
                    raise RuntimeError("the pinned Hermes exited before it was ready")
                if line.startswith(_READY_PREFIX):
                    break
        self.version = json.loads(line.removeprefix(_READY_PREFIX))["version"]
        self.url = f"http://127.0.0.1:{port}"
        # Hermes may keep writing to stdout; an undrained pipe would stall it.
        self._drain = asyncio.create_task(self._process.stdout.read())

    async def kill(self) -> None:
        """End Hermes the way a crash does: no shutdown, nothing settled."""

        if self._process is not None and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        if self._drain is not None:
            with contextlib.suppress(Exception):
                await self._drain


async def _host(hermes: _Hermes, record: Path, *arguments: str) -> tuple[int, str]:
    """Run one realtime host process on ``record``; return its exit code and last line."""

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        "--host",
        str(record),
        *arguments,
        env=os.environ | {"HERMES_CONTINUITY_URL": hermes.url, "HERMES_CONTINUITY_KEY": hermes.key},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 180)
    except TimeoutError:
        process.kill()  # A hung host would hold the record lock and the key.
        await process.wait()
        raise
    lines = stdout.decode("utf-8", "replace").strip().splitlines()
    assert process.returncode is not None
    return process.returncode, lines[-1] if lines else ""


async def _scenario(
    name: str, nonce: str, model: _StandInModel, hermes: _Hermes, directory: Path
) -> dict[str, object]:
    record = directory / f"{name}.json"
    crash = "after" if name == "crash_after_acknowledgment" else "before"
    code, _ = await _host(hermes, record, f"--crash={crash}", f"--nonce={nonce}")
    if code != _CRASHED:
        raise RuntimeError(f"{name}: the host did not crash where the scenario requires")
    # The crash must have left real work running for the restart to find.
    await model.wait_until(nonce, started=1, running=1, timeout=_SETTLE_SECONDS)
    recorded = _census(record)
    if name == "hermes_restart":
        await hermes.kill()
        await hermes.start()
    code, line = await _host(hermes, record)
    if code != 0:
        raise RuntimeError(f"{name}: the restarted host did not settle its record")
    # A stopped run hangs up on the model; allow Hermes the time a stop may take.
    with contextlib.suppress(TimeoutError):
        await model.wait_until(nonce, running=0, timeout=_SETTLE_SECONDS)
    return {"recorded": recorded, "settlement": json.loads(line), "remaining": _census(record)}


async def _qualify(python: Path) -> None:
    observed: dict[str, dict[str, object]] = {}
    evidence: dict[str, object] = {"version": 1}
    model = _StandInModel()
    with tempfile.TemporaryDirectory(prefix="hermes-continuity-") as temporary:
        directory = Path(temporary)
        home = directory / "hermes-home"
        home.mkdir()
        hermes = _Hermes(python, home, secrets.token_urlsafe(32))
        try:
            config = {
                "model": {"provider": "custom", "base_url": await model.start(), "default": "x"},
                "platform_toolsets": {"api_server": []},
                # Titling a run's session is the one auxiliary model call a run that never
                # finishes makes; without it, every model request is the dispatched work.
                "auxiliary": {"title_generation": {"enabled": False}},
            }
            (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
            await hermes.start()
            identity = installed_hermes_identity(hermes.version, PINNED_HERMES / "source")
            evidence["hermes"] = identity
            # Each scenario's nonce, and where its admissions begin in Hermes's store.
            marks: list[tuple[str, str, int]] = []
            for name in _EXPECTED:
                nonce = f"continuity-{secrets.token_hex(8)}"
                marks.append((name, nonce, len(_admissions(home))))
                observed[name] = await _scenario(name, nonce, model, hermes, directory)
            # Both witnesses are read once, at the end, so a late duplicate is still counted.
            await asyncio.sleep(_LATE_SECONDS)
            admissions = _admissions(home)
            ends = [start for _, _, start in marks[1:]] + [len(admissions)]
            for (name, nonce, start), end in zip(marks, ends, strict=True):
                observed[name] |= {"admissions": admissions[start:end], "work": model.work(nonce)}
            evidence["unattributed"] = model.unattributed
            evidence["passed"] = _passed(observed, model.unattributed, identity)
        except BaseException as error:
            evidence["failure"] = type(error).__name__
            raise
        finally:
            evidence["scenarios"] = observed
            print(_PREFIX + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)
            await hermes.kill()
            await model.close()
    if evidence["passed"] is not True:
        raise SystemExit(1)


async def _serve_hermes(port: int) -> None:
    """Run inside the pinned Hermes environment, with ``HERMES_HOME`` already isolated."""

    import hermes_cli  # type: ignore[import-not-found]
    from gateway.config import PlatformConfig  # type: ignore[import-not-found]
    from gateway.platforms.api_server import APIServerAdapter  # type: ignore[import-not-found]

    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": port,
                "key": os.environ.pop("HERMES_CONTINUITY_KEY"),
                "cors_origins": [],
            },
        )
    )
    if not await adapter.connect():
        raise RuntimeError("the pinned Hermes API server did not start")
    print(_READY_PREFIX + json.dumps({"version": hermes_cli.__version__}), flush=True)
    await asyncio.Event().wait()  # Until killed: this process never shuts down in order.


async def _run_host(record: Path, crash: str | None, nonce: str | None) -> None:
    """One realtime host on ``record``: dispatch and crash, or start, settle and report."""

    from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
    from hermes_realtime.protocol import WorkDispatchRequestedEvent, WorkDispatchRequestedPayload

    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=os.environ["HERMES_CONTINUITY_URL"],
            bearer=os.environ.pop("HERMES_CONTINUITY_KEY"),
            settlement_timeout_seconds=_SETTLE_SECONDS,
        ),
        session_id="session_continuity",
        run_record_path=record,
    )
    await session.start()
    if crash is None:
        settlement = session.restart_settlement
        assert settlement is not None
        await session.close()
        print(json.dumps({"stopped": settlement.stopped, "unknown": settlement.unknown}))
        return
    if crash == "before":
        submit = session._submit_run

        async def submit_then_crash(pending: Any) -> Any:
            await submit(pending)
            os._exit(_CRASHED)  # Hermes answered; the host never recorded what it said.

        session._submit_run = submit_then_crash  # type: ignore[method-assign]
    acknowledgment = await session.dispatch(
        WorkDispatchRequestedEvent(
            event_id="dispatch_continuity",
            session_id="session_continuity",
            sequence=1,
            timestamp=datetime.now(UTC),
            type="work.dispatch.requested",
            task_id="task_continuity",
            utterance_id="utterance_continuity",
            payload=WorkDispatchRequestedPayload(objective=f"Qualification {nonce}: reply."),
        )
    )
    os._exit(_CRASHED if acknowledgment.payload.accepted else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hermes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--host", type=Path, help=argparse.SUPPRESS)
    # Crash before or after the host learns Hermes's acknowledgment.
    parser.add_argument("--crash", choices=("before", "after"), help=argparse.SUPPRESS)
    parser.add_argument("--nonce", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.hermes is not None:
        asyncio.run(_serve_hermes(args.hermes))
    elif args.host is not None:
        asyncio.run(_run_host(args.host, args.crash, args.nonce))
    else:
        asyncio.run(_qualify(provision_pinned_hermes()))


if __name__ == "__main__":
    main()
