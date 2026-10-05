"""Qualify the operator's installed Hermes from the outside, the way the full host meets it.

One command, run with the install's own interpreter while its gateway is running:

    <HERMES_HOME>/hermes-agent/venv/Scripts/python.exe scripts/real_hermes_api_gate.py

``--hermes-home`` defaults to ``HERMES_HOME``, else the installer's default home, and
``--hermes-api-url`` to the full host's default. The gate:

1. refuses unless this interpreter's Hermes is the installer's checkout and its
   hermes-realtime is installed in that checkout's environment, and names both;
2. reads the API key and the companion endpoint from the home's ``.env`` with the full
   host's own loaders;
3. performs a real bridge hello with the companion, which answers only when the running
   gateway discovered, loaded and started the plugin, and requires every capability;
4. dispatches, approves and exactly cancels real work through the host's task session;
5. after the session closes, reads every run Hermes admitted for it back as terminal.

It prints one JSON line: identity, candidate, discovery, behaviors and cleanup, as counts,
kinds and statuses only. A refusal prints one bounded ``[real-hermes-gate]`` line instead.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sysconfig
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

import aiohttp
from real_gate_support import available_port, installed_hermes_identity

from hermes_realtime.companion.host import CompanionEndpoint
from hermes_realtime.host_launcher import (
    DEFAULT_HERMES_API_URL,
    _load_api_bearer,
    _load_voice_companion,
    build_local_host_launcher,
)
from hermes_realtime.integration import (
    HermesApiConfig,
    HermesApiTaskSession,
    LocalHermesBridgeClient,
)
from hermes_realtime.protocol import (
    VOICE_ARCHIVE_CAPABILITY,
    VOICE_REVIEW_CAPABILITY,
    CancelScope,
    ControlCancelEvent,
    ControlCancelPayload,
    WorkDispatchRequestedEvent,
    WorkDispatchRequestedPayload,
)

_MARKER = "[real-hermes-gate] "
# What the full host's archive and review senders ask the companion for.
_CAPABILITIES = (VOICE_ARCHIVE_CAPABILITY, VOICE_REVIEW_CAPABILITY)
_HELLO_TIMEOUT_SECONDS = 30.0
_MAX_STATUS_BYTES = 64 * 1024
# Every status Hermes reports for a run that holds no more work.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})


def default_hermes_home(environ: Mapping[str, str]) -> Path:
    """The home the installer uses: ``HERMES_HOME``, else the platform's default."""

    configured = environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path(environ["LOCALAPPDATA"]) / "hermes"
    return Path.home() / ".hermes"


def installed_identity(home: Path) -> dict[str, object]:
    """Name the install this interpreter runs; refuse an interpreter that is not the install.

    Hermes must be imported from the installer's checkout, ``<home>/hermes-agent``, and
    hermes-realtime must be installed in that checkout's ``venv``.
    """

    import hermes_cli  # type: ignore[import-not-found]

    checkout = (home / "hermes-agent").resolve()
    # hermes_cli sits at the root of the installer's checkout.
    if Path(hermes_cli.__file__).resolve().parents[1] != checkout:
        raise RuntimeError("this interpreter's Hermes is not the install's checkout")
    venv = str(checkout / "venv")
    site = Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv}))
    candidate = importlib.metadata.distribution("hermes-realtime")
    if Path(str(candidate.locate_file(""))).resolve() != site.resolve():
        raise RuntimeError("this interpreter's hermes-realtime is not installed in the install")
    direct_url = candidate.read_text("direct_url.json")
    editable = (
        direct_url is not None
        and json.loads(direct_url).get("dir_info", {}).get("editable") is True
    )
    return {
        "hermes": installed_hermes_identity(hermes_cli.__version__, checkout),
        "candidate": {
            "version": candidate.version,
            "install": "editable" if editable else "wheel",
        },
    }


async def _hello(companion: CompanionEndpoint) -> list[str]:
    """Only a companion the running gateway started answers; it must offer every capability."""

    async with asyncio.timeout(_HELLO_TIMEOUT_SECONDS):
        link = await LocalHermesBridgeClient.connect(
            host="127.0.0.1",
            port=companion.port,
            token=companion.token,
            participant_id="real-hermes-gate",
            capabilities=_CAPABILITIES,
        )
    async with link:
        if link.capabilities != frozenset(_CAPABILITIES):
            raise RuntimeError("the companion does not offer every capability the host asks for")
        return sorted(link.capabilities)


async def _read_back(config: HermesApiConfig, run_ids: frozenset[str]) -> dict[str, object]:
    """Read each run back from Hermes; any that is not terminal is still running work."""

    statuses: Counter[str] = Counter()
    timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
        for run_id in sorted(run_ids):
            async with client.get(
                f"{config.base_url}/v1/runs/{quote(run_id, safe='')}",
                headers={"Authorization": f"Bearer {config.bearer}"},
                allow_redirects=False,
            ) as response:
                body = await response.content.read(_MAX_STATUS_BYTES + 1)
            if len(body) > _MAX_STATUS_BYTES:
                raise RuntimeError("a run status exceeds the body bound")
            run = json.loads(body)
            if (
                type(run) is not dict
                or run.get("run_id") != run_id
                or run.get("status") not in _TERMINAL_STATUSES
            ):
                raise RuntimeError("a run the gate created is not terminal")
            statuses[run["status"]] += 1
    return {"runs": len(run_ids), "statuses": dict(sorted(statuses.items()))}


def _dispatch(
    *,
    event_id: str,
    sequence: int,
    task_id: str,
    utterance_id: str,
    objective: str,
) -> WorkDispatchRequestedEvent:
    return WorkDispatchRequestedEvent(
        event_id=event_id,
        session_id="session_real_api_gate",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        type="work.dispatch.requested",
        task_id=task_id,
        utterance_id=utterance_id,
        payload=WorkDispatchRequestedPayload(objective=objective),
    )


async def qualify(home: Path, api_url: str) -> dict[str, object]:
    """Qualify the install at ``home`` against its running gateway; the passing record."""

    evidence: dict[str, object] = {"stage": "identity", "version": 1}
    passed = False
    try:
        record = installed_identity(home)
        evidence["stage"] = "endpoint"
        env_file = home / ".env"
        config = HermesApiConfig(
            base_url=api_url,
            bearer=_load_api_bearer(env_file),
            request_timeout_seconds=120,
        )
        companion = _load_voice_companion(env_file, os.environ)
        if companion is None:
            raise RuntimeError("the install's .env names no companion endpoint")
        evidence["stage"] = "discovery"
        record["discovery"] = {"capabilities": await _hello(companion)}
        record["behaviors"], run_ids = await _behaviors(config, evidence)
        evidence["stage"] = "cleanup"
        record["cleanup"] = await _read_back(config, run_ids)
        if os.environ.get("HERMES_REALTIME_LIVEKIT_LOCAL") == "1":
            evidence["stage"] = "host"
            record["host"] = await _compose_host(config)
        passed = True
        return {"gate": "passed"} | record
    except BaseException as error:
        evidence["failure"] = type(error).__name__
        raise
    finally:
        if not passed:
            print(_MARKER + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


async def _behaviors(
    config: HermesApiConfig, evidence: dict[str, object]
) -> tuple[dict[str, str], frozenset[str]]:
    """Dispatch, approval and exact cancellation through the host's task session.

    Returns each behavior's terminal status and, once the session has closed, every run
    Hermes admitted for it.
    """

    approval_seen = asyncio.Event()
    approval_events: list[dict[str, str | int | bool | None]] = []
    run_events: list[tuple[str, str | None]] = []

    def observe_approval(data: dict[str, str | int | bool | None]) -> None:
        approval_events.append(data)
        approval_seen.set()

    session = HermesApiTaskSession(
        config=config,
        session_id="session_real_api_gate",
        private_id_factory=iter(
            ("completion_gate", "approval_gate", "cancel_gate")
        ).__next__,
        approval_observer=observe_approval,
        event_observer=lambda event, tool: run_events.append((event, tool)),
    )
    behaviors: dict[str, str] = {}
    try:
        evidence["stage"] = "completion"
        await session.start()
        completion_ack = await session.dispatch(
            _dispatch(
                event_id="dispatch_completion_gate",
                sequence=1,
                task_id="task_completion_gate",
                utterance_id="utterance_completion_gate",
                objective=(
                    "Do not use tools. Return one concise sentence confirming that the "
                    "Hermes realtime API completion gate passed."
                ),
            )
        )
        if not completion_ack.payload.accepted:
            raise RuntimeError("real completion dispatch was rejected")
        completion = await asyncio.wait_for(session.next_update(), timeout=300)
        behaviors["completion"] = completion.payload.status.value

        evidence["stage"] = "approval"
        run_events.clear()
        probe = Path(tempfile.gettempdir()) / f"hermes-approval-probe-{uuid.uuid4().hex}"
        if probe.exists():
            raise RuntimeError("approval probe path unexpectedly exists")
        approval_ack = await session.dispatch(
            _dispatch(
                event_id="dispatch_approval_gate",
                sequence=10,
                task_id="task_approval_gate",
                utterance_id="utterance_approval_gate",
                objective=(
                    "Use the terminal tool to run exactly this command and do not choose an "
                    f"alternative: chmod 777 {probe.as_posix()}. If approval is rejected, "
                    "report that rejection without retrying."
                ),
            )
        )
        if not approval_ack.payload.accepted:
            raise RuntimeError("real approval dispatch was rejected")
        terminal_task = asyncio.create_task(session.next_update())
        approval_task = asyncio.create_task(approval_seen.wait())
        done, _ = await asyncio.wait(
            {terminal_task, approval_task},
            timeout=300,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            terminal_task.cancel()
            approval_task.cancel()
            raise TimeoutError("real approval probe produced no terminal or approval event")
        if terminal_task in done and approval_task not in done:
            approval_task.cancel()
            terminal = terminal_task.result()
            raise RuntimeError(
                "approval probe completed without approval; "
                f"status={terminal.payload.status.value}; events={run_events!r}"
            )
        if len(approval_events) != 1 or approval_events[0].get("actionable") is not True:
            raise RuntimeError("real approval event was not actionable")
        approval_id = approval_events[0].get("approvalId")
        if not isinstance(approval_id, str):
            raise RuntimeError("real approval event lacks public request identity")
        await session.decide_approval(
            approval_id=approval_id,
            sequence=1,
            decision="reject",
        )
        approval_terminal = await asyncio.wait_for(terminal_task, timeout=300)
        if probe.exists():
            raise RuntimeError("approval probe changed the nonexistent target")
        behaviors["approval"] = approval_terminal.payload.status.value

        evidence["stage"] = "cancellation"
        cancel_ack = await session.dispatch(
            _dispatch(
                event_id="dispatch_cancel_gate",
                sequence=20,
                task_id="task_cancel_gate",
                utterance_id="utterance_cancel_gate",
                objective=(
                    "Without using tools, think carefully for several minutes before replying."
                ),
            )
        )
        if not cancel_ack.payload.accepted:
            raise RuntimeError("real cancellation dispatch was rejected")
        cancellation = await session.cancel(
            ControlCancelEvent(
                event_id="cancel_exact_gate",
                session_id="session_real_api_gate",
                sequence=22,
                timestamp=datetime.now(UTC),
                type="control.cancel",
                task_id="task_cancel_gate",
                payload=ControlCancelPayload(
                    scope=CancelScope.TASK,
                    reason="real boundary exact-stop gate",
                ),
            )
        )
        if not cancellation.payload.accepted:
            raise RuntimeError("real exact cancellation was rejected")
        interrupted = await asyncio.wait_for(session.next_update(), timeout=60)
        behaviors["cancellation"] = interrupted.payload.status.value
    finally:
        await session.close()
    return behaviors, session.admitted_run_ids


async def _compose_host(config: HermesApiConfig) -> str:
    """Start and close the Desktop full-host composition against the same gateway."""

    launcher = build_local_host_launcher(
        hermes_api_bearer=config.bearer,
        hermes_api_url=config.base_url,
        livekit_url=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        livekit_api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
        livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", "local-" + ("x" * 32)),
        room_name=f"real-host-gate-{uuid.uuid4().hex[:10]}",
        worker_identity=f"worker_{uuid.uuid4().hex[:16]}",
        browser_port=available_port(),
        inference_provider="codex",
        conversation_profile="natural_v1",
        natural_work_tools=True,
        knowledge_speculation=True,
        knowledge_recovery=True,
        knowledge_budget_seconds=0.35,
        stt_provider="moonshine",
        moonshine_model_tier="medium",
        tts_provider="kokoro",
        allow_unsandboxed_tasks=True,
    )
    try:
        launch_url = await launcher.start()
        parsed = urlsplit(launch_url)
        if parsed.hostname != "127.0.0.1" or not parsed.fragment.startswith("bootstrap="):
            raise RuntimeError("full host returned an invalid loopback launch URL")
    finally:
        await launcher.close()
    return "started_and_closed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hermes-home", type=Path, default=None)
    parser.add_argument("--hermes-api-url", default=DEFAULT_HERMES_API_URL)
    args = parser.parse_args()
    home = args.hermes_home if args.hermes_home is not None else default_hermes_home(os.environ)
    record = asyncio.run(qualify(home, args.hermes_api_url))
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
