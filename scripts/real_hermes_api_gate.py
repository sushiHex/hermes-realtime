"""Exercise hermes-realtime against the installed Hermes API adapter."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from gateway.config import PlatformConfig  # type: ignore[import-not-found]
from gateway.platforms.api_server import APIServerAdapter  # type: ignore[import-not-found]
from real_gate_support import available_port, load_api_key

from hermes_realtime.host_launcher import build_local_host_launcher
from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelEvent,
    ControlCancelPayload,
    WorkDispatchRequestedEvent,
    WorkDispatchRequestedPayload,
)


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


async def main() -> None:
    key = load_api_key(Path.cwd() / ".env")
    api_port = available_port()
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": api_port,
                "key": key,
                "cors_origins": [],
            },
        )
    )
    if not await adapter.connect():
        raise RuntimeError("installed Hermes API adapter did not start")
    approval_seen = asyncio.Event()
    approval_events: list[dict[str, str | int | bool | None]] = []
    run_events: list[tuple[str, str | None]] = []

    def observe_approval(data: dict[str, str | int | bool | None]) -> None:
        approval_events.append(data)
        approval_seen.set()

    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{api_port}",
            bearer=key,
            request_timeout_seconds=120,
        ),
        session_id="session_real_api_gate",
        private_id_factory=iter(
            ("completion_gate", "approval_gate", "cancel_gate")
        ).__next__,
        approval_observer=observe_approval,
        event_observer=lambda event, tool: run_events.append((event, tool)),
    )
    results: list[dict[str, str | bool]] = []
    try:
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
        results.append(
            {
                "accepted": True,
                "status": completion.payload.status.value,
                "task_id": completion.task_id,
            }
        )

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
        results.append(
            {
                "accepted": True,
                "status": approval_terminal.payload.status.value,
                "task_id": approval_terminal.task_id,
            }
        )

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
        results.append(
            {
                "accepted": True,
                "status": interrupted.payload.status.value,
                "task_id": interrupted.task_id,
            }
        )
        if os.environ.get("HERMES_REALTIME_LIVEKIT_LOCAL") == "1":
            await session.close()
            launcher = build_local_host_launcher(
                hermes_api_bearer=key,
                hermes_api_url=f"http://127.0.0.1:{api_port}",
                livekit_url=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"),
                livekit_api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
                livekit_api_secret=os.environ.get(
                    "LIVEKIT_API_SECRET", "local-" + ("x" * 32)
                ),
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
                if parsed.hostname != "127.0.0.1" or not parsed.fragment.startswith(
                    "bootstrap="
                ):
                    raise RuntimeError("full host returned an invalid loopback launch URL")
            finally:
                await launcher.close()
            results.append(
                {
                    "accepted": True,
                    "status": "started_and_closed",
                    "task_id": "task_full_host_gate",
                }
            )
        print(json.dumps({"gate": "passed", "results": results}, sort_keys=True))
    finally:
        await session.close()
        await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
