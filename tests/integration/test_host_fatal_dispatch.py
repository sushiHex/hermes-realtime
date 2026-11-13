"""A host-fatal exit leaves the run record for the restart to settle and announce (#531).

The real Hermes API session against a fake Hermes that loses both the dispatch response and
its resend, behind the production conversation wiring. The dispatch outcome is unknown, the
task controller latches, and the host ends itself. What matters here is what the exit does
*not* do: it does not close the work session, so nothing stops the run or empties the run
record. The entry written ahead of the request is still there, and the next start replays
it, stops the run it finds, and has something truthful to tell the user.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_realtime.host_launcher import HOST_FATAL_EXIT_STATUS, _restart_announcement
from tests.conversation.test_dispatch_failure_cascade import (
    _TASK_COMMAND,
    _healthy_turn,
    _stack,
)
from tests.integration.test_hermes_api_session import (
    _IdempotentHermes,
    _read_record,
    _record_session,
)


@pytest.mark.asyncio
async def test_a_host_fatal_exit_keeps_the_pending_dispatch_for_the_restart_to_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    hermes = _IdempotentHermes("lose_response", lose_resend=True)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    successor = _record_session(port, record)
    try:
        await session.start()
        async with _stack(
            tmp_path, capsys, session=session, session_id="session_api_1"
        ) as stack:
            generation = await _healthy_turn(stack)
            await stack.say(generation, _TASK_COMMAND)
            await stack.exited()

            assert stack.exit.statuses == [HOST_FATAL_EXIT_STATUS]
            assert stack.host_fatal.cause == "task_control"
            (printed,) = stack.exit.stdout
            assert (
                '[host-fatal-exit] {"cause":"task_control","flush":"done","input":"done",'
                '"version":1}'
            ) in printed.splitlines()

            # The run record still holds the dispatch written ahead of the request, and the
            # exit asked Hermes to stop nothing.
            snapshot = _read_record(record)
            assert type(snapshot) is dict and snapshot["admitted"] == []
            (entry,) = snapshot["pending"]
            assert entry["key"] == hermes.attempts[0][0]
            assert json.loads(hermes.attempts[0][1])["input"] == "Summarize the synthetic report."
            assert hermes.stop_calls == 0
            assert len(hermes.attempts) == 2

            # The process is gone: the kernel releases its lock, and nothing else ran.
            session._release_run_record()
            before_restart = record.read_bytes()

        assert record.read_bytes() == before_restart

        # The next start replays the stored key, learns the run and stops it.
        hermes.lose_resend = False
        await successor.start()
        settlement = successor.restart_settlement
        assert settlement is not None
        assert (settlement.stopped, settlement.unknown) == (1, 0)
        assert hermes.stop_calls == 1
        assert _restart_announcement(settlement) == (
            "I restarted, so background work from before will not resume. "
            "1 task was stopped or had already ended."
        )
        assert _read_record(record) == {"admitted": [], "pending": [], "version": 1}
    finally:
        await successor.close()
        await session._close_client()
        await runner.cleanup()
