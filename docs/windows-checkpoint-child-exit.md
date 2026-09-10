# Windows checkpoint child-exit investigation

Issue [#13](https://github.com/sushiHex/hermes-realtime/issues/13) includes a
five-second child-exit timeout at
`test_external_cli_checkpoint_failures_exit_nonzero_and_reap[ack_eof]`.
This mechanism remains separate from the browser ICU timing observations.

## Findings

The checkpoint test waited for the child to exit before reading its piped stderr.
A child writing more than the pipe can hold blocks until the parent reads it;
the parent can then time out waiting for that child's exit. Python documents this
hazard for [`Popen.wait()`](https://docs.python.org/3.11/library/subprocess.html#subprocess.Popen.wait).

A regression makes the test-owned launcher's close operation write 64 KiB of
synthetic bytes to stderr after the checkpoint failure. With the original wait,
the regression failed with `TimeoutExpired` at the existing five-second authority.
Draining stderr with `communicate(timeout=5.0)` made the same regression pass.
The regression checks the complete synthetic prefix, the original failure
diagnostic, nonzero exit, checkpoint EOF, and process reaping.

This demonstrates a test-harness backpressure defect. It does **not** establish
the cause of the historical hosted failure. Unmodified local `ack_eof` runs
produced only 2,035 stderr bytes and did not time out. The hosted run did not
retain the failing child's stderr. Additional diagnostic output or runner
conditions during that run remain unknown.

## Scope and authority

The implementation changes the checkpoint test harness only. The production
checkpoint transport, host lifecycle, and ICU setup are unchanged.

The completion operation drains stderr and waits for exit under one five-second
timeout. It does not grant separate five-second budgets to reading and waiting.
The async fixture retains and shields the communication task. Cancellation stops
the awaiter without cancelling that task's ownership of the subprocess streams.
Cleanup kills a live child and waits for the existing communication call to end.
If that call timed out, a sequential `communicate()` retry finishes the Windows
reader and reaps the child. Two communication calls never run concurrently.

Settlement and any retry share one five-second cleanup deadline, measured before
the kill. Repeated cancellation does not restart it or abandon the worker. The
fixture closes the parent stderr stream only after communication has settled,
then propagates the original timeout or cancellation. A cleanup deadline failure
is still fatal: it does not emit `cleanup_returned` or close a stream still owned
by active communication. Startup, checkpoint, workflow, and job timeout
authorities are unchanged.

The real-interruption regression runs the actual Windows `communicate()` and
reader against a child held in its close operation. After EOF, an event barrier
pauses reader-thread retirement to expose the ordering deterministically. The
original fixture returned while that thread was still active for a real timeout,
cancellation, and repeated cancellation. The corrected fixture remains pending
until the barrier is released. The regression then checks the original exception,
non-overlapping communication calls, reader retirement, stream closure, and
process reaping. The test's synchronization watchdogs do not change the fixture's
completion or cleanup budgets.

## Local comparison

Before changing the test, ten sequential invocations of each case were observed
on each revision:

| Revision | Case | Passed | Inside `Popen.wait()` | Stderr bytes |
| --- | --- | --- | --- | --- |
| Original candidate `9ce11c34…` | `ack_eof` | 10/10 | 99.131–118.370 ms | 2,035 |
| Original candidate `9ce11c34…` | `wrong_nonce` | 10/10 | 94.205–119.980 ms | 1,376 |
| Accepted main `08300b28…` | `ack_eof` | 10/10 | 105.450–130.471 ms | 2,035 |
| Accepted main `08300b28…` | `wrong_nonce` | 10/10 | 103.033–146.171 ms | 1,376 |

The checkpoint test, `_qualification.py`, `host_launcher.py`, and `uv.lock` were
byte-identical Git blobs across those two revisions. These were direct calls to
the unmodified async test function, with parent-side observation of `Popen`,
`wait`, and stderr reads. Each call launched a fresh child. They were not forty
full-suite runs or independent samples of hosted runner conditions. Local Python
was 3.11.15; the original hosted failure used 3.11.9. No hosted rerun was issued.

The [measurement file](evidence/windows-checkpoint-child-exit-2026-09-09.json)
retains exact source identifiers and per-invocation numbers. `wait_ms` measures
the original synchronous wait call, excluding executor queueing. `total_ms`
includes the complete direct invocation and local temporary-directory cleanup;
it is not timeout headroom.

## Retained observations and bounded repetition

The fixture emits one `[checkpoint-child]` JSON line after cleanup. Ordinary
pytest capture retains it on failure; `-s` exposes passing samples. Tests that
parse the line replay the captured output before asserting, preserving the line
if a later assertion fails. It contains
only a version, fixed case name, PID, outcome flags, exit code, stderr byte
count, parent monotonic event offsets, and fixed child lifecycle names. It contains
no stderr text, command line, environment, checkpoint nonce, or filesystem path.

The event offsets are milliseconds from the parent's process-creation attempt:

- `spawn_returned`: `Popen` returned;
- `checkpoint_received`: the parent received the startup frame;
- `failure_sent`: the parent closed the ACK writer or wrote the invalid ACK;
- `completion_entered` / `completion_returned`: the parent awaited communication
  and exit; their difference includes executor queueing and stderr draining;
- `cleanup_entered` / `cleanup_returned`: the fixture's cleanup interval.

These are parent observations, not child-side timestamps. Closing the ACK writer
does not measure when the child processes EOF. A missing event means that boundary
was not observed. Version 2 retains the byte count returned by either normal
communication or its sequential cleanup drain. A null count means neither returned a measurement;
it does not mean zero bytes. The count after a kill describes bytes recovered
through cleanup, not a naturally completed child.
A killed child's exit code is a cleanup result, not its natural exit status.

The separate inherited progress pipe carries at most four single-byte milestones:
`launcher_close_entered`, `launcher_close_returned`, `host_main_settled`, and
`atexit_entered`. Only those fixed names enter `child_events`; unknown, duplicate,
or reordered bytes yield null. An empty list means no milestone was observed.
The parent samples available bytes without waiting for EOF or adding a reader
thread. The child makes the handle noninheritable before importing the host.

`host_main_settled` observes the test child's `finally` around the real CLI entry
point, including an exception; it does not mean success. `atexit_entered` observes
one callback registered after host imports. It does not prove that other exit
callbacks or interpreter finalization completed. The milestones carry no child
clock measurements and cannot be subtracted from parent offsets.

A controlled child held inside launcher close and another held inside the exit
callback distinguish these boundaries under the unchanged five-second deadline.
Timeout, cancellation, and repeated cancellation tests compare the reported byte
count with bytes actually returned by the owned communication call. The progress
pipe remains independent of stderr and closes even if communication cleanup fails.

From a clean checkout of the candidate, run a bounded Windows comparison:

```powershell
git rev-parse HEAD
git status --short
uv run --frozen python --version
foreach ($sample in 1..10) {
    foreach ($case in @('ack_eof', 'wrong_nonce')) {
        uv run --frozen pytest -q -s "tests/test_qualification_checkpoint.py::test_external_cli_checkpoint_failures_exit_nonzero_and_reap[$case]"
        if ($LASTEXITCODE -ne 0) {
            throw "Checkpoint investigation stopped at sample $sample ($case)."
        }
    }
}
```

Stop on the first failure and retain its exact revision, case, observation, and
test outcome. A new timing sample does not authorize a CI rerun, timeout increase,
or issue closure. Further hosted evidence is required to resolve the historical
child-exit failure.

## Initial candidate validation (`48ce86b…`)

- RED: the injected 64 KiB stderr write timed out at the unchanged five-second
  wait before the drain fix.
- RED: observation assertions failed before timing output existed, and the
  injected-timeout regression exposed an unclosed parent stderr stream.
- GREEN: all 20 checkpoint tests pass after the fix and final fixture refactor.
- Full default Python suite: 3,199 passed, 21 skipped. Skips cover opt-in native,
  browser, and Linux checks, missing optional speech dependencies, and this
  account's unavailable symbolic-link privilege.
- `uv run --frozen ruff check .`: passed.
- `uv run --frozen mypy src`: passed, 76 source files.

Hosted run [34318730626](https://github.com/sushiHex/hermes-realtime/actions/runs/34318730626)
subsequently passed all four jobs on attempt 1 for this initial candidate.
Those results do not qualify a later revision. Issue #13 remains open.

## Communication ownership follow-up (`70f9e7e…`)

- RED: a real timeout, cancellation, and repeated cancellation each let the
  `48ce86b…` fixture return while its Windows reader thread was held at the
  retirement barrier. The observation parser also consumed the diagnostic line.
- GREEN: those cases pass with explicit communication ownership. Cancellation
  during the timeout-cleanup retry also preserves the original timeout.
- All 24 checkpoint cases passed locally: the 23-case checkpoint run plus the
  added retry-cancellation case, then all 24 again within the full default suite.
- Full default Python suite: 3,203 passed, 21 skipped in 357.51 seconds.
- Ruff and mypy pass; the latter checks 76 source files.

## Independent pipe cleanup follow-up

- RED: a controlled communication-settlement failure left two real protocol pipe
  descriptors open. The regression releases those descriptors even on failure.
- GREEN: protocol pipes now close in an independent `finally` block, including
  when communication settlement fails. Such a failure still propagates and does
  not record `cleanup_returned`; stderr is closed explicitly only after settlement.
- The interruption regression's teardown also accepts a cleanup `TimeoutError`,
  so it does not replace an earlier assertion with that secondary failure.
- All 25 checkpoint cases pass locally in 30.42 seconds. The full-suite result
  above belongs to `70f9e7e…`; later results must bind their exact revision.

The retained 40-sample JSON is unchanged. Hosted results must be bound to the
final exact commit separately.
