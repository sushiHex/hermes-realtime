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
An expired completion still fails the test; its existing cleanup kills a live
child and waits at most five seconds to reap it. The fixture now also closes its
parent stderr stream explicitly. A regression injects a completion timeout to
verify propagation, reaping, stream closure, and incomplete timing observations.
Startup, checkpoint, workflow, and job timeout authorities are unchanged.

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
pytest capture retains it on failure; `-s` exposes passing samples. It contains
only a version, fixed case name, PID, outcome flags, exit code, stderr byte
count, and monotonic event offsets. It contains no stderr text, command line,
environment, checkpoint nonce, or filesystem path.

The event offsets are milliseconds from the parent's process-creation attempt:

- `spawn_returned`: `Popen` returned;
- `checkpoint_received`: the parent received the startup frame;
- `failure_sent`: the parent closed the ACK writer or wrote the invalid ACK;
- `completion_entered` / `completion_returned`: the parent awaited communication
  and exit; their difference includes executor queueing and stderr draining;
- `cleanup_entered` / `cleanup_returned`: the fixture's cleanup interval.

These are parent observations, not child-side timestamps. Closing the ACK writer
does not measure when the child processes EOF. A missing event or a null stderr
count means it was not observed, not that its duration or byte count was zero.
A killed child's exit code is a cleanup result, not its natural exit status.

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

## Local validation

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

These are local Windows checks. Hosted release gates have not been run for this
change, and issue #13 remains open.
