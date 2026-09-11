# Owned-close fault qualification

[Collaborator guide](README.md) · [Implementation status](implementation-status.md)

The `owned_close_faults` producer qualifies retained close ownership after two
completed synthetic conversations through the real HTTP, LiveKit, and host
runtime. It executes the source-bound Pure candidate wheel under the existing
Windows process owner. The Native CI job prints a content-free receipt only after
normal worker exit and complete owned-process cleanup.

## Comparison and ownership

Seven fixed arms share one ephemeral content-commitment key:

| Arms | Close stimulus | Required observations |
| --- | --- | --- |
| Ordinary, capture disabled and consented | Close once after typed and PCM turns complete | Original caller returns; each runtime and provider stage completes once. |
| Retry, capture disabled and consented | Deterministic speech fixture's first close callback raises | First caller and launcher attempt remain failed; retry invokes only the unfinished provider slot and completes. |
| Cancelled caller, capture disabled and consented | Hold the provider callback, join the retained close owner, then cancel the first caller | First caller remains cancelled; provider and joining caller remain pending until release, then finish under the same owner. |
| Perturbed, capture disabled | Change the real typed input and close normally | Authoritative committed conversation context differs from the ordinary baseline. |

The [independent validator](../scripts/owned_close_faults.py) requires all six
conversation content records to match the ordinary baseline across the first six
arms. Each disabled/consented pair must also preserve its exact first-caller
outcome. Consented arms require both completed terminal settlements. Ordered
production close observations retain the failed launcher attempt before a
successful retry; runtime stages cannot repeat. Observed callback events bind
entry, failure, cancellation, release, and return to the actual close driver.

The [worker](../scripts/owned_close_faults_worker.py) observes final retained owner
success, completed runtime/provider stages, and settled driver tasks. Its fault
gates are released and cleanup joined even if an observation fails. The packaged
parent independently checks normal exit, waits every retained process handle,
and requires zero active owned processes before minting a receipt. Caller-supplied
rows cannot mint run authority. The receipt binds source commit, tree, archive,
wheel, and observation digests without paths, source content, or identifiers.

## Related runtime repair

[Issue #40](https://github.com/sushiHex/hermes-realtime/issues/40) identified a
completed consent-settlement task being gathered through its stopped owner loop.
The [evidence runtime](../src/hermes_realtime/evidence/runtime.py) now reads an
already-terminal result synchronously, retaining success, failure, or cancellation.
Pending tasks retain their cancellation and observation path. Public close joins
and retries reuse the original binding revocation instead of revoking twice.

[Runtime regressions](../tests/evidence/test_owned_close.py) cover successful,
failed, and cancelled completed tasks on stopped or closed loops, real delayed
SQLite consent activation, pending settlement cancellation, concurrent close
callers, caller cancellation, and retry after an early close-stage failure. This
is a narrow completed-task repair; it provides no general migration of pending
tasks between event loops.

## Evidence boundary

Faults enter the deterministic test provider's close callback after conversation
has completed. The production launcher, runtime, media transport, and consented
SQLite cleanup are real. The no-task qualification host does not exercise full
Hermes background-work ownership or external provider cleanup. This
matrix does not qualify active-turn shutdown, physical audibility, or every close
failure position. It complements the separate deterministic-equivalence scenario.

This is one `packaged_process` scenario at unchanged ordinal 19, with the governed
assertions `conversation_trace_equal_to_revised_close_baseline` and
`owned_process_cleanup`. Thirteen producers remain unavailable; full Slice 0
qualification remains incomplete. Capture remains disabled by default.

[Issue #43](https://github.com/sushiHex/hermes-realtime/issues/43) records exact
candidate, review, and run evidence. These results do not establish the historical
causes investigated in
[#13](https://github.com/sushiHex/hermes-realtime/issues/13),
[#24](https://github.com/sushiHex/hermes-realtime/issues/24), and
[#27](https://github.com/sushiHex/hermes-realtime/issues/27).
