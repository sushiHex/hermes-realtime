# Implementation status

[Collaborator guide](README.md) · [Work tracking](work-tracking.md)

Implementation and evidence reviewed on **2026-09-09** at main commit
[`c7b8975e33f5dac7203845d7aa9207ee3518d453`](https://github.com/sushiHex/hermes-realtime/commit/c7b8975e33f5dac7203845d7aa9207ee3518d453),
tree `478bf2a68035817214a98e3b02e08cf84bdf277f`. This is the reviewed implementation
baseline, not a claim that subsequent edits have been qualified. Relative source
and test links follow the checkout being read; the commit link identifies the
snapshot behind this review.

Read each capability on three separate dimensions: implementation present,
qualification evidence available, and activation/default behavior. A source or
test link establishes where to inspect a capability; it does not imply that every
optional or hardware-dependent test ran. Missing evidence remains unverified.

## Capability map

| Capability and implementation | Activation/default | Evidence and remaining qualification |
| --- | --- | --- |
| **Host and browser conversation:** [host composition](../src/hermes_realtime/host_launcher.py), [client](../src/hermes_realtime/client/), and [browser source](../web/src/) are present. | Explicit launcher; loopback is the default listener scope. | [Browser self-acceptance](../tests/integration/test_browser_self_acceptance.py) is exercised by the Native CI job. Synthetic and installed-path checks do not establish subjective quality or general iPhone/WebKit readiness. See [local gates](local-livekit.md). |
| **Hermes work and cancellation:** [API session](../src/hermes_realtime/integration/api.py) and [work tools](../src/hermes_realtime/conversation/work_tools.py) are present. | Authenticated full-host configuration; the conversation-only local launcher refuses work dispatch. | [API tests](../tests/integration/test_hermes_api_session.py) and [work-tool tests](../tests/integration/test_natural_work_tools.py) cover their boundaries. Real installed dispatch and acknowledgement latency require the separate [natural-work gate](release-gates.md#installed-natural-work-boundary-and-latency-gate). |
| **Conversation profiles:** [conversation state](../src/hermes_realtime/conversation/state.py) and [playback](../src/hermes_realtime/livekit/playback.py) are present. | `legacy` is the default; `natural_v1` is opt-in. | [Conversation tests](../tests/conversation/) and browser checks cover the exercised paths. `natural_v1` remains unqualified for general use; retain the [profile limitations](../README.md#conversation-profiles). |
| **Public search:** [lookup](../src/hermes_realtime/providers/current_facts.py), [coordination](../src/hermes_realtime/conversation/knowledge.py), and [egress controls](../src/hermes_realtime/search_egress.py) are present. | Default off; operator enablement and browser-binding consent are separate gates. Speculation/recovery are also default off. | [Consent-bound lookup tests](../tests/providers/test_consent_bound_current_facts.py) and [knowledge tests](../tests/conversation/test_knowledge.py) cover policy. Broader promotion still requires the [source-backed gate](source-backed-latency.md#promotion-gate). Search grants no work authority. |
| **Windows evidence capture:** [host controls](../src/hermes_realtime/host_launcher.py) and [admission, lifecycle, storage, and runtime](../src/hermes_realtime/evidence/) are present. | Default off; `--evidence-capture` only exposes consent-bound controls. The local conversation-only launcher remains capture-disabled. | [Evidence tests](../tests/evidence/), [full-host ingress](../tests/integration/test_qualification_full_host_ingress.py), and [synthetic audio](../tests/integration/test_qualification_full_host_synthetic_audio.py) provide implementation coverage. Full governed Slice 0 qualification remains incomplete; see the producer status below. |
| **Linux null capture:** platform refusal and noncreating behavior are implemented. | Evidence enable/status/purge refuse as `unsupported_platform`; ordinary imports and host use retain the null surface. | The required [Linux null-capture test](../tests/integration/test_linux_null_capture.py) exercises the offline-installed candidate wheel. This qualifies the null surface, not Linux evidence storage. |
| **Plugin discovery:** [entry point](../src/hermes_realtime/hermes_plugin.py) and [compatibility harness](../scripts/qualify_hermes_v020_pluginmanager.py) are present. | Discovery does not launch the host or grant capture consent. | [Entry-point tests](../tests/integration/test_plugin_entrypoint.py) and [harness tests](../tests/test_qualify_hermes_v020_pluginmanager.py) cover packaging and the supplied source surface. Public immutable Hermes-version compatibility is not established; v0.20 in-process bridge dispatch remains fail-closed. |
| **Learning/profile import:** no capture-to-learning or runtime captured-text import path is provided. | Excluded from Slice 0. | The [no-learning contract](evidence-capture.md#no-learning-boundary) requires new host-owned capabilities and a separate approved plan before this work can begin. |

## Qualification producer status

The [input-binding implementation](qualification-input-files.md) adds retained
file and tool owners, source-locked dependency verification, six independent
build recipes, purpose-specific Windows runtime installation, publisher and
provider-resource bindings, and an authenticated Linux prerequisite. Its final
acceptance boundary requires the complete matching input graph and completed
preparation cleanup. These are prerequisites for the scenario runner; they cannot
create a full qualification report. Native recipe evidence and synthetic
acceptance tests retain their distinct limits. [Issue #47](https://github.com/sushiHex/hermes-realtime/issues/47)
owns the input closure and its exact candidate evidence; [#62](https://github.com/sushiHex/hermes-realtime/issues/62)
owns composition with scenario and report evidence. This addition does not advance
the page-wide reviewed baseline above.

The [qualification module](../scripts/qualify_evidence_slice_zero.py) implements
candidate/input validation, report semantics, and Windows process ownership.
The process owner now associates its noninherited Job during suspended child
creation and exposes a single resume step. The separate
[private journal](../scripts/qualification_run_journal.py) durably records bounded
ownership observations through a borrowed file handle. Their
[implementation boundaries](qualification-execution.md#finalize-validate-and-retain)
and tests distinguish these primitives from verified recovery, producer
orchestration and complete-report acceptance. Exact candidate evidence belongs
to [#62](https://github.com/sushiHex/hermes-realtime/issues/62); this addition does
not advance the page-wide reviewed baseline.

The `SCENARIO_REGISTRY_V1` now registers eight real producers,
[`deterministic_equivalence`](deterministic-equivalence.md),
[`revoke_race`](revoke-race.md), [`capacity_rollover`](capacity-rollover.md),
[`over_budget_turn`](over-budget-turn.md),
[`spool_crash_matrix`](spool-crash-matrix.md),
[`synthetic_fault_matrix`](synthetic-fault-matrix.md),
[`full_purge_cleanup`](full-purge-cleanup.md), and
[`owned_close_faults`](owned-close-faults.md), with **12 governed producers still
explicitly unavailable**. The source-only equivalence producer
owns the archived child, collects nine fixed comparison arms, and independently
validates conversation, settlement, close, and process cleanup observations.
The native CI job invokes it against the checked-out committed candidate and
prints its exact source identities on success.

The first complete local source-only run used commit
`7b9cf17501c21feab30c91c6fcce7cf3aaba3234`, tree
`b8e77aa3a8cf873060479126592b62847ec5e4e3`, and archive SHA-256
`8d2fa37f7aa3822be9859629168a165448a4182e3e0a2996fffc82523f9611f4`.
It accepted nine arms and completed cleanup of five owned processes. This is a
candidate-specific implementation result; subsequent changes require their own
run. The [producer guide](deterministic-equivalence.md) states its source-only,
media-selection, and dependency-environment limits.

The packaged revocation producer executes the source-bound pure wheel through
the existing owned Windows process harness. Its independent validator requires
durable revocation, unchanged evidence admission while ordinary conversation
continues, verified purge, normal exit, and complete cleanup. The Native job
consumes the Pure candidate wheel artifact and prints the accepted source, wheel,
and observation identities. This is a single packaged scenario; dependency-closure
and physical claims retain the limits in its [implementation guide](revoke-race.md).

The packaged [capacity rollover producer](capacity-rollover.md) drives three
ordinary turns through the real session budget and writer transaction. It binds
before-commit reader isolation, durable predecessor/successor lineage, source and
persisted content equality, continued conversation, and owned cleanup. The
[implementation issue](https://github.com/sushiHex/hermes-realtime/issues/33)
records exact candidate and run evidence. This scenario does not qualify queue
saturation or cumulative turn-budget overflow.

The packaged [capture overflow producer](over-budget-turn.md) exceeds the real
64-record ordinary queue limit while conversation continues. Its independent
validator binds the persisted prefix to accepted source, requires incomplete
evidence to remain excluded, and scans the closed store for rejected source.
Admission permanently marks a lost one-shot publication as incomplete; completed
conversation retires only its exact evidence lease and advances pending
revocation. Capture remains disabled by default. [Issue #35](https://github.com/sushiHex/hermes-realtime/issues/35)
records exact qualification evidence. Cumulative 16 MiB turn-quota overflow
remains unqualified.

The [registry tests](../tests/test_qualify_evidence_slice_zero.py) preserve the
remaining refusal contract. These producers cannot generate an accepted full
`qualification-report-v1`; the remaining matrix, installed-path prerequisites,
and required human observations are still outstanding. Use
[GitHub Issues](https://github.com/sushiHex/hermes-realtime/issues?q=is%3Aissue%20is%3Aopen%20label%3Aqualification)
for current ownership, prerequisites, and acceptance criteria; this page records
implementation and evidence rather than work status.

The [packaged spool crash producer](spool-crash-matrix.md) covers all 82 governed
cases, including both restart clocks for rollback. Its independent observer
checks durable history, complete seals, recovery purge, and retained filesystem
authority. [Issue #38](https://github.com/sushiHex/hermes-realtime/issues/38) records
exact candidate evidence and the seeded historical-fixture boundary.

The [full-purge cleanup producer](full-purge-cleanup.md) uses two fresh workers to
verify exact database-artifact deletion, preserved adjacent decoy bytes, and an
unchanged repeated purge. Parent-side checks establish handle release before each
worker exits. Its synthetic sidecars qualify deletion only; initialization debris,
host orchestration, volume-full, and physical behavior remain separate.
[Issue #41](https://github.com/sushiHex/hermes-realtime/issues/41) records its evidence.

The [synthetic fault matrix](synthetic-fault-matrix.md) drives all five governed
injections through the real scheduler, spool and daemon, with independent source,
capacity and purge checks. Five retained processes must release storage and exit
normally before acceptance. This `synthetic_injected` proof does not establish
installed dependency closure or physical behavior. [Issue #59](https://github.com/sushiHex/hermes-realtime/issues/59)
records the exact candidate and run evidence.

<a name="ci-evidence-and-open-investigation"></a>

## CI evidence and investigation limits

The capability map retains its reviewed baseline above. The following dependency
and automated-qualification update is bound separately to main commit
[`8676b114696dd47a75f0ff58df0fc1eaf6b9846c`](https://github.com/sushiHex/hermes-realtime/commit/8676b114696dd47a75f0ff58df0fc1eaf6b9846c),
tree `714ebdbafba4ad47a00fdf23cc197b9248d092b2`.
[Main push run `34435214901`](https://github.com/sushiHex/hermes-realtime/actions/runs/34435214901)
completed on **attempt 1** with all four jobs successful:

| Job | Conclusion |
| --- | --- |
| Pure candidate wheel | success |
| Linux null capture | success |
| Hermetic release candidate | success |
| Native LiveKit release integration | success |

That run accepted nine archived deterministic-equivalence arms and cleaned up
five owned processes, using source archive SHA-256
`b43cc8b34bfcd791b5c8279b337255efd190b6af88c12174c39b0b66ee6ce4e3`.
It does not establish the full human-assisted Slice 0 matrix or production
readiness. Packaging and installed-boundary details live in
[Release gates](release-gates.md).

| Qualified migration | Candidate dependency boundary |
| --- | --- |
| [TypeScript](https://github.com/sushiHex/hermes-realtime/pull/10), [jsdom](https://github.com/sushiHex/hermes-realtime/pull/11), and [Vitest](https://github.com/sushiHex/hermes-realtime/pull/8) | TypeScript 7.0.2, jsdom 30.0.1, `@types/jsdom` 30.0.0, and Vitest 5.0.0; browser CI uses Node 22.22.2 on Windows and Linux. |
| [LiveKit browser client](https://github.com/sushiHex/hermes-realtime/pull/5) | `livekit-client` 2.22.2 with regenerated assets and disclosure hashes; Linux validates the disclosure manifest before wheel construction. |
| [Optional speech closure](https://github.com/sushiHex/hermes-realtime/pull/4) | Kokoro 0.6.1 and phonemizer 3.4.0; packaging 26.3 and protobuf 7.36.0 in the CPU and CUDA closures. Source-archive and installed-wheel probes exercised synthetic synthesis and owned cleanup on both providers. |

Vitest 5 is a new timing baseline relative to the earlier Vitest 3 and 4 runs.
Retain the exact lockfile and environment when comparing measurements. Installed
synthetic speech probes do not establish physical audibility or subjective quality.

The real SQLite consent and retention tests invoke owned runtime close from
`finally`, including when activation assertions fail. Their failure-path regression
starts the real dispatcher and SQLite writer, exercises a pending activation
through its unchanged timeout and injects a failed result or exception after
startup, then requires both retained thread objects to be stopped. The
[earlier local activation timeouts and subsequent thread-count failures](https://github.com/sushiHex/hermes-realtime/pull/30#issuecomment-5614964675)
remain recorded; repairing test cleanup does not establish their timing cause.

[Issue #13](https://github.com/sushiHex/hermes-realtime/issues/13) tracks
the historical Windows timing failures. The
[child-exit regression](windows-checkpoint-child-exit.md) demonstrates the
pipe-deadlock class with 64 KiB, while the historical `ack_eof` child wrote 2,035
bytes. Its hosted cause remains unproven. The earlier
[3112.190 ms ICU warm-up](https://github.com/sushiHex/hermes-realtime/issues/13#issuecomment-5594418906)
is a sub-threshold observation, not a timeout recurrence. Total pytest-node
duration is not child-exit timeout headroom, and missing diagnostics are not zero
measurements.

A [later Native recurrence](https://github.com/sushiHex/hermes-realtime/issues/13#issuecomment-5611252950)
exhausted the unchanged `communicate(timeout=5.0)` authority for `ack_eof` and
`duplicate_key`. Draining alone therefore does not exclude a hosted child-exit
timeout. The [checkpoint fixture](../tests/test_qualification_checkpoint.py) now retains
cleanup stderr byte counts and fixed child milestones distinguishing launcher
close, CLI settlement, and entry into an exit callback. These observations keep
the existing completion and cleanup deadlines. The hosted cause remains unproven;
a later run must qualify the new diagnostic candidate separately.

[Issue #24](https://github.com/sushiHex/hermes-realtime/issues/24) tracks a separate
archived-worker access violation after the complete ready/nine-arm/done exchange
in run `34431895476`, attempt 1. The worker's abnormal exit was correctly rejected.
The healthy main run above is an additional observation, not proof of a cause or
fix. The [worker exit diagnostic](deterministic-equivalence.md#worker-exit-observations)
now retains fixed shutdown milestones separately from scenario frames, including
on failure. Normal exit and owned-process cleanup remain mandatory for acceptance.

## Maintaining this snapshot

When implementation or defaults change, update the affected row in the same PR.
When qualification changes, record the exact candidate, run/attempt, exercised
scope, and remaining limitations. If a row is reviewed against a newer candidate
than this page's baseline, record that identity in the row; advance the page-wide
baseline only after reviewing the whole map. Keep issue discussions and run logs
as the detailed evidence record, and follow the
[contribution guidance](../CONTRIBUTING.md#documentation-maintenance).

A separate [browser readiness investigation](https://github.com/sushiHex/hermes-realtime/issues/27)
tracks main push run `34441271535`, attempt 1, at `e6d1ffb08967055f064c37b15187408cd14d5071`.
Native's integration gate and archived equivalence passed, then browser
self-acceptance found typed input still disabled 30 seconds after Connect. The
other three jobs passed. The exact-tree PR run was green; it does not replace
this failed main result. Browser and runtime sources were unchanged by that
checkpoint diagnostic change. The hosted cause remains unproven.
