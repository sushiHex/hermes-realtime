# Implementation status

[Collaborator guide](README.md) · [Roadmap](roadmap.md)

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

The [qualification module](../scripts/qualify_evidence_slice_zero.py) implements
candidate/input validation, report semantics, and Windows process ownership.
The `SCENARIO_REGISTRY_V1` now registers one real
[`deterministic_equivalence` producer](deterministic-equivalence.md), with
**19 governed producers still explicitly unavailable**. The source-only producer
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

The [registry tests](../tests/test_qualify_evidence_slice_zero.py) preserve the
remaining refusal contract. This producer cannot generate an accepted full
`qualification-report-v1`; the remaining matrix, installed-path prerequisites,
and required human observations are still outstanding. Follow the ordered
[roadmap](roadmap.md#ordered-milestones).

## CI evidence and open investigation

[Main push run `34398885362`](https://github.com/sushiHex/hermes-realtime/actions/runs/34398885362)
completed on **attempt 1** with all four jobs successful:

| Job | Conclusion |
| --- | --- |
| Pure candidate wheel | success |
| Linux null capture | success |
| Hermetic release candidate | success |
| Native LiveKit release integration | success |

This is automated evidence for the reviewed baseline. It does not establish the
full human-assisted Slice 0 matrix or production readiness. Packaging and
installed-boundary details live in [Release gates](release-gates.md).

[PR #21](https://github.com/sushiHex/hermes-realtime/pull/21) upgraded Vitest to
4.1.11. Treat its timing as a separate test-runner baseline from Vitest 3; retain
the lockfile and actual environment alongside comparisons.

[Issue #13](https://github.com/sushiHex/hermes-realtime/issues/13) remains open for
the historical Windows timing failures. The
[child-exit regression](windows-checkpoint-child-exit.md) demonstrates the
pipe-deadlock class with 64 KiB, while the historical `ack_eof` child wrote 2,035
bytes. Its hosted cause remains unproven. The earlier
[3112.190 ms ICU warm-up](https://github.com/sushiHex/hermes-realtime/issues/13#issuecomment-5594418906)
is a sub-threshold observation, not a timeout recurrence. Total pytest-node
duration is not child-exit timeout headroom, and missing diagnostics are not zero
measurements.

## Maintaining this snapshot

When implementation or defaults change, update the affected row in the same PR.
When qualification changes, record the exact candidate, run/attempt, exercised
scope, and remaining limitations. If a row is reviewed against a newer candidate
than this page's baseline, record that identity in the row; advance the page-wide
baseline only after reviewing the whole map. Keep issue discussions and run logs
as the detailed evidence record, and follow the
[contribution guidance](../CONTRIBUTING.md#documentation-maintenance).
