# Roadmap

[Collaborator guide](README.md) · [Implementation status](implementation-status.md)

The next milestone is a reproducibly qualified, constrained Windows desktop
evidence-capture path using the existing host. Capture remains disabled by default
and requires separate participant consent. Ordinary conversation must remain
independent of evidence persistence.

This roadmap describes public, hardware-generic work. Deployment-specific plans and
sensitive qualification material belong outside the repository. The
[implementation status](implementation-status.md) records the reviewed starting
point; the detailed [evidence contract](evidence-capture.md) governs acceptance.

## Ordered milestones

| Order | Deliverable | Prerequisite | Complete when |
| --- | --- | --- | --- |
| 1 - implemented | One complete deterministic-equivalence qualification producer | Review the existing scenario contract, production observations, and integration coverage. | The real producer binds observations to the candidate and owned process lifecycle; independent validation rejects missing observations, false identity, and incomplete cleanup. Other scenarios remain explicitly unavailable. |
| 2 - in progress | Remaining governed scenario producers | Establish the first complete producer without weakening its acceptance contract. | Every required scenario has a real producer with focused positive and failure coverage. A partial run cannot produce an accepted full qualification report. |
| 3 | Frozen-candidate Windows desktop qualification | Complete the producer and installed-path prerequisites. | All required scenarios, including human-assisted observations, have valid evidence on the same candidate. Missing prerequisites and failed cases block the corresponding claim. |
| 4 | Constrained alpha release decision | Complete desktop qualification and review the supported-use matrix. | The release decision identifies exact artifacts, demonstrated support, compatibility limits, and known issues. Publication requires release-owner authorization. |

The first producer and the [packaged revocation race](revoke-race.md) are registered in
[`qualify_evidence_slice_zero.py`](../scripts/qualify_evidence_slice_zero.py).
Its [implementation guide](deterministic-equivalence.md) documents the fixed
comparison arms, archive binding, and acceptance limits. Extend the remaining
producers through the same candidate and ownership boundaries. Existing
[full-host ingress tests](../tests/integration/test_qualification_full_host_ingress.py)
remain integration coverage to examine before adding another harness. Revocation
now has a producer under [issue #29](https://github.com/sushiHex/hermes-realtime/issues/29);
the [capacity rollover producer](capacity-rollover.md) adds real session-budget
transition coverage under [issue #33](https://github.com/sushiHex/hermes-realtime/issues/33).
The [capture admission overflow producer](over-budget-turn.md) adds durable
exclusion and rejected-source absence under [issue #35](https://github.com/sushiHex/hermes-realtime/issues/35).
The [packaged spool crash matrix](spool-crash-matrix.md) adds recovery coverage
under [issue #38](https://github.com/sushiHex/hermes-realtime/issues/38).
[Full-purge cleanup](full-purge-cleanup.md) adds exact deletion and fresh-process
idempotence under [issue #41](https://github.com/sushiHex/hermes-realtime/issues/41).
[Owned-close faults](owned-close-faults.md) adds paired conversation comparisons
across ordinary close, failed-provider retry, and caller cancellation under
[issue #43](https://github.com/sushiHex/hermes-realtime/issues/43), with narrow
consent-settlement and binding-ownership regressions for
[issue #40](https://github.com/sushiHex/hermes-realtime/issues/40).
Thirteen scenarios remain unavailable. Continue with installed-host crash and
filesystem fault prerequisites before physical qualification, preserving the
existing scenario IDs and acceptance contracts.

Keep each implementation change independently reviewable. For milestone 2, group
work by existing ownership boundaries: consent/revocation, bounded admission,
crash recovery/purge, and owned close. Preserve the canonical scenario IDs and
execution order in the qualification module. A producer records observations;
the validator determines acceptance. Neither a supplied success flag nor a
test-only replacement for production behavior establishes qualification.

For milestone 3, keep human observations separate from machine assertions.
Synthetic audio and server transport confirmation do not establish physical
audibility, acoustic quality, or general browser readiness. The
[evidence boundary](evidence-capture.md#physical-observation-boundary) defines what
may be claimed and retained.

## Maintenance alongside the milestone

Track Windows CI timing in [issue #13](https://github.com/sushiHex/hermes-realtime/issues/13).
Preserve a recurrence's exact candidate, run, attempt, failed node, measurements,
and same-run comparator before selecting a discriminating experiment. The
[checkpoint investigation](windows-checkpoint-child-exit.md) and
[status page](implementation-status.md#ci-evidence-and-open-investigation) explain
the remaining evidence limits. Do not raise global timeouts, weaken assertions,
or use a routine rerun as the sole response.

Track the separate archived-worker abnormal exit in
[issue #24](https://github.com/sushiHex/hermes-realtime/issues/24). Add bounded,
content-free teardown observations before selecting a runtime change. A complete
scenario exchange followed by an abnormal exit remains a failed qualification.
Once diagnostics are in place, continue milestone 2 while collecting recurrence
evidence; an unexplained intermittent failure is not evidence of a fix.

Track the separate browser readiness failure in
[issue #27](https://github.com/sushiHex/hermes-realtime/issues/27). Preserve its
readiness assertion and distinguish navigation, connection, and media-activation
observations before assigning a cause.

Security and demonstrated compatibility needs can take priority. Review dependency
updates against the current base and affected runtime: generated browser assets
and notices must accompany changes that alter them; Node types must match the
supported runtime; optional native providers need relevant coverage. Treat compiler,
DOM, and test-runner major upgrades as separate migrations. Keep current PR state
in [GitHub](https://github.com/sushiHex/hermes-realtime/pulls), rather than copying
a dependency queue that becomes stale here.

For each merge, qualify the exact candidate, resolve review findings, and follow
the [release-gate sequence](release-gates.md#required-automated-checks). Wait for the
resulting main push workflow before advancing to another candidate. Green tests
support their exercised scope; they do not clear an unexplained historical cause.

## Later work and explicit dependencies

| Work | Gate before proceeding |
| --- | --- |
| Promote `natural_v1` | Characterize the exact profile and complete its relevant browser and human checks; see [conversation profiles](../README.md#conversation-profiles) and [local gates](local-livekit.md). |
| Promote speculative lookup, recovery, or another search backend | Complete the existing [source-backed promotion gate](source-backed-latency.md#promotion-gate), including consent, paired measurements, failures in the denominator, and its physical matrix. |
| Claim public Hermes-version compatibility | Bind a retrievable immutable upstream identity and qualify entry-point discovery separately from functional API dispatch; see [Hermes integration](hermes-bridge.md). |
| Expand to general iPhone/WebKit use | Complete a separate physical browser matrix. Chrome CI does not supply this evidence. |
| Add learning or profile import | Obtain the missing host-attested profile and writable data-root capabilities, then establish the new threat model and owner-approved plan required by the [no-learning boundary](evidence-capture.md#no-learning-boundary). |
| Add services, background startup, PWA/offline behavior, or deployment | Define a separate scope and acceptance plan. These remain outside evidence-only Slice 0. |

When a milestone completes, update this page and the status page with the supporting
issue/PR and candidate evidence. Reorder work when evidence changes the priority;
do not broaden a completed milestone's claims retrospectively.
