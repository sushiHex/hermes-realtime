# Evidence capture boundary

This document defines the evidence-only Slice 0 policy. The host now contains
capture controls and admission, lifecycle, storage, status, and purge implementation.
Full Slice 0 qualification is incomplete. See [Implementation status](implementation-status.md)
for source, test coverage, and the outstanding scenario producers, and the
[Roadmap](roadmap.md) for the implementation sequence.

## Status and activation

Evidence capture is disabled by default.
Operator enablement is not consent.

The host's `--evidence-capture` flag makes authenticated controls available, but only
an exact, current disclosure accepted for the current browser/media binding may begin capture.
The conversation-only local launcher remains capture-disabled. A disable override must
win over every enablement path.

Slice 0 ships no web-app manifest, service worker, offline cache, update policy,
permission-retention policy, or installed PWA behavior. It runs only in an interactive
logged-on-user console. Windows service installation, Windows Scheduled Task installation,
Session 0, background start, stop/restart policy, reboot recovery, deployment, publishing,
Phase 1, and every learning phase are excluded.

## Qualification boundary

Repository policy tests validate this document, the disclosure resource, and the
release-gate packaging contract. Passing those checks proves only that the declared
boundary is present and internally consistent; it does not prove that production
evidence capture exists or that any physical, perceptual, privacy, or security claim
has been qualified.

## Threat boundary

Captured text is inert, attacker-controlled data. It must never be interpreted as an
instruction or used to change conversation behavior. The producer spool may detect
defined corruption, incomplete lifecycles, crash truncation, stale or replayed
identities, and ordinary cross-process ownership contention.

The spool does not authenticate bytes against another process running as the same OS user.
Filesystem permissions reduce cross-user exposure but do not establish same-user
integrity. Slice 0 makes no encryption-at-rest claim, and a deny filter does not make
arbitrary transcript text nonsensitive. Any future reader must treat producer bytes as
untrusted even when their hashes verify.

## Disclosure and consent

The package-authored disclosure cannot be replaced by CLI, environment, operator configuration, or model output.
Its version, exact UTF-8 text, digest, retention, and served-asset hashes are one
candidate-bound consent surface. An asset or digest mismatch refuses new consent while
allowing privacy maintenance on an existing store.

Evidence-capture consent authorizes only the disclosed local-storage behavior. It does not authorize
public-search egress. Public search has a separate package-authored disclosure, operator-enable gate,
browser-binding consent sequence, status projection, and revoke control; neither consent implies the
other.

Consent must be authenticated, explicit, and scoped to the current browser/media
binding, selected microphone and/or typed source, disclosure digest, and retention.
Operator enablement merely exposes the control. Capture may retain only authoritative final user text from a consented source, generated
assistant text, and independently confirmed server-transport text needed to classify the
lifecycle. Exact admitted transcript text remains sensitive and may itself contain paths,
URLs, secret-looking strings, or exception-like text; the narrow deny filter is not a
general secret detector. No separate metadata field may retain command text,
proactive/replay text, participant or speaker identity, task authority, profile data,
paths, secrets, URLs, exception strings, or arbitrary metadata.

The disclosure must identify transcript sensitivity, local SQLite storage without an
encryption-at-rest guarantee, a default 24-hour raw-evidence TTL configurable from 1 to
168 hours, bounded quotas, and capacity-based noncapture. Revocation closes current
epoch admission before scheduling that epoch for purge. Previously closed epochs remain
subject to their TTL or explicit full-store purge. Status is persistent but content-free
and exposes no transcript, identity, profile, path, count, or quota.

## Queue boundary

Foreground conversation never waits for evidence persistence.
Capacity failure drops or taints capture while the conversation operation continues.
Revocation and owner drain use independent reserved lanes.

The scheduler is exactly accounted by owned record and byte credits, never by an
approximate queue size. The ordered lane is bounded to 64 logical records, 64 physical
items, and 2 MiB of canonical bytes. Consent creation, every live lease, and lifecycle
close reserve their terminal credits before admitting content. Ordinary records cannot
consume those reservations. The coalescing revoke lane carries no text, and the
owner-only drain lane is accepted once. Admission gaps, oversize records, exhausted
terminal credit, or writer faults make affected evidence ineligible without changing
the foreground operation's outcome.

## Lifecycle boundary

Consent is not inherited across process restart, reconnect replacement, media-incarnation replacement, disclosure change, or revocation.
Each accepted binding owns a fresh epoch and bounded logical session. Command evidence
exists only after a real accepted dispatch acknowledgment; a declined command route may
open a user-response turn, while rejected or invalid input retires without captured
text. Proactive and replay turns carry lifecycle information only and never captured
text.

Generated text admission, complete server transport confirmation, the pre-cleanup
snapshot, authoritative cleanup, and terminal settlement are distinct transitions. A
snapshot occurs before delivery capability is destroyed and settlement occurs after
authoritative cleanup. Only a gap-free, conflict-free, untainted, unexpired session with
a complete valid lifecycle can seal atomically and become locally eligible for
qualification or inspection. A precommit crash leaves it open or tainted; an atomic
seal cannot be partial. Process death makes its active epoch subject to privacy recovery
before reads or new consent.

Owned close proceeds through browser ingress, foreground work, speech, seal-or-taint,
revoke finalization, and bounded writer drain even if an earlier stage fails. Model,
speech, playback, and conversation paths never wait for the drain acknowledgment.

## Platform boundary

Windows is the only Slice 0 evidence-storage target. Its storage root is an unbound
producer capability on a validated absolute local fixed disk; it is not a Hermes profile
or plugin-data-root capability. The host-local default is under
`%LOCALAPPDATA%\HermesRealtime\evidence`, and a custom target must retain the exact
`capture-v1.sqlite3` basename.

On Linux, evidence enable, status, and purge requests fail closed as `unsupported_platform` before any evidence path is created or opened.
Linux retains the null capture surface: imports, the ordinary host, and the local
launcher continue without a directory, lock, database, identity, or writer thread.
Status on an absent Windows store is also noncreating. Status and purge compose no media,
browser, inference, TTS, network, or Hermes task service.

## Purge boundary

Routine TTL, revocation, unclean-recovery, and full-store erasure first establish durable
erasure authority, perform logical deletion, then run bounded maintenance and verify
owned artifacts. A pending or failed erasure blocks reads, scoped consent, and any
purge-complete acknowledgment; restart resumes it.

Purge targets only the manifest-owned database artifacts; it never uses a glob or recursive delete.
Full purge removes and verifies only the database, journal, WAL, shared-memory, vacuum,
and temporary database names under the already validated parent. It preserves adjacent
files plus the owned root marker and stable sentinel. Only a later accepted consent may
create a new installation identity.

VACUUM and byte-absence checks are not SSD forensic erasure, and Python strings are not securely zeroized.
Restart is required for stronger clearing of process memory. Process-crash and SQLite
atomicity tests do not establish physical power-loss durability.

## Physical-observation boundary

The runtime cannot prove physical speaker identity, typed authorship, browser playout, audibility, hearing, comprehension, agreement, or truth.
Transport confirmation establishes only the defined server-side transport transition.
A human-assisted Windows Chrome-tab gate may record closed observation codes about
device selection, microphone permission, mute-state behavior, and required operator
actions. Those contemporaneous observations remain separate from deterministic machine
assertions and do not become identity, comprehension, or inferred-audibility claims.
Raw audio, screenshots, and transcripts are purged rather than retained as replayable
proof, so later physical reanalysis is impossible.

## Plugin boundary

Entry-point discovery does not activate the plugin or authorize capture.
Installing the Realtime wheel makes its existing `hermes_agent.plugins` entry point
discoverable only. Bare import, entry-point enumeration, and registration must create no
evidence filesystem, identity, thread, media/model/network service, consent, or profile
mutation. The plugin compatibility gate and the evidence-host gate remain separate.

Compatibility testing is limited to a supplied local Hermes v0.20 source `PluginManager`
harness. That harness is not bound here to a publicly retrievable immutable upstream release.
No Hermes wheel or public release-version compatibility is claimed. Only the exact
noninteractive isolated-profile enable flow may activate the tested candidate entry point.
Entry-point discovery provides packaging compatibility only; no v0.20 in-process bridge dispatch
is claimed.

## No-learning boundary

It captures no raw audio and has no profile, learning, or reviewer import path.
The spool has no runtime captured-text reader, export protocol, background review,
proposal, approval, prompt-injection, learned-context, memory, skill, Curator, SessionDB,
external-memory, code, deployment, or publishing path.

Captured text cannot affect conversation, inference, context, routing, tools, policy, prompts, tasks, canonical state, memory, skills, code, Curator, SessionDB, or a profile.
The writer may inspect text only for bounded validation and persistence-deny filtering.
The unbound data root cannot authorize profile attribution, and existing sessions can
never be automatically back-attributed.

Phase 1 remains blocked because Hermes v0.20.0 provides neither an attested
`HermesProfileBinding` nor an attested writable `HermesPluginDataRoot`. Any importer,
reviewer, proposal, projection, or learning work requires new host-owned capabilities,
a fresh threat model, adversarial approval, and a new owner-approved plan.
