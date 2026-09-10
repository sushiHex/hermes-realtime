# Packaged revocation race

[Collaborator guide](README.md) · [Qualification status](implementation-status.md#qualification-producer-status)

The `revoke_race` producer runs the real host, browser HTTP ingress, LiveKit,
evidence admission, and SQLite writer from a pure candidate wheel. It reuses the
[archived process owner](deterministic-equivalence.md#candidate-and-process-ownership)
and existing full-host composition. Inference and speech inputs are deterministic;
production consent, admission, durability, and erasure remain authoritative.

The bounded implementation and acceptance scope is tracked in
[issue #29](https://github.com/sushiHex/hermes-realtime/issues/29).

## Observations and acceptance

The producer completes one consented typed turn, then submits revocation through
the host's browser API. A scheduling barrier holds the writer immediately before
its real `finalize_revoke` call. The acknowledgment must already report durable
revocation. A separate read-only SQLite connection observes the pending request,
revoked epoch, retained evidence, and admission watermark.

While finalization is held, a second ordinary typed turn must complete. A second
store snapshot must match the first, and the production admission observations
must show no additional evidence admission. The barrier then releases the same
real writer operation. Final observations must show one completed revoke, released
capacity, no evidence events, sessions, epochs, or pending requests, and a matching
erasure tombstone with the exact erased counts and a valid final watermark.

Each SQLite observation checks integrity and closes its connection before the
writer can continue maintenance. An ephemeral HMAC key binds the pending request
to the final tombstone within the invocation. Request, epoch, session, and binding
identifiers remain in the child; neither the key nor raw store records cross the
pipe. The parent receives only closed counts, commitments, and lifecycle facts.

The independent validator derives `revocation_request_durable`, `purged`, and
`purge_verified` from these observations. It also requires complete production
observations, the successful ordered host-close sequence, normal worker exit,
waits for every retained process, complete Job finalization, and workspace removal.
A supplied success flag, constructed receipt, missing observation, or abnormal
exit cannot establish qualification. Cleanup releases the scheduling barrier even
when an earlier assertion fails.

## Package and source binding

Before execution, the verifier binds the supplied wheel digest to the same clean
source commit, tree, and verified archive as the runner. Every package file must
match its archived source blob. The wheel must have the exact bounded pure-Python
member set, package identity, license, and complete SHA-256 `RECORD`. Its UTF-8
metadata must parse without defects, declare core `Metadata-Version: 2.4` (the
format produced by the pinned builder), and declare `Wheel-Version: 1.0`.
Both files must be plain header documents with valid header names: Unix-from
envelopes and multipart payloads are rejected. `METADATA` may carry its ordinary
description body; `WHEEL` may not carry body content.
The [core metadata fields](https://packaging.python.org/en/latest/specifications/core-metadata/)
are checked before granting wheel authority. The existing locked development
`packaging` library [validates their semantics](https://packaging.pypa.io/en/stable/metadata.html),
including dependency and Python-version declarations. A closed projection of the
verified archive's static `pyproject.toml` profile binds every declared field:
requirements and extra markers, Python versions, identity, authors, license,
classifiers, keywords, URLs, and the exact README description. Repeated fields
compare as multisets of validated values; header order is immaterial. Entry points
match the source with exact case, and build metadata matches the pinned builder.
Unsupported source metadata profiles fail closed without executing build code.
Duplicate, missing, unexpected, native, or mismatched members fail before launch.

The pure wheel is unpacked into a fresh owned directory outside the source tree.
Only that directory supplies `hermes_realtime`; the archive supplies the runner
and qualification drivers. All source imports are checked, and both materialized
trees are reverified after execution. No installer or network dependency resolver
runs inside the scenario. The inherited development environment supplies third-party
dependencies; this producer does not attest an offline installed dependency closure,
console-script installation, DLL provenance, or physical devices. Python-private
receipts are not a boundary against a hostile process with the same OS identity.

## Running the producer

Use the pure wheel from the candidate's required CI artifact, or a wheel built
from its verified source archive. On Windows with the locked development environment:

```powershell
uv run --frozen --group dev python -m scripts.qualify_revoke_race `
  --candidate . --baseline origin/main `
  --candidate-wheel $candidateWheel --wheel-sha256 $candidateWheelSha256 `
  --livekit-executable $livekitExecutable --livekit-sha256 $livekitSha256
```

The command requires a clean committed candidate and nonoptimized 64-bit Python.
It prints a `revoke-race-summary-v1` only after independent acceptance, binding the
source commit, tree, archive, wheel, observation digest, process count, and derived
machine assertions. The Native CI job consumes the existing Pure candidate wheel
artifact and runs this scenario after archived deterministic equivalence.

This is one `packaged_process` scenario, not an accepted full
`qualification-report-v1`. Eighteen governed producers and the remaining installed
and human-assisted prerequisites are still unavailable or unqualified. Synthetic
turns do not establish physical audibility, subjective quality, or browser readiness.
