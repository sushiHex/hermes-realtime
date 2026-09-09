# Deterministic equivalence

[Collaborator guide](README.md) · [Qualification status](implementation-status.md#qualification-producer-status)

The source-only `deterministic_equivalence` producer compares conversation behavior
across capture states using the archived candidate's real host, HTTP ingress,
LiveKit transport, evidence controller, and SQLite writer. It uses the existing
[qualification composition](../tests/support/qualification.py) and
[full-host driver](../tests/integration/test_qualification_full_host_ingress.py).
Deterministic inference, speech generation, transcription, and synthetic PCM are
controlled inputs. Production lifecycle and evidence owners remain authoritative.

## Comparison boundary

The fixed sequence covers disabled capture, available-but-unconsented capture,
consented capture, a blocked writer, a writer completion fault, active-response
shutdown in the three capture states, and a deliberately changed typed input.

The validator compares ordered commitments to the adapter's committed context,
generated text, and transport-confirmed text. It also checks real terminal
settlement, cancellation, foreground cleanup, ordered close observations, and host
return. The writer-fault arm must preserve the conversation facts and retain its
failed host return. An unchanged committed context in the changed-input control
fails validation. Missing, duplicate, incomplete, or unexpected observations fail.

Raw content stays in the child process. A fresh in-memory HMAC key binds each
content observation to its kind; only commitments and closed lifecycle facts cross
the inherited pipe. The key is discarded with the child. Commitments support
comparison within one invocation and cannot be compared across invocations.
No transcripts, PCM, credentials, or exception text enter the summary.

## Candidate and process ownership

The existing archive oracle verifies the clean commit, tree, and repeated archive
bytes. The runner executes a fresh materialization outside the repository, verifies
its files before and after execution, and checks that source imports resolve inside
that materialization. The executing runner must match the archived runner blobs.

A suspended root enters a Windows Job before it resumes. The owner retains process
handles, creation identities, image hashes, and ancestry. Python, LiveKit, and
Windows console helpers have explicit roles; an unexpected descendant fails.
Every checkpoint rechecks Job membership and the signaling listener's ownership.
Signaling binds loopback; media uses the existing SDK's network selection. This
scenario makes no general RTC network-isolation claim.

The parent drains bounded canonical frames while the child runs. A single scenario
deadline bounds collection. Success requires the expected final frame, successful
child exit, retained-handle waits, zero active Job members, closed handles, and
removal of the owned temporary tree. Cleanup may retry after failure, but a cleanup
retry cannot turn a failed invocation into an accepted result.

## Running the producer

On Windows, install the locked development environment with `uv sync --frozen --dev`.
Supply a trusted LiveKit executable and its independently verified SHA-256:

```powershell
uv run --frozen --group dev python -m scripts.qualify_deterministic_equivalence `
  --candidate . --baseline origin/main `
  --livekit-executable $livekitExecutable --livekit-sha256 $livekitSha256
```

The command requires a clean committed candidate and nonoptimized 64-bit Python.
It prints one content-free summary only after independent acceptance. The summary
binds the source commit, tree, archive digest, observation digest, and counts.
It is a single-scenario summary, not `qualification-report-v1`.

## Evidence limits

This source-only scenario does not establish installed-wheel provenance, native
speech-provider quality, physical microphone input, audible playback, browser
readiness, or complete Slice 0 qualification. The other governed producers and
their human observations remain separate requirements. The supplied Python
environment is a development dependency environment; this result does not attest
the complete DLL or provider-artifact closure of a release installation.

Python-private capabilities prevent ordinary callers from promoting supplied
records or flags into a producer receipt. They do not protect against a hostile
process with the same operating-system identity and interpreter authority.
