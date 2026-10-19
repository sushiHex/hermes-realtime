# Desktop MVP acceptance

Use this procedure for the integrated operator result owned by
[#159](https://github.com/sushiHex/hermes-realtime/issues/159). The
[diagnostic guide](desktop-mvp-diagnostic.md) remains the setup, marker and
step reference. An automated rehearsal or a green CI run cannot establish
physical audibility, microphone behavior or the operator's experience.

## Freeze before observing

Record the candidate commit and installed wheel identity, the pinned Hermes
commit, foreground and Hermes providers/models, speech models, browser version,
and microphone/speaker classes. Use synthetic conversation data. Preserve the
existing personal profile and unrelated processes. A separate test home must
have its own state and loopback endpoints, not a copied personal memory store.

Bind the install to the candidate by equality, not by its version string:

1. Build one wheel from the clean frozen checkout and record its SHA-256.
2. For both the host and Hermes interpreters, compare the installed package and
   distribution metadata payload with that wheel's manifest: exact file set,
   content hashes and sizes. Read and hash actual files; a copied `RECORD` alone
   is not proof. Account explicitly for installer-generated artifacts such as
   bytecode and installation metadata; unexpected package files are a refusal.
3. After that comparison passes, record each install's `RECORD` digest. Require
   the gateway's authenticated runtime attestation to equal the verified Hermes
   install's digest, and run the host from the verified host install. A changed
   install requires comparison and process restart before observations.

The installed gate's version/install-kind report and gate-to-gateway attestation
equality do not perform step 2. A same-version stale wheel fails this binding.
Publish only the candidate/artifact digests, comparison counts and outcomes.

### Isolated-session setup overrides

The diagnostic guide describes a single default installation. For a separate
test installation, these rules override its setup and process-control examples:

- Set `HERMES_HOME` only in this session's child-process environment. Install the
  pinned checkout and its own `venv` under that home, and install the frozen wheel
  there. Do not switch the user's profile, overwrite their configuration, or run
  the guide's global gateway `stop`, `uninstall` or process-name kill commands.
- Allocate separate available literal-loopback API and companion ports. Pass
  the test home and its API URL to the installed gate; pass the same API URL and
  the test home's `.env` to the host. Keep the test voice tail and run record in
  separate state paths. A refused API or companion bind is a setup finding;
  select another free port instead of terminating its listener.
- Reuse the pinned shared LiveKit executable through the existing verified
  resolver. This procedure uses the documented default LiveKit/host ports; it
  does not add an alternate LiveKit launch mode. If either is occupied, mark
  setup `not_run` and defer the session rather than replacing its listener.
  No new binary path or firewall rule is needed for this session.
- Start owned long-lived processes without visible console windows, with stdin
  disconnected. Keep exact process ownership for restart and cleanup; restart
  only this session's host and stop only this session's gateway/LiveKit. A healthy
  preexisting gateway remaining at cleanup is expected, not a leak.

Do not execute a diagnostic recovery instruction against the personal gateway
when a test endpoint refuses. Diagnose and correct only the isolated setup.

These separate-home and process-ownership rules take precedence over the
diagnostic guide's default-home and gateway-stop examples. Follow it for the
detailed Hermes, wheel, `.env`, and LiveKit setup, but keep `HERMES_HOME` in the
session's PowerShell process environment. Do not run its global `hermes gateway
stop`, `hermes gateway uninstall`, or process-name kill examples; preserve any
personal gateway. Start and restart only the gateway installed under the test
home, and stop only processes owned by this session.

For the candidate in the current [#159 acceptance handoff](https://github.com/sushiHex/hermes-realtime/issues/159#issuecomment-6275503278),
record Phi at context 4096, Ollama 0.40.4, and the pinned Hermes commit
`29112bef099274229cadff79cdff7bf7b99c4b77` as the intended profile. This is a
candidate selection for qualification; it does not change project defaults or
claim that human acceptance has passed. Record the exact model identifier and
effective context reported by the host. Use the already-qualified Ollama
installation and the model name shown by `ollama list`; do not infer the model
name from “Phi.”

After the installed gate and wheel equality check pass, launch the verified
host installation's console script directly. The wheel defines
`hermes-realtime-host` as its Windows console entry point; `<host-venv>` must be
the interpreter whose installed package was compared with the candidate wheel.
The following shows the executable and arguments; start it under the hidden,
stdin-disconnected process ownership rules above:

```powershell
$env:HERMES_HOME = "<test-home>"
& "<host-venv>\Scripts\hermes-realtime-host.exe" `
  --hermes-env-file "$env:HERMES_HOME\.env" `
  --inference-provider ollama `
  --ollama-model "<exact name from ollama list>" `
  --ollama-num-ctx 4096 `
  --persistent-loopback-launch `
  --allow-unsandboxed-hermes-tasks
```

Keep the test home's API URL, separate free API and companion ports, and
session-owned state paths from the isolated setup above. Do not substitute the
diagnostic guide's `uv run` checkout command for this installed-host launch.

Run the installed-runtime gate from the same clean candidate, with its installed
interpreter, before starting the host. A stand-in Hermes model does not complete
[#67](https://github.com/sushiHex/hermes-realtime/issues/67)'s real-model gate.
Record refusals as well as passes; do not describe a different installed Hermes
revision as the pinned baseline. If a prerequisite fails, keep later results
diagnostic and leave acceptance pending.

Use the diagnostic guide's initial profile: Windows Chrome, literal loopback,
built-in memory, evidence capture and public search off, and explicit task
commands with natural work tools disabled. Acceptance of
[ADR 0004](adr/0004-model-action-authority.md) records automatic natural task starts
and explicit user task cancellation. It requires no confirmation click for a
model-proposed start and grants the model no cancellation or downstream approval
authority. Design acceptance does not establish implemented routing, actual-model
qualification, live activation or physical voice acceptance. The separately
scoped implementation and its candidate-bound evidence belong to
[#226](https://github.com/sushiHex/hermes-realtime/issues/226) and
[#248](https://github.com/sushiHex/hermes-realtime/pull/248).

A later automatic-routing acceptance session must first freeze a separately
reviewed and qualified candidate, its natural-work setting, model artifact and
provider/runtime configuration. Its observations extend this procedure: appropriate
natural requests start work once with the intended objective; conversation,
explanation, quotation, negation and instruction-shaped reference data are measured
separately for unintended starts. Record denominators, failures and latency.
Require an accepted acknowledgment with a run ID before active status, and test
exact explicit-user cancellation with unrelated work active. Model cancellation
and approval must have zero effect. Speech interruption must leave admitted
background work independent. A finite corpus or stand-in pass does not prove
arbitrary intent recognition, injection resistance or operator acceptance.

Before starting, declare the observations: final transcript to first token,
first token to audio, task acknowledgment/completion/cancellation, and reconnect
to Listening. Record the displayed milliseconds or measured seconds and sample
counts. Do not invent a pass threshold after seeing the values. Any product
latency target must be recorded in #159 before it is used as an acceptance test.
Record the operator's responsiveness assessment separately from those timings.

## Observe the existing steps

Follow [session steps 1–10](desktop-mvp-diagnostic.md#the-diagnostic-session),
recording `as_expected`, `different`, `failed` or `not_run` for each. In particular:

- Confirm actual microphone input and audible typed/spoken replies. Synthetic
  browser audio is not a substitute for either observation.
- Exercise the spoken command `start task <objective>` and the typed `task:`
  form. For cancellation, say `cancel task` when exactly one acknowledged task
  is active; when there are zero or several, verify the no-task guidance or use
  **Cancel** on the intended task card. Also exercise the typed
  `cancel task: <task-id>` form. Record accepted task acknowledgment before
  active status, exact cancellation and terminal settlement, and speech
  interruption leaving unrelated work active. The spoken grammar and its
  ambiguity behavior are detailed in [local-livekit.md](local-livekit.md#full-loopback-host-and-constrained-desktop-human-gate).
- Observe Stop/Connect, reload during the readiness cue, and host restart.
  Check context without restating the fact being tested. Record a recall failure
  even when the corresponding history is present.
- Exercise deletion only on the synthetic test conversation. Verify the stated
  archive scope and pending/complete transitions, with the guide's explicit
  limits: memory, skills and delegated tasks remain; no unlearning or physical
  erasure is promised. A model's failure to recall a phrase is not proof that
  the archive was deleted.
- Stop only processes owned by this session and verify their cleanup. Preserve
  preexisting gateways, services and conversations.

If the candidate changes, start a new candidate-bound record. A successful
repeat does not erase the first outcome. An unexercised step stays `not_run`.

## Record and disposition

Post the installed gate's bounded record to #67 and the operator record to #159.
Use the diagnostic guide's record sheet: outcomes, categories, counts, timings
and approved content-free markers only. Add a sample count for each named timing
and identify whether it came from the browser or an operator measurement.
Do not publish transcripts, audio,
memory, objectives, credentials, launch links, personal paths or identifiers.

The operator supplies hearing/speaking observations; automation supplies only
what it actually measured. Link any defect to its owning issue. Preserve the
separate dependency/security disposition, memory-provider egress check and
contributor setup check required by #159.

This record establishes only the tested desktop profile. It neither qualifies
Tailscale/mobile operation nor replaces the governed physical capture family,
release/tag approval, or the remaining Slice 0 gates.
