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
  separate state paths. A refused bind is a setup finding; select another free
  port instead of terminating its listener.
- Reuse the pinned shared LiveKit executable through the existing verified
  resolver. Refuse occupied LiveKit/host ports rather than replacing their
  processes. No new binary path or firewall rule is needed for this session.
- Start owned long-lived processes without visible console windows, with stdin
  disconnected. Keep exact process ownership for restart and cleanup; restart
  only this session's host and stop only this session's gateway/LiveKit. A healthy
  preexisting gateway remaining at cleanup is expected, not a leak.

Do not execute a diagnostic recovery instruction against the personal gateway
when a test endpoint refuses. Diagnose and correct only the isolated setup.

Run the installed-runtime gate from the same clean candidate, with its installed
interpreter, before starting the host. A stand-in Hermes model does not complete
[#67](https://github.com/sushiHex/hermes-realtime/issues/67)'s real-model gate.
Record refusals as well as passes; do not describe a different installed Hermes
revision as the pinned baseline. If a prerequisite fails, keep later results
diagnostic and leave acceptance pending.

Use the diagnostic guide's initial profile: Windows Chrome, literal loopback,
built-in memory, evidence capture and public search off, and explicit task
commands with natural work tools disabled. Acceptance of
[ADR 0004](adr/0004-model-action-authority.md) is not evidence that its proposed
confirmation control has been implemented.

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
- Observe accepted task acknowledgment before active status, exact task
  cancellation, and speech interruption leaving unrelated work active.
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
