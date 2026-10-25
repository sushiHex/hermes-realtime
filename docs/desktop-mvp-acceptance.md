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
  there. Do not switch the user's profile, overwrite their configuration, stop
  or uninstall a personal gateway, or kill processes by name.
- Allocate separate available literal-loopback API and companion ports. Pass
  the test home and its API URL to the installed gate; pass the same API URL and
  the test home's `.env` to the host. Keep the test voice tail and run record in
  separate state paths. A refused API or companion bind is a setup finding;
  select another free port instead of terminating its listener.
- Start the API through the pinned Hermes gateway (`hermes gateway run`), which
  enables its approval notification path. A standalone API adapter is not an
  equivalent acceptance setup. Use explicit `approvals.mode: manual`, no YOLO
  override and no command allowlist in the isolated profile; verify that the
  API toolset exposes `terminal` before the installed gate. This tests top-level
  API runs, not delegated subagents, whose approval policy is different.
- Reuse the pinned shared LiveKit executable through the existing verified
  resolver at its documented endpoint; this procedure does not add an
  alternate LiveKit launch mode. If the shared LiveKit endpoint is occupied or
  unavailable, mark setup `not_run` and defer rather than replacing its
  listener. If host port 8765 is occupied, choose a free literal-loopback host
  port with `--port`; the browser origin is derived from that port. Do not pass
  `--browser-origin` in loopback mode. No new binary path or firewall rule is
  needed for this session.
- Start owned long-lived processes without visible console windows, with stdin
  disconnected. Keep exact process ownership for restart and cleanup; restart
  only this session's host and stop only this session's gateway/LiveKit. A healthy
  preexisting gateway remaining at cleanup is expected, not a leak.

These separate-home and process-ownership rules override the default-home and
process-control examples in the [diagnostic guide](desktop-mvp-diagnostic.md).
Follow that guide for detailed Hermes, wheel, `.env`, and LiveKit setup. Diagnose
endpoint refusals within the isolated setup; start, restart, and stop only
processes owned by this session.

The candidate profile follows the [#226 owner decision](https://github.com/sushiHex/hermes-realtime/issues/226#issuecomment-6226820445):
phi4-mini at context 4096 with Ollama 0.40.4 and pinned Hermes commit
`29112bef099274229cadff79cdff7bf7b99c4b77`. This is a qualification candidate;
it does not change the project default or claim that human acceptance has
passed. Record the exact model identifier and effective context reported by the
host. Use the exact model name shown by `ollama list`. Run the foreground's
Ollama 0.40.4 as a session-owned process with `OLLAMA_GO_TEMPLATE=1`; leave
`LLAMA_ARG_CONTEXT_SHIFT` and `OLLAMA_CONTEXT_LENGTH` unset in that process
environment. Do not change a preexisting Ollama service. These are the
source-backed launch preconditions for this Phi artifact's Go-template path,
not a universal model-metadata context guard.

### Prepare the separate background model

The selected two-model profile uses **Phi only for the Realtime foreground** and
**Qwen3.5 9B for Hermes background work**. They run in separate, session-owned
Ollama processes. A foreground context of 4096 is not a suitable replacement for
Hermes's resolved background context of 64000. The pinned Hermes
[minimum-context policy](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/model_metadata.py#L410-L413)
and [route resolution](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/agent_init.py#L2414-L2439)
remain authoritative. [#292](https://github.com/sushiHex/hermes-realtime/issues/292)
tracks a general server-authoritative preflight; these settings describe one
explicit local route and do not replace that resolver.

Use the official [Ollama v0.40.4 Windows AMD64 archive](https://github.com/ollama/ollama/releases/tag/v0.40.4),
`ollama-windows-amd64.zip`, not an installer that replaces an existing service.
Its published archive SHA-256 is
`7485a1172f8527db4342a63521b9342fc466185ff0855da378cc864ecf6354cf`.
Download and verify it before extracting to a new, otherwise empty directory:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/ollama/ollama/releases/download/v0.40.4/ollama-windows-amd64.zip" `
  -OutFile "<test-downloads>\ollama-windows-amd64.zip"
$actual = (Get-FileHash "<test-downloads>\ollama-windows-amd64.zip" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne "7485a1172f8527db4342a63521b9342fc466185ff0855da378cc864ecf6354cf") {
  throw "Ollama archive identity differs"
}
Expand-Archive -LiteralPath "<test-downloads>\ollama-windows-amd64.zip" `
  -DestinationPath "<test-ollama-runtime>"
```

Keep the extracted runtime closure intact and record its actual file hashes;
do not copy just `ollama.exe` or substitute a different runner directory.
The reviewed runtime source is
[`45e6011815c74acde6a346512856b69dd79dd22b`](https://github.com/ollama/ollama/tree/45e6011815c74acde6a346512856b69dd79dd22b).

Model tags are mutable. Require these **manifest** identities, not just matching
model names, file sizes or declared tool support:

| Role | Model name in the frozen install | Manifest SHA-256 |
| --- | --- | --- |
| Foreground | `phi4-mini:latest` | `78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753` |
| Background | `qwen3.5:9b` | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |

For an available installation, hash the actual manifest, its config blob and
every referenced layer blob; require each descriptor's exact hash and byte size.
Compare the owned server's `/api/tags` digest and `/api/show` metadata with those
files. The frozen Qwen config has native renderer/parser `qwen3.5` and no template
layer. Its `/api/show` fallback template is metadata, not an artifact template.
A changed tag, a split projector/template artifact or cloud-backed weights is
a different candidate. Do not silently repull, create an alias or rewrite a
manifest to make it match. Fresh retrieval of the frozen Qwen manifest through
a supported public install command remains a setup prerequisite, not a promise
that `ollama pull qwen3.5:9b` obtains the table's artifact.

The 2026-10-19 metadata-only registry check found a different current tag:
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
Its config and weights changed, and it adds projector/template layers. The
frozen manifest's digest endpoint returned 404, while HEAD requests for its old
config and weights returned 200 with the expected sizes. This establishes
neither complete artifact loss nor a verified installation route. Keep this
retrieval gap explicit; the frozen local install is a diagnostic candidate,
not evidence that a fresh contributor can reproduce it from the current tag.

Reserve separate free literal-loopback ports for the two Ollama servers, API,
companion and host. Use a dedicated test model store; do not pull into, prune or
replace the personal service's store. Build each server's child environment
separately: remove inherited `OLLAMA_*` and `LLAMA_ARG_*` overrides first, then
apply the settings for that role. Do not set these in the user's environment.
The background column records the installed API gate's server settings; the
foreground column describes the prepared voice candidate, not human acceptance.

| Child setting | Phi foreground server | Qwen background server |
| --- | --- | --- |
| `OLLAMA_HOST` | `127.0.0.1:<foreground-ollama-port>` | `127.0.0.1:<background-ollama-port>` |
| `OLLAMA_MODELS` | `<test-model-store>` | `<test-model-store>` |
| `OLLAMA_NUM_PARALLEL` / `OLLAMA_MAX_LOADED_MODELS` | `1` / `1` | `1` / `1` |
| `OLLAMA_NO_CLOUD` | `1` | `1` |
| `OLLAMA_NOPRUNE` / `OLLAMA_DEBUG_LOG_REQUESTS` | `1` / `0` | `1` / `0` |
| `OLLAMA_CONTEXT_LENGTH` | Unset; the host sends `num_ctx=4096` | `64000` |
| `OLLAMA_GO_TEMPLATE` | `1` | Unset; the artifact selects its native renderer |
| `OLLAMA_KV_CACHE_TYPE` | Unset (`f16`) | `q8_0` |
| `OLLAMA_FLASH_ATTENTION` | No additional foreground requirement | `1` |
| `LLAMA_ARG_CONTEXT_SHIFT` | Unset | Unset |

These switches are defined by the pinned
[Ollama environment configuration](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/envconfig/config.go).
Leave speculative-decoding settings unchanged. The profile does not qualify
every draft/cache configuration or transfer the foreground no-truncation
contract to Hermes's background API.

In the fresh isolated Hermes home's `config.yaml`, use this model section with
the background server's actual port. Keep its other setup sections and explicit
manual-approval policy. Do not carry over a named custom-provider route or an
environment override pointing elsewhere:

```yaml
model:
  provider: custom
  default: qwen3.5:9b
  base_url: "http://127.0.0.1:<background-ollama-port>/v1"
  context_length: 64000
```

The frozen candidate omitted the `memory` section. An absent `memory.provider`
leaves the
[external provider inactive](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/agent_init.py#L1905-L1907);
omitted `memory_enabled` and `user_profile_enabled` use the
[pinned enabled defaults](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/tools/memory_tool.py#L1198-L1204).
Record a config-only inspection as such. It does not establish a running
gateway's effective memory state or prove that no data left the machine.
The diagnostic guide's `hermes memory off` / `status` commands belong to
intentional isolated-home setup: CLI startup can perform recovery and provider
discovery, so do not describe an unexecuted status command as a read-only runtime
readback. Do not print provider configuration, memory contents or credentials.

### Public walkthrough boundary

[#295](https://github.com/sushiHex/hermes-realtime/issues/295) owns the exact-profile
walkthrough. The preparation reference above does not supply a complete public
supervisor or claim that walkthrough passed. The existing
[`Processes` / `Containment` utilities](../scripts/rehearse_desktop_mvp.py)
provide hidden, stdin-disconnected Job ownership and bounded forced settlement;
they do not provide this two-model setup as a command. The full rehearsal uses a
stand-in Hermes model and cannot substitute for this real-model procedure.

Before treating this as a contributor-reproducible acceptance setup, the public
path must also provide and verify: retrieval of the exact frozen models; actual
wheel-to-install file equality from the opening steps; owned hidden launch of
both servers, gateway and installed host; graceful host/gateway stop before
forced fallback; and fresh final process/listener verification. A private helper
is not that public path. If any step lacks a published, reviewed mechanism,
record the walkthrough `not_run` with that precise gap instead of improvising
process-name kills or claiming setup completion.

Readiness metadata is not allocation proof: advertised model context and
`/api/ps` options alone do not show the backend's actual per-sequence context or
K/V allocation. Record bounded numeric backend witnesses for the selected run;
raw Ollama output can contain prompts and must not become public evidence.
Individual offload counts and one warm overlap sample do not establish spare
capacity, ordinary-length voice responsiveness or human acceptance. Record
resource headroom and actual latency observations under #159; do not evict
unrelated workloads to make the profile fit.
An explicitly nominated configuration may also use `--ollama-num-gpu` to send a
GPU-layer override through the normal host factory and adapter. Omission keeps
the runtime default; 0 requests CPU layers. The accepted range 0–1024 is an
application input bound, not a model-capability or zero-GPU guarantee. Readiness
and subsequent inference retain the same setting, including browser-selected
local models. A CPU candidate still needs its reviewed artifact, loader,
environment and owned-process closure and separate unchanged qualification.

Qualify the installed Hermes API first, in its own interpreter. Under the
hidden process ownership rules above, use the frozen candidate script:

```powershell
& "<test-home>\hermes-agent\venv\Scripts\python.exe" `
  "<candidate>\scripts\real_hermes_api_gate.py" `
  --hermes-home "<test-home>" `
  --hermes-api-url "<test-api-url>"
```

Leave `HERMES_REALTIME_LIVEKIT_LOCAL` unset for this API gate. Its optional
legacy Codex host-composition probe is a different configuration and is not
required for #67. Do not install the speech stack into Hermes merely to run
that probe. The API gate performs real background inference; its pass is not
voice-host acceptance. Test the selected voice host separately below, using
its own verified installation. Record both models and contexts. Declared tool
support does not establish that a model will produce the required tool call
or approval request.

After the installed gate and wheel equality check pass, launch the verified
host installation's console script directly. The wheel defines
`hermes-realtime-host` as its Windows console entry point; `<host-venv>` must be
the interpreter whose installed package was compared with the candidate wheel.
Replace every placeholder below with this session's values. Set
`--hermes-api-url` to the same owned literal-loopback API URL passed to the
installed gate, and use a dedicated existing `<test-state>` directory for both
state files; the explicit arguments prevent fallback to the personal API or
default state paths. Set `<test-host-port>` to a free numeric loopback port;
the printed browser URL will use that port.
The following shows the executable and arguments; start it under the hidden,
stdin-disconnected process ownership rules above:

```powershell
$env:HERMES_HOME = "<test-home>"
& "<host-venv>\Scripts\hermes-realtime-host.exe" `
  --hermes-env-file "$env:HERMES_HOME\.env" `
  --hermes-api-url "<test-api-url>" `
  --hermes-run-record "<test-state>\hermes-runs-v1.json" `
  --voice-tail "<test-state>\voice-tail-v1.json" `
  --port "<test-host-port>" `
  --inference-provider ollama `
  --ollama-base-url "<test-ollama-url>" `
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
commands with natural work tools disabled. The current [#226 owner decision](https://github.com/sushiHex/hermes-realtime/issues/226#issuecomment-6226820445)
keeps natural routing off and starts work only through deterministic explicit
commands. The model default remains unchanged unless and until this owner
acceptance passes and a separate default-change is made.

[ADR 0004](adr/0004-model-action-authority.md) remains the design record for
model-proposed work. If natural routing is considered again, the [#284 owner
decision](https://github.com/sushiHex/hermes-realtime/issues/284#issuecomment-6226820862)
requires explicit user confirmation of the proposed objective before dispatch.
That future flow is outside this MVP acceptance procedure and needs its own
implementation and qualification; it grants no model cancellation or approval
authority. Current acceptance covers explicit task commands, task-card Cancel,
and interruption independence only.

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
- Observe Disconnect/Connect, reload during the readiness cue, and host restart.
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
