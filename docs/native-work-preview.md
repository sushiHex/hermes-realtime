# Native work voice preview

`preview_cutover.py` prepares a source and wheel bound preview, then reuses the
Windows process owner for launch and cleanup. It does not select tools from a
task objective. A `native` plan stages the capability profile through
`native_work_profile`; a `lookup` plan keeps the separate ordinary-lookup
readiness gate and interactive lookup gateway.

The version 1 JSON plan binds the Hermes Realtime source revision/tree, wheel
and installed runtime RECORD, Hermes source revision/tree, profile directory,
model and speech settings, executable paths, and owned/protected ports. Native
plans also provide `native_status` with `command`, `sha256`, `state_dir`, and
`python_user_base` (a path or `null`). The status command hash and existing
state directory are checked before the native profile is staged. Lookup plans
set `native_status` to `null`. Native plans verify `model.default` and
`agent.reasoning_effort` against the declared background settings and require an
empty `agent.reasoning_overrides` mapping; they do not reimplement Hermes
provider-specific override resolution.

The native profile grants full MCP trust only to the reviewed, status-only server;
its exact tool include list excludes model loads and all other server actions.
`VRAM_MCP_AUDIT=0` avoids a separate tool audit payload; native run lifecycle and
this operator's bounded receipts remain available. Both profiles keep manual
consequential-effect approvals. Spoken source names are requested only when the
user asks for them; rich task results retain attribution.

DEV preparation reuses the scoped source/wheel/installed-byte and RECORD
comparator. Full source-archive qualification remains in release CI. Install the
isolated runtime with `uv pip install --no-deps --link-mode copy --target ...` so
it owns its files rather than sharing cache hard links. Git and the existing
interpreter environment remain trusted DEV prerequisites.

The plan has these required top-level keys: `version`, `output`, `realtime`,
`hermes`, `profile_mode`, `native_status`, `speech_assets`, `ports`, `models`,
`speech`, `executables`, and `host_options`. The optional `readiness_policy` is
`kernel_listeners` by default, or explicitly `dev_owned_services`. `speech_assets` has `receipt`,
`moonshine_cache`, and `kokoro_cache` absolute paths. The receipt binds each
cached speech file by size and SHA-256, along with the Moonshine tier and Kokoro
model/voice asset identity. Verified assets are copied into the isolated profile
cache before preflight; Moonshine receives that staged cache through
`MOONSHINE_VOICE_CACHE`, and Kokoro uses its normal profile-local cache path.
Nested keys are strict too:

- `realtime`: `source`, `revision`, `tree`, `wheel`, `wheel_sha256`,
  `runtime_target`, `record_sha256`, `git_executable`.
- `hermes`: `source`, `revision`, `tree`, `python`, `profile_source`.
- `ports`: `api`, `companion`, `livekit`, `host`, `protected` (the first four
  are distinct owned ports; `protected` is a list).
- `models`: `foreground`, `foreground_effort`, `background`, `background_effort`.
- `speech`: `stt_provider`, `stt_tier`, `tts_provider`, `tts_voice`,
  `tts_threads`, `tts_chunk_chars`, `moonshine_version`, `kokoro_version`,
  `onnxruntime_version`.
- `executables`: `codex`, `livekit`, `host_python`.
- `host_options`: `allow_unsandboxed_tasks`, `natural_work_tools`,
  `task_start_policy` (`confirm` or `automatic`), `conversation_profile`.

Source revisions/trees are full lowercase Git object IDs; SHA-256 values are
64 lowercase hex characters. Paths are absolute. Model, effort, voice, tier,
and version strings use the validated label format; the speech providers are
`moonshine` and `kokoro`. The plan contains no auth fields.

Use the existing host Python environment, which supplies the CPU speech packages,
PyYAML, and packaging helpers. The commands below use `$previewPython` for its
Python executable; they do not install dependencies into Hermes.

```powershell
& $previewPython -I -B scripts/preview_cutover.py prepare --plan .\preview-plan.json
# After the previous owner reports a clean normal stop:
& $previewPython -I -B scripts/preview_cutover.py finalize `
  --prepared .\preview\prepared.private.json `
  --previous .\previous-run
# Launch independently of the calling terminal or agent controller:
& $previewPython -I -B scripts/preview_cutover.py launch --active .\preview\active.private.json
# Later, request graceful stop and wait for cleanup settlement:
& $previewPython -I -B scripts/preview_cutover.py stop --output .\preview
```

`prepare` and its `preflight` are offline. `finalize` refuses unless the prior
run has a settled stop receipt, no owned survivors/listeners, and an empty task
journal. `launch` probes the bound host interpreter and starts its verified base Python
with CPython's standard virtual-environment launcher variable. This preserves
the same virtual environment without the Windows venv redirector's extra Job.
It uses the existing windowless Windows Job breakaway flags, NUL input and
exclusive private output files. It reports
only `launch_requested`, not readiness; wait for the owner's readiness receipt.
The actual `serve` process refuses inherited Job membership before creating its
own containment. It also verifies its actual executable image and virtual
environment against the plan-bound interpreter, consuming the launch identity
before service children start. There is no attached-launch fallback when Windows refuses
breakaway or membership cannot be determined. `serve` checks the prepared bindings
again, owns the gateway, LiveKit,
and host children, and reports readiness only after their checks pass. Stop
through the command so the owner can settle children and write its receipt; do
not edit private receipts to advance a run. Ordinary `Start-Process` is not the
persistent launch path. Console hiding alone does not establish lifetime
independence; the windowless flags and actual owner membership check serve
different purposes. See Microsoft's [Job Objects contract](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

On Windows, the preview owner preserves `PROGRAMFILES` as an OS runtime path in
its reduced child environment. NVIDIA capacity queries require it on the
observed host; this does not copy provider credentials. Use the maintained
`launch` command, which resolves the active receipt path and supplies the verified
owner-runtime binding; direct `serve` without that binding refuses.


## DEV service readiness

The source-bound plan can explicitly select `readiness_policy: dev_owned_services`
for a trusted local development account. This checks authenticated gateway health
against the owned gateway process tree and containment, then checks the exact host
readiness receipt against its owned process tree and containment and requires HTTP
readiness. Host startup already completes its authenticated LiveKit verifier room
connection before writing that receipt. The owned LiveKit process must stay alive.
Actual conversation media connects later when the browser session starts.

This DEV policy uses bounded loopback TCP availability checks for planned ports
before startup, during readiness, and after owned cleanup. It cannot establish
complete listener ownership or exclude every competing listener. Readiness and
stop receipts therefore report `listener_ownership: unavailable`;
`remaining_owned_listeners` and `protected_listeners_unchanged` are `null`.
Separate `remaining_accepting_endpoints` and `protected_endpoints_unchanged` fields
report TCP availability only. Protected-endpoint comparison is optional metadata;
when unavailable it is `null` and establishes neither personal-service identity
nor continuity. It does not authorize cleanup of another process.
Process containment, source/runtime/profile binding,
cooperative stop and saved-state preservation still apply.

A DEV stop receipt can finalize another explicitly selected DEV preview after
owned processes and accepting endpoints are gone. It cannot finalize a strict
preview or satisfy release qualification. Existing plans retain strict kernel
listener checks, and generic qualification/release code remains strict.
