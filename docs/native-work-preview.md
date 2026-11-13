# Native work voice preview

`preview_cutover.py` prepares a source and wheel bound preview, then reuses the
Windows process owner for launch and cleanup. It does not select tools from a
task objective. A `native` plan stages the capability profile through
`native_work_profile`; a `lookup` plan keeps the separate ordinary-lookup
readiness gate and interactive lookup gateway.

## Prerequisites

The preview runs on Windows. It needs no Hermes core patch: Hermes v0.20 and later
provide the `/v1/runs` surface the bridge uses, and the additive patch applies only
to v0.19 ([bridge requirements](hermes-bridge.md)). Start from the
[example plan](../scripts/preview-plan.example.json) and the redacted
[profile skeleton](../scripts/native-profile-config.example.yaml). The example is a
CPU Kokoro plan: it has no managed GPU worker fields (`kokoro_mcp_config`), so it
does not reproduce an operator's opt-in GPU worker profile. The skeleton is JSON,
which is also valid YAML, and holds no provider or credential fields. Digests and
paths are placeholders; a test keeps both accepted by the plan parser and the native
profile validators, and checks that they still refuse a missing field or mismatched
model.

| Interpreter | Plan field | Role and requirements |
| --- | --- | --- |
| Host Python (`$previewPython`) | `executables.host_python` | Runs this script and the host. Needs `moonshine-voice`, `kokoro-onnx` and `onnxruntime` at the plan's `speech` versions and no `onnxruntime-gpu`, which `prepare` checks, plus PyYAML and `packaging`. An opt-in managed Kokoro GPU worker runs in its own environment. |
| Hermes Python | `hermes.python` | The Hermes installer's environment. Install Hermes at an exact commit with the [recorded installer procedure](desktop-mvp-diagnostic.md#1-install-hermes-at-the-exact-commit). |
| Isolated runtime | `realtime.runtime_target` | The hermes-realtime wheel installed with `--no-deps` as described below; the host and preflight import it from this target. |

Two Hermes pins have different roles. `29112bef` (v0.21.0) is the owner-selected
qualification baseline used by the installed-boundary gates and the recorded
diagnostic (`scripts/real_gate_support.py`). `3c27eb62` (release v2026.8.3) is the
reviewed publisher source for release-candidate qualification
(`scripts/qualification_hermes_source.py`). The plan binds the revision and tree you
actually install.

The source profile's `config.yaml` must match the plan's background model and
effort, keep `agent.reasoning_overrides` empty, use manual approvals with
unattended consequential effects denied, and set the `hermes-realtime` plugin entry's
`allow_tool_override` to `false`; staging keeps only that plugin. It must not select
an external memory provider: at the pinned Hermes baseline `memory.provider` alone
selects one, and an absent key or its empty-string default means built-in memory only.
Staging and preflight refuse a non-empty `memory.provider` with the fixed error
`memory_provider_selected`, and a `memory` section or `memory.provider` of the wrong
exact type with `memory_provider_refused`. The category alone, never the provider's
name or settings, is retained as `failure_category` in `prepare`'s
`preparation-refused.private.json` and in the preflight `[voice-preview]` status line.
The `prepared.private.json`
receipt records `memory_provider_config_inspection` (`absent` or `builtin`). This is a
configuration inspection only; no runtime readback of the gateway's effective memory
state exists. Configure the model provider with `hermes setup`; the
optional live-Codex `key_cmd` route is described in
[live credentials](live-credentials.md). `native_status.command` is the launcher of
the public [vram-mcp](https://github.com/sushiHex/vram-mcp) server; the staged
profile exposes only its `vram_status` tool.

**Codex CLI.** `executables.codex` is the absolute path to an installed Codex CLI
executable. Release qualification verifies `rust-v0.145.0`, whose Windows package
(`codex-package-x86_64-pc-windows-msvc.tar.gz`) it admits by archive SHA-256
(`scripts/qualification_codex_files.py`); the preview itself does not check the CLI
version, and the repo documents no install command. Sign-in is your
own, done through the CLI (`codex login status` reports it); the repo documents no
other login step. The live-credential route requires file-based CLI credential
storage, and keyring-only storage is unsupported ([live credentials](live-credentials.md)).
The plan's `models.foreground` and `models.background` names (the example plan uses
`gpt-5.6-terra` and `gpt-6-luna`; the host launcher's `--codex-model` defaults to
`gpt-5.6-terra`) must be available to the signed-in account.

**vram-mcp.** Install [vram-mcp](https://github.com/sushiHex/vram-mcp) from a checkout
of that repository at a commit you record; at the time of writing its `main` head is
`5c88cc6c8215d7adbbab1003632bb4765af4bdc6`. That is a pin to copy, not a qualification
of that revision. The package installs a `vram-mcp` console script, and
`native_status.command` points at the installed launcher. Record its SHA-256 yourself
as `native_status.sha256`: the launcher is generated by the installer, so its bytes are
specific to each installation. `native_status.state_dir` must be an existing directory.
`python_user_base` is `null` unless the install is a Python user-base install, in which
case it must also be an existing directory (`scripts/native_work_profile.py`).

**Realtime source, wheel and runtime target.** `realtime.source` must be the checkout
whose `scripts/preview_cutover.py` you run, with `HEAD` at `realtime.revision`
(`git rev-parse HEAD`) and `realtime.tree` (`git rev-parse "HEAD^{tree}"`). Clone
with history, not `--depth 1`: the source check also reads the parent commit. The
checkout must be completely clean, untracked files included: keep the plan, the
output directory and every other working file outside it (`dist/` is ignored, so the
built wheel may stay there). Build the wheel with `uv build --wheel`.
`realtime.wheel_sha256` is the SHA-256 of that wheel, and `realtime.record_sha256` is
the SHA-256 of `hermes_realtime-<version>.dist-info\RECORD` inside
`realtime.runtime_target` after the `--no-deps` install described under
[Plan reference](#plan-reference); `Get-FileHash -Algorithm SHA256` gives both, in
upper case, so lower-case them. The runtime target may hold only that install. A
bytecode cache there is an unexpected file and refuses preparation: leave
`UV_COMPILE_BYTECODE` unset when installing, and run everything that imports from the
target with `-B`.

**Hermes source and profile.** `hermes.source` must be at `hermes.revision` with no
modified or untracked Python files. Keep no `.env` in that checkout: the gateway reads
one there for any key the profile's `.env` lacks. `hermes.profile_source` is a curated
directory, not a live Hermes home. It may contain only `config.yaml`, `.env`,
`SOUL.md`, `skills`, `hooks` and `plugins`, without links, within 4096 files and
64 MiB. `finalize` refuses any other entry, including a Hermes `auth.json`, so a
provider sign-in stored there cannot be carried: supply the background model's
credential through `.env` or the `key_cmd` route. `prepare` reads only `config.yaml`;
the other entries and the required `.env` are checked at `finalize`, so an accepted
`prepare` does not show that the profile source is acceptable.

The `.env` is copied unread into the staged profile by `finalize`. It must hold one
`API_SERVER_KEY` of at least 32 random characters, `HERMES_REALTIME_COMPANION_PORT`
equal to the plan's `ports.companion`, and a `HERMES_REALTIME_COMPANION_TOKEN` of 24 to
512 random characters. Generate the secrets as the
[diagnostic guide](desktop-mvp-diagnostic.md#5-configure-hermes_homeenv) describes and
never reuse personal credentials. Do not copy that guide's `API_SERVER_HOST` and
`API_SERVER_PORT` lines: Hermes lets `.env` override the host and port the owner
passes from the plan, and the preview then waits for a gateway that listens elsewhere.

**LiveKit and ports.** `executables.livekit` must be the shared verified server that
`uv run python -m scripts.local_livekit path` prints
([pinned development server](local-livekit.md#pinned-development-server)); the owner
refuses another copy as `livekit_path_binding` after `launch`. The owner starts its
own LiveKit process on `ports.livekit`. All
four owned ports must be free at launch, so stop a development LiveKit server on the
same port or choose another port. `output` must not exist yet and its parent directory
must.

### Speech caches

`prepare` copies speech assets only from the two existing cache directories the plan
names. It downloads nothing, so populate them once.

**Moonshine.** Download the model for the plan's `speech.stt_tier` into the directory
the plan names as `speech_assets.moonshine_cache`, using the host Python:

```powershell
& $previewPython -I -B -m moonshine_voice.download --stt --language en `
  --model-arch 5 --root "<moonshine-cache>"
```

`--model-arch` takes the package's numeric streaming architecture: `2` for `tiny`,
`4` for `small`, `5` for `medium`. The command needs network access to
`download.moonshine.ai`. It checks each file against the size and CRC32C in the
package's own catalog, and also tries to fetch the English spelling model. Files land
under `<moonshine-cache>\download.moonshine.ai\model\`, the layout the host later reads
through `MOONSHINE_VOICE_CACHE`. With a complete cache the host start downloads nothing
for Moonshine; if the spelling files are missing, the package tries to fetch them at
host start and continues without them. `prepare` does not check that the cache holds
the plan's tier, so download only that one architecture into a dedicated directory; a
cache for another tier is found only at host start. This repository does not pin
Moonshine model hashes: the receipt pins the bytes you downloaded. Before writing the
receipt, check
that an interrupted download left no `.partial` or `.lock` file in the directory; an
empty file is refused as `speech_asset_size`. This step follows from the
`moonshine-voice` 0.1.5 source and has not been exercised on a clean machine here.

**Kokoro.** `speech_assets.kokoro_cache` must be a directory named by the pinned model
SHA-256, `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a`, holding
`kokoro-v1.0.onnx` and `voices-v1.0.bin` from the kokoro-onnx
[`model-files-v1.1` release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1).
A host that has already spoken with Kokoro under this account has them in
`.cache\hermes-realtime\kokoro\<model-sha256>\` under the user profile. Otherwise
download the two files into a directory with that name. Their exact sizes and SHA-256
values are pinned in `src/hermes_realtime/providers/kokoro.py`, and the receipt command
refuses any other bytes.

**Receipt.** `prepare` requires a speech asset receipt at the plan's
`speech_assets.receipt`.
`python scripts/preview_cutover.py make-speech-receipt --plan <plan.json>` writes it
from the caches that already exist; it never downloads anything. Use a dedicated
Moonshine cache directory,
not a shared one: every file in it is admitted and staged. The two cache roots must
not contain each other. The receipt is a version 1
JSON object with `moonshine_tier`, `moonshine_package_version`, `moonshine_cache_root`
and `kokoro_cache_root` (resolved absolute paths equal to the plan's caches),
`kokoro_model_sha256`, `kokoro_model_bytes`, `kokoro_voice`, `kokoro_voice_sha256`,
`kokoro_voice_bytes`, and `files` and `file_sizes`, which map the same absolute
file paths to SHA-256 and byte size. It lists every regular file under the Moonshine
cache and exactly the pinned Kokoro model and voice files, whose directory is named by
the pinned model SHA-256. The command refuses links, pin mismatches including file
name case, an empty Moonshine cache, the consumer's count, size and path bounds, and
an existing receipt (it never overwrites).

### Web tools backend and credential

The native profile gives Hermes background work the `web` toolset (`web_search` and
`web_extract`). Hermes chooses the backend, not this repository, from the `web` section
of the profile's `config.yaml`, which staging copies unchanged. At the `29112bef`
baseline (`tools/web_tools.py`):

1. `web.search_backend` and `web.extract_backend` select a backend per tool.
2. Otherwise `web.backend` selects one for both. If only one per-tool key is set and
   `web.backend` is empty, the other tool uses Firecrawl.
3. With no stored selection, Hermes picks the first available of: an Exa, Parallel or
   Keenable API key; a Firecrawl API key or self-hosted Firecrawl URL; the managed tool
   gateway; a SearXNG URL; a Brave Search key; the `ddgs` package if it is importable;
   then any plugin-registered provider.
4. With none of those, it uses the keyless tier. If `web.keyless_fallback` is `false`,
   it selects Firecrawl, which fails without a key.

A stored selection is never replaced by another configured backend. These keyless
behaviors still apply to it:

- A selected Exa, Parallel, Firecrawl or Keenable backend with no key uses that
  vendor's keyless endpoint, unless `web.provider_tier.<name>` is `paid`. The value
  `free` forces the keyless endpoint even when a key is set.
- A failed keyed call is retried once through the keyless tier, unless
  `web.keyless_rescue` is `false`.
- A keyless call that is rate limited can be answered by another keyless vendor, so
  the backend you selected is not always the one that served a result.

Setting `web.keyless_fallback: false` turns the whole keyless tier off, including the
retry.

A keyed backend needs that provider's own API key, or for SearXNG and self-hosted
Firecrawl a service URL, in the environment variable Hermes names for it. Put it in the
profile source's `.env`. The owner starts the gateway with a reduced environment, so a
key exported in your shell does not reach it. Never put a key in the plan,
`config.yaml`, a command line or an issue.

The keyless tier needs no credential. It sends search queries and requested URLs to
third-party public endpoints without an account or user identifier, from this
machine's network address. A profile with no web selection and no web key therefore
still performs real retrieval.

The owner sets `HERMES_DISABLE_LAZY_INSTALLS=1` for the gateway, so Hermes does not add
packages to its own environment while running. Install any client package the selected
backend needs into the Hermes environment first.

No search backend is pinned or qualified for the native profile. Record the backend you
selected with each observation. Web-only results can additionally be presented from
bounded same-run passages with a validated sources list; that is the separate opt-in
described in [native web evidence](native-web-evidence.md).

### Support limits

- **First launch.** `finalize` is the only command that writes `active.private.json`,
  and it requires `--previous`: an earlier preview run directory with its stop records
  and a saved `state\voice-tail-v1.json`. No maintained command creates that first
  directory, so a machine with no earlier preview run cannot reach `launch` through
  this procedure. The [manual full-host composition](local-livekit.md#full-loopback-host-and-constrained-desktop-human-gate)
  can run a first session, but it is not source and wheel bound, and its state is not
  a `--previous` directory.
- **Clean-machine walkthrough.** These steps were checked against the source. Nobody
  has yet followed them end to end on a clean machine with real installs
  ([#295](https://github.com/sushiHex/hermes-realtime/issues/295)).
- **Moonshine tier.** The plan parser accepts `stt_tier` `base`, but the host supports
  only `tiny`, `small` and `medium`. Do not select `base`.
- **Managed GPU speech.** The example plan is CPU-only. The GPU worker environment,
  the scoped vram-mcp server and its operator profile are set up as
  [managed Kokoro](kokoro-mcp.md) describes; the operator profile's values are
  specific to one machine. The host Python needs the `kokoro-mcp` extra, and `prepare`
  does not detect its absence: it shows only when the host starts.
- **Codex CLI and vram-mcp installs.** This repository documents no install command
  for either, as stated above.

## Plan reference

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
  `onnxruntime_version`; optionally `pronunciations`.
- `executables`: `codex`, `livekit`, `host_python`.
- `host_options`: `allow_unsandboxed_tasks`, `natural_work_tools`,
  `task_start_policy` (`confirm` or `automatic`), `conversation_profile`.

Source revisions/trees are full lowercase Git object IDs; SHA-256 values are
64 lowercase hex characters. Paths are absolute. Model, effort, voice, tier,
and version strings use the validated label format; the speech providers are
`moonshine` and `kokoro`. The plan contains no auth fields.

The optional `speech.pronunciations` object forwards explicit operator term-to-spoken-text
choices to Kokoro's existing pronunciation lexicon. Omission or an empty object keeps
the lexicon empty. It accepts at most 64 case-insensitively unique whole terms, each
1–64 characters, with spoken replacements of 1–128 characters; both must be trimmed
and contain no control characters. The host must support the option when entries are
provided. These substitutions affect synthesis only, leaving displayed text intact;
they do not infer abbreviation meanings or provide phoneme overrides. For example,
an operator can explicitly map `A P I` to `API` without expanding its meaning. Keep
such choices profile-specific; text projection alone does not qualify pronunciation.

## Commands

Setup order, each step using the section named:

1. Install Hermes at the commit the plan will bind, and prepare the profile source
   (Prerequisites).
2. Install the shared LiveKit server, the Codex CLI and vram-mcp (Prerequisites).
3. Build the wheel from a clean checkout and install it into the runtime target with
   `--no-deps` (Plan reference).
4. Populate the two speech caches and write the receipt (Speech caches).
5. Select the web backend and place its credential, if it needs one (Web tools backend
   and credential).
6. For managed GPU speech only, prepare the worker and its configuration
   ([managed Kokoro](kokoro-mcp.md)).
7. Write the plan outside the checkout, then run the commands below from the checkout
   root.

Use the existing host Python environment, which supplies the CPU speech packages,
PyYAML, and packaging helpers. The commands below use `$previewPython` for its
Python executable; they do not install dependencies into Hermes. `<preview-root>` is a
private directory outside the checkout that holds the plan and each run's output
directory; `<output>` is the plan's `output` and `<previous-output>` is the earlier
run's.

```powershell
& $previewPython -I -B scripts/preview_cutover.py make-speech-receipt `
  --plan "<preview-root>\plan.json"
& $previewPython -I -B scripts/preview_cutover.py prepare --plan "<preview-root>\plan.json"
# After the previous owner reports a clean normal stop:
& $previewPython -I -B scripts/preview_cutover.py finalize `
  --prepared "<output>\prepared.private.json" `
  --previous "<previous-output>"
# Launch independently of the calling terminal or agent controller:
& $previewPython -I -B scripts/preview_cutover.py launch --active "<output>\active.private.json"
# Later, request graceful stop and wait for cleanup settlement:
& $previewPython -I -B scripts/preview_cutover.py stop --output "<output>"
```

Each command prints one `[voice-preview]` status line with a JSON `category`. A
refused `prepare` prints only a general `failure_category`; the specific one is in
`<output>\preparation-refused.private.json`. The refused output directory remains, so
use a new one for the next attempt. `launch` starts the owner, whose own output goes
to `owner.private.log` and `owner.stderr.private.log` in `<output>`; both are private
and are kept. The
owner then writes
`ready-receipt.private.json` when the preview is ready and
`result-receipt.private.json` when it has stopped, both in `<output>`. Open
`http://127.0.0.1:<ports.host>/` in a local browser once the ready receipt exists. The
[acceptance procedure](native-preview-acceptance.md) lists what to observe and record
from there.

`prepare` and its `preflight` are offline. `finalize` refuses unless the prior
run has a settled stop receipt, no owned survivors/listeners, and an empty task
journal. A prior run whose owner died before writing any receipt is accepted as
`abandoned_stop_verified` only when every recorded PID, including the supervisor's,
is dead, no port of its hash-bound plan or the new plan is listening, neither host
lock file is held, and its task journal is empty or absent. The host writes that
journal before its first Hermes dispatch, so absence means nothing was submitted.
A receipt that is present but not a settled regular file is refused as before, and
one that appears during the checks is refused as `previous_receipt_appeared`. Either
way, `finalize` carries the voice tail together with its forget record
(`voice-tail-v1.deletes.json`) when one exists, so a forgotten conversation stays
retired. `finalize` only reads the prior run. `launch` probes the bound host interpreter and starts its verified base Python
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
