# Experimental ordinary-lookup backend setup reference

This is the explicit, disabled-by-default backend configuration delivered by
[#301](https://github.com/sushiHex/hermes-realtime/pull/301), not a foreground
model nomination or natural-routing activation. It extends the isolation and
install-binding rules in [desktop acceptance](desktop-mvp-acceptance.md).
[#295](https://github.com/sushiHex/hermes-realtime/issues/295) still owns the full
selected-profile contributor walkthrough. The
[public backend command](lookup-backend-command.md) now composes these settings,
scoped realtime wheel/target equality, owned hidden launch, one production-adapter
task and cleanup. Use that command rather than an unpublished operator helper.
Its backend-only execution and source/semantic evidence must be recorded before
claiming a walkthrough passed; it does not qualify the foreground profile or a
fresh model download.

## Freeze the source and artifacts

Use a clean public checkout. The delivered main commit is
`f0312425a6b651317a0f4183e4c6d76cd915f85e`; its
[five post-merge checks passed on attempt 1](https://github.com/sushiHex/hermes-realtime/actions/runs/40158579344).
Official Hermes is pinned to
[`29112bef099274229cadff79cdff7bf7b99c4b77`](https://github.com/NousResearch/hermes-agent/tree/29112bef099274229cadff79cdff7bf7b99c4b77).
Use the [existing install instructions](desktop-mvp-diagnostic.md#setup), with
the desktop acceptance guide's isolated-home overrides, and compare actual
installed files with the frozen wheel. The
[installed API gate](hermes-bridge.md) observes that setup.
A version string, copied RECORD or gateway digest alone is insufficient.

The historical successful task and the separate public reproduction input have
these bindings:

| Component | Binding |
| --- | --- |
| Realtime source for the task | `26c7f54d411d74d4c4ad6f44fcc5608b43593508` |
| Source tree for that task | `fff5b8fa50e58d7264d2e0073297ed1d624e1c33` |
| Background and presentation model | `devstral-small-2:24b` |
| Installed model manifest SHA-256 | `24277f07f62db8f9cb68e9dfc679ea1818a7fbac47a50eff0a701d3f645b63c8` |
| Historical retained Ollama runtime closure | `d1cb1fb752fa1181fb2ad4e4c6fa01dd55457ca8340da3b3bc5f4d3d4435d36d` |
| Public reproduction archive pin | v0.40.4, `7485a1172f8527db4342a63521b9342fc466185ff0855da378cc864ecf6354cf` |

The historical task verified the retained runtime closure, not its equality to
that ZIP. A later comparison of the retained directory with a freshly verified
official archive refused `runtime-content`: the retained directory contained
extra files not in that archive, while the selected executable digest matched.
Changed bytes in the shared files have not been established; the historical
directory was preserved. No causal claim follows from this inventory difference.
The public command used a fresh extraction instead. Neither the archive pin nor
the historical useful task is relabelled as proof of the new command's outcome.

The model name is a mutable tag. Before loading, require the exact manifest,
config and complete referenced layer closure by SHA-256 and byte size, then
compare owned-server metadata. A current tag or advertised tool support is not
identity proof. A clean public retrieval walkthrough for this exact closure has
not been performed by this documentation change. The historical Qwen manifest
retrieval failure remains recorded in the desktop guide; Devstral is an explicit
backend replacement, not evidence that the old Qwen artifact became retrievable.

The task reused an earlier installed realtime payload after exact equality of
112 source files and 122 digested RECORD rows. It did not build a new wheel or
obtain a new import attestation. Final main also includes a separately tested
typed-confirmation admission-order fix. Neither the old install receipt nor the
task is relabelled as an execution of final main. A new installation must bind
its own wheel, files, imports and authenticated gateway digest to its candidate.

## Stage only into a fresh owned profile

Create an empty disposable directory outside the checkout and personal Hermes
home. Run this preparation from the frozen repository root with a prepared
Python interpreter; replace the placeholder before running:

```python
import json
from pathlib import Path
from scripts.stage_lookup_policy import stage_lookup_policy

profile = Path("<fresh-owned-profile>")
profile.mkdir(exist_ok=False)
receipt = stage_lookup_policy(
    profile, Path("tests/fixtures/plugin_generation_candidate")
)
print(json.dumps(receipt, sort_keys=True))  # public hashes and fixed labels only
```

The [stager](../scripts/stage_lookup_policy.py) requires an empty directory and
verifies exact reviewed entrypoint/policy bytes. It creates only four plugin
files under `plugins/ordinary-lookup-experimental`. It supplies a minimal native
bootstrap and experimental manifest. Never copy the fixture's synthetic
`__init__.py` or offline-only manifest. Preserve a partial profile on failure;
do not bypass the emptiness check to update a profile. Add config and credentials
only **after** staging. The plugin is not installed in a user's home.

## Configure the dedicated backend

Use separate free literal-loopback model/API/companion ports. Set `HERMES_HOME`
only in owned children. The model server uses the intact archive closure and a
dedicated model store, `OLLAMA_CONTEXT_LENGTH=64000`,
`OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_FLASH_ATTENTION=1`,
`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NO_CLOUD=1`,
`OLLAMA_NOPRUNE=1`, and `OLLAMA_DEBUG_LOG_REQUESTS=0`. Set `OLLAMA_HOST` to its
owned loopback endpoint and `OLLAMA_MODELS` to that store. Remove inherited
`OLLAMA_*`/`LLAMA_ARG_*` overrides before setting this child environment; leave
`OLLAMA_GO_TEMPLATE` and `LLAMA_ARG_CONTEXT_SHIFT` unset. Do not retune or stop a
personal server. Metadata readiness is not inference readiness or memory proof;
64,000 is the configured context, not a claim of spare capacity or all-GPU fit.

The isolated `config.yaml` uses matching native and presentation routes. Replace
the numeric port placeholder in both locations with the same owned model port:

```yaml
approvals:
  mode: manual
  unattended_mode: deny
auxiliary:
  title_generation:
    enabled: false
model:
  provider: custom
  default: devstral-small-2:24b
  base_url: "http://127.0.0.1:<model-port>/v1"
  api_key: local
  context_length: 64000
platform_toolsets:
  api_server: [web]
plugins:
  enabled: [hermes-realtime, ordinary-lookup-experimental]
  entries:
    hermes-realtime:
      allow_tool_override: false
    ordinary-lookup-experimental:
      allow_tool_override: true
      settings:
        presentation:
          provider: custom
          base_url: "http://127.0.0.1:<model-port>/v1"
          model: devstral-small-2:24b
          api_key_env: HERMES_LOOKUP_PRESENTATION_API_KEY
web:
  cache_enabled: false
  search_backend: exa
  extract_backend: exa
  extract_char_limit: 12000
  keyless_fallback: true
  keyless_rescue: false
  provider_tier:
    exa: free
    firecrawl: paid
    keenable: paid
    parallel: paid
```

The pinned native web tools perform actual retrieval; no proof fixture handler
is installed. The native realtime plugin must come from the verified realtime
installation, not this four-file staging operation. Keep a fresh profile so an
old provider-resolution cache cannot mask changed context or route settings.
The qualified profile omitted memory settings: built-in enabled defaults and
no configured external provider are configuration evidence, not runtime memory
readback or proof that nothing left the machine.

Put a freshly generated strong `API_SERVER_KEY` in the isolated `.env`; never
reuse personal credentials or put the bearer in command arguments. Set
`HERMES_LOOKUP_PRESENTATION_API_KEY` in the same child environment; for this
local endpoint its value is the nonsecret placeholder `local`. It has no ambient
credential fallback. Companion credentials and port, if used, are separately
generated and isolated. Keep raw config, environment and gateway logs private.

## Select the production adapter explicitly

Use the existing Python composition seam, not an extra per-event field:

```python
from hermes_realtime.integration import HermesApiConfig

config = HermesApiConfig(
    base_url=owned_api_origin,
    bearer=isolated_api_bearer,
    lookup_policy="ordinary-lookup/v1",
)
```

Supply it to `HermesApiTaskSession(config=config, ...)` through the existing
conversation controller. Alternatively `build_local_host_launcher` exposes
`hermes_lookup_policy="ordinary-lookup/v1"` for its API adapter. Also select
`hermes_model_source="gateway"` in that Python composition so background runs
use the gateway's configured model route, reasoning and service tier, independently
of the foreground model. This delegates selection; it does not attest the model's
identity. The default `"foreground"` preserves the existing Codex overrides.
The voice-host CLI exposes neither selection. The separate backend setup command
selects its own gateway route. Every run through that configured
adapter is designated; use a dedicated deployment. `None` preserves ordinary
native runs. Keep native `run_provider`, `run_model`, `run_reasoning_effort` and `run_service_tier`
unset for this profile so Hermes uses the configured route/default reasoning.
The separate presentation call sends `reasoning_effort="none"`, no tools,
nonstreaming strict JSON and a 1024-token cap. Unsupported controls refuse;
there is no repair call or fallback.

Before dispatch the adapter authenticates readiness and binds the generation.
It requires the real accepted run's policy acknowledgment and authenticated
final policy status. Do not replace this with raw POST success or native
`completed`. Unknown/missing/mismatched authority refuses while preserving
claim, settlement, recovery and cancellation of a possibly accepted run.
Pending recovery resends its persisted policy, not today's adapter setting.

The exact accepted nonblank objective (at most 1024 characters) becomes one
explicitly host-origin search REQUEST. Hermes's normal executor runs the tool;
the request is neither a model inference nor a retrieved result. Subsequent
provider work resumes only after actual successful same-run retrieval. Existing
search/extraction tickets, approvals and the original 120-second admission
deadline remain authoritative. Source IDs select exact catalog URLs for detail;
voice need not speak a URL. Source eligibility does not itself prove factual
support: inspect retained results and answer separately.

## Execution and remaining walkthrough boundary

Start only the pinned Hermes gateway, using `hermes gateway run`, under the
existing [Processes/Containment owner](../scripts/rehearse_desktop_mvp.py).
Use hidden Windows children, stdin disconnected, bounded startup and an outer
deadline. A listening socket alone is not HTTP readiness. Verify authenticated
API/runtime/policy readiness before the single admitted task; stop the owned
gateway gracefully before forced settlement, then stop only the owned model
server. Record final process/listener absence and unchanged unrelated services.
No process-name kill, personal profile update, shared listener replacement or
prewarming is part of this procedure.

The startup helper's empty load selects `keep_alive=-1`, retaining the model
through idle periods until the dedicated owned server stops. Later requests
without a retention override preserve that lease in the
[pinned scheduler](https://github.com/ollama/ollama/blob/v0.40.4/server/sched.go#L495-L505).
The caller must retain its Job, watchdog and final settlement for the entire
consumer session. This prevents idle expiration; it does not prove future runner
survival or replace process/listener cleanup.

Startup timing remains separate from the model/context profile and the task's
120-second authority. `start_model(..., startup_timeout_seconds=300)` explicitly
selects a bounded cold-start allowance in an existing Python composition; accepted
values are exact integers from 1 through 600. `None` preserves the 60/120-second
profile defaults, and the caller's earlier absolute deadline always wins.
Requested and effective startup budgets appear in the receipt. The separate
public backend command keeps its existing 60-second startup and 360-second watchdog.

For an interactive session, prove background startup and task readiness before
inviting the operator. Reserve an owner window for cold startup, operator use and
cleanup; publish the ready URL with an explicit usable end time, then hand control
to the operator promptly. A URL published after scheduled shutdown is unavailable.
An earlier load acknowledgment alone does not establish later consumer reuse.

The desktop rehearsal remains synthetic. The separate
[public backend command](lookup-backend-command.md) now composes this real-provider
profile, the scoped supplied-target comparator and the existing owner. It also
migrates the qualification's bounded Exa search/extraction transport guards and
the separate exact same-model-origin metadata GET lane. These are additional
harness bounds, not native tool-policy guarantees; the config above alone does
not install them. The API gate remains an observer, not a new supervisor or
alternate dispatcher.

The prepared Hermes interpreter/dependencies remain trusted inputs, and the
command's successful structural completion still requires separate answer/source
review. Record any actual public-command walkthrough against its exact candidate;
do not reuse the historical task as that observation. Fresh full-profile setup,
foreground and human acceptance remain under #295/#159.

The [recorded backend task](https://github.com/sushiHex/hermes-realtime/issues/337#issuecomment-6392752249)
completed in 34.297 seconds, with one search, one permitted extraction, genuine
SSE completion, source review and clean owned shutdown. This is one stable task,
not reliability, broad model compatibility, natural routing, voice acceptance or
[#315 resource qualification](https://github.com/sushiHex/hermes-realtime/issues/315).
Readable punctuation remains a separate [#338 follow-up](https://github.com/sushiHex/hermes-realtime/issues/338).
