# Prepare and run one isolated lookup backend

This Windows-only experimental command composes the existing process owner,
native plugin stager and production ordinary-lookup adapter. It does not install
software, download models, enable natural routing or start a voice host. A
successful command establishes structural completion and cleanup of one explicit
task; `semantic_review: pending` requires separate review of the retained answer
against its actual sources. It is a backend slice of
[#295](https://github.com/sushiHex/hermes-realtime/issues/295), not completion of
the selected foreground/background MVP profile.

## Prepare the inputs

Use a clean frozen checkout of the public repository. Follow the
[isolated desktop rules](desktop-mvp-acceptance.md#isolated-session-setup-overrides)
for a fresh Hermes home. Its official source commit must be
`29112bef099274229cadff79cdff7bf7b99c4b77`. The supplied interpreter and its
dependencies are trusted prepared prerequisites: the command refuses tracked
Python edits and nonignored untracked Python sources, but does not attest every
ignored dependency/cache file or recreate the environment.

Build one wheel and materialize it into a **new** Windows uv target with the
prepared Hermes interpreter. These commands are preparation, not inference:

```powershell
uv build --wheel --out-dir "<private-inputs>/dist"
uv pip install --python "<hermes-python>" --target "<new-runtime-target>" `
  --no-index --no-deps --no-compile --link-mode copy `
  "<private-inputs>/dist/hermes_realtime-0.0.3-py3-none-any.whl"
```

The comparator checks the realtime package's exact tracked source file set and
bytes against the wheel, then the supplied target's exact package/metadata set,
payload bytes and RECORD. Its supported layout includes uv's `.lock`, two `bin`
console executables and generated dist-info metadata. It is not a generic venv
comparator or a complete dependency-environment attestation. Import origins and
the gateway RECORD/config binding are checked again at execution.

Use the official Ollama v0.40.4 Windows AMD64 ZIP with SHA-256
`7485a1172f8527db4342a63521b9342fc466185ff0855da378cc864ecf6354cf` and its
otherwise unchanged extracted directory. The command compares the whole
authenticated archive with that directory. Do not substitute a copied executable
or omit the ZIP because an older extracted runtime happens to work.

Prepare an Ollama-format model store using the official CLI, outside the personal
service's store. Acquisition must itself use an isolated, owned server and the
existing Processes/Containment lifecycle; do not repoint or stop the personal
server. This command does not perform acquisition. The selected manifest's exact
SHA-256 and every declared config/layer descriptor hash and byte size must match;
a mutable tag alone is insufficient. A different artifact needs its own evidence.
An already acquired exact closure may instead be copied into a separate prepared
store: copy the original manifest and every referenced blob without rewriting
them, then let this command verify all bytes. Do not share writable hardlinks with
the personal store. This is retained-artifact preparation, not a fresh download.

The published example identifies the installed Devstral artifact used by the
successful backend task. A bounded public registry check found the exact manifest
and matching declared sizes at all four descriptor endpoints; it did not download
the model bodies or complete a fresh model-store walkthrough. The historical
Qwen retrieval failure is not overwritten by that result. No fresh download is
claimed by the example.

## Fill the private manifest

Copy [the template](../scripts/lookup-backend-manifest.example.json) outside the
checkout. Replace every path and zero digest. Record the candidate and tree with
`git rev-parse HEAD` and `git rev-parse 'HEAD^{tree}'`; hash the actual wheel with
`Get-FileHash -Algorithm SHA256`. Keep the manifest, output and model contents
private. Paths in the template are examples, not a personal installation.

Use distinct free literal-loopback model/API/companion ports. Port 11434 is
reserved for the untouched personal service and is refused as any selected port.
The model name is an exact local identifier, not a behavior allowlist. The
objective must be a nonblank exact string of at most 1024 characters and is never
rewritten or truncated. Unknown fields, duplicate JSON keys and unsupported
versions refuse.

The generated profile uses matching custom native/presentation routes,
64,000 context, Q8 KV, flash attention, parallel one, manual approvals and the
native Exa search/extraction tools without cache/rescue. Native reasoning uses
the provider default; presentation uses the existing tool-free, nonstreaming
strict schema with `reasoning_effort=none`. No fallback or unsupported-provider
repair is added. The supplied endpoint must support the selected configuration.

Set the manifest's presentation environment variable to the nonsecret value
`local` only in the launching session. Fresh API and companion credentials are
generated inside the new protected output profile; no personal `.env`, SOUL,
memory, sessions or credentials are copied.

## Invoke explicitly

Default preparation verifies inputs and stages the native plugin before adding
configuration. It starts no long-lived process and performs no inference:

```powershell
$env:HERMES_LOOKUP_PRESENTATION_API_KEY = "local"
& "<hermes-python>" -I -B "<candidate>/scripts/setup_lookup_backend.py" `
  --manifest "<private-manifest.json>" --output "<fresh-preparation-output>"
```

Each invocation requires a fresh output directory. To execute after review, use
a different fresh output and add `--execute`. The command must run with the same
prepared Hermes interpreter. Launch it with the existing hidden stdlib boundary,
stdin disconnected and an explicit outer timeout; for example:

```python
import subprocess

subprocess.run(
    [hermes_python, "-I", "-B", candidate_script,
     "--manifest", private_manifest, "--output", fresh_output, "--execute"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW, timeout=1020, check=True,
)
```

Read-only archive/model preflight has a separate 600-second bound. Actual owned
execution has a 360-second watchdog covering model metadata startup (60 seconds),
gateway startup, one task and settlement. The task retains its original
120-second admission authority; preflight/startup/cleanup time does not extend
that authority. A watchdog/outer timeout is a failed observation with cleanup
unproven, never permission to retry automatically.

The command authenticates the gateway only after OS listener ownership is known,
rejects readiness redirects, checks native policy readiness and requires the real
accepted run's acknowledgment and final authenticated policy status. The initial
search REQUEST is explicitly policy-origin; actual results enter through Hermes's
normal tool executor. The adapter preserves recovery, cancellation and settlement.
Bounded terminal/timeout snapshots are retained before cleanup and cannot promote
a live or refused run to completed work.

Only the two owned long-lived processes are stopped, through the existing owner;
final process/listener settlement and unchanged personal-service listeners are
required. Child output is private, capped at 1 MiB per process and continues
draining after overflow. Read/drain/settlement failures refuse. The output profile
and evidence are preserved for review, including refusals. Public stdout carries
only fixed categories, counts, timings and hashes. Raw source/answer snapshots,
credentials, launch paths and process identifiers must not be posted.

This does not claim graceful remote inference cancellation, physical audibility,
natural routing, model reliability, or resource qualification. Do not silently
repair retained text; punctuation fidelity remains
[#338](https://github.com/sushiHex/hermes-realtime/issues/338).
