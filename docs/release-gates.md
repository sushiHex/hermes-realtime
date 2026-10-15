# Release gates

Automated candidate acceptance requires all five jobs in the tracked
[Release Gates workflow](../.github/workflows/release-gates.yml). These checks
qualify their exercised scope; human-assisted and installed-service claims retain
the separate gates documented below. See [Implementation status](implementation-status.md)
for the reviewed candidate and remaining qualification.

The Windows release gates build from a verified committed source archive, not
dirty checkout bytes. Ignored build output, virtual environments, and local tool
downloads cannot enter that archive. The pre-archive committed-blob secret scan
still inspects export-ignored paths.

## Required automated checks

| Job | Responsibility |
| --- | --- |
| Pure candidate wheel | Verify the Linux browser build and committed asset parity, then build the candidate wheel and hash-identified offline dependency closure. |
| Linux null capture | Install that wheel offline and verify the Linux null-capture boundary. |
| Hermetic release candidate | Run the Windows source, browser, packaging, and isolated-installation gates. |
| Native LiveKit release integration | Run the Windows gates with pinned LiveKit, the [archived source-equivalence producer](deterministic-equivalence.md), and real-browser self-acceptance. |
| Packaged capture and spool recovery | Qualify the [packaged revocation race](revoke-race.md), [capacity rollover](capacity-rollover.md), [capture admission overflow](over-budget-turn.md), [spool crash recovery](spool-crash-matrix.md), [synthetic fault matrix](synthetic-fault-matrix.md), [full-purge cleanup](full-purge-cleanup.md), and [owned-close faults](owned-close-faults.md) against the Pure candidate wheel. |

Before an authorized merge, require the reviewed candidate's five PR checks to
pass. After merging, wait for all five jobs in the resulting exact-commit `main`
push run to complete on attempt 1 before advancing to the next candidate. Preserve
any failed run and investigate it; an earlier green PR run does not replace the
main push result. Head or base changes require requalification of the new candidate.

Browser checks use Node 22.22.2 on Windows and Linux. The Linux wheel job first
installs the locked browser dependencies with scripts disabled, runs the browser
tests and TypeScript build, and rejects any generated change to committed assets
or their disclosure manifest. This exercises platform-specific compiler packages
before producing the wheel consumed by the Linux installation check.
The Pure job applies the same [canonical wheel serialization](qualification-input-files.md#produce-independent-artifacts)
as the native qualification builds before creating any wheel hash or receipt.

Before provisioning the candidate environment, the gate verifies that `uv.lock`
matches the project metadata. A stale lock fails without being rewritten. The
check explicitly selects the materialized candidate. Gate commands ignore ambient
uv project, working-directory, environment-location, and configuration overrides;
CI cache and interpreter-installation locations remain available.

The `release-candidate` job runs, from that fresh candidate:

1. a forced import from `src/`, then default `pytest -q`, `ruff check .`, and
   `mypy src` with the lockfile and development group, plus a `speech-verification`-extra run of
   `tests/providers/test_speech_presence.py` and a typed pass over the governed
   qualification and real-gate scripts, including the private run journal;
2. `npm ci --ignore-scripts`, browser unit tests, and the production browser
   build;
3. source/static parity after CRLF-to-LF normalization for package files captured from the archived
   candidate *before* the web build with that build's freshly regenerated
   package-static output. This snapshot is necessary because `web/build.mjs`
   writes directly to `src/hermes_realtime/client/static`; comparing only after
   that write could overwrite stale committed output and falsely pass;
4. a fresh wheel and sdist build, a 20 MiB per-artifact ceiling, strict sdist allowlisting and
   ambient/forbidden-path rejection, packaged Silero license/notice/model presence with an exact
   model checksum, plus an isolated installed-wheel import; and
5. two secret scans that fail on credential files and common private key,
   GitHub token, OpenAI key, or AWS access-key signatures without printing
   matched values. TLS certificates and keys used by hostname-boundary tests are generated
   in pytest-owned temporary directories and are never committed. The first scan enumerates
   and reads every `HEAD` Git blob before
   `git archive`, so `.gitattributes export-ignore` cannot hide a committed
   secret; the second scans the extracted distributable candidate.

The Packaged capture and spool recovery job consumes the existing Pure candidate wheel artifact. Its
packaged producers verify the wheel digest and every runtime source blob,
then execute the unpacked pure wheel in an owned process. These packaged
scenarios do not establish the full installed dependency or physical matrix.

The `native-livekit` job downloads the documented Windows LiveKit 1.13.4
binary, verifies its SHA-256, starts it on `127.0.0.1` in development mode, and
then runs the native local LiveKit, browser LiveKit, and local launcher tests.
`InsecureKeyLengthWarning` is promoted to an error. This is a required release job rather than an
ambient local prerequisite.

## Publishing the repository

Publishing experimental alpha source does not establish desktop qualification or
production readiness. Use this sequence when publishing a private repository:

1. Confirm the target repository and record the reviewed main commit and tree.
   Complete the [required automated checks](#required-automated-checks), including
   the resulting main push run, before making the publication decision.
2. Review every surface that will become public: Git history and refs, issue/PR
   text and retained edit history, Discussions, workflow logs, and downloadable
   artifacts. Record coverage and unavailable material. Apply the
   [public-repository boundary](../CONTRIBUTING.md#public-repository-boundary) to
   each surface; candidate-archive scans alone do not cover GitHub discussions or
   historical content. Preserve the evidence while removing private details from
   both current text and affected revision diffs. GitHub documents
   [how to remove sensitive revision diffs](https://docs.github.com/en/communities/moderating-comments-and-conversations/tracking-changes-in-a-comment).
3. Check licenses, third-party notices, artifact provenance, and dependency
   advisories for the candidate. Ensure the README and
   [implementation status](implementation-status.md) distinguish implemented
   behavior, default activation, and completed qualification.
4. Obtain release-owner authorization for the visibility change. During the
   transition, enable and verify
   [private vulnerability reporting](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository)
   so the route in [SECURITY.md](../SECURITY.md) works. Read back main protection,
   workflow token permissions, outside-contributor workflow approval settings,
   and available secret-scanning/push-protection settings; inspect fresh alerts.
5. Before announcing the public alpha, verify the published documentation,
   sanitized history, and security-reporting route from an outside-reader
   perspective. Record the published commit, CI run/attempt, findings resolved,
   and remaining qualification limits in the release decision.

GitHub's [visibility documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)
describes the transition's effects, including public access to Actions history
and logs. GitHub's [secret-scanning documentation](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning)
describes automatic public-repository scanning. Keep sensitive review evidence
outside the public repository; publish only sanitized findings and decisions.

## Conversational style characterization

Before changing the realtime Codex instructions, run the versioned synthetic multi-turn corpus
against the release model and effort:

```sh
env -u PYTHONPATH uv run --frozen python scripts/evaluate_conversation_quality.py \
  --model gpt-5.6-terra \
  --effort low \
  --label candidate \
  --output .runtime/conversation-quality.json
```

The evaluator uses only synthetic dialogue. Its shadow work handler and shadow knowledge lookup
reject or empty-answer and record every tool attempt without dispatching external work or search. It
fails on missing or duplicate cases, empty replies, case-specific lexical relevance requirements,
word/sentence/question ceilings, unrequested spoken list or Markdown structure, common assistant
boilerplate, or any work/knowledge-tool attempt. The lexical requirements catch generic or
context-free replies; they do not establish factual correctness. The report retains model responses
so it must still be inspected before sharing, even though the checked-in prompts contain no private
user data.

This is a bounded transcript characterization, not an acoustic or subjective acceptance claim.
Blind A/B review can compare prompt candidates, but desktop/iPhone listening remains authoritative
for warmth, pacing, silence, echo, repetition, barge-in feel, and whether optional follow-up
questions make the exchange feel engaged rather than interrogative.

## Opted-in Windows evaluator ownership

The configured-provider routing evaluator requires the exact clean candidate commit and
tree for every invocation. It selects the host's provider factory, makes one tool-less
readiness call, binds the shared effect-free work handler, then makes one decision call
per corpus case. This is a source-bound synthetic routing check, not a real-Hermes
dispatch, voice, or semantic lifecycle review. A failed case is never retried.

The schema-five routing report accepts the exact internal `HostSpeechRequest` as a
terminal selection and reports its count, case IDs and request latency separately in
`host_status_requests`. This is request-only evidence: the evaluator renders no host status,
consults no live task controller and delivers no audio. The effect-free shadow handler owns
the active case and refuses a status request mixed with recorded lifecycle calls. A later
stream item, an unknown item or a marker subclass fails the attempt; readiness continues to
accept speech strings only. Earlier model prose remains ordinary unreviewed prose.

A status request never counts as a lifecycle tool call, a successful start or legacy
string-stream `first_speakable` latency. That string stream still includes the existing
fixed start acknowledgments; #272 separately owns their provenance. A status request does
not satisfy the separate lifecycle-claim review. The corpus, routing thresholds and pending
semantic review remain unchanged. Task-state
projection, current foreground rendering and heard provenance are covered by the host's
deterministic tests and require their own installed and voice qualification.

On Windows, opt into the independent base-Python guardian with the same binding:

```sh
python -m scripts.evaluate_natural_work_routing \
  --source-head "$(git rev-parse HEAD)" \
  --source-tree "$(git rev-parse 'HEAD^{tree}')" \
  --launch-binding interpreter-binding.json \
  --launch-binding-sha256 APPROVED_SHA256 \
  --launch-receipt private-launch.jsonl \
  --provider ollama --mode diagnostic_override \
  --model SELECTED_MODEL
```

The private binding has exactly this versioned shape; supply absolute local paths
and SHA-256s of their exact executable bytes. The guardian must be the current
interpreter's `sys._base_executable`; the controller must be `sys.executable`.
Keep this file, receipt and retained output private; do not commit local paths.

```json
{"schema_version":1,"guardian":{"path":"ABSOLUTE_BASE_PYTHON","sha256":"SHA256"},"controller":{"path":"ABSOLUTE_CONTROLLER_PYTHON","sha256":"SHA256"}}
```

All three launch options are required together. Without them the CLI runs in process
and claims no guardian ownership. `--mode production_default` takes provider settings
from the host factory and refuses explicit model/runtime overrides. It is the CLI
default. Select `--mode diagnostic_override` for model or runtime overrides. `--provider codex`
remains the default; `--provider ollama` selects the same host factory and optional
`--ollama-base-url` and `--ollama-num-ctx` settings. The entrypoint refuses a missing or
changed source commit/tree, a dirty worktree, or production imports from another
checkout before model execution, and verifies source stability again afterward.

The version-four report declares a 20-second readiness bound, a 60-second decision
bound, a 10-second adapter-close bound and a 3500-second total evaluation bound. It
uses a 3600-second outer guardian deadline, leaving room for bounded cleanup and
receipt writing without silently doubling the evaluation budget. The total clock
starts before Codex version discovery when Codex is selected. It
records one attempt per case and distinguishes a fresh per-case adapter from unknown
external runtime process/cache freshness. Reports contain model labels, aggregate
routing counts and bounded failure categories; they do not contain prompts, replies,
private paths, or raw provider errors. A provider decision can be a false or misleading
lifecycle claim; the separate semantic review remains pending until reviewed.
If the owned controller completes with a nonzero result and source remains bound,
the caller preserves its reviewed report (including a pending semantic review) and
the retained private result artifact. If post-run source binding fails, it emits only
bounded `owned_result_failed` with a secondary source category, without echoing the
controller output. A valid unsuccessful guardian
receipt retains its recorded stage, category and cleanup fields in the public failure
summary, distinct from a missing or malformed receipt. Source drift observed after
either failure is reported as a secondary category without replacing the first one.

The default launch requests explicit Job breakaway, hides the process and verifies
the actual base guardian is outside inherited Jobs. Windows may refuse breakaway;
`breakaway_not_permitted` is a recorded refusal with no fallback. The generic
Windows kernel associates the controller with its noninherited kill-on-close Job
atomically while suspended. The controller retains its explicit interpreter and
uses private output files plus NUL input. A completed launch needs a normal
controller completion record, retained exit evidence, and an observed empty owned
Job. Completion is containment/quiescence evidence, not a model/scoring pass or an
authenticated inventory of all descendant images.

An outer Windows caller that starts the bound virtual-environment interpreter
must use `scripts.qualification_launch.WINDOWLESS_BREAKAWAY_FLAGS`, including in
its model-free proof, with NUL stdin and private redirected output. Do not add
`DETACHED_PROCESS`: [Windows ignores `CREATE_NO_WINDOW` when they are combined](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags).
Console suppression and lifetime independence are different properties; the
existing base guardian still checks breakaway and owns the controller Job. This
constant grants no interpreter, source or execution authority and does not bound
a caller's output or wait; those remain the caller's existing responsibilities.

The Windows regression observes the actual virtual-environment redirector's
interpreter child from a console-attached parent on private desktops that are
never selected for input. Removing `CREATE_NO_WINDOW` or adding `DETACHED_PROCESS`
leaves the child with a console; the corrected flags leave it without one. This is not attribution
of the historical visible window's owner, which was not retained. A host that
cannot run the private-desktop/redirector observation does not establish that
native coverage. No model or provider runtime participates in this test.

An explicit breakaway can succeed while leaving the guardian in a
[restrictive ancestor Job](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs).
That host receives `guardian_inherited_job` before controller
execution; it is not a supported fallback execution mode. Windows tests use an
independent base-Python/Win32 capability probe and verify the actual API and CLI
refuse without starting a controller on constrained hosts. Controller-dependent
native fault tests are then explicitly reported as unexercised, rather than
claimed as completion coverage. Supported hosts still run those paths; synthetic
kernel, binding and watchdog checks remain independent of host capability.

The fresh receipt and adjacent `.work` directory are retained, including after
caller interruption. Missing final evidence remains incomplete; do not infer
cleanup or automatically retry. The guardian has its own bounded deadline
(four hours maximum, plus at most thirty seconds for cleanup). Retained stdout is
limited to 8 MiB, stderr to 64 KiB; size polling rejects excess every 50 ms. This
is an acceptance/polling bound, not a hard disk-write quota. The Job's explicit
64 GiB per-process / 128 GiB aggregate limits do not replace a provider's GPU or
memory admission contract.

The observer checks its execution deadline after each bounded wait, before
accepting a signaled exit, and its cleanup deadline before accepting a zero-active
observation. After verified Job quiescence, final stdout, stderr
and completion reads enforce their acceptance bounds again.

`scripts.qualification_launch.run_owned_controller` is also a caller-bound script
API: it hashes and executes the exact external controller bytes without importing
the tooling package into that controller's namespace. The caller remains responsible
for controller dependencies, runtime provenance, source/bootstrap checks, provider
budgets and sanitized public reports. Interpreter hashes alone do not attest their
complete dependency closure. This ownership slice does not complete reusable
provider qualification, real dispatch, or human voice acceptance.

### Bounded development diagnostics

The existing scenario evaluator also accepts `--development-fixture` instead of
`--scenario`. It binds the unchanged development fixture bytes and selects, in order,
`research`, `greeting`, `explanation`, `quotation`, `negation`,
`cancel_model_authority`, `approval_model_authority`, and `ambiguity`. A separate
`host_status` observation uses the existing synthetic question “How is work?”;
it does not add a row to either frozen corpus. Cancellation, approval and status
observations carry one synthetic active task. No project context or access-proof
override is accepted in this mode.

```sh
python -m scripts.evaluate_natural_work_scenario \
  --development-fixture tests/fixtures/natural_work_development_cases.json \
  --source-head APPROVED_CLEAN_HEAD --model SELECTED_MODEL \
  --base-url http://127.0.0.1:11437 --private-receipt private-development.jsonl --live
```

The runtime must already be owned by the caller. For owned Windows execution, pass
this script and these arguments to the existing `run_owned_controller` API with its
verified interpreter binding and a 1360-second deadline; do not copy its guardian.
The evaluator uses a fresh production Ollama adapter/context/shared rejecting work
surface per observation, ordinary synthetic-PCM preflight, context 4096 and unchanged
production sampler settings. At most nine readiness and nine decision requests run
once, including after semantic misses. Readiness and decisions each retain their
60-second bounds; adapter and surface cleanup each retain ten seconds. The diagnostic
has a 1300-second overall bound. It does not start or evict a model runtime.

The private receipt is exclusively reserved before calls and flushed after each
attempt, retaining synthetic replies, objectives and partial failure evidence for
manual review. Keep it private. Public reports expose counts, kinds and latency only.
A typed host-status marker measures **selection only**, without rendering, synthesis,
playback or a claim that status was heard. Text-output latency remains separate.
The Ollama-only diagnostic rejects prose/effects mixed with status selection and
output after a terminal marker.
Objective fidelity, clarification and lifecycle review remain pending; `pass` stays
false. This diagnostic does not invoke the 100-case scorer, admit its resource
prerequisites, qualify explicit cancellation, or establish voice acceptance.

## Installed natural-work boundary and latency gate

Before enabling natural Codex work routing by default, run the installed-boundary
gate with the exact release model, effort, Codex executable, Hermes source tree,
and Hermes API credential that will be used by the host. The gate starts a
loopback-only authenticated API adapter and performs two benign real Hermes runs:
one must complete authoritatively and one deliberately long run must be stopped
through the explicit user command `cancel task: <public task ID>`. The existing
command router delegates to the shared exact-task work surface; cancellation
never enters model inference. The accepted cancellation and interrupted terminal
must match that frozen task ID. Ordinary conversation and an old-task continuation
response must create no run or work-tool activity.
Codex completes one tool-less preflight turn before the measured requests. The gate provider has
no knowledge lookup, so unlike the production host it exposes no `search_knowledge` tool. This
preflight keeps app-server initialization from contaminating acknowledgement timing before the
natural-work tools are bound.

On the native Windows installation, use the Hermes Python environment because it
contains the installed API adapter. Point `HERMES_REALTIME_SOURCE` at this candidate's `src` and
add it before the Hermes Agent source tree on `PYTHONPATH`:

```sh
export HERMES_REALTIME_SOURCE='C:/path/to/hermes-realtime/src'
export HERMES_AGENT_SOURCE='C:/path/to/hermes-agent'
export HERMES_HOME='C:/path/to/hermes-home'
env -u PYTHONPATH \
  PYTHONPATH="$HERMES_REALTIME_SOURCE;$HERMES_AGENT_SOURCE" \
  python -c "import hermes_realtime, gateway; print(hermes_realtime.__file__); print(gateway.__file__)"
env -u PYTHONPATH \
  PYTHONPATH="$HERMES_REALTIME_SOURCE;$HERMES_AGENT_SOURCE" \
  python scripts/real_natural_work_gate.py \
    --env-file "$HERMES_HOME/.env" \
    --model gpt-5.6-terra \
    --effort low \
    --pairs 125 \
    --warmups-per-arm 5 \
    --task-ack-budget-ms 8000 \
    > .hermes/real-natural-work-gate.json
```

The latency portion creates separate persistent Codex processes for the
tools-absent and tools-present-but-unused arms. It reports five discarded warmups
per arm, then 125 paired measurements in alternating AB/BA order. The gate
enforces a minimum of 30; the release command uses 125 after a preserved
100-pair run showed unstable remote-service p95 variance. Every
pair uses the same bounded conversational prompt. Acceptance requires the
present-but-unused foreground p95 regression to remain within the greater of
100 ms or 10% of the absent-arm p95. Every natural start or explicit cancellation is
timed from transcript acceptance until the installed Hermes handler returns its
accepted result; every sample must remain below the explicitly configured
task-acknowledgement budget. `8000` ms is the current candidate budget,
calibrated from repeated real runs after the production-equivalent tool-less
preflight while retaining a finite user-facing ceiling. It must be changed on
the command line only when the release owner agrees a different conversational
budget.

The report separates the absent/present p95 comparison from the signed paired
deltas and their absolute distribution. `present_arm_work_activity_boundaries`
and `present_arm_handler_attempts` must both be exactly zero; transport
observation catches malformed or rejected work attempts before handler dispatch.
Every wire-level `thread/start` for the idle present arm must expose exactly the
independently specified `start_work` dynamic-tool schema; its measured count,
name, canonical schema SHA-256, observed thread-start count, and zero-active-task state are published
under `present_arm_configuration`. Invalid sample counts, warmups, and
acknowledgement budgets are bounded above and below before credentials or real
runs are touched. Runtime failures emit only a bounded diagnostic code on stderr.

The boundary report uses `evidence_version: 2` and
`cancellation_route: "explicit_command"`. `observed_cancel_tool_requests` must
be exactly zero, including shutdown; the explicit command still produces one
accepted handler cancellation and one request-to-acknowledgment latency sample.
Earlier evidence that required a model cancellation tool call is incompatible
with this start-only model policy and must be regenerated.

The JSON contains only the model, effort, Codex version and binary SHA-256,
sanitized observed counters, terminal-state classifications, and timing
distributions. The acknowledgement clock starts at transcript acceptance, so the acceptance budget
remains user-facing request-to-accepted-result latency. The report separately publishes the derived
pre-handler decision/tool-call interval and the installed Hermes handler's own acceptance interval;
this localizes an outlier without weakening the combined budget or labeling it transport-only. The
gate fails closed on private `run_`/`deleg_` authority tokens
and discards incidental Python-level adapter output in a bounded sink while
redirecting native runtime writes to the operating-system null device, so stdout
remains one parseable JSON document. The saturated Python discard byte count is
published as `discarded_python_output_bytes`. Never publish the Hermes env file
or raw prompts, credentials, launch capabilities, or private run handles.

## Source-backed foreground qualification

Default-off, consent-bound final-transcript public-RSS lookup (Bing RSS, plus Google News RSS for
outcome-shaped queries) is the maintained Codex runtime backend. Stable-partial overlap and one
weak/empty recovery attempt remain default-off until they pass the source-backed benchmark and the
physical browser matrix on the same frozen candidate. Run the hermetic policy tests and unmetered
characterization first:

```sh
env -u PYTHONPATH uv run --frozen --group dev pytest -q \
  tests/providers/test_current_facts.py \
  tests/conversation/test_knowledge.py \
  tests/test_foreground_knowledge_benchmark.py

env -u PYTHONPATH uv run --frozen --group dev python \
  scripts/benchmark_foreground_knowledge.py \
  --backend ddgs \
  --mode characterize \
  --output artifacts/ddgs-characterization.json
```

A release report must retain routed/unrouted outcomes, weak/empty/timeout/error rates, detached-call
and saturation evidence, and lookup elapsed/blocking/overlap distributions. Query text, transcript
text, page bodies, credentials, and provider payloads are forbidden. Experimental Tavily, Exa, and
OpenAI hosted-search arms are benchmark-only, require explicit environment credentials plus
`--allow-metered-api`, and cannot become runtime defaults from vendor claims or retrieval timing
alone. See [`source-backed-latency.md`](source-backed-latency.md).

The physical full-host gate must exercise operator-disabled and participant-unconsented refusal,
all four route labels, consent-bound final-only public-RSS lookup, optional
Moonshine speculation/recovery, correction of a speculative partial, barge-in, media replacement,
no-search conversation, offline/timeout behavior, and browser `knowledge_timing` projection.
Evidence must show that stale or insufficient lookup never becomes a memory-filled exact claim and
that knowledge controls grant no Hermes work authority.

## Local use

Run the ordinary hermetic candidate gate:

```sh
uv run --frozen --group dev python scripts/release_gate.py --candidate .
```

The script validates committed `HEAD`, not staged, dirty, or untracked files.
Its first output line is `archived revision: <commit>`, which names the exact
commit it archived, so that claim is checkable against `git rev-parse HEAD`.
Commit review changes before treating its result as release evidence; staging is
not sufficient. It intentionally leaves
no release artifacts in the checkout.

For a fast adversarial regression check of the release script itself, run:

```sh
uv run --frozen --group dev python scripts/release_gate.py --self-test
```

The self-test proves that a tracked secret excluded by `export-ignore` is absent
from `git archive` yet still rejected from its committed blob, and that stale
packaged static files fail the pre-build snapshot comparison.

To reproduce the native integration portion locally, first install the shared pinned server
with `uv run python -m scripts.local_livekit install` as described in
[`local-livekit.md`](local-livekit.md), but do not start the server separately. From
a clean, committed repository root, run this PowerShell block. It retains the exact
executable hash and process ID, verifies that the owned process has the only loopback
listener, and supplies all three ownership facts required by the native gate:

```powershell
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath '.').Path
$server = uv run --frozen --group dev python -m scripts.local_livekit path
if ($LASTEXITCODE -ne 0) { throw 'the shared LiveKit server is missing or unverified' }
$serverHash = (Get-FileHash -LiteralPath $server -Algorithm SHA256).Hash.ToLowerInvariant()
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('hermes-livekit-' + [guid]::NewGuid().ToString('N'))
$names = @('LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_KEYS')
$prior = @{}
$present = @{}
foreach ($name in $names) {
  $present[$name] = Test-Path -LiteralPath "Env:$name"
  if ($present[$name]) { $prior[$name] = (Get-Item -LiteralPath "Env:$name").Value }
}
$process = $null
$gateExit = $null
New-Item -ItemType Directory -Path $scratch -ErrorAction Stop | Out-Null
try {
  $env:LIVEKIT_URL = 'ws://127.0.0.1:7880'
  $env:LIVEKIT_API_KEY = 'dev' + 'key'
  $env:LIVEKIT_API_SECRET = 'local' + '-' + ('x' * 32)
  $env:LIVEKIT_KEYS = "${env:LIVEKIT_API_KEY}: ${env:LIVEKIT_API_SECRET}`n"
  $process = Start-Process -FilePath $server -ArgumentList '--dev', '--bind', '127.0.0.1' `
    -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $scratch 'livekit.out') `
    -RedirectStandardError (Join-Path $scratch 'livekit.err')
  $ready = $false
  for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
      if ((Invoke-WebRequest 'http://127.0.0.1:7880/' -UseBasicParsing -TimeoutSec 2).Content.Trim() -eq 'OK') {
        $ready = $true
        break
      }
    } catch {}
    Start-Sleep -Seconds 1
  }
  if (-not $ready) { throw 'LiveKit did not become ready on loopback' }
  $listeners = @(Get-NetTCPConnection -LocalPort 7880 -State Listen -ErrorAction Stop)
  if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1' -or
      $listeners[0].OwningProcess -ne $process.Id) {
    throw 'LiveKit signaling listener is not the owned loopback process'
  }
  uv run --frozen --group dev python scripts/release_gate.py --candidate $repo `
    --require-livekit `
    --livekit-executable $server --livekit-executable-sha256 $serverHash `
    --livekit-pid $process.Id
  $gateExit = $LASTEXITCODE
  if ($gateExit -ne 0) { throw "Native release gate failed with exit code $gateExit" }
} finally {
  try {
    if ($null -ne $process) {
      try {
        if (-not $process.HasExited) {
          $process.Kill()
          if (-not $process.WaitForExit(10000)) { throw 'LiveKit cleanup timed out' }
        }
      } finally {
        $process.Dispose()
      }
    }
  } finally {
    try {
      foreach ($name in $names) {
        if ($present[$name]) { Set-Item -LiteralPath "Env:$name" -Value $prior[$name] }
        else { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
      }
    } finally {
      foreach ($logName in @('livekit.out', 'livekit.err')) {
        $logPath = Join-Path $scratch $logName
        if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force -ErrorAction Stop }
      }
      Remove-Item -LiteralPath $scratch -ErrorAction Stop
    }
  }
}
```

Readiness uses at most 30 attempts with a two-second timeout per request. Cleanup
runs for both gate success and failure. The block exercises the native automated gate; it does not establish the
separate physical, perceptual, provider-account, or full-host qualification claims.
Never use these public development credentials outside loopback mode.
