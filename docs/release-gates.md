# Release gates

Automated candidate acceptance requires all four jobs in the tracked
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
| Native LiveKit release integration | Run the Windows gates with pinned LiveKit, the [archived source-equivalence producer](deterministic-equivalence.md), the [packaged revocation race](revoke-race.md), [capacity rollover](capacity-rollover.md), [capture admission overflow](over-budget-turn.md), the [synthetic fault matrix](synthetic-fault-matrix.md), and real-browser self-acceptance. |

Before an authorized merge, require the reviewed candidate's four PR checks to
pass. After merging, wait for all four jobs in the resulting exact-commit `main`
push run to complete on attempt 1 before advancing to the next candidate. Preserve
any failed run and investigate it; an earlier green PR run does not replace the
main push result. Head or base changes require requalification of the new candidate.

Browser checks use Node 22.22.2 on Windows and Linux. The Linux wheel job first
installs the locked browser dependencies with scripts disabled, runs the browser
tests and TypeScript build, and rejects any generated change to committed assets
or their disclosure manifest. This exercises platform-specific compiler packages
before producing the wheel consumed by the Linux installation check.

Before provisioning the candidate environment, the gate verifies that `uv.lock`
matches the project metadata. A stale lock fails without being rewritten. The
check explicitly selects the materialized candidate. Gate commands ignore ambient
uv project, working-directory, environment-location, and configuration overrides;
CI cache and interpreter-installation locations remain available.

The `release-candidate` job runs, from that fresh candidate:

1. a forced import from `src/`, then default `pytest -q`, `ruff check .`, and
   `mypy src` with the lockfile and development group, plus a `speech-verification`-extra run of
   `tests/providers/test_speech_presence.py` and a typed pass over the three real-gate scripts;
2. `npm ci --ignore-scripts`, browser unit tests, and the production browser
   build;
3. source/static parity after CRLF-to-LF normalization for package files captured from the archived
   candidate *before* the web build with that build's freshly regenerated
   package-static output. This snapshot is necessary because `web/build.mjs`
   writes directly to `src/hermes_realtime/client/static`; comparing only after
   that write could overwrite stale committed output and falsely pass;
4. a fresh wheel and sdist build, a 15 MiB per-artifact ceiling, strict sdist allowlisting and
   ambient/forbidden-path rejection, packaged Silero license/notice/model presence with an exact
   model checksum, plus an isolated installed-wheel import; and
5. two secret scans that fail on credential files and common private key,
   GitHub token, OpenAI key, or AWS access-key signatures without printing
   matched values. TLS certificates and keys used by hostname-boundary tests are generated
   in pytest-owned temporary directories and are never committed. The first scan enumerates
   and reads every `HEAD` Git blob before
   `git archive`, so `.gitattributes export-ignore` cannot hide a committed
   secret; the second scans the extracted distributable candidate.

The Native job also consumes the existing Pure candidate wheel artifact. Its
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

## Installed natural-work boundary and latency gate

Before enabling natural Codex work routing by default, run the installed-boundary
gate with the exact release model, effort, Codex executable, Hermes source tree,
and Hermes API credential that will be used by the host. The gate starts a
loopback-only authenticated API adapter and performs two benign real Hermes runs:
one must complete authoritatively and one deliberately long run must be stopped
through natural cancellation. An ordinary conversational turn must create no run.
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
100 ms or 10% of the absent-arm p95. Every natural start or cancellation is
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
python scripts/release_gate.py --candidate .
```

The script validates committed `HEAD`, not staged, dirty, or untracked files.
Commit review changes before treating its result as release evidence; staging is
not sufficient. It intentionally leaves
no release artifacts in the checkout.

For a fast adversarial regression check of the release script itself, run:

```sh
python scripts/release_gate.py --self-test
```

The self-test proves that a tracked secret excluded by `export-ignore` is absent
from `git archive` yet still rejected from its committed blob, and that stale
packaged static files fail the pre-build snapshot comparison.

To run the native integration portion locally, first follow
[`local-livekit.md`](local-livekit.md) to start the pinned server on loopback,
then run:

```sh
python scripts/release_gate.py --candidate . --require-livekit
```

The native gate fails closed when `http://127.0.0.1:7880/` is unavailable or
not the expected LiveKit readiness response. Hosted CI derives and propagates the same
38-character loopback-only test secret documented in `local-livekit.md`; an HMAC key-length
warning is a gate failure. Never use development credentials outside loopback mode.
