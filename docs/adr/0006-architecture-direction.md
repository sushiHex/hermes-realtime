# ADR 0006: One owner per fact

Status: Proposed for review. Direction approved by the owner on 2026-11-12; this record is not
accepted until the open decisions in the last sections are answered on the pull request that
carries it. No umbrella work item exists; each move links its own issue, and a move without
one gets an issue before work starts ([ADR 0001](0001-github-work-tracking.md)).
Reviewed source: [`ae48cad8`](https://github.com/sushiHex/hermes-realtime/commit/ae48cad8a1fdd39c5c7a96b044bf58dde7274966).
Every `file:line` is at that commit, relative to `src/hermes_realtime/` unless it starts with
`scripts/`, `tests/`, `web/`, `docs/` or `.github/`. Nothing below is implemented. This record
changes no schema, prompt, default or gate, and contains no line-level plan.

## Context

Four reviews of the same commit (structure, the foreground model path, lifecycle and state,
the evidence system) reached one diagnosis from different sides: **each fact has several
owners, and what is operated is not what is typed and tested.** The rows below were re-read
in source for this record. "Latent" means two definitions exist and no wrong outcome was shown.

| Fact | Owners found | What was verified |
| --- | --- | --- |
| A private handle | `client/projection.py:65-67` and `conversation/result_projection.py:20-22` (three prefixes); `web/src/protocol.ts:510` (two); six unanchored `deleg_` patterns (`conversation/commands.py:27`, `context.py:34`, `updates.py:22`, `work_tools.py:50`, `providers/codex_app_server.py:192`, `ollama.py:71`); `conversation/tasks.py:39-40` adds a look-behind | The browser pattern lacks `subagent_`. Python is the producer and stricter, so this is drift at a privacy boundary, not a shown leak |
| A valid turn id | Letter first at `client/projection.py:63`, `conversation/work_tools.py:54`, `web/src/protocol.ts:509`; digit allowed at `conversation/knowledge.py:28` | Latent |
| A result's length | `result_limits.py:3` says Unicode scalars; `client/http.py:665` counts UTF-16 units. Presentation bounds 220 and 1800 (`result_limits.py:6-7`) against 240 and 3200 (`conversation/result_projection.py:28-29`) | Two counting units for one limit |
| The layering rule | No tracked document or test states it. `protocol/events.py:583` imports `memory`; `conversation/knowledge.py:19` imports `providers.current_facts`; `conversation/result_projection.py:47` imports `providers._speech_text`; `conversation/streaming.py:16-20` imports `_qualification` and `evidence` | Violations landed unopposed |
| Which routes this turn has | Advertisement and prose are assembled separately. `providers/codex_app_server.py:1255` appends "use advertised start_work" on every tools-branch turn, including those where `:1149-1150` withdrew the start tool. `providers/ollama.py:454` sends the objective instruction on reply-only turns | Prose outlives its route (also ADR 0005 finding 3) |
| Which rows were host speech or interrupted | Rows carry the markers (`conversation/context.py:84-113`); index lists are rebuilt at `providers/ollama.py:999-1009`, `providers/codex_app_server.py:3252-3255`, `conversation/streaming.py:1009-1012` | One fact, three renderings |
| Where a sentence splits | `providers/ollama.py:1147` uses `>`; `providers/codex_app_server.py:3337` uses `>=` | The adapters segment differently at the hard limit |
| Where the saved conversation lives | The package has one fixed path (`host_launcher.py:3443-3447`), a kernel lock (`integration/run_record.py:90-110`) and atomic replace (`:41-56`). The preview overrides it per run and copies the tail forward (`scripts/preview_cutover.py:1816-1873`; `write_bytes`, no fsync, `:1872`), judging the predecessor dead from bare PIDs (`:1638-1656`) written non-atomically (`scripts/rehearse_desktop_mvp.py:653-656`) | Every [#496](https://github.com/sushiHex/hermes-realtime/issues/496) item exists because of the copy |
| The run-record schema | Owner `integration/run_record.py:227-230`; restated by hand at `scripts/preview_cutover.py:1678-1687` | Latent: the owner writes version 1 whenever `pending` is empty (`:195-198`), which is the only case the script accepts |
| "The host is ready" | Prose at `host_launcher.py:3299`; the same sentence as a constant at `scripts/rehearse_desktop_mvp.py:128` | A copy edit would read as a product failure |
| What the composed product is | `host_launcher.py:1874` (one function to `:3111`), `launcher.py:300`, each with its own `_project_conversation_observation` (`:889`, `launcher.py:80`). The preview's process owner is imported from a rehearsal (`scripts/preview_cutover.py:2250`), and `scripts/` is outside the wheel (`pyproject.toml:90-91`) | Operated lifecycle is neither packaged nor strictly typed |
| That the suite passed | `scripts/release_gate.py:986-998` runs the whole suite unconditionally; two jobs call it (`.github/workflows/release-gates.yml:320`, `:432`) | Every node runs twice per run |
| What is qualified | A registry in code (`scripts/qualify_evidence_slice_zero.py:445-504`: 20 scenarios, 8 with producers), prose in `docs/implementation-status.md`, and a family of issues | Only the code copy is checked |

The diagnosis is weaker than the reviews put it in one place: eleven test files do call the
shipped `build_local_host_launcher`. What strict typing and the wheel do not reach is the
preview lifecycle in `scripts/`, and tests cannot reach the launcher's closures in isolation.

## Decision

**One owner per fact, inside the package; everything else is a rendering of it.**

The test for any proposed change: *which single place owns this fact, and what renders it?*
If the answer names two places, or a place outside `src/hermes_realtime/`, the change adds an
owner and is refused in that form. A check that two copies agree is acceptable only where a
single definition cannot span the boundary (Python to TypeScript), and it binds by equality.

Two fixes this refuses:

- **Marking the copied conversation as consumed** ([#496](https://github.com/sushiHex/hermes-realtime/issues/496)
  item 1). The marker is a second record about the first copy. It must then define "consumed"
  (when `finalize` ran, or when the successor opened the tail?), and the lifecycle review
  found that refusing an already-handed-off run strands the conversation when a successor
  fails to start. The owner of the conversation is one store. Remove the copy.
- **Adding `subagent_` to the browser pattern** (`web/src/protocol.ts:510`). It repairs today's
  drift and leaves two definitions. The owner is one definition in `protocol`; the browser
  constant is generated from it or pinned to it by a test, as
  `tests/client/test_projection.py:632-640` already does for two result limits.

### Move 1: one definition

Identifiers, limits and evidence-marker names are defined once in `protocol`. Browser constants
are generated or test-pinned. A ratcheting test enforces the inward import rule.

- **Makes impossible:** two Python definitions of one identifier or bound; a new outward
  import landing unnoticed.
- **Does not:** stop a fact being redefined under a new name (review only); prevent browser
  drift by construction when the constant is test-pinned, not generated; fix existing
  violations, which stay as named exceptions until later moves delete them; cover loopback or
  canonical-JSON definitions, which the reviews reported and this record did not verify.
- **Must preserve:** `protocol` imports nothing and stays strict and fail-closed; exact-type
  checks; lazy provider imports. Unifying a *divergent* copy changes behaviour at a privacy
  boundary and is a labelled change with its own RED tests, never part of a move.
- **First step:** the ratchet test, with an exception list bound by equality to today's
  violations so that both a new edge and a stale exception fail; then imports replacing
  copies only where the copy is byte-identical.
- **Deletes:** the local pattern copies, `result_limits.py` as a root module, the lazy import at
  `protocol/events.py:583`, and one exception entry per later step.

### Move 2: one turn request

`conversation/` builds one immutable request per inference: routes as data, host-derived
facts, history rows carrying their own markers. Adapters render it and return one event
stream under one contract: *speech, then at most one terminal route*.

- **Makes impossible:** a host sentence naming a route the turn does not offer (guidance is a
  field of the route); adapters disagreeing on host facts (they hold no clock and no store);
  index drift (no index list exists); two segmenters or two guard sets; an adapter
  re-deriving policy from `messages[-1].role` (`providers/ollama.py:423-428`).
- **Does not:** stop a paraphrased capability sentence in an unconditional fragment (a
  vocabulary lint catches names only); stop the model promising work it did not propose
  (ADR 0005 cases A and B stay undecidable); control what a provider adds around the request;
  guarantee cache hits, only a fixed, testable order; stop an in-band marker being echoed by
  the model unless the output gate and a user-text test cover it; stop a route that was open
  when the request was built from being refused at admission.
- **Must preserve:** delivered-only history; a route selection is a proposal, and nothing is
  claimed started before an accepted acknowledgment (ADR 0004 admission is unchanged);
  independent cancellation scopes; atomic publication through the coordinator's leases;
  `conversation/` performs no I/O, so the clock is injected.
- **First step:** a golden matrix of the exact bytes each adapter sends today, proved by
  mutation. It changes nothing and makes every later step either byte-identical or visibly not.
- **Deletes:** both `_trusted_snapshot` rebuilds, the adapter-side route booleans, both
  `_extract_segments`, the adapter guard methods, the three index-list sites, the lookup
  policy inside `stream()` (`providers/codex_app_server.py:1026-1094`), and the
  `conversation` to `providers` import.

Design authority for the context fields stays with
[ADR 0005](0005-foreground-decision-context.md) and
[#353](https://github.com/sushiHex/hermes-realtime/issues/353); cache shape with
[#499](https://github.com/sushiHex/hermes-realtime/issues/499).

### Move 3: one state root

A run acquires the conversation store by kernel lock. Nothing is copied between runs.
Ownership is a lock, not a PID. A host-wide fault exits the host and restart recovers
(crash-only); errors are not reclassified inside ordered check chains.

- **Makes impossible:** fork, rollback, double consumption and a half-finished copy (there is
  no copy); a conversation stranded because cleanup was unclean; PID reuse deciding liveness;
  and, once crash-only applies, a live host that can no longer speak or work.
- **Does not:** prevent a manual copy or restore of the root, or two configured roots, which
  is a fork by configuration; work on a synced or network volume, where byte-range locks and
  rename are not dependable; release a hung predecessor's lock (the new owner refuses, which
  is correct, and the operator ends the old Job); keep the last coalesced snapshot if the host
  dies before its write lands; carry Hermes-side state while the Hermes home is per run.
- **Must preserve:** write-ahead order (deletes record before tail, run record before the
  POST); atomic replace under the lock, with temporaries removed only by the lock holder;
  tasks stopped, never resumed; the kill-on-close Job; terminal by default inside ordered
  check chains; the forget guarantee; voice history surviving a crash
  ([ADR 0003](0003-hermes-owned-conversation-continuity.md)).
- **First step:** [#496](https://github.com/sushiHex/hermes-realtime/issues/496): the owner
  takes a lock (stricter only), then the plan names a stable root and `finalize` stops copying.
- **Deletes:** `_copy_voice_tail`, the deletes copy, the `saved_voice_*` refusal categories,
  `--previous` as a source of state, and PID files as authority.

### Move 4: one composition

One `compose()` that ships and that tests run. The preview lifecycle and the process owner
move into the package; scripts become thin commands.

- **Makes impossible:** two launchers wiring the same graph differently; a script restating a
  package schema or importing a private symbol (`scripts/preview_cutover.py:2165`) or a
  signature (`:414`, `:1072`, `:2066`); the product's process owner living in a rehearsal.
- **Does not:** stop a test wiring its own graph (a lint is needed); make fakes faithful;
  exercise real providers.
- **Must preserve:** lazy provider imports and a base suite that passes without the `local`
  extra; explicit provider selection with no fallback; the Job's containment; the release
  gate's snapshot of committed bytes.
- **First step (after acceptance):** move one closure cluster behind a characterization test,
  starting with `ProactiveCompletionUpdatePolicy` (`host_launcher.py:1741`).
- **Deletes:** the duplicate projection function, most of `launcher.py:300` onward, the
  launch-sentence parser, and the product-to-rehearsal import.

### Consequence for evidence

The suite runs once per CI run. Each cross-component contract is tested once, at its
boundary, from one shared vector set. Qualification becomes one executable matrix in the
repository, generated into the status document and bound to it by equality, replacing the
issue family. Linked work: [#300](https://github.com/sushiHex/hermes-realtime/issues/300),
[#507](https://github.com/sushiHex/hermes-realtime/issues/507),
[#324](https://github.com/sushiHex/hermes-realtime/issues/324).

What this gives up: a timing-flaky test gets one pre-merge exposure, not two, and the second
copy was the only detector of a test that fails in one job and passes in the other. A cache
hit no longer proves the pinned binary is still downloadable. The hash check stays the only
authority; the cache is transport.

## Sequencing

Every step ships alone and preserves behaviour, unless it is labelled a behaviour change and
carries its own RED tests. A step that cannot meet that is split or deferred.

| Before desktop MVP acceptance ([#159](https://github.com/sushiHex/hermes-realtime/issues/159)): stricter-only or small | After acceptance |
| --- | --- |
| State root ([#496](https://github.com/sushiHex/hermes-realtime/issues/496)) | Turn-request steps that change what the model sees (derived facts, per-row markers, eviction, cache shape) |
| Import ratchet | `compose()` |
| Identifier consolidation where byte-identical | Lifecycle and process owner into the package |
| Golden request matrix | Hermes home persistence |
| CI: suite once; pinned-binary cache | Crash-only escalation of host-wide latches |
| The fixed host sentence for an unaccepted start ([#509](https://github.com/sushiHex/hermes-realtime/pull/509)) | Unifying any divergent identifier or bound |

## Do not build

- A routing model, keyword classifier, promise parser, nudge turn or second model call.
- An append-only log for the conversation. The tail is a bounded snapshot; atomic replace fits.
- A persistent Codex thread that holds undelivered text.
- Reclassification of terminal failures as turn-only inside ordered check chains.
- A restructuring of the Codex wire-protocol code for its own sake. Only its prompt and
  policy half moves.
- A split of `evidence/`, `StreamingSpeechLoop` or `ConversationSessionWorker` for size alone.

Advised against by the reviews and not adopted here, each a separate decision if raised: a
name-keyed model table, a third adapter before Move 2, change-scoped CI.

## Conflicts and their resolution

| Conflict | Resolution |
| --- | --- |
| The #499 design note proposes a session-constant tool superset with host rejection, for cache stability. The foreground review and ADR 0005 (routes equal the advertised set) require exact per-turn routes. | Exact. A superset advertises a route the request does not contain, which is a second owner. Open decision 1. |
| ADR 0005 Decision 1: adapters construct the context and read the clock. Move 2: `conversation/` builds it; adapters hold no clock. | Move 2 governs, with the clock injected. ADR 0005's fields, bounds and no-authority rule stand; its RED case "no module outside the providers imports the type" is withdrawn. Open decision 2. |
| ADR 0005 puts the facts block inside the Ollama system message, ahead of history. The foreground review puts per-turn facts after history on both adapters, for a stable prefix. | Open decision 3. Not resolved here: whether small-model templates accept a trailing system message is unverified. |
| ADR 0004: provenance is bounded to the process and the saved tail is unchanged. The foreground review persists row source and time. | ADR 0004 stands until amended. Open decision 4. |
| ADR 0004: Ollama buffers the whole envelope. The foreground review offers early reply streaming. | Not adopted. A separate latency decision, outside this record. |
| The lifecycle review moves the run record into the state root. It also warns that a pending entry replayed against a fresh Hermes home would start work and then stop it. Sequencing puts the state root before acceptance and Hermes home persistence after. | Open decision 5. Derived from reading, not exercised. |
| The lifecycle review offers two interim fixes: a `HEAD` record, and carrying state from a run whose cleanup was refused. | Both harden the copy. Refused by the principle; Move 3 replaces them. |
| The lifecycle review would keep an unparseable tail aside. ADR 0003 says the next write replaces it. | Not adopted. ADR 0003 stands. |
| The evidence review proposes one rerun for failures classed as external input. The development guide requires attempt 1. | Not adopted. The pinned-binary cache removes the measured cause. |
| The structure review reports the package at 62,995 lines and scripts at 73,709. The evidence review reports the reverse. | The evidence review is right: 73,709 lines in 112 package files, 62,995 in 125 scripts. |

## Open owner decisions

Each blocks a move. A recommendation is not a decision.

1. Should routes be advertised exactly per turn, accepting a second cache lineage on turns
   that withdraw a route? **Recommend yes.**
2. Should ADR 0005 be amended so that `conversation/` builds the decision context from an
   injected clock? **Recommend yes.**
3. Should per-turn host facts sit after the history on both adapters? **Recommend yes**, after
   acceptance, and only once the template question is answered for the configured local model.
4. Should ADR 0004 be amended so the saved tail records each row's source and time, in a new
   tail version that still reads the old ones? **Recommend yes**, after acceptance.
5. Should the run record stay per run until the Hermes home persists, so that the state root
   first holds only the tail and its deletes record? **Recommend yes.**
6. Should Hermes-side state (archive, companion store, memory) persist across preview runs?
   **Recommend yes**, after acceptance; ADR 0003's purpose depends on it.
7. May the owner restart the host child in place after a host-wide latch?
   **Recommend yes**, after acceptance.
8. Should a scheduled cold fetch of the pinned binary accompany the cache?
   **Recommend yes**; without it an upstream deletion goes unseen.
9. Should the qualification issues be closed into registry rows once the matrix and its stored
   results exist, each row linking its former issue? **Recommend yes.**
10. May heard-context eviction move to whole-turn blocks, changing the documented window?
    **Recommend yes**, after acceptance, as a labelled behaviour change.

## Not verified

- CI figures in the evidence review (145 runs, failure causes, shares of suite time). One
  green run on the reviewed source tree was checked and matched the reported job durations.
  All savings are estimates.
- Any Codex app-server protocol fact, including where dynamic tools may be declared, and any
  prompt-cache effect of request order on either adapter.
- That importing `conversation` loads an HTTP client. It follows from module-level imports;
  nothing was run.
- Archive fencing after a handoff, and the run-record replay in open decision 5.
- The reported divergences in run-id patterns, loopback definitions and canonical-JSON
  helpers. They are not in the table above for that reason.
- That a host-wide latch leaves a host unable to work. The latches were read
  (`conversation/tasks.py:462-467`, `providers/kokoro_mcp.py:409`); the missing exit path
  rests on the shutdown event having no setter for them (`host_launcher.py:3176-3305`).
- Line counts for what each move deletes.
- No test, model, backend or voice session was run for this record.
