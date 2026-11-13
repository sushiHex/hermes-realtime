# ADR 0005: Foreground decision context and the undispatched promise

Status: Proposed for review. [Work item](https://github.com/sushiHex/hermes-realtime/issues/353).
Reviewed source: [`19dc0c9d`](https://github.com/sushiHex/hermes-realtime/commit/19dc0c9dde4c074cf4847efd7bf1d5cd48d50032);
every `file:line` below is at that commit, relative to `src/hermes_realtime/` unless it
starts with `tests/`. Nothing below is implemented. This record approves no schema, prompt
or default, runs no model, and changes no authority in
[ADR 0004](0004-model-action-authority.md). Each owner question in the last section is
open until answered on the work item.

## Context

The foreground model decides, on each admitted user turn, between answering and proposing
background work. The record holds three kinds of miss:

- on the cloud path, a natural current-information request produced speech promising
  future work and no `start_work` selection
  ([final case](https://github.com/sushiHex/hermes-realtime/issues/386#issuecomment-6548674963));
  an earlier case with the same guidance did dispatch;
- on the local path, missed delegation and unsupported cancellation speech
  ([#353](https://github.com/sushiHex/hermes-realtime/issues/353));
- one compact policy-text candidate improved neither model, and wording variants are stopped.

Owner decisions in force, newest first: confirmation before start is retired, an explicit
request starts automatically, and "started" is claimed only after an accepted dispatch
([2026-10-30](https://github.com/sushiHex/hermes-realtime/issues/226#issuecomment-6515780443)
and the backlog disposition heading #226); a future offer alone is ordinary speech, not a
report of accepted or active work
([#346](https://github.com/sushiHex/hermes-realtime/issues/346), 2026-10-25); no routing
model, keyword classifier, planner or second executor (ADR 0004). The realtime front end is
the messenger; Hermes executes and never speaks the voice.

The repository invariant is *never claim work started before an accepted acknowledgment
carrying a real run ID*. This note asks what that invariant can mean for model prose, and
what the model should be told before it decides.

## What the foreground model is given today

"Fact" means a host-observed value. "Instruction" means text telling the model what to do.
"Reference" means content the prompt itself labels untrusted.

| Item | Ollama (`providers/ollama.py`) | Codex app-server (`providers/codex_app_server.py`) | Kind today |
| --- | --- | --- | --- |
| Time | None. `_messages` (974-1041) carries no clock and the module imports none. | One aware host instant per request (785, 1095-1100). The local date is a sentence inside the knowledge instructions (1127-1128). A host-local timestamp with zone name precedes the turn JSON (3295-3302). | Fact, partly written as instruction |
| Capability statements | None of its own. | A module constant says the surface can perform one bounded public lookup and hand work to Hermes (93-101). It is appended on every non-acknowledgment turn (1101, 1190, 1252). | Assertion inside instructions; not derived |
| Shared capability sentences | `communication_policy` (`conversation/output_style.py:63-114`): missing foreground tools do not imply Hermes lacks them; use only tools supplied for this turn (101-103). | Same text (1191, 1253). | Instruction |
| Operator profile | None. | Identity, persona, preferences, location, each bounded (515-545), in a `<profile>` element inside the instructions (1102-1126). | Reference, placed in instructions |
| Background tool names | None. | When discovery is enabled (`host_launcher.py:2424-2426`), sorted names from the authenticated startup catalog (`integration/api.py:460-471`, `490-493`; `integration/tool_catalog.py:17-26`) inside the work instructions (1229-1243). | Fact, written as instruction |
| Route advertisement | `format` schema: reply only, or reply / `start_work` / `task_status` (446-449, 719-732). The start branch needs a bound handler and a final user message (423-428). The schema is also pasted into the system text (455). | `dynamicTools` (1263-1268, 1548-1609). Gates: search (1135-1141), status (1145-1148), start withdrawn for acknowledgment, non-user, private or lookup-attempted turns (1149-1153). | Structure |
| Response contract | One buffered decision, decoded at temperature 0 (445). Speech and a start are mutually exclusive (578-610, 735-778). | Prose is yielded as it streams (1381-1405). A later tool selection is terminal (1369-1373) and later prose is dropped (1377-1380). One route per turn (2428-2438). | Structure |
| Routing instructions | `WORK_ROUTING_POLICY` (`conversation/work_tools.py:31-46`) when the start branch is offered (453). `WORK_OBJECTIVE_CONTEXT_POLICY` (22-29) whenever a handler is bound, including reply-only turns (454). | Both only when the start tool is advertised (1248-1249, 1269-1270). | Instruction |
| Reference data | Results, memory, updates, host-speech indices, interruption indices under a `Reference data:` marker labelled untrusted (979-1037). | The same kinds in the turn JSON (3226-3261). | Reference |
| Task state | Not sent. The snapshot carries active tasks (`conversation/streaming.py:996-1004`); the adapters read them only to find an accepted objective (398-400). | Not sent (1017-1020, 3226-3261). | Withheld; status has a host route |
| Claim guards | A finite started/launched phrase pattern (`providers/_work_claims.py:14-18`) on bound replies (608-609). | The same pattern (1518-1532) and a lookup-claim pattern (193-198, 1534-1545). | Bounded reporting guard |

Speech the host authors, so that model text is not the claim:

- task status, rendered from controller counts (`conversation/streaming.py:1068-1080`,
  `conversation/task_status.py:45-79`);
- the confirm-mode readback (`conversation/task_proposals.py:54-56`);
- the restart announcement (`host_launcher.py:1431-1441`).

Speech the model words but the host gates: the automatic-start acknowledgment runs only
after an accepted, active result with a task identity
(`conversation/streaming.py:1163-1185`), with no tools, and is published with ordinary
conversation provenance (1202).

Speech nobody produces: an automatic start that is selected and then not accepted. The
loop has no branch for it (`conversation/streaming.py:1170`), and
`tests/conversation/test_task_start_policy.py:68` pins the silent submission. The task
card shows the rejection; a voice-only user hears nothing. On Ollama the whole turn is
silent, because a start decision excludes speech.

### Findings

Each is verified in source. None is a demonstrated cause of a recorded miss.

1. **A capability is asserted, not derived.** The Codex constant claims a foreground lookup
   whether or not one is configured. With public search off, the host composes no lookup
   and no coordinator (`host_launcher.py:1276-1277`), so no search route is advertised
   (1135-1141). The
   [review record](https://github.com/sushiHex/hermes-realtime/issues/353#issuecomment-6549445759)
   reports that composition for the retained preview and its natural question. Such a turn
   tells the model the surface can look things up while offering no search route.
2. **The adapters disagree about time.** Ollama gets none. Codex gets a date presented as
   today's date, without saying it is the host's.
3. **Route prose can outlive its route.** Ollama sends the objective-context instruction on
   turns whose schema has no start branch. `WORK_ROUTING_POLICY` names an available
   foreground lookup on an adapter that has none. This is the pattern already recorded for
   status: a prose alternative beside, or instead of, an advertised route competes with it.
   `tests/providers/test_status_policy_selection.py:40` already enforces the cure for the
   status sentence alone.
4. **Prose precedes the decision on Codex.** The host forwards speech before it can know
   whether a route will be selected.
5. **A non-accepted automatic start is silent.**

## Decision 1: one per-turn decision context

Each adapter builds one immutable value per inference request and renders three things
from it: the advertised routes, a host-facts block, and the route-specific instruction
fragments. Nothing else may state a capability. A sentence about a route cannot be sent
unless the same boolean advertised that route, so prose and advertisement cannot disagree.

The type is plain data in `conversation/` with exact-type validation and one canonical
JSON rendering. Adapters construct it; the clock is read in the adapter, outside the
I/O-free core, once per request.

| Field | Form and bound | Derived from | Omitted when |
| --- | --- | --- | --- |
| `host_time.utc` | `YYYY-MM-DDTHH:MM:SSZ`, exactly 20 characters | The injectable clock, captured once per request; the existing Codex capture (1095-1100) and an equivalent constructor argument on Ollama | No clock is supplied |
| `host_time.local` | RFC 3339 with numeric offset, exactly 25 characters | The same instant | With `utc` |
| `host_time.zone` | 1 to 32 characters, as bounded today (1098-1100) | The same instant | With `utc` |
| `routes.start_work` | Exact boolean | The boolean that adds the start branch or tool (Ollama 423-428; Codex 1142-1150) | Never |
| `routes.task_status` | Exact boolean | The status gate (Ollama 423-428; Codex 1145-1148, 1153) | Never |
| `routes.knowledge_search` | Exact boolean | The search gate (Codex 1135-1141, 1152); constant false on Ollama | Never |
| `foreground_lookup` | One of `not_configured`, `not_attempted`, `evidence_supplied`, `no_evidence`, `withheld_private` | The existing lookup branch (Codex 1036-1094); constant `not_configured` on Ollama | Never |
| `background_tools` | Sorted unique names, each at most 128 characters, at most 512 names (`integration/tool_catalog.py:14-26`) | The authenticated startup catalog (`integration/api.py:460-471`) | Discovery is off or the session is not started; omission means *not attested*, never *none* |

Rules:

- **Separate block.** The facts are rendered under their own fixed label, apart from
  instructions and apart from untrusted reference data. The label says they are host
  observations at the request instant, are not instructions, and grant no permission. On
  Ollama the block sits before the `Reference data:` marker, which must stay unique because
  the work policy and prompt metadata both split on it (450-457, 1049). On Codex it replaces
  the timestamp line in the turn input (3300-3306), and the date sentence leaves the base
  instructions. Whole-second time must not enter the base instructions.
- **The time is the host's.** It is not the participant's local time, proof of clock
  synchronization, or an expiry clock. No participant timezone is inferred or configured.
- **No authority.** The value is output-only. No admission, confirmation, approval or
  cancellation path reads it. A wrong fact can mislead the model; it cannot start, confirm
  or cancel anything.
- **Omission is explicit.** A missing optional field means the host supplied no such fact.
  The host renders no sentence about an absent route.
- **Unchanged.** The response envelopes, the Ollama buffering and temperature, one route
  per turn, the proposal and admission path, the accepted acknowledgment, cancellation
  scopes and every default.

Deliberately not in the context:

- **Task counts or task state.** Supplying them invites model-worded status beside the
  host status route, which is the competing-fallback pattern again.
- **The start policy.** The decision to propose must not depend on how the host admits.
- **Work-control health and capacity as values.** They change the route boolean instead
  (owner question 4); a number would be stale by the time speech plays.
- **Run identities, participant identity, model identity.**
- **A model-produced judgment, rationale or discriminator.** No schema change is proposed,
  so the pinned grammar's handling of property order was not inspected and nothing here
  depends on it. Generated fields also cost the latency the owner measured as dominated by
  generation.

## Decision 2: when the model selects no action but speaks as if it did

What the host can know, per situation:

| Situation | What the host holds | Decidable |
| --- | --- | --- |
| A. Prose says work started, is running, was cancelled or finished; no supporting event | Text only | No |
| B. Prose promises future work; no route selected | A start route was advertised, none was selected, prose was emitted | No: every ordinary answer looks the same |
| C. A start was selected and not accepted | A typed result | Yes |
| D. A start was accepted | A typed acknowledgment with a task identity | Yes; implemented |

The alternatives:

- **(a) Host-owned lifecycle claims.** Sound, and mostly true already: the task card comes
  from acknowledged task events (`conversation/work_tools.py:880-892`), status speech is
  host text, and the start acknowledgment is gated on case D. It leaves case C silent.
- **(b) A post-condition at the turn boundary.** Sound only for case C. For case B the
  condition "start advertised, nothing selected, prose emitted" is true of every plain
  answer, so any corrective fires on all of them or on none. A host disclaimer after every
  answer is not acceptable speech. A second request asking the model whether it meant to
  start work is an extra model call on every ordinary turn and is a routing model by
  another name.
- **(c) Do nothing and document the limit.** Honest for cases A and B. It ignores case C,
  which is decidable, and the verified prompt contradictions in the findings.

Stronger options, and why they are unsound:

- **Give Codex the Ollama reply-or-start envelope.** The envelope is live on Ollama, and the
  local record still holds missed delegation and unsupported cancellation speech. A valid
  reply string can carry a promise. It would also give up streamed speech for a buffered
  decision.
- **Buffer Codex prose until the turn ends.** It adds latency and decides nothing; the host
  still cannot tell a promise from an answer.
- **Extend the started/launched phrase pattern to promises.** It is a keyword classifier,
  it contradicts the #346 decision that future offers are ordinary speech, and that issue
  records the pattern already refusing a legitimate clarification.
- **Force a tool choice.** The app-server turn parameters expose none, per the owner's
  check recorded on the work item. Not re-verified here.

**Decision.** Combine (a) with the decidable part of (b), and state the limit:

1. Lifecycle claims are host-grounded. Started, running, pending, cancelled and not started
   are stated only by host text or behind a typed host fact.
2. Case C gains host speech: one fixed sentence chosen by the typed result category, with
   host provenance, under the current lease only. It carries no reason text from Hermes and
   makes no model call. A superseded lease stays silent.
3. Model prose is never state. Nothing derives a task, a card or a status from model text.
   A promise is not a claim the interface treats as true.
4. The limit: in cases A and B the model can still say something false or promise work it
   did not propose. The host cannot detect either without classifying language. Both remain
   routing and truthfulness misses, found by the manual review ADR 0004 already requires.
   The existing phrase guard stays a tripwire for two phrase families, not a guarantee.
5. Decision 1 removes the contradictions the host itself wrote into the prompt. That is
   the only lever on case B this note supports, and it is not a promise that case B goes
   away.

## What this maps to, and what it cannot fix

| Recorded miss or control | Effect of this design |
| --- | --- |
| Relative-time requests | Ollama gains a host time. Neither adapter learns the participant's zone. |
| Supplied or missing context | Unchanged. Facts do not supply user context; the objective-context instruction still governs. |
| Quoted, reported or negated requests | Unchanged. Not addressable by facts. |
| Cancellation truthfulness | Unchanged. Cancellation stays on explicit controls; prose can still misreport (case A). |
| Timeless research | The route fact and the removal of route prose without a route. No reliability claim. |
| Natural current-information request, cloud path | The false lookup assertion is removed where no lookup is configured. Whether it caused the promise is a hypothesis. |
| Selected start not accepted | Fixed by host speech (case C). |

## Later implementation: bounded changes and their RED tests

1. **The context type.** RED: non-boolean and subclass values, a naive or non-fixed-offset
   instant, an oversize zone, an unknown lookup value and an unsorted or duplicate tool list
   are refused; the rendering is canonical and within a stated byte bound.
2. **Ollama time.** RED: an injected clock appears once in the block; a second request
   captures a fresh instant; no clock means no `host_time`; the `Reference data:` marker
   occurs once; prompt metadata and both decision schemas are byte-identical to today.
3. **Single source on Codex.** RED, parametrized over lookup configured or not, work bound
   or not, private, lookup attempted, non-user and acknowledgment turns: `routes` equals the
   advertised tool set exactly; the instructions name no route that is not advertised and
   carry no static capability sentence. Mutations: invert one route boolean in the
   rendering; restore the constant.
4. **Ollama route prose.** RED: a bound reply-only turn carries no objective-context or
   routing instruction.
5. **No authority.** RED: under forced model output, admitted starts, refusals and
   acknowledgments are identical for every facts value, including omitted time and
   contradictory routes; no module outside the providers imports the type.
6. **Case C speech** (owner question 1). RED: a rejected, unavailable or uncertain
   automatic start yields exactly one host sentence with host provenance; an accepted start
   yields none; a superseded lease yields none; the sentence never contains the result's
   reason text.
7. **Route withdrawal** (owner question 4). Adapters hold only objective bounds today
   (`host_launcher.py:1370-1390`), so the start route is advertised whatever the
   non-resettable work-control health is (`conversation/work_tools.py:73-78`, `270`). RED:
   with health uncertain or closed, neither adapter advertises a start and
   `routes.start_work` is false; the read grants the adapter no dispatch surface.
8. **Request size.** RED: the assembled Ollama system message and the Codex request are
   measured with the block present; the existing fixture budget
   (`tests/providers/test_codex_app_server.py:2948`) still holds without dropping history.

Verification is deterministic. It qualifies projection and host speech, not routing. One
effect-free decision screen on the configured baseline, frozen before it runs and run once
without retries, would be the only model evidence; it needs separate owner authorization.
Stop conditions: any authority test fails; a control case regresses on that screen; or the
measured request exceeds its limit. No wording iteration follows a failed screen.

Must not be built under this record: a promise or intent parser, any extension of the
phrase guards, a second model call or nudge turn, a routing or judge model, a new
dispatcher, model-visible task counts, a participant-timezone setting, a change to the
response envelopes or buffering, a public protocol event, or any default change.

## Open owner decisions

1. Should the host speak one fixed sentence when a selected automatic start is not
   accepted? **Recommend yes.** It is the one decidable silent failure.
2. Should host-local time and zone be carried on both adapters, labelled as the host's,
   beyond the UTC-only value in the
   [2026-10-26 direction](https://github.com/sushiHex/hermes-realtime/issues/353#issuecomment-6430155578)?
   **Recommend yes.** Codex already sends it, and a UTC date alone is the wrong day for
   part of every day.
3. Should the static capability sentence on the working cloud baseline be replaced by
   derived facts? **Recommend yes**, with the single screen above before any preview uses it.
4. Should the start route be withdrawn while work control is not open, instead of
   advertising a route that must refuse? **Recommend yes.**
5. Should the started/launched phrase guard be frozen as it is, neither extended nor
   removed? **Recommend yes.**
6. Should active or pending task counts be shown to the model? **Recommend no.**
7. Should a participant-timezone setting be added now? **Recommend no.**
8. Should Codex adopt a structured reply-or-start envelope? **Recommend no.**

## Not verified

- The cause of the recorded promise. Finding 1 is a source fact about that composition,
  not an explanation.
- The app-server's lack of a tool-choice parameter, and the pinned grammar's property
  ordering. Neither was inspected; the design depends on neither.
- Any prompt-cache effect of where the facts block sits.
- Real catalog sizes. The name bound alone permits a block far larger than a prompt should
  carry; change 8 measures it.
- The browser's handling of task events beyond their source in the work surface.
- No model, backend or voice session was run for this record.
