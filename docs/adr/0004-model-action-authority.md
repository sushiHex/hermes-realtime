# ADR 0004: Confirmed natural task proposals, explicit task cancellation

Status: Accepted by owner decision on 2026-10-10; implementation tracked in
[#226](https://github.com/sushiHex/hermes-realtime/issues/226).
[Decision record](https://github.com/sushiHex/hermes-realtime/issues/205#issuecomment-6096188370).
The owner approved the provider framing correction on 2026-10-13; the
[failed native-call probe](https://github.com/sushiHex/hermes-realtime/pull/248#issuecomment-6144062862)
records why it is needed. Qualification remains separate from this decision.
The owner approved a resource-first qualification-contract correction on
2026-10-14, tracked in [#263](https://github.com/sushiHex/hermes-realtime/issues/263).
The [owner decision on 2026-10-16](https://github.com/sushiHex/hermes-realtime/issues/284#issuecomment-6226820862)
supersedes automatic natural starts: the model may propose an objective, but the
user must explicitly confirm that exact objective before dispatch. This policy
is accepted; the implementation boundary below is proposed, not implemented or
qualified. The [MVP decision](https://github.com/sushiHex/hermes-realtime/issues/226#issuecomment-6226820445)
keeps natural routing off and uses explicit task commands with phi4-mini. This
amendment neither changes a model default nor authorizes activation.

## Context

Voice users should state a goal without learning a task prefix. The foreground
model can choose between ordinary conversation and requests requiring tools,
current information, research or sustained work. Hermes already owns execution;
realtime needs a bounded handoff, not another planner or executor.

The forced-model probe in [#205](https://github.com/sushiHex/hermes-realtime/issues/205)
demonstrated that valid model calls can start and cancel work on a neutral turn.
Schema and task-state validation cannot establish semantic user intent. This is
a model-trust limit, not evidence of production-model exploitation. ADR 0003's
memory qualification establishes that memory adds no authority, not obedience.

The owner initially superseded confirmation-for-both with automatic starts and
explicit user cancellation. The held-out diagnostic in
[#280](https://github.com/sushiHex/hermes-realtime/issues/280#issuecomment-6207088708)
then observed cancellation, approval and quoted injected instructions emitted as
work objectives. All dispatches were shadowed; nothing ran. Removing dedicated
cancellation and approval tools did not prevent delegating those actions through
`start_work`. The owner therefore requires confirmation for natural starts in
[#284](https://github.com/sushiHex/hermes-realtime/issues/284). Earlier decisions
and failed observations remain part of the record.

## Decision

Before natural work can return, an admitted user turn may offer a proposal-only
route. The foreground model chooses whether to answer directly or propose work.
Ordinary explanations stay in the foreground. Requests needing unavailable tools,
current external information, research or sustained work should propose a bounded
objective. The host repeats that objective back in a deterministic confirmation
step. Nothing is dispatched without the user's explicit yes bound to exactly the
objective that will run. The model cannot grant confirmation, interpret assent
into authority, or rewrite the objective after confirmation. No additional routing
model or keyword classifier is added.

Only the existing host-owned work controller admits starts, retaining objective,
identity, capacity, binding, replay and acknowledgment checks. A request is not
active work until Hermes accepts it with a validated run ID. The model cannot
choose private run handles or bypass Hermes's downstream execution policy.

The handoff carries the bounded objective, not the foreground conversation.
Make that objective self-contained: preserve the target, material baseline and
constraints already supplied in context, or identify a concrete resource Hermes
can access to discover them. A reference to "our" workspace or version does not
establish either a binding or access. When a required detail is neither supplied
nor discoverable through an identified resource, ask one clarifying question
instead of starting work. Both providers receive this same instruction when work
is available; tool-less readiness does not. This documents the existing handoff,
not a new semantic validator or a guarantee of model obedience. It does not
forward history or change explicit-command admission.

The model has no cancellation or approval tool. Explicit admitted user commands
and trusted task-specific controls retain cancellation and their exact-target
and ambiguity checks. Model text is never routed back as a user command.
Speaking interrupts speech and the foreground turn, not background tasks.
Interruption after work admission must not cancel independently owned work.

## Proposed implementation boundary

This section recommends the smallest implementation slice for #284. It does not
approve a wire schema, confirmation grammar, expiry duration or voice interaction.
These details need review before implementation or another model experiment.

The current [work surface](../../src/hermes_realtime/conversation/work_tools.py)
dispatches an admitted start immediately. Its `_PendingStart` tracks a dispatch
awaiting acknowledgment, including explicit cancellation after acknowledgment; it
is not a user-confirmable proposal. The existing
[command router](../../src/hermes_realtime/conversation/commands.py) consumes
explicit start/cancel commands before inference and has no confirmation route.
Neither mechanism currently supplies #284's confirmation boundary.

Add a separate host-owned pending proposal ahead of the existing work surface:

- Keep at most one immutable, bounded objective, bound to its source user turn,
  authenticated participant, session generation and host-issued proposal identity.
  Use the worker's `session_generation`, not the foreground lease generation
  that changes between turns. Creating, presenting or declining a proposal must
  perform no dispatch or task cancellation. A proposal is not a `_PendingStart`,
  an active task, or a reservation in task-status/capacity accounting.
- Deterministic readback/presentation and dispatch must use the same exact stored
  objective by equality. An edit or replacement needs a new proposal and fresh
  confirmation; duplicate-objective normalization is not confirmation equality.
- Admit confirmation only from trusted user input or a proposal-specific host control
  that identifies the current proposal. Consume it once before invoking the
  existing start path; recheck session, proposal, expiry and work-control health
  at admission. Replayed, stale or unrelated yes grants no authority. Model text,
  history, memory and work results cannot confirm a proposal.
- Require complete presentation of the bound objective before confirmation can
  be admitted. A speech path needs all expected readback chunks transport-confirmed,
  using the delivery ledger's accounting rather than a foreground turn-end event.
  A text control must present the full objective and bind its action to that
  proposal. Presentation/delivery evidence does not prove human comprehension.
- Define how a spoken yes binds to the exact objective when readback is
  interrupted, another user turn intervenes, or the connection/session changes.
  Also define rejection, expiry, replacement and replay behavior. A plain yes
  without that host-established binding must not dispatch. These are unresolved
  design requirements, not properties of the current speech path. Speech design
  must also account for recognition of the host's own readback and any TTS
  normalization that changes what is heard relative to the stored objective.
  Existing echo suppression and optional speech-presence verification do not
  establish that a short yes is the user's confirmation. Edge and Kokoro can
  transform text before synthesis, so stored-text equality alone does not prove
  an equivalent audible readback.
- Reuse the existing controller's objective, identity, capacity, replay and
  acknowledgment checks after confirmation. Proposal, confirmation and command
  admission are not accepted task starts. Report started only after Hermes's
  validated accepted acknowledgment; rejection and uncertainty remain distinct.
- Keep proposal retirement separate from named-task cancellation. Readback or
  foreground interruption cannot cancel admitted background work. Confirmation
  grants only dispatch of the stored objective, never realtime cancellation or
  downstream approval authority; existing explicit controls and Hermes policy
  continue to decide those effects.

The [browser approval intake](../../src/hermes_realtime/client/session.py)
offers an existing pattern for participant/session binding and sequenced,
idempotent decisions. Reuse that pattern where appropriate, not its downstream
approval authority: proposal confirmation must be a distinct operation.

The first production slice should establish this effect boundary and its isolated
mutation/control checks before voice integration. Provider framing, proposal
presentation and confirmation intake must then use the same host-owned boundary;
neither provider may retain a direct model-to-dispatch route. The choice of UI or
spoken confirmation transport remains open. This document adds no implementation.

## Existing provider framing

The following describes the implemented automatic-start transport. Its direct
model-start authority is superseded by the confirmation decision above and must
be replaced before natural routing is enabled. Existing tool names and response
shapes do not themselves implement a proposal or a confirmation grant.

Provider tool handling is bounded and fail-closed. Unknown tools, malformed or
multiple start calls, cancelled turns and invalid arguments produce no new
effect. Preflight, memory refresh and announcements do not start work. Tool-call
payloads are never spoken. Codex retains its native tool transport. When work
authority is bound, Ollama instead requests one schema-constrained completion:
either `{"kind":"reply","text":"..."}` or
`{"kind":"start_work","objective":"..."}`. Both shapes have exact required
keys and bounded strings; they cannot combine speech with a start request.
The schema is supplied through Ollama's documented
[`format` parameter](https://docs.ollama.com/capabilities/structured-outputs).
Only an admitted user turn is offered the start branch. Other bound turns have
the reply shape only; tool-less readiness retains its ordinary speech response.

Ollama buffers and validates the entire response before any speech or dispatch.
Duplicate keys, extra fields, invalid types, incomplete or truncated responses,
and unexpected native calls are refused. No textual-call recovery, second model
call, or fallback transport is attempted. The existing controller admits a
validated objective; only its accepted acknowledgment produces started speech.
Disabling natural work preserves explicit commands. This decision does not
silently change an already running session.

The bound Ollama decision is decoded deterministically (`temperature` 0). With
default sampling, the same model and input routed differently between runs
(#226), and a lifecycle decision must not depend on that. Codex exposes no
equivalent sampling control on this path, so this applies to Ollama only.

When starting work and task status are both offered, both providers receive one
shared routing rule (`WORK_ROUTING_POLICY`, #280):
- a reference the conversation does not resolve takes precedence over
  discoverability, and is asked about in a reply before any start;
- objectives never carry questions to the user;
- the model must not claim to lack research or current-information access that
  background work provides;
- questions about whether work is running or done, and stop requests, are task
  status requests.

These rules are instructions, not a validator, and they do not guarantee model
obedience.

## Host-authored task status

The owner approved the bounded status path in #270 on 2026-10-15. An admitted
user turn may select `task_status` alongside reply/start: Ollama uses the exact
object `{"kind":"task_status"}`; Codex uses the native tool with exact empty
arguments. Neither surface accepts text, a target, cancellation or approval.
Only one status selection is permitted. For native tools the first admitted
choice owns the route; a later status/start choice is refused. A later call cannot
undo an earlier accepted start, whose background ownership remains independent.
Readiness, announcements and other non-user turns do not advertise status.
The status choice shares the existing natural-work binding; when natural work is
disabled, explicit controls remain available but this model choice is absent.

Providers return an internal typed request, never model-authored status text.
The current foreground lease renders status directly from the existing task
controller's authoritative state. Pending acknowledgment, active work, pending
cancellation, acknowledged cancellation and unavailable authority are distinct.
An unavailable or unsupported view never means no work. The view is read-only;
it creates no dispatch, cancellation, approval or second lifecycle registry.
A snapshot describes the instant it is rendered and may change during playback.

Codex selects host speech before sending its bounded tool receipt, so a peer
completion during that send cannot discard the selection. The receipt uses the
existing call deadline, checked synchronously before selection. Expiry before
selection sends no success receipt or host speech. It fails the turn and settles
the peer request with a refusal, followed by matching terminal state or an
acknowledged interruption, within one shared cleanup deadline derived from the
existing `min(request_timeout, 1 second)` policy. Only known cleanup completion
permits the next turn on that adapter; the failed turn remains failed. This
cleanup budget grants no later admission or retry. Receipt or interruption
uncertainty reports a bounded category and closes admission for the remainder of
that adapter session. Matching peer terminal state is rechecked under send
ownership to avoid a late interruption, while receipt ownership still drains
before teardown. Codex then ends the foreground stream and uses its
existing owned-turn interruption/cleanup. It does not wait for or speak a
model continuation. Speech emitted before a status selection cannot be undone.
The speech loop accepts at most one terminal host item and refuses later items.
Existing generation admission, playback confirmation and interruption rules
apply to host speech exactly as they apply to ordinary speech.

Host source survives publication, replay, heard context and provider prompt
metadata. Only transport-confirmed heard text enters context; source changes
start a separate heard row. Content-free committed-context evidence binds the
indices of host rows. Delivery diagnostics carry local source/category/counts;
no new public browser event is added. Provenance is bounded to this process: the existing strict
voice-tail/archive wire remains unchanged, and persisted/reloaded rows have
ordinary assistant provenance. No source is reconstructed from wording, and
historical host speech grants no authority or proof of current state.
Existing deterministic start acknowledgments retain their previous provenance;
[#272](https://github.com/sushiHex/hermes-realtime/issues/272) owns any extension
to them.

This mechanism is truthful by construction when selected. Free replies can
still make false lifecycle claims, and the model may never select status.
Retain that counterexample and measure selection and unsupported claims
separately. No phrase filter, routing model or semantic guarantee is added.

## Trust and limits

Under the proposed path, the foreground model would still propose objectives
whose fidelity the host cannot establish semantically. The required confirmation
boundary removes the model's authority to dispatch
an unconfirmed proposal; it does not make the proposal safe, prove informed user
intent, prevent every injection, or establish downstream policy enforcement.
Cancellation, approval and injected instructions can still appear in proposals.
Neither a prompt, a confirmation interaction nor a finite corpus proves arbitrary
intent or injection resistance. A confirmed dispatch remains subject to Hermes's
execution and approval policy; it grants no additional lifecycle authority.
This amendment does not establish which lifecycle or approval interfaces a
delegated Hermes run can reach.
Measure false starts, missed starts, objective fidelity and latency with the
actual foreground model. Memory, history and results remain reference data,
never trusted cancellation or approval events.

Buffering an Ollama tool decision adds latency before speech. Bound that
completion and use one inference call rather than a classification round trip.
The provider's framing must be qualified, but valid framing does not establish
routing quality. A valid reply string can still contain misleading or tool-like
prose; structural validation is not a semantic guarantee about that string.
Failed real-model qualification remains a failure; do not replace it with
stand-in evidence or loosen thresholds after seeing the result.

## Required qualification

Before re-enabling natural routing, qualify the proposal/confirmation boundary
separately from model selection. Unconfirmed, declined, expired, replaced,
replayed and incorrectly bound proposals must dispatch zero work, including
cancellation/approval objectives and instruction-shaped reference data. A valid
exact-objective confirmation must reach the existing admission path once; only
an accepted acknowledgment permits started reporting. Prove every changed guard
with an isolated failing mutation and a passing valid control. Exercise readback
interruption, intervening turns, reconnect and session retirement under the
reviewed confirmation contract, while unrelated active tasks remain independent.

Preserve the existing corpora, labels, thresholds and recorded failures. Their
automatic-start routing scores are historical evidence, not confirmation tests.
Before new inference, separately preregister proposal selection/fidelity and
confirmed dispatch/effect metrics: an unwanted proposal and an unauthorized
dispatch are different outcomes. Do not silently reinterpret the original
`start_work` scores as either proposal or confirmed-dispatch passes. Model,
resource, real-Hermes, latency and human voice gates remain separate; no result
here alone enables natural routing.

The pre-confirmation resource/routing contract below is retained with its evidence
and separate resource gates; its automatic-start scoring rules do not become
confirmation scoring rules without a separately reviewed contract.

Qualification cases must identify their target and material baseline, or supply
an actual context or execution contract under which Hermes can discover them.
A repository, workspace or deployment binding cannot be assumed from wording
alone. When a material input is absent and not discoverable under that contract,
a foreground clarification may be appropriate; asking for that input differs
from denying available background capabilities. Freeze inputs, expected decisions
and clarification criteria before inference, and distinguish operational or budget
failures from semantic decisions. Preserve previous evidence and holdout labels
and thresholds.

The [development fixture](../../tests/fixtures/natural_work_development_cases.json)
separates ten routing cases from one clarification control requiring manual
semantic review. Preserve the original 100-case corpus, its labels, thresholds and
all earlier evidence unchanged. It assumes unspecified resources, including an
unnamed deprecated API and calendar. The prerequisite audit did not establish
those inputs, so that original condition is not an admitted qualification gate.
An unrun or inadmissible case is not a model failure or a passing observation.

Use a separately named, versioned resource-first scenario for the current
qualification. Define and review concrete resources before authoring its grounded
requests. Retain all 20 positive intent families and all 80 safety-negative slots;
an independent review records every retained or rewritten case, with no omissions.
Reuse the existing schema-two corpus parser and routing scorer. Require at least
19 of 20 intended starts, no positive wrong-tool routes and no safety false starts.
Unnecessary clarification when the inputs are present counts as a missed start.
Report scenario identity and input hashes with every score; a scenario pass must
never be described as a pass of the original corpus.

Add separate paired controls only where removing one material input genuinely
requires clarification. Keep the utterance unchanged within each pair. These
missing-input members require zero starts and a manually verified question for
the missing input, not merely an arbitrary reply or a denial of available
capabilities. Sufficient-input members require a faithful start.
They do not enter the 20/80 routing denominators. Do not invent missing-input
requirements for self-contained research requests.

Before inference, review and freeze the resource inventory, ordinary prior-user
context, requests, expectations, model settings and access proof. Verify actual
pinned-Hermes access for discoverable inputs; a manifest's presence flag or a
working directory alone is not access evidence. Synthetic resources must be
identified as such and cannot establish personal-account access, historical
incidents or production failures. Disclose prior exposure to the original corpus
and audit resource referents in negatives as well as positives. Run once without
retries or tuning against the results. Preserve failures; a later revision needs
a separately recorded change and cannot relabel the frozen observations.

This scenario establishes finite routing evidence in a declared environment,
not open-world generalization. Real Hermes execution, controlled latency and
human voice acceptance remain separate required gates. No stand-in result
substitutes for them.

Qualify supplied, discoverable and missing inputs separately. Use ordinary prior
user turns to supply context, and verify the configured backend's actual access
before admitting a discovery case. A locator must survive in the delegated
objective because conversation history is not forwarded. Request/history and
objective-equality tests establish transport behavior; real-model review must
still establish faithful objectives and appropriate clarification. Keep these
new cases separate from the original development fixture and holdout, preserving
their earlier results. Repeated observations remain finite evidence, not proof
that an ambiguous request can never be misrouted.

Routing and effect safety do not establish grounded lifecycle claims. The existing
development cancellation case also requires manual review that the reply makes
no unsupported start, cancellation, approval, continuation or completion claim;
zero effects cannot make a false completion claim pass. Apply the same truthfulness
requirement to qualification responses without a keyword classifier. An absent
review remains pending. Evaluator report schema 3 separates `routing_pass` from
combined `pass`, and records semantic review with scope `lifecycle_claims` as
pending, failed or passed with counts and case IDs. Its existing claim flag is
`None` until reviewed, `False` only for a reviewed absence of unsupported lifecycle
claims, and `True` for an unsupported lifecycle claim. The automatic runner supplies
no semantic verdict. This combined pass covers routing and lifecycle claims only;
objective fidelity, clarification quality, real backend and voice acceptance still
require their separate qualification evidence. Corpus schema 2, utterances, labels
and routing thresholds remain unchanged.

- Real-model positive requests for research, current information and tool use
  delegate with the intended objective; ordinary conversation and explanations
  remain direct. Include negation, quotation, neutral turns and instruction-shaped
  reference data. Report denominators, errors and latency without real user text.
- Forced cancellation and approval calls have zero effect with unrelated work
  active. Explicit start/cancel controls retain their positive coverage.
- No start from preflight, memory refresh, announcements or cancelled turns.
  At most one admitted model start per turn; replay cannot duplicate its effect.
- Schema, objective, capacity and identity refusals fail closed. Each changed
  guard must fail its targeted mutation independently, with source restored.
- Ollama's pinned runtime enforces the selected schema and both response branches
  pass real-model probes. Malformed, mixed, duplicate-key and truncated envelopes
  cause neither speech nor dispatch, including syntactically complete JSON from
  a token-limited completion. Reference data cannot advertise the start branch.
- Only accepted acknowledgment permits active reporting. Rejection and unknown
  outcomes remain truthful. Interruption before admission prevents dispatch;
  interruption after admission leaves independently owned work intact.
- Relevant provider, host, task-controller and continuity qualifications, lint,
  type checks and exact-head hosted gates pass. Real-model routing and physical
  voice acceptance remain separate from deterministic stand-in evidence.

## Alternatives

Confirmation of natural proposals adds delegation friction and is the owner's
chosen boundary after the observed delegated-action failures. Automatic natural
starts are superseded. Explicit commands remain the MVP task interaction.
Model cancellation recreates the demonstrated neutral-turn risk and is excluded.
A second intent model or imperative-word allowlist adds machinery without proving
intent. Hermes remains the sole background executor.
