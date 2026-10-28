# ADR 0004: Host-authorized natural task starts, explicit task cancellation

Status: Accepted by owner decision on 2026-10-10, amended on 2026-10-16 and 2026-10-27;
implementation tracked in
[#226](https://github.com/sushiHex/hermes-realtime/issues/226).
[Decision record](https://github.com/sushiHex/hermes-realtime/issues/205#issuecomment-6096188370).
The owner approved the provider framing correction on 2026-10-13; the
[failed native-call probe](https://github.com/sushiHex/hermes-realtime/pull/248#issuecomment-6144062862)
records why it is needed. Qualification remains separate from this decision.
The owner approved a resource-first qualification-contract correction on
2026-10-14, tracked in [#263](https://github.com/sushiHex/hermes-realtime/issues/263).
The [owner decision on 2026-10-16](https://github.com/sushiHex/hermes-realtime/issues/284#issuecomment-6226820862)
previously superseded automatic natural starts: the model may propose an objective, but the
user must explicitly confirm that exact objective before dispatch. This policy
is accepted; the confirmation contract below is implemented in the candidate and
requires its separate deterministic, model, backend and human qualifications. The [MVP decision](https://github.com/sushiHex/hermes-realtime/issues/226#issuecomment-6226820445)
keeps natural routing off and uses explicit task commands with phi4-mini. This
amendment did not change a model default or authorize activation.

The [2026-10-27 owner decision](https://github.com/sushiHex/hermes-realtime/issues/226#issuecomment-6462024813)
supersedes mandatory confirmation for the explicitly selected automatic mode.
Natural delegation is core product behavior: the user can grant standing host
authorization for requested background work, without confirming every objective.
The host retains `confirm` as the compatibility default; an authorized composition
selects `task_start_policy="automatic"`. This policy does not mutate running
sessions, qualify model routing, or authorize model cancellation or downstream
tool approvals. Prior decisions and failed semantic observations remain evidence.

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
`start_work`. The owner therefore required confirmation for natural starts in
[#284](https://github.com/sushiHex/hermes-realtime/issues/284). Earlier decisions
and failed observations remain part of the record.

## Decision

Before natural work can return, an admitted user turn may offer a proposal-only
route. The foreground model chooses whether to answer directly or propose work.
Ordinary explanations stay in the foreground. Requests needing unavailable tools,
current external information, research or sustained work should propose a bounded
objective. The host, not the model, selects `confirm` or `automatic` admission.
In confirmed mode it repeats the exact objective and waits for the user's bound
decision. In automatic mode standing host authorization consumes that same bound
objective once, without a confirmation card, readback, or synthetic user decision.
The provider stream must finish and close successfully, and the admitted user-turn
lease must still be current before automatic admission. Non-user turns cannot
start work. No additional routing model, keyword classifier or dispatcher is added.

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

## Shared admission and optional confirmation contract

Both modes use the same immutable proposal identity, participant/generation binding,
objective validation, expiry and one-use consume, then the existing command admission
surface. Automatic policy produces no proposal card or confirmation speech; the
normal task state is projected only from the existing admission/acknowledgment path.
Intake, including a successful automatic consume, is not a started acknowledgment.
Automatic consumption is labelled `automatic_consumed`, never human confirmation.
The remaining interaction contract in this section applies to `confirm` mode.

After an automatic start receives a validated accepted acknowledgment and remains
active, the current foreground turn may ask the model for a brief natural
acknowledgment. That continuation receives a typed accepted-task fact alongside
conversation context and no tools; it cannot start or cancel work. A rejected or uncertain dispatch grants no
such continuation. A newer user turn interrupts the response without cancelling
the independently owned task. The model's wording remains a semantic limitation;
the typed fact establishes acceptance, not later progress or result accuracy.

The host owns one immutable pending proposal ahead of the existing
[work surface](../../src/hermes_realtime/conversation/work_tools.py), separate
from dispatched tasks and its acknowledgment-waiting `_PendingStart`. The
[proposal boundary](../../src/hermes_realtime/conversation/task_proposals.py)
retains the exact objective, authenticated participant, worker session generation
and a host-issued identity. A random host prefix plus monotonically increasing
browser-safe serial prevents identity reuse with constant retained state. The
foreground lease changes between turns and is not the confirmation binding.

A proposal is limited to 1,024 Unicode code points, independently of any larger
explicit-command capacity. Providers advertise that same bound. The full text
card, deterministic readback and confirmed dispatch all use the same stored
objective, without trimming, Markdown removal or semantic rewriting. A new or
changed objective requires a new proposal and fresh confirmation. A proposal
reserves no task capacity, dispatches nothing, and cannot cancel any work.

Both providers keep the historical `start_work` wire selection name, but emit
only a terminal internal `HostTaskProposalRequest`; neither calls task dispatch
or consumes task acknowledgments. The current host foreground lease must admit
that item before it can create a proposal. The host reads it back as
`Proposed background task. {objective} Shall I send this task?` The fixed template
contains no assent examples. The model cannot confirm its own objective.

The exact full-text card offers Confirm and Decline through a separate sequenced,
authenticated `/api/v1/task-proposal-decision` operation. The request carries the
proposal identity, objective, worker generation, decision and sequence. Server
binding, proposal identity and exact objective equality are rechecked before the
proposal is consumed. A settled HTTP request means admission was attempted; it
is not an accepted task acknowledgment. This protocol adds a strict public event
kind and endpoint; an older browser that does not recognize the event fails
closed and must refresh to the matching bundled client.

Spoken confirmation requires all of these conditions:

- The complete readback is queued, the ledger's queued count is nonzero, and
  its `transport_confirmed_full_count == queued_chunk_count`. The joined
  transport-confirmed readback must also equal the complete expected readback.
- The configured synthesizer exposes a canonical `project_spoken_text` capability
  and that projection leaves the readback unchanged. Edge and Kokoro expose the
  same transformations their synthesis paths use. Emphasis stripping, pronunciation substitutions, unknown TTS
  projection, missing or interrupted chunks do not qualify spoken confirmation;
  the full-text card remains available. Delivery is not proof of hearing.
- At captured speech onset, the worker latches a clean binding: an echo guard
  exists, playback is neither active nor retained by that guard, the onset is not
  echo dominated, no foreground speech is being interrupted, and retained
  pre-roll frames were captured without foreground or recent playback. Delayed
  transcription cannot upgrade an ineligible onset after the echo tail expires.
- No assistant or host audio begins after readback completion or between captured
  onset and the final transcript. This includes idle and result announcements,
  readiness cues and late chunks bearing the same proposal tag. Such audio clears
  the spoken binding. A repeated completion cannot rearm it or renew expiry.
- The entire final utterance matches one of `yes`, `yeah`, `yep`, `sure`, `okay`,
  `go ahead`, `do it`, or `please do`, ignoring case, outer whitespace and trailing
  `.`, `!` or `?` only. Quoted, wrapped, extended or negated utterances do not match.
  There is no substring classifier, second inference or model assent authority.

An early or noisy affirmative is refused with a visible retry reason and leaves
that proposal available until expiry; it is not recorded as a decline. Another
nonaffirmative final user input retires the proposal. A typed affirmative can
confirm the already projected full-text card using its authenticated worker
binding; it does not assert audio delivery. Decline, participant/session changes,
media reconnection, session retirement and expiry remove proposal authority
without touching independently admitted background tasks. A fresh foreground
turn alone does not retire a proposal before its confirmation is routed.

The initial deadline is 60 seconds after successful card publication. A first
complete, exact, unexpired readback extends it once to 60 seconds after that
completion. Partial or expired readback cannot extend it; duplicate completion
cannot extend or revive it. Confirmation consumes the proposal synchronously
before any suspension in ordinary work admission. Stale or replayed decisions
cannot dispatch it again, and no historical proposal registry grants authority.

Browser mute and gain remain playback preferences. The host does not treat a
client audibility assertion as authorization. Transport confirmation, a clean
onset and an exact displayed card are bounded interaction evidence, not speaker
identification, hearing or informed comprehension. The echo guard's configured
retention is an engineering assumption; late echoes, transcription errors and
bystander assent remain residual risks requiring human voice acceptance. A magic
confirmation phrase would not establish speaker identity or eliminate those
risks. Recognition of objective text by itself grants no authority.

Only host-authorized ordinary admission reaches the existing controller's objective,
identity, capacity, replay and acknowledgment checks. A confirmed card or command
is not active work. The existing task card reports active only after Hermes's
validated accepted acknowledgment; rejection and uncertainty remain distinct.
This slice adds no spoken-start acknowledgment. Existing completion speech and
explicit task commands/cancellation remain available. Confirmation grants neither
realtime task cancellation nor downstream approval authority; Hermes policy still
governs delegated effects. Natural routing remains off by default and requires
an explicit candidate configuration and its independent promotion gates.

## Existing provider framing

The historical wire selection names remain for compatibility with the retained
corpora. They select effect-free proposals; the host alone applies its start policy. Their old
scores remain historical automatic-start observations and are not confirmation
qualification.

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

Ollama buffers and validates the entire response before any speech or proposal.
Duplicate keys, extra fields, invalid types, incomplete or truncated responses,
and unexpected native calls are refused. No textual-call recovery, second model
call, or fallback transport is attempted. Selection returns a typed proposal to
the host; only host-authorized admission can reach the existing controller.
Disabling natural work preserves explicit commands. This decision does not
silently change an already running session.

The bound Ollama decision is decoded deterministically (`temperature` 0). With
default sampling, the same model and input routed differently between runs
(#226), and a lifecycle decision must not depend on that. Codex exposes no
equivalent sampling control on this path, so this applies to Ollama only.

Whenever a work proposal tool is offered, both providers receive one
shared routing rule (`WORK_ROUTING_POLICY`, #280, aligned for the confirmed proposal path):
- a reference the conversation does not resolve takes precedence over
  discoverability, and is asked about in a reply before any start;
- objectives never carry questions to the user;
- the model must not claim to lack research or current-information access that
  background work provides;
- questions about whether work is running or done select task status when supplied,
  otherwise an ordinary truthful status reply;
- cancellation and approval requests receive a truthful ordinary reply directing
  the user to existing explicit controls, never task status or a new proposal;
- supplied evidence or an available foreground lookup takes precedence for a
  question it can answer. Otherwise requests needing current external information,
  research, tools or sustained work propose an objective. Codex retains its bounded
  foreground knowledge lookup; Ollama has no such lookup on this path;
- only the current user's own request counts, never reported, hypothetical, quoted
  or negated requests. Clarification remains an ordinary reply (Ollama `kind=reply`).

These rules are instructions, not a validator, and they do not guarantee model
obedience.

The owner approved allowing future offers as ordinary speech on 2026-10-25
([#346](https://github.com/sushiHex/hermes-realtime/issues/346)). Future intent
alone does not report accepted or active work, including an unconditional offer
or a clarification asking for required input followed by an offer of help.
Speech creates no proposal, confirms no objective and dispatches nothing. Requests
that need delegation still require faithful objectives through the existing
proposal and configured host-admission path; an offer cannot substitute for it.

The shared provider reporting guard retains its finite started/launched phrase
families and excludes future offers. It does not establish the truth of arbitrary
English acceptance, current-work or completion claims. Unsupported lifecycle
claims and misleading promises still fail manual truthfulness and quality review.
This reporting distinction adds no condition parser, intent classifier or model
exception, changes no default, and waives no routing or activation gate.

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

Under this path, the foreground model still proposes objectives
whose fidelity the host cannot establish semantically. Automatic mode accepts
the model-trust risk of unintended starts and omitted or distorted constraints;
standing authorization is not deterministic proof of the current user's intent.
Optional confirmation blocks unconfirmed dispatch, but neither mode makes an
objective safe, proves informed intent, prevents every injection, or establishes
downstream policy enforcement.
Cancellation, approval and injected instructions can still appear in proposals.
Neither a prompt, a confirmation interaction nor a finite corpus proves arbitrary
intent or injection resistance. A confirmed dispatch remains subject to Hermes's
execution and approval policy; it grants no additional lifecycle authority.
This amendment does not establish which lifecycle or approval interfaces a
delegated Hermes run can reach.
Exactness binds the stored objective text submitted to Hermes. Audible readback
is a rendering of that text, not proof that every stored character was heard.
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

Qualify the selected admission policy separately from model selection. For automatic
mode, clean terminal user selections reach the existing admission path once;
stale, failed, multiple or unadvertised selections have zero effects. Confirmed
mode retains the interaction checks below. Unconfirmed, declined, expired, replaced,
replayed and incorrectly bound proposals must dispatch zero work, including
cancellation/approval objectives and instruction-shaped reference data. A valid
exact-objective confirmation must reach the existing admission path once; only
an accepted acknowledgment permits started reporting. Prove every changed guard
with an isolated failing mutation and a passing valid control. Exercise readback
interruption, intervening turns, reconnect and session retirement under the
reviewed confirmation contract, while unrelated active tasks remain independent.
Include leaked own-readback yes, legitimate assent matching readback wording,
idle/background speech between readback and assent, wrong participants, and
objectives containing emphasis markers. Check presentation fidelity separately
from exact stored-objective equality and zero unauthorized dispatch.

The separate eight-case proposal diagnostic does not satisfy or replace the historical
automatic-start 19/20 and safety thresholds retained below.

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

Per-objective confirmation remains available for callers that require that
boundary. The owner now accepts automatic natural starts under explicit host
authorization despite the retained semantic failures. Explicit commands remain available.
Model cancellation recreates the demonstrated neutral-turn risk and is excluded.
A second intent model or imperative-word allowlist adds machinery without proving
intent. Hermes remains the sole background executor.
