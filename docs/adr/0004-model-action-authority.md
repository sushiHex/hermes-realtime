# ADR 0004: Natural task starts, explicit task cancellation

Status: Accepted by owner decision on 2026-10-10; implementation tracked in
[#226](https://github.com/sushiHex/hermes-realtime/issues/226).
[Decision record](https://github.com/sushiHex/hermes-realtime/issues/205#issuecomment-6096188370).

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

The owner superseded the earlier confirmation-for-both policy with automatic
starts and explicit user cancellation. The earlier design remains in #205 and
Git history. No confirmation UI or proposal/grant lifecycle is required now.

## Decision

Advertise `start_work` for an admitted user turn when natural work is enabled.
The foreground model chooses whether to answer directly or delegate to Hermes.
Ordinary explanations stay in the foreground. Requests needing unavailable tools,
current external information, research or sustained work should delegate with a
bounded objective. No additional routing model or keyword classifier is added.

Only the existing host-owned work controller admits starts, retaining objective,
identity, capacity, binding, replay and acknowledgment checks. A request is not
active work until Hermes accepts it with a validated run ID. The model cannot
choose private run handles or bypass Hermes's downstream execution policy.

The model has no cancellation or approval tool. Explicit admitted user commands
and trusted task-specific controls retain cancellation and their exact-target
and ambiguity checks. Model text is never routed back as a user command.
Speaking interrupts speech and the foreground turn, not background tasks.
Interruption after work admission must not cancel independently owned work.

Provider tool handling is bounded and fail-closed. Unknown tools, malformed or
multiple start calls, cancelled turns and invalid arguments produce no new
effect. Preflight, memory refresh and announcements do not start work. Tool-call
payloads are never spoken. Ollama uses native tools and acknowledgment-derived
speech rather than speaking a buffered promise before dispatch is accepted.
Disabling natural work preserves explicit commands. This decision does not
silently change an already running session.

## Trust and limits

Automatic delegation deliberately trusts the foreground model to recognize the
user's request. A structurally valid unwanted start remains possible. Neither a
prompt nor a finite corpus proves arbitrary intent or injection resistance.
Measure false starts, missed starts, objective fidelity and latency with the
actual foreground model. Memory, history and results remain reference data,
never trusted cancellation or approval events.

Buffering an Ollama tool decision adds latency before speech. Bound that
completion and use one inference call rather than a classification round trip.
Native tool capability is necessary but does not establish routing quality.
Failed real-model qualification remains a failure; do not replace it with
stand-in evidence or loosen thresholds after seeing the result.

## Required qualification

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
- Only accepted acknowledgment permits active reporting. Rejection and unknown
  outcomes remain truthful. Interruption before admission prevents dispatch;
  interruption after admission leaves independently owned work intact.
- Relevant provider, host, task-controller and continuity qualifications, lint,
  type checks and exact-head hosted gates pass. Real-model routing and physical
  voice acceptance remain separate from deterministic stand-in evidence.

## Alternatives

Confirmation on every start avoids autonomous work effects but adds delegation
friction. Explicit commands remain a fallback, not the requested interaction.
Model cancellation recreates the demonstrated neutral-turn risk and is excluded.
A second intent model or imperative-word allowlist adds machinery without proving
intent. Hermes remains the sole background executor.
