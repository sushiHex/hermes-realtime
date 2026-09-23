# ADR 0003: Realtime converses; Hermes keeps the turns

Status: Proposed for review (revised; owner-agreed direction). Work items:
[#77](https://github.com/sushiHex/hermes-realtime/issues/77),
[#80](https://github.com/sushiHex/hermes-realtime/issues/80), and
[#159](https://github.com/sushiHex/hermes-realtime/issues/159).

Reviewed upstream source: Hermes Agent
[`v0.21.0` at `29112bef`](https://github.com/NousResearch/hermes-agent/commit/29112bef099274229cadff79cdff7bf7b99c4b77),
the owner's qualification baseline rather than a compatibility ceiling. This record does not
claim that the design is implemented.

## Context

hermes-realtime is the orchestrating messenger agent. It owns the voice: foreground inference,
speech, barge-in, and the ledger of what was delivered. It wields the Hermes agent harness for
everything else: real work, recall, and Hermes's self-learning. Hermes never speaks through
realtime. To the user this is one assistant.

#77 already fixes the integration approach, and this record applies it. Hermes stays
authoritative for sessions, history, memory, search, and compaction. Realtime adds the smallest
layer it can. Reading history, saving an externally produced turn, and running a Hermes agent
turn are three distinct operations. A missing capability is met by a small upstream extension,
never by writing Hermes's private files.

The #80 audit of v0.21.0 established four facts this design rests on:

- **No write without a model run.** No route adds a message without running a model. The
  session routes create, read, patch, delete, fork, and chat
  ([routes](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L2237-L2246)),
  and the server reports no external memory-write API.
- **Learning happens only in a live turn.** Hermes learns through its background memory and
  skill review. That review starts only at the end of a live agent turn
  ([spawn](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_finalizer.py#L806-L817)),
  and its memory cadence counts the user turns in the loaded history
  ([counter](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_context.py#L782-L790)).
  Nothing reviews a stored session.
- **Runs accept what delegation needs.** `/v1/runs` accepts a client `session_id`, an
  `Idempotency-Key` backed by a durable unique reservation, and explicit `conversation_history`
  that is used as context but not re-persisted
  ([session](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L546),
  [idempotency](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L452-L465),
  [history](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L502-L520),
  [not re-persisted](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L2234-L2238)).
- **Runs cannot be listed.** Hermes can report one run by ID, but has no route that lists runs
  ([routes](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L83-L91)).

## Decision

Three mechanisms, and nothing else.

### 1. One Hermes session holds the voice conversation

Each voice conversation is bound to one Hermes session, the voice session, in the one selected
profile. Every voice message is appended to it individually, carrying a realtime-generated
message ID:

- a user row when its transcript is final;
- an assistant row when its delivery settles.

An assistant row may belong to an ordinary reply, an announcement with no user turn, or a
resumed replay. A `task:` command becomes a user row plus its delivered announcement.

Appends form one ordered queue: a message is sent only after every earlier message is
acknowledged. When an outage loses messages, appending resumes at the next user message. The
record therefore never holds a reply without the question it answers.

The assistant text is the transport-confirmed prefix, followed by a fixed interruption marker
when speech was cut off. Confirmation comes from the delivery ledger. It proves the audio
reached the room's verifier, not that a person heard it, and no wording in this record claims
more. The marker is a constant, never model-authored. Generated but undelivered text is never
written.

Resume reads the newest bounded page of the voice session. It admits only user and assistant
text rows and skips anything else. Reads already follow Hermes's compression continuation
([resolution](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4508)).
Older recall is delegated: a run can search sessions.

### 2. Every delegated run is its own Hermes session

Each run carries three things:

- a realtime-generated session ID derived from the task;
- `Idempotency-Key` set to the task ID;
- a bounded tail of the voice conversation as `conversation_history`.

Runs never use the voice session. A run holds its session's cross-process turn lease for its
whole duration, and a voice write must never wait behind background work. With separate
sessions it cannot. The tail gives the work the conversation's context. It also lets Hermes's
own skill review, which runs over the whole message snapshot, learn from voice conversations
wherever real work happens.

### 3. Realtime keeps one binding record

The binding record holds `{profile, voice session ID, task ID → run ID and run session ID}`. It
is bounded, and it is the only persisted realtime state. #77 item 4 admits exactly this kind of
session reference and bounded transient context.

- **Before dispatch.** A task is recorded before its request is sent, together with that exact
  request. Hermes's acknowledgment promotes the entry to its run ID and drops the request. The
  request is the only transcript-bearing content the record ever holds, and only until it is
  acknowledged.
- **Restart, pending entry.** Realtime replays the stored request verbatim under the same
  idempotency key. Hermes returns the original run if it accepted the request, or starts the
  work if the request never arrived. A replay must be byte-identical: a different body is
  refused as a conflict
  ([replay](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L562-L594)).
- **Restart, promoted entry.** Realtime reconciles it with `GET /v1/runs/{id}`. A run Hermes no
  longer knows is reported as lost, never as running.

No accepted run can therefore go untracked, and no retry can start work twice.

### Live context follows the same rule

Today the live foreground context records every generated segment
([`record_assistant_generation`](../../src/hermes_realtime/conversation/streaming.py#L803)),
while the Codex prompt tells the model that earlier assistant messages "represent only speech
confirmed delivered"
([prompt](../../src/hermes_realtime/providers/codex_app_server.py#L2800)). That claim is
false, and the live context diverges from any resumed one.

Live context becomes heard-first: the transport-confirmed prefix plus the same fixed marker. The
undelivered remainder stays in the resumable-replay machinery. The live context, the Hermes
record, and what the user experienced then match by construction.

### The one upstream extension

`POST /api/sessions/{id}/messages` appends externally produced turns without a model call:

- **Body:** `{"messages": [{"client_id", "role", "content"}]}`, with role `user` or `assistant`.
- **Idempotency:** idempotent per `(session, client_id)`, enforced by the route itself. The
  existing `platform_message_id` index is not unique.
- **Target session:** resolves the ID through the compression continuation, as reads do.
- **Writing:** takes the session's turn lease and writes through the existing batch append.
- **Learning:** runs the accounting a live turn runs. It hydrates the memory counter from
  stored user rows and spawns Hermes's existing background review when the profile's cadence
  is reached. Externally produced turns are turns, and no client flag changes that.

This single route provides persisted voice history, memory learning at Hermes's own cadence, and
searchable voice recall, with no realtime store. A memory-read route is separable and deferred.
Until the route ships, voice history is not persisted. No interim adapter writes Hermes's
storage.

## Degraded mode

A backend outage never silences the voice.

- **Appends are best-effort.** A bounded queue holds messages until they are acknowledged. On
  overflow, realtime drops the unacknowledged tail, announces the gap once, and resumes at the
  next user message.
- **A failed resume read** starts an explicit no-continuity conversation.
- **Task truth stays live.** It always comes from the live backend, never from the record, and
  restored text cannot grant dispatch, approval, or cancellation authority.

## Consequences

- **Voice latency is unchanged.** It is today's foreground path.
- **Learning on v0.21.0 is partial.** Skills learning reaches voice through delegated runs.
  Memory learning waits for the route, because the counter would fire only if a supplied tail
  happened to land on the cadence, and the tail is not sized to game it.
- **What the MVP resumes.** On v0.21.0, #159's MVP resumes tasks across a restart but not
  voice history. That limit is stated in #159, not hidden.
- **Memory egress.** Before a voice tail reaches a run, the profile must use built-in memory
  only, because a run's memory sync passes its messages to any configured external provider
  ([sync](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L4556)).
- **Forgetting.** "Forget this conversation" deletes every run session in the binding record,
  which hold objectives rather than voice text, and the voice session's whole compression chain.
  Hermes deletes only the named session and leaves compression successors in place. So forget
  resolves the latest continuation and walks `parent_session_id` back to the bound session,
  deleting each link. If the walk cannot reach the bound session, forget reports itself
  incomplete. Curated memory is a
  separate, explicit operation, and complete file-level erasure is not promised until the delete
  path is qualified
  ([delete semantics](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L14027-L14116)).
- **Retention.** Hermes session auto-pruning stays disabled for the MVP.

## Rejected alternatives

- **Hermes as the voice foreground** (session chat per utterance). It contradicts the product's
  shape. It also cannot hold the invariants on v0.21.0:
  - tools are per profile only;
  - an interrupted reply is persisted as an ordinary assistant message
    ([interruption](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/conversation_loop.py#L4581-L4595));
  - the stream's terminal event is unreliable;
  - no client-defined tools exist to keep delegation with realtime.
- **Runs in the voice session.** The lease would make voice writes wait behind background work.
- **An in-process plugin writer.** It is unreachable in the current host, and it breaks #77 item
  4.

## Implementation sequence

1. **Run adapter.** Add the idempotency key, a per-task session, and a bounded voice tail to
   `/v1/runs`.
2. **Heard-first live context.** Correct the Codex prompt to match.
3. **Binding record.** Add restart reconciliation.
4. **Upstream route.** Propose it. After it lands and a pinned target carries it, append voice
   messages and resume from the voice session.

## Required qualification before acceptance

Each step must prove its guarantees against a qualified exact target:

- a retried or replayed run never starts twice;
- a crash between dispatch and acknowledgment leaves a pending entry that restart resolves;
- a restarted host reports every bound task truthfully;
- live context and the record contain only transport-confirmed text and markers;
- appends are idempotent, ordered, and never block the voice;
- the record never holds a reply without its question;
- resume admits only text rows from the bound session;
- forgetting removes every bound run session and the voice session's whole compression chain.

Each guard follows the repository evidence rules: one bounded, content-free refusal marker, and
one mutation per guard shown to fail alone.
