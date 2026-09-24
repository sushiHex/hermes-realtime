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
  `Idempotency-Key` backed by a unique reservation (durable when the server advertises it), and
  explicit `conversation_history`
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

Hermes already gives every run a session of its own, named by its run ID
([default](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L605-L607)).
Realtime therefore sends no session ID. Each run carries two things:

- **An `Idempotency-Key` minted for that one dispatch.** Hermes scopes keys by profile and API
  key, which every client of the same credentials shares. A task can also be dispatched more
  than once. A per-dispatch key is the only identity that means "this request".
- **A bounded tail of the voice conversation as `conversation_history`.** The tail is sent only
  once live context is heard-first (see below), and only on a profile whose memory is
  built-in, so unheard speech never leaves realtime.

Runs never use the voice session. A run holds its session's cross-process turn lease for its
whole duration, and a voice write must never wait behind background work. With separate
sessions it cannot. The tail gives the work the conversation's context. It also lets Hermes's
own skill review, which runs over the whole message snapshot, learn from voice conversations
wherever real work happens.

A dispatch whose first attempt got no response is resent once, unchanged, under the same key.
Hermes admits a run at most once per key, so the resend either returns the run the first
attempt started or admits the request if it never arrived. After such an ambiguous attempt,
only an admission settles the dispatch. A refused resend proves nothing about the first
attempt, so its outcome is reported as unknown, never as a rejection. The resend happens even
for an abandoned dispatch, because it is the only way to learn a run that must then be stopped.
Resending requires Hermes to advertise durable run idempotency, the only record that answers a
resend truthfully across a Hermes restart
([capability](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L94-L99)).
A resend is also refused once the dispatch is older, by the wall clock, than the retention
Hermes advertises. A host suspended past that window could otherwise find its key pruned and
start the work again. Durability is known only as advertised at startup. If Hermes restarts
onto its in-memory fallback store, the resend guarantee does not hold, and no capability check
can make it atomic.

### 3. Realtime keeps one binding record

Crash recovery stops what a crashed process may have left running; tasks are not resumed. A
crash must end like an orderly shutdown, which already stops every active run. Hermes cannot
list runs, so without a record a restarted host could not find them.

The binding record holds only the runs this process may have running on Hermes, as two kinds of
entry:

- **Pending:** a dispatch sent but not yet admitted, with its idempotency key, when that key was
  minted, and the exact request. The request is the only transcript-bearing content the record
  ever holds, and only until admission. Without durable run idempotency no key is sent, so none
  is recorded.
- **Admitted:** the run's Hermes ID, which also names its session. Never the private protocol
  ID, never the objective.

The record is versioned, bounded by the active-run capacity, and the only persisted realtime
state. It is owned by one host at a time: a host holds an exclusive lock on it from before it
settles the record until after its final write, and a second host fails to start rather than
stop runs it does not own. #77 item 4 admits exactly this kind of bounded transient reference. The profile and the
voice session join it in step 4.

- **Write-ahead.** A dispatch is recorded before its first request is sent. The entry becomes
  the run ID as soon as the run is claimed. It is removed when the run is authoritatively
  terminal, when a stop of it completes, or when Hermes truthfully rejects the dispatch. A
  dispatch whose outcome is unknown stays recorded, and holds a capacity slot, until a restart
  settles it. Every write replaces the record atomically: a temporary file in the same
  directory, flushed and synced, then renamed over the record.
- **Restart, admitted entry.** Realtime stops the run and waits for its terminal status. A run
  Hermes answers `404 run_not_found` for has nothing left running
  ([status](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L1052-L1068),
  [stop](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L1371-L1387)).
- **Restart, pending entry.** An entry with a key younger than the advertised retention is
  replayed once, verbatim, under its stored key, by the rules of the in-process resend: only an
  admission naming an exact run counts. Hermes returns the original run if it accepted the
  request, or starts the work if the request never arrived; either way realtime then stops it. A
  replay must carry the same parsed body, because Hermes fingerprints the parsed JSON; a
  different body is refused as a conflict
  ([replay](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L562-L594)).
  An entry without a key, older than the retention, or whose replay is refused or unanswered is
  never resent, because Hermes may have pruned its record and would start the work again
  ([pruning](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_run_idempotency.py#L237-L294)).
  Its outcome is reported as unknown.
- **Reported once.** The restart then writes the record empty, emits one content-free marker
  with the counts, and tells the operator once how many runs were stopped or had already ended
  and how many outcomes are unknown. If any settlement fails, start fails closed and the record
  keeps exactly the unsettled entries.

With the record in place, a recorded run cannot outlive the next successful start, and no
replay can start work twice. Before it, the in-process resend recovers one lost acknowledgment;
a dispatch whose resend also gets no response ends with its outcome unknown.

### Live context follows the same rule

Live foreground context is heard-first. Assistant rows fill only from delivery-confirmed chunks:
each generated segment has one row, holding the exact slice of the segment text through its
latest confirmed chunk. Interruption is data, not text: a turn that ends abnormally after some of
its speech was confirmed sets `interrupted` once on its last confirmed row, whose text stays
exactly what was delivered. Providers render the flag deterministically, as an `"interrupted":
true` field in the Codex snapshot and as a fixed suffix in Ollama's chat messages. Model text
can never forge the flag. A turn with nothing confirmed writes no assistant row. A replay adds
only newly confirmed text, as its own row. The undelivered remainder stays in the
resumable-replay machinery. The Codex prompt's statement that earlier assistant messages
"represent only speech confirmed delivered"
([prompt](../../src/hermes_realtime/providers/codex_app_server.py#L2801)) is therefore true, and
the live context, the Hermes record, and what the user experienced match by construction.

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
- **What the MVP resumes.** Not tasks: a restart stops the background runs a crashed process
  left running and says so. On v0.21.0 it does not resume voice history either. These limits
  belong in #159's statement of the MVP, not hidden.
- **Memory egress.** Before a voice tail reaches a run, the profile must use built-in memory
  only, because a run's memory sync passes its messages to any configured external provider
  ([sync](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L4556)).
- **Forgetting.** "Forget this conversation" deletes every run session in the binding record,
  which hold objectives rather than voice text, and the voice session's whole compression chain.
  The record keeps a run only while it may be running, so how forget names the sessions of
  finished runs is open until step 4.
  Hermes deletes only the named session and leaves compression successors in place. So forget
  resolves the latest continuation and walks `parent_session_id` back to the bound session,
  deleting each link. If the walk cannot reach the bound session, forget reports itself
  incomplete. Curated memory is a
  separate, explicit operation, and complete file-level erasure is not promised until the delete
  path is qualified
  ([delete semantics](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L14027-L14116)).
  A run dispatched with a key also has its status persisted by Hermes, including its output,
  error, and any approval request still pending
  ([persistence](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L138-L152)).
  Under the default policy for bearer clients, a terminal record is pruned once it is more than
  24 hours past its last status update. A record left non-terminal by a gateway crash is not
  pruned until Hermes rehydrates it as interrupted
  ([pruning](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_run_idempotency.py#L237-L294)).
  No API route deletes it, so forget does not remove it.
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

1. **Idempotent dispatch.** Add the per-dispatch key and the single lost-acknowledgment resend.
   Settle Hermes's `interrupted` status, which a replay after a Hermes restart reports.
2. **Heard-first live context.** Correct the Codex prompt to match, then add the bounded voice
   tail to runs.
3. **Binding record.** Crash recovery stops what a crashed process may have left running; tasks
   are not resumed. The record holds pending entries (key, mint time, request) and admitted run
   IDs only.
4. **Upstream route.** Propose it. After it lands and a pinned target carries it, append voice
   messages and resume from the voice session. The profile and the voice session join the
   binding record.

## Required qualification before acceptance

Each step must prove its guarantees against a qualified exact target:

- a retried or replayed run never starts twice;
- a crash between dispatch and acknowledgment leaves a pending entry that restart resolves;
- a restarted host stops every recorded run, and reports what it stopped and what stayed
  unknown;
- live context and the record contain only transport-confirmed text and markers;
- appends are idempotent, ordered, and never block the voice;
- the record never holds a reply without its question;
- resume admits only text rows from the bound session;
- forgetting removes every bound run session and the voice session's whole compression chain.

Each guard follows the repository evidence rules: one bounded, content-free refusal marker, and
one mutation per guard shown to fail alone.

`scripts/qualify_hermes_continuity.py` qualifies the replay, crash and restart guarantees of
steps 1–3 against the baseline, unattended and without credentials. It uses real processes and
the pinned Hermes. Each scenario crashes a host, restarts it on the same record, and checks two
independent witnesses:

- Hermes's own durable admission store, which must show one admission per dispatch, ended by
  the restart's stop, or by Hermes's own restart;
- a stand-in model that never finishes, which must see the work start once and nothing left
  running.

The in-process resend and the unknown outcomes rely on the same Hermes replay contract. They are
proven by unit tests, not against Hermes.
