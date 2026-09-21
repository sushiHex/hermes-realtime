# ADR 0003: Hermes owns persisted foreground conversation

Status: Proposed for review. Work items: [#77](https://github.com/sushiHex/hermes-realtime/issues/77),
[#80](https://github.com/sushiHex/hermes-realtime/issues/80), and
[#159](https://github.com/sushiHex/hermes-realtime/issues/159).

Reviewed upstream source: Hermes Agent
[`v0.21.0` at `29112bef`](https://github.com/NousResearch/hermes-agent/commit/29112bef099274229cadff79cdff7bf7b99c4b77).
That version is the owner's baseline, not a compatibility ceiling. This record
does not claim that the proposed integration is implemented.

## Context

The current realtime context is deliberately bounded and event-loop-local. It
records a model response separately from confirmed speech delivery
([context.py](../../src/hermes_realtime/conversation/context.py) and
[delivery.py](../../src/hermes_realtime/speech/delivery.py)). It is therefore
useful for one live room, but it cannot be the user's durable conversation or
memory after restart.

Hermes v0.21.0 advertises authenticated session chat and streaming session chat,
while it explicitly advertises no memory-write API
([capabilities](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L3341-L3371)).
The session chat handlers reload the named session and run Hermes Agent with that
session identity
([synchronous handler](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4605-L4719),
[streaming handler](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4722-L4942)).
This is the supported path for Hermes to own inference and persistence. Running
a second foreground model and writing its output into Hermes is not a supported
v0.21.0 contract.

A blocking evidence gap is the terminal event. A streamed turn emits a random stream-local
message ID, then emits `assistant.completed` with `partial: false` and
`interrupted: false` after the agent call returns
([event construction](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4790-L4894)).
The agent finalizer separately records its real `interrupted` result and catches
persistence failures as `cleanup_errors`
([finalizer result](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_finalizer.py#L718-L766)).
The stream handler does not bind those facts to its terminal event. It requests a
hard interrupt and drains the agent task when the transport disconnects
([disconnect drain](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4944-L5000)),
but the disconnected client receives no terminal result. The present terminal
event consequently cannot prove which assistant row was persisted, whether the
turn was interrupted, or whether final persistence succeeded.

## Decision

Hermes will own each persisted foreground conversation. Realtime will use one
explicit Hermes profile and one Hermes transcript session for a conversation.
It will call the supported streaming session-chat route once for each final user
utterance and will not also run a separate foreground inference for that turn.
Realtime will never write Hermes state files or its SQLite database directly.

Implementation is gated on a narrow upstream-supported terminal contract, or an
equivalent verified contract in a later Hermes release. A terminal result must
bind all of these facts:

- the selected profile and transcript session;
- stable persisted identities for the admitted user row and completed assistant
  row;
- whether the generation completed, was partial, was interrupted, or failed;
- whether the terminal transcript write succeeded.

The extension should strengthen session chat rather than create a general
external-assistant append API. If session chat cannot meet the measured latency,
tool, and cancellation requirements, an external-turn extension needs a
separate upstream design and review. Its existence must not be inferred from
the current `memory_write_api: false` capability.

### Truth boundaries

A Hermes message row proves only persisted conversation content. It does not
prove that speech synthesis began, audio entered the room, or the user heard it.
Hermes' message response fields contain content and generation metadata, but no
speech-delivery state
([response projection](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4190-L4197)).
Realtime remains authoritative for the delivery stages it observes. After a
restart, a persisted assistant row without a matching delivery receipt is shown
as delivery unknown and is not silently replayed or described as heard.

Task truth also remains outside conversation history. Live run status, events,
approval state, and cancellation decide whether background work exists or may
be controlled. Text in a restored transcript cannot recreate a dispatch,
authorize a tool, or prove that a task is active or complete. Before adoption,
qualification must also show how the session-chat tool set is restricted so a
foreground turn cannot duplicate the realtime task-dispatch path; v0.21.0 source
review did not establish that restriction. The opaque run ID emitted by session
chat is also a foreground-operation identifier; it must not be parsed with, or
admitted into, the existing background-task identifier grammar.

### Identity and minimal realtime state

The binding consists of a schema version, the exact profile route, and the
Hermes transcript session ID. Realtime may retain delivery receipts keyed to the
stable message identity supplied by the required terminal contract. It will not
persist transcript text, audio, summaries, embeddings, or a second searchable
history database.

Profile, transcript session, and memory scope are separate identities. Hermes
rejects unknown profile routes and uses profile-scoped runtime state
([profile routing](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L2097-L2206)).
`X-Hermes-Session-Key` is an optional long-term-memory scope, not the transcript
session ID
([header validation](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L2285-L2335)).
Resume therefore fails closed on a missing or changed profile, an inaccessible
session, or a conflicting binding. It never falls back to another profile or
guesses a memory key.

### Resume and history

Realtime resumes only when no foreground writer is active for the bound session.
The v0.21.0 API exposes no snapshot or writer lease, and a realtime-local lock
cannot exclude another Hermes client. Implementation must either obtain an
upstream serialization contract or fail qualification under a concurrent
external writer; this design does not assume quiescence from its own process.
It reads the newest bounded page, follows further pages only within an explicit
message and character budget, and presents the retained rows oldest first. The
API caps pages at 500 and supports `limit`, `offset`, and latest/oldest order
([messages endpoint](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4498-L4555)).
It supplies no snapshot token or `has_more` marker, so stable offset pagination
under a concurrent writer is not assumed. Long-session latency and compaction
behavior require measurement against the installed Hermes version.

Hermes persists the user message before the first model call for crash recovery
([turn admission](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_context.py#L1567-L1588)).
An interrupted or failed turn can therefore leave a durable user row without an
ordinary final assistant row. Resume preserves that history; it does not invent
or backfill a reply. Hermes compaction is non-destructive: inactive pre-compaction
rows remain stored and searchable
([compaction contract](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12235-L12284)).

### Retention, deletion, and correction

Starting a separate conversation creates a fresh Hermes session binding and does
not delete the previous session. Ending a realtime room does not end or delete
the Hermes conversation.

For the initial MVP, Hermes session auto-pruning must remain disabled. Upstream
documents it as opt-in and limited to ended sessions; its default retention is
90 days when enabled
([session retention](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/website/docs/user-guide/sessions.md#L872-L906)).
This prevents elapsed time alone from deleting an ended conversation in the
supported profile.

"Forget this conversation" cannot yet promise complete erasure through the
v0.21.0 API. Its authenticated delete handler calls
`db.delete_session(session_id)` without a transcript-directory argument
([API handler](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4485-L4496)).
That removes the SQLite session and message rows, including delegate children,
but the database implementation removes `.json`, `.jsonl`, and request-dump
files only when its optional `sessions_dir` argument is supplied
([delete semantics](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L14027-L14116)).
Full conversation erasure therefore needs an upstream API fix and a filesystem
proof before it can be offered. The user must also be told that branch and
compression children are preserved as orphaned sessions. v0.21.0 exposes no
per-message edit/delete route, so a correction is a new user turn unless the
whole conversation is deleted after that gap is closed.

Conversation deletion does not claim to erase long-term memory. Built-in memory
reset deletes the profile's complete `MEMORY.md` and/or `USER.md`, not one
session's contributions
([reset command](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_cli/main.py#L12872-L12914)).
That is a separate, explicit user operation.

### Privacy and memory egress

The initial supported profile uses built-in memory only. Preflight refuses an
external memory provider until its destination, content sent, retention, and
message/session deletion behavior have a reviewed disposition. This is needed
because the Honcho integration sends sanitized user and assistant text to its
configured session
([turn synchronization](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/plugins/memory/honcho/__init__.py#L1419-L1468)),
and an unset base URL can let the SDK select an environment default, including a
hosted endpoint
([client target resolution](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/plugins/memory/honcho/client.py#L1283-L1323)).
Rotating a Honcho session removes only local cache bindings and explicitly keeps
the old remote session for user modeling
([rotation behavior](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/plugins/memory/honcho/session.py#L821-L850)).
The existence of a conclusion-deletion operation does not establish remote
message or session erasure.

Evidence capture and optional search remain outside this decision. Neither may
be enabled as a side effect of conversation continuity.

## Required qualification before implementation acceptance

Qualification must pin the installed Hermes source and prove the terminal
contract on normal completion, interruption before first token, interruption
after partial output, persistence failure, transport disconnect, process restart,
profile mismatch, and long-history pagination. It must also prove that a restored
assistant row is never labeled delivered without a matching realtime receipt,
and that restored history cannot grant live task or approval authority.

New terminal-authority, profile, resume-serialization, delivery, task-authority,
and refusal guards must follow the repository evidence rules. Every refusal path
emits one stable, bounded, content-free JSON marker from a `finally`: counts,
kinds, and categories only, with no transcript text, paths, or identifiers.
Mutation tests must prove, one at a time, that missing or misbound terminal
authority, a wrong profile, a concurrent writer, false delivery or task
authority, a missing refusal marker, and leaking refusal evidence each fail
alone while an adjacent passing case remains green.

Until those proofs pass, #77 remains a design and compatibility dependency for
the integrated MVP in #159 rather than a supported runtime promise.
