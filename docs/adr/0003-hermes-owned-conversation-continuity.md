# ADR 0003: Realtime converses; Hermes keeps the turns

Status: Proposed for review (revised; owner-agreed direction). Work items:
[#77](https://github.com/sushiHex/hermes-realtime/issues/77),
[#80](https://github.com/sushiHex/hermes-realtime/issues/80), and
[#159](https://github.com/sushiHex/hermes-realtime/issues/159).

Amended 2026-09-25 with the owner's approval. Voice archiving and learning no longer wait for
the upstream route
([NousResearch/hermes-agent#121045](https://github.com/NousResearch/hermes-agent/issues/121045)).
A Hermes-side companion, hosted in this repository's Hermes plugin, archives voice rows into
Hermes storage and runs Hermes's own memory and skills review over them (mechanism 4). The
amendment lifts this record's earlier ban on interim writers into Hermes storage and its
rejection of an in-process plugin writer. The sections below are edited in place.

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
turn are three distinct operations. Realtime never writes Hermes's files. Where Hermes lacks a
capability, the owner-approved exception is the voice companion: code that runs inside the
Hermes process and writes only through Hermes's own storage layer.

The #80 audit of v0.21.0, as corrected by the companion's source review, established these
facts:

- **No route writes without a model run; a plugin can.** No HTTP route adds a message without
  running a model. The session routes create, read, patch, delete, fork, and chat
  ([routes](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L2237-L2246)),
  and the server reports no external memory-write API. But the gateway discovers plugins in its
  own process
  ([discovery](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/run.py#L13683)),
  and Hermes's session store appends rows with no model call
  ([batch append](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11672)).
  The plugin bridge lacked only the active parent turn that delegation needs. A model-free
  writer needs none.
- **Hermes reviews only live turns, but its review is reusable.** Hermes learns through its
  background memory and skill review. That review starts only at the end of a live agent turn
  ([spawn](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_finalizer.py#L806-L817)),
  and its memory cadence counts the user turns in the loaded history
  ([counter](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_context.py#L782-L790)).
  Nothing in Hermes reviews a stored session. The review's admission and thread target are
  module functions that take a parent agent and any message snapshot
  ([admission](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L75-L94),
  [target](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1761-L1769)).
  The skills trigger counts the tool iterations of the current turn
  ([skills trigger](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/turn_finalizer.py#L785-L791)),
  so a model run that only records voice rows never fires it.
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
- **Session reads are capped.** The messages route returns at most 500 rows per read
  ([cap](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server.py#L4537)).
- **Writers need not take the turn lease.** Agent turns serialize on a per-conversation turn
  lease, but a transcript write checks it only when its caller names a holder
  ([guard](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11423)).
  The gateway transcript writer names none
  ([writer](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/session.py#L3898-L3925)).
  Manual compression bypasses turn admission and commits under a compression lock instead
  ([CLI](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/cli.py#L14051-L14058),
  [gateway](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/slash_commands.py#L4739-L4747),
  [commit](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/conversation_compression.py#L4622-L4631),
  [lock check](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12300-L12315)).
  `replace_messages` is unleased by default
  ([replace](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12102-L12108)),
  and a metadata setter overwrites a row's display metadata whole
  ([setter](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11766-L11798)).
  The index on `platform_message_id` is not unique
  ([index](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state_schema.py#L1001-L1005)),
  and plugins have no hook on transcript writes
  ([hooks](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_cli/plugins.py#L163)).

## Decision

Four mechanisms, and nothing else.

### 1. One Hermes session archives the voice conversation

Each voice conversation is bound to one Hermes session, the voice session, in the one selected
profile. The companion creates it with an ID it chooses and `source` set to
`hermes-realtime-voice`
([create](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L6520)).
Only the companion writes it. Every voice message is archived as its own row, with its role and
a realtime-assigned identity:

- a user row when its transcript is final;
- an assistant row when its delivery settles.

An assistant row may belong to an ordinary reply, an announcement with no user turn, or a
resumed replay. A `task:` command becomes a user row plus its delivered announcement.

The assistant text is exactly the transport-confirmed prefix. Confirmation comes from the
delivery ledger. It proves the audio reached the room's verifier, not that a person heard it,
and no wording in this record claims more. Interruption is data, not text: a deterministic
`interrupted` flag in the row's display metadata, never model-authored. Generated but
undelivered text is never written.

Rows leave realtime in order, through the outbox (see the durable voice tail below). One batch
is in flight per conversation, and the archive cursor advances only over contiguous,
acknowledged coverage. When overflow loses rows, archiving resumes at the next user row, and
the lost interval is recorded as an explicit gap. The archive therefore never holds a reply
without the question it answers.

Realtime never reads the voice session. Resume restores the local voice tail. Older recall is
delegated: a run can search sessions, and session search does not hide the voice source
([hidden sources](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/tools/session_search_tool.py#L46)).

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
whole duration, and the companion holds the voice session's lease while it owns the archive.
With separate sessions neither waits on the other. The tail gives the work the conversation's
context. Learning from voice does not depend on it: the companion reviews the archive.

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

The record is versioned, bounded by the active-run capacity, and, with the durable voice tail
below, one of only two persisted realtime files. It is owned by one host at a time: a host
holds an exclusive lock on it from before it
settles the record until after its final write, and a second host fails to start rather than
stop runs it does not own. #77 item 4 admits exactly this kind of bounded transient reference.
In milestone M1 the record also names the conversation's profile, its voice session, and its
generation, the fencing number that forget retires.

- **Write-ahead.** A dispatch is recorded before its first request is sent. The entry becomes
  the run ID as soon as the run is claimed. It is removed when the run is authoritatively
  terminal, when a stop of it completes, or when Hermes truthfully rejects the dispatch. A
  dispatch whose outcome is unknown stays recorded, and holds a capacity slot, until a restart
  settles it. Every write replaces the record atomically: a temporary file in the same
  directory, flushed and synced, then renamed over the record. A kill between the two can leave
  that temporary behind, so a start deletes this record's temporaries once it holds the lock.
  Deletion is best effort: one a scanner holds open is left for a later start, never a reason
  to fail this one.
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
the live context, the Hermes archive, and what the user experienced match by construction. A
turn that completes normally closes its last row, so only speech still in progress can be
cut off.

A final user input becomes exactly one user row: an ordinary input through the turn it starts,
and an explicit command, which starts no turn, before the command acts. Live speech settles
first, so a user row never lands inside an assistant row. A command is invalid only when the
store's pure text check refuses it: over the per-item bound, blank, not encodable as UTF-8, or
carrying a private run token. So a command row must fit `max_item_chars`, prefix included. Any
other failure while recording it propagates exactly as it would for an ordinary turn.

### Durable voice tail and outbox

Voice history survives a host restart, crash or clean shutdown, through one realtime file, the
voice tail. Every start restores it, with no age limit. Tasks are still stopped, never resumed.

The live context store stays the only owner of the conversation and does no I/O. After every
change to its rows, and whenever its `prior_work` flag flips, it hands a callback its durable
view: exactly the live rows, except that a row whose speech is still in progress is flagged
`interrupted`, because a crash then would cut it off, plus `prior_work`, true once any task was
active or ended. That view is the only thing persisted. One writer task off the voice path owns
every write, the final one included: it coalesces changes, writes the latest atomically, and
retries any failed write with bounded backoff. Closing never cuts a backoff short; it waits a
bounded time for the tail to be clean, and on timeout keeps the tail locked so a later close
waits again. The file is versioned JSON, bounded by the store's message and per-item limits,
and locked by one host at a time like the binding record; a start deletes its crashed-write
temporaries on the same best-effort terms.

A start restores the tail before preflight and so before the first turn. Restore is allowed
only into an empty store and validates every row against the store's own bounds, UTF-8 text
included. A malformed, oversized, or unknown-version tail, or one the store refuses, starts a
fresh conversation with one content-free marker; the next write replaces the file. A refusal
never blocks start; a tail that cannot be read fails it, so an unread tail is never
overwritten.

Its contract:

- the recorded rows are a prefix of the heard sequence;
- a recorded assistant text is a prefix of its delivered text;
- a final `interrupted` flag may be conservative, so the Codex prompt says the user "may not
  have heard the rest".

Tail version 2 (milestone M1) adds the archive outbox to the same file, under the same lock and
atomic-write rules:

- **Identities.** Each closed row gets an immutable identity, `(generation, seq)`, and a
  timestamp. A user row closes when its transcript is final; an assistant row closes when its
  delivery settles. A row still in progress is never eligible.
- **Durable before eligible.** A row becomes eligible for the archive only after a completed
  tail write contains it.
- **Frozen batches.** A batch is written to the tail before it is sent. While its outcome is
  unknown it is never rebatched, rewritten, or discarded, and it is resent unchanged.
- **Bounded, with explicit gaps.** On overflow the outbox discards only rows it never sent,
  up to the next user row, and records the discarded interval durably. The gap travels as data
  on that next user row, `gap_before: [first, last]`, so it is archived, hashed, and
  acknowledged with the row that follows it. Archived rows and recorded gaps stay distinct.
- **Cursor.** The cursor names the last `seq` covered by acknowledged rows and the gaps they
  carry. A trailing gap with no following row is recorded in the tail only; it needs no
  acknowledgment, because nothing after it awaits archiving.

An archive batch has three outcomes. *Present*: the companion acknowledged it after its commit
and an exact identity and content check, and the cursor advances. *Unknown*: no acknowledgment
arrived, and the frozen batch is resent unchanged. *Absent* is never computed: realtime never
reads the voice session, so no resend is keyed on a negative read.

Work from before a restart is ended history, never active. A restored tail with `prior_work`,
or a restart settlement that stopped work or left it unknown, marks it ended, once, so the
prompt's `work_state` reads `inactive_with_history`. After a settlement, one fixed
announcement with counts only is spoken once voice input is ready and nothing else holds the
floor. It becomes a context row like any other speech.

The tail is plaintext user data under the host state directory, outside evidence capture and
purge. Forget clears the store and the file follows; from M3, forget also clears and fences the
outbox and ignores stale acknowledgments. No forget operation exists yet (#77); until one does,
delete the file while the host is stopped. The conversation-only launcher keeps no tail and
archives nothing.

### 4. A Hermes-side companion archives and reviews

The companion lives in this repository's Hermes plugin, the `hermes_agent.plugins` entry point,
installed into the user's Hermes. It runs inside the Hermes gateway process. Realtime reaches it
over the existing authenticated loopback bridge ([bridge](../hermes-bridge.md)), which gains
archive, review, and forget events. The companion does two things:

- a model-free, role-preserving archive of delivery-confirmed rows into the voice session;
- confined native memory and skills review over archived ranges.

It never runs a foreground turn, never compacts, never ends a voice session it owns, and holds
no transcript text of its own.

#### Hosting

Registration builds the companion's runtime; it is not readiness. An owned start binds exactly
one profile and its database, recovers the companion's fences, acquires each owned lease,
verifies each archive, and only then starts the bridge and announces readiness. One companion
serves one profile; multiplexing is refused. Unload stops reviews, releases leases, and closes
the bridge. The bridge hello gains a capability list and `protocol_version` becomes `"0.2"`.
Realtime sends no voice event until the companion advertises the capability.

#### The archive operation

`voice_archive{conversation_id, generation, seq_from, seq_through, rows[{seq, role, text,
interrupted, ts, gap_before}]}` is answered by
`voice_archive_ack{generation, seq_from, seq_through}`.

Each row's `platform_message_id` is `voice:<conversation>:<generation>:<seq>`, and its display
metadata is `{"voice": {"gen", "seq", "interrupted", "gap_before"}}`. Before writing, the
companion validates the batch exactly. Its rows and their gaps must partition
`[seq_from, seq_through]` with no hole and no overlap: each row's `seq` follows the previous
row's, or, when the row carries `gap_before`, follows that gap's end. A gap may be carried only
by a user row, and its interval must begin right after the previous row. It also validates roles
(`user` or `assistant` only), timestamps, and the flag. An identity already present with equal
content is a duplicate. One with different content is a conflict, refused with no mutation. A
malformed batch is refused whole.

One operation, `archive_voice_rows`, runs in a single `BEGIN IMMEDIATE` transaction through
Hermes's own writer
([write](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L5675-L5744)).
Inside it, the operation:

1. runs the transcript guard with the companion's lease holder, which refuses a lost lease
   ([guard](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11430-L11436));
2. enforces the row cap and verifies the integrity chain (below);
3. compares every identity already present, by full content;
4. inserts only the missing rows through `_insert_message_rows`
   ([insert](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12008));
5. updates `message_count` as the public batch append does
   ([count](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11747-L11757)).

Any failure rolls the transaction back. The operation does not call the public batch append,
which opens its own transaction. A batch commits whole or not at all, so a resend of a batch
the cursor already covers inserts nothing: the operation checks that every row is present and
equal, and the batch is acknowledged again.

#### Integrity chain

The turn lease fences participating agent turns only. Unleased writes, manual compaction, and
replacement can still change the archive. So the companion keeps durable integrity evidence and
verifies the whole archive against it.

- **Rows.** Every row of the voice session, active, inactive, and compacted, in ascending
  `messages.id` order, the order Hermes itself reads
  ([order](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12513-L12514),
  [every row](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12533-L12535)).
  Each row is serialized canonically, with sorted keys, typed fields, and a normalized
  timestamp: `platform_message_id`, role, content, timestamp, voice metadata, `active`, and
  `compacted`. Numeric row IDs are left out, so the chain after a batch is known before insert.
- **Header.** `parent_session_id`, `source`, whether `ended_at` is null, and `end_reason`.
- **Chain.** `C_0 = SHA-256(header)` and `C_i = SHA-256(C_{i-1} || SHA-256(row_i))`. The
  fingerprint is `(count, C_n)`.
- **Same pass.** The expected identity sequence, gaps included, is checked in the same order,
  and no session may name the voice session as its parent.

A companion SQLite file in the plugin's data directory, the plugin store, holds one row per
conversation: the `committed` fingerprint, a `pending` fingerprint or null, the quarantine and
tombstone fences, the cursor, the lease holder, and the review ledger. It holds fences and
progress only, never transcript text. `state.db` stays the only source of truth for archive
content. No transaction ever spans the two files. Committing a batch the cursor does not yet
cover is write-ahead:

1. In a plugin-store transaction, assert that `pending` is null and the conversation is neither
   quarantined nor tombstoned, then write `pending`, the fingerprint after the batch.
2. In the archive operation, the recomputed chain must equal `committed` before insertion.
3. In a plugin-store transaction, set `committed` to `pending`, `pending` to null, and advance
   the cursor. Only then is the batch acknowledged.

Recovery runs at startup, before any admission. When `pending` is set, the chain is recomputed
once. Equal to `committed`, the write never landed, and `pending` is cleared. Equal to
`pending`, the write already applied, and it is promoted. Anything else quarantines. A startup
without `pending` verifies against `committed`.

#### Verification and quarantine

The chain is verified at three points: inside every archive transaction before an
acknowledgment, at review admission, and at restart readiness. Each verification reads every row,
because the threat is a change to older rows; the row cap bounds its cost.

A mismatch, a missing voice session, or an ambiguous recovery quarantines the conversation. The
durable quarantine fence is written before any definitive refusal is returned, and the refusal
fails if the fence cannot persist. Quarantine fences companion mutations, cursor advances, and
review admissions until an explicit reconciliation, which this record does not automate.
Re-acquiring a lease never clears quarantine and never recreates a missing archive. The voice
itself continues: realtime keeps the outbox and reports the quarantine once, with a
content-free marker.

#### The lease

The companion holds the voice session's turn lease for as long as it owns the archive. The
holder is `pid=<pid>:voice=<conversation>:boot=<uuid4>`, fresh for each process instance. The
TTL is 300 seconds, refreshed every 100 seconds
([refresh](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L8538-L8559)).
A refresh that returns false or raises fences all work. Hermes never reclaims a same-process
holder as dead, but TTL expiry still frees it, and on Windows without `psutil` expiry is the only
reclaim
([reclaim](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L209-L226)).
The transcript guard renews a matching lease only once it has expired
([renewal](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11437-L11451)).
The lease key is the lineage root
([key](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L8421-L8424)).

Every agent turn on a durable session acquires the lease, waiting up to 30 minutes
([turn lease](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L8853-L8857)).
So a CLI, gateway, TUI, or API resume of the voice session waits and then fails. That is all the
lease prevents. The review fork, whose persistence is disabled, takes no lease
([exemption](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L8815)).

Leases do not survive a gateway restart. A start recovers the fences, acquires the lease under a
new boot nonce, verifies the archive, and only then announces readiness. The lease is released
only as the last step of forget, or at unload.

#### Review

`voice_review{conversation_id, generation, seq_from, seq_through, memory, skills}` asks for one
review of an archived range.

- **Parent.** A factory bound to the companion's profile builds a parent `AIAgent` for the voice
  session, with verified runtime and credential resolution, `skip_memory=True`, and the `memory`
  and `skills` toolsets. Built-in memory loads, and no external memory provider exists
  ([built-in](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/agent_init.py#L1862-L1896),
  [external](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/agent_init.py#L1900-L1903)).
  There is one authoritative parent per conversation and generation. It never runs a foreground
  turn. The companion owns its cleanup and reroutes its callbacks.
- **Admission.** The companion calls `prepare_background_review_run` and
  `spawn_background_review_thread` itself and starts a thread it owns. It does not call
  `_spawn_background_review`, which returns nothing whether it spawned a review, found review
  disabled, or found one busy, and discards its thread
  ([wrapper](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L1907-L1977)).
  Under the conversation lock, one `BEGIN IMMEDIATE` transaction verifies the archive, captures
  an immutable snapshot, and takes the run token before the database lock is released. The
  thread starts only after that transaction succeeds; a failed admission returns the token.
  *Accepted* means a run token and a started, owned thread.
- **Windows.** A snapshot holds at most 24 messages, within byte and token limits. Above 24
  messages, a review routed to another model replays a digest that cuts older user text to 300
  characters and assistant text to 200
  ([digest](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L417-L458)).
  Windows stay under that bound. A larger range splits deterministically; an oversized window is
  split or refused.
- **One prompt.** Both flags select Hermes's single combined review prompt
  ([prompt](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1792-L1797)).
- **Trigger.** Realtime policy, not Hermes cadence. A review fires every N acknowledged user
  rows, where N is the profile's memory nudge interval
  ([interval](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/agent_init.py#L1885)),
  and once at conversation end, after the final archive settles. A review range is always
  inside acknowledged coverage. An empty conversation starts none. A busy refusal keeps the
  pending coverage, the closing range included. The profile's
  `auxiliary.background_review.enabled` switch produces an explicit refusal and is never
  bypassed. Hermes reads that switch fail-open on a config error
  ([switch](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L266-L288));
  the companion refuses instead.
- **Acknowledgment.** `voice_review_ack` carries the review identity, the exact range, and
  `accepted` or `refused` with a reason. The outcomes finished, failed, cancelled, and unknown
  are reported separately and kept in the review ledger. None of them means "learned".

#### Execution boundary

The review runs in Hermes's own review fork, under its limits:

- a tool whitelist of `memory`, `skills_list`, `skill_view`, `skill_manage`, `read_file`, and
  `search_files`
  ([whitelist](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1539-L1607));
- dangerous commands auto-denied
  ([approvals](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1426-L1433));
- no session persistence
  ([isolation](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1262-L1263))
  and no external memory provider
  ([fork](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1234)).

`auxiliary.background_review.extra_tools` widens that whitelist
([extra tools](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1575-L1586)).
The companion requires it empty in the configuration each spawn uses, not only at startup, and
requires the dispatched whitelist to equal the qualified set. A whitelist is not a sandbox, so
provider, plugin, and hook behaviour inside the fork is qualified too. Native review callbacks
([callback](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L1716-L1723))
are suppressed or routed explicitly, and no archive or review event, summary, or failure reaches
speech or task dispatch. No full-toolset turn exists anywhere in this design.

#### Forget

`voice_forget{conversation_id, generation}` retires a generation. One lock per conversation
serializes archive commits, review admission, and tombstoning, and each of them also reads the
tombstone and quarantine inside its own plugin-store transaction. A lock on `state.db` never
stands in for one on the plugin store. The order is fixed:

1. Persist the tombstone. Later archive and review events for that generation are refused.
2. Set the in-memory generation fence.
3. Cancel the review run. Before admission this fences it
   ([fence](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L48-L63));
   after admission it interrupts the fork
   ([interrupt](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L115-L149)).
4. Join the owned thread. Quiescence means the join returned and the thread is not alive.
   Hermes's own cancellation waits a bounded time and then lets the review continue
   ([bounded wait](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/background_review.py#L152-L187)),
   so a cancel never proves quiescence. On a join timeout, forget reports itself incomplete,
   defers deletion, and keeps the lease. Memory or skill writes may land while a review unwinds.
5. Resolve the latest continuation
   ([resolution](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L12742)),
   capture every `parent_session_id` link back to the bound session, then delete each link with
   `delete_session(expected_delete_ids=...)`, persisting progress after each so a partial
   deletion resumes. Hermes orphans compression children rather than deleting them
   ([orphaning](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L14068-L14073)).
   A walk that cannot reach the bound session reports forget incomplete.
6. Release the lease, last.

Realtime clears and fences the outbox and ignores stale acknowledgments. The generation recheck
applies to companion work; nothing rechecks inside Hermes's review thread. Forget does not remove
memory or skills already learned.

#### Compatibility

`integration/hermes_compat.py` is the only module that names Hermes internals:

- storage: `_execute_write`, `_check_transcript_write_guards`, `_insert_message_rows`,
  `_TRANSCRIPT_WRITE_PATIENCE_S`, `create_session`, and `get_messages`;
- leases: `try_acquire_session_turn_lease`, `refresh_session_turn_lease`, and
  `release_session_turn_lease`;
- sessions: `resolve_resume_session_id` and `delete_session(expected_delete_ids)`;
- review: `prepare_background_review_run`, `spawn_background_review_thread`,
  `finish_background_review_run`, and `_BackgroundReviewRun.cancel`;
- profile: `_profile_runtime_scope` and `set_hermes_home_override`;
- shapes: the review whitelist and the `messages` and `sessions` columns the chain reads.

Each Hermes pin runs a qualification test over that surface: the imports, the signature
parameters relied on, and behaviour. The behaviour checks cover whitelist equality, disabled
persistence in the review fork, a `platform_message_id` round-trip, and `delete_session`
orphaning children. They also require that `archive_voice_rows` and `append_messages_batch`
write byte-identical rows for the same fixture, and that the chain over both is equal. The
public append also repairs transcripts
([repair](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L11731-L11739)),
so the equivalence test is what proves the repair a no-op for voice rows. A pin that fails the
test is red.

#### Row cap

A voice session holds at most 4,096 rows. The number is provisional and fixed in M0. The cap
is enforced before every insertion and at bootstrap and recovery. Its check reads at most
cap + 1 rows, counts inactive rows, and never truncates: an insertion over the cap is refused.
The cap bounds every verification.

#### What this gives up, and its limits

- **The pure public-contract story.** The companion depends on private Hermes names in some of
  Hermes's most-changed files. The compatibility test makes that cost visible at every pin.
- **`/v1/runs` journal runs.** None are used.
- **Hermes's native cadence for voice.** An explicit trigger replaces it.
- **A learning receipt.** A review reports finished, never learned.
- **Rejecting unleased writers.** v0.21.0 has no transcript-write hook, so an unleased foreign
  write is detected and quarantined, not rejected. Rejection needs a Hermes change.
- **Preventing foreign compaction.** Manual compaction and replacement bypass the lease. They
  are detected and quarantined; zero foreign compactions is not guaranteed.
- **Memory in the realtime foreground.** Learned memory lands in the profile's built-in memory
  files. Reading it back into the realtime foreground is milestone M4, not this amendment.

### The upstream route is optional

The proposed route
([NousResearch/hermes-agent#121045](https://github.com/NousResearch/hermes-agent/issues/121045)),
`POST /api/sessions/{id}/messages`, appends externally produced turns without a model call,
idempotently per `(session, client_id)`, under the session's turn lease, and runs the learning
accounting a live turn runs. It remains welcome, but nothing in this design waits for it.

If a pinned Hermes carries it, the archive can write through a public, idempotent contract
instead of the private archive operation, and voice can reach Hermes's native memory cadence.
The route alone does not fence unleased writers or manual compaction, so the integrity chain
and quarantine stay until Hermes offers that fence. Any move to the route is a separate,
qualified change.

## Degraded mode

A backend outage never silences the voice.

- **Archiving is best-effort.** The bounded outbox holds rows until they are acknowledged. On
  overflow, realtime drops rows it never sent, records the gap, announces it once, and resumes
  at the next user row.
- **A companion that is absent, not ready, or quarantined** stops archiving and review only.
  The outbox keeps its rows, and the voice continues.
- **Task truth stays live.** It always comes from the live backend, never from the record, and
  restored text cannot grant dispatch, approval, or cancellation authority.

## Consequences

- **Voice latency is unchanged.** It is today's foreground path; archiving runs off it.
- **Learning.** Hermes's own memory and skills review runs over voice conversations through
  the companion, on realtime's trigger rather than Hermes's cadence. Completion reports
  finished, never learned. The realtime foreground reads learned memory only from M4.
- **What the MVP resumes.** Not tasks: a restart stops the background runs a crashed process
  left running and says so. It resumes the bounded voice tail locally, while Hermes holds the
  archived conversation. These limits belong in #159's statement of the MVP, not hidden.
- **Memory egress.** Voice text reaches built-in memory only; no external memory provider
  receives it. The companion builds its review parent with `skip_memory=True`, so no external
  provider exists on the review path. A run carries a voice tail only on a profile whose memory
  is built-in, because a run's memory sync passes its messages to any configured external
  provider
  ([sync](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/run_agent.py#L4556)).
- **Forgetting.** "Forget this conversation" clears the live store, which rewrites the voice
  tail empty. From M3 it also runs the companion's forget, which deletes the voice session's
  whole compression chain in the order above. It deletes every run session in the binding
  record, which hold objectives rather than voice text. The record keeps a run only while it
  may be running, so how forget names the sessions of finished runs remains open. Curated
  memory and skills are a separate, explicit operation, and complete file-level erasure is not
  promised until the delete path is qualified
  ([delete semantics](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L14027-L14116)).
  A run dispatched with a key also has its status persisted by Hermes, including its output,
  error, and any approval request still pending
  ([persistence](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_runs.py#L138-L152)).
  Under the default policy for bearer clients, a terminal record is pruned once it is more than
  24 hours past its last status update. A record left non-terminal by a gateway crash is not
  pruned until Hermes rehydrates it as interrupted
  ([pruning](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/gateway/platforms/api_server_run_idempotency.py#L237-L294)).
  No API route deletes it, so forget does not remove it.
- **Retention.** Hermes session auto-pruning stays disabled for the MVP. Pruning selects only
  ended sessions
  ([filter](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/hermes_state.py#L14371)),
  and the companion never ends a voice session it owns.

## Rejected alternatives

- **Hermes as the voice foreground** (session chat per utterance). It contradicts the product's
  shape. It also cannot hold the invariants on v0.21.0:
  - tools are per profile only;
  - an interrupted reply is persisted as an ordinary assistant message
    ([interruption](https://github.com/NousResearch/hermes-agent/blob/29112bef099274229cadff79cdff7bf7b99c4b77/agent/conversation_loop.py#L4581-L4595));
  - the stream's terminal event is unreliable;
  - no client-defined tools exist to keep delegation with realtime.
- **Runs in the voice session.** The lease would make voice writes wait behind background work.
- **`/v1/runs` journal runs** (a model run per voice batch, only to record it). Each batch
  would cost a model call. The run persists its own input and reply, not the voice rows with
  their roles, because supplied history is not re-persisted. The skills trigger counts tool
  iterations in the current turn, so a journal turn never fires skills review, and aligning
  the memory cadence would mean gaming it.
- **A companion-owned unique index or table in Hermes's `state.db`.** It changes Hermes's
  schema. The archive operation and the integrity chain need no schema change.
- **Silently following a changed continuation.** A rotation or in-place compaction of the voice
  session quarantines it instead.

Withdrawn by the 2026-09-25 amendment:

- **An in-process plugin writer** was rejected as unreachable in the current host and as
  breaking #77 item 4. The plugin loads in the gateway process, and a model-free writer needs
  no active parent turn. The owner approved the exception to #77 item 4. It is now mechanism 4.
- **An interim writer into Hermes storage** was banned until the upstream route shipped. The
  owner lifted the ban. The companion is the only such writer, and realtime still writes no
  Hermes file.

## Implementation sequence

1. **Idempotent dispatch.** Add the per-dispatch key and the single lost-acknowledgment resend.
   Settle Hermes's `interrupted` status, which a replay after a Hermes restart reports.
2. **Heard-first live context.** Correct the Codex prompt to match, then add the bounded voice
   tail to runs.
3. **Binding record.** Crash recovery stops what a crashed process may have left running; tasks
   are not resumed. The record holds pending entries (key, mint time, request) and admitted run
   IDs only.
4. **Durable voice tail.** Mirror the live store's durable view into one bounded, locked file,
   restore it before the first turn, and report restart-stopped work as ended history with one
   fixed announcement.
5. **Voice companion,** in milestones. Each proves its criteria below against the pinned Hermes.
   - **M0, spike.** The compatibility module, the archive operation, the lease, and the
     integrity chain with quarantine; the row cap is fixed. Criteria 2, 3, 5, and 12. If
     exclusive ownership cannot be established without a Hermes schema change, the design
     fails M0 and this record is revisited.
   - **M1, archive.** Companion hosting and capability negotiation, tail version 2 with the
     outbox, and the archive events; the binding record gains the profile, voice session, and
     generation. Criteria 1 and 4.
   - **M2, review.** The review parent, admission, windows, trigger, and execution boundary.
     Criteria 6, 7, 8, 9, and 11.
   - **M3, forget.** Criterion 10.
   - **M4, memory readback.** A model-free bridge read returns fresh, bounded built-in memory
     for the foreground's context snapshot, under the same profile binding. M4 qualifies the
     realtime foreground applying what Hermes learned.

The upstream route is not on this path. Proposing it stays worthwhile; adopting it is a
separate change.

## Required qualification before acceptance

Each step must prove its guarantees against a qualified exact target:

- a retried or replayed run never starts twice;
- a crash between dispatch and acknowledgment leaves a pending entry that restart resolves;
- a restarted host stops every recorded run, and reports what it stopped and what stayed
  unknown;
- live context and the archive contain only transport-confirmed text and deterministic flags;
- a crash keeps every row the voice tail recorded, and a malformed tail starts a fresh
  conversation;
- archiving never blocks the voice, and the archive never holds a reply without its question.

The companion's milestones must meet these criteria:

1. **Archive fidelity.** Archived rows equal the frozen eligible rows in identity, role, text,
   interruption, and timestamp. Every discarded interval appears exactly once as the
   `gap_before` of the next archived user row, or remains a trailing gap in the tail; a batch
   whose rows and gaps do not exactly partition its range is refused.
2. **Exactly once, and ownership.** Sequential and concurrent resends leave exactly one row per
   identity. A conflicting payload is refused with no mutation. A leased foreign agent turn is
   refused with 0 rows. An unleased foreign row, a manual `/compress`, a `replace_messages` that
   keeps the count and every identity but changes one content, and a metadata overwrite that
   drops the interruption flag each cause 0 companion mutations and a durable quarantine that
   survives a restart.
3. **Crash safety.** Kills between lookup and insert, during the transaction, between the
   `state.db` commit and the plugin-store commit, after the acknowledgment but before the
   cursor write, on lease takeover, during tombstoning, and during deletion recovery cause 0
   duplicates and 0 false advances. The boot nonce, the refresh at a third of the TTL, a failed
   or raising refresh fencing all work, and the startup order (fences, lease, verify, ready)
   are each proven.
4. **No negative reads.** 0 reads of the messages route, and 0 resends keyed on a negative read.
5. **Foreign compaction.** Forced compaction, rotation, and retention attempts are exercised.
   Any change they make to the archive is detected by chain verification and quarantined before
   any refusal; zero physical foreign compactions is not claimed. Detached in-memory compaction inside the review fork is allowed, and the fork
   has no session database.
6. **Confinement.** Serial and parallel forced tool requests, in both routing modes, produce 0
   executions outside the whitelist. Whitelist equality and an empty `extra_tools` are checked
   at each spawn.
7. **Review coverage.** Conversations with 0, 1, 8, 9, 10, 11, 19, and 20 user rows, a review
   already busy at close, a disconnect, and a restart: an empty conversation starts 0 reviews,
   and every non-empty closing range gets an admission or a retained failure.
8. **Attribution.** 0 attribution errors on an adversarial corpus on both the main and the
   routed model. Production windows never produce the digest; an oversized snapshot is split
   or refused.
9. **Corrections applied.** Every predeclared eligible correction is persisted to memory or a
   skill and applied in a fresh Hermes task, with a declared model and repeat count.
10. **Forget.** Late events, work already admitted, a hung cancellation, a restart, and a
    partial deletion never resurrect history or produce a false "complete". Forget stays
    incomplete until the join returns and the thread is not alive.
11. **Nothing reaches speech.** 0 archive or review events, native summaries, or failures reach
    speech or task dispatch.
12. **Compatibility.** The compatibility test is green at `29112bef` and fails when one
    enumerated name or the whitelist set is mutated. Every relied-on check has negative
    evidence, including wrong-profile credentials or store, a lifecycle reload, and missing
    admission evidence. The equivalence and fingerprint fixtures pass.

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
