# Local Hermes bridge

`hermes-realtime` communicates with the loaded Hermes plugin through an authenticated, loopback-only NDJSON socket. The realtime worker never consumes Hermes's internal completion queue.

## Hermes backends

Hermes v0.20 needs no realtime-specific core patch and loads the plugin through its public
`PluginManager`. The plugin currently exposes no Hermes-turn command surface from which to retain
the lifecycle API's active-parent authority. Local-bridge dispatch therefore fails closed on
v0.20; realtime asynchronous work uses the authenticated full-host `/v1/runs` adapter instead.
The plugin does not retain opaque lifecycle capabilities, start supervisor threads, or claim
cancellation authority it cannot exercise.

This does not remove a shipping realtime feature. Both current executable hosts are independent
of the plugin bridge: `hermes-realtime-host` uses `HermesApiTaskSession`, while
`hermes-realtime-local` remains intentionally conversation-only. The full host retains concurrent
background work, proactive terminal updates, exact task cancellation, approval decisions, natural
work tools, browser projection/replay, and bounded shutdown on v0.20. The bridge implementation is
kept exported and tested as the v0.19 backend and protocol reference rather than deleted.

The v0.20 lifecycle surface also does not restore completion correlation by itself. Its
`subagent_stop` hook does not carry the exact `delegation_id` required by the bridge completion
router. A future in-process v0.20 bridge therefore requires both safe out-of-turn dispatch/cancel
authority and exact terminal correlation; adding only one half must remain fail-closed.

The v0.19 compatibility backend requires the companion patch's three additive behaviors:

1. `PluginContext.dispatch_reserved_delegation(args, delegation_id)` submits one asynchronous
   top-level `delegate_task` under the exact supplied handle and fails closed;
2. asynchronous `subagent_stop` hooks include that exact `delegation_id`;
3. `PluginContext.interrupt_delegation(delegation_id)` signals one exact running delegation.

Reserved dispatch requires an active top-level parent agent in the hosting Hermes process.
Starting the bridge without that context is permitted, but dispatch requests are rejected rather
than silently running synchronously or detached from a Hermes session.

Synchronous and capacity-fallback delegations report `delegation_id=None`. A completion hook proves child execution stopped; it does not prove that another Hermes delivery adapter consumed its completion record.

## Plugin host

After Hermes loads the `hermes-realtime` plugin, construct a bridge around the registered runtime:

```python
import secrets

from hermes_realtime.hermes_plugin import create_local_bridge
from hermes_realtime.integration import SessionBindings

bindings = SessionBindings()
bindings.bind("participant_001", "session_001")

# Provision this out of band to the worker. Do not log it or expose it to a browser.
token = secrets.token_urlsafe(32)
bridge = create_local_bridge(bindings=bindings, token=token)
await bridge.start()
```

The server rejects non-loopback bind addresses, malformed participant identifiers, invalid tokens, and NDJSON lines larger than 64 KiB. It counts unauthenticated sockets toward the connection limit, requires authentication within the configured timeout, and bounds shutdown even if a connection or completion task stalls. Protocol free-text fields are limited to 8,192 characters. Hermes terminal summaries and reasons are each truncated to 4,096 characters before retention and delivery so that two maximally JSON-escaped fields still fit one wire line.

The default limits are `authentication_timeout=5.0`, `max_connections=32`, and `shutdown_timeout=5.0`. Override them through `LocalHermesBridgeServer` only when the local deployment requires different bounds. Completion-hook ingress is deduplicated by exact run ID and bounded by `max_pending_completions` before cross-thread loop scheduling. Call `await bridge.close()` during plugin shutdown. Closing is terminal for that bridge instance; construct a new bridge and service for a new lifecycle rather than restarting an instance whose routes and completion subscription were discarded.

Plugin registration succeeds when Hermes exposes either the v0.20 lifecycle contract or all three
v0.19 exact-delegation APIs. Under v0.20, registration provides packaging compatibility while
bridge dispatch rejects with the full-host route. Registration fails when neither complete host
surface is available. Do not apply the v0.19 companion patch to v0.20.

The upgrade qualification gate is `scripts/real_hermes_api_gate.py`. It observes the operator's
installed runtime from the outside, the way the full host meets it, and simulates nothing: run it
with the install's own interpreter while the installed `hermes gateway` is running.

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\python.exe" scripts\real_hermes_api_gate.py
```

`--hermes-home` defaults to `HERMES_HOME`, else the installer's default home, and
`--hermes-api-url` to the full host's default, `http://127.0.0.1:8642`; pass the same URL the host
is given. Every observation is bound to the process under test:

1. **Identity.** This interpreter's `hermes_cli` must be exactly the installer's checkout,
   `<home>/hermes-agent`, and its own attestation must name every field: the Hermes version,
   the checkout's detached commit, and the hermes-realtime wheel imported from exactly that
   checkout's `venv`, identified by the SHA-256 of the wheel's installed `RECORD` (which hashes
   every file it put down). An editable install or any other source is refused: only a built
   wheel names exact content. The gate refuses a checkout with tracked changes, because no
   commit describes that code.
2. **Endpoint.** `API_SERVER_KEY` and the companion endpoint come from `<home>/.env` through the
   full host's own loaders; only a literal loopback API URL is accepted.
3. **The serving process.** The gateway names its process through the authenticated
   `GET /health/detailed`. A real bridge hello with the companion must offer `voice_archive`,
   `voice_review` and `runtime_attestation`, and the attestation must name that same process
   (otherwise another process, such as a CLI or cron run, owns the companion) and exactly the
   identity inspected in step 1, `RECORD` hash included (otherwise the gateway is still running
   what it loaded before an update, a reinstall or a rebuilt wheel of the same version: restart
   it). The gate never discovers plugins in its own process.
4. **Behaviors.** Through `HermesApiTaskSession`, the class the host uses, it dispatches work
   that must complete, rejects an actionable approval and requires the run to complete without
   running the command, and stops a run that must end interrupted, not completed.
5. **Cleanup.** After the session closes, every run Hermes admitted for it must read back
   through `GET /v1/runs/{run_id}` as 200 with that run's terminal status; `stopping` is not
   terminal. A refusal still reads the runs back and counts those left running.
6. **End.** `/health/detailed` must name the same process at the end: a gateway that restarted
   during the gate fails it, even though its runs read back as `interrupted`.

It prints exactly one line on stdout and nothing on stderr. A pass prints the JSON record: the
Hermes version, commit and whether that pair is the qualification baseline (v0.21.0 at
`29112bef`, a reference rather than a version ceiling); the candidate's version and install
kind; the negotiated capabilities; each behavior's terminal status; and the cleanup counts. A
refusal, including a bad argument, a missing default home or an interpreter that cannot import
the gate, prints one `[real-hermes-gate]` marker with the stage, a bounded category, the
failure's type and, once runs were admitted, `left_running`. Markers other components print
while the gate runs are folded into that line under `components` (at most 8, each at most 512
characters), and library log records at WARNING and above are counted under
`log: {logger: {level: count}}`, never printed. Process IDs, paths, commits and record hashes
are compared, never printed beyond the record.

The refusal categories:

- `not_install`: this interpreter's Hermes is not the checkout at `<home>/hermes-agent`.
- `unnamed`: a field of the install cannot be named: no detached commit, no single installed
  wheel `RECORD`, `hermes_realtime` imported from anywhere but the install's wheel, or an
  unreadable Hermes version.
- `no_companion`: the `.env` names no companion endpoint.
- `health`: `/health/detailed` did not name the gateway's process.
- `hello_refused`: a wrong token, or a gateway still running a plugin without runtime
  attestation: restart the gateway.
- `capability`: the companion does not offer every capability the gate asks for.
- `foreign_companion`: another process, not the gateway, owns the companion.
- `restart_gateway`: the gateway is still running what it loaded before an update, a
  reinstall or a rebuilt wheel: restart it.
- `behavior`: a dispatch, approval or cancellation did not end exactly as required.
- `left_running`: a run the gate created did not read back as terminal.
- `gateway_restarted`: the gateway's process changed during the gate.
- `arguments`: the command line was not understood.
- `error`: anything else; the failure's type names it.

What it does not prove:

- Processes a run spawned (a terminal command's children) are not checked; Hermes reaps those
  on stop.
- Edits to the installed wheel's files in `site-packages` after it was installed: `RECORD` is
  the installer's list of what it put down, and is not recomputed from the files.
- Hermes edits that were uncommitted when the gateway loaded and reverted before the gate ran:
  the commit is read from `HEAD` only, and the tracked-changes check sees the tree as it is now.

With `HERMES_REALTIME_LIVEKIT_LOCAL=1`, it also starts and closes the actual full-host
configuration used by Desktop against the same gateway: Codex natural-work tools, the
`natural_v1` conversation profile, knowledge overlap/recovery, Moonshine v2 Medium, and Kokoro.
Spoken-turn media remains covered by the LiveKit acceptance tests.

Paths that differ only in case are exempt from the tracked-changes refusal on a
case-insensitive checkout only while their one file on disk matches one of their committed
versions: `29112bef` tracks such paths, and Windows can hold only one of each.

## Protocol 0.3: the hello, voice archive, review and memory

The hello is exactly
`{"token", "participant_id", "protocol_version": "0.3", "capabilities": [...]}`. The server
answers `{"ok": true, "protocol_version": "0.3", "capabilities": [...]}` with the requested
capabilities it offers, or `{"ok": false}`. A hello of another version, with an unknown or
repeated capability, or with any other key, is refused. The voice capabilities are
`voice_archive`, `voice_review` and `voice_memory`, offered by the ready companion.
Memory is an additive 0.3 capability, not a new wire version; peers without it keep
dispatch, archive and review on their existing connections. A welcome that
negotiates review also carries its profile's bounded `review_interval`. Realtime
sends no voice event on a connection whose welcome did not list it, and the server closes a
connection that sends one anyway. A voice connection carries voice events only. A refused
hello leaves one `[hermes-bridge-hello]` marker with its rejection category (`shape`,
`token`, `participant`, `version` or `capability`), never the token or the participant.

`runtime_attestation` is offered by a companion the plugin built, which captured what its
process loaded once, when Hermes loaded the plugin. A welcome that negotiates it carries
`runtime{pid, hermes_version, hermes_commit, realtime_version, realtime_install,
realtime_record}`, and a welcome that does not must not. `hermes_commit` is the checkout's
detached commit read from `.git/HEAD`, or `unknown`. `realtime_install` is `wheel` when
`hermes_realtime` was imported from exactly the install `venv`'s
`site-packages/hermes_realtime/__init__.py`, else `elsewhere`; `realtime_record` is then the
SHA-256 of that wheel's single installed `RECORD`, else `unknown`. A process whose install
cannot be read attests every field as `unknown` or `elsewhere`. The model is strict: exact
types, `pid` in `[1, 2^53 - 1]`, versions of 1 to 64 characters of `[0-9A-Za-z.+_-]`, lowercase
hex hashes, and no other field; the client refuses a welcome that breaks it. The capability is
additive, so existing clients and servers are unchanged; a plugin that predates it refuses the
hello, since the capability is unknown to it.

Versions are mixed on purpose: the hello and every voice event name `protocol_version`
`"0.3"` explicitly (a voice event without it is refused), while the work events
(`work.*`, `control.*`) keep `"0.1"`. The two event families parse separately, and
neither parser accepts the other's events.

Memory uses a dedicated subscription connection. One
`voice_memory{conversation_id, generation}` request receives ordered
`voice_memory_snapshot{conversation_id, generation, revision, memory, user, truncated}`
events. Each block is capped at 4,096 UTF-8 bytes. A
`voice_memory_refused{conversation_id, generation, category}` event clears the foreground
snapshot for binding, integrity or unavailable states. Transient `pending` and memory-read
`capacity` refusals retain the last snapshot while the connection remains bound; disconnect
still clears it. Revisions order snapshots within a connection, not across companion restarts.
The foreground keeps a per-connection revision high-watermark. No refusal lowers that
mark; a later snapshot must have a revision at least as high, and an equal revision is
accepted.
Reconnect delays grow to a bounded maximum after connection or stream failures, including
failures after a successful handshake. The delay resets only when the connection delivers
its first valid snapshot.
The companion reads its bound profile through Hermes's native memory parser, sanitizer
and renderer at subscription open and after a review finishes. Reads use bounded file
input, Hermes's configured character limits and the rendered byte cap. Oversize input
keeps the longest suffix of whole sanitized entries that fits both limits, with the
truncation marker first. Newest entries win, including repeated entries. A bounded tail
read discards any partial first entry before Hermes parses the remaining complete entries.

The foreground never waits for this read on a turn. It uses the latest received snapshot
as context data explicitly labelled untrusted. Memory does not add tool permissions. A new
conversation can read before its first archived row without creating an archive session.
An existing binding must be verified and ready. Quarantine, generation changes, binding/integrity refusals
and connection loss clear memory. Missing or unsupported capability leaves voice running
without memory; no periodic refresh or second durable memory store is introduced.

**Authority qualification:** ADR criterion 17 compares admitted calls with and without
memory under identical forced model behavior. It also checks that refresh alone starts no
foreground turn or tool call, that the model cannot grant approval, and that memory appears
only in its labelled prompt data section. These are structural checks on what memory adds,
not a test of model obedience.

**Existing boundary's limit:** valid advertised model tool calls can dispatch or cancel on
a neutral-user turn. The forced stand-in probe records one dispatch and one cancellation;
M4 leaves natural routing unchanged. This does not prove a production model will follow an
adversarial memory entry. A deterministic boundary is a separate decision in
[#205](https://github.com/sushiHex/hermes-realtime/issues/205); no option is implemented here.

Memory in Codex and Ollama is labelled reference data. Ollama carries its labelled JSON
in a system message, leaving conversation messages unchanged; placement and labelling do
not establish model prompt-injection resistance. Ollama's `num_ctx` is not set by this
adapter, so the configured model/server context window may truncate input despite the
memory byte cap. Adapter-boundary evidence includes the SHA-256 and UTF-8 byte length of
canonical memory JSON, never the memory contents.

**Freshness limits:** an open conversation refreshes after this companion's reviews report
`finished`, not after failed or cancelled reviews. Memory written by other Hermes sessions,
or written before a failed/cancelled review, waits until the next conversation opens.
Hermes may replace blocked entries with a `[BLOCKED: ... use memory(action=remove)]`
placeholder. That is retained as untrusted data; the foreground has no memory-removal tool
and cannot perform that suggested action.

`voice_archive{conversation_id, generation, seq_from, seq_through, rows[{seq, role, text,
interrupted, ts, gap_before}]}` is answered on the same connection by
`voice_archive_ack{conversation_id, generation, seq_from, seq_through}` after the commit, or
by `voice_archive_refused{..., category}`. The category is a bounded string (1 to 32
characters of `[a-z_]`), not a closed list, so one a newer companion adds still parses.
The models are strict: exact types, `ts` an exact
finite non-negative float, identities in `[0, 2^53 - 1]`, 1 to 256 rows, text of 1 to 65,536
characters. Semantic rules (rows and gaps partition the range, only a user row carries a
gap, only an assistant row is interrupted) belong to the companion, which refuses a batch
that breaks them as `invalid` or `partition`. When the outcome is unknown, the companion
closes the connection without answering; realtime then resends the same frozen batch.

Refusal categories split into two closed sets whose union is the companion's set. A
transient refusal (`not_ready`, `lease_held`, `lease_lost`, `conversations`, `pending`,
`stale`, `fenced`, `incompatible`, `durability`, `unbound`, `bound`) never means the archive
or the batch is wrong: realtime retries the same frozen batch with bounded backoff, with one
marker per episode. Every other category is an integrity refusal and fences that
conversation's archiving until realtime restarts, and drops its connection; so does a
category neither set lists.

Realtime's tail holds a closed row to the companion's own row rules (one shared, pure
validator) before the row can be archived: a row the companion would refuse, such as one
with a NUL, becomes a gap instead of entering a frozen batch. A version-1 tail migrates on
the first write; its restored rows get identities then, so their `ts` is the time of that
restart, not the time their speech settled, which a version-1 tail never recorded.

## Voice companion hosting

Registration builds the companion when the environment the gateway runs in names both
`HERMES_REALTIME_COMPANION_PORT` (a loopback port, 1 to 65535) and
`HERMES_REALTIME_COMPANION_TOKEN` (24 to 512 characters). A partial or malformed endpoint, or
a plugin context without `on_unload` and a `state.data_dir`, is refused with one
`[voice-companion]` marker; dispatch still registers.

Hermes discovers plugins in every process (the gateway, the CLI, cron), so the owned start
first takes an exclusive OS lock beside the plugin store and holds it until unload. A
process that finds it held stands down: it touches no lease, row or store, and prints one
`{"refusal":"held"}` marker. The owner then runs on its own event-loop thread. It binds
the profile's `state.db`, as Hermes resolves it, and the plugin store `voice-companion.db`
in the plugin's data directory, checks compatibility and durability, and only then starts
the bridge on the configured port. Start opens no conversation: each opens on contact
whenever it is not ready (M0's order: fences, compatibility and durability, lease,
verification), so an unknown outcome or a lost lease recovers in the same process, while a
quarantined or tombstoned one stays refused. At most 16 are live at once; the store binds
at most 4,096 and refuses a new one past that (`conversations`), which M3's forget will
prune. Unload closes the bridge, releases every lease, closes the store and `state.db`, and
releases the lock; a close that fails keeps the companion owned for a later retry. A
second owned start in one process is refused as `multiplexed`.

The full host (`hermes-realtime-host`) reads the same two variables from `--hermes-env-file`,
or from its environment. With both present and a voice tail enabled, it drains the tail's
outbox to the companion. With neither, it archives nothing.

Known limit: the token sits in the Hermes env file the gateway also reads. Any process of
the same user can read that file anyway, so keeping the token out of it would not change
who can reach the bridge; it is deferred.

## Voice review

`voice_review{conversation_id, generation, seq_from, seq_through, memory, skills, closing}`
requests Hermes's combined memory and skills review. Both review flags must be exact
`true` booleans. The companion answers with `voice_review_ack`, naming the same range,
closing flag and a review identity, only after it owns a started review thread. A
`voice_review_refused` retains the range for retry or investigation. Admission is not
completion: the companion's durable ledger records `finished`, `failed`, `cancelled`
or `unknown`. Completion also emits a content-free `[voice-review]` outcome marker.
None of those is a receipt that something was learned.

The companion verifies the archive chain and captures the snapshot while taking the
Hermes admission token under the same database write transaction. A mismatch quarantines
the conversation. Review text comes exclusively from archived rows; realtime sends
identities and flags, not a second copy of the conversation. Each request is bounded to
24 archived messages and the companion also bounds snapshot size. An oversized snapshot
is refused rather than passed to Hermes's routed-history digest.

The parent belongs to the archive's profile, never runs a foreground turn, and uses
`skip_memory=True` with the built-in memory and skills toolsets. Each spawn checks the
profile's review switch, empty `extra_tools`, and the qualified tool whitelist. The native
parent and fork have no session database or session persistence, so closing the parent cannot
end the archive. The companion owns cancellation and joining; a timeout
does not prove that the thread stopped. Native summaries and failures are not forwarded
to speech or task dispatch.

Realtime schedules review from acknowledged user-row counts using the negotiated profile
interval, and retains a final boundary after 300 seconds without conversation activity or
at shutdown. Shutdown gives the live archive and review senders one five-second budget
to acknowledge the final range; an expired budget retains the checkpoint and emits a
content-free refusal marker. Active foreground work and unsettled speech defer idle closure.
Each idle boundary ends a conversation period within the same Hermes voice session; resuming speech
does not erase earlier closing ranges or create a second history. A final review has a
distinct identity even when a periodic review just covered the same ending range.
Changing the profile's nudge interval requires restarting the companion; a mismatch with
the negotiated interval refuses review. The enabled switch and empty extra-tools policy
are checked again for every spawn. A cached parent also refuses changed profile configuration,
model routing or credential authority; restart the companion to bind the new configuration.

Tail version 3 retains bounded acknowledged-row identities, frozen review requests and
ordered closing checkpoints. Pending coverage survives busy replies, disconnection and
restart. The archive remains the history authority; review bookkeeping contains only
identities, counts and outcomes. Migrating a version-2 tail preserves its outbox and begins
counting new acknowledgments at its existing cursor; it does not claim to have reviewed
older archived history. Older bridge versions are refused, so the two sides must be
upgraded together.

The installed qualification is `uv run python scripts/qualify_voice_review.py`, against
the pinned Hermes with a stand-in model and synthetic fixtures. Its attribution and
correction checks establish the integration's behavior with that declared model; they
do not establish the reasoning quality of a production model. Memory readback into the
realtime foreground remains M4, and forgetting remains M3.

## Worker

```python
from hermes_realtime.integration import (
    LocalHermesBridgeClient,
    RealtimeHermesSession,
)

client = await LocalHermesBridgeClient.connect(
    host="127.0.0.1",
    port=bridge.port,
    token=token,
    participant_id="participant_001",
)

async with client, RealtimeHermesSession(
    client=client,
    session_id="session_001",
) as hermes:
    acknowledgment = await hermes.dispatch(request)
    terminal_update = await hermes.next_update()
```

`RealtimeHermesSession` owns the single inbound reader, correlates dispatch and cancellation acknowledgments, enforces the bound session and strictly increasing incoming event sequences, and pushes terminal updates without polling. Its proactive terminal buffer is bounded (`max_pending_updates=256` by default); overflow terminates the session explicitly instead of dropping evidence or growing memory without limit.

Retained acknowledgments and terminal evidence keep their semantic `event_id` but receive a fresh transport `sequence` whenever replaying the original sequence would regress on a connection. Terminal and dispatch-retry retention are coupled: after terminal evidence expires, a matching old retry receives an explicit rejected acknowledgment and must use a new `task_id`; the bridge never installs a live route for expired terminal work.

A cancellation acknowledgment means Hermes accepted an interruption signal for the listed exact run IDs. Only a later `work.completed` event with `status="interrupted"` proves terminal interruption.

## Full-host work routing

The full host keeps one `ConversationWorkControlSurface` between foreground requests and
Hermes task authority. The deterministic `task:` and `cancel task:` commands use that same
surface, so explicit fallback and natural routing share validation, acknowledgment, projection,
replay, and cancellation behavior.

Natural work routing is experimental and disabled by default. Enable it only with the exact
Codex app-server provider:

```powershell
hermes-realtime-host `
  --inference-provider codex `
  --natural-work-tools `
  --allow-unsandboxed-hermes-tasks
```

`--no-natural-work-tools`, or omitting both natural-work flags, leaves only the explicit-prefix
path enabled. `--natural-work-tools` with Ollama is rejected during composition; Ollama and
disabled Codex sessions remain tool-less. The unsandboxed-task acknowledgment is required in
either mode.

Source-backed foreground knowledge is a separate authority plane. It is disabled unless the host
operator passes `--enable-public-search`, and each active browser binding must separately consent.
When both gates are open, Codex hosts route source-sensitive final transcripts through bounded
public-RSS evidence from Bing Search RSS and, for outcome-shaped queries, Google News RSS.
`--knowledge-speculation` overlaps eligible Moonshine partials and
`--knowledge-recovery` permits one deadline-sharing weak/empty recovery; both are default off and
neither grants bridge, task, cancellation, approval, or tool authority. See
[`source-backed-latency.md`](source-backed-latency.md).
Retrieved source text is untrusted evidence, never an instruction, and cannot authorize work.

The host starts and attests the Hermes API session, then runs the Codex readiness turn with no
work tools; the deterministic `search_knowledge` tool may be present. It binds exactly the two work
tools only after inference and TTS preflight both succeed and before the browser can admit a user
turn. Binding requires the exact
`CodexAppServerStreamingInference` and `ConversationWorkControlSurface` types. Provider drift,
preflight failure, an unsupported provider, a second bind, or an incompatible capability fails
closed without exposing work authority to the readiness prompt or browser.

Shutdown first stops browser and foreground-worker admission. It then closes Codex inference,
including outstanding server-request routing, before settling and closing the work surface,
closing the task controller, and finally closing the Hermes API session. A failed
dependency close is retryable and prevents later work-authority dependencies from closing out
of order.
