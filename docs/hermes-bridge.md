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
is given. The gate then:

1. refuses unless this interpreter's `hermes_cli` is the installer's checkout,
   `<home>/hermes-agent`, and its `hermes-realtime` distribution is installed in that
   checkout's `venv`;
2. reads `API_SERVER_KEY` and the companion endpoint from `<home>/.env` with the full host's own
   loaders, and accepts only a literal loopback API URL;
3. performs a real bridge hello with the companion and requires both `voice_archive` and
   `voice_review`. Only a companion the running gateway discovered, loaded and started answers,
   so the gate never discovers plugins in its own process;
4. dispatches, rejects an actionable approval and exactly cancels real work through
   `HermesApiTaskSession`, the class the host uses;
5. after the session closes, reads every run Hermes admitted for it back through
   `GET /v1/runs/{run_id}` and requires each to be terminal, so none is left running.

It prints one JSON line naming the Hermes version, commit and whether that pair is the
qualification baseline (v0.21.0 at `29112bef`, a reference rather than a version ceiling); the
candidate's version and whether it is a wheel or an editable install; the negotiated
capabilities; each behavior's terminal status; and the cleanup counts. A refusal prints one
`[real-hermes-gate]` line with the stage and the failure's type only. With
`HERMES_REALTIME_LIVEKIT_LOCAL=1`, it also starts and closes the actual full-host configuration
used by Desktop against the same gateway: Codex natural-work tools, the `natural_v1` conversation
profile, knowledge overlap/recovery, Moonshine v2 Medium, and Kokoro. Spoken-turn media remains
covered by the LiveKit acceptance tests.

The gate refuses a checkout with tracked changes, because no commit describes that code. Paths
that differ only in case are exempt on a case-insensitive checkout only while their one file on disk matches one of their committed versions:
`29112bef` tracks such paths, and Windows can hold only one of each.

## Protocol 0.3: the hello, voice archive and review

The hello is exactly
`{"token", "participant_id", "protocol_version": "0.3", "capabilities": [...]}`. The server
answers `{"ok": true, "protocol_version": "0.3", "capabilities": [...]}` with the requested
capabilities it offers, or `{"ok": false}`. A hello of another version, with an unknown or
repeated capability, or with any other key, is refused. The voice capabilities are
`voice_archive` and `voice_review`, offered by the ready companion. A welcome that
negotiates review also carries its profile's bounded `review_interval`. Realtime
sends no voice event on a connection whose welcome did not list it, and the server closes a
connection that sends one anyway. A voice connection carries voice events only. A refused
hello leaves one `[hermes-bridge-hello]` marker with its rejection category (`shape`,
`token`, `participant`, `version` or `capability`), never the token or the participant.

Versions are mixed on purpose: the hello and every voice event name `protocol_version`
`"0.3"` explicitly (a voice event without it is refused), while the work events
(`work.*`, `control.*`) keep `"0.1"`. The two event families parse separately, and
neither parser accepts the other's events.

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
