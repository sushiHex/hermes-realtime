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

The upgrade qualification gate is `scripts/real_hermes_api_gate.py`. With
`HERMES_REALTIME_LIVEKIT_LOCAL=1`, it composes the actual full-host configuration used by Desktop:
Codex natural-work tools, the `natural_v1` conversation profile, knowledge overlap/recovery,
Moonshine v2 Medium, and Kokoro. It behaviorally exercises authenticated Hermes API work,
actionable approval rejection, exact cancellation, proactive completion, LiveKit composition, and
bounded startup/shutdown. Spoken-turn media remains covered by the LiveKit acceptance tests.

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
