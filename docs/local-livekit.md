# Local LiveKit development

The first media tracer bullet uses a native LiveKit server on the Hermes host. LiveKit Cloud is intentionally deferred until local media, interruption, and reconnect behavior are measured.

## Pinned development server

- LiveKit Server: `v1.13.4`
- Windows asset: `livekit_1.13.4_windows_amd64.zip`
- SHA-256: `a326e025de516e93dfb3719bcd28e5a4ac16f21bcf1ef562499403ca98cc65fe`
- Release: <https://github.com/livekit/livekit/releases/tag/v1.13.4>

Every checkout, worktree and clone shares one copy of the server, installed once per user
outside all of them:

```text
%LOCALAPPDATA%\hermes-realtime\tools\livekit-1.13.4\livekit-server.exe
```

Install it from the root of any checkout:

```sh
uv run python -m scripts.local_livekit install
```

[`scripts/local_livekit.py`](../scripts/local_livekit.py) downloads the pinned archive, checks
its SHA-256 and the SHA-256 of the executable inside it, and writes only the verified
executable. A verified copy is left alone, so the command is safe to repeat. Every local
launcher, test and gate that starts LiveKit resolves it through that module and checks its hash
again before running it. Do not copy the server into a checkout: Windows Firewall remembers its
decision per program path, and each new copy asks again.

The qualification producers and the native release gate take the server as an explicit
`--livekit-executable` with its SHA-256. Locally, give them the shared copy:

```powershell
$livekitExecutable = uv run --frozen --group dev python -m scripts.local_livekit path
if ($LASTEXITCODE -ne 0) { throw 'the shared LiveKit server is missing or unverified' }
$livekitSha256 = (Get-FileHash -LiteralPath $livekitExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
```

## Start the local server

From Git Bash in the repository root:

```sh
uv run python -m scripts.local_livekit serve
```

It runs the shared server with `--dev --bind 127.0.0.1` until `Ctrl-C`. It also overrides
LiveKit's short built-in development secret with a deterministic 38-character loopback-only
test secret:

```text
API key: devkey
API secret: "local-" followed by 32 lowercase x characters
WebSocket URL: ws://127.0.0.1:7880
```

These credentials remain public and predictable. They must never be used for a LAN,
internet-facing, or production deployment.

On Windows, `--bind 127.0.0.1` restricts the HTTP/WebSocket signaling listener, but LiveKit
1.13.4 still opens its RTC listeners beyond loopback: TCP 7881 on the wildcard address and
UDP 7882 on every interface address. Keep Windows Firewall enabled and do not add inbound
allow rules for this development process.

These listeners cannot usefully be held to loopback. A configuration can do it
(`rtc.tcp_port: 0` and `rtc.ips.includes: [127.0.0.1/32]` with
`rtc.enable_loopback_candidate: true` and `rtc.node_ip: 127.0.0.1` leave only 127.0.0.1
sockets), but no client can then connect. libwebrtc, which both the LiveKit Python SDK and
Chrome use, gathers ICE candidates only on non-loopback adapters, and Windows refuses to send
from such an address to 127.0.0.1 (`WSAEADDRNOTAVAIL`). Room joins time out.

So the first run of the shared server shows one Windows Firewall prompt, for its one path.
Answer it with **Cancel**; do not allow access. Cancel adds two inbound **Block** rules (TCP
and UDP) for that path on the Private and Public profiles, and because there is only one path,
that decision holds for every checkout.

Local media still works under those Block rules. With them in place, the native integration
tests and the real-Chrome self-acceptance gate pass, and every ICE candidate pair LiveKit selected
was UDP to port 7882 with both ends on this computer's own adapter addresses (LAN, Tailscale,
WSL and link-local, IPv4 and IPv6). The client and the server are the same host, so that
traffic is not subject to the inbound filter, while the same ports stay blocked to every other
machine. No allow rule is needed.

Windows Firewall applies a Block rule over any Allow rule for the same traffic, so those Block
rules also stop every remote client of this binary, whatever Allow rule is added for its
ports. LAN, Tailscale or phone use of LiveKit needs a server binary at a different path, or the
removal of the shared path's Block rules.

Earlier per-checkout copies each left their own pair of Allow rules. To remove them, run this
in an elevated PowerShell, review what `-WhatIf` reports, then run it again without `-WhatIf`.
It touches only Allow rules for copies under a checkout's `.tools\livekit\`:

```powershell
Get-NetFirewallApplicationFilter |
  Where-Object { $_.Program -like '*\.tools\livekit\livekit-server.exe' } |
  Get-NetFirewallRule |
  Where-Object { $_.Action -eq 'Allow' } |
  Remove-NetFirewallRule -WhatIf
```

Copies extracted elsewhere, such as under the system temporary directory, may have left
rules too. This lists every remaining LiveKit Allow rule outside the shared path, for review
by hand:

```powershell
$shared = Join-Path $env:LOCALAPPDATA 'hermes-realtime\tools\livekit-1.13.4\livekit-server.exe'
Get-NetFirewallApplicationFilter |
  Where-Object { $_.Program -like '*\livekit-server.exe' -and $_.Program -ne $shared } |
  Where-Object { ($_ | Get-NetFirewallRule).Action -eq 'Allow' } |
  Select-Object -ExpandProperty Program -Unique
```

`$env:LOCALAPPDATA` belongs to the account running PowerShell. If you elevate as a different
administrator account, it names that account's directory, not yours, so set `$shared` to
your own shared path.

## Verify readiness

```sh
curl --fail http://127.0.0.1:7880/
```

Expected response:

```text
OK
```

## Run the media tracer bullet

With the server running, execute:

```sh
HERMES_REALTIME_LIVEKIT_LOCAL=1 uv run pytest tests/integration/test_local_livekit.py -v
```

The test creates two uniquely named participants in a unique room, publishes synthetic signed-16-bit PCM through LiveKit's Opus/WebRTC transport, verifies non-silent decoded PCM at the subscriber, proves that production playback begins before scripted inference completes, and verifies that a cancelled physical stream cannot confirm a replacement even when public turn and chunk identifiers are reused. Production speech uses one peer-owned persistent publication; each chunk still receives an exact stream identity and a duration validation before PCM enters that publication. All peers and publications are then closed. Without the opt-in environment variable, the tests are skipped so ordinary CI requires neither a server nor credentials.

## Milestone 5.2c human-test procedure

This milestone exposes the transport and streaming loop, not an end-user microphone or speaker client. Human testing at this gate therefore exercises the real local server and inspects deterministic lifecycle evidence; audible browser/iPhone testing belongs to Milestone 6.

1. Start the server in one Git Bash terminal and leave it running:

   ```sh
   uv run python -m scripts.local_livekit serve
   ```

2. In a second terminal, verify signaling readiness:

   ```sh
   curl --fail http://127.0.0.1:7880/
   ```

   Continue only if this prints `OK`.

3. Run the real transport gate from the repository root:

   ```sh
   HERMES_REALTIME_LIVEKIT_LOCAL=1 env -u PYTHONPATH PYTHONPATH=src \
     uv run --offline pytest tests/integration/test_local_livekit.py -v
   ```

   Expected result: all three tests pass. In particular:

   - `test_streaming_loop_confirms_livekit_pcm_before_inference_completes` proves remote, materially non-silent PCM is confirmed before inference is released to finish.
   - `test_cancelled_chunk_track_cannot_confirm_replacement` reuses the same public turn/chunk IDs and proves the cancelled transport incarnation cannot confirm its replacement.

4. Run the deterministic interruption and accounting surface:

   ```sh
   env -u PYTHONPATH PYTHONPATH=src uv run --offline pytest \
     tests/conversation/test_streaming.py \
     tests/conversation/test_foreground.py \
     tests/conversation/test_context.py \
     tests/speech/test_delivery.py \
     tests/speech/test_types.py \
     tests/livekit -q
   ```

   Any failure, timeout, pending-task warning, stale-audio confirmation, or context admission before exact delivery is a stop condition. Preserve the complete command output when reporting a failure.

5. Stop the native server with `Ctrl-C`. Confirm that no test participant or publication remains in its final log output. Do not expose development mode beyond the local host.

The derived development secret satisfies PyJWT's minimum HMAC key-length guidance. Any
`InsecureKeyLengthWarning` is unexpected and is a stop condition.

Default local ports are:

| Port | Transport | Purpose |
|---:|---|---|
| 7880 | TCP | HTTP and WebSocket signaling |
| 7881 | TCP | WebRTC media fallback |
| 7882 | UDP | WebRTC media |

## Scope

The automated browser tracer proves that the one-use HTTP bootstrap provisions the exact browser identity, the returned restricted token joins the fixed room, and microphone PCM reaches the identity-bound production LiveKit worker. The full host supplies concrete VAD, STT, Codex/Ollama inference, Kokoro/Edge synthesis, and Hermes task/approval adapters. Deterministic tests cover their authority and cleanup boundaries; natural-conversation quality remains a separate physical desktop/iPhone claim.

## Milestone 5.6 automated browser gate

Build and test the browser bundle, then run both real transport gates:

```sh
cd web
npm ci
npm test
npm run build
cd ..

HERMES_REALTIME_LIVEKIT_LOCAL=1 env -u PYTHONPATH PYTHONPATH=src \
  uv run --offline --python 3.11 --isolated --extra local --group dev pytest -q \
  tests/integration/test_local_livekit.py \
  tests/integration/test_browser_livekit.py \
  tests/integration/test_local_launcher.py
```

The browser test is not a mocked token check. It starts `BrowserClientRuntime`, consumes a one-use bootstrap capability over its bounded HTTP server, provisions an exact participant binding through `LiveKitConversationWorker`, connects an SDK room with the returned browser token, publishes a microphone-class audio track, and requires materially non-zero decoded PCM from that exact participant.

Expected result with the pinned development server is five passing tests. Any warning,
failure, timeout, leaked credential, unexpected participant, or pending task is a stop condition.

The packaged wheel must also contain the generated client:

```sh
env -u PYTHONPATH uv build --offline
python -c "import glob,zipfile; p=glob.glob('dist/*.whl')[0]; z=zipfile.ZipFile(p); required={'hermes_realtime/client/static/index.html','hermes_realtime/client/static/assets/app.js','hermes_realtime/client/static/assets/styles.css'}; missing=required-set(z.namelist()); print(sorted(missing)); raise SystemExit(bool(missing))"
```

## Production composition contract

`BrowserClientRuntime` is the composition boundary. The host supplies:

- one `LiveKitConnection` whose API key and secret remain server-side;
- the fixed room name and worker identity;
- the real `LiveKitConversationWorker` wrapping the reconnect-safe conversation runtime;
- the authoritative approval callback;
- optionally, the same `BrowserEventProjection` passed as the observer to the session worker and streaming speech loop.

The LiveKit peer owns one persistent speech source/publication for its generation. Logical chunks
are serialized through exact physical stream identities; normal completion waits for playout,
cancellation clears queued PCM, and stale receipts cannot confirm a replacement. In
`natural_v1`, provider-neutral segmentation additionally rejects rendered chunks above three
seconds. `legacy` retains the historical provider chunk contract.

Foreground inference may continue into a bounded queue while playback is blocked. The production
host admits up to 256 response segments (the loop supports at most 4,096), each normally bounded to
4,096 characters, while synthesis keeps at most one logical speech chunk of look-ahead. An
interruption synchronously revokes active stream authority, cancels consumer/provider work, and
retains only unconfirmed text eligible for explicit replay under current generation authority.
A barge-in lets the interrupted word finish: the peer keeps playing the already-queued PCM to the
next gap between words (a run of at least 25 ms of 5 ms frames 20 dB below the chunk's
90th-percentile frame level), capped at 300 ms, then cuts the queue and appends a 15 ms
raised-cosine fade. Every other stop (Stop speaking, session end, explicit cancellation) cuts at
once with only the fade. The interrupted chunk never confirms delivery, so none of it enters the
transcript or model context, and the next chunk cannot start until the tail has drained. Each stop
prints one `[speech-stop]` JSON line with its mode, outcome (`gap`, `cap`, `end`, `immediate`,
or `unpublished`), tail length, the 10 ms block it cut at (`stop_block`), how far past the
planned stop it landed (`late_ms`), whether the clock had run past the chunk (`clamped`) and
whether it faded. Known limits, unmeasured on real transport so far: the playout position comes
from an open-loop `perf_counter` clock that assumes one native 10 ms block per tick from the
moment the chunk was queued, so any timer drift grows with the chunk; and the user's first
~300 ms of speech overlaps the tail, so agent audio could leak into the user's transcript where
echo cancellation is weak. `scripts/measure_word_boundary.py` reproduces the threshold, click
and latency measurements on generated Kokoro speech.
Under `natural_v1`, local microphone onset is presentation-only: the browser attenuates the exact
current stream through a 750 ms local-quiet debounce and retains the exact renderer claim for at
most 10 seconds while server evidence settles. A rejected claim restores full volume; a matching
server event carrying `(turn, generation, chunk, stream)` commits gain-zero silence and pause.
The browser uses LiveKit Web Audio mixing so attenuation and committed silence do not depend on
iPhone's ineffective `HTMLMediaElement.volume`. Local onset never calls the yield endpoint, mutates
transcripts, or gains task authority. The explicit **Stop speaking** control still submits an
immediate exact `(turn, generation, chunk, stream)` yield claim.

Pass `projection.publish` as the observer when constructing both `ConversationSessionWorker` and `StreamingSpeechLoop`. This projects admitted user transcripts, delivered assistant transcript segments, first foreground token, and first playable audio without exposing internal turn, delegation, or run handles. Task and approval owners publish `task_state`, `completion_received`, `notification_queued`, and actionable `approval_state` events only after their own authoritative transitions; the browser runtime does not duplicate that authority.

Loopback composition derives an `http://127.0.0.1:<port>` origin. Keep the launch capability in the URL fragment; the client removes it from browser history before its first request. `BrowserClientRuntime.start()` returns that launch URL, and `close()` settles the HTTP server and LiveKit worker. The runtime is one-shot: terminal stop or inactivity cleanup closes that worker rather than attempting to reuse it for another bootstrap.

Disconnect is a transient media pause. It retains the authenticated server lease, event polling, and credential refresh so **Connect** can rejoin with the same participant identity and worker generation. Stop is terminal and only changes the UI to stopped after the authoritative stop endpoint succeeds. Authenticated polling/refresh/input/approval activity renews a bounded server lease; an abandoned client is reaped after the configured inactivity timeout (300 seconds by default).

The bootstrap and refresh responses also bind the expected reserved `worker_...` identity. The browser attaches remote playback only when the current room reports an audio track whose publisher exactly matches that identity and whose publication source is microphone; audio from every other participant or source is ignored. Transcript DOM retention is capped at 128 entries and 131,072 characters, while objective/status marker retention is capped at 256 entries and 32,768 characters, evicting the oldest entries first.

## LAN and iPhone constraints

Do **not** expose LiveKit `--dev` or its documented development credentials to the LAN. A phone test requires all of the following:

1. A non-development LiveKit deployment reachable through a trusted `wss://` URL.
2. `BrowserClientRuntime(lan_mode=True, ssl_context=..., canonical_origin="https://<trusted-host>:<port>")`.
3. A certificate chain trusted by the iPhone, with the canonical hostname in its SAN.
4. A narrowly scoped Windows Firewall rule for only the intended TLS listener and LiveKit media ports. The shared development binary's Block rules override any Allow rule for it (see [Pinned development server](#pinned-development-server)), so the deployment runs from a different path.
5. Delivery of the one-use launch URL through a private channel; never logs, shell history, screenshots, issue text, or query parameters.
6. Verification that API key/secret values do not appear in HTML, JavaScript, HTTP bodies, browser storage, or evidence artifacts.

The runtime rejects LAN mode without TLS, non-loopback binds in local mode, wildcard origins, widened browser publication grants, replayed bootstrap capabilities, stale identities/generations, duplicate or out-of-order typed/approval sequences, duplicate JSON fields, transfer encoding, oversized input, and non-allowlisted static paths.

### Remote full-host activation seam

The full host has an explicit remote profile. It does not deploy LiveKit, issue certificates, or change Windows Firewall by itself. It only enables the already-hardened remote runtime boundary after every secure input is supplied.

Remote LiveKit credentials are read from `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`; they are not accepted as command-line arguments. The loopback development values are rejected. The Hermes API remains on its authenticated loopback URL.

```bash
export LIVEKIT_API_KEY='<non-development key>'
export LIVEKIT_API_SECRET='<strong non-development secret>'

env -u PYTHONPATH uv run --frozen --extra local hermes-realtime-host \
  --hermes-env-file C:/path/to/active/hermes/.env \
  --remote \
  --livekit-url wss://<trusted-livekit-host> \
  --browser-host <intended-private-interface> \
  --browser-origin https://<trusted-browser-host>:8765 \
  --tls-cert C:/path/to/trusted/fullchain.pem \
  --tls-key C:/path/to/private/key.pem \
  --inference-provider codex \
  --tts-provider kokoro \
  --kokoro-worker-python C:/path/to/kokoro-cuda/Scripts/python.exe \
  --allow-unsandboxed-hermes-tasks
```

Remote mode requires a credential-free, path-free `wss://` LiveKit origin, an exact canonical HTTPS browser origin matching the listener port, and a server TLS context loaded from both certificate files. Local mode retains its previous loopback-only defaults. Do not use a self-signed probe certificate on the phone; the iPhone must trust the complete chain and the canonical hostname must appear as an exact DNS or IP subject-alternative name. This lean tracer intentionally does not accept wildcard SANs.

For an explicitly persistent, private Tailnet launch broker, add `--tailnet-launch` to
`--remote` and supply the exact permitted Tailscale node identity only through
`HERMES_REALTIME_TAILNET_NODE_STABLE_ID`. Use a placeholder in scripts and documentation;
keep the real value in the existing access-restricted runtime environment. This mode:

- returns the stable canonical HTTPS origin without a URL fragment;
- authorizes each Connect request from the real TCP peer using
  `tailscale whois --json --proto tcp` and exact `Node.StableID` equality;
- resolves the WhoIs CLI only from known system installation locations; nonstandard installs
  require an absolute `HERMES_REALTIME_TAILSCALE_CLI` path in the restricted runtime
  environment, and current-directory/PATH discovery is intentionally rejected;
- does not mint or register the diagnostic bootstrap capability route while Tailnet launch is
  enabled;
- rejects proxy/forwarding headers, bearer credentials, cross-origin requests, missing
  Fetch Metadata, CLI failures, and malformed identity data;
- keeps the HTTPS listener and shared conversation authority alive across explicit stop or
  inactivity while replacing the LiveKit peer, browser identity, and session generation.

It does not configure Tailscale, firewall policy, certificates, DNS, or a phone launcher.
Remote diagnostic mode without `--tailnet-launch` retains the one-use fragment flow.

The same persistent broker implementation has a local front door. Add
`--persistent-loopback-launch` without `--remote` to return the stable
`http://127.0.0.1:<port>/` origin, authorize each Connect from the socket-derived loopback peer,
and keep the listener and conversation authority alive across sequential sessions. It accepts no
proxy-derived client identity and cannot be combined with `--remote` or `--tailnet-launch`.
Use this loopback front door for desktop latency baselines; use the Tailnet front door for remote
path parity. Both use `/api/v1/stable-bootstrap` and `/api/v1/stable-rebind`, rotate browser
identity and session generation, and issue only short-lived LiveKit credentials.
Unlike the one-use fragment and exact-node Tailnet modes, this local front door trusts the entire
machine loopback boundary: any local process can construct the same HTTP request and claim a new
session while no browser session is active. Enable it only on a single-user trusted workstation;
retain the one-use diagnostic flow where local processes are outside the trust boundary.

A stable-launch tab survives a reload. The tab keeps in its own `sessionStorage` only what
`/api/v1/stable-rebind` needs: the session's current participant identity, which rotates on
every rebind, and the request ID of a reload rebind still awaiting its answer. It never stores
the LiveKit token. After a reload, **Connect** presents that identity with `"freshView": true`;
the session rotates the identity as any rebind does and resets everything the page holds, since
the reloaded page kept nothing: the event projection restarts at sequence one with the session's
own description and every task still running and approval still actionable, and the typed-input
and approval counters restart at zero, as the page's do. Only a definitive verdict changes what
the tab remembers. A rotation (2xx) remembers the new identity; when no session is active (409)
the tab forgets the identity and bootstraps; an identity that is not the active one is refused
(403) and forgotten, so a reload never replaces another tab's session. Every other outcome keeps
the identity and its request ID for the next **Connect**, which replays the same request, so a
rotation whose answer was lost is answered again instead of refused: a transient refusal (503),
a network error, or a failure later in connect. A reload during the readiness cue is not
refused. The cue is the binding's own speech, so closing the binding first stops it through
playback's usual hard stop, which releases its chunk. Before, the chunk still in flight made the
audio publisher refuse at once to unbind the old connection. The server answers 409 for exactly
"no session is active"; every other state refusal is 503. A leaving page never reconnects the session it is
leaving: the LiveKit SDK's own page-leave disconnect is turned off, since the page would read it
as a dropped connection and rebind, rotating the identity the reloaded tab is about to present.
Only a stop that succeeded forgets the stored identity. A duplicated tab copies
`sessionStorage` and can therefore rebind the session away from the original, which then needs
**Connect** again. The one-use fragment launch stores nothing and still needs a fresh launch.

## Conversation-only loopback launcher

Milestone 6.1 supplies a runnable local profile for the first desktop conversation slice:

- WebRTC VAD at 48 kHz mono with bounded start/end hysteresis;
- faster-whisper `tiny.en` on CPU with no automatic provider fallback;
- explicit loopback Ollama inference using `hermes-4.3-36b-iq4xs-16k:latest` by default;
- Edge neural TTS decoded by FFmpeg to transport-aligned 48 kHz mono PCM;
- reconnect-safe publication under the reserved `worker_...` identity;
- immutable LiveKit PCM delivery confirmation before assistant transcript admission.

Install and run it only with the local development server bound to loopback:

```bash
uv sync --dev --extra local
uv run --extra local hermes-realtime-local
```

The command preloads inference before exposing a one-use browser URL. Open that URL in one
local browser, grant microphone access, and use typed fallback if needed. Press `Ctrl-C` in the
launcher terminal to settle the browser runtime, worker, verifier, and providers.

This profile is deliberately conversation-only: background task dispatch and approval authority
fail closed, and it does not compose Codex public-RSS retrieval, natural work tools, or the `natural_v1`
profile. Those belong to `hermes-realtime-host`. It keeps no durable voice tail, so its
conversation ends with the process. It is not the full human-test launcher described
below and must not be used for LAN/iPhone exposure. The automated real-boundary gate is:

```bash
HERMES_REALTIME_LIVEKIT_LOCAL=1 \
  uv run --extra local pytest tests/integration/test_local_launcher.py -q
```

That gate keeps launch capabilities and credentials inside the test process; evidence must
record only pass/fail status and non-secret latency values.

## Full loopback host and constrained desktop human gate

`hermes-realtime-host` is the Milestone 6.1b launcher. It keeps the browser, LiveKit,
Hermes API, inference, STT, and TTS surfaces on loopback. Explicit
`start task <objective>` starts background work through the deterministic command router.
The whole final utterance must begin with those exact words (case does not matter); no colon
or other STT punctuation is required. The objective is bounded by the existing work surface.
`cancel task` (optionally ending in `.`, `!` or `?`) cancels only when exactly one acknowledged
task exists at command intake. Its public identity is frozen before any await; if that task
ends meanwhile, cancellation is refused instead of selecting newer work. With no task the
browser says there is no active task; with several it asks the operator to use the intended
task card's **Cancel** control. Pending dispatches are not selected by this spoken command.
`cancel task <public-task-id>` names an exact task. The legacy `task: <objective>` and
`cancel task: <public-task-id>` forms remain supported. Negated, quoted or prefaced commands
are not direct commands, and bare `stop` grants no task cancellation. A dispatch remains
pending until Hermes accepts it; cancellation acknowledgment means stopping was requested,
while a terminal event establishes that the task ended.

If a typed or card command loses its acknowledgment, the browser pauses input because the
command may already have been admitted. It never retries that command automatically. Select
**Connect** on a stable launch to restore a fresh authoritative view and input counters;
a one-use launch requires a fresh launch from the host. Cancellation feedback keeps the
last authoritative task state until a real task update arrives.
Stable recovery retains its bounded public identity and request record in memory even
when browser storage is unavailable. An unresolved input can retire only its own binding.
An ordinary media reconnect keeps the admitted typed-input and approval counters;
only a fresh view or a new session starts those counters again.

With Codex and `--natural-work-tools`, ordinary language may invoke the
same authoritative start/cancel surface; without that flag, ordinary speech and typed text remain
tool-less foreground inference. The launcher requires Hermes's authenticated server-side
`/v1/runs` capabilities before exposing the one-use browser URL; there is no socket, subprocess,
provider, or unauthenticated fallback.

The full host keeps its recent delivery-confirmed voice conversation in a durable voice tail,
`HermesRealtime/state/voice-tail-v1.json` beside the Hermes run record (override with
`--voice-tail`), and restores it before the first turn on every start, after a crash or a
clean shutdown. Work from before the restart is ended history, never resumed. The file is
plaintext user data outside evidence purge. One host holds it at a time. A malformed tail starts
a fresh conversation. To forget the conversation, delete the file while the host is stopped.

Codex may receive a separate bounded public-RSS foreground-evidence path for routed source-sensitive
turns. It is default-off, requires `--enable-public-search`, and remains closed until the active
browser binding grants separate public-search consent. It is independent of background work and
grants no Hermes task authority. Moonshine speculation and one weak/empty recovery attempt are
additional default-off qualification controls. See
[`source-backed-latency.md`](source-backed-latency.md).

> **Security boundary:** Hermes command approvals are transported with an exact public request
> identity, but they are not a sandbox and do not comprehensively gate every tool or filesystem
> mutation. A server-side task may mutate state without producing an approval request. The full
> host therefore requires explicit operator acknowledgement and must be used only with bounded,
> non-destructive objectives during this human gate. It admits only one active Hermes background
> task at a time; foreground conversation remains available while that task runs.

### One-time Hermes API setup

Set these values in the active Hermes `.env`; generate `API_SERVER_KEY` with a cryptographic
random generator and do not paste it into shell history, screenshots, logs, or issue text:

```dotenv
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
API_SERVER_KEY=<at-least-32-random-characters>
```

Restart Hermes from a separate terminal outside the running gateway process, then verify
only the unauthenticated health surface:

```bash
hermes gateway restart
curl --fail http://127.0.0.1:8642/health
```

If the gateway runs in the foreground under `hermes gateway run`, restart it with `Ctrl-C` and
`hermes gateway run` instead: on Windows, `restart` replaces it with a detached gateway and may
offer to install a scheduled task.

Do not continue unless the gateway is healthy and port 8642 is bound only to literal
loopback. The full launcher performs authenticated capability discovery itself.

To qualify that installed runtime, with a built `hermes-realtime` wheel (not an editable install)
installed in the install's `venv` and the companion endpoint (`HERMES_REALTIME_COMPANION_PORT`
and `HERMES_REALTIME_COMPANION_TOKEN`) in the same `.env`, run the gate with the install's own
interpreter from the repository root:

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\python.exe" scripts\real_hermes_api_gate.py
```

It dispatches real work, so the gateway's model is called. Restart the gateway after installing
or updating either Hermes or `hermes-realtime`: the gate refuses a gateway still running what it
loaded before (`restart_gateway`). See [`hermes-bridge.md`](hermes-bridge.md) for what it checks,
what it records and what it does not prove.

### Automated dress rehearsal of the desktop MVP session

`scripts/rehearse_desktop_mvp.py` runs the
[diagnostic session](desktop-mvp-diagnostic.md#the-diagnostic-session) unattended against the
composed stack, so integration findings surface before the operator's session:

```bash
env -u PYTHONPATH uv run --frozen --extra local --extra browser-acceptance \
  python scripts/rehearse_desktop_mvp.py --ollama-model <name from ollama list>
```

**What runs is HEAD.** The record is bound to a commit, as the release gates are. The script
clones `HEAD` into the run directory and builds the wheel there. The host runs from that wheel,
in its own environment made from the locked dependencies. The installed-runtime gate and the
host's child wrapper come from the same clone. The setup line records the commit, whether the
tree it came from was clean, whether the running harness matches `HEAD`, and that the host
imported the wheel.

**The composed stack.** It builds a throwaway Hermes home under the system temporary directory,
laid out the way the upstream installer lays it out:
- the pinned `29112bef` checkout at `home/hermes-agent`, with its locked environment in
  `home/hermes-agent/venv` and the candidate wheel installed there;
- the plugin enabled, and memory turned off, with the runbook's own `hermes` commands;
- for Hermes's model, a stand-in served by the script, so no provider key is involved.

It then starts each of these with a log:
- `hermes gateway run` from that home, with the companion inside it;
- the shared pinned LiveKit server, verified by SHA-256 before it starts;
- the full host with the runbook's flags (`--persistent-loopback-launch`,
  `--allow-unsandboxed-hermes-tasks`, Ollama with the named model, default speech providers),
  with its voice tail and run record inside the run directory;
- a headless system Chrome, driven through the DevTools protocol.

**Containment.** The rehearsal joins a kill-on-close Job Object before it starts anything, and
every process it starts inherits that job. A rehearsal killed outright therefore leaves no
orphan. The job is also what the final count of running processes reads. A command that times
out is ended with its whole process tree.

**Every verdict comes from an independent observation.** A verdict never rests on one of the
script's own actions succeeding.
- *Speech.* The final transcript must contain the spoken words. The microphone is a synthetic
  track that plays Kokoro-synthesized clips, so speech crosses WebRTC, LiveKit, VAD and Moonshine
  for real. "Audible" means the decoded remote audio track carried energy.
- *Stale audio.* After each Connect, the page is watched for audio no one asked for. Only the
  host's readiness cue is excused, and only when it is identified: one burst of at most two
  seconds around the page's own voice-input confirmation. Any other energy, or any assistant row,
  is stale.
- *A delegated task.* The result Hermes produced must appear in a background-result row.
- *Cancel and restart.* A cancel must hang up the stand-in's stream. A restart needs work running
  through the crash, and that work stopped after settlement.
- *Context.* Context carried over a reconnect or a restart means two things.
  - **The host kept the fact.** The user's own step-3 row stating it is among the rows the host
    keeps (its voice tail). After a restart, these are the rows the new host read, captured
    before it started, and the host must report restoring exactly that many.
  - **The question's prompt carries it.** The host child counts, at the Ollama adapter
    boundary, the user rows of that exact prompt that state the fact (`[rehearsal-prompt]`).

  Losing the fact is `context_lost`. A kept fact missing from the prompt is
  `context_not_in_prompt`. Both are host findings. The model's echo of the fact never counts.
  Nothing restates the fact, so the bounded context window shows up as a `different`
  verdict.

  Whether the model then names the bird is model quality, so it is a note, not a finding:
  - **The answer** is recorded as `context_*_answered`.
  - **The recall rate** is measured with `--recall-trials N`. The host child resends that exact
    prompt N times after its reply and prints only the count of replies naming the fact
    (`[rehearsal-recall]`).
  - **The prompt size** is recorded too: `[ollama-prompt]` gives the size the host sent beside
    Ollama's `prompt_eval_count` ([heard context window](heard-context-window.md)).
- *Listeners.* The Hermes API, the companion and LiveKit signaling must listen on loopback only.
  Any other address fails the step, and nothing runs behind that listener.
- *Reload inputs.* A step after the deletion spends a typed turn and an approval decision, reloads
  the page, and makes one of each again. Each is judged by its status as the browser's network
  layer saw it, and by its effect: a reply, or Hermes's own report to the model that the user
  denied the command. It runs last so its turns cannot push the step-3 fact out of the tail.
- *Deletion.* The phrase must be found before deletion, in the voice tail and in Hermes's
  database. Afterwards it must be gone from four places:
  - the page;
  - the tail;
  - every column of every table and every full-text index of `state.db`;
  - the `sessions/` and `memories/` files.

  The check runs again after the follow-up turn and at cleanup, and every assistant row of the
  follow-up reply is read. A missing database is a failure. A phrase left only in built-in
  memory is the documented no-unlearning limit, so it reads as `different`.
- *Tracebacks.* A traceback outside a named allowlist of known upstream noise makes its step
  `different`. The allowlist holds LiveKit's FFI handle disposal at exit and Hermes's Unix-socket
  watchdog on Windows.

**Records.** Each step prints one `[desktop-mvp-rehearsal] {...}` line in the record sheet's
categories: `outcome`, `timings`, markers and `notes`. Markers are kept by name, from the
components this session runs. They are re-rendered from their values, which may only be counts,
flags and short categories; anything else is counted and dropped. A failed step is recorded, a
later step reconnects the page if it must, and the rehearsal continues. The final `summary` line
counts what is left running by image name, and it must be empty. The exit code is 0 only when
every step was `as_expected` and nothing was left.

**One run at a time.** Before anything else, a run takes an exclusive lock:
`rehearsal.lock` in the per-user `hermes-realtime\tools` directory that holds the shared
LiveKit server. It keeps the lock until its process exits, and the operating system releases
it even after a crash. While another run holds it, the script prints one `failed` preflight line
with category `rehearsal_busy` and exits 1. Two runs can't overlap and share LiveKit and Ollama;
they used to race past the port check below. If the lock's directory can't be created, the
line's category is `rehearsal_lock_unavailable`.

**Preflight.** When a required piece is missing, the script prints one `not_run` preflight line
and exits 0. The pieces are Windows, the pinned LiveKit binary, system Chrome, Playwright,
Kokoro, git, uv, and a running Ollama with the named model. A listener already on port 7880 is
a `failed` preflight with exit code 1. With the lock in place, that means a server no rehearsal
started, such as an orphan.

**Setup failures name their cause.** Every setup command runs through one bounded runner. That
covers the pinned Hermes cache (git and `uv sync`), the candidate clone, wheel build, host
install and its import check, and the Hermes clone, sync and wheel install.
- **A command that fails** ends the `setup` step with a short category, such as
  `host_requirements`, and the record carries its `exit_code`.
- **A command that times out** records `timed_out` instead, and its process tree is ended.
- **Either way, its stderr is kept.** The last 64 KiB stay in the run directory as
  `logs/setup-<category>.stderr`, cut on a character boundary.
- **A run directory too long for the pinned Hermes checkout** is refused before the checkout,
  as `run_dir_too_long`, and the record carries `path_chars`.
  - The test adds the checkout directory to the longest thing the checkout creates: a file, or
    a directory plus the 12 units Windows reserves for a short name inside it.
  - Lengths count UTF-16 units, as Windows does, against its 260-unit limit.
  - The default directory under `%TEMP%` fits.

**Differences from the operator's session.** Ctrl-C cannot reach a detached process, so:
- The host runs as `<host env>/python <clone>/scripts/rehearse_desktop_mvp.py --host-child <stop
  file> -- <host flags>`. When the stop file appears, the child interrupts its main thread
  exactly as Ctrl-C does. The forced restart ends the whole host process tree abruptly.
- The gateway is stopped through Hermes's own Windows stop path, the planned-stop marker that
  `hermes gateway stop` writes. LiveKit is killed.
- `hermes gateway status` and `stop` themselves are not run: on a machine with its own gateway,
  their process scan could reach that one.

**Limits.**
- The throwaway gateway and host use free ports, so a gateway already running on 8642 is left
  alone.
- The run directory is left in place. It holds private logs that may contain paths, the
  throwaway home's `.env` with its generated API and companion keys, and a `state.db` with the
  synthetic conversation. Publish only the rehearsal's own lines.
- The stand-in model never writes memory, so the deletion step does not exercise a review
  copying the conversation into memory.

### Launch

1. In terminal A, start the pinned local LiveKit server and leave it running:

   ```bash
   uv run python -m scripts.local_livekit serve
   ```

2. In terminal B, from the repository root, install the locked local profile and launch
   with the active Hermes env file. The file is parsed for `API_SERVER_KEY` and the companion
   endpoint (`HERMES_REALTIME_COMPANION_PORT` and `HERMES_REALTIME_COMPANION_TOKEN`) only;
   unrelated secrets are not copied into the realtime process environment:

   ```bash
   env -u PYTHONPATH uv sync --frozen --dev --extra local
   bash scripts/setup-kokoro-cuda-worker.sh
   env -u PYTHONPATH uv run --frozen --extra local hermes-realtime-host \
     --hermes-env-file C:/path/to/active/hermes/.env \
     --inference-provider codex \
     --conversation-profile natural_v1 \
     --natural-work-tools \
     --enable-public-search \
     --knowledge-speculation \
     --knowledge-recovery \
     --tts-provider kokoro \
     --kokoro-voice bf_isabella \
     --kokoro-intra-op-threads 8 \
     --kokoro-stream-chunk-chars 400 \
     --kokoro-worker-python C:/path/to/hermes-realtime/.hermes/runtime/kokoro-cuda/Scripts/python.exe \
     --allow-unsandboxed-hermes-tasks
   ```

   This is the qualification profile, not the conservative default. Omit
   `--conversation-profile natural_v1` to retain `legacy`; omit `--natural-work-tools` to retain
   explicit task prefixes only; omit `--enable-public-search` to disable all transcript-derived
   public-search egress. The knowledge flags are rejected unless public search is enabled; omitting
   them retains consent-bound final-transcript public-RSS lookup without overlap or recovery.
   Knowledge overlap requires Codex plus Moonshine and
   remains independent of background-work authority. The exact-turn knowledge deadline defaults
   to 3.5 seconds and may be reduced with `--knowledge-budget-seconds`.

   Moonshine v2 streaming tiers are selected explicitly with
   `--moonshine-model-tier tiny|small|medium`; `tiny` remains the compatibility default.
   This public bound deliberately excludes unqualified upstream architectures.
   The native Windows path runs on CPU and never silently changes tier or falls back to
   Faster Whisper. For public qualification, launch otherwise identical hosts with
   `--moonshine-model-tier small` and `--moonshine-model-tier medium`.

   Tier performance and memory use vary by CPU, operating system, audio, and package build.
   Use the generic benchmark and qualification tooling on the target deployment before changing
   a default. Do not treat upstream accuracy tables or one machine's measurements as Hermes
   acceptance evidence.

   On Windows, the full host defaults to local `kokoro-onnx==0.6.1` with the full English
   v1.0 model and British female Isabella voice (`bf_isabella`), upsampled from native
   24 kHz to the existing 48 kHz mono LiveKit transport. Other platforms default to Edge
   because the locked Kokoro dependency is Windows-only. Explicit provider selection never
   falls back. Model and voice assets are pinned by
   exact size and SHA-256. The CPU session defaults to 8 ONNX intra-op threads; override it with
   `--kokoro-intra-op-threads` when profiling different hardware. `--kokoro-speed` remains
   available for a deliberate speaking-rate tradeoff. `--tts-provider edge` remains an
   explicit diagnostic alternative; it is never selected as a fallback when Kokoro fails.

   Kokoro synthesis is demand-driven and conditionally emits deterministic provider chunks
   when an inference segment exceeds `--kokoro-stream-chunk-chars` (default 400; accepted
   range 32-1024). Source slices concatenate to the exact generated segment. The adapter does
   not synthesize another provider chunk until its consumer requests one; the host conversation
   loop intentionally keeps at most one chunk of look-ahead while the current chunk plays. Under
   `natural_v1`, the provider-neutral duration wrapper further splits text and validates actual PCM
   until every emitted logical chunk is at most three seconds; a provider attempt that cannot
   satisfy the finite bound fails closed. Cancellation revokes unpublished chunks and suppresses
   the remaining suffix. If
   chunking would change Markdown or pronunciation speech projection, synthesis fails closed
   to one chunk. Repeat `--kokoro-pronunciation TERM=SPOKEN` for up to 64 case-insensitively
   unique whole-term substitutions; each term is 1–64 characters and each spoken form is 1–128.
   Pronunciations affect backend speech
   only; transcript and conversation context retain the exact model text. The lexicon is empty
   unless entries are explicitly supplied.

   Use [`scripts/benchmark_kokoro.py`](../scripts/benchmark_kokoro.py) to measure one explicit
   model/provider row on the intended deployment. The benchmark records requested and actual
   execution providers so silent CUDA fallback cannot be presented as a GPU result. Treat local
   results as machine-specific; publish them only with deliberate provenance and disclosure.
   CPU FP32 remains the default when `--kokoro-worker-python` is omitted. When supplied,
   that value must name an existing absolute native Windows Python executable. The opt-in
   CUDA path uses the separately version-pinned environment created by
   `scripts/setup-kokoro-cuda-worker.sh`. Its [hashed dependency closure](../requirements/README.md)
   keeps `onnxruntime-gpu` as the sole owner of the ONNX Runtime import namespace; the CPU
   distribution stays in the host environment. The host owns one authenticated
   loopback worker process, verifies pinned model and voice hashes, and profiles hidden startup
   synthesis. Startup requires Conv/Gemm/LSTM on `CUDAExecutionProvider`; CPU execution is
   limited to an explicit control/spectral-op allowlist required by this graph. The worker
   refuses startup on any provider-policy drift. Benchmark process-boundary behavior on the
   intended deployment before drawing latency or throughput conclusions.

3. Copy the one-use fragment URL only into one local browser. Open it once, grant
   microphone permission, select the intended input device, and press **Connect**. Never
   paste the URL into chat, evidence, shell history, or an issue.

### Human procedure

Perform these checks in order. Stop on the first contradiction.

1. Say a short ordinary question. Require the final user transcript, first-token marker,
   first-playable-audio marker, audible speech, and delivery-confirmed assistant transcript.
2. Enter one ordinary typed question. Require the same foreground path and audible answer.
3. Ask one current, explicit-source, historical, or distinctive technical question. Require one
   bounded `knowledge_timing` inspector update and either source-backed evidence or a concise
   inability to verify. No query, page text, credential, or private handle may appear in browser
   events. Repeat once with knowledge flags omitted to exercise final-transcript lookup only.
4. Say, without a prefix, “Start background work to return one concise sentence confirming the
   completion test.” Require exactly one active public `task_...` state followed by terminal state
   and a proactive spoken completion. Then enter the explicit
   `task: Return one concise sentence confirming the background completion test.` fallback once.
   No `run_...`, `deleg_...`, bearer, or raw tool argument may appear.
5. Start a longer harmless task, then ask an ordinary foreground question while it runs.
   Interrupt assistant playback with speech. Require immediate attenuation followed by committed
   silence after server floor promotion, while the background task remains active and later
   completes. Repeat with one cough or brief backchannel: require bounded full-volume continuation
   without a new answer or whole-sentence replay. Then press **Stop speaking** during another reply
   and require immediate exact-stream silence.
6. Start another harmless long task and submit
   `cancel task: <the displayed task_... identifier>`. Require cancelling then interrupted
   state for exactly that task.
7. **Reject gate:** submit a task that asks Hermes to run
   `chmod 777 <a fresh guaranteed-nonexistent temp path>`. Require one redacted actionable
   request carrying a fresh request identity, click **Reject**, and verify Hermes reports
   rejection without retrying.
8. **Approve gate:** repeat with a different guaranteed-nonexistent temp path, inspect the
   displayed command and request identity, and click **Approve** once. The command should fail because the path
   does not exist; the purpose is approval routing, not a filesystem change. Never approve
   a command targeting an existing path during this gate.
9. Force a transient transport interruption while leaving the connected page open and without
   pressing **Disconnect**. Restore the same endpoint and credentials, wait for automatic
   reconnect, and require the same browser lease and still-live microphone track to resume without
   stale audio, duplicate transcripts, cross-participant media, replayed approval actions, or a
   second `getUserMedia` call. Reapply the user's mute choice before accepting recovered media.
10. Stay connected beyond the original bearer lifetime, exercise typed fallback once more,
   then press **Disconnect**. Require terminal stopped state and complete worker/provider cleanup.
   Finally press `Ctrl-C` in terminal B and then terminal A.

Retain raw monotonic first-token, first-audio, microphone-onset-to-attenuation, and
microphone-onset-to-committed-silence samples locally and report count, median, p95, and maximum.
Also count rejected-onset recoveries and false ducks. Record complete failure output but never tokens, launch fragments, API
credentials, internal delegation handles, private run IDs, or approval arguments. Stop
immediately on stale playback, authority duplication, credential disclosure, approval
without the exact actionable pending request identity, any action accepted after stop, or any command that
changes an existing approval-test target.

Use [`source-backed-latency.md`](source-backed-latency.md) for knowledge policy and benchmark
evidence, and [`release-gates.md`](release-gates.md) for committed-candidate qualification.

Official documentation: <https://docs.livekit.io/transport/self-hosting/local/>
