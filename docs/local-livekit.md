# Local LiveKit development

The first media tracer bullet uses a native LiveKit server on the Hermes host. LiveKit Cloud is intentionally deferred until local media, interruption, and reconnect behavior are measured.

## Pinned development server

- LiveKit Server: `v1.13.4`
- Windows asset: `livekit_1.13.4_windows_amd64.zip`
- SHA-256: `a326e025de516e93dfb3719bcd28e5a4ac16f21bcf1ef562499403ca98cc65fe`
- Release: <https://github.com/livekit/livekit/releases/tag/v1.13.4>

Downloaded binaries belong under `.tools/livekit/`, which Git ignores. Verify the archive against the release `checksums.txt` before extracting it.

## Start the local server

From Git Bash in the repository root:

```sh
LIVEKIT_KEYS="devkey: $(python -c "print('local-' + 'x' * 32)")"$'\n' \
  ./.tools/livekit/livekit-server.exe --dev --bind 127.0.0.1
```

The command overrides LiveKit's short built-in development secret with a deterministic
38-character loopback-only test secret constructed at runtime:

```text
API key: devkey
API secret: "local-" followed by 32 lowercase x characters
WebSocket URL: ws://127.0.0.1:7880
```

These credentials remain public and predictable. They must never be used for a LAN,
internet-facing, or production deployment.

On Windows, `--bind 127.0.0.1` restricts the HTTP/WebSocket signaling listener, but LiveKit 1.13.4 still opens its RTC TCP and UDP listeners on available interfaces. Keep Windows Firewall enabled and do not add inbound allow rules for this development process. Verify the actual listeners and firewall rules rather than assuming the media sockets are loopback-only.

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
   LIVEKIT_KEYS="devkey: $(python -c "print('local-' + 'x' * 32)")"$'\n' \
   ./.tools/livekit/livekit-server.exe --dev --bind 127.0.0.1
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
4. A narrowly scoped Windows Firewall rule for only the intended TLS listener and LiveKit media ports.
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
profile. Those belong to `hermes-realtime-host`. It is not the full human-test launcher described
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
`task: <objective>` and `cancel task: <public-task-id>` transcripts are always the deterministic
background-work fallback. With Codex and `--natural-work-tools`, ordinary language may invoke the
same authoritative start/cancel surface; without that flag, ordinary speech and typed text remain
tool-less foreground inference. The launcher requires Hermes's authenticated server-side
`/v1/runs` capabilities before exposing the one-use browser URL; there is no socket, subprocess,
provider, or unauthenticated fallback.

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

Do not continue unless the gateway is healthy and port 8642 is bound only to literal
loopback. The full launcher performs authenticated capability discovery itself.

### Launch

1. In terminal A, start the pinned local LiveKit server and leave it running:

   ```bash
   LIVEKIT_KEYS="devkey: $(python -c "print('local-' + 'x' * 32)")"$'\n' \
   ./.tools/livekit/livekit-server.exe --dev --bind 127.0.0.1
   ```

2. In terminal B, from the repository root, install the locked local profile and launch
   with the active Hermes env file. The file is parsed for `API_SERVER_KEY` only; unrelated
   secrets are not copied into the realtime process environment:

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
   pressing **Stop session**. Restore the same endpoint and credentials, wait for automatic
   reconnect, and require the same browser lease and still-live microphone track to resume without
   stale audio, duplicate transcripts, cross-participant media, replayed approval actions, or a
   second `getUserMedia` call. Reapply the user's mute choice before accepting recovered media.
10. Stay connected beyond the original bearer lifetime, exercise typed fallback once more,
   then press **Stop session**. Require terminal stopped state and complete worker/provider cleanup.
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
