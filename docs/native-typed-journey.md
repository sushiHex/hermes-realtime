# Typed native journey client

`python -m scripts.native_typed_journey` checks an already running isolated
preview through its existing browser HTTP contract. It starts no services and
adds no executor. Use a preview owned by the operator, with no other participant
or active task. The client joins its existing LiveKit room without subscribing
to audio, sending microphone audio, or retaining PCM.

The client requires the version 2 events endpoint and event-batch response from
the matching updated preview. Other input and session operations retain their
version 1 endpoints. An older preview is refused explicitly; the client does not
downgrade or interpret a legacy display-limit notice as the full result.

Provide an external JSON scenario and a **new** receipt filename. Preflight is
offline by default; only `--execute` submits input or contacts the running host.

```text
uv run python -m scripts.native_typed_journey \
  --preview-dir /private/preview \
  --native-profile /private/preview/input/home \
  --browser-origin http://127.0.0.1:28765 \
  --api-origin http://127.0.0.1:59876 \
  --livekit-url ws://127.0.0.1:7880 \
  --scenario /private/scenario.json \
  --receipt /private/preflight.json
```

For the authorized real check, repeat with `--execute` and another receipt
filename. On Windows use absolute Windows paths and PowerShell's continuation
syntax, or put the command on one line. Authentication is read internally from
the selected native profile's `.env`, never from a command-line token. Loopback
origins and the selected preview's room/worker identities must match exactly.

An example input shape (replace the synthetic objective outside Git):

```json
{
  "version": 1,
  "steps": [
    {"label": "lookup", "kind": "dispatch", "text": "Synthetic research request", "wait": "accepted"},
    {"label": "status", "kind": "status", "task": "lookup", "text": "How is that work going?"},
    {"label": "recall", "kind": "recall", "task": "lookup", "text": "What did that work find?"}
  ]
}
```

There are at most 12 steps and 4096 characters per input. `dispatch` defaults to
waiting for completion; `wait: "accepted"` allows the next status question while
work runs. An optional `tool` requires paired successful native evidence for
`web_search`, `web_extract`, or `mcp__vram__vram_status`. Specifying it waits for
completion. Web assertions additionally require a usable result as defined below.
The status-tool witness requires exact integer `cmdline_chars: 0`
and structured status data with command lines omitted. Set that request in the
external objective. This assertion never alters the model's tool arguments.

`status` and `recall` reference an earlier dispatch label. Completed status is
valid; neither step requires another dispatch. Recall waits for its referenced
task to settle. A `cancel` step has only `label`, `kind`, and `task`: the client
submits the existing explicit cancellation command with the **exact bound task
ID**. A task that already completed cannot qualify active cancellation. Start a
distinct authorized task for that case; do not repeatedly ask an answered
research question merely to force dispatch.

A `natural_cancel` step has exactly `label`, `kind`, `task`, and `text`. It
submits the natural-language `text`, which must not begin with the typed
`cancel task:` or `task:` grammar (that route is the `cancel` kind). It passes
only when all of these hold: the bound task is `active` at step start; a
`cancelling` or `interrupted` task state for exactly that task, and no other,
arrives after this step's own input admission (compared by browser event
sequence) and during the step, never from earlier state or the cleanup
`cancel task:`; the turn completes on its exact identity; and the native witness
for that task ends `cancelled`. Turn completion alone is never task success.

A turn's completion binds through its first-token pair or, when the product
emits no token (a natural-cancel selection, acknowledgment, or refusal), through
the exact admitted `inputSequence`/`bindingGeneration` of this step's input,
excluding any pair seen before the step. Each turn row records which in
`terminalBinding` (`token` or `admittedInput`). A tokenless completion never
satisfies `status`, `recall`, or `conversation` (`journey_turn_tokenless`).

If the native witness says `completed` but the task failed with one of the
host's exact fixed web-refusal notices for that task, the failure is
`journey_task_failed` and the receipt's `taskFailures` records
`classification: host_refused` with the refusal category. Without that exact
evidence the mismatch stays `journey_terminal_binding`. Both are failures.

`conversation` has exactly `label`, `kind`, and `text`, with no task reference,
wait option, or tool assertion. Use it for an ordinary fixed-text turn, including
speech timing comparisons. It waits for the identity-bound foreground turn to
complete and, like status/recall, refuses any new task or observed native admission.
It uses the same timing markers and cleanup; it does not dispatch work to obtain
a latency measurement or establish that the spoken wording matched the request.

Acceptance binds one new task card to one observed native journal admission.
The client samples the journal around event polling and retains admitted IDs
only in memory. If a fast run was never observed, it reports unproven evidence;
it does not rerun the task. Native evidence uses authenticated durable GET Run
status and read-only, session-scoped SQLite request/result pairs. It **never**
subscribes to native Runs SSE, whose queue belongs to the production adapter.
`toolEventCount` is zero; `toolRecordCount` counts database records. Successful
pairing proves recorded tool execution, not the truth or completeness of a
model's prose. Cancelled runs can have explicitly counted unpaired requests.

`pairedResults` and `completed` count recorded request/result pairs, including
failures. `successfulResults` preserves the legacy result-validation rules;
`pairSuccess` means every paired result passed those rules with no pending call.
It is neither a pure transport flag nor proof of relevance or truth.

For web tools, `usableResults` counts **legacy-valid results** containing at least
one usable row in that same result. A usable row has a structural HTTP(S) target,
nonempty `description` (search) or `content` (extract), and no item/call error or
policy block. Extraction also requires supplied request URLs and the existing
exact returned/requested URL-set match. A title alone is insufficient. The URL
check reuses the existing parser/literal-IP predicate; it does not resolve DNS,
prove public reachability, or establish source authenticity.

`usableRows`, `errorRows`, `emptyRows`, and `unboundRows` aggregate retained row
observations across calls. These categories can overlap. Unbound includes a URL
or request binding that was not established. A partial extract may have usable
rows while its `successfulResults` and `usableResults` remain zero; it cannot
qualify the web assertion. Counts do not assess factual support, passage
relevance, or completeness, and never include the retained URLs or text.

Receipts contain labels, categories, counts, terminal states, and timings only.
They omit IDs, credentials, text, arguments, URLs, tool results, and audio.
Existing receipt files are never overwritten. On failure, bound active tasks
are cancelled by exact task ID before the client stops its browser lease and
disconnects; services and native processes remain owned by the preview owner.
Unresolved cleanup is reported, never treated as a successful journey.

An executed receipt may also carry `diagnostics`, which explains a verdict but
never changes it or the cleanup status. For each step it records whether this
step's `typed_input_admitted` arrived and carried a valid tuple. It counts the
step's terminal kinds and, for up to eight terminals, records whether `turnId`,
`inputSequence` and `bindingGeneration` were present, whether the correlation
equalled the admitted tuple (null when none was admitted), whether the pair
predates the step, and whether it was the bound one. For a task-bound step it
records whether a `cancelling`/`interrupted` state for that step's task was seen
during the step and after its admission, plus the native terminal category.
`preCleanupTasks` records each task's tracker status and native category, captured
synchronously before cleanup acts, so nothing optional can delay or skip mandatory
cleanup. Diagnostics perform no native I/O: the native category is the one
`collect()` already retained for that task, or null meaning "not read; collect()
did not run". Cleanup's own later collection does not change the snapshot. No
tuple, ID or text is recorded, and `diagnostics` is dropped first if it would
overflow the receipt.

When the existing failure handler catches an `OSError`, its receipt may include
`osError`: a fixed source-context function and line, `journal`/`scenario`/`other`
path role, fixed exception kind, and nullable bounded `errno`/`winerror` numbers.
This is the nearest supported client context, not attribution of the OS cause.
No exception message, filename, frame locals or credentials are retained. Pair
the context with the client revision used for the check. Collection is best
effort; it adds no reads or retries and does not change failure or cleanup rules.

Warm measurements require matching turn ID **and** generation. A foreground
pair is bound only after the matching typed admission and user transcript;
background/replayed speech cannot supply that pair. Missing identities remain
unknown. Same-pair token-to-playable timing measures server preparation, not
browser audibility or first nonzero PCM.

Three optional passive fields partition that interval without changing acceptance.
`firstPublicationMs` records the first matching current-pair
`assistant_text_generated` timestamp relative to the input start.
`tokenToPublicationMs` measures from the existing first-segment marker to that
publication; it can include opening coalescing and model continuation, rather than
raw first-model-delta latency. `publicationToPlayableMs` includes publication
queueing, synthesis and consumer admission before the transport play call. It is
neither isolated backend/GPU inference time nor browser-audible latency.
Missing timestamps or reversed interval endpoints leave the affected interval null.
These observations do not bind another turn or expand the existing marker cache. The three fields may be
omitted if they would overflow the unchanged receipt bound; existing failure
evidence takes precedence. The client does not infer semantic
recall accuracy, source quality, phonemes, microphone recognition, or voice
interruption from an HTTP transcript. Those remain separate acceptance checks.

## Synthetic `speak` step

A `speak` step has exactly `label` and `kind`; a scenario may hold one. It streams
the sha256-pinned synthetic clip `tests/fixtures/audio/sapi_hello_can_you_hear_me.wav`
(`Hello, can you hear me?`, Windows SAPI, no human speech) into the preview's LiveKit
room as microphone audio: one `microphone-1` track at 48 kHz mono in 10 ms frames at
real-time pace, with bounded leading and trailing silence, then one `/api/v1/media`
activation. The audio crosses the real VAD, STT, foreground model and TTS. Only the
worker's audio track (the exact validated worker identity) is subscribed before
readiness, one track only (a second distinct worker track is ambiguity), and received
PCM is measured by RMS and discarded; nothing is written to disk or the receipt. The
clip's bytes, duration, channels, sample rate and SHA-256 are verified before anything
is published. After readiness the step waits for one whole local quiet interval,
measured from the later of gate entry and the last observed worker audio (so a cue the
host has not started yet cannot pass for silence), bounded by a timeout, before speaking.
The receipt records `cueAudioObserved` and `quietGateMs` (client clock only). Frames are paced in real time through a queue-bounded `AudioSource`,
and measurement windows (settle, input, response) only advance, so they never overlap.
The worker confirms readiness only after it receives PCM, so a warm-up silence producer feeds
the published track from activation until the clip starts. It is bounded by frames, bytes and
time, is the only PCM producer at that moment, is cancelled and settled on every exit path,
and stops before the clip without disposing the shared `AudioSource`. A `speak` step clears
only a completed preceding typed step's observation state (so the microphone final is not
also checked as that step's typed input) and keeps seen identities, admissions, task
baselines and sequence authority; a typed turn still active is rejected
(`journey_speak_previous_turn_active`), never erased.
Client and server clocks are never mixed; the step computes no latency.

Microphone input emits no `typed_input_admitted`, so the step is attributed strictly.
It passes only when the session starts quiet (no active task, no open foreground turn,
no worker audio for a settle period), voice activity started then ended, exactly one
new user `transcript_final` whose lowercased, punctuation-stripped words equal the
fixture's, exactly one new foreground pair bound by `first_foreground_token` after that
final and completed, no new task, and non-silent worker audio inside that response
window. More than one final, an unbound or concurrent turn, a missing pair or no audio
fails. The receipt's `speak` object holds booleans and counts only: no transcript text,
identity or audio.

### `speak_barge`

A `speak_barge` step (also exactly `label` and `kind`; a scenario holds one microphone step,
`speak` or `speak_barge`) first runs the same flow with the pinned request
`sapi_long_ocean_story.wav` (`Please tell me a long story about the ocean.`). The one new
token-bound pair is TARGET. Only once TARGET has `first_playable_audio`, is still running and
has at least 1500 ms of worker audio in its window does it inject the pinned `sapi_stop.wav`
(`Stop.`) on the same track, under the same stream bound. If TARGET ends first the step is
inconclusive (`journey_barge_target_terminal`); it never claims an interruption.

It passes only when all of these hold: a fresh voice start after injection and exactly one new
user final with the Stop words; for TARGET's exact pair, `interrupt_requested` for its turn
and/or `playback_silenced`, plus `assistant_turn_interrupted`, and no `assistant_turn_completed`
through the final drain; no other pair active while TARGET runs and at most one fresh pair;
no new task; and audio quiescence. Quiescence is a window from injection end that closes at the
earlier of a fresh pair's first token or a bounded drain (the 1500 ms budget plus the 500 ms
quiet interval). Worker audio must stop within 1500 ms of injection end and then stay silent
for a whole 500 ms observed interval. If a fresh reply closes the window before that interval
is observable, quiescence is inconclusive (`journey_barge_quiescence_inconclusive`) and the step
does not pass. Audio after a fresh token is not counted against TARGET. These thresholds are
declared constants, never tuned to pass.

The receipt keeps booleans, counts and client-local milliseconds (`targetAudioMsBeforeStop`,
`stopInjected`, `stopRecognized`, `targetInterrupted`, `targetCompleted`,
`interruptEventObserved`, `postStopAudioMs`, `quiescenceObserved`, `quiescenceInconclusive`,
`freshReplyObserved`, `stopLatencyMs`). `stopLatencyMs` is the time from the end of the injected
Stop speech to the last non-silent worker frame observed in the quiescence window (0 if none).
Every window edge is a client observation: events are seen through 50 ms polls and injected
audio through a 200 ms `AudioSource` queue, so these are coarse bounds, not perceived speed,
and no per-pair PCM lineage is inferred from poll timestamps.

This is a controlled synthetic path. It bypasses the physical microphone and acoustic
echo cancellation. It does not prove source-attributed microphone qualification or the
naturalness of the conversation.

Offline checks: `uv run pytest -q tests/test_native_typed_journey.py`.
