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
plus `assistant_turn_interrupted` (`playback_silenced` has an empty payload, so it is recorded as `playbackSilencedObserved` supporting evidence only and never satisfies this), and no `assistant_turn_completed`
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
`interruptEventObserved`, `playbackSilencedObserved`, `postStopAudioMs`, `quiescenceObserved`, `quiescenceInconclusive`,
`freshReplyObserved`, `stopLatencyMs`). `stopLatencyMs` is the time from the end of the injected
Stop speech to the last non-silent worker frame observed in the quiescence window (0 if none).
Every window edge is a client observation: events are seen through 50 ms polls and injected
audio through a 200 ms `AudioSource` queue, so these are coarse bounds, not perceived speed,
and no per-pair PCM lineage is inferred from poll timestamps.

TARGET's exclusivity (one pair, one matching final, no unbound foreground event) is
re-validated after the precondition wait and again at the first Stop speech frame; worker audio
from an ambiguous window never qualifies TARGET (`journey_barge_ambiguous`). On any refusal the
receipt keeps whatever evidence exists (a pure, bounded snapshot taken in `finally` before
cleanup, which can neither change the verdict nor block cleanup: acceptance reads the validated
trace, never the snapshot, and a failed snapshot sets `snapshotError`). `targetInterrupted`,
`targetCompleted`, `postStopAudioMs`, `stopLatencyMs` and `quietTailMs` are null when that stage
was never reached, so "not measured" is distinct from an observed false or zero.

The receipt also carries diagnostic counts of projected events after the injection cursor,
which never change a verdict: `userPartialCount` (user `transcript_partial` events) with
`userPartialAtLeast3Alnum` (a boolean evaluated in memory, never storing text),
`echoSuppressedCount`, `transcriptEchoSuppressedCount`, `speechEndedCount`,
`voiceActivityEndedCount`, `bargeInNonSpeechSuppressedCount`,
`bargeInVerifierUnavailableCount`, `echoBargeInConfirmedCount`, and `interrupt_requested`
split into `interruptTargetCount` (the strict TARGET pair), `interruptEmptyCount` (an empty
payload) and `interruptOtherCount`. Published partial counts sit downstream of the host's
echo and playback-authority filtering, so a zero does not prove the STT backend emitted no
partial.

### `speak_task`

A `speak_task` step (exactly `label` and `kind`; a scenario holds one microphone step) tests
natural spoken delegation and a sole-active natural spoken cancellation. It is labelled for
#159 with a #237 cross-reference. It does not cover the explicit `task:`/`cancel task:` command
grammar or the ambiguous-cancel category.

1. **Delegate.** It runs the speak flow (warm-up, quiet gate) with the pinned request
   `sapi_task_dispatch.wav`. It needs voice activity ordered, exactly one user final with the
   exact normalized words, and exactly one new accepted dispatch admission and one new
   `task_state` active whose first state follows that final. Microphone input has no
   `typed_input_admitted`, so none is faked: attribution is by event order around the user
   final. A replayed earlier task, a competing second task or admission, or a task whose first
   state precedes the final fails. That task and its run are TARGET, frozen. Exactly one new
   task and one new run are owned for cleanup as soon as they are observed, and again on every
   exit path, so a later failure (a stream error included) still lets cleanup cancel exactly that
   task; with several candidates nothing is guessed and cleanup stays honestly unsettled. No
   sleep, hold or budget change is added; the work is genuinely substantial public work.
2. **Quiet with TARGET active.** From the freeze through the cancel injection there must be
   exactly one matching dispatch final and no further voice start, unbound or replayed
   foreground event (`journey_speak_finals`, `journey_task_unexpected_input`). It waits for no open foreground turn, no pending admission, one
   whole quiet interval, and TARGET still active. If TARGET already ended (including a
   rate-limit or lookup refusal) the result is inconclusive, `journey_task_target_terminal`.
   No replacement task is ever selected.
3. **Cancel.** It injects `sapi_task_cancel.wav` on the same track. At its first speech frame
   it re-checks that TARGET is still the sole live task with no competing task or admission and
   aborts before any cancel speech if not. It needs a fresh voice start, exactly one final with
   the exact words, and a `cancelling` or `interrupted` `task_state` for exactly TARGET after
   that final. A cancel state before the final is stale; any other task cancelling is foreign;
   any new dispatch fails.
4. **Native evidence before cleanup.** The native witness is read in the step (through
   `collect`) and must say `cancelled`. The cancellation's whole validate-and-snapshot (a voice
   start, exactly one final with the exact words, nothing foreign cancelled, no competing task or
   admission) runs before the witness read and again after it and the final poll, with the trace
   still live; a late second final is refused with its updated count kept in the receipt.
   Cleanup never creates acceptance evidence: a task that is only cancelled later is a failure.

The receipt holds booleans and counts only (no ids or text). It is controlled synthetic
evidence.

### `speak_confirm_cancel`

A `speak_confirm_cancel` step (exactly `label` and `kind`; it is the scenario's one microphone
step) tests the confirmed natural spoken cancel with several running tasks (#581, ADR 0004
"Confirmed natural cancel with several tasks"). It starts both tasks by **typed explicit
`task:` commands** so their stored objectives, and therefore the host's fixed readbacks
(`cancel_readback`, imported from the package, never retyped), are exact. It does not cover
spoken delegation; `speak_task` covers that separately.

1. **Two typed starts.** After the speak flow's quiet start, warm-up and quiet gate, it sends
   `task: <A>` then `task: <B>` (two fixed harmless objectives) through the typed input route.
   Each must yield exactly one new task and one new run whose first projected state is
   `active` after that send (`journey_confirm_not_dispatched`, `journey_confirm_competing`,
   `journey_confirm_dispatch_unbound`). Each is owned for cleanup as soon as it is unambiguous
   (bindings `<label>.a` / `<label>.b`). From then until the first clip only those typed finals
   may arrive: a voice start, another user final, or an unbound or replayed foreground event
   fails (`journey_confirm_unexpected_input`).
2. **Quiet with both active**, then `sapi_confirm_request_b.wav` (names B). At its first
   speech frame both tasks must still be active with nothing competing, otherwise it aborts
   before any speech. The one reply turn (a pair first seen after the request's final) must
   complete. Its last assistant final must be exactly B's readback, and that readback must
   appear only once. Model prose may precede it: the host speaks the readback after the
   provider stream closes and refuses model text after it. Any other shape around a readback
   is `journey_confirm_ambiguous`: text after it, a duplicate, or both readbacks.
3. `sapi_confirm_no.wav`: through a bounded drain window after its final nothing may be
   cancelling or interrupted and both tasks stay active (`journey_confirm_no_cancelled`).
4. **Quiet again**, then `sapi_confirm_request_a.wav` (names A): A's exact readback, completed.
5. **Quiet after A's completed readback**, then `sapi_confirm_yes.wav`. The existing
   drain, quiet interval and clip lead let the default playback tail and capture pre-roll
   expire; the gate resets on worker audio and refuses on its existing deadline. The only
   proof is the outcome: A observed `cancelling` or
   `interrupted` by `task_state` after the Yes final (`journey_confirm_not_cancelled`
   otherwise; a terminal result alone is not arming). B must never be cancelled
   (`journey_confirm_wrong_target`) and stay active (`journey_confirm_bystander_ended`).
6. **Native evidence before cleanup.** A's witness is read (through `collect`) and must say
   `cancelled`; then B's run status is read with `NativeRunWitness.live_status` and must be
   `queued`, `running` or `waiting_for_approval` (`stopping`, `cancelled` or `interrupted` is
   the wrong target; `completed` or `failed` is an ended bystander). The validation runs
   before and after each native read with the trace live. It never waits for B to finish.
7. **Cleanup** then cancels B by exact id, counted separately as `successPathCleanupCancels`.
   Cleanup never creates acceptance: a target only cancelled by cleanup is refused.

Classification precedence is failure before inconclusive. Every side effect is ruled out
before any reply is classified or any natural end is named, so a mismatch, a clarification or
an early end can never mask one. A cancel of either task during request B is
`journey_confirm_premature_cancel`; any cancel from the No through the Yes injection is
`journey_confirm_no_cancelled`; a cancel of A in the Yes clip before its final is premature;
any cancel of B after Yes is the wrong target. Only then are these inconclusive (no further
answer is sent and no replacement is chosen): a readback naming the other task
(`journey_confirm_nomination_mismatch`), a completed reply without a readback
(`journey_confirm_clarified`, including a readback with extra words), or a task that ends on
its own (`journey_confirm_target_terminal`). An interrupted or unfinished reply is
`journey_confirm_readback_incomplete`; more than one reply turn, or a reply that starts before
the request's final, is `journey_confirm_ambiguous`. Wrong words, a missing voice start or a
second final for any clip fail (`journey_confirm_transcript`, `journey_confirm_vad`,
`journey_confirm_finals`).

#### Short finals near playback

For a streaming recognizer, a two-character final near playback can retain authority
when its whole alphanumeric text agrees with the latest partial from that same capture
(ignoring case and terminal sentence punctuation). This path requires an explicit speech
presence verifier; acoustic promotion and transcript echo filtering still apply. Agreement
is consistency, not independent recognition: a recognizer wrapper may reuse its last partial
at the endpoint. Missing, stale, mismatched or echo evidence refuses the final.

This endpoint path does not authorize interruption from a short partial, make a capture
clean, or grant positive cancellation confirmation. A final is checked again after the
interruption await so a replaced foreground cannot inherit the old capture's authority.
Refusal diagnostics contain bounded counts and categories only, never recognized text.

#### Controlled background-model lifetimes

Short research tasks can finish before this spoken sequence reaches confirmation.
For an explicitly scripted-model rehearsal, an externally owned qualification controller
can use [`ControlledTaskModel`](../scripts/controlled_task_model.py). Pass a mapping from
separate local labels to the two unchanged, exact objectives, await `start()`, and configure
only the isolated Hermes profile's custom model URL/default to the returned URL and
`controlled`. The endpoint binds loopback on an ephemeral port and admits each objective
once. Unknown, stale, duplicate and nonstreaming requests refuse; no fuzzy matching occurs.

Each held response yields unfinished model chunks until its client disconnects or its
own local gate is released. These chunks let the native stream consumer observe cancellation;
SSE comments alone may be swallowed by the SDK. They do not impose a minimum duration or
complete work on a timer. The endpoint never starts a task, grants cancellation, emits task
events, or edits native run state. There is no HTTP release control.

Keep both gates unreleased throughout the successful journey. Use the existing task/run
bindings and native witnesses, unchanged: A must be natively cancelled and B natively live
before cleanup. `snapshot()` exposes admission/open/closed/refusal counts only; a closed
model stream is not a native cancellation witness. Afterwards the controller must settle
the owned tasks and call `close()` in its cleanup path, including on failure. Use the
[existing process owner](contained-validation.md) for the controller and native descendants;
this library neither owns native processes nor replaces their settlement checks.

Retain a separate receipt explicitly labelled **native runtime with scripted background-model
lifetimes and synthetic audio**. It may prove the confirmation/cancellation mechanism under
controlled task longevity. It cannot qualify the configured background provider, research
results, natural task duration, physical microphone, AEC, or perceived speech quality. A
controlled success does not close those separate acceptance gates.

**Outcome classes.**
- **Inconclusive:** these mean a precondition of the scenario was lost. They are never a product
  failure:
  - `journey_confirm_nomination_mismatch`;
  - `journey_confirm_clarified`;
  - `journey_confirm_target_terminal`;
  - `journey_confirm_bystander_ended`, meaning B ended on its own before the witness. If any
    cancel evidence for B exists, the outcome is `journey_confirm_wrong_target` instead.
- **Failures:** every other `journey_confirm_*` refusal, including
  `journey_confirm_readback_incomplete`. Failure precedence always applies.

`successPathCleanupCancels` counts only the cleanup cancel of B on the success path, after
the native evidence. Cleanup after a refusal goes through the mandatory owned-task cleanup
and its cleanup flags. That cleanup is authoritative and is not counted here.

The receipt holds booleans, counts, client-local milliseconds and the fixed reply categories
`readbackBOutcome` / `readbackAOutcome` (`target`, `other`, `clarified`, `interrupted`,
`incomplete`, `none`, `ambiguous`, or null if not reached). Assistant text is only compared in
memory with the two pinned readbacks; no text or identity is stored.

Flat `readbackA` / `readbackB` fields retain event, distinct-pair, assistant-final and
readback-final counts. Their `Reason` is null or a fixed ambiguity category: `unbound_reply`,
`foreign_phase`, `reply_before_final`, `multiple_turns`, `multiple_readbacks`, or
`text_after_readback`. These explain an existing refusal without changing its classification
or failure precedence. Counts remain bounded by the trace capacity.

This is a controlled synthetic path. It bypasses the physical microphone and acoustic
echo cancellation. It does not prove source-attributed microphone qualification or the
naturalness of the conversation.

Offline checks: `uv run pytest -q tests/test_native_typed_journey.py`.
