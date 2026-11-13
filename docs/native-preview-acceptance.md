# Native preview acceptance procedure

This is the operator procedure for the profile the native preview runs today: a cloud
foreground model through the Codex app server, Hermes background work, Moonshine
speech-to-text, Kokoro speech on a managed GPU worker, the `natural_v1` conversation
profile and automatic task start. Set it up with the
[native work voice preview](native-work-preview.md) and
[managed Kokoro](kokoro-mcp.md) guides.

It is a procedure, not a result. Nobody has run it end to end from these pages, and
writing it down qualifies nothing. The outcome and its acceptance record belong to
[#159](https://github.com/sushiHex/hermes-realtime/issues/159). The
[desktop MVP acceptance](desktop-mvp-acceptance.md) and
[diagnostic](desktop-mvp-diagnostic.md) guides are historical: they preserve the
explicit-command session with the Phi/Qwen local-model profile. This page reuses their
marker catalog and deletion limits by link and replaces their session steps for the
current profile.

## Rules for the record

Record categories, counts, states, milliseconds, seconds and yes/no only. Never record
transcripts, audio, objectives, model output, tokens, keys, launch links, paths, user
names, host names or device names. Use synthetic conversation content and harmless,
bounded task objectives. For each step record one outcome: `as_expected`, `different`
(with a one-word category), `failed` or `not_run`. A failed step is a finding. Record
it and continue while the setup still allows. A step you did not exercise stays
`not_run`. If the candidate changes, start a new record; a later pass does not erase
an earlier outcome.

No machine timing establishes that a person heard speech. The operator supplies the
hearing and speaking observations.

## Before the session

Record once, so the session binds to one candidate:

- the hermes-realtime commit, and whether `prepare` accepted the plan (its status
  category; for a refusal, the `failure_category` in
  `preparation-refused.private.json`);
- the Hermes commit the plan binds;
- the foreground and background model names and efforts from the plan;
- the Moonshine tier and Kokoro voice, and whether the plan carries both
  `kokoro_mcp_config` fields (managed GPU speech: yes or no);
- `host_options`: `natural_work_tools`, `task_start_policy` and
  `conversation_profile`;
- the web backend category Hermes will select (keyed, self-hosted or keyless) and
  whether `web_evidence` is selected in the plugin settings;
- the readiness policy (`kernel_listeners` or `dev_owned_services`);
- the browser version, microphone and speaker class, and CPU and GPU class.

## Where the evidence comes from

- **Command status lines.** Each `preview_cutover.py` command prints one
  `[voice-preview]` line with a JSON `category`. Record the category.
- **Receipts** in the plan's output directory: the owner's
  `ready-receipt.private.json` and `result-receipt.private.json`, and
  `handoff.private.json` from `finalize`. They are private. Record only the fields
  this page names.
- **Child output.** While the preview runs, the owner keeps the first 1 MiB of each
  child's output in `gateway.private.log`, `livekit.private.log` and
  `host.private.log` in the output directory, and deletes them when it writes the stop
  receipt. Extract marker lines before you stop, with the command in the diagnostic
  guide's [Markers](desktop-mvp-diagnostic.md#markers) section, which also explains
  each marker. The files are private; publish marker lines only. Writes are buffered,
  so a marker can appear late. A marker missing from the file is unknown, not absent.
  Markers printed during shutdown cannot be collected, because the files are gone when
  `stop` returns. The owner's own `owner.private.log` and `owner.stderr.private.log`
  remain; they are private too.
- **The browser.** The **Session model** panel shows providers, models, **Response
  last** and **Response average**. **Diagnostics → Latency markers** lists named
  timings in milliseconds.
- **The operator.** Wall-clock seconds for task and connection steps, and every
  hearing observation.

The [typed native journey client](native-typed-journey.md) can cross-check typed
dispatch, status, recall and cancellation without a microphone. It does not replace
any step here.

## Session steps

### 1. Launch and readiness

- **Do:** run `prepare`, `finalize` and `launch`, then wait for the ready receipt.
  `finalize` needs an earlier preview run directory; see the preview guide's
  [support limits](native-work-preview.md#support-limits).
- **Expect:** `prepared`, `ready_to_launch` and `launch_requested`, then a ready
  receipt with `category: ready`.
- **Record:** the three status categories; the receipt's `category`,
  `readiness_policy` and `listener_ownership`; seconds from `launch` to the receipt.
  If no ready receipt appears, record the result receipt's `failure_category` and the
  category names and counts under its `child_output_metadata`, which include any
  managed speech refusal (`kokoro_mcp_...`) and host startup failure.

### 2. Connect and audibility

- **Do:** open `http://127.0.0.1:<ports.host>/`, press **Connect**, grant the
  microphone and wait for **Listening — speak naturally**.
- **Expect:** a short spoken readiness cue. The **Session model** panel names the
  plan's foreground model and speech providers.
- **Record:** seconds from **Connect** to **Listening**; cue audible yes or no; panel
  matches the plan yes or no.

### 3. Spoken turn

- **Do:** ask one short ordinary question aloud.
- **Expect:** your final transcript, an audible reply and its transcript as it is
  delivered. No task card.
- **Record:** audible yes or no; task cards created (expect 0); **Response last**;
  `transcript_to_first_token` and `first_token_to_audio`.

### 4. Typed turn

- **Do:** type one short ordinary question and send it.
- **Expect:** the same path as step 3.
- **Record:** audible yes or no; **Response last**.

### 5. Natural task request with a sourced, dated result

- **Do:** ask in ordinary words, with no command prefix, for something that needs
  current public information and has a checkable date.
- **Expect:** no **Proposed background task** card, because the start policy is
  automatic. Exactly one **Background task** card appears after the turn that
  announces it, reads **Running**, then **Completed in**, and the completion is
  spoken. The result states a date and its detail lists sources. No `run_...` or
  `deleg_...` identifier appears anywhere.
- **Record:** proposal card shown yes or no; task cards created; the state sequence;
  seconds from the request to active and to completed; completion heard yes or no;
  number of sources listed; date stated yes or no; date agrees with a listed source
  yes or no (open the source yourself); private identifier visible yes or no.

A sources list shows where the text came from. It does not prove the answer is
correct, current or complete.

### 6. Status while a task runs

- **Do:** ask naturally for a longer harmless task. While its card is active, ask how
  that work is going.
- **Expect:** a spoken reply consistent with the card, no second task card, and the
  card still active.
- **Record:** reply consistent with the card yes or no; task cards created by the
  status question (expect 0); **Response last**.

### 7. Barge-in during an active task

- **Do:** while the task from step 6 is still active, ask an ordinary question and
  speak over the reply. Then press **Stop speaking** during another reply.
- **Expect:** playback lowers when you start speaking and then stops; **Stop
  speaking** silences the reply at once. The task card stays active through both and
  later completes: interrupting speech must not cancel work. None of the assistant's
  words appear in your own transcript line.
- **Record:** playback yielded yes or no; cut mid-word or clicked yes or no; assistant
  words in your transcript yes or no; every `[speech-stop]` line; the task state
  before, after and at the end.

### 8. Explicit task commands

Run these in order and record each part separately. A task card shows its state as
**Running** (active), **Stopping** (cancelling), **Stopped after** (interrupted),
**Completed in**, **Failed after** or **Not started**.

1. **Spoken start.** Say `start task` and a short harmless objective without pausing
   after the words. Expect one new task card that reads **Running** and then
   **Completed in**. The command is recognized only when the final transcript begins
   with `start task` followed by a space, so a transcript with a comma after `task` is
   an ordinary turn. If that happens, record the punctuation category, not the words.
2. **Typed start.** Type `task: ` followed by a short harmless objective. Expect the
   same.
3. **Cancel with no task.** With no active task, say `cancel task`. Expect the page to
   say there is no active task to cancel, and nothing else to change.
4. **Cancel with one task.** Start one longer harmless task, then say `cancel task`.
   Expect exactly that task to read **Stopping** and then **Stopped after**.
5. **Ambiguous cancel.** Start two longer harmless tasks with different objectives,
   then say `cancel task`.
   Expect the page to say several tasks are active and to point to the task card's
   **Cancel** control, with neither task changing. Press **Cancel** on one card and
   expect only that task to read **Stopping** and then **Stopped after**. Then press
   **Cancel** on the other card and expect the same for it.

- **Record:** for each part, the state sequence and task-card count; seconds from the
  command to **Running**, **Completed in** or **Stopped after**; for parts 3 and 5,
  guidance shown yes or no and tasks changed by the bare command (expect 0).

A cancellation acknowledgment means stopping was requested. Only the terminal state
shows that the task ended. The card's **Cancel** sends the typed
`cancel task: <task-id>` command for that card. The card does not display the
identifier, so typing that form by hand is not part of this step.

### 9. Disconnect and reconnect

- **Do:** state one synthetic fact and wait for the reply. Press **Disconnect**, then
  **Connect**. Separately, reload the page and press **Connect**. After each, ask a
  question that needs the fact without restating it.
- **Expect:** **Listening** returns, with no stale audio and no duplicated transcript
  entries, and the question is answered in context.
- **Record:** each reconnect's outcome; seconds from **Connect** to **Listening**;
  recall yes or no; stale audio yes or no; duplicated entries yes or no. Record a
  recall failure even when the history is visibly present.

### 10. Stop, hand off and restart

- **Do:** start one longer harmless task. While it is active, extract the markers, then
  run `stop`. Prepare a plan with a new output directory, run `finalize` with
  `--previous` naming the stopped run, then `launch` and **Connect**. With managed GPU
  speech, the new output needs its own operator profile and configuration
  ([managed Kokoro](kokoro-mcp.md)). Ask a question that needs the fact from step 9.
- **Expect:** `stop` reports `stopped`. The host stops the active task as it shuts
  down. If Hermes confirms that within the 45 seconds the owner waits for the host to
  exit, the stop receipt has `run_journal_empty: true`. `finalize` reports
  `ready_to_launch` and the handoff records `normal_stop_handoff_verified`. The new
  host's log has a `[voice-tail]` line with a positive `restored` count. After
  **Connect**, the question is answered in context, no task is active and the stopped
  task does not start again. A clean stop makes no restart announcement.
- **Record:** the `stop` category; from the stop receipt, `category`,
  `failure_category`, `cleanup_settled`, `cleanup_failed`, `owned_survivor_count`,
  `run_journal_empty`, `host_graceful_stop_timed_out`, `host_exit_code`, and
  `remaining_owned_listeners` or
  `remaining_accepting_endpoints`; the handoff category; the `[voice-tail]` line;
  recall yes or no; active task cards after **Connect** (expect 0); before any new
  request, the new run's `state\hermes-runs-v1.json` absent, or its `pending` and
  `admitted` counts (expect 0 and 0); stopped work started again yes or no.

If the stop receipt is `refused` or the journal is not empty, `finalize` refuses.
Record that as the outcome and mark the rest of this step `not_run`. Do not edit a
receipt to continue.

### 11. Delete the voice conversation

- **Do:** follow [session step 9](desktop-mvp-diagnostic.md#9-delete-the-voice-conversation)
  of the diagnostic guide with a synthetic unique phrase.
- **Expect:** the transcript clears at once and the status moves from pending to
  **Voice conversation deleted.** The follow-up reply does not know the phrase. The
  [support limits](desktop-mvp-diagnostic.md#support-limits) apply: learned memory,
  skills and delegated tasks remain, and deletion is logical.
- **Record:** the status transitions in order; seconds from confirming to complete;
  phrase returned yes or no; any `[voice-forget-send]` or `[voice-forget]` line. Never
  record the phrase.

### 12. Final cleanup

- **Do:** press **Disconnect**, extract the markers, then run `stop`. Then check
  independently that nothing is left:

  ```powershell
  $pids = Get-Content "<output>\pids.json" | ConvertFrom-Json
  $owned = @($pids.gateway, $pids.livekit, $pids.host,
             [int](Get-Content "<output>\supervisor.pid.private.txt"))
  @(Get-Process -Id $owned -ErrorAction SilentlyContinue).Count
  @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object LocalPort -in <ports.api>, <ports.companion>, <ports.livekit>, <ports.host>).Count
  ```

  With managed GPU speech, also read the vram-mcp coordinator's own status and count
  the Kokoro claims this session still holds.
- **Expect:** `stop` reports `stopped`. Both counts are 0. A process ID can be reused
  by an unrelated process after a restart, so a nonzero process count needs a look at
  what the process is before it is called a survivor. The check assumes the run
  reached readiness, so that `pids.json` names all three children. No Kokoro claim
  from this session remains. Processes and services you did not start are untouched.
- **Record:** the same stop receipt fields as step 10; the two counts; remaining
  Kokoro claims. Shutdown markers such as `[voice-review-close]` cannot be collected
  here; record them `not_run`.

Stop only through the `stop` command. Do not end processes by name.

## Operator contingency: the host stops answering

This is not a pass/fail step. It applies at any point in steps 3 to 11. Two faults leave
the host process running while it can no longer converse
([#531](https://github.com/sushiHex/hermes-realtime/issues/531)).

**Recognise it.** The assistant stops answering, with nothing spoken and no error shown,
after one of these:

- a task start that ends in an error with an unknown outcome. The page shows one task
  card with status `uncertain`, and the host prints one
  `[task-command-settlement]` line whose JSON has `kind` `start` and `category`
  `acknowledgment-uncertain`. The next ordinary turn then fails silently;
- a foreground failure after the Codex app-server child dies. The next turn prints one
  `[codex-send-bound]` line whose JSON has an `outcome` of `send_failed`, and every
  later turn fails with no line.

The page's advice to restart the session, and a browser reconnect, do not clear either
fault. Only a new host does.

**Recover.** Extract the markers, then stop the host and start it again through the
maintained commands in the preview guide's
[Commands](native-work-preview.md#commands) section. Press **Connect** and continue
from the step that was interrupted. The restart restores the saved conversation. A task
that was active is stopped, not resumed.

**Record.** The occurrence is a `failed` outcome for the step in progress. Record only:

- the step number;
- the marker line or lines, with their categories and outcome;
- whether the conversation was recalled after the restart, yes or no;
- the dispatch count before and after the restart.

Do not record transcripts, audio, objective text or paths.

The planned fix is tracked in [#531](https://github.com/sushiHex/hermes-realtime/issues/531)
and the proposed [ADR 0008](https://github.com/sushiHex/hermes-realtime/pull/529): a
host-wide latch ends the host and leaves one evidence line.

## Not covered

These are outside the procedure today. Leave them `not_run` and name the reason.

| Not covered | Why |
| --- | --- |
| Recovery of a host that is running but cannot converse, after a task-start error or a dead foreground child | There is no automatic recovery. Use the [operator contingency](#operator-contingency-the-host-stops-answering); it counts as a failure of the step in progress. Tracked in [#531](https://github.com/sushiHex/hermes-realtime/issues/531). |
| Approval prompts, and what an approval card shows after Hermes times out, is cancelled or receives a late answer | The card states are tracked in [#291](https://github.com/sushiHex/hermes-realtime/issues/291) and are not implemented. |
| Handoff from a run whose owner died, from an older run, or from a run already handed off | `finalize` accepts an abandoned run only when no work was recorded, and nothing ties `--previous` to the newest unconsumed run. Hardening is tracked in [#496](https://github.com/sushiHex/hermes-realtime/issues/496). |
| Abrupt owner or host death while a task is active | The run record is then not empty and `finalize` refuses it. The preview has no maintained recovery command for that state. |
| The first launch on a machine with no earlier preview run | `finalize` requires `--previous`; see the preview guide's [support limits](native-work-preview.md#support-limits). |
| Evidence capture, foreground public search, Tailscale and mobile | Not part of this profile. |

## Record and disposition

Post the record to [#159](https://github.com/sushiHex/hermes-realtime/issues/159):
the profile record, then each step's outcome, timings with their sample counts,
counts, and marker lines verbatim. Say for each timing whether it came from the browser
or from the operator. Link each defect to its owning issue.

The record establishes only the profile and candidate it names. It does not qualify
another profile, release or tag, and it does not replace the governed physical capture
family or the remaining Slice 0 gates.
