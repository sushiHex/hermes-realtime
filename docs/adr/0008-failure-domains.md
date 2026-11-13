# ADR 0008: A latch ends its holder

Status: Proposed. The owner's instruction of 2026-11-13 on
[#159](https://github.com/sushiHex/hermes-realtime/issues/159) settles decisions 1 to 4 below
as recommended, for the two reproduced faults tracked as
[#531](https://github.com/sushiHex/hermes-realtime/issues/531); decisions 5 to 10 are open.
Each other migration step gets an issue before work starts
([ADR 0001](0001-github-work-tracking.md)). It elaborates Move 3 and decision 7 of [ADR 0006](0006-architecture-direction.md) ("a host-wide
fault exits the host and restart recovers") and changes neither.
Reviewed source: [`5b66d584`](https://github.com/sushiHex/hermes-realtime/commit/5b66d5842d626531d81d4911348f6a4b8a60b972).
Every `file:line` is at that commit, relative to `src/hermes_realtime/` unless it starts with
`scripts/`, `tests/`, `web/` or `docs/`. Nothing below is implemented. This record changes no
schema, prompt, default or gate. Every claim is from reading source; nothing was run.

## Context

The package handles an unexpected failure with a **latch**: an object records the first error
and refuses every later call. Refusing is right, because the object's state is no longer
known. The defect is what follows: for most latches, nothing. Twenty-five latches were found;
the ones cited here were re-read. Each is held by an object with one of three lifetimes: a
turn, a binding (one browser connection, built per bind at `host_launcher.py:2618-2642`), or
the host (built once by `build_local_host_launcher`). Only the binding's latch ends its own
holder: the next audio frame raises, the receiver fails, and the transport is torn down
(`conversation/worker.py:349`, `livekit/worker.py:343-365`). A host-lifetime latch ends
nothing. The host waits on one event (`host_launcher.py:3305`) with three setters, namely a
qualification channel failure, SIGTERM off Windows and SIGBREAK in qualification mode (`:3182`,
`:3266`, `:3272`), so no product latch can reach it.

| Domain (holder) | Failures that land here today | The user hears or sees | Survives | Assigned wrongly |
| --- | --- | --- | --- | --- |
| Speech chunk (one ledger entry) | None ends only a chunk. A synthesis error or a delivery confirmation timeout (`livekit/playback.py:711-720`) fails the turn, then the binding | Speech stops; the page shows the turn interrupted | Confirmed chunks; unconfirmed text is discarded (`conversation/streaming.py:2409-2416`) | Wider than needed by two levels |
| Turn (one response) | Exact `ModelOutputRefused`, twice running at most (`conversation/worker.py:1179-1184`, `:50`) | Silence for that turn; no sentence | Everything | Codex tool-argument mistakes and `OllamaDecisionRefused` are model mistakes that end the binding ([#478](https://github.com/sushiHex/hermes-realtime/issues/478) item 1, deliberately) |
| Task (one accepted run) | A rejected dispatch (`integration/api.py:721-728`); a failed run | A fixed sentence that the task was not started (`conversation/task_proposals.py:39-42`); "I ran into a problem with that work." (`conversation/result_projection.py:137`) | Everything | None |
| Session (one binding) | Any error in the audio path (`conversation/worker.py:589-591`); any response error but the exact refusal, and the third refusal (`:1187-1188`); peer loss | "Speech worker disconnected. Select Connect to retry." (`web/src/connection-recovery.ts:50`, `:81`); nothing is spoken | Heard context, tasks, every host-lifetime object | A rebind rebuilds only the worker and its detector. It reuses the model adapter, recognizer, synthesizer, speech loop, update executor and task controller (`host_launcher.py:2316`, `:2554`, `:2597`, `:2647`), so a fault that began in one of those is given a domain whose restart does not rebuild it |
| Host (one process) | Start refusals only: a held tail lock (`integration/voice_tail.py:1319-1322`), a malformed run record (`integration/api.py:531-534`), a failed preflight | Nothing; reconnecting fails for good | The saved conversation and the run record | Nine latches leave a live host that cannot do its job (below) |
| Owner (process tree) | Any child exit (`scripts/preview_cutover.py:2434-2438`) ends the run as `refused` (`:2452`, `:2522`, `:2552`); owner death closes the Job (`scripts/rehearse_desktop_mvp.py:503-526`) | A dead page until an operator starts a run | The run directory's files | A host exit is treated as a tree failure, and nothing restarts |

**Live hosts that cannot do their job.** Each is a latch whose holder outlives it.

1. **One dispatch exception ends all conversation, not only work.** The task controller
   latches (`conversation/tasks.py:462-467`, `:725-729`) and queues the error. The update
   director takes it from `next_completion` and latches (`conversation/updates.py:601`,
   `:655-659`); the update executor takes that and latches (`conversation/actions.py:744`,
   `:775-777`). Every later turn then fails at `reserve_response` (`:349`, from
   `conversation/worker.py:1131`), and every new binding fails at `start` (`:332`, from
   `conversation/worker.py:317`). One unanswered resend (`integration/api.py:882-886`) is enough.
   Reproduced by execution in [#530](https://github.com/sushiHex/hermes-realtime/pull/530):
   after the failed dispatch an ordinary turn is refused before any inference and nothing is
   spoken, and a rebind fails and leaves no binding. It is not silent: the command path
   shows one `uncertain` task state and prints one `[task-command-settlement]` line. Nothing
   says the conversation has ended, and nothing exits.
2. The work-control health is per host (`host_launcher.py:2326`,
   `conversation/work_tools.py:684-686`, `:1014-1016`). The user is told to "Restart the
   session" (`conversation/commands.py:363`), which cannot clear it.
3. A dead Codex child is never replaced: start-up returns while a transport is held
   (`providers/codex_app_server.py:1899-1901`), and a failed send closes the adapter for the
   process (`:2137-2154`, `:1797`). The preview's foreground model is Codex
   (`scripts/preview_cutover.py:2097`). Reproduced in #530: the next turn fails at its first
   send with one `[codex-send-bound]` line, every later turn is refused, and the transport
   factory is never called again.
4. The managed speech worker launches once (`providers/kokoro_mcp.py:409-413`); after a fault
   the client asks for a second launch (`providers/kokoro_worker.py:459-461`, `:492-493`,
   `:509-510`) and is refused on every later synthesis.
5. The verifier peer that confirms delivery joins once (`launcher.py:223-224`,
   `host_launcher.py:2472-2481`) and closes for good on a disconnect
   (`livekit/adapter.py:184-187`, `:231-232`).
6. A failed rebind rollback refuses every new session (`client/session.py:957-958`, `:2024`).
7. In one-use mode a lapsed browser lease closes the page server and returns
   (`client/runtime.py:389-393`); the process then waits with nothing to serve.
8. A failed model re-selection closes the selection owner (`providers/ollama_session.py:224`).
9. The tail writer serializes outside its retry (`integration/voice_tail.py:1498-1500`). If
   that raises, saving stops with no evidence until close.

## Decision

**A latch ends its holder.** A failure handler answers one question: *what state is no longer
known, which object owns it, and what ends that object?*

1. **Domain.** After a failure, name the state that is no longer known-good and the object
   that owns it. The failure's domain is that owner's lifetime, never narrower.
2. **Latch.** A latch is never reset. Tripping it ends its holder (the turn fails, the binding
   closes, the host exits) and leaves one evidence line. The next larger owner builds a fresh
   holder from state that survived.
3. **Readiness.** A dependency is *derived at use* (no fact about it outlives the call, so its
   failure belongs to the caller's domain) or *leased* (one owner, an expiry, and a lapse
   fails in the owner's domain). A stored "ready" that is neither is refused.

When no later decision reads the failed component's state, the holder is that component
alone: it ends by itself and says so. The archive fence (`integration/voice_archive.py:13-15`,
`:239`), the review sender, the memory receiver and evidence capture are such leaves today, and
stay so. Declaring a leaf is the review point: the claim is "nothing between the user's speech
and a reply, a start, an announcement or a saved row reads this".

**The rule reproduces what is right today.** An exact refusal touches nothing beyond its turn
([#476](https://github.com/sushiHex/hermes-realtime/pull/476) proved the cleanup), so the
domain is the turn. A rejected dispatch leaves nothing unknown, so the task. A malformed run
record is host state, so the host refuses to start. A failed tail write leaves the old file
whole (`integration/run_record.py:41-56`) and memory whole, so nothing is unknown, no holder
ends, and retrying is correct (`integration/voice_tail.py:1509-1518`). The work-control health
is not resettable (`conversation/work_tools.py:74`) and the rule never resets it.

**It corrects the wrong assignments, and every correction widens.** The nine cases above each
end the host. A host exit then stops being a tree failure: the host's state is rebuilt from
the saved conversation and the run record, and the gateway and LiveKit hold nothing unknown,
so the owner restarts the host alone (decision 7 of ADR 0006).

**It disagrees with the earlier lifecycle review in one place.** That review proposed
narrowing a dispatch transport error to the task. What is unknown after it is whether Hermes
started a run. The only owner that can settle that is the restart settlement, which replays
the stored key and stops what it finds (`integration/api.py:584-604`,
[ADR 0003](0003-hermes-owned-conversation-continuity.md)). The domain is the host. Today's
latch has the right scope and no exit.

**It leaves two assignments wider than needed, deliberately.** Chunk failures and model
mistakes other than the exact refusal still end the binding. The rule states what a narrowing
would have to show: every piece of state the failure touched is owned by a turn-lifetime
object. That is not true while the Codex thread lives in a host-lifetime adapter.

### Why this does not shadow an integrity check

The four holes found in [#482](https://github.com/sushiHex/hermes-realtime/pull/482) had one
cause: a verdict chosen from *what the error says*, at a point *inside* the chain, so that a
narrow verdict returned before later checks ran, or a first-error slot dropped a wider one.

- The domain is a property of the holder, fixed where the holder is constructed. No code
  inspects an error to choose a domain, and the chains raise exactly as they do today.
- This record narrows no error. Each change turns a latch into an exit, or is made by the
  process owner from an exit status.
- Domains only join upward. Two failures in one operation take the wider domain; an exception
  group stays terminal for its holder, as #476 left it.
- The exit keeps the first category only for its evidence line. Every escalation has the same
  consequence, so a dropped later one loses a diagnosis and never a refusal.

### Escalation: a crash that saves the conversation first

One latch type lives in `conversation/` and does no I/O. It cannot be constructed without the
function that ends its holder. The composition root is the only constructor of host-domain
latches and passes them one `escalate(category)`. In the host process `escalate`:

1. arms a timer on its own thread, not the event loop, that hard-exits at a fixed bound;
2. keeps the first category;
3. awaits the tail writer's final write, which is already bounded
   (`integration/voice_tail.py:1463-1485`);
4. prints one line and exits the process with one reserved status, distinct from the statuses
   in use (0, 1, 2, 3, 4 and 130; `host_launcher.py:3403-3420`, `:3774-3780`).

Component close paths do not run. They are ordered chains over objects now in an unknown
state: the executor's close re-raises its own latch (`conversation/actions.py:1109-1111`), so
an orderly close after a latch already ends as an unhandled error. An orderly close also stops
every run and writes the record empty (`integration/api.py:1148-1167`), after which the next
start has nothing to report (`host_launcher.py:1433`) and the user is never told that their
task was stopped. The crash path keeps the record; the restart stops the runs and says so
once. No interpreter or native teardown runs either, which is where
[#24](https://github.com/sushiHex/hermes-realtime/issues/24) observed an access violation.

- **What survives.** The saved conversation, by atomic replace under its lock; step 3 failing
  costs at most the last coalesced snapshot. The run record, written before the request
  (`integration/api.py:711-713`, `:720`). The kernel releases both locks. Tasks are stopped at
  restart, never resumed. For a turn in flight, only confirmed chunks are in the context, so
  what was heard is what is kept, and the open row is saved as interrupted (ADR 0003).
- **The owner.** It holds each host incarnation in its own kill-on-close Job inside the tree's
  Job. On a host exit it closes that Job, which ends the Codex child and the speech worker.
  If the status is the reserved one and budget remains, it starts a host against the same run
  directory, gateway and LiveKit. On any other status it tears down as today.
- **No restart loop.** The budget is a fixed count for the owner's run and is not refilled by
  time, since a refill would need a stored "served well" fact. A restarted host that fails to
  start exits with an ordinary status. A new host must acquire the tail lock: the lock, not a
  process ID, says the predecessor is gone, and a hung predecessor means no restart.
- **The user.** The page shows the worker as disconnected or updates as unavailable. The
  preview's launch address is stable (`scripts/preview_cutover.py:2111`,
  `host_launcher.py:3143`). On the next connection the conversation is restored heard-first,
  and if work was running the host says "I restarted, so background work from before will not
  resume." with its counts (`host_launcher.py:1431-1441`). With no work running it says nothing.
- **Evidence.** The host prints `[host-exit]` with a category from a closed set and the flush
  outcome (`written`, `failed`, `timeout`). The owner prints `[host-restart]` with the
  category, the outcome (`restarted`, `budget`, `refused`) and the remaining count, and its
  result receipt carries restarts by category. Counts and categories only, one line per exit,
  per the development guide. The names are proposals, owned by the marker catalogue. This is
  compatible with, and does not depend on, the contract proposed in
  [#517](https://github.com/sushiHex/hermes-realtime/pull/517).

Escalation is not cancellation. The four cancellation scopes stay independent and unchanged.

### Readiness

Start-up receipts stay as evidence that something was ready at a time. They carry no authority.

| Dependency | Rule and owner | Today | Not ready at use |
| --- | --- | --- | --- |
| Hermes API | Derived at use. The accepted acknowledgment is the readiness; the web and lookup generations are already read per dispatch (`integration/api.py:621`, `:707`) | Correct, apart from the missing exit | Refused: the fixed not-started sentence. Unknown outcome: the host exits, and the restart says it could not confirm whether a task started |
| Background model | Derived at use by Hermes. The host holds no fact about it. Residency is a lease owned by whoever starts the model server, expiring with that process (`scripts/lookup_backend_model.py:559`) | Nothing in `src/` or the preview owner sends a retention value or reads residency. A start-up receipt outliving the model is the [#358](https://github.com/sushiHex/hermes-realtime/issues/358) defect: a second owner | The dispatch is accepted and the run fails: "I ran into a problem with that work." That is truthful |
| Foreground Ollama | Leased by the host and renewed by each request, never unbounded on a server the host does not own | No retention value is sent (`providers/ollama.py:433-461`); the server default applies | A lapsed lease reloads inside the request bound, or the turn fails and the binding ends |
| Codex child | Derived at use, where login expiry already is (`providers/codex_app_server.py:1895-1901`): a live process and a live reader | Only the login is re-derived | The child holds the thread, and its owner is host-lifetime: the host exits |
| Speech worker, direct | Derived at use (`providers/kokoro_worker.py:487-493`, `:818-833`) | Correct | The turn fails; the next synthesis relaunches |
| Speech worker, managed | Leased by the host: GPU residency that another party can evict. One launch per host is kept | The lapse is a refusal with no exit | The host exits; the new host launches once |
| LiveKit | Per-binding peers are derived at use (`client/runtime.py:179-197`). The verifier peer is leased by the host | The verifier is a stored fact | The binding ends; a lost verifier ends the host |
| The owner's view of its children | Derived from their exit, never from a probe | Three process IDs are watched (`scripts/preview_cutover.py:2434-2438`) | See escalation |

The owner's loop is correct once every host-wide fault is an exit. A health probe in the owner
would be a second owner of each readiness fact. A gateway that is alive with a dead API is
found by the host at dispatch. This record adds no spoken sentence: a host that cannot think
or speak cannot say so, and the page and the restart announcement carry it.

## What this does not make impossible

- **A fault in the owner.** If it dies, the Job ends the tree, the conversation survives and
  nothing restarts it. If it hangs, nothing notices, and a host exit is never followed by a
  restart.
- **A fault during exit.** A hung flush is cut by the timer and a failed one loses the last
  snapshot. A process too damaged to run the timer thread remains a live host that cannot work.
- **A dependency that is alive but wrong.** Derivation at use detects absence, not wrongness,
  unless the use carries an attestation (an exact run ID, a readiness generation bound into the
  run, a worker's ready frame). Ollama's context contract is checked once and cached
  (`providers/ollama.py:271-278`).
- **A restart into the same fault.** It is bounded, not prevented: the user is disconnected
  once per restart until the budget ends the run.
- **A component that is wrong or stuck without raising.** No latch trips, so nothing exits.
- **A host that cannot save.** A tail write that fails forever is retried forever with a log
  line only. Exiting would turn a possible loss into a certain one.
- **Unsupervised runs.** Between the exit and the restart settlement Hermes runs continue;
  without an owner they continue until the next start, as after any crash today.
- **A misdeclared leaf**, and **a binding latch that waits** for the next audio frame.
- **Model mistakes that end the session.** Unchanged.

## Migration

Each step ships alone. "After" means after desktop MVP acceptance
([#159](https://github.com/sushiHex/hermes-realtime/issues/159)). "Now" marks what the owner's
instruction of 2026-11-13 moved ahead of acceptance, limited to the two faults of
[#531](https://github.com/sushiHex/hermes-realtime/issues/531): the dispatch latch with its
director and executor, and the dead Codex child. The remaining host latches follow after.

| # | Step | Label | RED test | When |
| --- | --- | --- | --- | --- |
| 1 | A ratchet test listing every latch with its holder's lifetime, bound by equality | Preserving; test only | A new latch, or a stale entry, fails it | Before |
| 2 | One evidence line when a host-lifetime latch first trips, and one per tail write-failure episode | Preserving; adds lines only | Trip each with a fake: exactly one line. Mutation: drop the print | Before |
| 3 | One latch type that requires its terminator; today's latches re-expressed with today's outcomes | Preserving | The existing suite, plus construction without a terminator refused | After |
| 4 | `escalate`: host-domain latches exit the host with the reserved status | Change | Per latch: exit status, one line, the tail equals the heard rows, the record still holds the pending entry. A hung flush still exits. Mutations: drop the timer; run the orderly close | Now for #531; the rest after |
| 5 | The owner holds a Job per host incarnation and restarts in place within a budget | Change | Reserved status: the new host takes the lock, restores, announces stopped work. Budget spent, another status, or a held lock: torn down as today, with the category | Now, once [#496](https://github.com/sushiHex/hermes-realtime/issues/496) lands |
| 6 | Readiness at use: Codex liveness, verifier loss and one-use lease expiry end the host | Change | One fake per dependency that dies after preflight | Now for the Codex child; the rest after 4 |
| 7 | A retention lease on foreground Ollama requests | Change; request bytes differ | The golden request matrix of ADR 0006 Move 2 shows the one new field | After |
| 8 | Correct "Restart the session", which a session restart cannot honour | Change | The sentence no longer promises it | Now, with 4 |

Only steps 1 and 2 are stricter-only evidence. Until the state root of #496 lands, a host
that exits leaves a `refused` receipt, and `finalize` does not carry a conversation from such
a run (`scripts/preview_cutover.py:1515-1522`), so step 4 ahead of #496 trades a live host
that cannot converse for a stopped preview whose conversation is not carried forward. Step 7 loads a model, so its real-path check
goes through `vram-mcp` as the development guide requires.

## Owner decisions

Each is yes or no; the recommendation is yes throughout. Decisions 1 to 4 were decided yes by
the owner's instruction of 2026-11-13 on #159, for the purpose of fixing #531.

1. Adopt "a latch ends its holder" as the test for failure handling: no latch reset, no error
   reclassified to choose a domain.
2. An escalated exit is a crash that saves the conversation first. Component close paths do
   not run, and runs are stopped by the restart, not by the exit. The cost is runs unsupervised
   for the length of a restart.
3. A dispatch whose outcome is unknown stays host-wide and is not narrowed to the task.
4. A dead Codex child, a lapsed managed speech worker and a lost verifier end the host. None
   is respawned in place.
5. The restart budget is a fixed count per owner run, three, with no refill by time.
6. The owner holds each host incarnation in its own nested Job.
7. No new spoken sentence for a dependency that is not ready at use.
8. Foreground Ollama requests carry a finite retention value, never an unbounded one.
9. The leaves (archive, review, memory readback, evidence capture) keep ending alone.
10. Steps 1 and 2 may ship before acceptance; everything else waits.

## Not verified

- Cases 4, 5 and 7 are read from source and were not exercised. Cases 1 and 3 were exercised
  in #530 with these limits: the dispatch failure was driven through the typed command only,
  not a natural start or the audio path; what the browser then shows is read from source; and
  a Codex reader that dies while the child still accepts writes was not driven.
- The latch inventory came from a delegated read. The rows cited here were re-read; the leaves
  (review sender, memory receiver, evidence capture) and the companion-side latches were not.
- That case 9 is reachable. The serializer was not read for what it can raise.
- What the browser shows first when a host dies, that it never reconnects by itself after a
  worker departure, and that a page can start a session against a new host at the same address.
- Nested Jobs under the owner's flags, and that closing one ends the managed speech worker,
  whose parent is the resident GPU coordinator's server. What that coordinator keeps after a
  hard-killed host is unknown.
- The #358 retention figures. No five-minute value exists in this repository; the report
  describes a private helper.
- Ollama's default idle unload, any Codex app-server behaviour, and whether a Codex child
  exits when its parent does.
- Which fixed sentence a rejected dispatch produces, between the two at
  `conversation/task_proposals.py:39-42`.
- No test, model, backend or voice session was run for this record.
