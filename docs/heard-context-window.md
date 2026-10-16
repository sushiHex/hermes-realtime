# Heard context window

The foreground model sees a bounded window of the conversation as it was heard. A restart
restores exactly that window from the voice tail. This page sets out what a row is, how
much the window keeps, and why the numbers are what they are.

## One heard row per turn

A reply's heard row is one per turn, not one per sentence. Providers stream a reply as
several publications (Ollama and Codex both split at sentence ends), and every publication
used to open its own row. A few multi-sentence replies then evicted the user's earlier
statement, and a restart restored only the rows that were left
([#210](https://github.com/sushiHex/hermes-realtime/issues/210)).

- **The key is per turn.** The speech loop creates one row key when a turn starts. Each
  confirmed chunk upserts the turn's heard text into that one open row: the exact heard
  slice of each publication, with one space between publications.
- **Delivery accounting is unchanged.** Only transport-confirmed chunks enter the row, and
  only after the ledger confirms them. A chunk that cannot be located in its publication
  adds nothing.
- **The row closes once.** It closes when the turn completes, or it is flagged interrupted
  once when the turn ends any other way.
- **Long replies roll over.** If the next chunk would take the row past `max_item_chars`, the
  row ends at the previous chunk and a new row starts at this one. Pushing the new row closes
  the old one, so an interrupted long reply flags only its last row. A rollover never
  raises, because a publication is already admitted only within `max_item_chars`
  (`validate_assistant_generation`), so one chunk always fits a fresh row.
- **A crash cuts off the open row.** The durable view flags the open row interrupted, so
  after a restart that turn is one archived, flagged utterance. A replay (`resume_interrupted`)
  is a new turn: its newly heard text is its own row.
- **Word-boundary stops don't change rows.** [#213](https://github.com/sushiHex/hermes-realtime/pull/213)
  plays queued audio to a word gap, but the ledger still confirms whole chunks. A row holds
  exactly the delivered chunks.

The archive identity `voice:<conv>:<gen>:<seq>` keeps its scheme. `seq` counts utterances:
one per turn, or one per row of a reply that rolled over. A reply archives when its row
closes, and archived rows keep their numbers.

## The window is a text budget

The window evicts its oldest rows once their text exceeds `max_window_chars`.
`max_messages` stays as a hard row cap. Retention therefore depends on how much was said,
not on how a provider split it.

| Bound | Default | Why |
| --- | --- | --- |
| `max_item_chars` | 1,024 | One row's text; unchanged. |
| `max_window_chars` | 22 × `max_item_chars` = 22,528 | One full user row and one full reply row for each of 10 turns, plus the question being asked and a restart announcement. |
| `max_messages` | 32 | Hard cap: 22 full rows, plus room for short rows, rollovers and announcements. |

- **The lower bound is one review interval.** M2 reviews every `memory.nudge_interval` user
  turns, which is 10 by Hermes's default. With every row at full length, ten complete turns
  still fit after a restart announcement and the next question; a test holds this. Twenty
  rows alone would fill the budget exactly, and the next row would evict turn 1. A reply that
  rolls over uses more of the budget.
- **Older turns live elsewhere.** M4 memory is the long-term store the foreground sees, and
  the M1 archive keeps every row. The window is the short-term conversation, not the history.
- **The window holds at least two full rows** (`max_window_chars >= 2 × max_item_chars`). A
  row closes when the next row is pushed, so it always survives that push long enough for
  the tail to give it an identity.
- **The restart announcement is a row like any other** and counts against the budget.

### Restoring tails across versions

- **Upgrade.** An older build's tail had at most 16 rows of at most 1,024 characters, which is
  16,384 characters, within the 22,528-character budget. It restores whole, with its
  per-sentence rows as they are, and those rows age out as new rows arrive. `seq` continues
  from the tail's `next_seq`, one per turn. Review state it froze also restores (below).
- **Downgrade hazard.** An older build refuses a tail of more than 16 rows as malformed. It
  also refuses review rows that carry a byte cost, and the retained `recent` rows (below). In
  every case it starts a fresh voice conversation. The restored window is lost, and so are
  the outbox rows not yet acknowledged by the archive; the archive keeps only what it had
  already acknowledged. Don't run an older build against a tail this version wrote.

## Ollama `num_ctx`

The adapter sends `options.num_ctx` (default 16,384) with every request. Without it, the
server's default applies, and that depends on the machine.

### Model capacity and session budget

A model's advertised maximum, the requested session budget, and its observed runtime
allocation are different values. Model metadata supplies the first; configuration chooses
the second; the running backend may report the third. None establishes recall quality or
tool-use reliability. A missing observation stays unknown.

The full host accepts `--ollama-num-ctx <tokens>` for an explicit session choice. It reads
the selected installed model's metadata before warmup, checks the requested value against
the advertised maximum when known, and observes the loaded allocation after warmup. The
choice is frozen before inference; restart the host to change it. The supported request
range is 2,048–262,144 tokens. This application bound does not redefine a model's advertised
maximum. Without the option, the existing 16,384-token configuration remains in place.

For browser selection, launch the Ollama host with `--local-model-selection` instead of
`--ollama-num-ctx`. Before connecting, choose an installed model and enter a bounded context
budget. Discovery reads metadata without loading a model. Preparing the choice checks its
digest again and runs the host's inference and speech readiness checks before Connect can
establish a voice session. Serving the configuration page does not mean voice is ready.
The installed-model list is a host snapshot; restart after installing or replacing models.

A successfully prepared choice stays fixed while its browser session is active, including
a dropped network connection and reconnect. Before joining, or after a successful explicit
session Disconnect on a persistent launch, choose **Change model or context** to edit it.
The form retains the last requested budget; Cancel keeps the existing prepared choice.
Connecting with an edited choice prepares a fresh adapter and checks readiness before
replacing the old adapter. Merely editing the form does not release the old choice.
Foreground and proactive action admission pauses only during that transaction; durable
Hermes jobs and stored history remain independent. Busy actions refuse preparation.
Preparation does not issue browser session credentials; the existing bootstrap authority
still governs connection. A consumed one-use launch is never renewed by model selection;
use the persistent loopback/Tailnet launch for successive sessions, or start a fresh host.
Local-selection bootstrap sends the exact versioned prepared snapshot and checks it under
the same session lock before issuing credentials or consuming a one-use capability. A stale
tab cannot join a different tab's replacement choice. Older local-selection browser clients
that send empty bootstrap requests fail closed; ordinary non-selection bootstrap is unchanged.
Waiting for that lock uses the existing ten-second session transition bound; the existing
provisioning budget is separate, so this is not a guarantee against every late response.
An expired one-use launch requires a fresh launch, even if preparation completed. The browser
reports advertised capacity, requested context and observed allocation separately; unknown
values remain unknown. Choosing a model does not qualify it for automatic task delegation.
Preparation has a 120-second budget and a separate three-second cleanup allowance. If a
candidate cannot be cleaned up, the host refuses further choices and retains that candidate
for cleanup retry; restart the host after resolving the failure. It never accumulates new
candidate adapters behind a failed cleanup.
If a replacement candidate has been created and its preparation fails, the host also fails
closed and requires restart. The candidate may have changed Ollama's resident allocation;
retaining the old adapter object cannot establish rollback readiness. A refusal before
candidate creation leaves the existing choice untouched. After an unanswered or refused
replacement request, the browser refreshes bounded host status and accepts only the
selection the host confirms; it never restores readiness from a cached form snapshot.

This control does not fit prompts automatically or change the retained foreground window.
An oversized request still fails explicitly under the no-truncation contract below.

Every Ollama adapter requires a source-qualified Ollama runtime, 0.34.4 or 0.40.4, before
its first prompt, including default and legacy launcher paths. Model metadata that declares
non-GGUF weights or a runner other than llama.cpp is refused wherever it appears, and the
installed-model catalog leaves such models out. Successful validation is cached
for that adapter; later turns do not repeat the metadata read. Recreate the session after
replacing or upgrading the backend. Unknown runtimes are refused rather than assumed to
honor no-truncation flags. Explicit context configuration supplies
a versioned browser report showing the advertised, requested and observed values separately;
an unavailable allocation remains unknown. Older browser clients do not support this new
report. Metadata identity is checked during discovery and allocation observation. Ollama
resolves mutable model names at inference, so concurrent external replacement of that name
is unsupported; these checks do not provide an atomic model lease.

Smaller models and smaller context budgets are valid choices for constrained applications.
They do not require model-name exceptions or allocation of the advertised maximum. A model's
original positional-scaling threshold is also not its advertised maximum. Qualification
belongs to the particular model artifact, runtime and configuration, separately for ordinary
conversation, task delegation and latency.

The foreground text window and durable storage have separate responsibilities. Choosing a
smaller inference budget does not delete the Hermes archive or memory. Conversely, stored
history is not necessarily present in every model request. The text-window limits above are
not exact token counts: templates, tools, memory, policy text and the generated reply also
consume context. An explicit capacity refusal is preferable to silently dropping those
sections. No character-to-token ratio proves that a request fits.

**Measured on Ollama 0.34.4 here.** These tests used `phi4-mini`, a 24 GiB GPU and no
`OLLAMA_CONTEXT_LENGTH`.

- **The default context loaded at 32,768.**
- **A 10,052-token prompt was silently truncated.** With `num_ctx` 4,096 or 8,192, Ollama
  evaluated 4,026 or 8,034 tokens. It dropped older conversational rows and returned no error, and
  the early fact went unanswered.
- **Smaller GPUs get a smaller default.** Ollama picks the default by available VRAM, so the
  same prompt can be truncated on a smaller GPU.

**What Ollama preserves by default.** At the measured version's immutable source commit,
[`chatPrompt`](https://github.com/ollama/ollama/blob/b2da9e468af2479058ae18c6d908ed29de410684/server/prompt.go#L20-L86)
keeps all system messages and the latest message while trimming older conversational rows.
The leading policy/reference message therefore survives that message-level trimming.
If the system messages plus the latest message still exceed the effective context, they
reach a separate
[token-level truncation path](https://github.com/ollama/ollama/blob/b2da9e468af2479058ae18c6d908ed29de410684/llm/llama_server.go#L282-L321).
With context shifting enabled, that path preserves only the configured prefix token count
and a suffix; it can cut policy or reference text inside the rendered prompt. This extreme
overflow limit also applied to the previous separate system messages. Request assembly
does not establish which complete sections the model evaluated.

**Explicit overflow refusal.** The adapter sends top-level `truncate: false` and
`shift: false`. `truncate: false` disables the message-trimming loop in `chatPrompt`, and
Go then skips its own token-level truncation. An oversized prompt therefore reaches
llama-server, whose own context check refuses it with HTTP 400 whether or not context
shifting is on. At the pinned 0.34.4 GGUF source, the
[`Shift` request field](https://github.com/ollama/ollama/blob/b2da9e468af2479058ae18c6d908ed29de410684/api/types.go#L160-L166)
sets the runner's context-shift configuration. `shift: false` leaves out `--context-shift`,
which prevents generation-time token discarding. Neither flag cuts the rendered
policy/reference prompt. `LLAMA_ARG_CONTEXT_SHIFT` in Ollama's environment could still turn
shifting on, so it must stay unset. The adapter
propagates an opening refusal without a reply or automatic retry. This is source-based
qualification. An isolated paired-options check at realtime commit `43b28a7` on Ollama
0.34.4 completed a 372-token synthetic prompt and refused a 6,138-token prompt at a
4,096-token allocation with HTTP 400 before any content or spoken segment. The oversized
case used 13 intact rows within the host's row and text-window limits. This verifies that
case's refusal, not every backend or model. The cold positive took 64.716 seconds under a
120-second diagnostic bound, exceeding the host's unchanged 60-second budget; it is not
host-readiness or voice-latency qualification.
The [source-bound result and limits](https://github.com/sushiHex/hermes-realtime/issues/249#issuecomment-6115056437)
remain in the owning issue.
Ollama 0.40.4 was admitted after a
[source comparison](https://github.com/sushiHex/hermes-realtime/issues/282#issuecomment-6206646192)
and [paired witnesses](https://github.com/sushiHex/hermes-realtime/issues/282#issuecomment-6226666160).
Both versions gave the same `prompt_eval_count`, the same HTTP 400 on overflow and the same
trimmed count in a truncate-omitted control. The error text differs between versions, so
nothing may key on it. Other versions and native-chat, MLX or cloud model paths remain
unqualified. Disabling shifting
can also end generation at the context limit; it does not reserve reply tokens or prove
model obedience. A change to this flag can
[`reload an already loaded runner`](https://github.com/ollama/ollama/blob/b2da9e468af2479058ae18c6d908ed29de410684/server/sched.go#L1422-L1427).

**The derivation.** These are the steady-state prompt parts, in characters. The Ollama
adapter now prepends one shared communication-policy message with labelled reference JSON,
followed by the unchanged ordered conversational rows. Reference text adds no work authority.

| Part | Characters |
| --- | --- |
| Heard window | 22,528 |
| Interrupted delivery indices, JSON wrapper and index/authority framing | at most 500 |
| M4 memory, two blocks of at most 4,096 UTF-8 bytes | 8,192 |
| Memory label and JSON wrapper | about 150 |
| Foreground communication/style policy and work-state framing | about 2,200 |
| **Total** | **about 33,600** |

- **Delivery metadata.** At most 32 zero-based indices for the bounded heard window are
  carried in labelled system reference JSON. The wrapper and all indices take 139 characters;
  the index/authority explanation takes 295, so 500 is a conservative allowance. Dialogue text
  stays unchanged, and the heard-window limits remain unchanged. A model may still verbalize
  metadata; this representation is not a guarantee of model obedience.
- **Tokens.** At an estimated 3 characters per token, that is about 11,200 tokens. The chat
  template adds about 6 tokens for each of 33 messages (about 200), for about 11,400 in all.
  For comparison, English filler measured 5.5 characters per token here.
- **Estimated headroom.** `num_ctx` 16,384 leaves about 5,000 tokens for active-task and update reference
  sections and for the reply.
- **What it doesn't cover.** Every bound at once, such as 8 maximal task objectives plus 16
  maximal updates, would not fit. Neither would text in a script denser than 3 characters per
  token, which may need a larger `num_ctx`.
- **The marker.** Every completed request prints `[ollama-prompt]` with the prompt's message
  count, characters and UTF-8 bytes, the `num_ctx` requested, and Ollama's
  `prompt_eval_count`, plus the captured output-style enum and actual rendered reference-section
  kinds (`memory`, `active_work`, `updates`). It carries no text. The desktop rehearsal records
  those kinds for each context check; one combined system message is no longer a count of notes.
  Conversation-context equality evidence remains separate and does not bind the selected style.
- **It is a request, not a guarantee.** The server may cap it, for example at a model's
  trained context length, and the marker shows only what was requested. So the evidence is
  `prompt_eval_count`: a count near the context the model actually ran with means the prompt
  may have been truncated. The count does not prove retention of any particular row or
  reference section. This removes the hardware-dependent default; it can't stretch a model's
  context.
- **The value overrides the model's own.** An explicit `num_ctx` also overrides a Modelfile's
  value, such as a `-32k` model variant's. The window estimate targets 16,384; it is not an
  exact token bound or a guarantee for every supported model and input.

## Codex

The Codex adapter sends the whole snapshot as one prompt. About 30,000 characters is roughly
10,000 tokens. The default model, `gpt-5.6-terra`, reports a context window of 272,000 tokens
in the local Codex model catalog, so the snapshot is under 4% of it. Nothing changes for
Codex.

## Review windows within the companion's byte budget

The companion admits a review window only if its snapshot has 1 to 24 rows and its JSON is
at most `MAX_REVIEW_TOKENS` = 16,384 bytes, the smaller of its two byte bounds
(`MAX_REVIEW_BYTES` is 65,536). That one check, `review_snapshot_admitted`, lives in
`companion/review.py`, and the Hermes adapter applies it. Otherwise the refusal is
`window`, which is not transient, so the sender stops reviewing that conversation. That is a
deliberate fail-closed choice. A 24-row limit was enough for per-sentence rows. It is not
enough for per-turn rows, so the tail now chooses every window within the byte budget, and
a `window` refusal can no longer be reached.

There is one rule for every window: it is trimmed to the budget from known per-row costs, and
it always holds at least one row.

- **Each row carries its cost.** When an archive batch is acknowledged, each row enters review
  bookkeeping with its exact snapshot cost: `[seq, is_user, bytes]`. The cost is capped one
  byte past the budget.
- **Periodic and closing windows.** A window is the longest prefix of the eligible rows (at
  most 24) whose snapshot fits. A closing window is final only when it covers every eligible
  row; otherwise it is an ordinary window and the close follows.
- **The tail keeps the last reviewed rows.** When a review is acknowledged, the tail keeps the
  costs of the last reviewed rows, at most 24, ending at the review cursor (`recent`).
- **An empty close replays what fits.** A close that finds no unreviewed row replays the
  longest suffix of those rows that fits, and then spans the empty gap to its checkpoint.
  Parsing recomputes that start from the retained costs and refuses any other, so the format
  checks itself.
- **One maximal row always fits.** At the default `max_item_chars` of 1,024, the worst row is
  1,024 control characters, each a 6-byte escape: 6,180 bytes, plus the list brackets. A test
  holds this for control, astral and CJK text. It is why "at least one row" can never build a
  window the companion refuses. A store with rows over about 2,700 characters would break it,
  and the fix then is a lower `max_item_chars`.

Tails written before this change still parse, so the tail version is unchanged. They keep
their conversation and fail closed:
- **Rows without a cost** are costed as the largest row the store can hold:
  `36 + 6 × max_item_chars`, which is 6,180 bytes at 1,024. They are reviewed in smaller
  windows.
- **A tail without retained rows** has a new empty close replay only the last reviewed row.
  With nothing reviewed yet, it starts at the checkpoint itself.
- **An empty-close replay an older build froze** (24 positions back) is accepted, but only
  while no retained rows exist. It is resent exactly as frozen, because the companion
  recognizes a review by its exact range; a different range would start a second review.
  Its size can't be checked, since the reviewed rows' costs were never kept. It behaves as
  it did before the upgrade.
- **A pending window an older build froze** is planned again if its costs exceed the
  budget. A window that large was refused as too large, so it was never reviewed, and
  planning it again loses nothing. The tail prints `[voice-tail-outbox]
  {"replanned":"review_budget"}` when it does.
