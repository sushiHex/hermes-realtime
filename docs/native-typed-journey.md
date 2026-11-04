# Typed native journey client

`python -m scripts.native_typed_journey` checks an already running isolated
preview through its existing browser HTTP contract. It starts no services and
adds no executor. Use a preview owned by the operator, with no other participant
or active task. The client joins its existing LiveKit room without subscribing
to audio, sending microphone audio, or retaining PCM.

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

Warm measurements require matching turn ID **and** generation. A foreground
pair is bound only after the matching typed admission and user transcript;
background/replayed speech cannot supply that pair. Missing identities remain
unknown. Same-pair token-to-playable timing measures server preparation, not
browser audibility or first nonzero PCM. The client does not infer semantic
recall accuracy, source quality, phonemes, microphone recognition, or voice
interruption from an HTTP transcript. Those remain separate acceptance checks.

Offline checks: `uv run pytest -q tests/test_native_typed_journey.py`.
