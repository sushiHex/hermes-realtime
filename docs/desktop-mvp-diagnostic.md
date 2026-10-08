# Desktop MVP: setup and diagnostic session

This guide sets up the exact profile of the Windows desktop MVP
([#159](https://github.com/sushiHex/hermes-realtime/issues/159)) and walks the operator through
the diagnostic session, step two of its plan.

**This is a diagnostic session, not the acceptance session.** Its purpose is to bring
integration findings forward. A failed step is a finding to record, not a reason to stop: record
it, then continue with the next step whenever the setup still allows it.

Record outcomes and latency observations as categories and timings only. Never record
transcripts, audio, objective text, model output, tokens, keys, launch URLs, paths, user names
or host names. The bounded `[name] {...}` markers described below are designed to be recorded
verbatim: they carry counts, kinds and categories only.

The guide links to the detailed references rather than repeating them:
[`local-livekit.md`](local-livekit.md) for the LiveKit server and the full host,
[`hermes-bridge.md`](hermes-bridge.md) for the installed-runtime gate and the bridge protocol,
and the [README](../README.md) for building from source.

## Deleting a voice conversation

Use the browser's **Delete this voice conversation** control and confirm the action.
There is no voice command for deletion: a misheard utterance must never delete history.
The control states this limit beside it:

> What Hermes learned from it (memories and skills) stays and may still shape replies. There is no unlearning in the MVP.

> Delegated tasks remain in Hermes and are managed with Hermes's own session controls.

The live tail and queued archive rows clear immediately and a new generation fences late
events. The status remains **pending** until the companion verifies that the old generation's
archive chain has been deleted. An already running review finishes normally; deletion waits
for it, and anything it learns remains. If the companion is unavailable, pending does not
mean complete. A restart or successor owner resumes reconciliation from durable intent.

Deletion is logical, not physical erasure. Hermes run records remain (terminal records are
pruned 24 hours after their last status update); unvacuumed SQLite pages and backups or sync
copies remain too. A non-terminal run record left by a crash must first be recovered by
Hermes before its terminal retention applies. Built-in memory and skills are not removed.
Completion verifies session absence in Hermes's database. Hermes removes session sidecar
files on a best-effort basis and can silently retain them if filesystem removal fails;
completion is not a verified filesystem-erasure receipt.

For the acceptance session, use synthetic data: verify that a unique phrase disappears from
the voice tail, each archive-chain session and the next conversation's history; exercise
pending deletion during a review and after restart; verify separately that a deliberately
learned fact can still return through M4. Record only results and counts, never the phrase
or memory content. See [ADR 0003](adr/0003-hermes-owned-conversation-continuity.md).
Delegated-task session retention is an explicit acceptance witness, not a deletion failure.

## The profile

| Aspect | MVP setting |
| --- | --- |
| Operator | One operator, one Chrome browser on Windows. |
| Hermes | v0.21.0 at `29112bef099274229cadff79cdff7bf7b99c4b77`, by the upstream installer. |
| Hermes profile | The default profile, selected explicitly (see setup step 2). |
| Hermes memory | Built-in memory only: no external memory provider. |
| Host | `hermes-realtime-host` on literal loopback, with the stable loopback front door. |
| Conversation | The default `legacy` profile; explicit `task:` and `cancel task:` commands. |
| Speech | The defaults: Moonshine `tiny` speech-to-text, Kokoro text-to-speech on Windows. |
| Evidence capture | Off: `--evidence-capture` is not passed. |
| Public search | Off: `--enable-public-search` is not passed. |

Before the session, record the profile you actually ran, so the session binds to one candidate:

- the hermes-realtime commit, and the wheel version printed by `uv build`;
- the Hermes version and commit, from the gate's record (session step 1);
- the inference provider and model, and the speech providers and models, as the browser's
  **Session model** panel shows them;
- the Chrome version, the microphone and speaker class (built-in, USB headset, ...), and the
  CPU and GPU class. Never record a device name that identifies a person or a machine.

## Setup

Use four terminals: **A** for LiveKit, **B** for the host, **C** for the Hermes gateway and
**D** for one-off commands. Commands are Windows PowerShell unless they say Git Bash.

### 1. Install Hermes at the exact commit

The one-line installer (`iex (irm ...)`) cannot pass a commit. Download the installer from that
commit into a temporary folder and run it as a file:

```powershell
$commit = "29112bef099274229cadff79cdff7bf7b99c4b77"
$source = "https://raw.githubusercontent.com/NousResearch/hermes-agent/$commit"
$installer = Join-Path $env:TEMP "hermes-install.ps1"
Invoke-WebRequest -OutFile $installer "$source/scripts/install.ps1"
powershell -ExecutionPolicy Bypass -File $installer -Commit $commit
```

The installer clones Hermes into `<HERMES_HOME>\hermes-agent` and checks out the commit as a
detached `HEAD`. It creates the environment at `<HERMES_HOME>\hermes-agent\venv` and sets
`HERMES_HOME` for your user, `%LOCALAPPDATA%\hermes` by default. It also puts `hermes` on your
`PATH` and runs `hermes setup` to configure a model. At the end it may offer to start the
gateway in the background; answer no, since setup step 6 starts it in a terminal. On a machine
with an existing newer install, add `-ForceCommit`, or the installer keeps the newer checkout.

Open a new terminal so the user environment applies, then confirm the commit:

```powershell
git -C "$env:HERMES_HOME\hermes-agent" rev-parse HEAD
```

It must print the commit above. The gate (session step 1) is the authority on whether the
checkout is clean. Plain `git status` is not: `29112bef` tracks paths that differ only in case,
such as spellings of a file under `contributors/emails/`, and a Windows checkout can hold only
one of each. `git status` therefore lists one of them as modified on every Windows checkout. The
gate exempts exactly those paths while the one file on disk matches one of their committed
versions, and refuses any other tracked change; see the gate in
[`hermes-bridge.md`](hermes-bridge.md#plugin-host).

### 2. Select the default profile

```powershell
hermes profile use default
hermes profile list
```

The gate expects the checkout and the `.env` under one home, `<HERMES_HOME>\hermes-agent` and
`<HERMES_HOME>\.env`, which is the default profile's layout. A named profile keeps its `.env`
elsewhere, so it is outside this MVP profile. The host itself accepts any `--hermes-env-file`.

### 3. Use built-in memory only

```powershell
hermes memory off
hermes memory status
```

`status` must report no external provider (built-in only).

### 4. Install the hermes-realtime wheel and enable the plugin

From a checkout of the hermes-realtime candidate commit (see the README's
[Install from source](../README.md#install-from-source)):

```powershell
uv sync --frozen --dev --extra local
uv build --wheel
uv pip install --reinstall-package hermes-realtime `
  --python "$env:HERMES_HOME\hermes-agent\venv\Scripts\python.exe" `
  dist\hermes_realtime-<version>-py3-none-any.whl
hermes plugins enable hermes-realtime --no-allow-tool-override
hermes plugins list --enabled
```

Substitute the wheel file name `uv build` printed. Install a built wheel, never an editable
install: the gate identifies the candidate by the installed wheel's `RECORD`. The plugin must
come from the same commit as the checkout that runs the host. Installing the wheel can change
packages it shares with Hermes in that environment; record any such change in the session notes.

### 5. Configure `<HERMES_HOME>\.env`

Edit the file in Notepad:

```powershell
notepad "$env:HERMES_HOME\.env"
```

Do not write to it with `>>` or `Out-File` from Windows PowerShell 5.1: those write UTF-16,
which the host's UTF-8 read cannot parse. Keep each key exactly once: the host
refuses a file that sets `API_SERVER_KEY` or a companion key twice. Add:

```dotenv
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
API_SERVER_KEY=<at least 32 random characters>
HERMES_REALTIME_COMPANION_PORT=<a free loopback port>
HERMES_REALTIME_COMPANION_TOKEN=<24 to 512 random characters>
```

Generate each secret with a cryptographic random generator, straight to the clipboard so it
never appears on screen. Paste it into Notepad, then clear the clipboard:

```powershell
$python = "$env:HERMES_HOME\hermes-agent\venv\Scripts\python.exe"
& $python -c "import secrets; print(secrets.token_urlsafe(32))" | Set-Clipboard
Set-Clipboard -Value " "
```

If Windows clipboard history is on, delete the entries from it too. Never paste these values
into shell history, issues, screenshots or logs. The gateway reads the whole file. The host
reads only `API_SERVER_KEY` and the two companion keys from it. Built-in memory needs no
provider keys. See [`local-livekit.md`](local-livekit.md#one-time-hermes-api-setup) for the API
server keys.

### 6. Start the Hermes gateway

First make sure no other gateway is running, for example one the installer started in the
background, or an installed scheduled task. In terminal D:

```powershell
hermes gateway status
hermes gateway stop
hermes gateway status
```

Run `stop` only if `status` reports a running gateway, and continue only once it reports none.
If `status` reports an installed service, `hermes gateway uninstall` keeps it from starting
again at logon during the session.

Then, in terminal C:

```powershell
hermes gateway run
```

It runs in the foreground; `Ctrl-C` stops it. To restart it, press `Ctrl-C` and run it again.
After `run`, `hermes gateway restart` would start a detached gateway instead and may offer to
install a scheduled task. Restart the gateway after any install or update of Hermes or the
wheel. Hermes writes its own gateway log to `<HERMES_HOME>\logs\gateway.log`; the plugin's
markers are printed on the gateway's standard output, in terminal C.

Check from terminal D that the API server answers and listens on loopback only:

```powershell
curl.exe --fail http://127.0.0.1:8642/health
Get-NetTCPConnection -State Listen -LocalPort 8642 | Select-Object LocalAddress, LocalPort
```

Every `LocalAddress` must be `127.0.0.1` or `::1`. Do not continue otherwise; this is the same
requirement as [`local-livekit.md`](local-livekit.md#one-time-hermes-api-setup).

### 7. Prepare the LiveKit server

Follow [Pinned development server](local-livekit.md#pinned-development-server) once. The server
itself is started in session step 2.

## Capturing evidence

### Markers

Bounded markers are single stdout lines of the form `[name] {json}`. They carry counts, kinds
and categories only, so you can record them verbatim. They come from three processes:

- **Host** (terminal B): the realtime host and its voice tail.
- **Gateway** (terminal C): the plugin's voice companion runs inside the Hermes gateway.
- **Gate** (terminal D): the installed-runtime gate prints exactly one line.

To keep a local copy of a terminal's output while you watch it, pipe it through `Tee-Object`
(PowerShell) or `tee` (Git Bash) into a file outside the repository, then pull out the markers:

```powershell
Select-String -Path <log-file> -Pattern '^\[[a-z0-9-]+\] \{' | ForEach-Object Line
```

The log file is private: it may hold paths and error text. Publish only marker lines and your
record sheet. Never tee a host started without `--persistent-loopback-launch`, whose one-use
launch URL is a credential. How `Tee-Object` encodes a native program's output in Windows
PowerShell 5.1 has not been checked; if the extraction finds nothing, record that and copy the
marker lines from the terminal instead.

From the gate:

- `[real-hermes-gate]`: the gate refused. Session step 1 gives the next step for each category;
  `components` folds markers other code printed during the run, such as `[hermes-identity]`
  for a modified Hermes checkout.

From the host:

- `[voice-tail]`: at start, `restored` and `outbox` counts when the conversation was restored,
  or `refusal: malformed` when the tail was unusable and a fresh conversation began. No marker
  means no tail existed yet.
- `[voice-tail-lock]`: `cause: held`, another host holds the voice tail; this one does not start.
- `[voice-tail-outbox]`: unsent archive rows were dropped (`overflow`, `evicted`, `invalid`),
  or a review or close bound was reached.
- `[voice-archive-send]`: the archive link to the companion. `outcome: unavailable` means the
  companion was unreachable; `outcome: unknown`, that the batch is resent unchanged;
  `transient: <category>`, a refusal that is retried; `fence: <category>`, a refusal that stops
  archiving for this host.
- `[voice-review-send]`: a review was refused (`refusal`), or kept for retry
  (`outcome: retained`).
- `[voice-review-close]`: at shutdown, a review close could not settle (`deadline`,
  `unsettled`, `not_open`), or had nothing to cover (`empty_window`).
- `[hermes-run-record-lock]`: `cause: held`, another host holds the Hermes run record.
- `[hermes-restart-settlement]`: at start after a crash, the runs `stopped`, the dispatches with
  an `unknown` outcome, and those `unresolved` (a failure: the start aborts); or
  `refusal: malformed`.
- `[hermes-dispatch-recovery]`: a dispatch got no complete answer and its resend did not settle
  it; `cause` says why.
- `[codex-session-auth]`: the Codex sign-in is unusable (`malformed`) or about to expire
  (`expiring`); run `codex` once.
- `[codex-tool-refusal]`: natural work tools only; not expected in this profile.
- `[speech-stop]`: one per stopped speech chunk. `mode: word` is a barge-in, which plays on to
  the next gap between words (`outcome: gap`), to the 300 ms cap (`cap`) or to the chunk end
  (`end`), with `tail_ms` of extra audio; `mode: hard` is any other stop (`immediate`).
  `faded` says whether the 15 ms fade was appended.

From the gateway (the companion runs inside it):

- `[voice-companion]`: the companion refused or failed to start (`refusal`, `failure`).
  `held` is expected from a Hermes CLI or cron process, such as a `hermes` command in terminal
  D, while the gateway owns the companion; from the gateway itself, it is a finding.
- `[voice-archive-open]`, `[voice-archive]`, `[voice-archive-lease]`: a conversation could not
  be opened, a batch was refused, or the archive lease was lost.
- `[voice-review]`: a review was refused, or ended (`outcome: finished`, `failed` or
  `cancelled`).
- `[hermes-bridge-hello]`: a hello was refused: `shape`, `token`, `participant`, `version` or
  `capability`.

The host also prints one plain line, not a marker, after a crash left runs behind:
`NOTICE: a previous session left Hermes background work behind: ...`, with counts only.
`[consent-activation]` (evidence capture) and `[qualification-checkpoint]` (qualification runs)
do not occur in this profile. A category not described here is still a finding: record it
verbatim.

### Timings

The browser's **Session model** panel shows **Response last** and **Response average**: the time
from your final transcript to the first foreground token. **Diagnostics → Latency markers** lists
named timings in milliseconds, such as `transcript_to_first_token` and `first_token_to_audio`.
For task steps, use wall-clock seconds from submitting the command to each task state.

### The record sheet

For each step, record:

- **outcome**: `as_expected`, `different` (with a one-word category), `failed` or `not_run`;
- **timings**: the values named in the step;
- **markers**: every marker line that appeared during the step, verbatim;
- **notes**: categories only.

Some behavior is not yet established and is recorded as an observation rather than checked
against a claim: whether `Ctrl-C` in Git Bash stops the host cleanly or kills it, whether the
gate and a running host interfere through the companion, and the order of the host's markers at
a restart.

Post the record sheet to [#159](https://github.com/sushiHex/hermes-realtime/issues/159).

## The diagnostic session

### 1. Run the installed-runtime gate

Run it after the gateway is up (setup step 6) and before the host and the browser start, so its
work is not mixed with the session's and a gateway restart never happens under a live host. In
terminal D, from the hermes-realtime checkout:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& "$env:HERMES_HOME\hermes-agent\venv\Scripts\python.exe" scripts\real_hermes_api_gate.py
```

- **Do:** let it finish. With a responsive model it takes a minute or two; its own waits allow
  up to about 16 minutes (300 seconds each for the completion, for the approval request and for
  the approval run to end, and 60 for the cancellation). Stopping it with `Ctrl-C` mid-run is a
  finding and may leave runs behind; restart the gateway to stop them.
- **Observe:** exactly one line. A pass is a JSON record with `"gate": "passed"`; a refusal is a
  `[real-hermes-gate]` marker. It dispatches real work, so the gateway's model is called, and
  it raises and rejects one approval.
- **Record:** the line verbatim. A pass names the Hermes version, commit and baseline, the
  candidate's version, and each behavior's status; a clean pass is also the check that the
  Hermes checkout is unmodified. For a refusal, record its `stage` and `category`, take the next
  step below, run the gate again and record both lines.

Next steps by category, from the gate's source:

- `not_install`: this interpreter's Hermes is not `<HERMES_HOME>\hermes-agent`. Run the gate
  with that checkout's `venv\Scripts\python.exe`, and check `HERMES_HOME` (setup steps 1 and 2).
- `unnamed`: part of the install cannot be named: no detached commit, no single installed wheel
  `RECORD`, a `hermes_realtime` imported from anywhere but the install's wheel, or an unreadable
  Hermes version. Reinstall at the commit (setup step 1) or reinstall the wheel (setup step 4),
  and run the gate without `PYTHONPATH`.
- `no_companion`: `<HERMES_HOME>\.env` names no companion endpoint. Add both companion keys
  (setup step 5) and restart the gateway.
- `health`: the gateway did not answer the authenticated `/health/detailed` with its process.
  Check that the gateway runs, that `API_SERVER_KEY` matches, and that port 8642 answers.
- `hello_refused`: a wrong companion token, or a gateway still running a plugin without runtime
  attestation. Check the token, then restart the gateway.
- `capability`: the companion does not offer every capability. Reinstall the candidate wheel
  (setup step 4) and restart the gateway.
- `foreign_companion`: another process owns the companion, such as a Hermes CLI or cron process
  that started first. Stop every `hermes` process, then start only the gateway.
- `restart_gateway`: the gateway still runs what it loaded before an update, a reinstall or a
  rebuilt wheel. Restart the gateway.
- `behavior`: a dispatch, approval or cancellation did not end exactly as required. This is a
  finding about Hermes or its model; record the stage.
- `left_running`: a run the gate created did not read back as terminal. Restart the gateway:
  that stops every run it holds, and Hermes reports a run whose gateway restarted as
  interrupted.
- `gateway_restarted`: the gateway's process changed during the gate. Run the gate again with
  the gateway left alone.
- `arguments`: the command line was not understood.
- `host`: only with `HERMES_REALTIME_LIVEKIT_LOCAL=1`, which this session does not set.
- `error`, with `stage`:
  - `import`: the interpreter cannot import the gate. Use the install's interpreter, and
    reinstall the wheel (setup step 4).
  - `identity`: a `RuntimeError` with `components` naming `[hermes-identity]` means the
    checkout has tracked changes (`modified`) or is not a git checkout (`unidentifiable`):
    reinstall at the commit.
  - `endpoint`: the `.env` is missing, its key is weaker than 32 characters, or a value is
    malformed (setup step 5).
  - `discovery`: a `TimeoutError` or a refused connection means nothing answered on the
    companion port. Check that the plugin is enabled (setup step 4) and the gateway was
    restarted after the `.env` change.
  - any other stage: the failure's type names it; record it.

### 2. Start the host, open the browser and talk

The host restores the previous conversation from its voice tail on start. Earlier history would
make later "context carried over" observations ambiguous, so move the old tail aside before the
first start. Moving it keeps that conversation; do not touch the Hermes run record beside it,
which lets the host stop work a crash left behind:

```powershell
$state = "$env:LOCALAPPDATA\HermesRealtime\state"
if (Test-Path "$state\voice-tail-v1.json") {
  Move-Item "$state\voice-tail-v1.json" "$state\voice-tail-v1.before-session.json"
}
```

In terminal A, start the LiveKit server with
[Start the local server](local-livekit.md#start-the-local-server) (Git Bash) and check
[Verify readiness](local-livekit.md#verify-readiness).

Choose the inference provider and record it:

- **Ollama:** pull a model with `ollama pull <model>`, confirm its exact name with
  `ollama list`, and pass `--inference-provider ollama --ollama-model <that name>`. The default
  `--ollama-model` names a custom local build that these docs do not provide, so always pass
  one.
- **Codex:** pass `--inference-provider codex` with a signed-in Codex CLI; the host refuses an
  unusable or expiring sign-in with `[codex-session-auth]`.

In terminal B, from the hermes-realtime checkout (Git Bash):

```bash
env -u PYTHONPATH uv run --frozen --extra local hermes-realtime-host \
  --hermes-env-file "$HERMES_HOME/.env" \
  --inference-provider ollama --ollama-model <name from ollama list> \
  --persistent-loopback-launch \
  --allow-unsandboxed-hermes-tasks
```

For Codex, replace the inference line with `--inference-provider codex \`.

- `--allow-unsandboxed-hermes-tasks` is required: without it the host does not start, because
  Hermes command approvals are not a sandbox.
- The first start downloads the speech models. Kokoro's files are pinned by size and SHA-256 and
  verified; the Moonshine model is fetched by the `moonshine` package and is not hash-pinned by
  this repository.
- The flags not passed keep the MVP profile: no `--enable-public-search`, no
  `--evidence-capture`, no `--natural-work-tools` and no `--conversation-profile`.

The host prints an unsandboxed-tasks `WARNING`, a `Knowledge path:` line and then
`Open this stable loopback URL in a local browser:` followed by `http://127.0.0.1:8765/`. The
stable loopback front door trusts every local process while no browser session is active, so
use it only on a single-user workstation; see
[`local-livekit.md`](local-livekit.md#remote-full-host-activation-seam).

In Chrome, open `http://127.0.0.1:8765/`, press **Connect**, grant microphone access, choose the
microphone, and wait for **Listening — speak naturally**. Then:

- **Do:** ask one short ordinary question aloud.
- **Observe:** no `[voice-tail]` marker at this first start, since the old tail was moved aside.
  Your final transcript appears, the reply is audible, and the assistant transcript appears as
  it is delivered. The **Session model** panel shows the providers and models in use.
- **Record:** any marker from the start; audible yes or no; **Response last**;
  `transcript_to_first_token` and `first_token_to_audio`.

### 3. Typed response

- **Do:** type one short ordinary question and send it.
- **Observe:** the same path: an audible reply and its transcript.
- **Record:** audible yes or no; **Response last**.

### 4. Delegate a task

- **Do:** type `task: ` followed by a short, harmless objective that asks for one sentence.
- **Observe:** one task card with a `task_...` identifier becomes active, then completes, and
  the result is reported. No `run_...` or `deleg_...` identifier appears anywhere.
- **Record:** the state sequence; seconds from sending to active and to completed; any marker.

### 5. Interrupt speech while a task runs

- **Do:** start a longer harmless task with `task: `. While it is active, ask a question, and
  speak over the reply while it plays.
- **Observe:** playback stops or lowers when you speak, finishing the word it was on (at most
  about 300 ms more) and fading out rather than cutting mid-syllable; the host prints
  `[speech-stop]` with `mode: word`. The task card stays active and later completes:
  interrupting speech must not cancel the task.
- **Record:** whether playback yielded; the task's states before and after the interruption,
  and its final state; any latency markers that appeared.

### 6. Cancel a task

- **Do:** start another longer harmless task, then type `cancel task: ` followed by the
  `task_...` identifier its card shows.
- **Observe:** exactly that task moves to cancelling and then to interrupted; no other task
  changes.
- **Record:** the state sequence; seconds from sending the cancel to interrupted.

### 7. Reconnect the browser

- **Do:** press **Stop session**, then **Connect** again. Separately, reload the page and press
  **Connect**.
- **Observe:** the session returns to **Listening**, without stale audio or duplicated
  transcript entries, and a question that depends on the earlier conversation is answered in
  context.
- **Record:** each reconnect's outcome; seconds from **Connect** to **Listening**; whether
  context carried over (yes or no).

### 8. Restart the host

- **Do:** start a longer harmless task. While it is active, end the host abruptly, so it cannot
  settle its runs. Find its processes in terminal D (`uv` and the host's interpreter), then
  stop each one it lists:

  ```powershell
  Get-CimInstance Win32_Process |
    Where-Object CommandLine -like '*hermes-realtime-host*' |
    Select-Object ProcessId, Name
  Stop-Process -Force -Id <id>
  ```

  Then start the host again with the same command (session step 2), open the same URL and
  press **Connect**.
- **Observe:** at start, terminal B prints `[voice-tail]` with a positive `restored` count,
  `[hermes-restart-settlement]` with `stopped` of at least 1, and the `NOTICE:` line. After
  **Connect**, the assistant says, once, that it restarted, that background work will not
  resume, and how many tasks were stopped. A question that depends on the conversation before
  the restart is answered in context.
- **Record:** the markers verbatim, in the order they appeared; the `NOTICE:` counts;
  announcement heard yes or no; context resumed yes or no.

A clean stop with `Ctrl-C` instead stops the session's runs as it shuts down, so the next start
restores history but prints no settlement and makes no announcement. If you also try it, record
it as a separate observation.

### 9. Cleanup

- **Do:** press **Stop session**, then `Ctrl-C` in terminal B (host), C (gateway) and A
  (LiveKit). Check that nothing is left running:

  ```powershell
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like '*hermes-realtime-host*' -or
                   $_.CommandLine -like '*gateway run*' -or
                   $_.Name -like 'livekit-server*' } |
    Select-Object ProcessId, Name
  ```

- **Observe:** the browser shows the stopped state, each terminal returns to its prompt, and the
  query lists nothing.
- **Record:** whether each `Ctrl-C` exited cleanly; the number of processes left; any marker
  printed at shutdown, such as `[voice-review-close]`.

The conversation persists on purpose. The host keeps the recent heard conversation in
`%LOCALAPPDATA%\HermesRealtime\state\voice-tail-v1.json` beside its Hermes run record, and the
companion archives it into Hermes. To forget the local copy, delete that file while the host is
stopped; the tail moved aside in session step 2 is yours to keep or delete the same way.
Deleting the archived voice history from Hermes is not available yet.
