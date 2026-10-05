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
and the [README](../README.md) for building from source and inference prerequisites.

## The profile

| Aspect | MVP setting |
| --- | --- |
| Operator | One operator, one Chrome browser on Windows. |
| Hermes | v0.21.0 at `29112bef099274229cadff79cdff7bf7b99c4b77`, by the upstream installer. |
| Hermes profile | The default profile, selected explicitly (see step 2). |
| Hermes memory | Built-in memory only: no external memory provider. |
| Host | `hermes-realtime-host` on literal loopback, with the stable loopback front door. |
| Conversation | The default `legacy` profile; explicit `task:` and `cancel task:` commands. |
| Speech | The defaults: Moonshine `tiny` speech-to-text, Kokoro text-to-speech on Windows. |
| Evidence capture | Off: `--evidence-capture` is not passed. |
| Public search | Off: `--enable-public-search` is not passed. |

Before the session, record the profile you actually ran, so the session binds to one candidate:

- the hermes-realtime commit, and the wheel version printed by `uv build`;
- the Hermes version and commit, from the gate's record (step 1 of the session);
- the inference provider and model, and the speech providers and models, as the browser's
  **Session model** panel shows them;
- the Chrome version, the microphone and speaker class (built-in, USB headset, ...), and the
  CPU and GPU class. Never record a device name that identifies a person or a machine.

## Setup

Use four terminals: **A** for LiveKit, **B** for the host, **C** for the Hermes gateway and
**D** for one-off commands. Commands are PowerShell unless they say Git Bash.

### 1. Install Hermes at the exact commit

The one-line installer (`iex (irm ...)`) cannot pass a commit. Download the installer from that
commit and run it as a file:

```powershell
$commit = "29112bef099274229cadff79cdff7bf7b99c4b77"
$source = "https://raw.githubusercontent.com/NousResearch/hermes-agent/$commit"
Invoke-WebRequest -OutFile install.ps1 "$source/scripts/install.ps1"
powershell -ExecutionPolicy Bypass -File install.ps1 -Commit $commit
```

The installer clones Hermes into `<HERMES_HOME>\hermes-agent` and checks out the commit as a
detached `HEAD`. It creates the environment at `<HERMES_HOME>\hermes-agent\venv` and sets
`HERMES_HOME` for your user, `%LOCALAPPDATA%\hermes` by default. It also puts `hermes` on your
`PATH` and runs `hermes setup` to configure a model. On a machine with an existing newer
install, add `-ForceCommit`, or the installer keeps the newer checkout.

Open a new terminal so the user environment applies, then confirm the checkout:

```powershell
git -C "$env:HERMES_HOME\hermes-agent" rev-parse HEAD
git -C "$env:HERMES_HOME\hermes-agent" status --porcelain --untracked-files=no
```

The first command must print the commit above and the second must print nothing: the gate
refuses a checkout with tracked changes, because no commit describes that code.

### 2. Select the default profile

```powershell
hermes profile use default
hermes profile list
```

The gate and the host expect the checkout and the `.env` in one home, `<HERMES_HOME>`. That is
the default profile's layout, so a named profile is outside this MVP profile.

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

Add these keys. Generate each secret with a cryptographic random generator, for example:

```powershell
$python = "$env:HERMES_HOME\hermes-agent\venv\Scripts\python.exe"
& $python -c "import secrets; print(secrets.token_urlsafe(32))"
```

```dotenv
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8642
API_SERVER_KEY=<at least 32 random characters>
HERMES_REALTIME_COMPANION_PORT=<a free loopback port>
HERMES_REALTIME_COMPANION_TOKEN=<24 to 512 random characters>
```

Never paste these values into shell history, issues, screenshots or logs. The gateway reads the
whole file. The host reads only `API_SERVER_KEY` and the two companion keys from it. Built-in
memory needs no provider keys. See [`local-livekit.md`](local-livekit.md#one-time-hermes-api-setup)
for the API server keys.

### 6. Start the Hermes gateway

In terminal C:

```powershell
hermes gateway run
```

It runs in the foreground; `Ctrl-C` stops it. To restart it, press `Ctrl-C` and run it again.
After `run`, `hermes gateway restart` would start a detached gateway instead and may offer to
install a scheduled task. Restart the gateway after any install or update of Hermes or the
wheel. Then check the API server from terminal D:

```powershell
curl.exe --fail http://127.0.0.1:8642/health
```

### 7. Start the LiveKit server

In terminal A, follow [Pinned development server](local-livekit.md#pinned-development-server)
and [Start the local server](local-livekit.md#start-the-local-server) (Git Bash), then
[Verify readiness](local-livekit.md#verify-readiness).

### 8. Start the full host

In terminal B, from the hermes-realtime checkout (Git Bash):

```bash
env -u PYTHONPATH uv run --frozen --extra local hermes-realtime-host \
  --hermes-env-file "<HERMES_HOME>/.env" \
  --inference-provider <ollama|codex> \
  --persistent-loopback-launch \
  --allow-unsandboxed-hermes-tasks
```

- `--allow-unsandboxed-hermes-tasks` is required: without it the host does not start, because
  Hermes command approvals are not a sandbox.
- The inference provider is your recorded choice. `ollama` needs a local Ollama with the
  default model. `codex` needs a signed-in Codex CLI. See the README's
  [Local conversation profile](../README.md#local-conversation-profile).
- The first start downloads the pinned speech model files and verifies their hashes.
- The flags not passed keep the MVP profile: no `--enable-public-search`, no
  `--evidence-capture`, no `--natural-work-tools` and no `--conversation-profile`.

The host prints an unsandboxed-tasks `WARNING`, a `Knowledge path:` line and then
`Open this stable loopback URL in a local browser:` followed by `http://127.0.0.1:8765/`. The
stable loopback front door trusts every local process while no browser session is active, so
use it only on a single-user workstation; see
[`local-livekit.md`](local-livekit.md#remote-full-host-activation-seam).

### 9. Open the browser

In Chrome, open `http://127.0.0.1:8765/`, press **Connect**, grant microphone access, choose the
microphone, and wait for **Listening — speak naturally**. The **Session model** panel shows the
providers and models in use.

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
launch URL is a credential.

From the gate:

- `[real-hermes-gate]`: the gate refused. Its categories are listed in
  [`hermes-bridge.md`](hermes-bridge.md); `components` folds markers other code printed during
  the run, such as `[hermes-identity]` for a modified Hermes checkout.

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

Post the record sheet to [#159](https://github.com/sushiHex/hermes-realtime/issues/159).

## The diagnostic session

### 1. Run the installed-runtime gate

In terminal D, from the hermes-realtime checkout, with the gateway running:

```powershell
& "$env:HERMES_HOME\hermes-agent\venv\Scripts\python.exe" scripts\real_hermes_api_gate.py
```

- **Do:** run it before connecting the browser, so its work is not mixed with the session's.
- **Observe:** exactly one line. A pass is a JSON record with `"gate": "passed"`; a refusal is a
  `[real-hermes-gate]` marker. It dispatches real work, so the gateway's model is called, and
  it raises and rejects one approval.
- **Record:** the line verbatim; it names the Hermes version, commit and baseline, the
  candidate's version, and each behavior's status. For a refusal, its stage and category; on
  `restart_gateway` or `hello_refused`, restart the gateway, run it again and record both.

### 2. Talk

- **Do:** after **Listening — speak naturally**, ask one short ordinary question aloud.
- **Observe:** your final transcript appears, the reply is audible, and the assistant transcript
  appears as it is delivered.
- **Record:** audible yes or no; **Response last**; `transcript_to_first_token` and
  `first_token_to_audio`.

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
- **Observe:** playback stops or lowers when you speak, and the task card stays active and later
  completes: interrupting speech must not cancel the task.
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

  Then start the host again with the same command (setup step 8), open the same URL and press
  **Connect**.
- **Observe:** at start, terminal B prints `[voice-tail]` with a positive `restored` count, then
  `[hermes-restart-settlement]` with `stopped` of at least 1, then the `NOTICE:` line. After
  **Connect**, the assistant says, once, that it restarted, that background work will not
  resume, and how many tasks were stopped. A question that depends on the conversation before
  the restart is answered in context.
- **Record:** the markers verbatim; the `NOTICE:` counts; announcement heard yes or no; context
  resumed yes or no.

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
- **Record:** clean exits yes or no; the number of processes left; any marker printed at
  shutdown, such as `[voice-review-close]`.

The conversation persists on purpose. The host keeps the recent heard conversation in
`%LOCALAPPDATA%\HermesRealtime\state\voice-tail-v1.json` beside its Hermes run record, and the
companion archives it into Hermes. To forget the local copy, delete that file while the host is
stopped. Deleting the archived voice history from Hermes is not available yet.
