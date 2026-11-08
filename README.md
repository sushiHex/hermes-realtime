# Hermes Realtime

[![Release gates](https://github.com/sushiHex/hermes-realtime/actions/workflows/release-gates.yml/badge.svg)](https://github.com/sushiHex/hermes-realtime/actions/workflows/release-gates.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Realtime voice conversations with [Hermes Agent](https://github.com/NousResearch/hermes-agent), built around LiveKit. Talk, delegate work, and keep the conversation going while Hermes works in the background.

Speech recognition and synthesis can run locally. The conversation model can use an authorized Codex CLI or a local Ollama runtime; Hermes owns background execution, tools, approvals and task sessions. These are separate choices: local speech does not make cloud inference or network tools offline.

> [!IMPORTANT]
> Independently authored alpha software for development and constrained testing. The Windows desktop preview has exercised native delegation and local speech, but it is not a generally qualified production assistant. This project is not affiliated with or endorsed by Nous Research or LiveKit.

## What works

| Area | Current capability |
| --- | --- |
| **Natural delegation** | The foreground can hand a self-contained objective to Hermes. Choose per-task confirmation or explicitly configured standing authorization, so requested work can start without a confirmation card. Work is reported as started only after an accepted dispatch acknowledgment. |
| **Conversation while work runs** | Task status, completion updates and follow-up result recall share bounded conversation context. Speaking or **Stop speaking** interrupts the reply, independently of background work. Task cancellation and approvals remain explicit user controls. |
| **Local speech** | Moonshine Voice streaming STT, including the Medium tier; Kokoro ONNX TTS on CPU or an explicitly managed Windows CUDA worker. Faster Whisper and Edge TTS remain alternative providers. |
| **Browser controls** | Microphone and voice selection, speaker volume/mute, output style, typed input and diagnostics. Model/effort selection is available where the configured provider supports it. |
| **Speech and readable results** | Short conversational replies and supporting result detail, supported unit/time projection, and explicit bounded pronunciation overrides. The optional native web-evidence handoff builds speech and display from the same passage-backed answer. |
| **Continuity** | A durable local voice tail. With the optional Hermes-side companion, bounded voice archiving, native memory/skills review, asynchronous memory readback and conversation-archive deletion. |
| **Deployment and verification** | Source/wheel/runtime-bound desktop preview preparation, owned startup and shutdown, saved-state handoff, typed native journey checks, and installed-wheel/browser/LiveKit CI gates. |

These are implemented paths, not guarantees of model obedience, factual completeness or universal pronunciation quality. Tools and local-system access depend on the configured Hermes profile. Current delivery and remaining acceptance work live in [the desktop MVP tracker](https://github.com/sushiHex/hermes-realtime/issues/159); [Implementation status](docs/implementation-status.md) records evidence and qualification limits.

## Architecture

![The browser exchanges microphone and speaker audio with the realtime host through LiveKit. The foreground conversation delegates durable work to Hermes Agent; task results return as bounded updates. Speech and background-task cancellation are independent.](docs/assets/architecture.svg)

The realtime host handles voice, foreground inference and playback. Hermes Agent runs durable work separately. The foreground receives bounded context and acknowledged task state, rather than taking ownership of Hermes's complete executor. Speech, foreground turns, named tasks and sessions have independent cancellation scopes.

The local voice tail restores bounded context after a restart. A configured, ready companion can archive eligible voice rows into Hermes and invoke its native memory and skills machinery; a bounded memory snapshot can return asynchronously. Deleting a voice conversation does **not** unlearn memories or skills, remove delegated-task sessions, or guarantee physical erasure. See [ADR 0003](docs/adr/0003-hermes-owned-conversation-continuity.md) and the [deletion guide](docs/desktop-mvp-diagnostic.md#deleting-a-voice-conversation).

## Requirements

- Python **3.11** and [`uv`](https://docs.astral.sh/uv/) for the documented source workflow.
- A local LiveKit server for the native media path.
- An authenticated Hermes Agent API server for task-capable profiles.
- The selected inference and speech providers' prerequisites. GPU Kokoro additionally needs the reviewed CUDA worker environment and vram-mcp configuration.

Windows is the primary desktop development target. Linux CI exercises supported generic paths and the null evidence-capture surface; it does not qualify Linux evidence storage. General iPhone/WebKit readiness, physical-room acoustics and production security hardening are not established. Python 3.12+ is not declared supported.

This repository does not claim compatibility with a publicly released Hermes Agent version. Use the reviewed Hermes source and runtime identities required by the selected guide. A matching version label alone is not a compatibility guarantee. Plugin discovery does not launch a voice session or provide the functional work-dispatch route.

## Install from source

Build a wheel from this repository, then install it into the Python 3.11 environment that will run Hermes Realtime:

```bash
git clone https://github.com/sushiHex/hermes-realtime.git
cd hermes-realtime
uv sync --frozen --group dev
uv build
uv pip install --python <path-to-python-3.11> dist/hermes_realtime-<version>-py3-none-any.whl
```

Replace the interpreter path and wheel filename with those for your environment. Install optional provider dependencies using the selected setup guide. Keep the GPU worker environment separate from the host; GPU ONNX Runtime does not need to replace the host's CPU runtime.

A source checkout alone is not an installation. The functional work-dispatch route is the separately launched, authenticated `hermes-realtime-host`. It talks to the configured Hermes API. Installing the `hermes_agent.plugins` entry point is an optional discovery/packaging step, not a replacement for launching the host.

For that optional check, install the wheel into the environment owning the `hermes` command, then run `hermes plugins enable hermes-realtime --no-allow-tool-override` and `hermes plugins list --enabled`. Use wheel installation rather than the directory-style `hermes plugins install owner/repository`. The reviewed Hermes v0.21.0 `plugins doctor` checks directories, not entry points, so it is not a registration check for this package.

On the locally exercised Hermes v0.20 source surface, enabling the entry point provides discovery and packaging compatibility only; in-process bridge dispatch remains fail-closed. This is the discovery harness's scope, not a claim about the native preview's Hermes source version.

## Choose a launch path

| Goal | Start here |
| --- | --- |
| **Task-capable Windows desktop preview** | [Native work preview](docs/native-work-preview.md): prepare a source-bound plan, verify prerequisites, then perform normal stop, saved-state handoff and persistent launch. This is the maintained path for native Hermes work. |
| **Manual full-host composition** | [Local LiveKit and full-host setup](docs/local-livekit.md): authenticated Hermes API, explicit providers, browser launch and optional companion configuration. |
| **Conversation-only demo** | [Local conversation profile](#local-conversation-profile) below: Ollama, Faster Whisper and Edge TTS. Task dispatch and approvals are disabled. |
| **Check an existing native preview without microphone input** | [Typed native journey client](docs/native-typed-journey.md): admitted input, task evidence, delivery and cleanup checks. It does not establish audible quality. |

The managed preview's `prepare` step is offline. `launch` reports a request to start, not readiness; wait for the owner's readiness receipt. Use the maintained owner commands for stop and handoff. The [native preview guide](docs/native-work-preview.md) also explains the narrower DEV readiness policy and its limits.

### Automatic background work

Natural work tools are opt-in, and `confirm` remains the start-policy default for natural proposals. To enable natural requests with standing authorization, an operator selects both natural work tools and the automatic start policy. The full-host CLI options are:

```text
--natural-work-tools --task-start-policy automatic
```

In a managed plan, configure the corresponding `host_options.natural_work_tools` and `host_options.task_start_policy` fields. These settings do not approve downstream tool effects or give the model cancellation authority. Full-host task execution requires explicit `--allow-unsandboxed-hermes-tasks` authorization (the managed plan uses `allow_unsandboxed_tasks`); Hermes command approvals are not a sandbox. Follow [the authority contract](docs/adr/0004-model-action-authority.md) and the selected deployment guide rather than treating these options as a complete launch command.

### Speech selection

Select Moonshine's tier explicitly when it matters: the full-host default remains Tiny; `--moonshine-model-tier medium` selects Medium. Portable Kokoro uses CPU unless a worker backend is selected. Managed GPU speech is explicit through `--kokoro-mcp-config` or the paired, hash-bound managed-plan fields; it has no silent CPU or direct-launch fallback. See [managed Kokoro GPU speech](docs/kokoro-mcp.md) and the [pinned worker dependencies](requirements/README.md).

GPU reservation/headroom are admission accounting, not preallocated memory or a measured hard VRAM cap. Choose and verify the budget for the deployment; this README does not prescribe a universal allocation.

Pronunciation overrides are empty by default. The host accepts bounded whole-term `--kokoro-pronunciation TERM=SPOKEN` entries; managed plans can supply `speech.pronunciations`. Overrides affect synthesis while preserving displayed text, and do not infer abbreviation meanings. See [the preview configuration](docs/native-work-preview.md).

## Local conversation profile

For the separate conversation-only demo, run a local LiveKit server and Ollama with an installed model. FFmpeg must be on `PATH`; Edge TTS requires network access.

```bash
uv sync --frozen --dev --extra local
uv run --frozen --extra local hermes-realtime-local --ollama-model your-installed-model
```

Replace `your-installed-model` with the model you intend to use. After provider preflight, open the emitted one-use loopback URL in one local browser. This profile deliberately refuses task dispatch and approvals. Development LiveKit credentials are for loopback only. Keep launch fragments out of logs, screenshots and issues.

## Conversation profiles

- `legacy` remains the default conversation profile.
- `natural_v1` is opt-in: pause-tolerant endpointing, microphone-onset ducking, explicit **Stop speaking**, rendered-duration segmentation and renderer recovery.

The desktop preview's selected settings do not change portable defaults. `natural_v1` is not generally qualified: microphone energy initially lowers renderer volume; an exact pause is committed after the server promotes the speech floor. **Stop speaking** has immediate exact-stream authority. See [the full-host guide](docs/local-livekit.md) for the profile contract.

## Privacy and security

Cloud inference receives admitted conversation context; Hermes tasks and their configured tools may send objectives or retrieved data to external services. Local STT/TTS alone is not an offline guarantee. Edge TTS sends assistant reply text to Microsoft's speech service, not microphone audio from this project.

Hermes-native web tasks and the optional foreground public-search feature are separate paths. Native work follows Hermes's configured tool and approval policy. Public search is disabled by default. Operator enablement is not participant consent. The foreground feature requires both operator `--enable-public-search` and separate browser-visible consent from the active binding. It can send transcript-derived or typed queries to Bing Search RSS and, for outcome-shaped queries, Google News RSS; it does not send raw microphone audio. Consent closes on revocation, rebind, resynchronization, inactivity expiry and stop, but cannot recall an already admitted request. See [source-backed routing](docs/source-backed-latency.md).

Evidence capture is an opt-in development surface. Full Slice 0 qualification is incomplete. It remains disabled by default; operator enablement is not consent. See [Evidence capture boundary](docs/evidence-capture.md).

- Browser launch fragments, access tokens and bearer material are credentials; never paste them into issues or logs.
- Retrieved text is untrusted evidence, not instructions. Evidence binding does not prove factual entailment or completeness.
- Loopback trust assumes a trusted single-user workstation. Use one-use launch mode when local processes are not mutually trusted.
- Conversation deletion and evidence erasure have different scopes; learned memory and skills can remain.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Dependency and runtime-downloaded asset notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Documentation

| Topic | Guide |
| --- | --- |
| Native desktop deployment and state handoff | [Native work preview](docs/native-work-preview.md) |
| Host, LiveKit and provider configuration | [Local LiveKit setup](docs/local-livekit.md) |
| Managed CUDA speech and CPU rollback | [Kokoro via vram-mcp](docs/kokoro-mcp.md) |
| Managed provider credentials | [Live access credentials](docs/live-credentials.md) |
| Task results and web evidence | [Native web evidence](docs/native-web-evidence.md) |
| Automated typed checks and timing limits | [Native journey client](docs/native-typed-journey.md) |
| Desktop diagnostic and acceptance procedures | [Diagnostic guide](docs/desktop-mvp-diagnostic.md), [acceptance guide](docs/desktop-mvp-acceptance.md) |
| Architecture, source evidence and contribution workflow | [Collaborator guide](docs/README.md), [implementation status](docs/implementation-status.md), [work tracking](docs/work-tracking.md) |
| Release qualification | [Release gates](docs/release-gates.md) |

Substantive work and acceptance outcomes are tracked in [GitHub Issues](https://github.com/sushiHex/hermes-realtime/issues) and [Milestones](https://github.com/sushiHex/hermes-realtime/milestones).

## Development

```bash
uv sync --frozen --group dev
uv run --frozen --group dev pytest -q
uv run --frozen --group dev ruff check .
uv run --frozen --group dev mypy src
```

The canonical committed-`HEAD` qualification is:

```bash
uv run --frozen --group dev python scripts/release_gate.py --candidate .
```

It builds and inspects the wheel and sdist, scans committed Git blobs for secret patterns, verifies browser assets and tests an isolated installation. Dirty, staged and untracked bytes are excluded from the candidate, not qualified by this command. Required CI also covers native LiveKit/browser integration, packaged capture/spool recovery and Linux null capture.

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before contributing. Public changes must be hardware-generic, secret-free and reproducible; tests must not depend on a contributor's private account, paths, named audio device or physical room.

## License

[MIT](LICENSE). Browser dependencies, bundled components and runtime-downloaded model/voice assets have separate notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
