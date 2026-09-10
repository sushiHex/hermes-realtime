# Hermes Realtime

[![Release gates](https://github.com/sushiHex/hermes-realtime/actions/workflows/release-gates.yml/badge.svg)](https://github.com/sushiHex/hermes-realtime/actions/workflows/release-gates.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hermes Realtime is an independently authored, experimental realtime voice runtime for [Hermes Agent](https://github.com/NousResearch/hermes-agent), built around LiveKit.

It separates latency-critical conversation from autonomous work:

- streaming speech and natural interruption stay in the foreground;
- Hermes sessions, tools, memory, and long-running work continue independently;
- important results can re-enter the conversation through bounded, context-sensitive updates;
- speech, foreground turns, background tasks, and sessions have separate cancellation scopes.

> [!IMPORTANT]
> Hermes Realtime is alpha software. The local and authenticated full-host launchers are intended for constrained testing and development. This project is not affiliated with or endorsed by Nous Research or LiveKit.

## What works

The current baseline includes:

- a strict, versioned work-dispatch and cancellation protocol;
- provider-neutral VAD, streaming STT, and streaming TTS interfaces;
- deterministic conversation states and conservative playback-delivery accounting;
- independent cleanup scopes for speech, turns, background work, and disconnects;
- an official Python LiveKit SDK adapter for room-scoped tokens and raw PCM transport;
- a loopback-by-default HTTP host and packaged TypeScript browser client;
- one-use browser bootstrap, short-lived microphone credentials, refresh, authenticated event polling, approval decisions, and stop;
- a `hermes_agent.plugins` entry point exercised against a local Hermes v0.20 source surface for discovery and packaging compatibility;
- authenticated Hermes run dispatch, exact task stop, request-bound approvals, and bounded concurrent run ownership;
- opt-in local speech providers and explicit Codex or Ollama inference selection;
- synthetic browser, LiveKit, packaging, and installed-wheel release gates.

Evidence-only Slice 0 includes host capture controls, storage, and lifecycle code.
Full Slice 0 qualification is incomplete.
It remains disabled by default; operator enablement is not consent.
See [Implementation status](docs/implementation-status.md) for code, tests, and
outstanding qualification. The project does not claim subjective audio quality,
physical-room acoustic performance, general iPhone/WebKit readiness, or production
security hardening.

## Architecture

```text
Browser microphone
       │
       ▼
   LiveKit room ──► conversation worker ──► STT ──► inference
       ▲                    │                           │
       │                    └──── state/events ─────────┘
       │
       └──────── streaming TTS / interruption control

Hermes Agent runs, approvals, tools, and long work remain separately owned.
```

The realtime coordinator receives compact context rather than Hermes's complete tool schema. Provider integrations remain replaceable, and task-state claims are emitted only from acknowledged Hermes events.

## Requirements

- Python 3.11
- a [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) API server for authenticated full-host task dispatch
- an optional Hermes installation exposing the plugin CLI when exercising entry-point discovery
- [`uv`](https://docs.astral.sh/uv/) for source builds
- a local LiveKit server for the native integration path
- additional system/provider prerequisites for optional speech profiles

This repository does not claim compatibility with a publicly released Hermes Agent version. The local v0.20 source-surface harness checks a supplied source archive, but the repository does not publish an immutable upstream source identity for that harness. Python 3.12+ is not yet declared supported.

## Install from source

Hermes Realtime has no PyPI release. The only supported distribution channel is a wheel built from this repository. Install the wheel into the Python 3.11 environment that will run the selected Hermes Realtime executable. Use the environment that runs Hermes Agent only when separately exercising `hermes_agent.plugins` entry-point discovery.

```bash
git clone https://github.com/sushiHex/hermes-realtime.git
cd hermes-realtime
uv sync --frozen --group dev
uv build
```

Then install the generated wheel into the environment that will run Hermes Realtime:

```bash
python -m pip install dist/hermes_realtime-*.whl
```

The functional work-dispatch route is the separately launched, authenticated `hermes-realtime-host`. It talks to the configured Hermes API server and does not require in-process plugin bridge dispatch.

Entry-point discovery is an optional packaging check. To exercise it, install the wheel with the Python environment that owns your `hermes` command, confirm that interpreter can import `hermes_cli`, and then run:

```bash
python -c "import hermes_cli, sys; print(sys.executable)"
hermes plugins enable hermes-realtime --no-allow-tool-override
hermes plugins doctor hermes-realtime --ci
hermes plugins list --enabled
```

If Hermes uses an isolated environment, use that environment's Python or install with `uv pip install --python <path-to-hermes-python> dist/hermes_realtime-*.whl`.

A source checkout alone is not an installation. Do not use `hermes plugins install owner/repository` for this package: that command targets directory-style plugins, while Hermes Realtime is a pip entry-point distribution.

On the locally exercised Hermes v0.20 source surface, enabling the entry point provides discovery and packaging compatibility only; in-process bridge dispatch remains fail-closed. Enabling the entry point does **not** start the realtime host or provide functional v0.20 bridge dispatch. Launch `hermes-realtime-host` separately with an explicit profile. The minimal loopback development profile below is the safest first run.

## Local conversation profile

Prerequisites include the pinned local LiveKit development server, Ollama with your selected model, FFmpeg on `PATH`, and network access when using Edge TTS. Development LiveKit credentials are safe only on loopback and must not be exposed to a LAN.

```bash
uv sync --frozen --dev --extra local
uv run --frozen --extra local hermes-realtime-local
```

After provider preflight, open the emitted one-use loopback URL in one local browser. Task dispatch and approvals deliberately fail closed in this profile. Do not place the launch fragment in logs, screenshots, issue text, shell history, or evidence artifacts.

The full host can keep one stable loopback entry across sequential sessions:

```bash
uv run --frozen --extra local hermes-realtime-host \
  --persistent-loopback-launch \
  --hermes-env-file /path/to/hermes/.env \
  --hermes-context-file /path/to/hermes/realtime-context.json \
  --inference-provider codex \
  --allow-unsandboxed-hermes-tasks
```

The `--allow-unsandboxed-hermes-tasks` flag is intentionally explicit. Hermes command approvals are not a sandbox, and server-side tasks may mutate state without a command-approval prompt.

The optional context file is a strict version-1 JSON object. `identity` is required; `persona`, `user_preferences`, and `location` are optional bounded strings. It is sent to the configured inference provider. Do not include secrets, filesystem paths, complete memories, or sensitive personal data.

## Conversation profiles

- `legacy` is the default and best-tested profile.
- `natural_v1` is default-disabled. It adds pause-tolerant endpointing, bounded microphone-onset ducking, explicit **Stop speaking**, rendered-duration segmentation, and renderer recovery.

`natural_v1` is not yet qualified for general use. Microphone energy alone lowers renderer volume; an exact pause is committed only after the server promotes the speech floor, while **Stop speaking** retains immediate exact-stream authority.

## Privacy and security

Public search is disabled by default. Operator enablement is not participant consent. To make it
available, the host operator must explicitly pass `--enable-public-search`; each active browser
binding must then grant separate browser-visible consent before any lookup. If both gates are open,
Hermes Realtime may send a query derived from stable partial or final speech transcript text, or
final typed input text, over HTTPS to Bing Search RSS and, for outcome-shaped queries,
Google News RSS. Raw microphone audio is not sent by this feature.
Consent closes on revocation, rebind, projection resynchronization, inactivity expiry, and stop.
Revocation cannot cancel a lookup admitted while consent was active or recall a request that was
already sent. See
[`source-backed-latency.md`](docs/source-backed-latency.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

- Browser launch fragments and bearer material are credentials. Never paste them into issues or logs.
- Source text retrieved from the web is treated as untrusted evidence, not instructions.
- Private-looking queries are withheld from external lookup by the source-backed path.
- Edge TTS sends assistant reply text to Microsoft's speech service. It does not receive microphone audio or user transcripts from this project.
- Loopback trust assumes a trusted single-user workstation; use the one-use launch mode when local processes are not mutually trusted.
- No repository test may require a contributor's named audio device, private host path, physical room, or personal account.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Documentation

Start with the [Collaborator guide](docs/README.md),
[Roadmap](docs/roadmap.md), and [Implementation status](docs/implementation-status.md)
for priorities, code and test locations, and remaining qualification work.

- [Local LiveKit setup and automated gates](docs/local-livekit.md)
- [Hermes bridge and natural-work authority](docs/hermes-bridge.md)
- [Source-backed routing, privacy, and benchmark controls](docs/source-backed-latency.md)
- [Evidence capture boundary](docs/evidence-capture.md)
- [Release and installed-boundary gates](docs/release-gates.md)

## Development

```bash
uv sync --frozen --group dev
uv run --frozen --group dev pytest -q
uv run --frozen --group dev ruff check .
uv run --frozen --group dev mypy src
```

The canonical committed-`HEAD` qualification is:

```bash
python scripts/release_gate.py --candidate .
```

It builds and inspects the wheel and sdist, scans committed Git blobs for secret patterns, verifies browser assets, and tests an isolated installation. It does not qualify dirty, staged, or untracked bytes.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Public contributions must be hardware-generic, secret-free, reproducible, and useful outside one private deployment.

## License

Hermes Realtime is released under the [MIT License](LICENSE). Bundled browser code, the Silero VAD model, optional dependency licenses, and runtime-downloaded model/voice assets are documented separately in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
