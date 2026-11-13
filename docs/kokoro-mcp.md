# Managed Kokoro GPU speech

The managed Windows host can use the existing Kokoro CUDA worker through an
operator-configured, scoped vram-mcp server. vram-mcp performs the actual GPU
admission, load and owned stop. The host retains the authenticated loopback speech
protocol and verifies the expected execution policy before rendering.

Install the `kokoro-mcp` extra in the host environment. Keep the GPU worker's
environment separate: installing GPU ONNX Runtime into the host is unnecessary.
Moonshine and the host's CPU providers keep their existing configuration.

Select `--tts-provider kokoro --kokoro-mcp-config <absolute-private-config>`.
This rejects a simultaneous `--kokoro-worker-python` setting and does not silently
fall back to CPU or a direct process launch. Omit the MCP configuration to select
the existing CPU path explicitly. Portable installations retain the CPU default;
the operator's managed launch profile determines its default backend.

The private JSON configuration has version `1` and these exact fields:

| Fields | Meaning |
| --- | --- |
| `server_python`, `server_python_sha256` | Native base Python executable and digest. |
| `server_venv_python`, `server_venv_python_sha256` | Scoped MCP environment interpreter and digest. |
| `server_source`, `server_commit`, `server_tree` | Clean reviewed vram-mcp source and exact Git identities. |
| `server_state_dir` | Existing coordination directory shared with the other vram-mcp clients, independent of the app's isolated home. |
| `profile`, `profile_sha256` | Existing trusted vram-mcp Kokoro operator profile and digest. |

Use a dedicated source checkout with no extra importable modules or bytecode
caches. The adapter compares its importable source inventory to Git, including
ignored shadows; run source checks with `-B` to preserve that inventory.

The operator profile binds the device, worker/interpreter and model/voice assets,
execution policy, reservation estimate and free headroom. These values are not
model-controlled tool arguments. Keep the configuration and profile private.
Never include provider credentials in either file.

The operator profile is a version `1` JSON file that the scoped vram-mcp server reads.
The host checks only its shape: exactly the fields `version`, `python`,
`python_sha256`, `venv_python`, `venv_python_sha256`, `worker`, `worker_sha256`,
`model`, `model_sha256`, `voices`, `voices_sha256`, `gpu_index`, `gpu_uuid`,
`reserve_gb`, `headroom_mb`, `voice` and `language`, plus an optional
`execution_policy`. `venv_python`, `worker`, `model` and `voices` are absolute paths.
`venv_python` is the worker environment's interpreter, created by
[`scripts/setup-kokoro-cuda-worker.sh`](../scripts/setup-kokoro-cuda-worker.sh) from
the [pinned worker dependencies](../requirements/README.md). The meaning and checking
of the remaining values belong to vram-mcp; this repository documents no procedure for
choosing a reservation or headroom.

The host refuses a profile whose `model`, `voices`, their digests, `voice` or
`language` differ from what it is about to load. The host's Kokoro language is
`en-gb`. In a managed preview, the host loads the staged copies, so `model` and
`voices` must name the files under the plan's output,
`<output>\input\home\.cache\hermes-realtime\kokoro\<model-sha256>\`, and `voice` must
equal the plan's `speech.tts_voice`. Each `prepare` uses a new output directory, so
each prepared run needs a new operator profile, a configuration carrying that
profile's digest, and that configuration's digest in the plan.
Before the first GPU load, verify that a temporary control claim made through the
existing coordinator is visible through the configured scoped server. A separate
home directory must not silently create a separate reservation domain.

Managed preview plans may add the pair `kokoro_mcp_config` and
`kokoro_mcp_config_sha256` to `host_options`. Both are required together; the
configuration is checked before preparation and host construction. Existing CPU
plans remain valid. Preserve the previous prepared CPU profile and saved state
for rollback through the normal preview-owner procedure.

One SDK owner task manages the scoped server session. Speech cancellation discards
the cancelled turn's output while retaining the same worker for the next turn.
Host shutdown closes the speech connection, stops and checks the matching owned
operation, then closes the scoped server. A timeout or uncertain acknowledgment
does not authorize another load or the release of someone else's reservation.

Reservation and free-headroom settings are admission accounting, not preallocated
memory or a measured hard cap. Check longer speech and cancellation with the
selected budget before adopting a managed default; do not evict unrelated work.
