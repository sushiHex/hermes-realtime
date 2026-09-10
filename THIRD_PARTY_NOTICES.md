# Third-party notices and model provenance

Hermes Realtime's own source is licensed under the [MIT License](LICENSE). That license does not replace the licenses or service terms of third-party packages, models, voices, or network services.

## Bundled release artifacts

### Silero VAD

The wheel bundles `src/hermes_realtime/providers/models/silero_vad.onnx` from `snakers4/silero-vad` revision `76e3dc408eb2a5c655c34e230d2d5459b4439daa`.

- License: MIT
- SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- Full notice and license: `src/hermes_realtime/providers/models/SILERO_NOTICE.md` and `SILERO_LICENSE`

### Browser bundle

The packaged browser bundle is built from `web/package-lock.json`. `web/build.mjs` generates `src/hermes_realtime/client/static/assets/app.js.LEGAL.txt` from esbuild's production input graph plus the non-development npm lock closure. The release gate requires that notice file and verifies regenerated static-byte parity before building artifacts.

### Synthetic test audio

`tests/fixtures/audio/sapi_stop.wav` is synthetic Windows SAPI speech, not a human recording. Its phrase, generator, PCM format, and SHA-256 are recorded in `tests/fixtures/audio/README.md`. It is included in the source distribution, not the wheel.

## Required Python runtime dependencies

These are installed by a default `pip install hermes-realtime`; they are not optional extras.

### Outbound web search from the Codex host

> [!IMPORTANT]
> Public search is disabled by default. The operator must explicitly pass `--enable-public-search`.
> A participant must separately accept the browser-visible public-search disclosure for the
> active binding before a lookup can run. This is outbound network egress of user-derived content.
> It is separate from, and not covered by, local evidence-capture consent.

When both gates are open, `PublicRssCurrentFactLookup`
(`src/hermes_realtime/providers/current_facts.py`) issues direct HTTPS requests —
`trust_env=False`, public-address-only DNS resolution, 512 KB response cap — to:

| Endpoint | When |
|---|---|
| `https://www.bing.com/search?q=…&format=rss` | every routed lookup |
| `https://news.google.com/rss/search?q=…` | additionally, for outcome-shaped queries |

The query may be derived from stable partial or final speech transcript text, or final typed input
text. Raw microphone audio is not sent by this feature. Consent is scoped to one authenticated
browser binding and closes
on revocation, rebind, projection resynchronization, inactivity expiry, and stop. Revocation cannot
cancel a lookup admitted while consent was active or recall a request that was already sent.
`--knowledge-speculation` and `--knowledge-recovery` control
optional overlap/recovery only; they require public search to be operator-enabled and do not grant
participant consent.

Production composition sets result-page enrichment to zero, so it does not fetch result publishers'
pages. Neither Microsoft nor Google is affiliated with or endorses this project. Their services are
governed by their own terms, and requests carry no project credentials.

### `ddgs`

`ddgs>=9.5,<10` (locked at `9.14.4`, MIT) is a **required** dependency, so it and its closure —
`click`, `fake-useragent`, `httpx` with the `brotli`, `http2`, and `socks` extras, `lxml`, and
`primp` — install with every default install. Their licenses remain their own and are pinned by
`uv.lock`.

`ddgs` is **not** used by the production lookup above. `DdgsCurrentFactLookup` is the base class
that `PublicRssCurrentFactLookup` overrides, and it is instantiated only through
`providers/knowledge_backends.py`, which is imported solely by
`scripts/benchmark_foreground_knowledge.py`. A default host install therefore ships the dependency
without calling it at runtime.

## Optional Python runtime dependencies

Optional extras are installed from `uv.lock`; they are not bundled into the Hermes Realtime wheel. Their licenses remain their own. In particular:

- `edge-tts==7.2.8` declares LGPL-3.0;
- `phonemizer==3.4.0` declares GPL-3.0-or-later;
- `moonshine-voice==0.1.0` and `kokoro-onnx==0.6.1` ship MIT license files;
- the locked npm and Python dependency closures should be re-audited when lockfiles change.

Installing an optional extra creates a combined runtime environment with those dependencies; it does not relicense Hermes Realtime's source.

## Model and voice assets downloaded at runtime

### Kokoro v1.0

`src/hermes_realtime/providers/kokoro.py` downloads exactly these assets from the `thewh1teagle/kokoro-onnx` `model-files-v1.0` GitHub release and rejects size or SHA-256 drift:

| Asset | Size | SHA-256 |
|---|---:|---|
| `kokoro-v1.0.onnx` | 325,532,387 bytes | `7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5` |
| `voices-v1.0.bin` | 28,214,398 bytes | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` |

Upstream identifies `kokoro-onnx` as MIT and the Kokoro model as Apache-2.0. The model and voice assets are not redistributed in this repository.

### Moonshine v2 English streaming models

The public host selects English `TINY_STREAMING`, `SMALL_STREAMING`, or `MEDIUM_STREAMING` through `moonshine-voice==0.1.0`. Upstream's license states that all streaming speech-to-text models are MIT. The upstream package obtains a download manifest and validates expected size plus CRC32C; Hermes Realtime does **not** independently pin SHA-256 digests for those downloaded files. Reproducible or high-assurance deployments must pre-provision and independently verify the selected model cache rather than treating an upstream manifest fetch as release provenance.

### Faster Whisper

`faster-whisper==1.2.1` is MIT. Symbolic model names such as `tiny.en` and `base.en` resolve through the upstream Systran Hugging Face repositories, whose model cards declare MIT. Hermes Realtime does **not** pin a Hugging Face revision or model-file digest for symbolic names. For reproducible deployments, pass an operator-provisioned local model directory and verify that directory under your own artifact policy.

### Edge TTS

Edge TTS is a network service path, not a bundled or downloaded local model. Its availability and service terms are controlled by the upstream service and dependency.

This file records technical provenance and packaging boundaries; it is not legal advice.
