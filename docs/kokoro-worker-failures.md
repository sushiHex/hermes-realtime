# Kokoro worker failure categories

The isolated worker reserves the following nonzero exit codes. These codes report
where an ordinary Python exception or an existing placement guard failed. They do
not establish a root cause, CUDA readiness, process settlement or safe capacity.
The authenticated ready message and existing CUDA placement checks remain required.

| Exit code | Fixed category |
| --- | --- |
| 80 | `authentication_failed` |
| 81 | `parent_watchdog_failed` |
| 82 | `asset_attestation_failed` |
| 83 | `dependency_import_failed` |
| 84 | `dll_preload_failed` |
| 85 | `cuda_session_failed` |
| 86 | `cuda_provider_rejected` |
| 87 | `kokoro_engine_failed` |
| 88 | `attestation_synthesis_failed` |
| 89 | `cuda_profile_unavailable_or_invalid` |
| 90 | `cuda_profile_missing_core_ops` |
| 91 | `cuda_profile_forbidden_cpu_ops` |
| 92 | `loopback_connection_failed` |
| 93 | `worker_service_failed` |

Dependency imports retain their original order and share code 83. Codes 90 and 91
identify the existing required-core-operation and forbidden-CPU-operation guards;
neither guard is relaxed. Code 93 covers ready-message/service processing after the
loopback connection is entered, so it must not be described as proof of a startup
failure or of successful authentication.

On a caught exception, the worker attempts one bounded stderr line prefixed
`[kokoro-worker-failure]`, containing only `version: 1`, the fixed category and an
allowlisted exception kind. Unknown exception types become `unknown_exception`.
Exception messages, arguments, paths, tokens, request identifiers, provider output
and stderr are not copied into that marker. A broken diagnostic stream cannot
replace the selected exit status. Collectors should use the reserved exit-code
mapping rather than parsing or replaying arbitrary worker stderr.
After a diagnostic write/flush failure, the failed Python stderr is detached so
CPython shutdown cannot retry that stream and replace the status with exit 120.
No raw fallback diagnostic is emitted.

Success remains exit code 0 with the existing version 1 authenticated wire protocol.
Parent-watchdog exits 70 and 71 remain unchanged. Argument-parser exits, native
crashes and other exit codes are outside this mapping and must remain unclassified.
An unrecognized internal phase returns ordinary exit code 1 with the fixed marker
category `unknown_worker_failure`; it does not invent a service-phase witness.
An exit code is not evidence that a controller caused, observed or completed cleanup.


## Authenticated rejected-operation witness

Only the existing forbidden-CPU-operation guard (exit 91) may attempt a diagnostic
startup connection. Before returning its failure status, it sends one version 1
`startup_failure` header on the existing loopback transport. Its exact fields are
`version`, `type`, `token`, `modelSha256`, `voicesSha256`, `category` (fixed to
`cuda_profile_forbidden_cpu_ops`) and `cpuOperationTypes`.

The operation list is the complete sorted, unique, nonempty rejected type set,
with at most 64 types of at most 64 characters each. Missing/non-string CPU type
metadata, an unknown type or an overflow makes the entire witness unavailable;
no recognized subset or truncated list is reported. The finite 85-type witness
vocabulary is bound to model SHA-256
`beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a`.
It comprises the pinned model's 67 operation types and 18 reviewed ONNX Runtime
or existing-policy labels. This vocabulary controls diagnostic disclosure only;
it does **not** expand the CPU placement allowlist or authorize any placement.
The reviewed additional labels come from ONNX Runtime 1.28's
[contributed operators](https://github.com/microsoft/onnxruntime/blob/v1.28.0/docs/ContribOperators.md),
[layer normalization kernels](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/contrib_ops/cpu/layer_norm.cc),
[NCHWc schemas](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/graph/contrib_ops/nchwc_schema_defs.cc)
and [memory-copy transformer](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/optimizer/transformer_memcpy.cc).

The client requires exact schema, version, token, both pinned asset hashes and
catalog membership. A valid header raises `KokoroCpuPlacementRejected`, whose
fixed message carries no operation names and whose read-only
`cpu_operation_types` property contains an immutable tuple. Its constructor also
validates the model-bound tuple. No node names, profile paths, tensors, exception
messages, arbitrary stderr or user content are transported.

If the child already exited with code 91, the client may nonblockingly accept an
already queued failure connection; a dead child's `ready` header is rejected.
All header reads, including fragmented length and body reads, share the original
startup deadline. There is no extra grace, acknowledgment or retry. Transmission
failure cannot replace exit 91. Existing fault/close and owned-process settlement
remain unchanged.

This header proves only authenticated profile rejection on the attempted pinned
worker. It publishes no providers or synthesis socket and establishes neither
CUDA readiness nor a natural OS exit. Client cleanup can precede natural exit;
only separately observed MCP exit metadata may establish that exit category.
