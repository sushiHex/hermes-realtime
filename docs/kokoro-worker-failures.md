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
