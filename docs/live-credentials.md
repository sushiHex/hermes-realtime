# Live access credentials for managed native profiles

The explicit live-Codex profile reads the authorized CLI owner's current access
cache instead of copying a bearer into a deployment profile. It uses the pinned
Hermes named custom-provider `key_cmd` mechanism and the presenter's optional
`key_cmd_argv`. Existing `api_key_env` presenter profiles remain supported.

This is read-only credential resolution, not a refresh service. Codex owns its
normal sign-in and renewal. If its source cache is expired, missing or belongs
to another account, the helper refuses; it does not sign in, rotate a
refresh token, replay a task or change account state. File-based CLI storage is
required. Keyring-only storage is not supported by this resolver.
Server-side revocation remains a server authentication failure.

## Trusted operator configuration

Build one argument list with `codex_access_credentials.command_argv` using the
verified native Hermes **base** Python interpreter, the installed helper file,
the existing absolute CLI auth-file path and the expected account's SHA-256
fingerprint. The isolated command uses `-I -B -S`; a Windows venv redirector is
not a substitute for the base interpreter. The reviewed helper creates no
children, makes no network calls and writes no files.

The helper's stdout is credential material for a captured pipe. Do not run it
interactively, echo its output, or attach it to an issue. Tokens and refresh
tokens never belong in command arguments. Operator paths and the account pin
are private configuration, not model-controlled task arguments.

For this dedicated native profile:

- `model.provider` is `custom:<name>` with one canonical lowercase provider name.
- The `providers` table contains exactly that entry, with only `base_url`,
  `default_model`, `api_mode`, `key_cmd` and `extra_headers`. The latter contains
  only `ChatGPT-Account-ID`, whose value must match the helper's account fingerprint.
  This preserves native account routing when the native SDK holds a callable key.
  Use the exact existing Codex endpoint
  `https://chatgpt.com/backend-api/codex`, without a trailing slash, and
  `api_mode: codex_responses`.
- `model.api_key` is absent. Any model-level base URL or API mode must agree with
  the selected entry. The selected provider and presenter use the same model.
- Presenter configuration uses the normalized provider `custom`, the same route,
  and `key_cmd_argv` instead of `api_key_env`.
- Native `key_cmd` is exactly `shell_command(argv)` for the presenter's argument
  list. Generate it for the deployment platform rather than hand-quoting paths.

The existing profile and native source checks still apply. Credential pools take
precedence over `key_cmd` in the pinned native resolver, so matching configuration
strings alone do not establish the actual credential source. The native probe
checks the real resolver's exact callable type, command and route without calling
the credential. Resolution-time network/process I/O causes refusal, including a
pool refresh that fails and falls back. Preparation checks an exact bounded copy
of the staged configuration in a retained private scratch home under the output,
using the selected read-only environment file. Native initialization stays outside
the profile handoff. This checks the configured resolver input, not the finalized
home. Startup separately checks the real finalized home and the gateway's loaded
environment before advertising the existing web-evidence readiness contract.
Native pool/cache bookkeeping may occur inside those owned homes; the probe is
not a filesystem-read-only operation. The source profile, source environment file
and primary CLI sign-in remain unchanged.

## Lifetime and cancellation

The helper checks the expected account and requires more than 60 seconds of JWT
lifetime. It exports only `access_token` and a short `expires_in` freshness window.
Pinned Hermes has a five-second minimum command cache; the presenter resolves
again for each presentation. Neither contract guarantees validity after server
revocation or throughout an arbitrarily long request.

The presenter executes only the trusted, reviewed childless helper with a bounded
captured pipe and no shell. It retains the existing per-run cancellation owner,
waits for an acknowledged process handle, and settles that child before reporting
credential completion. An unknown OS spawn acknowledgment remains owned rather
than being reported as settled at the request deadline. Arbitrary helper process
trees and unattended renewal of an expired primary sign-in are not qualified.
