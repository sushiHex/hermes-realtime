# Experimental native Runs policy proof

Refs [#306](https://github.com/sushiHex/hermes-realtime/issues/306), especially the
[controlled lifetime decision](https://github.com/sushiHex/hermes-realtime/issues/306#issuecomment-6349139236)
and [implementation boundary](https://github.com/sushiHex/hermes-realtime/issues/306#issuecomment-6350105618).
This is an opt-in, synthetic proof, outside ordinary pytest collection. **Never
install this fixture in a user profile.** It does not activate product behavior.

`policy.py` contains the candidate policy. `entrypoint.py` separates the generic
installer from `__init__.py`, which supplies synthetic model completions, counted
handlers and fault controls. Its separate `register(ctx)` entrypoint is source
for review; direct packaging/loading of that entrypoint is not qualified here.
The fixture invokes the same installer through supported Hermes plugin APIs.

## Interface and scope

The exact supported source is official Hermes Agent
[`29112bef099274229cadff79cdff7bf7b99c4b77`](https://github.com/NousResearch/hermes-agent/tree/29112bef099274229cadff79cdff7bf7b99c4b77).
No Hermes core modification, replacement dispatcher, new dependency or supervisor
is required by this proof. It uses native HTTP Runs, public plugin registration,
execution middleware, registry lookup and the approval-session accessor.

- `GET /lookup-policy/ready` reports readiness, generation and policy version.
- Explicit `X-Lookup-Policy: ordinary-lookup/v1` on bare `POST /v1/runs` binds the
  real accepted run ID before execution. Native status/body stay native; response
  headers acknowledge that ID, generation and version. Policy-bearing profile
  alias admission is refused before dispatch; ordinary aliases remain native.
- One complete batch of 1ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ5 searches may be followed by one extraction of 1ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ3
  distinct, exact URLs returned by successful searches. No classification,
  normalization, retry, extra batch or budget reset occurs.
- Immutable designation/refusal snapshots remain for the process lifetime.
  Missing or replaced mutable budgets refuse designated runs selectively;
  detected authority or guard loss closes the dedicated generation. This does
  not promise ordinary same-process continuation through total authority loss.
- Single-use run/request/call tickets check effective arguments at the actual
  handler. Native authenticated stop closes new admission before acknowledgment.
  Earlier effects may finish; late results cannot authorize more work.
- Successful authenticated native status GETs expose separate policy-refusal
  headers without rewriting status/body. Unknown authority is unavailable, not
  false. Ordinary native errors and unauthorized/non-200 responses stay native.

Only the OpenAI chat-completions path has synthetic evidence. Other transports,
active plugin removal, real model/provider access, source freshness, voice and
useful-answer acceptance remain unqualified. A truthful native `completed` plus
policy refusal does not pass useful acceptance. SSE/client integration is absent.

## Reproduce one bounded case

Use an already prepared, disposable, pinned Hermes environment. Do not install or
upgrade packages to run this proof. From the repository root, provide these local
environment variables: `HERMES_PROOF_PYTHON` (that environment's venv interpreter),
`HERMES_PROOF_SOURCE` (its checkout), and `HERMES_PROOF_RECEIPT` (a new output file).
Optional `HERMES_PROOF_CASE` and `HERMES_PROOF_MUTATION` default to `healthy` and
`none`. Run the following with an ordinary Python interpreter; it launches the
existing Windows-only runner hidden, discards standard streams, filters common
credential-variable names, refuses an already-existing receipt and bounds the
whole controller, worker, export and cleanup to the same 60 seconds. The controller
reuses the existing rehearsal kill-on-close Job; it must itself be a disposable
hidden subprocess. It never closes that Job while it contains the controller.
Use a unique receipt destination without concurrent
writers; this preflight is not an atomic filesystem reservation:

```python
import json
import os
from pathlib import Path
import subprocess

receipt = Path(os.environ["HERMES_PROOF_RECEIPT"])
if receipt.exists():
    raise RuntimeError("receipt-exists")
command = [
    os.environ["HERMES_PROOF_PYTHON"], "-I", "-S", "-B",
    "tests/offline_plugin_generation_proof.py",
    "--hermes-source", os.environ["HERMES_PROOF_SOURCE"],
    "--receipt", str(receipt),
    "--case", os.environ.get("HERMES_PROOF_CASE", "healthy"),
    "--mutation", os.environ.get("HERMES_PROOF_MUTATION", "none"),
]
environment = {
    key: value for key, value in os.environ.items()
    if not any(part in key.upper() for part in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))
}
category = "launch-refused"
try:
    result = subprocess.run(
        command, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    category = "child-zero" if result.returncode == 0 else "child-nonzero"
except subprocess.TimeoutExpired:
    category = "deadline-expired"
finally:
    print("[plugin-proof-launch] " + json.dumps({"category": category}))
```

The controller is the sole overall-success authority. The internal worker writes
`<receipt>.worker.json` with `ok: false` and separate semantic results, then exits
normally so Hermes/Python finalizers release process-owned handles. The controller
retains its exact child handle through exit and requires matching nonce, source
bindings, semantic success and no live/unknown survivors or sticky settlement refusal from
the existing process owner before one profile deletion. Only successful deletion permits the final receipt to become
true. Timeout, abnormal exit, bad evidence or unsettled execution preserves the
profile and overall false. There is no per-resource teardown fallback or retry.
The parent also checks a nonce-bound audit sidecar after worker exit. Protected
network/process denials persist fixed counters atomically throughout worker
lifetime; write failure terminates the proof worker nonzero. Missing, malformed,
stale or adverse audit evidence prevents success, including an event occurring
after the worker's semantic receipt. Known blocked metadata probes are reported
separately from unclassified provider attempts. Optional auxiliary title generation
is disabled only in this model-free fixture; primary calls retain the stand-in.
The outer 60-second caller is mandatory; it bounds cleanup as well as the worker.
Linux product behavior is unchanged; this opt-in owner composition is Windows-only.

Inspect the bounded receipt and worker receipt, not just the launch category. Setup refusal, timeout,
missing receipt and an unexpected assertion are failures, never permission to
retry or repair automatically. `--help` lists cases; mutations are defined in
`tests/generation_policy_mutations.py` and affect only the copied disposable
policy. For example, `rewrite` with mutation `final-arguments` must fail
`exact-policy-effects`; the same mutation with `healthy` or `readiness` must pass.
The runner requires isolated, no-site, no-bytecode startup and rejects optimized
Python. It verifies upstream origin/HEAD/unchanged tracked Python, then exports
committed Git blobs into its disposable profile. Only the exact pinned
`contributors/emails` attribution-metadata subtree is omitted: its case aliases
cannot be represented on Windows. The receipt binds the full tree, the omitted
subtree OID `04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d`, its count and the projected
payload manifest. No runtime source is excluded; no collision check is relaxed.

The original checkout, its ignored files and caches never supply runtime imports.
The runner derives the existing venv site directory from the lexical interpreter
path, without relying on Python 3.11 `-S` prefixes or resolving a Linux venv symlink
to the base interpreter. It adds that prepared dependency directory without
processing `.pth` files. Unsupported finders or preloaded upstream modules refuse.
Loaded upstream module origins must stay inside the projected source. The prepared
interpreter/dependencies remain trusted prerequisites, not attested outputs.
Reviewed proof helpers execute their exact source bytes; only named reviewed
fixture files are copied, so neighboring extension/bytecode overlays cannot enter.
Their actual executed bytes and the projection are bound in the receipt.

Content-free stage checkpoints are retained before expensive phases. Intermediate
snapshots always say `ok: false`; overall success requires final origin checks,
profile deletion and final evidence publication. Atomic replacement preserves the
preceding complete snapshot when evidence writing is interrupted.

After complete blob/path validation, the controller prepares parent directories
and materializes the finite input with two standard-library worker threads.
Ordered map results preserve the manifest; the executor joins before an error can
reach profile cleanup. `*_work_seconds` aggregate overlapping work and must not be
added to wall durations; `source_export_materialization_wall_seconds` reports the
materialization interval. There is no parallel cleanup or source cache.

The audit barrier denies subprocesses and non-test
connections, except narrow stdlib loopback self-pipe setup. Denied constructor
metadata/pricing attempts are classified; unclassified attempts must be zero.
The outer deadline bounds the proof child; it is not evidence of descendant
settlement if interrupted during Git preparation. No complete process-owner
cleanup qualification is claimed by this command.

## Finite defensive-guard map

`tests/generation_policy_checks.py` is the fixed direct-policy target map. Each
named term uses the same candidate class, installed registration handles and
public approval context, with synthetic request/callback/result inputs. Its
matrix runs a baseline, one source-copy omission and two healthy controls for
each target. The final installer group uses a public, disposable registration
override to verify unsupported captured handler shape is refused and restored.
These checks do not dispatch native Runs or invoke a provider.

Three term omissions are explicitly non-independent: non-object arguments still
fail argument access, missing tickets still fail subsequent ticket access, and
non-string extraction URLs still fail returned-string membership. Their baseline
refusal remains required; their surviving removals are not called killed mutants.
Exact-type validation remains in the policy. This map does not claim every
syntactic boolean is independently observable or prove the full product.

Source commit `e7c53c1a8de29cfd180d1ade0045d2a49e6083a4` completed the
source-bound 49-group direct matrix in 16.797 seconds, including profile cleanup.
All expected outcomes matched: 46 isolated omissions changed the target outcome
with two healthy controls each; the three redundant removals retained refusal.
The direct replay target includes same-version changed-objective conflict and
retained budget identity. The receipt verified 194 upstream origins and the same
9,990-file manifest, with zero provider/native Run/denied-network attempts.
Receipt SHA-256: `340f5c84f514af185950418d1e456ba9ccf77e8f603b7b7f58b356cd316d1d03`.

The first subsequent native smoke completed one native Run with three synthetic
continuations and passed its semantic checks, but profile deletion raised
`PermissionError`. The child exited 1 with overall `ok: false` and cleanup refused;
the smoke **failed**. The failing file/operation was not retained and cannot be
identified from remaining files alone. The remaining four smoke cases were not
run. Receipt SHA-256:
`4c5acb998c902eaa84a252b1030271f83b1a57fa1bf0fb457c708c7513fc4136`.
The private parent mislabeled its category as `passed` while preserving exit 1
and `ok: false`; that classifier was corrected separately without rewriting the
failed receipt. This PR stays draft; native smoke, cleanup and #329 remain held.

Later exception-derived diagnostics identified an owned logging lock, then an
owned database sharing violation. The supported logging queue drain was added;
no private database or agent state was patched. The ownership correction moves
profile deletion after normal worker exit rather than guessing individual handle
owners.

The first parent-owned attempt at source commit
`c1529d711f86410e41684f7e3ad7d6936f96cf98` exited overall false in 15.547 seconds.
Its worker exited normally, completed one native Run and passed semantic checks;
the controller refused before deletion and preserved the profile. All retained
source/envelope values match, but the expected nonce and Job snapshot were not
retained, so the exact refusing guard is unknown. Receipt SHA-256:
`8b70982d8343111036d1638b630f7b92518f54c40c68c70678ce81893d86656a`.
The diagnostic correction records only named refusal categories, nonce-match and
Job count/self-presence. No guard is weakened and no cleanup pass is claimed.

The named diagnostic at `f88e46b6d41fdcf013489b65ed7ee6153d1a02f5` refused
with two Job members after normal worker exit; the nonce matched. Raw membership
was an incorrect execution-settlement proxy because exited process objects can
remain listed. The correction reuses `Processes.survivors(containment)`:
retained exited members are allowed, live/unknown members and sticky stop/refusal
state remain failures. No member is filtered by a new observer or guessed wait.
The failed diagnostic and preserved profile are not relabeled a cleanup pass.

At `54e19192ae0c7833da7c4fc896fe0a18fce16ef7`, canonical process settlement
and profile deletion succeeded in 21.234 seconds. That record is **not a qualified
proof**: one unclassified blocked provider attempt occurred after the semantic
check. The final audit now gates parent acceptance after process exit. The fixture
also disables the supported optional title-generation lane; the truncated earlier
stack does not prove that lane caused the historical attempt. Earlier receipts
remain unchanged, including SHA-256
`6491e3b254b574a571159ae22226bc60b3be8a65712788b5aadae9a0cc3a4e7e`.

Earlier 60-second failures remain preserved, including a cleanup timeout and a
later interruption after 7,992 files in the serial materialization/per-file-hash
region. No deadline was extended. The corrective parallel-export experiment
preserved the exact payload; its observed success is not a universal latency or
filesystem-cause claim. The worker tests include a real serial-source RED,
candidate GREEN and three isolated mutations with two healthy controls each.
The affected source test file passed all 43 tests after the correction.

## Preserved evidence boundary

Prior receipts remain outside the repository. They contain no public deployment
instructions and are not replaced by this PR. Three separately source-bound
groups reached their expected outcomes:

| Group | Expected outcomes | SHA-256 of matrix |
| --- | --- | --- |
| Main | 99: 30 baselines, 23 isolated mutations, 46 controls | `2fc2b1e21ffdabe35f6f29921957b72910fbfde458d8857e9e159e61d0ac8b9b` |
| Alias correction | 17: 8 baselines, 3 isolated mutations, 6 controls | `2e8caf397143195a663709722c34361195a484d23aadc2d26f72f70c0798bf21` |
| Alias authorization | 4: baseline, repeated authorization mutation, 2 controls | `2061edf29366b931d6e7161a31679ec280d36e4f2d9a1ae79ff5db7917c4f8b8` |

All three groups predate the clean-source-export correction. Their original
HEAD/tracked-file checks did not exclude ignored import overlays; those historical
receipts are not retroactively given the stronger import-source qualification.
The main group also predates the small alias correction, with its own archived policy
SHA `c6f467b9ff7d0b81a04797385766500939cd1b3f1b33d17aa362b46358d422e1`.
The alias groups bind current policy SHA
`7967d036cb9117f8a927cd950d81f690307f7ef49700fbf558228fde1e34733b`.
All groups bind installer SHA
`eb998e0b8a517aa5bd6e905d17bbd0dd873e2a96961379743fafdc3bf33ac282`.
The deliberately unsupported transport ended natively `failed` before provider
continuation, with zero provider/effect calls; this is an unqualified negative.
Exploratory setup and fixture failures were retained, not relabeled guard REDs.

The actual built-in sync/async handlers were captured and wrapped without calling
them or their availability probes. Normal lazy registration retained wrapper
identity; native budget-builder threshold comparisons, including overrides,
matched before/after registration. Synthetic tools still supply budget-case
results; no real retrieval or model continuation was invoked.

Actual native-Hermes generation replacement is the next integration gate. The
separate existing-owner witness used stdlib children and proves neither plugin
readiness nor native worker settlement. A follow-up must compose that owner with
the native fixture, retain old process handles and policy records until verified
owned shutdown, then accept a different, ready generation in the same disposable
profile. This PR does not add that composition or claim full #306 acceptance.
