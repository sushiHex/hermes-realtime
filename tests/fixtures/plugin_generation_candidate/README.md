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
- One complete batch of 1–5 searches may be followed by one extraction of 1–3
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
environment variables: `HERMES_PROOF_PYTHON` (that environment's interpreter),
`HERMES_PROOF_SOURCE` (its checkout), and `HERMES_PROOF_RECEIPT` (a new output file).
Optional `HERMES_PROOF_CASE` and `HERMES_PROOF_MUTATION` default to `healthy` and
`none`. Run the following with an ordinary Python interpreter; it launches the
existing runner hidden on Windows, discards standard streams, filters common
credential-variable names, refuses an already-existing receipt and bounds the
whole child to 60 seconds. Use a unique receipt destination without concurrent
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
    os.environ["HERMES_PROOF_PYTHON"], "tests/offline_plugin_generation_proof.py",
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

Inspect the bounded receipt, not just the launch category. Setup refusal, timeout,
missing receipt and an unexpected assertion are failures, never permission to
retry or repair automatically. `--help` lists cases; mutations are defined in
`tests/generation_policy_mutations.py` and affect only the copied disposable
policy. For example, `rewrite` with mutation `final-arguments` must fail
`exact-policy-effects`; the same mutation with `healthy` or `readiness` must pass.
The runner rejects optimized Python, verifies upstream origin/HEAD/unchanged
tracked Python before importing it, verifies native import origin and binds its
sources in the receipt. Its audit barrier denies subprocesses and non-test
connections, except narrow stdlib loopback self-pipe setup. Denied constructor
metadata/pricing attempts are classified; unclassified attempts must be zero.

## Preserved evidence boundary

Prior receipts remain outside the repository. They contain no public deployment
instructions and are not replaced by this PR. Three separately source-bound
groups reached their expected outcomes:

| Group | Expected outcomes | SHA-256 of matrix |
| --- | --- | --- |
| Main | 99: 30 baselines, 23 isolated mutations, 46 controls | `2fc2b1e21ffdabe35f6f29921957b72910fbfde458d8857e9e159e61d0ac8b9b` |
| Alias correction | 17: 8 baselines, 3 isolated mutations, 6 controls | `2e8caf397143195a663709722c34361195a484d23aadc2d26f72f70c0798bf21` |
| Alias authorization | 4: baseline, repeated authorization mutation, 2 controls | `2061edf29366b931d6e7161a31679ec280d36e4f2d9a1ae79ff5db7917c4f8b8` |

The main group predates the small alias correction, with its own archived policy
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
