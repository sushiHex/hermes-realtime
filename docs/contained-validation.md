# Deadline-bound local validation on Windows

Use the existing `scripts.qualification_launch.run_owned_controller` API when a
local validation command needs an execution deadline and owned descendant cleanup.
Do not put a timeout around `uv run ... pytest` and treat the wrapper's exit as
settlement of Python and its descendants. This procedure reuses the existing
guardian; it adds no launcher or provider-qualification path.

## Prerequisites

- A trusted checkout and the Python 3.11 development environment prepared with
  `uv sync --frozen --group dev`.
- A Windows host that permits the guardian's explicit breakaway and verifies it
  is outside inherited Jobs. Unsupported hosts refuse; there is no attached or
  uncontained fallback.
- A fresh private evidence directory outside the checkout. Keep the receipt and
  its adjacent `.work` directory, including after failure or caller interruption.
- A fixed test selection and an appropriate deadline. Interpreter hashes do not
  attest the dependency closure, model budget, or every imported source file.

For short focused iteration, the ordinary commands in [CONTRIBUTING](../CONTRIBUTING.md)
remain available; they do not themselves claim a bounded process-tree lifetime.
The existing [release-gate procedure](release-gates.md) remains authoritative for
release qualification and its owned native commands.

## Run a focused pytest selection

Run this from the intended checkout. Replace the evidence-directory placeholder
with a new absolute private path whose parent exists. The example checks import
origins, creates the existing version-1 interpreter binding, and uses the same
dependency environment for a three-line pytest controller. That controller only
converts pytest's enum exit value to an integer; it contains no launching or
containment code. The guardian's strict completion rules remain unchanged.

```powershell
$validationRoot = (Get-Location).Path
$validationPython = (Resolve-Path .venv/Scripts/python.exe).Path
$validationEvidence = "REPLACE_WITH_ABSOLUTE_NEW_PRIVATE_DIRECTORY"
$validationPriorPath = $env:PYTHONPATH
$env:PYTHONPATH = "$validationRoot;$(Join-Path $validationRoot src)"
try {
@'
import hashlib
import json
import sys
from pathlib import Path

import hermes_realtime
import pytest
from scripts import qualification_launch as launch

root = Path.cwd().resolve()
assert Path(launch.__file__).resolve() == root / "scripts/qualification_launch.py"
assert Path(hermes_realtime.__file__).resolve() == root / "src/hermes_realtime/__init__.py"
assert Path(pytest.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
evidence = Path(sys.argv[1])
assert evidence.is_absolute()
evidence = evidence.resolve()
assert evidence != root and root not in evidence.parents
evidence.mkdir(exist_ok=False)
controller = evidence / "pytest-controller.private.py"
with controller.open("x", encoding="utf-8") as stream:
    stream.write("import sys\nimport pytest\nraise SystemExit(int(pytest.main(sys.argv[1:])))\n")

def interpreter(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

binding = {"schema_version": 1,
           "guardian": interpreter(sys._base_executable),
           "controller": interpreter(sys.executable)}
raw = json.dumps(binding, sort_keys=True).encode()
binding_path = evidence / "interpreter-binding.private.json"
with binding_path.open("xb") as stream:
    stream.write(raw)
receipt = evidence / "validation.private.jsonl"
result = launch.run_owned_controller(
    controller, ("-q", "tests/conversation/test_state.py"),
    binding_path=binding_path,
    binding_sha256=hashlib.sha256(raw).hexdigest(),
    receipt=receipt, timeout_seconds=120,
)
final = json.loads(receipt.read_bytes().splitlines()[-1])
print(json.dumps({"pytest_exit_code": result.exit_code,
                  "stage": final["stage"],
                  "controller_completed": final["controller_completed"],
                  "cleanup_verified": final["cleanup_verified"]}))
raise SystemExit(result.exit_code)
'@ | & $validationPython -B - $validationEvidence
} finally {
    $env:PYTHONPATH = $validationPriorPath
}
```

Check the Python process's exit code as well as its final receipt. The PowerShell
block's completion is not a test or cleanup result. Select a different reviewed
test path or deadline deliberately; do not silently repeat a timed-out full suite.
Do not substitute a `uv` wrapper for the bound controller interpreter.

The import-only checks prevent a sibling editable checkout from silently supplying
the application or launcher. The guardian additionally hashes the selected
controller and launcher/kernel inputs. Keep the trusted source and dependencies
fixed throughout execution; this is not protection against hostile concurrent
interpreter or dependency replacement. Do not add model/provider tests without
their separate configuration, privacy and resource authorization.

## Interpret the evidence

| Observation | Meaning |
| --- | --- |
| `stage: complete`, `controller_completed: true`, `cleanup_verified: true` | Normal controller completion and observed owned-Job quiescence. Read the exit code separately. |
| Pytest exit code `0` | The selected tests passed. It is meaningful alongside the completion and cleanup evidence. |
| Nonzero pytest exit with complete containment | Test/collection failure or another pytest outcome; it is not an ownership failure and is not a test pass. |
| Refusal, timeout, missing completion, or uncertain cleanup | Validation did not establish complete execution/settlement. Retain evidence and investigate; do not infer success from the outer process ending. |

The Job owns the controller before it resumes. The guardian retains its own
deadline if the observing caller disappears; loss of the guardian closes its
kill-on-close Job. Do not kill unrelated processes by name, infer ownership from
a process list, or replace this mechanism with another per-session guardian.

The API accepts deadlines of 1–14400 seconds, with up to 30 seconds for cleanup.
Retained stdout and stderr have 8 MiB and 64 KiB acceptance bounds respectively;
polling is not a hard disk-write quota. The Job's memory limits do not establish
GPU capacity or replace vram-mcp admission. See the [source and tests](../tests/test_qualification_launch.py)
and [ownership contract](release-gates.md#opted-in-windows-evaluator-ownership).

Keep paths, process identities, raw logs and any sensitive test data private.
Public reports should retain source identities, exit/result categories, timings
and settlement booleans, not transcript or credential material. The original
failure remains evidence even after a later corrected candidate passes.

This routing guidance does not by itself complete the broader wrapper-versus-tree
timeout proof tracked in [#351](https://github.com/sushiHex/hermes-realtime/issues/351).
