# Fresh-clone contributor setup validation

## Scope and binding

This record describes one execution of the contributor setup commands documented in
`CONTRIBUTING.md` and `README.md`, performed against a fresh clone with no inherited build state.

- Commit: `4d41e03ff7c48a4ad205ba3c6c47156c5d85fe28`, checked out detached and confirmed with
  `git rev-parse HEAD`.
- Platform: Windows 11, the repository's primary target.
- Clone path length: 54 characters. Path length is recorded because Windows path limits are a known
  source of build and packaging failures, and a longer clone root is not covered here.
- No repository `.venv` and no `web/node_modules` were inherited. Both were absent immediately after
  the clone and were created only by the documented commands.
- `git status --short` was empty before any documented command ran.

Tool versions:

| Tool | Version |
| --- | --- |
| `git` | 2.53.0.windows.1 |
| `uv` | 0.11.28 |
| `node` | v22.16.0 |
| `npm` | 10.9.2 |
| bare `python` | 3.14.3 |
| interpreter provisioned by `uv sync` | 3.11.15 |

The documented Requirements are Python 3.11, `uv`, and Node.js 22.22.2 or newer 22.x. Two
environment facts follow from the table and bear on the results below. Bare `python` resolves, but
to 3.14.3, which is outside `requires-python = ">=3.11,<3.12"` in `pyproject.toml`; the supported
3.11.15 interpreter is provisioned by `uv` into its managed store and is not on `PATH`. Node is
v22.16.0, below the documented 22.22.2 minimum, so the Node results are provisional.

## Method

Every command was run verbatim, in the order documented, from the fresh clone. Wall-clock elapsed
time was recorded for each. Failures were preserved and the sequence continued; nothing was
repaired, skipped, or substituted to make a later command succeed. No repository file was modified
except the deliberate, reverted probe described under Finding 1.

Clone paths are written as `<clone>` throughout, including inside quoted command output. Gate
temporary directories are written as `<temp>/hermes-realtime-release-<random>`.

## Results

| Command (as documented) | Outcome | Elapsed |
| --- | --- | --- |
| `uv sync --frozen --group dev` | Passed. Provisioned CPython 3.11.15. | 3 s |
| `uv run --frozen --group dev pytest -q` | Passed. `4668 passed, 21 skipped in 771.70s (0:12:51)`. | 773 s |
| `uv run --frozen --group dev ruff check .` | Passed. `All checks passed!` | 0 s |
| `uv run --frozen --group dev mypy src` | Passed. `Success: no issues found in 76 source files`. | 11 s |
| `uv run --frozen --group dev pytest -q tests/conversation/test_state.py` | Passed. `1 passed in 0.48s`. | 1 s |
| `uv run --frozen --group dev ruff check src/hermes_realtime/conversation/state.py tests/conversation/test_state.py` | Passed. `All checks passed!` | 0 s |
| `npm ci --ignore-scripts` (from `<clone>/web`) | Passed with four `EBADENGINE` warnings. See Finding 4. | 5 s |
| `npm test -- --reporter=verbose --slowTestThreshold=100` (from `<clone>/web`) | Passed. `Test Files 9 passed (9)`, `Tests 163 passed (163)`. | 4 s |
| `npm run build` (from `<clone>/web`) | Passed. Working tree remained clean. See Finding 5. | 1 s |
| `git status --short` after `npm run build` | Empty. | — |
| `python scripts/release_gate.py --candidate .` | Passed. `release gate passed`. | 831 s |
| `uv run --frozen --group dev python scripts/release_gate.py --candidate .` | Passed. `release gate passed`. | 753 s |
| `python scripts/release_gate.py --candidate .` against a deliberately invalid working tree | Passed. `release gate passed`. See Finding 1. | 823 s |
| `uv build` | Passed. Built `hermes_realtime-0.0.3.tar.gz` and `hermes_realtime-0.0.3-py3-none-any.whl`. | 3 s |
| `python -m pip install dist/hermes_realtime-*.whl` (PowerShell) | Failed. Glob not expanded. See Finding 2. | 1 s |
| `python -m pip install dist/hermes_realtime-*.whl` (Git Bash) | Failed. Interpreter version rejected. See Finding 2. | 1 s |
| `python -c "import hermes_cli, sys; print(sys.executable)"` | Failed. `ModuleNotFoundError: No module named 'hermes_cli'`. See Finding 3. | 0 s |

The documented command `python -m pip install dist/hermes_realtime-*.whl` is listed twice because it
behaves differently in the two shells a Windows contributor is likely to use, and the difference is
the finding. The release gate is listed three times: as documented, under the supported interpreter,
and against a modified working tree.

The 21 skips in the default suite are 14 for the local LiveKit boundary, 4 for the absent
`onnxruntime` provider dependency, 1 for the Linux null-capture gate, 1 for the opt-in browser gate,
and 1 reporting `this account cannot create symbolic links`. The last of these is an account
privilege condition on the audit machine rather than a repository property.

Elapsed times are reported to the nearest second and are lower bounds. See Limitations.

## Findings

Each finding records what a contributor observes, the evidence, and the smallest correction that
would address it. Every correction below is a proposal. None has been applied; this record changes
no repository behaviour and no repository file other than itself.

### Finding 1 — the committed-candidate gate passes on a modified working tree

`CONTRIBUTING.md` line 77 states:

> The gate intentionally refuses dirty or untracked candidate bytes.

The gate does not refuse. It ignores the working tree and validates committed bytes.

To separate "ignored the working tree" from "read it and found nothing wrong", the probe used an
edit that three checks the gate already runs would each reject. The line `def (` was appended to
`src/hermes_realtime/protocol/events.py`, the innermost module of the package. Locally,
`uv run --frozen --group dev ruff check src/hermes_realtime/protocol/events.py` reported
`Found 2 errors.`, and any import of the module raises `SyntaxError`.

With that edit present and `git status --short` reporting `M src/hermes_realtime/protocol/events.py`,
the documented gate command was run again. It completed with exit status 0 and printed:

```
release gate passed
```

The gate's internal `pytest` run reached completion without a collection error, which it could not
have done had the modified file been read.

The first line of output is identical across the clean run and the modified run:

```
canonical candidate diff SHA-256: 253b3796d22a0accc7611f20096a3f8f25721aa7e3c4311c89e3091e45f3c1ea
```

The digest is computed from the committed diff, so it does not change when the working tree changes,
and it therefore offers no signal that an uncommitted edit was excluded.

The probe was reverted with `git checkout -- src/hermes_realtime/protocol/events.py` immediately
after the run. `git status --porcelain --untracked-files=all` was then empty and `git rev-parse HEAD`
still reported `4d41e03ff7c48a4ad205ba3c6c47156c5d85fe28`.

The implementation describes the behaviour correctly. `scripts/release_gate.py` lines 5 to 8 read:

```
The gate deliberately archives the selected Git revision into a new temporary
checkout.  This prevents ignored build output, virtual environments, and other
ambient worktree directories from affecting a release artifact.
```

`scripts/release_gate.py` contains no working-tree status check; `git status`, `git diff-index`, and
`git ls-files` do not appear in it. The candidate is materialised by `git_archive`, which runs
`git archive --format=tar HEAD`.

Isolation and refusal are different guarantees. The contributor-facing sentence promises the one the
gate does not provide. A contributor who runs the gate on an uncommitted change reads
`release gate passed` as validating the change in front of them.

What a contributor sees: a passing gate, and nothing in the output that distinguishes their working
tree from the commit that was actually validated. The temporary checkout path appears in echoed
subprocess lines, for example
`+ uv lock --check --project <temp>/hermes-realtime-release-<random>/candidate --no-config --python 3.11`,
which indicates a separate checkout but does not name the revision or state that uncommitted work
was excluded.

Proposed corrections, smallest first:

1. Replace the sentence at `CONTRIBUTING.md` line 77 so it describes isolation rather than refusal,
   for example: "The gate validates committed bytes. It archives `HEAD` into a temporary checkout,
   so uncommitted and untracked working-tree changes are excluded rather than rejected. Commit
   first, or the result is not evidence for the change in your working tree."
2. Additionally, print the archived revision alongside the existing digest, so the passing output
   names the bytes that were validated.

The second proposal changes behaviour and is offered only as a follow-up for the review owner's
judgement; the documentation correction alone resolves the contradiction.

### Finding 2 — the documented wheel install fails in both common Windows shells

`README.md` line 84 documents:

```
python -m pip install dist/hermes_realtime-*.whl
```

In PowerShell the glob is not expanded for a native executable, so `pip` receives the literal
pattern:

```
WARNING: Requirement 'dist/hermes_realtime-*.whl' looks like a filename, but the file does not exist
ERROR: Invalid wheel filename (wrong number of parts): 'hermes_realtime-*'
```

In Git Bash the glob does expand, to `dist/hermes_realtime-0.0.3-py3-none-any.whl`, and the command
then fails on the interpreter instead:

```
ERROR: Package 'hermes-realtime' requires a different Python: 3.14.3 not in '<3.12,>=3.11'
```

Both runs exited 1. The command is presented in a `bash`-style fenced block, immediately after
`uv build`, in a repository whose primary target is Windows.

What a contributor sees: an error whose text depends on their shell, neither of which points at the
actual requirement, which is a Python 3.11 environment holding the built wheel.

Proposed correction: document a form that is shell-independent and names the interpreter, for
example `uv pip install --python 3.11 dist/hermes_realtime-0.0.3-py3-none-any.whl`, or instruct the
contributor to substitute the concrete filename printed by `uv build`. `README.md` line 98 already
shows the `uv pip install --python <path-to-hermes-python>` form for the isolated-environment case,
so the vocabulary exists.

### Finding 3 — three documented commands use bare `python` while every other command uses `uv run`

`README.md` lines 84, 92, and 186, and `CONTRIBUTING.md` lines 64 and 74, use a bare `python`. Every
other documented command is `uv run --frozen --group dev ...`, which binds the supported interpreter.

On this machine bare `python` resolves to 3.14.3, outside `requires-python = ">=3.11,<3.12"`. The
supported 3.11.15 interpreter exists but lives in the `uv` managed store and is not on `PATH`. A
contributor who follows the Requirements exactly, installing `uv` and letting it provide Python, has
no bare `python` bound to the supported version; a contributor with an unrelated system Python has
one bound to the wrong version. Present and wrong fails less legibly than absent.

The three commands behave differently:

- `python scripts/release_gate.py --candidate .` succeeded under 3.14.3. The script imports only the
  standard library, carries no `sys.version_info` guard, and delegates the actual work to
  subprocesses that pin the interpreter themselves, for example `scripts/release_gate.py` line 788,
  `uv sync --frozen --python 3.11 --dev`. The same command under
  `uv run --frozen --group dev python ...` also passed, so the gate's result does not depend on which
  interpreter launches it. The documented command therefore works, but by circumstance rather than by
  construction.
- `python -m pip install dist/hermes_realtime-*.whl` failed. See Finding 2.
- `python -c "import hermes_cli, sys; print(sys.executable)"` failed with
  `ModuleNotFoundError: No module named 'hermes_cli'`. `README.md` lines 89 to 90 frame this as an
  optional packaging check to be run with "the Python environment that owns your `hermes` command",
  so the failure is consistent with the documentation on a machine where Hermes is not installed. It
  is recorded as a result rather than a defect.

Proposed correction: state in the Requirements that the documented bare-`python` commands need a
Python 3.11 interpreter on `PATH`, and note that `uv`-provided Python does not satisfy that. For the
gate specifically, the smallest change is to document
`uv run --frozen --group dev python scripts/release_gate.py --candidate .`, which binds the supported
interpreter and was observed to produce the same result.

### Finding 4 — `npm ci` reports an engine constraint the Requirements do not mention

`npm ci --ignore-scripts` succeeded but emitted four warnings, the first of which is:

```
npm warn EBADENGINE Unsupported engine {
npm warn EBADENGINE   package: 'hermes-realtime-client@0.0.3',
npm warn EBADENGINE   required: { node: '^22.22.2' },
npm warn EBADENGINE   current: { node: 'v22.16.0', npm: '10.9.2' }
npm warn EBADENGINE }
```

The remaining three concern `jsdom@30.0.1`, `machina@7.0.1`, and `undici@8.10.2`.

This reflects the audit environment, which is below the documented Node minimum, not a repository
defect. It is recorded because the warning is visible only to a contributor who does not meet the
documented minimum, `npm` does not fail on it without `engine-strict`, and `CONTRIBUTING.md` states
the Node requirement in prose without mentioning that `web/package.json` encodes it as an `engines`
constraint. A contributor on an older Node therefore gets warnings rather than a stop, and the
subsequent commands appear to succeed.

Proposed correction: note in the Requirements that `web/package.json` declares
`"engines": { "node": "^22.22.2" }` and that `npm` warns rather than fails when it is unmet.

### Finding 5 — conditions tested that did not reproduce as problems

Two conditions were tested and not observed. They are recorded so the negative results are available.

`npm run build` did not leave the tree dirty. `web/build.mjs` writes into
`src/hermes_realtime/client/static/`, which is tracked, and it rewrites all four assets plus the
disclosure manifest on every run: `assets/app.js` and `assets/app.js.LEGAL.txt` are written,
`index.html` and `assets/styles.css` are copied from `web/`, and
`src/hermes_realtime/evidence/disclosure_manifest_v1.json` is rewritten with fresh digests. After the
documented build on a fresh clone at this commit, `git status --porcelain --untracked-files=all` was
empty, so the committed assets match a fresh regeneration byte for byte. A contributor at this commit
can distinguish an intended regeneration from an accidental one, because any output at all from
`git status` means the assets changed.

`README.md` line 109 uses `uv sync --frozen --dev --extra local` where `README.md` lines 77 and 177
and `CONTRIBUTING.md` use `--group dev`. This is a naming inconsistency, not a functional difference:
`uv`'s `--dev` selects the `dev` group, and `scripts/release_gate.py` line 788 also uses `--dev`. The
command was not run, because `--extra local` pulls provider dependencies outside the scope of this
record. No correction is proposed beyond using one spelling consistently, should the review owner
consider it worthwhile.

## Not attempted

Four documented paths were identified and not executed. None is counted as passing.

- **Native Windows or LiveKit behaviour**, per the local native-gate procedure in
  `docs/release-gates.md`. This requires a verified, owned local LiveKit development server on a
  pinned build. `HERMES_REALTIME_LIVEKIT_LOCAL` was not set, and the default suite's skips record the
  boundary explicitly, for example
  `set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit`. The release gate was run
  without `--require-livekit`.
- **Installed Hermes work routing**, per the installed natural-work boundary in
  `docs/release-gates.md`. This requires the exact installed Hermes environment and an authorized
  credential context, neither of which is available to a contributor working from a clone alone.
- **Slice 0 provider and physical qualification**, per `docs/qualification-execution.md`. This
  requires an admitted execution environment and an operator making physical observations. Ordinary
  test success cannot substitute for it.
- **`README.md` § Local conversation profile.** Its prerequisites are the pinned local LiveKit
  development server, Ollama with a selected model, FFmpeg on `PATH`, and network access for Edge
  TTS. It also installs the `local` extra's provider dependencies.

## Limitations

- The `uv` and `npm` caches were warm. Every elapsed time is a lower bound against a contributor's
  genuine first run, which must download the Python interpreter, the locked Python dependencies, and
  the npm packages. The two long rows, the full suite at 773 seconds and the release gate at 831
  seconds, are the least affected, because their cost is dominated by execution rather than download.
- One platform only. This is Windows 11, the repository's primary target. No Linux or macOS
  contributor path was exercised.
- One clone path length only, 54 characters. Longer clone roots, which are the usual trigger for
  Windows path-length failures, were not tested.
- The Node toolchain was v22.16.0, below the documented 22.22.2 minimum. The three `npm` rows and the
  gate's internal `npm ci`, `npm test`, and `npm run build` steps all ran on that version, so those
  outcomes are provisional for a contributor who meets the documented minimum.
- Bare `python` resolved to an interpreter that a contributor following only the documented
  Requirements would not necessarily have. The bare-`python` rows describe this machine, not every
  machine.
- One skip, `this account cannot create symbolic links`, reflects an account privilege on the audit
  machine. A contributor whose account holds that privilege will run that test rather than skip it,
  and it was not exercised here.
- This record validates the documented contributor path. It does not qualify a release candidate,
  does not substitute for the required automated checks, and does not cover any path listed under Not
  attempted.
