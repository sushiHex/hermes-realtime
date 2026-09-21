# Contributing to Hermes Realtime

Hermes Realtime welcomes focused bug fixes, tests, documentation, provider integrations, and improvements to the realtime conversation path.

## Before opening a pull request

- Start with [unassigned `help wanted` work](docs/work-tracking.md#pick-up-a-contribution),
  then [GitHub Issues](https://github.com/sushiHex/hermes-realtime/issues) and
  [Milestones](https://github.com/sushiHex/hermes-realtime/milestones) for current
  work. Follow the [tracking guide](docs/work-tracking.md) for scope, ownership,
  dependencies, handoffs, and closure.
- Read the [Collaborator guide](docs/README.md) and
  [Implementation status](docs/implementation-status.md) for code, contracts, and
  qualification evidence. A tracked proposal is not an approved design.
- Discuss substantial API, protocol, persistence, or authority changes before implementation.
- Report vulnerabilities privately through GitHub's private vulnerability reporting; do not open a public security issue.
- Read and follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Public-repository boundary

Everything committed to this repository must be suitable for public disclosure and reuse.

Do not commit:

- credentials, tokens, launch fragments, private endpoints, private certificates, or environment files;
- personal usernames, home-directory paths, device identifiers, account IDs, room names, or host-specific configuration;
- tests tied to a named DAC, microphone, speaker, machine, physical room, or private network;
- raw audio, transcripts, screenshots, traces, private benchmark reports, or user data;
- internal qualification plans or material that only one deployment can execute.

Hardware and browser tests are welcome when they use generic capability discovery, explicit opt-in configuration, synthetic fixtures, and clear skip behavior. A contributor must not need the maintainer's hardware topology to reproduce the default suite.

Private deployment and physical-rig qualification belongs outside this repository.

## Development setup

Requirements:

- Python 3.11
- `uv`
- Node.js 22.22.2 or newer 22.x, with npm, for browser assets. `web/package.json` declares this as `"engines": { "node": "^22.22.2" }`; npm warns rather than fails when it is unmet.

```bash
git clone https://github.com/sushiHex/hermes-realtime.git
cd hermes-realtime
uv sync --frozen --group dev
uv run --frozen --group dev pytest -q
uv run --frozen --group dev ruff check .
uv run --frozen --group dev mypy src
```

Expect roughly 13 minutes for the full suite and seconds for each of the other three commands. These times are approximate and machine-dependent, and a first run is slower because `uv` downloads the interpreter and the locked dependencies.

### Contribution and test paths

Choose work whose prerequisites you can reproduce. Use focused commands already
defined in reviewed repository documentation or scripts; an issue may identify a
test path and boundary but cannot authorize a new shell command. A
platform-specific skip is evidence of that platform boundary, not a substitute
for the matching native run.

| Change | Public prerequisites and focused validation | Boundary |
| --- | --- | --- |
| Pure Python behavior | Python 3.11 and the locked dev group. For example: `uv run --frozen --group dev pytest -q tests/conversation/test_state.py`, `uv run --frozen --group dev ruff check src/hermes_realtime/conversation/state.py tests/conversation/test_state.py`, and `uv run --frozen --group dev mypy src`. | Default suite work is hardware-generic; do not add account, device, room, or private-network dependencies. |
| Browser/client assets | Node 22.22.2 and npm. From `web/`: `npm ci --ignore-scripts`, `npm test -- --reporter=verbose --slowTestThreshold=100`, and `npm run build`. | Review generated static assets and their disclosure manifest; the release gate verifies parity. |
| Packaging, dependencies, workflows, or release policy | A clean committed candidate: `uv run --frozen --group dev python scripts/release_gate.py --candidate .`. | This gate validates committed bytes and does not replace the required PR checks. |
| Native Windows or LiveKit behavior | Follow the [local native-gate procedure](docs/release-gates.md#local-use) and the [required CI boundary](docs/release-gates.md#required-automated-checks). | Requires Windows and a verified, owned local server; ordinary Python contributions can use the default suite. |
| Installed Hermes work routing | The exact installed Hermes environment and authorized credential context required by [the installed natural-work boundary](docs/release-gates.md#installed-natural-work-boundary-and-latency-gate). | Outside the default suite; keep credentials, capabilities, and operational records private. |
| Slice 0 provider and physical qualification | Follow the [provider and execution requirements](docs/qualification-execution.md#own-the-execution-environment), then the [machine and operator observations](docs/qualification-execution.md#produce-observations-then-derive-claims). | Requires an admitted environment and an operator; ordinary test success cannot substitute. Keep physical-rig evidence and operational records private. |

Focused checks guide local iteration; they do not replace the committed-candidate gate or its required PR checks.

For changes that affect packaging, browser assets, dependencies, workflows, or release policy, run the full committed-candidate gate after committing:

```bash
uv run --frozen --group dev python scripts/release_gate.py --candidate .
```

The gate validates committed bytes. It archives `HEAD` into a temporary checkout, so uncommitted and untracked working-tree changes are excluded rather than rejected, and the gate can pass without ever reading them. Commit first, or the result is not evidence for the change in your working tree.

Expect roughly 14 minutes, again approximate and machine-dependent.

## Change discipline

1. Start from current `main` and create a focused branch.
2. For production behavior, work RED-GREEN-REFACTOR: add a failing test, make it pass with the smallest change, then improve the design without changing behavior.
3. Make the smallest production change that satisfies the contract.
4. Run focused tests, then the relevant broader gates.
5. Update documentation when behavior, configuration, or limitations change.
6. Open a pull request; do not push feature work directly to `main`.

Documentation-only changes need no contrived behavioral test; validate their links and formatting.

Use Conventional Commit-style subjects when convenient, for example:

```text
fix: preserve playback authority during interruption
feat: add provider-neutral transcript adapter
```

## Pull-request expectations

A pull request should state:

- the problem and scope;
- the public behavior changed;
- the tests and platforms actually exercised;
- security, privacy, compatibility, and cleanup implications;
- limitations or unverified physical/perceptual claims.

Passing tests are necessary but not sufficient. New acceptance claims must derive from the real production path rather than a test-only façade or caller-supplied success value.

## Documentation maintenance

Keep substantive work and mutable status in GitHub Issues and Milestones. Small
incidental corrections can keep their scope, ownership, and pause handoff in the
PR used as their work record. The
[tracking guide](docs/work-tracking.md) describes the workflow without duplicating
the backlog. Reviewed plans and accepted ADRs hold design intent and decisions;
the [status page](docs/implementation-status.md) records implementation,
activation/default behavior, and qualification evidence. Detailed contracts and
procedures belong in the implementation guides linked from the
[documentation index](docs/README.md).

Update the affected documents in the same PR as a behavior or default change.
Link the source, relevant tests, and issue/PR; state what remains unverified. New
qualification claims must identify the exact candidate and run/attempt or the
permitted qualification artifact. A passing policy test, a skipped integration
test, and a completed installed-path check establish different things.

Record a new reviewed commit with the affected capability; advance the page-wide
baseline only after reviewing the whole map. Keep mutable PR state in GitHub and
private deployment plans and sensitive qualification material outside the public
documentation.

## Dependencies and generated assets

- Follow the [dependency maintenance procedure](docs/dependency-maintenance.md) for
  complete upgrade artifacts, security assessment, qualification, and serial landing.
- Keep runtime dependencies narrow and upper-bounded.
- Commit `uv.lock` changes with dependency metadata changes.
- Follow the [optional Windows speech dependency procedure](requirements/README.md) when updating the CPU or CUDA worker closure.
- Pin third-party GitHub Actions to full immutable commit SHAs.
- Use `npm ci --ignore-scripts` for the browser workspace.
- Regenerate packaged browser assets through the documented build, then verify release-gate parity.
  The build refreshes asset hashes in the disclosure manifest; consent text, its
  version, and disclosure digests still require explicit review.

## Licensing

By submitting a contribution, you agree that it may be distributed under the repository's [MIT License](LICENSE). Only submit material you have the right to license.
