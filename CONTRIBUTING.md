# Contributing to Hermes Realtime

Hermes Realtime welcomes focused bug fixes, tests, documentation, provider integrations, and improvements to the realtime conversation path.

## Before opening a pull request

- Start with [GitHub Issues](https://github.com/sushiHex/hermes-realtime/issues)
  and [Milestones](https://github.com/sushiHex/hermes-realtime/milestones) for current
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
- Node.js 22.22.2 or newer 22.x, with npm, for browser assets

```bash
git clone https://github.com/sushiHex/hermes-realtime.git
cd hermes-realtime
uv sync --frozen --group dev
uv run --frozen --group dev pytest -q
uv run --frozen --group dev ruff check .
uv run --frozen --group dev mypy src
```

For changes that affect packaging, browser assets, dependencies, workflows, or release policy, run the full committed-candidate gate after committing:

```bash
python scripts/release_gate.py --candidate .
```

The gate intentionally refuses dirty or untracked candidate bytes.

## Change discipline

1. Start from current `main` and create a focused branch.
2. Add a failing test for behavior changes where practical.
3. Make the smallest production change that satisfies the contract.
4. Run focused tests, then the relevant broader gates.
5. Update documentation when behavior, configuration, or limitations change.
6. Open a pull request; do not push feature work directly to `main`.

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
