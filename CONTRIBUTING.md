# Contributing to Hermes Realtime

Hermes Realtime welcomes focused bug fixes, tests, documentation, provider integrations, and improvements to the realtime conversation path.

## Before opening a pull request

- Use GitHub Issues for reproducible bugs and bounded feature proposals.
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
- Node.js/npm for browser assets

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

## Dependencies and generated assets

- Keep runtime dependencies narrow and upper-bounded.
- Commit `uv.lock` changes with dependency metadata changes.
- Pin third-party GitHub Actions to full immutable commit SHAs.
- Use `npm ci --ignore-scripts` for the browser workspace.
- Regenerate packaged browser assets through the documented build, then verify release-gate parity.

## Licensing

By submitting a contribution, you agree that it may be distributed under the repository's [MIT License](LICENSE). Only submit material you have the right to license.
