# Dependency maintenance

Dependency upgrades are complete changes, not version-number edits. The branch offered for
review must contain every repository artifact derived from the changed dependency and the
evidence needed for its actual installed profile. Dependabot proposes candidates; it does not
own source selection, generated artifacts, compatibility decisions, or qualification.

## Keep routine proposals bounded

`.github/dependabot.yml` checks the root `uv` and worker `pip` ecosystems weekly and the
browser `npm` and GitHub Actions ecosystems monthly. Each entry permits at most two open
version-update pull requests and applies a seven-day cooldown to version releases. GitHub's
cooldown and version-update limit do not apply to security updates.
[GitHub's Dependabot options reference](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)
defines these native keys and their version-update scope.

Patch and minor version updates are grouped within each ecosystem to reduce review traffic.
Major updates remain separate so a migration has an attributable diff and qualification
record. Vite and esbuild are the narrow exception: Dependabot may propose them together at any
version level because their peer ranges can require a coordinated browser-toolchain move.
Grouping by semantic version level is queue shaping, not evidence that the grouped versions
are compatible.

The root Python patch/minor group leaves `hatchling` and `cryptography` as individual
proposals without ignoring them. A Hatchling minor can require a governed metadata/profile
migration, while `cryptography` is shared with a source-derived qualification pin and can
produce same-major residual noise. Isolating them keeps either proposal from holding an
otherwise routine group; it does not approve, defer, or suppress the change.

The two-pull-request cap bounds new version proposals; changing it does not close proposals
that are already open. Do not leave a deliberately deferred governed migration occupying a
slot indefinitely. Record the bounded follow-up in its GitHub issue, then close the bot
proposal with the reason and successor link. Do not add a blanket ignore merely to clear the
queue, and do not create a Markdown queue or synchronization service.

## Produce a complete upgrade

Review the proposed release notes, supported runtimes, licenses, advisories, and the complete
diff. Then update the artifacts owned by the affected profile:

- A root Python manifest change includes `pyproject.toml` and `uv.lock`.
- A browser change includes `web/package.json`, `web/package-lock.json`, the rebuilt tracked
  client assets, and their disclosure hashes. Run the browser tests and build with the
  repository's supported Node version.
- A GitHub Actions change includes immutable full commit SHAs and the reviewed pin record in
  `tests/test_release_workflow.py`; review the action source and runtime migration represented
  by each SHA.
- An optional speech change follows the
  [Windows speech dependency procedure](../requirements/README.md). The Kokoro package pin,
  worker roots, hashed CUDA closure, package manifest, and `uv.lock` move together when their
  common profile changes. Exercise installed CPU synthesis and the CUDA worker's real provider
  selection and owned cleanup; ordinary required checks do not install the `local` extra and
  therefore do not establish this coverage.
- A Hermes source-profile change starts with the selected upstream source and derives its
  mirrored dependency constraints and lockfile from that authority. The source commit,
  purpose-specific profile, and any affected CPU or CUDA closure receive one coherent
  qualification; an independent bump must not weaken the exact-source guard.

A bot proposal that lacks any required artifact is an input to an owned change, not a branch
to repair. Create a repository-owned branch, adopt the proposed version there, complete and
qualify the artifacts, and close the bot pull request as superseded with a link to the owned
work. Never commit fixes to a Dependabot branch because the bot can force-push it away.

## Assess security reports immediately

Review every dependency alert when it appears, independently of the ordinary version cadence.
Identify the exact advisory, affected locked occurrences, installed and purpose-specific
profiles, direct call sites, upstream source constraints, and any unknown transitive or
operator-enabled paths. Lack of a direct call is not proof of no exposure. Record a reviewed
remediation or bounded risk disposition through the repository's private vulnerability path
when the report is not already public.

Existing source-root ignores are intentionally narrow and must not be broadened to make an
alert or proposal disappear. An ignore can suppress a security update pull request, so monitor
Dependabot alerts themselves and do not infer safety from the absence of a bot branch. The
seven-day cooldown and two-PR version cap do not delay security updates.

Keep proposal noise and exposure assessment separate. The standing `cryptography` major-only
ignore suppresses the observed cross-major proposal but can still permit a later 48.x proposal;
[the residual proposal-noise record](https://github.com/sushiHex/hermes-realtime/issues/158)
owns that limitation. The distinct
[advisory-exposure record](https://github.com/sushiHex/hermes-realtime/issues/160) owns the
affected-profile assessment and disposition. Closing or suppressing one does not resolve the
other.

## Land one qualified candidate at a time

Auto-merge may be enabled only after the complete review and exact-candidate merge
authorization described here; a green bot proposal alone is insufficient.

1. Refresh the selected proposal against current `main`; use a repository-owned branch when
   completing missing artifacts. Review every review thread, review, and conversation comment,
   and leave nothing unaddressed. Do not resolve a thread merely to clear a gate.
2. Run the focused checks for every changed artifact and exercise the actual installed profile,
   including optional providers. Then run the committed-candidate release gate. Required checks
   must pass on attempt 1; a rerun does not establish the gate.
3. Merge only when the user has explicitly authorized that exact candidate. Existing
   authorization remains valid within its stated scope and does not need to be requested
   again. Immediately before an authorized squash merge, record the current `main` commit and
   the approved head tree.
4. Verify the resulting squash commit has exactly one parent, that parent is the recorded
   pre-merge `main` commit, and its tree is byte-identical to the approved head tree. Wait for
   the exact-commit `main` push checks to pass on attempt 1.
5. Because strict base checks make every merge stale, refresh the next owned candidate from the
   new `main`, review its resulting diff again, and repeat the qualification and authorization
   process. Do not merge from a stale green run.

Merge permission does not authorize a release, deployment, dependency-profile migration, or
advisory disposition beyond the reviewed candidate.
