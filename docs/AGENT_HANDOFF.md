# Agent Handoff

Durable rules live in [`AGENTS.md`](../AGENTS.md); contribution paths live in
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Accepted design lives in
[ADRs](adr/0001-github-work-tracking.md). [GitHub Issues](https://github.com/sushiHex/hermes-realtime/issues)
owns current work, ownership, status and acceptance, and
[the tracking guide](work-tracking.md) governs it. Query the owning issue and its comments
before resuming anything.

**This document provides context, not a parallel task ledger.** It records what a session
learned and how that was established, so the next one does not re-derive it or repeat a
refuted reading. Nothing here confers authority: a finding recorded below is still bound by
the evidence its issue carries.

## Continue this session

**2026-09-19/20 — dependency policy, the recovery seam, and the upstream continuity audit
(base `ec7170e1` through `141b3772`).**

Finding: **a bot may only propose a change that is complete in the files it writes.** Every
red Dependabot pull request in this repository shared one shape — the change needed a second
artifact the bot does not write: `uv.lock` after a manifest edit, the tracked browser assets
after a `/web` bump, the reviewed action pin in `tests/test_release_workflow.py` after an
action bump, the recompiled closure after one of its declared roots moves. The three that
were green touched `requirements/*.txt`, which has no second artifact at all. That single
criterion explained the evaluated cases and guides `.github/dependabot.yml` (#107, #119,
#129, #144). The `cryptography` exception below does not cover every proposal in that class.

Three readings were refused by checking the artifact rather than reasoning forward, and each
refusal improved the result. "Exact pins are reviewed decisions" was over-theorised: the
failures were the lockfile gate, and exactness was irrelevant. "The closure is generated
output, so remove the entry" was wrong: at exact public commit
[`8e427265`](https://github.com/sushiHex/hermes-realtime/commit/8e4272651a06bcb6720d282cab147356a20e30f7),
`nvidia-cuda-runtime` 13.3.29 [published three wheels](https://pypi.org/pypi/nvidia-cuda-runtime/13.3.29/json),
and the [committed file](https://github.com/sushiHex/hermes-realtime/blob/8e4272651a06bcb6720d282cab147356a20e30f7/requirements/kokoro-cuda-worker-win-py311.txt#L101-L104)
lists exactly those three hashes. That was read as showing a derived entry carrying every
published hash is one a bot *can* complete. **That reading was itself wrong** (#169): the hash
set matching is necessary, not sufficient. The bot resolves without the closure's
`--python-version 3.11` and platform, so #112 moved joblib to a release that newly required
cloudpickle without adding it, and #130 moved numpy to a release requiring Python 3.12 — both
with complete, correct hash sets, both green, both breaking the worker setup. The original
premise was right after all: the closure is generated output. The wrong step was the
conclusion drawn from it. Removing a producer is not the mechanism; checking the output against
its generation is, and `tests/test_worker_closure.py` now does that whoever proposes the change.
"The entry scoped
to a directory is the entry that reaches its files" was disproved by the bot itself, which
proposed a declared-root change in `kokoro-cuda-worker.in` through the `uv` entry — an ignore
is per entry, so a root ignored on one entry is proposed by the other.

`cryptography` was the standing exception, its pull request reopening weekly and never able to
merge because the pin is derived from `_COMMIT`. A blanket ignore would also silence the dev
group's updates. `update-types: ["version-update:semver-major"]` suppresses major proposals
while preserving the dev range's eligible updates, and `tests/test_dependabot_policy.py`
asserts that the mirrored pin (`==48.0.1`) and dev range (`>=50,<51`) remain in different
majors. It does **not** suppress a later 48.x patch or minor proposal: that proposal still
cannot move independently of `_COMMIT`, and the exact-source pin test in
`tests/test_qualification_candidate_files.py` refuses it.

Recovery: ADR 0002 (#106) mapped the merged process owner and run journal onto an ordered
state table. Its finding is that **the kernel decides process liveness and the journal decides
only whether effects were possible** — within the owner's documented single-Job topology, the
kill-on-close Job is associated at creation and the child is created suspended, so a controller
death at any boundary terminates it. That removes custody transfer, adoption, reconciliation
and recovery-side termination from #62's
scope entirely. `scripts/qualification_recovery.py` (#128) implements the first slice as one
call site, because the ordering it enforces is one nothing else can: the owner has no journal
awareness and the journal knows nothing of the kernel calls around it. The ADR is **proposed,
not accepted**, so that module cites the accepted execution protocol instead.

Upstream (#80): at the qualified commit, Hermes has **no API surface for saving an externally
generated assistant turn** — the only message write in the API server is a fork handler
copying an existing transcript, and `GET /v1/capabilities` advertises `"memory_write_api":
False`. Compounding it, the message schema carries no delivered or interrupted state, so
treating saved history as delivered context would erase the distinction this project's
delivery ledger preserves. Read history and correlate runs; do not write turns. One precondition
was missed on the first pass and corrected: enabling an external memory provider forwards raw
turn text off-machine, which is configured in Hermes and invisible from here.

Instrumentation: #121 distinguished the durable typed acceptance from the missing microphone
acceptance in #120. The first reading of `turn_opened: 1` incorrectly excluded the evidence
writer: `try_admit_user_final` enqueues `turn_opened` and `user_final_accepted` together, so
their shared absence cannot establish whether the runtime opened a second turn. The test
double did return a microphone final; filtering, routing, admission and persistence remained
undistinguished. The [source-bound correction](https://github.com/sushiHex/hermes-realtime/issues/120#issuecomment-5752463790)
records that limit. Observe an independent boundary before assigning a cause; preserve the
bound that made the missing acceptance visible.

Process: a review found a genuine defect in each of three consecutive changes here, twice a
check that passed in the broken state and once a fail-open in production code where a caller's
value was laundered into both sides of the comparison meant to refuse it. Those are now rules
in `AGENTS.md` rather than lessons.

## Deployment context

Whether a candidate may be released, tagged or packaged is current status, so it lives in the
owning issue and its milestone, never here. Use that issue, its pull request and the actual
landing manifest for deployment and rollback revisions, and record checks and acceptance
evidence there.

What is durable: a merge is not a release, and local review is not acceptance. Keep
credentials, one-use launch fragments, private URLs, transcripts and user audio out of
commits, logs, screenshots and evidence artifacts alike.
