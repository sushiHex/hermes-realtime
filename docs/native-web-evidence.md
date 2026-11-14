# Native web evidence

The optional `hermes.web-evidence/v1` contract presents web-only background results
from bounded, same-run passages. It reuses the selected lookup route's package and
tool-free formatter. It does not enable the selected lookup policy, replace native
tools, change approval policy, or add a dispatcher.

Enable it explicitly in the `hermes-realtime` plugin's settings with
`web_evidence: true` and a `presentation` mapping containing `provider`, `base_url`,
`model`, `api_key_env`, and optionally `api_mode`. The formatter must match the
actual native model route. Credentials remain in the native environment; they do
not cross the completion handoff. Omitting `web_evidence` leaves it disabled.

The host independently opts in with `HermesApiConfig.web_evidence_contract`, or
the launcher's `hermes_web_evidence_contract`, set to `hermes.web-evidence/v1`.
The maintained native preview derives that host choice from the explicit staged
plugin setting. This contract and `lookup_policy` cannot both be selected.

## Handoff

1. Authenticated `GET /hermes-realtime/v1/web-evidence/ready` returns exactly
   `ready`, `contract`, and a process-generation identifier.
2. Before dispatch, the host durably records the contract and generation. It
   sends `X-Web-Evidence-Contract` and `X-Web-Evidence-Generation` on the existing
   native `POST /v1/runs`. The acknowledgment must bind those values and the exact
   accepted run through `X-Web-Evidence-Run`. An idempotent resend retains the
   recorded generation; it cannot silently adopt a replacement producer.
3. The plugin binds the admitted run to Hermes's exact native task. Worker hooks
   use Hermes's run-scoped approval key, not a reusable conversation/session ID
   or an HTTP context variable. Hermes retains execution authority.
4. Authenticated terminal `GET /v1/runs/{run_id}` carries one bounded
   `X-Web-Evidence-Attestation`. The host validates its exact contract, generation,
   run and SHA-256 of the raw published output before projecting voice/detail.
   It uses the existing Runs event consumer; there is no second SSE subscription.

Attestation version 2 has exactly these fields: `version`, `contract`,
`generation`, `run_id`, `disposition`, `reason`, `output_sha256`, `input_sha256`,
`input_bytes`, `catalog_sha256`, `catalog_count`, `passage_count`,
`truncated_count`, and `missing_count`. It contains no source text or URLs.
Version 1 (no `reason`) is no longer accepted.

| Disposition | `reason` | Meaning |
| --- | --- | --- |
| `presented` | `null` | The bounded passage-backed formatter produced the published output. |
| `no_web` | `null` | No web tool was observed; ordinary native output is preserved. |
| `mixed_ungrounded` | `null` | Web and other effect tools were used; ordinary output is preserved without a grounding claim. |
| `ungrounded` | `unobserved` | No middleware call observed the final answer of a no-web or mixed run; ordinary output is preserved without a grounding claim. |
| `refused` | one refusal reason | Evidence capture or presentation could not establish this contract; the host withholds the result. |

`ungrounded` is decided at the terminal status read. Hermes's turn-budget summary
(`handle_max_iterations`) makes its final model call outside LLM middleware, so the
observer never sees that answer; any other unobserved call path lands here too. The
unobserved final gets the disposition an observed final would have had without
presentation: a no-web or mixed run is delivered `ungrounded`, and a web-only run is
`refused` with reason `unobserved`, because web-only output reaches the user only
through the passage-backed presenter. The observer binds the decision to the
SHA-256 of the exact completed output it reads from the status body. Any row retired
before that read while not yet bound to an output, undecided or refused, is kept
source-free until it, so a refusal still reports its reason. Nothing previously refused
is ever delivered this way. A final that Hermes builds outside middleware in a
no-web run (for example an error text) is delivered `ungrounded` and relies on the
host's ordinary result-envelope checks, as non-web runs already do.

Refusal reasons form a closed set: `binding` (task, run or presentation ownership
mismatch), `capture` (a web result could not be read), `closed` (stopped, cancelled
or unloaded), `deadline`, `final_response` (the observed final answer was not a
complete stop), `input_bound`, `insufficient` (no usable passage), `internal`,
`presentation_failed`, `route` (formatter route does not match the native route),
`source_changed`, and `unobserved` (a web-only run's final answer was never observed,
or a completed status had no bindable output). The
first reason of a refusal is the one reported. The plugin's content-free
`[web-evidence]` line carries the same `category` and `reason`, plus where and why it
refused: `phase` (the observer hook or final-response step), `cause` (the fixed
category literal when declared, `undeclared` for any other literal, `none` for a
non-`ValueError`), `error_class` (from a fixed allowlist, else `other`), the
presenter's `stage`, `selected_source_count` and `rendered_detail_length` when
reached. `tests/test_web_evidence_cause_ratchet.py` requires every literal category
raised on the refusal path to be declared; the host's
`[hermes-web-evidence]` line carries `reason` too, or `attestation` when the
attestation is missing or unbound. The host speaks and shows a fixed notice per
reason, and says "Please try again" only for `deadline`, `presentation_failed`
and `source_changed`.

Tool discovery metadata does not make a run mixed. Attempted, failed or malformed
web retrieval cannot be classified as `no_web`. Missing, evicted, malformed,
duplicated or mismatched metadata is unavailable; it never permits a raw-draft
fallback. Non-presented dispositions have no input/catalog hashes and zero
evidence counts. Run records write version 3 when this selection is pending;
versions 1 and 2 remain readable. Restart settlement still stops admitted runs
rather than publishing historical results.

## Bounds and limits

The immutable formatter input includes the objective, exact catalog, retained
passage prefixes, and missing/truncation indicators. Native draft prose is omitted
from synthesis; its hash remains local diagnostic metadata. Its
complete UTF-8 representation is at most 16 KiB. Presentation runs inside the
native task's lifetime, with a plugin-owned 120-second allowance measured from
admission. That allowance does not shorten ordinary or mixed native work.
Cancellation/unload closes owned formatter work and prevents late publication.

The formatter returns one `answer` object with required `lead` and `support`
strings, plus `source_ids`. The lead is a standalone gist of at most 220 characters,
used verbatim as canonical speech text. Display composes that same lead and any
nonempty support into one narrative, followed by the validated sources footer.
The combined display still fits the existing 1800-character detail bound. Support
can be empty when the lead fully answers the request, including honest uncertainty;
its explicit slot retains requested supporting dates and qualifications for richer
answers. Overlong leads are refused, never truncated or independently summarized.
Normal speech normalization still applies after this projection. An empty source
list is permitted for uncertainty and produces no sources footer. Shared wording
and a required support slot do not prove agreement or factual completeness.

The observer bounds active rows, retained source bytes, source membership and
observations before presenting. Exceeding an observation bound refuses
presentation; it does not deny or retry a native tool effect. Settled source text
is discarded; terminal metadata is bounded and may become unavailable on eviction.

These checks establish provenance, bounds, cancellation and output binding.
They do **not** prove factual entailment, date selection, latestness or completeness.
The captured passage is the native tool handler's result before any later result
transformation, not a claim about every byte subsequently shown to the model.
Semantic quality remains tracked in [#386](https://github.com/sushiHex/hermes-realtime/issues/386).
