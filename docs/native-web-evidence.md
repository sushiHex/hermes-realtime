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

Attestation version 1 has exactly these fields: `version`, `contract`,
`generation`, `run_id`, `disposition`, `output_sha256`, `input_sha256`,
`input_bytes`, `catalog_sha256`, `catalog_count`, `passage_count`,
`truncated_count`, and `missing_count`. It contains no source text or URLs.

| Disposition | Meaning |
| --- | --- |
| `presented` | The bounded passage-backed formatter produced the published output. |
| `no_web` | No web tool was observed; ordinary native output is preserved. |
| `mixed_ungrounded` | Web and other effect tools were used; ordinary output is preserved without a grounding claim. |
| `refused` | Evidence capture or presentation could not establish this contract; the host withholds the result. |

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

The formatter returns one `answer` narrative and `source_ids`. Its first paragraph
is a standalone gist of at most 220 characters, used verbatim as the canonical
speech text. Display retains the full narrative and the validated sources footer,
within the existing 1800-character detail bound. Overlong openings are refused,
never truncated or independently summarized. Normal speech normalization still
applies after this projection. An empty source list is permitted for uncertainty
and produces no sources footer. This shares wording between speech and display;
it does not mechanically prove agreement between paragraphs or factual support.

The observer bounds active rows, retained source bytes, source membership and
observations before presenting. Exceeding an observation bound refuses
presentation; it does not deny or retry a native tool effect. Settled source text
is discarded; terminal metadata is bounded and may become unavailable on eviction.

These checks establish provenance, bounds, cancellation and output binding.
They do **not** prove factual entailment, date selection, latestness or completeness.
The captured passage is the native tool handler's result before any later result
transformation, not a claim about every byte subsequently shown to the model.
Semantic quality remains tracked in [#386](https://github.com/sushiHex/hermes-realtime/issues/386).
