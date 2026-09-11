# Slice 0 qualification execution protocol

[Collaborator guide](README.md) · [Evidence contract](evidence-capture.md)

## Decision and authority

Adopt this protocol for implementing and reviewing full evidence-only Slice 0
qualification. The decision takes effect when its reviewed PR is merged; an
unmerged proposal is not execution authority. [Issue #46](https://github.com/sushiHex/hermes-realtime/issues/46)
records the decision and exact review evidence. This document owns the execution
design; GitHub Issues owns implementation scope, dependencies, and completion.

The [evidence contract](evidence-capture.md) governs runtime and privacy behavior.
The [input schema](../scripts/schemas/qualification-input-v1.schema.json),
[report schema](../scripts/schemas/qualification-report-v1.schema.json), and
[qualification module](../scripts/qualify_evidence_slice_zero.py) govern exact
input closure, scenario order, proof classes, and independent report validation.
At the reviewed source baseline
[`11900a84583853d877bb19eea3626dec92ac5542`](https://github.com/sushiHex/hermes-realtime/commit/11900a84583853d877bb19eea3626dec92ac5542),
the schema SHA-256s are:

| Schema | SHA-256 of exact file bytes |
| --- | --- |
| `qualification-input-v1` | `ffe645767c3621a69076bf364dceacbb5b236884ef7f4c2d73c578cd44f15da3` |
| `qualification-report-v1` | `e3c8c509612bba844d9329202beb83bee7cbdf6b7a3d98501e2511be9da3e399` |

A run binds the exact committed bytes of this plan and its candidate's schemas,
runner, and source archive. Changing any bound input requires a new input digest
and run. A plan/schema disagreement stops acceptance for explicit review; neither
a permissive schema nor an issue description can silently extend a proof.

Accept the existing twenty-scenario v1 contract with the coverage limits below.
Do not add another runner authority, a caller-supplied success switch, or a second
report format for full qualification. This decision approves the implementation
design. It supplies no physical observations, installed environment, release
decision, deployment permission, or public disclosure of private evidence.

## Freeze and verify inputs

Use a clean committed source candidate with its baseline, commit, tree, canonical
diff digest, and deterministic archive identity. Materialize only that archive.
Freeze the governing plan and qualification runner with the candidate; never
execute a dirty checkout or replace a bound artifact during a run.

Prepare one owned input root matching every direct and transitive role in
`qualification-input-v1`. It contains the direct and repeated wheels, sdist and
repeated sdist, wheels rebuilt from each sdist, all five purpose-specific
wheelhouse manifests and their hash-required requirements and wheels, build tool
identities, provider distributions and model manifests, browser image manifest,
LiveKit and Codex executables, Hermes source/harness, benchmark report and machine
manifest, and the exact governed schemas. The schema is the exhaustive role list.

Reopen and verify actual files through `verify_qualification_input_closure` before
launch and before accepting output. Reject missing, changed, aliased, indirect,
out-of-root, noncanonical, duplicate, or mismatched inputs. Claimed hashes and
version strings alone do not prove installed identity or source provenance.
Reuse the existing archive and wheel validators for runtime blob parity. Verify
build reproducibility, dependency closure, installed import origins, and required
environment identities independently before they support an installed claim.

Install each runtime purpose offline in its own fresh environment from the bound
wheelhouse, with dependency hashes enforced and ambient configuration excluded.
Build inputs and runtime inputs retain separate roles. Installed checks must
execute the installed candidate and bound dependency environment with no editable
installation, source-checkout fallback, user-site imports, or unbound downloads.
Source-only producer success and the Pure wheel job do not establish this full
closure. Missing provider, Hermes, browser, benchmark, or platform prerequisites
must remain unavailable; do not fabricate placeholders to satisfy the schema.

## Own the execution environment

Full storage qualification targets an interactive Windows desktop under the
logged-on operator. Use capability discovery and explicit opt-in fixture paths;
require no named machine, device, room, account, or private network. Credentials
and deployment-specific provisioning remain outside the repository and reports.
Keep capture disabled by default and require fresh browser-binding consent for
each applicable scenario. Search and Hermes task authority remain separate;
qualification uses the governed disabled Hermes task mode.

Bind the actual Python, browser version directory, provider/model resources,
LiveKit binary, benchmark machine context, and report environment to the frozen
inputs. Retain process identity with image hashes, creation times, and handles
through the existing Windows Job owner. A matching PID alone is insufficient.
Launch only owned descendants, retain membership observations, wait on retained
handles, and require zero active processes after cleanup. Preserve the existing
bounded process, checkpoint, foreground, and close authorities.

Use loopback-only signaling and the pinned LiveKit credential profile; keep
credentials out of retained output. Independently inventory RTC sockets and bind
them to owned processes. Do not infer RTC routing from the signaling address or
create firewall rules to obtain a pass. Use an owned disposable browser profile;
do not modify or clean an operator's ordinary profile or browser installation.

Filesystem faults require an owned fixture root, exact allowed artifact names,
adjacent decoys, and verified restoration of any changed DACL or file attributes.
Before a volume-full experiment, verify a disposable VHDX and both its volume
identity and backing disk extents against the system volume. Its backing must be
on a non-system volume with bounded capacity and sufficient host headroom. Never
fill ordinary operator storage. Missing ownership, privileges, restoration, or
non-system proof blocks that experiment before destructive work begins.

## Produce observations, then derive claims

Invoke the canonical registry in the exact twenty-scenario order required by
the input and report schemas. Validate registration identity, candidate binding,
proof class, case order, and the independently observed attempt. A producer must
execute its real path and its independent validator must derive the assertions.
Generic pytest success, supplied booleans, replayed receipts, mocks, or an enum
label cannot provide a production capability or a stronger proof class.

The physical family uses the installed full host and actual browser/media path.
Packaged scenarios retain their installed process boundary; real Windows
filesystem scenarios retain their OS and storage boundary. Synthetic injection
remains `synthetic_injected`. Deterministic equivalence establishes its bounded
comparison and does not replace any physical observation.

Collect only contemporaneous operator confirmations from the schema's closed
observation codes. Bind each fresh attestation to this attempt, UTC observation
time, and exactly one referencing scenario; do not reuse attestations across
scenarios or candidates. An absent, declined, stale, or out-of-window observation
blocks acceptance. These are operator observations, not identity, authorship,
hearing, comprehension, or truth verification.

| Physical observation | Required use of existing codes |
| --- | --- |
| Consent controls | `disclosure_visible_controls_accessible` where capture is available and for fresh consent after a binding replacement |
| Microphone response | `chrome_microphone_permission_allowed`, `physical_input_selected`, and `physical_phrase_spoken`; machine evidence separately compares the synthetic fixture and STT result |
| Unmuted transport | `physical_output_selected` and `unmuted_output_audible` |
| Muted transport | `physical_output_selected`, `chrome_mixer_zero`, `muted_output_not_audible`, and subsequent `mixer_restored` |
| Interruption matrix | `barge_in_performed` and `stop_performed` for the corresponding actions; machine evidence covers each governed case |
| Reconnect | `reconnect_performed`, followed by a fresh disclosure observation and machine verification that previous consent is not inherited |
| Media replacement | A new `physical_input_selected` or `physical_output_selected` observation for the actual changed medium, fresh disclosure observation, and machine verification of media-incarnation and consent isolation |

Capture-disabled and unconsented noncreation require real absence observations;
an unavailable consent control cannot be attested as visible. Typed-response
equality derives from the accepted input and persisted source, not an operator
authorship claim. The codes do not encode every interruption case or a generic
media-replacement action; case-specific machine observations supply that binding.
Do not invent additional v1 codes or automatically confirm an operator action.

## Coverage dispositions

The existing `over_budget_turn` name is retained for wire compatibility. Its
accepted scope is the real 64-record queue-admission overflow documented in
[the producer guide](over-budget-turn.md). It does not establish cumulative
16 MiB per-turn quota exhaustion. Session rollover is also a separate proof.
Accept this bounded v1 scope; decline any inference that twenty passing v1 rows
qualify the separate cumulative quota. A future claim for that quota requires a
separately reviewed producer/contract extension and new qualification evidence.

The `synthetic_fault_matrix` must execute its five schema-defined cases in order:
coupled queue capacity, deny filtering, clock rollback, injected SQLite failure,
and blocked writer drain. Derive reachability from the production configuration:
64 record credits, 64 physical items, 32,768 maximum canonical bytes per record,
and 2,097,152 aggregate canonical bytes. The byte ceiling equals the lawful
record-count product. Observe the record-capacity refusal and actual accounted
counts/bytes; do not lower capacities to manufacture independent byte or physical
overflow. Rejected-source absence, durable purge, and cleanup require real
observations for the applicable cases.

The v1 report's empty measurements for volume-full do not encode its headroom
calculation. Derive that assertion from the actual owned allocation and free-byte
observations using `required_free_bytes` and `check_maintenance_headroom` in
[storage security](../src/hermes_realtime/evidence/storage_security.py): required
free space is the nonnegative difference between the physical maintenance ceiling
and owned allocated bytes; exceeding that ceiling is itself a refusal. The
producer independently validates this before deriving `headroom_formula_equal`;
retain permitted producer evidence outside the full row. Schema acceptance alone
cannot verify operator presence, artifact provenance, or physical hardware
behavior. Producing and independently accepting authorities establish those facts.

## Finalize, validate, and retain

Attempt cleanup after success, failure, cancellation, or unavailable prerequisites.
Stop admission, finalize consent/revocation and owned processes, purge raw capture
artifacts, restore fixture permissions and mixer changes, dismount the owned VHDX,
and remove only owned temporary environments, profiles, extraction, and backing
files after verifying their absolute paths and ownership. Preserve adjacent
decoys, stable ownership markers as required by the purge contract, the browser
image, and firewall state. Verify each applicable cleanup assertion from the
responsible owner. An unknown or incomplete cleanup result prevents acceptance.
Do not claim forensic SSD erasure or secure clearing of Python process memory.

Compose a complete report only from genuine independently accepted scenario
capabilities, verified installed inputs, actual environment/topology observations,
and the fresh operator attestations. Reopen the complete input closure, serialize
canonical JSON, and run `validate_qualification_report` on the resulting bytes.
The producer must derive `passed`; the semantic validator checks that derivation.
Schema-valid failure records are not accepted qualification. Missing or
unexecuted producers must cause explicit refusal, never invented pass rows or a
partial report presented as the complete v1 report. Benchmark acceptance and
release-manifest acceptance retain their separate validators and authorities.

Keep failed attempts and their bounded, sanitized conclusions associated with
the exact source/input identities. Do not overwrite them with a later success,
cancel hosted runs, rerun a failed workflow, relax a timeout, or combine different
candidates' successful rows into one run. Raw audio, transcripts, screenshots,
databases, credentials, and private diagnostic output are purged under the
contract; preserving failure evidence does not authorize retaining those bytes.

A schema-valid input or report is not automatically safe to publish: it can
contain relative paths, process identities, hardware context, and attestation
identifiers. Keep full operational records private. Public PR/issue summaries
contain only reviewed source/tree and permitted artifact digests, aggregate
counts, closed outcome codes, exact CI run/attempt links, and evidence limits.
Apply the [public-repository boundary](../CONTRIBUTING.md#public-repository-boundary)
to every publication surface, including comments and attachments.

Require both exact reviews and the four [automated PR/main checks](release-gates.md#required-automated-checks)
for each implementation candidate. Full desktop qualification additionally needs
one frozen complete run with independently accepted machine and human evidence.
The subsequent release decision names supported scope and unresolved risks;
qualification does not authorize publication, deployment, default enablement,
learning, background startup, or any other excluded phase.
