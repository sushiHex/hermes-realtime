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

Accept the existing twenty-scenario v1 shape with the coverage limits below.
Do not add another runner authority, a caller-supplied success switch, or a second
report format for full qualification. This decision approves the implementation
design. It supplies no physical observations, installed environment, release
decision, deployment permission, or public disclosure of private evidence.

This is an implementation protocol, not a claim that its complete acceptance path
exists. The current `validate_qualification_report` checks supplied bytes and
their semantic consistency. It does not authenticate observations, establish
operator presence, or prove that any producer ran. Full qualification is refused
until the input authority and capability-based composition described here exist;
their implementation is tracked in [#47](https://github.com/sushiHex/hermes-realtime/issues/47)
and [#62](https://github.com/sushiHex/hermes-realtime/issues/62). A validator return
value alone is never an accepted qualification result.

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
launch and before accepting output. Those calls are point-in-time byte checks;
they do not retain file handles, compare file identities, detect hard-link
aliases, or prevent a file from changing between verification and use. The input
authority in #47 must seal the complete input closure through every consumer:
retain validated file and ancestor identities with handles that deny writes,
replacement and deletion, or use an equivalently immutable owned snapshot.
Verify bytes through those seals and preserve the seals across build, install,
launch, observation and final acceptance. If sealing cannot be established,
refuse before executing a consumer. A final matching hash does not excuse a
transient replacement during execution.

Reject missing, changed, aliased, indirect, out-of-root, noncanonical, duplicate,
or mismatched inputs. Require distinct file identities and a single hard link for
each declared artifact role, including direct/repeated build outputs. Separate
files with equal hashes still do not prove separate builds: reproducibility
requires independently owned build invocations and their bound output receipts.
Claimed hashes and version strings alone do not prove installed identity, build
execution, or source provenance. These filesystem seals do not isolate a hostile
process that already controls the runner's interpreter or execution authority.
Reuse the existing archive and wheel validators for runtime blob parity. Verify
build reproducibility, dependency closure, installed import origins, and required
environment identities independently before they support an installed claim.

Install each runtime purpose offline in its own fresh environment from the bound
wheelhouse, with dependency hashes enforced and ambient configuration excluded.
Build inputs and runtime inputs retain separate roles. Installed checks must
execute the installed candidate and bound dependency environment with no editable
installation, source-checkout fallback, user-site imports, or unbound downloads.
Before the first installed import, independently manifest the complete executable
environment: interpreter and native libraries, standard library, installed
packages, generated entry points, and every allowed import/resource location.
Bind that manifest and its owned installation receipt to the sealed input closure
and candidate. Retain immutable file and ancestor seals, or an equivalent owned
immutable snapshot, over those generated bytes through every scenario and final
acceptance. Disable bytecode writes and keep writable data in separately owned
locations excluded from import and executable search paths. A Python image hash
alone does not bind imported modules. These generated-tree receipts belong to the
input authority; do not invent additional v1 report fields to imply that the
current `verifiedArtifacts` array covers them. Current archived producers extract
into writable workspaces and do not yet supply this installed-tree authority.
Source-only producer success and the Pure wheel job do not establish this full
closure. Missing provider, Hermes, browser, benchmark, or platform prerequisites
must remain unavailable; do not fabricate placeholders to satisfy the schema.

## Own the execution environment

Full storage qualification targets an interactive Windows desktop with an
operator present. Use capability discovery and explicit opt-in fixture paths;
require no named machine, device, room, account, or private network. Credentials
and deployment-specific provisioning remain outside the repository and reports.
Keep capture disabled by default and require fresh browser-binding consent for
each applicable scenario. Search and Hermes task authority remain separate;
qualification uses the governed disabled Hermes task mode.

Keep the trusted qualification controller and its private recovery state outside
the candidate execution boundary. Before any candidate code runs, #62 must
establish a dedicated low-privilege execution identity or equivalent OS sandbox
with explicit filesystem, handle, credential and network access restrictions.
The candidate may read sealed runtime inputs, write only owned scenario data, and
use the explicitly permitted loopback/media resources. It must not inherit the
operator's access token, credentials, home-directory access, arbitrary handles,
or outbound network access, nor be able to modify the controller, input authority,
installed seals or recovery journal. A separate ordinary account alone is not
sufficient without those verified restrictions. Provisioning requires the
separate environment authorization; capability discovery cannot silently create
accounts, grant permissions or change firewall policy. Refuse before dispatch if
the required boundary or desktop/media capability cannot be established.

The existing Windows Job owner provides lifecycle and resource limits, not this
security boundary. The current archived producers execute candidate code under
the invoking identity; they do not qualify execution of an adversarial candidate
on an operator's ordinary desktop. Full execution must verify its isolation
before those producers are integrated. Windows documents
[Job security limits](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
separately from [AppContainer access restrictions](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer).
The controller, OS isolation implementation and independently reviewed validator
are trusted authorities; this protocol does not claim to withstand compromise of
those authorities or the operating system.

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

The v1 full report permits twelve attestations total. Use exactly the twelve
observations in the table, each once at its designated scenario. Bind each to the
current attempt, its UTC observation time, and exactly one referencing scenario;
do not reuse an attestation across scenarios or candidates. These are operator
observations, not identity, authorship, hearing, comprehension, or truth
verification.

| Scenario | Required codes, each observed once |
| --- | --- |
| `physical_available_unconsented` | `disclosure_visible_controls_accessible` |
| `physical_microphone_response` | `chrome_microphone_permission_allowed`, `physical_input_selected`, `physical_phrase_spoken` |
| `physical_unmuted_transport` | `physical_output_selected`, `unmuted_output_audible` |
| `physical_muted_transport` | `chrome_mixer_zero`, `muted_output_not_audible`, `mixer_restored` |
| `physical_interruption_matrix` | `barge_in_performed`, `stop_performed` |
| `physical_reconnect` | `reconnect_performed` |

Capture-disabled and unconsented noncreation require real absence observations;
an unavailable consent control cannot be attested as visible. Typed-response
equality derives from the accepted input and persisted source, not an operator
authorship claim. Other scenarios have no report attestations. The muted scenario
must independently verify continuity of the output selected in the unmuted
scenario; a changed device blocks that comparison. Reconnect and media replacement
require machine evidence of fresh consent and binding isolation. The v1 report
does not attest repeated disclosure viewing, repeated device selection, every
interruption action, or a human-confirmed media-replacement action. Accept those
limits; stronger human claims require a reviewed format extension, not invented
codes, extra rows, reused attestations, or automatic confirmations.

The composition authority in #62 must enforce this exact scenario/code mapping,
one occurrence of every code, fresh unique IDs, and observation times inside both
the run and corresponding scenario observation windows. It must collect operator
input through an owned contemporaneous interaction; a supplied JSON confirmation
is not that capability. Missing, declined, stale, misassigned, or out-of-window
observations refuse acceptance. The current byte validator does not enforce these
requirements and must not be used as their substitute.

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

Existing storage producers deliberately retain a failed private workspace when
successful cleanup cannot be established; see [full-purge cleanup](full-purge-cleanup.md).
This can include a database and diagnostic material. The existing context manager
does not persist a recovery record or return a surviving cleanup capability after
failure. Therefore its retained directory alone is not a recoverable quarantine
authority, and this plan does not claim that existing failed workspaces will be
automatically discovered or purged.

Before creating sensitive files or dispatching workers, the complete runner in
#62 must durably record private workspace ownership and cleanup obligations.
Bind the record to the run/candidate, exact workspace and parent file/volume
identities, an owned marker, allowed artifacts, and retained process creation and
image identities. Keep the record in an access-controlled private recovery root
outside the disposable workspace, update it durably before each new owned
resource, and preserve it across exceptions and runner restart. If recording
fails, refuse dispatch. The journal itself contains private paths and identifiers;
it cannot enter Git, public reports, logs, comments, or attachments.

Recovery must reopen and match the recorded ownership before touching files,
establish that every recorded process has exited without confusing reused PIDs,
perform only the recorded owned cleanup, and independently verify completion
before retiring the recovery record. Unknown ownership or incomplete cleanup
remains a visible private obligation and prevents acceptance. Never scan a name
prefix and adopt arbitrary directories, retroactively claim an unrecorded
workspace, or delete beneath a possibly live writer. Existing partial producers
retain their documented limits until integrated with this authority; a complete
qualification entry point must refuse until that integration is available.

The composition authority must accept only genuine producer-minted capabilities,
revalidate their independent observations, and bind each to the same candidate,
artifact closure, scenario and invocation. It must verify actual installed input,
environment, topology, cleanup, and operator capabilities before deriving report
rows; a dictionary of supplied outcomes or a serialized receipt is insufficient.
Only this authority may issue a full acceptance result. Missing registration or
evidence refuses before an acceptance result can exist.

After this authority verifies the complete run, reopen the input closure,
serialize canonical JSON, and use `validate_qualification_report` as a final
consistency check on the derived bytes. The composer derives `passed`; the byte
validator checks internal consistency but supplies no execution or human proof.
Schema-valid failure records are not accepted qualification. Missing or
unexecuted producers must cause explicit refusal, never invented pass rows or a
partial report presented as the complete v1 report. Benchmark acceptance and
release-manifest acceptance retain their separate validators and authorities.

Keep failed attempts and their bounded, sanitized conclusions associated with
the exact source/input identities. Do not overwrite them with a later success,
cancel hosted runs, rerun a failed workflow, relax a timeout, or combine different
candidates' successful rows into one run. Raw audio, transcripts, screenshots,
databases, credentials, and private diagnostic output cannot enter public failure
records. Purge owned raw artifacts when cleanup is safe; unresolved ownership
uses the private quarantine disposition above. Preserving a sanitized failure
conclusion is separate from claiming raw-data cleanup.

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
