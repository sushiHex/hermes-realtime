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

Accept qualification of the governed evidence behavior on a trusted host.
Decline turning v1 into a new adversarial execution or infrastructure-security
certification architecture. In particular, this report does not prove a maximum
upstream credential lifetime, global certificate-revocation freshness, or complete
browser/OS network isolation. Those properties are outside the existing contract
and cannot be inferred from a passing report. They remain distinct from the
required native TLS use, explicit remote-input consent, verified local media
topology, immutable execution inputs and independently verified owned cleanup.

## Freeze and verify inputs

Use a clean committed source candidate with its baseline, commit, tree, canonical
diff digest, and deterministic archive identity. Materialize only that archive.
Freeze the governing plan and qualification runner with the candidate; never
execute a dirty checkout or replace a bound artifact during a run.

Establish the closure in three stages so no invocation depends on its own output:

1. Seal the independently selected tool environments and pre-existing build
   resources first. Authenticate every trusted tool/runtime against a governed
   digest allowlist or independently verified publisher/distribution provenance
   before executing it. A caller-chosen digest or locally generated manifest
   cannot establish that trust. Under those authenticated seals, use the existing Git/archive authority to
   verify the selected candidate and capture its source. Bind the verified source,
   governing plan, runner, schemas and tool/resource manifests into a sealed
   pre-build closure. No final qualification-input digest exists at this stage.
2. Execute each build and repeated build independently against that pre-build
   closure, writing into separate owned output locations outside executable/import
   paths. Retain each invocation's actual ownership, input identity and output
   receipt; independently verify and seal outputs before later consumers use them.
   Obtain the Linux prerequisite receipt described below against the same source
   and applicable output/resource digests. Preserve the original seals and receipts.
3. Once all required outputs and prerequisites exist, construct the final canonical
   `qualification-input-v1` document and seal its complete direct/transitive
   closure. The input authority must match every final role to its pre-build input
   or genuinely produced output and verify that the original seals remain live.
   It then binds those retained capabilities to the final input digest. This later
   binding must not rewrite an invocation's original input identity or substitute
   a final-file hash for evidence that the invocation occurred.

Prepare one owned input root matching every direct and transitive role in
`qualification-input-v1`. It contains the direct and repeated wheels, sdist and
repeated sdist, wheels rebuilt from each sdist, all five purpose-specific
wheelhouse manifests and their hash-required requirements and wheels, build tool
identities, provider distributions and model manifests, browser image manifest,
LiveKit and Codex executables, Hermes source/harness, benchmark report and machine
manifest, and the exact governed schemas. The schema is the exhaustive role list.

After stage 3, reopen and verify actual files through
`verify_qualification_input_closure` before scenario launch and final acceptance.
Those calls are point-in-time byte checks;
they do not retain file handles, compare file identities, detect hard-link
aliases, or prevent a file from changing between verification and use. The input
authority in #47 must retain the pre-build and intermediate-output seals for
stage 1/2 consumers. Only after stage 3 can it seal the complete final input
closure, whose seals must remain live for every subsequent consumer:
retain validated file and ancestor identities with handles that deny writes,
replacement and deletion, or use an equivalently immutable owned snapshot.
Verify bytes through each stage's applicable seals. Builds use only pre-build
and independently produced intermediate capabilities; they never require their
own future outputs or the final closure as inputs. After stage 3, preserve the
final closure seals across subsequent installation, launch, observation and
final acceptance. If the applicable seals cannot be established,
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

Install each runtime purpose offline on its matching OS in its own fresh
environment from the bound wheelhouse, with dependency hashes enforced and ambient
configuration excluded. The Windows direct, sdist-built and Hermes environments
are local prerequisites for the desktop run. The `realtime_linux_runtime`
wheelhouse requires a separate Linux execution authority; never install it into
a Windows environment or silently change its platform tag.

The #47 input authority must retain a genuine Linux installation/null-capture
receipt binding the same verified candidate source, direct wheel, Linux
wheelhouse, requirements, dependencies, tool environment and actual Linux
execution. Accept only a receipt authenticated by the independently owned Linux
executor, with its exact source/workflow identity, attempt, artifact digests and
cleanup outcome. For GitHub Actions, verify the exact repository, workflow source,
head, run/attempt, job result and transferred artifact digests through the trusted
GitHub service; a supplied JSON result or log string cannot mint that authority.
Before freezing the final input document, match its applicable roles to that
receipt's retained input/output identities. This is an input prerequisite, not a
twenty-first scenario or physical proof. Missing or mismatched Linux execution
refuses complete input acceptance. The existing Linux CI job proves its exercised
scope but does not yet supply this complete qualification receipt.
Its Linux tool/runtime manifest is a separate retained prerequisite capability;
the v1 Windows `build_python_executable` role is not a claim that both operating
systems execute the same interpreter file.

Build inputs and runtime inputs retain separate roles. Installed checks must
execute the installed candidate and bound dependency environment with no editable
installation, source-checkout fallback, user-site imports, or unbound downloads.
Before any tool invocation or installed import, independently manifest the complete
executable environment: interpreter and native libraries, standard library, installed
packages, generated entry points, and every allowed import/resource location.
Bind each manifest and owned invocation receipt to the pre-build closure or,
after it exists, the final input closure and candidate. Retain immutable file and
ancestor seals, or an equivalent owned
immutable snapshot, over those generated bytes through every execution and its
last independent identity observation. Stop and verify exit of every consumer
before releasing an installed environment's live seals. Retain its authenticated
input/installation/execution receipt, release only its execution-tree seals, then
remove that owned environment and verify removal before final acceptance. Final
acceptance revalidates the retained receipt and cleanup evidence; it cannot demand
live handles to an already removed environment. The separate final input-root
seals remain live through acceptance and are not disposable environment cleanup
targets. Disable bytecode writes and keep writable data in separately owned
locations excluded from import and executable search paths. A Python image hash
alone does not bind imported modules. These generated-tree receipts belong to the
input authority; do not invent additional v1 report fields to imply that the
current `verifiedArtifacts` array covers them. Current archived producers extract
into writable workspaces and do not yet supply this installed-tree authority.
Apply this requirement to the build-tool environment before the first source
capture, build, repeated build or installation invocation, including Git/uv
helpers and the build interpreter's DLL, standard-library and import closure.
The existing single-executable pins do not establish that complete tool closure.
Build outputs must be written outside sealed executable/import locations and
independently verified and sealed before consumption. Equal repeated outputs do
not excuse an unsealed build environment.
Source-only producer success and the Pure wheel job do not establish this full
closure. Missing provider, Hermes, browser, benchmark, or platform prerequisites
must remain unavailable; do not fabricate placeholders to satisfy the schema.

## Own the execution environment

Use a dedicated interactive Windows qualification account/session on a host
under exclusive qualification control, with an operator present. Exclude
unrelated user sessions, files, profiles, credentials and untrusted workloads
until every owned consumer has stopped. Trusted OS services remain part of the
environment. A dedicated account alone does not protect the public LiveKit
development profile from other local users. Missing admission or exclusive
control refuses physical execution. Capability discovery and explicit fixture
paths must work without a named machine, device, account, room or private network.
This decision does not authorize account, credential, firewall or browser
provisioning, physical execution, or capture enablement outside its scenarios.

The trusted computing base consists of the independently reviewed controller and
validator, exact reviewed candidate host, admitted tool/provider/browser software,
and OS. Native client TLS and its admitted trust material are part of that base.
Qualification does not reimplement certificate validation or certify that base
against compromise. Windows Jobs, private Python capabilities and file seals
establish their documented process/data properties; they are not a malware
sandbox. Untrusted input data, messages and observations still require strict
validation. A malicious candidate, controller, admitted tool or OS is outside this
claim and must not be qualified on an operator's ordinary desktop. A different
threat model requires a separate reviewed runtime/design decision.

Construct a closed full-host launch profile from bound inputs and the scenario;
never forward arbitrary caller CLI arguments or configuration overrides. Before
resource creation, independently verify and freeze the merged effective profile:

- Explicitly select `inference_provider="codex"`, `stt_provider="moonshine"`,
  `tts_provider="kokoro"`, and `moonshine_model_tier="medium"`.
- Require the report's `ProviderV1` constants: Codex `gpt-5.6-terra` with `low`
  effort, Moonshine distribution `0.1.0` with the medium backend, and Kokoro voice
  `bf_isabella`. Match all expected versions and admitted distribution/model
  resources to the input. Ordinary Ollama/tiny/automatic-TTS defaults cannot
  substitute; missing or mismatched providers refuse without fallback.
- Use the governed disabled Hermes-task mode. Reject task-enabling overrides.
  Require `public_search`, `knowledge_speculation` and `knowledge_recovery` false,
  with no search lookup or knowledge coordinator admitted. Reject all public-search
  and knowledge flags, including `--knowledge-budget-seconds`, and equivalent
  API/configuration overrides. Other optional features retain source-bound defaults.
- Disable model/effort/voice mutation at the native selection boundary, including
  authenticated `/api/v1/model` and `/api/v1/voice` requests and equivalent internal
  controls. Hiding UI controls alone is insufficient. Authenticated adverse
  requests must demonstrate refusal without changing the selected configuration.

Construction must consume that same frozen profile. Before invoking any provider
constructor, resolve its complete model/resource selection through the input
authority and retain the admitted paths, identities and sealed bytes. In
particular, the Moonshine constructor imports its package, resolves the model
and opens it natively: its selected medium model must already match the sealed
manifest before that native load. Constrain lookup to the admitted resource
namespace; refuse ambient fallback, downloads, changed selection or unavailable
pre-load verification. A post-construction mismatch cannot undo an unbound load.
Verify actual provider, backend and resource identities again after construction
and before input admission;
verify effective selections before each inference/synthesis dispatch and final
acceptance. Any configuration change invalidates admission. This profile applies
to full-host execution and does not broaden a synthetic producer's proof class.
The current no-task composition still admits public-search and selection controls;
#62 owns enforcement. Its present behavior is not evidence of compliance.

Keep evidence capture disabled by default and collect fresh browser-binding
storage consent for every applicable scenario. That local-storage consent is
separate from permission to send conversation text to the remote inference service.

Before admitting microphone or typed input, the operator must read and explicitly
accept this v1 remote-inference disclosure: the Codex path sends conversation text
to OpenAI, including final microphone transcriptions, typed user text, retained
conversation messages and supplied context. The data is processed remotely.
Withdrawing consent cannot retract text already submitted. Use synthetic scenario
content and admit no uninformed participant's speech or text.

An independently controlled input station must observe that explicit consent
before input admission and bind it privately to the exact disclosure/policy blob,
candidate, final input digest and run. Gate input and every remote dispatch on
the live consent. Refusal, missing evidence or withdrawal blocks further input
and dispatch, cancels owned outstanding work and refuses the attempt while
preserving cleanup obligations. Account usage permission, proposal approval,
stored defaults and prior runs cannot supply consent. This admission record is
separate from the twelve scenario attestations, adds no report field or claim of
human identity/truth, and must remain independently verifiable at final acceptance.
The current storage disclosure/provider composition does not supply this
capability; #62 must integrate it before physical execution.

The `codex-openai-service-v1` profile preserves the real published subscription
client and existing host/provider path. Require a separately authorized account
and bounded attempt/usage budget. Bind the complete admitted client/configuration
and its native TLS/trust material to the candidate/input authority. Use the
documented first-party ChatGPT service configuration with normal server-name and
certificate validation; reject alternate providers, API-key routes and unbound
endpoint, proxy or custom-CA overrides before launching the client. The official
[sample configuration](https://learn.chatgpt.com/docs/config-file/config-sample)
identifies the ChatGPT service base. Review every effective configuration layer,
including the temporary home and environment. The wrapper currently forwards
custom-CA variables, so its environment mapping alone does not establish admission.
Preserve the admitted configuration through reconnects. Do not substitute a broker
or mock provider, or infer a service-side spending cap from a local timeout.

Use the existing `livekit_local_v1` development profile. Its public predictable
constants are not private credentials and cannot be revoked per run. Its bounded
use relies on exclusive host control, loopback signaling, the effective inbound
firewall policy and the independently verified participant/media topology below.
It cannot authenticate against untrusted local peers or qualify a shared/external
deployment. Cleanup retires the owned service instance and temporary state; it
does not revoke the public constants or their use by another instance.

Bind the actual Python, browser version directory, provider/model resources,
LiveKit binary, benchmark machine context, and report environment to the frozen
inputs. Retain process identity with image hashes, creation times, and handles
through the existing Windows Job owner. A matching PID alone is insufficient.
Launch only owned descendants, retain membership observations, wait on retained
handles, and require zero active processes after cleanup. Preserve the existing
bounded process, checkpoint, foreground, and close authorities.

Use loopback-only signaling and the pinned LiveKit credential profile; keep
credentials out of retained output. The documented Windows LiveKit 1.13.4
`--dev --bind 127.0.0.1` path opens RTC listeners on available interfaces;
see [local LiveKit](local-livekit.md) and the official
[port roles](https://docs.livekit.io/transport/self-hosting/ports-firewall/).
Its actual socket inventory must not be relabeled as loopback-only.

Accept this bounded local RTC topology for the complete runner in #62: the owned
LiveKit process may listen on RTC TCP 7881 and UDP 7882 with loopback or wildcard
bindings. Owned browser/host media clients may use OS-assigned UDP endpoints
bound to loopback, wildcard or verified addresses of the same qualification
machine. Independently retain process, interface and selected ICE-pair evidence;
require both media peers to be on that same machine, the expected room and
participant identities, and no external TURN/media peer. Independently collect
participant admission, identity and departure events continuously from service
startup through shutdown, with no observation gap; an unexpected or impersonated
peer refuses the run even if it leaves before the final inventory. Combine that
evidence with exclusive host control: the public credentials and inbound firewall
do not authenticate or exclude same-host loopback clients. Missing continuous
observation refuses qualification. Reject unowned endpoints,
unexpected server ports, remote media peers or unclassified sockets. Collapse
only genuinely observed duplicate `(protocol, port, addressClass)` rows for the
v1 inventory; preserve their process/interface bindings in the private topology
capability. This qualifies a local media path, not loopback-only RTC listeners or
network isolation from arbitrary peers.

Require Windows Firewall enabled with effective policy preventing external
inbound access to those owned RTC endpoints; pre-existing broad allow rules or an
unverifiable effective policy refuse the physical run. Do not create or modify
rules to obtain a pass. The current byte validator rejects all non-loopback
inventory rows even though the v1 schema permits `wildcard` and `other`. Revise
that semantic restriction in #62 to the bounded policy above, backed by the
genuine topology capability; accepting supplied address-class labels alone is
insufficient. Until that implementation exists, this profile cannot produce an
accepted full report. Do not infer RTC routing from the signaling address.
Use an owned disposable browser profile;
do not modify or clean an operator's ordinary profile or browser installation.

Use an admitted browser image with an owned fresh profile, no operator sign-in,
extensions or restored sessions. Keep the qualification UI and actions on the
owned application and media topology; unexpected navigation, participants or
scenario input refuses the run. Bind the browser configuration and observations
to the input authority. The browser and its native networking remain admitted
software in the trusted base. This is not a claim that every browser/OS background
connection was blocked or inventoried; neither disposable profiles nor the RTC
inventory establish whole-host egress isolation. Stronger browser/network
certification is a separate scope, not a new prerequisite silently imposed on v1.

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
input through an owned contemporaneous interaction on a separately secured
desktop/input station or an out-of-band trusted attestation device. Verify that
candidate processes cannot observe or inject input into that collector; do not
collect confirmations in candidate-controlled browser UI. The operator may view
the qualified UI while confirming through the independent channel. A supplied JSON confirmation
is not that capability. Missing collector separation, declined, stale, misassigned, or out-of-window
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
files after verifying their absolute paths and ownership. Preserve fixture decoys
and stable ownership markers through the independent purge/recovery observation,
as required by that scenario's contract. After every consumer has exited and the
accepted observation receipt is retained, final cleanup may remove the enclosing
qualification-owned workspace, including its fixture markers and decoys. This is
how successful storage producers satisfy `evidenceRootsRemoved`; it does not
retroactively claim that purge deleted the protected fixtures. Adjacent files
outside the owned workspace, the browser installation image and firewall state
remain outside cleanup authority. Verify each applicable cleanup assertion from the
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

Include a dedicated private host-temporary root in this journal before launching
any full-host process. The actual Codex provider creates random temporary
credential homes and inference workspaces; forced process termination bypasses
their normal Python cleanup. Launch fresh host processes with `TEMP`, `TMP` and
`TMPDIR` all bound to that pre-owned root in their actual process environment,
before Python can cache a temporary directory. Supplying a mapping only to the
Codex child launcher does not redirect the parent provider's `tempfile` calls.
Before constructing any provider or copying credentials, a trusted bootstrap in
that fresh host must call `tempfile.gettempdir()` and prove its selected, cached
directory matches the retained private root's path and file/volume identities.
Python may otherwise skip an unavailable environment directory and choose an
ambient fallback. Missing, unwritable or different selection refuses before any
sensitive creation. Retain the root and its ancestry against replacement for the
entire consumer lifetime; after selection, a failed creation must fail in that
same cached root, without resetting it or selecting another directory. Prove the
pre-provider refusal with missing and unwritable-root tests, including termination
at the credential-copy boundary. Post-creation detection alone is insufficient.
Also verify containment of both the copied credential home and inference workspace,
including descendant tool temporary files, before accepting their use. An
ambient fallback or unowned temporary path refuses full qualification.

Record this root as an owned private subtree cleanup obligation, so recovery
covers randomly named descendants without needing their names before creation.
Require its protected access, exact parent/root identities and marker; reject
links, reparse points or unexpected ownership during recovery. After all recorded
writers have exited, verify and remove only this recorded subtree, including any
credential copies left by termination. Never discover it by scanning a temporary
directory prefix. The complete runner must demonstrate this path under forced
termination and restart; the current provider's `TemporaryDirectory` cleanup
alone does not satisfy it. Credential content and these paths remain private.

Reusable subscription credentials remain an explicitly authorized prerequisite
on this trusted host. Before copying them, bind the credential context and
responsible custodian to the run and its private recovery record; preserve the
original credential source outside disposable cleanup authority. Verify protected
access and containment of every owned copy, and stop all consumers before safe
removal. Missing authorization, containment or ownership refuses introduction.

If cleanup cannot be proved, retain a private recovery obligation with the
custodian, keep the workspace access-controlled, stop further dispatch and refuse
qualification. Resolve it through verified owned cleanup and, when required by
the incident, the provider's supported credential-invalidation procedure. Record
an invalidation as successful only with independently confirmed evidence. A
failed or unavailable action remains unresolved; local logout or deletion is not
proof of upstream invalidation. Never claim a bounded upstream grant lifetime
from this runner, or require a new scheduled-expiry/revocation service as part of
this design. This accepts the existing credential path under the stated trusted
host model, not an assertion that a copied refresh grant becomes harmless after
a local deadline. Invalidation does not prove file removal or authorize deletion
without verified filesystem/process ownership. Cleanup and credential disposition
remain distinct private obligations; no unresolved cleanup can yield acceptance.

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
