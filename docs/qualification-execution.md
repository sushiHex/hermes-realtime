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

Full storage qualification targets a dedicated interactive Windows qualification
desktop with an operator present. Use capability discovery and explicit opt-in fixture paths;
require no named machine, device, room, account, or private network. Credentials
and deployment-specific provisioning remain outside the repository and reports.
Keep capture disabled by default and require fresh browser-binding consent for
each applicable scenario. Search and Hermes task authority remain separate;
qualification uses the governed disabled Hermes task mode.

The trusted computing base includes the independently reviewed candidate host,
qualification controller and validator, authenticated tools/providers/services,
and operating system. Admit only an exact candidate with clear code and security
reviews and the required automated gates. Run it in an explicitly provisioned
qualification account/session containing no unrelated operator files, browser
profiles or credentials, on a host under exclusive qualification control for the
entire run. Exclude other user sessions and untrusted local workloads; trusted OS
services remain part of the computing base. A dedicated account alone is
insufficient for the public LiveKit profile. Verify host admission before launch
and retain that exclusive control until every owned process has stopped. Keep
fixtures synthetic except for the separately authorized physical observations.
Missing candidate admission or exclusive host control refuses full execution.
This does not authorize account provisioning,
credential installation, firewall changes or physical execution by this PR.

This is qualification of reviewed code, not a malware sandbox. Job membership,
Python-private receipts, independent observer implementations and file seals do
not protect against malicious candidate code or a compromised trusted controller,
toolchain, validator or OS. The current archived producers execute under the
invoking identity. They must not be presented as safe execution of an unreviewed
or adversarial candidate on an operator's ordinary desktop. Windows documents
[Job limits](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
separately from [AppContainer access restrictions](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer).

Preserve the real provider and host paths being qualified. The pinned Codex role
needs an explicitly authorized subscription and remote inference access; use a
qualification-specific credential context with an owner-approved attempt/usage
budget. The trusted controller admits only the governed scenarios and retains
their dispatch and observation evidence under the existing runtime bounds.
Adopt the closed `codex-openai-service-v1` admission policy below. Its authority
comes from this reviewed source document, not an operator-supplied hostname or CA
approval. Bind its exact source blob to the candidate archive and governing-plan
closure, and bind the admitted client/configuration/trust evidence to that policy,
the Codex executable digest and the final input digest.

- Permit subscription service HTTPS/WSS only at `chatgpt.com:443`, using the
  first-party `/backend-api/` service base, and HTTPS token refresh only at
  `auth.openai.com:443`. No wildcard hosts, IP-literal service authorities,
  alternate providers, API-key routes or cross-origin redirects are admitted.
  A required destination outside this set needs a new reviewed policy version;
  the environment being checked cannot approve it for itself. The official
  [sample configuration](https://learn.chatgpt.com/docs/config-file/config-sample)
  identifies the ChatGPT service base; it does not prove a particular binary's
  complete connection behavior.
- Use only server-authentication roots admitted by the Microsoft Trusted Root
  Program, excluding its Disallowed set. Independently authenticate the published
  AuthRoot/Disallowed material. Require every effective trust root to belong to
  AuthRoot and be absent from Disallowed. Require every member of each actual
  service connection chain, including leaf and intermediate certificates, to be
  absent from Disallowed before sensitive application bytes. Client admission
  must establish that the native same-connection verifier enforces this rule
  against the initial admitted catalogs; a root-only comparison, machine store
  inspection or later rejection cannot supply that pre-dispatch guard. Before
  each run, the independent trusted controller must obtain
  the latest published catalogs from an authenticated Microsoft distribution
  source; a supplied snapshot, filesystem timestamp or successful verification
  of an old Microsoft signature does not establish freshness. The policy permits
  at most 24 hours from that authenticated retrieval through the last service
  connection. Use independently trusted UTC and a monotonic elapsed-time bound;
  unavailable time/source authentication or expiry refuses admission or further
  dispatch. Validate the catalogs' signed update times and any specified validity
  end, and reject future update times, expired material, or sequence/update-time
  rollback against controller-retained high-water records for each catalog
  identity. Bootstrap those records from the same authenticated current
  publication, never from the admitted host's cache. Recheck the publication
  before final acceptance; changed catalogs require reevaluation of every
  effective root and retained connection chain against both refreshed catalogs.
  Final acceptance applies the same predicate to every effective root and every
  retained connection chain under both the initial and refreshed catalogs:
  roots must be admitted by AuthRoot and all chain members must be absent from
  Disallowed. Missing evidence or any violation refuses acceptance, regardless
  of when a certificate became disallowed. This is a bounded
  freshness policy, not instantaneous notice of later root-program changes.
  Freeze catalog identities, sequence/update times, retrieval/expiry evidence,
  effective-root digests and final recheck in the private tool environment
  receipt. No client admission occurs without the initial evidence. See Microsoft's
  [CTL verification procedure](https://learn.microsoft.com/en-us/windows-server/identity/ad-cs/configure-trusted-roots-disallowed-certificates#verify-trusted-and-untrusted-ctls).
  Private, enterprise-interception and locally added roots are excluded. This
  retains the ordinary public-CA trust assumption, not protection against a
  compromised admitted CA or OS.
- The Disallowed catalogs do not replace issuing-CA revocation checks. Native
  same-connection validation must establish authenticated, fresh non-revoked
  status for every non-root service-chain certificate before sensitive
  application bytes. Require correctly signed, certificate/issuer-bound CRL or
  OCSP evidence with an applicable signed validity interval; reject revoked,
  unknown, missing, expired or unverifiable status without soft-failing. Check
  status validity with the trusted UTC and elapsed-time authority above, including
  resumed sessions and reconnects, and retain the status-to-connection bindings
  for final acceptance. For each connection, bind a monotonic stop deadline to
  the earliest applicable status/certificate validity end, catalog admission
  expiry or owned run deadline. The admitted native client must forbid every
  sensitive application write at or after that deadline, including writes on
  an already-open HTTPS/WSS connection and automatic credential refresh. #62 must
  block further dispatch and terminate its owned client by that deadline; native
  write admission must also cover queued/background work so a controller timer
  or handshake-only check cannot substitute for the guard. Refuse a client whose
  deadline enforcement cannot be established before launch. Test a persistent
  connection crossing the earliest status expiry with queued application traffic
  and background refresh, and require absence of post-deadline application writes.
  Ordinary path validation, certificate validity, usage and
  constraints remain required in addition to both catalog and revocation checks.
  Microsoft's [chain validation API](https://learn.microsoft.com/en-us/windows/win32/api/wincrypt/nf-wincrypt-certgetcertificatechain)
  distinguishes offline/unknown revocation errors from successful validation.
  Under this closed endpoint policy, status must arrive as authenticated stapled
  evidence or as independently authenticated, fresh status material prepared by
  the trusted controller and sealed into the admitted client trust environment.
  Its provenance, certificate/issuer scope and expiry bind to that environment's
  receipt; stale host caches or supplied success flags cannot establish it.
  The client receives no general permission to contact certificate-supplied
  responder URLs. A native client that needs additional responder destinations
  requires a reviewed policy version defining those authorities and strictly
  credential-free status requests; missing native support for the current
  stapled/sealed-status policy refuses admission. Adverse client-admission tests
  must include a revoked non-root certificate absent from Disallowed, and
  missing, stale, wrong-certificate and unverifiable status. This policy supplies
  no claim of instantaneous knowledge of later CA revocations.
- Require direct service connections with authenticated hostname/chain checks;
  reject system/environment proxies, unverified DNS policy, endpoint overrides
  and ambient `CODEX_CA_CERTIFICATE`, `SSL_CERT_FILE` or `SSL_CERT_DIR` values.
  Inspect every effective configuration layer before launch. OpenAI documents
  [custom CA overrides](https://learn.chatgpt.com/docs/auth#custom-ca-bundles);
  the current provider forwards the two SSL variables, so its allowlist alone
  does not implement this qualification restriction.

Enforce the first-request boundary inside the admitted native client's transport.
Before placing real credentials in its temporary home or launching it, #62 must
have independently verified that the exact authenticated client/configuration
natively rejects a wrong hostname, untrusted certificate, prohibited endpoint or
cross-origin redirect before writing credentials or prompt bytes. No sensitive
TLS early data is allowed. Bind this admission evidence to exact client/tool bytes
and credential-free adverse-transport tests using synthetic data. Missing native
enforcement or an unverifiable configuration refuses that client before launch;
an executable hash by itself supplies no such capability.

The safe sequence is configuration/client admission, then the real client's
authenticated TLS handshake, then application data on that same connection.
The TLS implementation performs the in-path authentication before application
data; this does not require a nonexistent external pause in the stdio wrapper.
Independently collect content-free connection evidence during the run and require
it for final acceptance, preserving the admitted policy across reconnects. A
separate probe, a model string or after-the-fact detection cannot establish the
pre-dispatch guard. The existing wrapper and the cited user documentation do not
yet provide the required client-admission capability; full execution stays
unavailable until #62 establishes it. Do not substitute a broker or mock provider,
or claim that local size or
timeout bounds impose a service-side spending cap, or continue when authorization
or the applicable usage limit is unavailable. The reviewed host uses the existing
`livekit_local_v1` development profile. Its key and secret are fixed public,
predictable values; they cannot be made private or revoked per run. Their bounded
use relies on the dedicated environment, loopback signaling, effective firewall
policy and independently verified participant/media topology. The profile is not
an authentication boundary against untrusted local peers and cannot qualify a
shared or externally accessible deployment. Stop and remove the owned service and
its temporary state during cleanup; this retires that service instance, not the
public constants or their potential use by another instance. Real subscription
credentials and operational grants remain private.

Decline extending v1 into an adversarial-code execution architecture. Separate
credential brokers, per-request hostile-client capabilities and isolation from a
malicious host would require reviewed runtime changes and a new qualification
decision. Replacing the host's existing credential/provider path with those
components would qualify a different composition. These limits do not relax
strict validation of untrusted input data, protocol messages or observations.

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
