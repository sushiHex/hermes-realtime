# Qualification input file bindings

[Execution protocol](qualification-execution.md) · [Evidence contract](evidence-capture.md)

Qualification inputs have separate file, source, build, installation and publisher
authorities. Their final acceptance boundary requires matching lineage and
completed preparation cleanup while the final input files remain sealed. The
result supplies prerequisites for scenario execution; it does not accept a full
qualification report.
[Issue #47](https://github.com/sushiHex/hermes-realtime/issues/47) owns the complete
input authority; [#62](https://github.com/sushiHex/hermes-realtime/issues/62) owns
its composition with execution and cleanup evidence.

## Admit complete tool distributions

The [tool distribution policy](../scripts/qualification_tool_distributions.py)
selects complete Windows Git, uv and Python distributions by reviewed release
digest. The verifier checks the selected role, exact archive size and digest
before parsing any archive member. Its source comments link the official release
records used to admit those bytes. Callers cannot supply a replacement digest,
version or origin. Updating the selection requires normal source review.

The verifier retains every ordinary file as immutable bytes and rejects unsafe
paths, links, case aliases, file/directory collisions, duplicate members and
unbounded expansion. It preserves ordinary Windows resource names such as those
containing spaces or `+`; qualification-input-v1 artifact references retain their
separate, stricter path grammar. Source-locked Hatchling and its dependencies use
the dependency-file authority below, rather than a second package inventory.

The [tool environment owner](../scripts/qualification_tool_environment.py) copies
all three complete distributions into separate role directories in one immutable
snapshot. It retains file and ancestor handles and protects the directory
namespace before exposing any selected image to a consumer. Missing, swapped or
forged distribution capabilities refuse before filesystem work. Its metadata
expires with the owner, and cleanup removes only its newly created tree.

This establishes admitted tool bytes and their filesystem protection. It does not
execute a tool, establish build dependencies, bind an invocation to a candidate,
or prove an independent build. Every consumer still needs its governed command,
environment, process ownership, durable recovery and final-exit observations.
Those observations must precede release of the tool tree's seals and cleanup.

After verified tool-tree removal, `complete_tool_environment` can retain a
historical cleanup receipt tied to the exact original owner and admitted
distribution facts. Active or failed cleanup cannot mint that receipt. The
receipt neither reopens removed paths nor restores executable access; it lets
final acceptance check cleanup after normal tool use has ended.

`capture_candidate_source_archive_from_tools` passes the live tool owner to the
existing native Git child and source-archive authority. It derives the selected
image and expected version from the admitted distribution and revalidates the
owner around every Git invocation. It reuses the same candidate identity, object,
metadata and repeated-archive checks. The ordinary Program Files pin entry point
keeps its existing policy. Source capture still does not prove builds, installed
runtime execution or the complete runner's durable recovery protocol.
The complete-tool entry point retains its observed distribution identities only
after source capture and Git-child cleanup succeed. Build input acceptance requires
that provenance to match its admitted tools; a legacy image-pin archive cannot be
upgraded by attaching unrelated tool metadata afterward. Completed capture facts
remain readable after disposable tool trees are removed, without retaining
execution authority over them.

Canonical source comparison disables external diff and text conversion helpers.
Attribute inspection reads the exact candidate tree and refuses clean/process
filters before comparing the worktree. Every Git child uses that same committed
attribute source, so changed index or worktree attributes cannot select a local
helper. Hook, fsmonitor and ambient configuration exclusions remain in force.

## Bind complete browser and LiveKit files

The [Chrome distribution admission](../scripts/qualification_chrome_files.py)
authenticates the complete reviewed Chrome for Testing archive before parsing it.
It preserves every ordinary publisher file and derives the canonical version
manifest from those bytes. The final binding requires that exact manifest, every
sealed member, the selected executable and version, and the same final candidate
authority. An updated manifest hash cannot authorize a missing or changed file.

Chrome resource names use the shared ordinary Windows path rules, including
publisher names such as `First Run`. Direct qualification artifact references
retain their stricter grammar. Public Chrome artifact IDs use the SHA-256 of each
relative resource name; the sealed version manifest retains its exact spelling.
This keeps both report schemas unchanged and covers nested resource directories.

The [LiveKit admission](../scripts/qualification_livekit_files.py) derives its
archive and executable selections from the candidate's existing workflow and
browser integration test. It owns the archive and its complete publisher file
set under separate namespace prefixes, then compares the final archive and
executable while both sets of seals remain live. Completed source facts and final
seals preserve that binding after the preparation workspace closes.

Both use the existing bounded archive parser. The Chrome policy permits its
reviewed large native library without widening the default tool-member limit.
These bindings do not execute a browser or server, establish version output or
readiness, create an operator profile, or authorize changes to an installation.
Execution requires the separately owned immutable namespace and process lifecycle.

## Bind the complete Codex publisher package

The [Codex distribution admission](../scripts/qualification_codex_files.py)
authenticates the reviewed Windows package before archive parsing. It retains all
six publisher files, including the command runner, code-mode host, resource
directory and search-path directory, and validates the publisher's layout
metadata. The final executable must match both that package and the fixture in
the genuine candidate archive. The frozen model and effort come from the
candidate's governed report contract.

The existing `codex_executable` role identifies the selected executable. The
separate immutable package capability preserves the rest of its publisher
namespace for the later execution owner; it does not invent another artifact
role or make a single executable stand for the whole installation. Admission and
binding do not execute Codex or its sandbox setup, provision credentials, or
establish the full host's isolated launch environment.

## Retain before consuming

`bind_build_inputs` binds the genuine candidate archive and a live complete tool
owner before any build output exists. It requires the source-selected Hatchling
backend, authenticates each immutable wheel against the archived source lock
before ZIP parsing, and closes only the backend's transitive dependency graph.
Unrelated distributions, alternate backends, source-relative backend imports and
rehashed replacement files refuse. Its private capability expires with the tool
owner. It retains source, project, lock, requirements and wheel identities for a
later installation consumer; it cannot be used as proof that a build ran.

After final input files exist, `bind_build_input_files` compares their candidate,
tool images and versions, backend wheel, build-wheel set, requirements and
constraints with that earlier capability. A final manifest cannot substitute new
tool or package bytes by updating its own hashes. This binds the earlier inputs
to the final input digest; independent build and installation receipts remain
separate requirements.
The original tool seals must still be live when this binding is created. After
that comparison, its retained source and input facts remain verifiable through
the final input seals without reopening removed tool trees. It cannot authorize
a new build, or create a new binding after the original tool owner closes.

`retain_qualification_input_files` seals the canonical input manifest before
discovering its direct and transitive files. It retains the exact artifacts and
their ancestors while the existing strict input validator reopens them. The
Windows file owner requires a local fixed NTFS volume, ordinary nonredirected
paths, distinct single-link file identities and bounded file counts and bytes.
Open handles deny file writes, replacement and deletion for the lease's lifetime.
Hard links, alternate streams, ambiguous paths and closed or forged capabilities
are refused.

`bind_candidate_files` requires the genuine existing Git archive authority. It
compares the input's full candidate identity, governing source documents, runners
and schemas with that archive. It reuses the wheel authority to compare all four
wheel roles with source runtime bytes and static package metadata. Both sdists
must contain exactly the source-selected files and matching package metadata;
inspection does not extract or execute them. The execution-plan blob is retained
as a separate source digest within the binding.

The sdist profile includes the tracked root `.gitignore` that
[Hatchling 1.27.0 force-includes](https://github.com/pypa/hatch/blob/hatchling-v1.27.0/backend/src/hatchling/builders/sdist.py#L297).
Its bytes must match the candidate just like every explicitly selected file;
ambient exclusion files or other undeclared extras are refused.

Equal repeated artifact hashes do not establish independent repeated builds.
Those require the separately owned invocations and output receipts described in
the execution protocol.

## Verify dependency contents

`bind_dependency_files` reads wheelhouse manifests, resolved requirements,
constraints and every wheel through the retained file handles. It checks each
purpose's platform and Python version, the exact Hatchling file role, wheel
metadata, RECORD contents, pins and dependency edges. The three realtime runtime
closures and the Hermes plugin runtime also include the genuine candidate wheel
in that check. The Windows direct and sdist-built runtime purposes activate its
`local` extra; optional and transitive extras must have their required packages.
Build traversal starts at Hatchling; runtime traversal starts at the candidate.
Every supplied distribution must be reachable from that purpose's root, including
its activated extras. Presence in the source lock alone does not admit an
unrelated package into the installed namespace.

Windows manifests must name the full admitted interpreter version. The standalone
Hermes PluginManager consumer accepts that canonical `3.11.x` spelling as well as
its legacy `3.11` spelling; complete input binding still requires the exact patch
version. This format compatibility supplies no interpreter or installed-host
execution evidence by itself.

The Moonshine and Kokoro distribution roles must also match the exact versions
and wheel bytes admitted in both Windows runtime purposes. A separately hashed
provider file cannot substitute for the distribution that would be installed.
This comparison supplies no model-loading or provider-execution evidence.

`bind_dependency_purpose` applies that same verifier to one selected installation
purpose. Its separate capability carries only that purpose's coverage and cannot
be submitted as an all-purpose dependency binding or relabelled for another
installation. This allows Windows and Linux prerequisites to be established
independently. `bind_dependency_files` still requires all five closures for
complete input acceptance; a valid Windows receipt supplies no Linux authority.

Every third-party wheel must also match the exact candidate's archived `uv.lock`
entry: package, version, basename, SHA-256 and size, with the governed PyPI
registry and file origin. The binding reads that source blob through the genuine
archive authority. It checks those bindings before opening wheel ZIP contents.
Replacing a wheel and updating all supplied pins and manifests
cannot authorize the replacement. The candidate wheel retains its separate
source comparison. This reuses the reviewed source lock; it does not create a
second dependency inventory or claim independent publisher signatures. A build
tool or dependency absent from that lock remains unavailable until its exact
closure is reviewed and added through the normal dependency procedure.

Requirement files use exact pins and one hash per selected wheel. Installer
options, URLs, editable sources and unresolved markers in those files are
refused. Ordinary native resource names use the same Windows path validation as
the immutable execution owner; names containing `+` or spaces retain their exact
bytes, while traversal, aliases and namespace collisions refuse. Native
Linux compatibility requires a genuine Linux ABI inventory; this preliminary
inspector only admits generic `linux_x86_64` and compatible pure wheels. Reading
a Linux wheelhouse on Windows does not establish Linux installation or execution.

Before installation, the shared wheel inventory also rejects two distributions
owning the same import file, Windows case aliases, and file/directory collisions.
It accounts for the [wheel format's data relocation](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#installing-a-wheel-distribution-1-0-py32-none-any-whl):
`purelib` and `platlib` members join the same import namespace. Other data schemes
need an independently observed target layout and remain explicitly unavailable in
this preliminary inspector. It never assumes their destinations or permits an
overwrite merely because both input wheels have valid hashes.

The metadata reader supports the pre-2.4 `License-File` extension found in
source-locked dependencies. It requires exactly one corresponding file in the
legacy `.dist-info` or early Hatchling `.dist-info/licenses` layout. Metadata 2.4
and later use the standard `licenses` layout. Missing or ambiguous references,
unsafe paths, invalid dependency declarations, unknown fields and changed RECORD
contents still refuse. This interpretation preserves all original wheel bytes;
it neither rewrites license metadata nor changes package digests. See the
[historical packaging layouts](https://peps.python.org/pep-0639/appendix-license-survey/#setuptools-and-wheel)
and [current license-file requirements](https://peps.python.org/pep-0639/#add-license-file-field).

Only the wheel's top-level `.dist-info` directory supplies its distribution
metadata. Nested vendored metadata remains original payload covered by RECORD,
and cannot replace the installed distribution owner.

The ordinary wheel inspector refuses `.pth` startup hooks by default. Qualification
recipes select its explicit no-site-processing profile: the original files remain
in the byte inventory, but every Python consumer must run under `-I -S -B` and
add only its sealed package directory directly. The worker checks these flags
before importing installed code. It never calls `site.addsitedir` or `site.main`.
This follows [Python's site processing contract](https://docs.python.org/3.11/library/site.html);
it grants no permission to execute a hook or launch the environment with ordinary
site initialization. Tests exercise a hook with a visible side effect: it remains
inert under the worker and executes under an explicit site-processing control.

`inspect_installed_wheels` independently compares an observed Windows uv target
with the original wheel members, including purelib/platlib relocation. Rewritten
RECORD files must cover exactly each distribution's original files, declared
entry-point images and the pinned installer metadata. Added imports, source
changes hidden behind rewritten hashes, cross-distribution ownership, namespace
collisions and indirect cache metadata refuse. The complete inventory includes
the empty target lock and every generated file. Its result is ordinary byte
metadata, not installation or entry-point execution authority; an installed
receipt still requires the actual owned installer, live runtime/package seals,
observed imports and cleanup.

## Bind provider resources before loading

`bind_kokoro_resources` reads the existing `_MODEL_ASSET` and `_VOICES_ASSET`
literal selections from the candidate's source-verified wheel. It does not import
the provider or maintain a second model pin list. It requires the same live input,
candidate and Windows dependency-purpose authorities, including the admitted
Kokoro distribution, before comparing the complete resource manifest and reopening
each sealed model file. Rehashing a replacement manifest cannot authorize different
model bytes, names, missing resources or additional resources.

The resulting capability covers only the selected direct or sdist-built Windows
purpose and expires with the input seals. Its metadata contains digests, counts
and the closed purpose value; resource paths and bytes remain private to the
consumer. This establishes model provenance before loading. It does not construct
Kokoro, select an execution namespace, prove provider behavior, or admit Moonshine
resources. The full host still needs the execution protocol's sealed pre-load
namespace and fixed provider profile.

The [Moonshine catalog observer](../scripts/qualification_moonshine_catalog.py)
uses one live installed Windows runtime authority. It derives the candidate,
source-selected worker, package distribution, tool environment and sealed import
namespace from that owner. The fixed native request observes English medium
resources with and without spelling resources before model construction. The
controller verifies the actual process, implementation bytes, output, publisher
origin, resource bounds and the identical primary selection in both responses.

The catalog owns resource names, sizes and CRC32C checksums; it does not turn a
caller-supplied model SHA-256 into publisher evidence. Catalog observations
survive successful execution-tree cleanup while their final input authorities
remain live; they provide no execution authority over removed trees.

The [preparation owner](../scripts/qualification_moonshine_preparation.py) acquires
resources before the final input document exists. It authenticates the
source-locked wheel and invokes the candidate-archived worker with that wheel's
native catalog API. The worker fetches the seven English medium resources and
two spelling resources from their fixed publisher origins, with normal TLS
verification and no redirects. Both worker and controller verify sizes and
CRC32C checksums; SHA-256 then identifies each acquired resource. The execution
owner must finish cleanup before preparation exposes its sealed outputs.
The separate resource owner retains those originals for final transfer.

The [final resource binder](../scripts/qualification_moonshine_resources.py)
compares those originals with the final sealed manifest and resource bytes. It
requires the same candidate, dependency purpose, wheel, worker, native API and
complete catalog selection. Transfer occurs while both sets of seals are live.
The resulting receipt becomes consumable only after original-resource cleanup
succeeds, then relies on retained transfer facts and the final input seals.
Unknown, pending or failed cleanup refuses consumption. The two Windows runtime
purposes each require their corresponding catalog authority.

Public provider-resource identifiers contain the provider and SHA-256 of the
relative resource name. Exact names remain in the sealed manifest and its model
identity calculation. Nested publisher cache paths therefore fit both existing
report schemas without weakening either schema's identifier grammar. Catalog,
acquisition and transfer evidence do not establish model loading or speech
recognition behavior.

## Admit the Linux runtime descriptor

The [Linux image admission](../scripts/qualification_linux_image.py) selects the
official Python 3.11.16 slim-bookworm image's exact `linux/amd64` manifest. It
authenticates that descriptor before following its configuration reference and
retains the complete compressed-layer and root-filesystem digest graph. Caller
tags or replacement digests cannot change the reviewed selection. The Windows
build interpreter retains its separate role.

Descriptor admission does not establish downloaded image layers, an immutable
container, observed Linux ABI compatibility, installed execution, cleanup, or
an authenticated GitHub Actions receipt. The Linux executor must establish those
before this prerequisite can support complete input acceptance.

## Bind authenticated Linux execution records

The [GitHub receipt consumer](../scripts/github_actions_linux_receipt.py) binds
service observations to one final Linux dependency-purpose authority and the
admitted image. It checks the exact repository, workflow source, candidate,
first run attempt, successful Linux job and artifact identities, then hashes the
transferred archive and parses its single canonical receipt. Supplied JSON or a
successful job label alone cannot mint its authenticated capability. Signed
artifact transfers receive no GitHub API bearer credential.

The authenticated receipt retains its input, runtime, ABI and installation facts
and revalidates the final input seals on consumption. Parsing the private payload
produces ordinary validated data, with no execution or service authority.

The [prerequisite owner](../scripts/qualification_linux_prerequisite.py) accepts
the earlier source/build capabilities, independently produced direct wheel,
source-locked Linux wheelhouse and admitted image descriptor. It seals that
recipe before observing the service, without requiring a final input document.
Both phases reuse one HTTPS and service-identity verifier.

`bind_prefinal_linux_receipt` compares the authenticated prerequisite with the
final Linux dependency and image facts while the original and final seals are
live. Transfer is single-use. The resulting binding retains completed service
facts and the final input seals, allowing intermediate workspaces to close.
The earlier final-only consumer cannot substitute for that phase transition.

The preliminary recipe admits source-locked wheel bytes without claiming ABI
compatibility. After service authentication, the observed CPython ABI and glibc
version determine the supported manylinux tags. The verifier checks the complete
dependency graph and retains that target with the exact source, wheelhouse,
candidate wheel and image identities. Both the single-purpose and five-purpose
final dependency binders accept that same opaque target; a target from another
recipe cannot authorize the final Linux inputs.

The [hosted producer](../scripts/qualification_linux_producer.py) owns two
containers and one fresh installation volume. The first container installs the
runtime-only, hash-pinned wheel recipe offline. After it exits and is removed,
the second container observes the same volume read-only. It runs as the verified
numeric owner of the newly created private output directory, with all Linux
capabilities dropped. Its private scratch mount has the same owner. The admitted
image, source and input mounts are immutable during observation; writable
temporary data stays outside the installed import namespace. The
[stdlib worker](../scripts/qualification_linux_worker.py) compares installed
wheel bytes, installer-generated entry points and RECORD coverage, checks import
origins, and runs the installed Linux null-capture behavior. The producer writes
the receipt only after container, volume and observation-workspace cleanup
succeeds. The existing portable tests use a separate environment.

Docker commands drain both output streams into bounded buffers while running:
16 MiB for stdout and 1 MiB for stderr. Exceeding either bound stops the command.
One 360-second deadline includes termination, process reaping and reader cleanup;
the final five seconds are reserved for cleanup. Success requires complete reads,
valid UTF-8 and a reaped process. Refusals expose fixed messages rather than child
output, and interrupted observation cannot produce an installation receipt.

If observation fails, the worker attempts to write one bounded, transient marker
containing only its version and a stage from a closed vocabulary. The producer
validates the marker before reporting the stage; missing or malformed markers
leave a generic failure. Exception text and process output remain private. This
diagnostic cannot mint a receipt or bypass cleanup.

The workflow publishes the receipt from its Linux job. Acceptance still requires
the trusted service's completed first-attempt run and artifact observations;
local controller tests and an uploaded JSON file do not supply that authority.
Adapter tests use synthetic build/service observations and real Windows file
owners. Native hosted execution remains a distinct validation requirement.

## Own immutable execution files

File handles alone do not prevent new import files from appearing in a directory.
`owned_execution_files` copies a bounded selection into a newly owned private
snapshot, independently compares the written bytes, applies protected directory
permissions that prevent new entries, and retains all file and ancestor handles.
It never adopts or changes an existing caller-selected directory. Its context
removes only its own verified snapshot after the lease ends.

The controller retains noninheritable security handles before freezing the
snapshot. Frozen permissions deny ordinary changes to permissions and ownership,
including the owner's implicit permission-changing rights through Windows
[Owner Rights](https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/manage/understand-special-identities-groups#owner-rights).
Cleanup restores permissions through those retained handles. Capability access
and completion recheck the exact bounded namespace and frozen security state.
These properties apply within the protocol's
[trusted computing base](qualification-execution.md#own-the-execution-environment);
they do not establish isolation from malicious software exercising privileged
OS authority.

This owner supplies filesystem evidence. It neither launches nor owns processes,
authenticates the selected contents, nor turns a copied directory into an installed
environment. The trusted process owner must verify that every consumer has stopped
before leaving the context. Full execution additionally requires the protocol's
durable ownership journal, authenticated tool environments, installation and
import-origin evidence, and independent cleanup observations.

`qualification_tool_process` invokes an admitted tool through the existing
Windows Job owner. Root exit and zero active Job accounting share one bounded
wait; a nonzero exit or incomplete finalization produces no completed invocation.
It observes live descendants and retains their classified process identities.
Before completion, the Job's cumulative process count must equal the complete
retained inventory, with zero active processes and no limit-terminated processes.
An unobserved short-lived child therefore refuses completion. Every retained
process must signal and exit normally within the same budget; the invocation
also bounds total process lifetimes to 128. See Microsoft's
[Job lifetime accounting](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information).

System directories come from native Windows APIs, with fixed-drive and nonreparse
validation. The Windows console host, which can accompany a windowless tool, and
the command shell have explicit OS roles bound to native System32 image digests
and retained parent identities. The shell is needed by
[CPython 3.11.16's Windows version query](https://github.com/python/cpython/blob/v3.11.16/Lib/platform.py),
which the admitted uv interpreter probe invokes. Ambient `SystemRoot`,
`SystemDrive` and `WINDIR` values cannot select those inputs. The accepted OS
trust boundary remains unchanged.

Its receipt retains the actual command, closed environment, process identities and
cleanup observation privately. Counts and OS image digests carry no paths or
process identifiers. That proves an invocation, not a governed build
recipe, independent repeat, installed import closure or full qualification.

`OwnedQualificationWorkV1` composes those existing owners with their file
dependencies. Enter the complete tool and input file contexts into this owner
before invoking a consumer. If process finalization fails, it retains those
contexts and the original Job owner for cleanup retry. Nested work failures
retain the outer dependency tree too. Further dispatch is refused after failure;
resource release follows successful process cleanup. File cleanup failure remains
a terminal refusal. This in-process owner supplies no durable journal or recovery
after controller death; the complete runner must establish those separately.

`install_build_environment` uses that work owner to install the source-locked
Hatchling dependency closure with the admitted uv and Python distributions. The
recipe disables indexes, dependency resolution and source builds, enforces hashes,
and selects only its sealed wheel directory. It independently reopens the installed
files, compares the complete layout with the original wheels, then copies and seals
the executable/import namespace before any package import. The isolated import
probe admits only the sealed packages and complete Python runtime. Its result is
bound to the retained process identity; supplied or ambiguous observations refuse.
The worker's executable bytes come from the genuine candidate source archive.
A missing archived worker refuses; the controller never substitutes its local
checkout. Both import checks and builds run that worker under `-I -S -B`.

The resulting live build-environment capability records the actual installer and
importer invocations and expires when its owning work starts closing. Invalid
installed bytes or import observations also stop further dispatch in that work.
This supplies an installation recipe, not a candidate build, independent repeated
build, generated artifact receipt or complete final qualification input.

## Produce independent artifacts

`build_candidate_artifacts` owns six separate builds: direct and repeated wheels,
direct and repeated sdists, and a wheel from each produced sdist. Each invocation
gets a fresh sealed source tree, independently installed and inspected backend,
and separate output workspace. The source tree comes from the genuine archive;
sdist-built wheels instead use the exact files admitted by the existing sdist
comparison. The worker pins the admitted Hatchling reproducibility timestamp.

The worker and the Pure CI job pass each newly built wheel through the same
`canonicalize_wheel_file` function in the archived
[build worker](../scripts/qualification_build_worker.py). It limits input reads
and replaces the output only after serialization succeeds. It fixes ZIP member
order and metadata and stores member bytes without compression, removing host
platform and compressor differences from the artifact identity. Member contents,
including RECORD, remain byte-identical. The existing wheel size limit and
source/metadata/RECORD validation still apply. Linux receipt authentication
requires the exact resulting direct wheel; compatible package contents alone
cannot satisfy that comparison.

The recipe requires normal process exit and complete Job cleanup before reading
outputs. It independently compares every wheel with the source/wheel authority
and every sdist with the source-selected files and package metadata. It seals an
immutable copy of each output before releasing the original output's handles.
The repeated builds must have distinct retained process identities and output
workspaces, the same source and backend closure, correct source lineage, and
byte-identical results. Supplied observations cannot create a build receipt.

Each temporary build environment is removed before the recipe completes. Completed
invocation, installation, output and cleanup facts remain available afterward;
they do not retain execution authority over deleted directories. Intermediate
output consumption still requires its live seals. `bind_build_output_files`
compares all six actual final file roles and the candidate identity with those
produced outputs while both sets of seals remain live. Subsequent validation
retains the final input seals and completed build facts, without reopening removed
intermediate trees. Closing the final input owner expires that binding.

The [build tests](../tests/test_qualification_builds.py) cover source selection,
output comparison, replay refusal, incomplete cleanup and binding lifetimes.
Their fault-injection rows test individual comparisons; they do not constitute
native build evidence. A successful native recipe establishes its exercised
builds and cleanup, without supplying the separate Linux, runtime, provider,
Hermes, browser, benchmark or full-run prerequisites.

## Produce benchmark inputs together

The [benchmark recipe](../scripts/qualification_benchmark.py) consumes the
pre-build source and admitted Python authority. One archived isolated worker
creates the live machine manifest and admission benchmark report in the same
owned invocation. The existing benchmark validators check the output grammar,
source identity and threshold results; the owner additionally binds those bytes
to the actual process and its cleanup. Machine and input identities use the
same canonical Python patch version.

Produced artifacts stay sealed for transfer into the two final input roles.
`bind_benchmark_output_files` requires the source, Python identity and both
outputs to agree while the produced and final seals are live. Completed facts
survive disposal of the benchmark execution tree; final bindings expire with the
final input owner. A supplied report, synthetic test timing or ordinary local
preparation run does not establish qualification on a dedicated target host.

## Install each runtime purpose

The runtime recipe reuses the same offline installer and installed-file inspector
as the build recipe. It selects the exact purpose's resolved dependency files,
including the candidate wheel from its separately bound artifact role when the
wheelhouse lists only third-party dependencies. It requires the final input
digest, genuine source, produced artifact binding, and complete admitted tool
environment to agree before allocating an installation workspace.

`install_runtime_environment` supports the two Windows realtime purposes and the
Hermes plugin purpose. Each call creates a fresh installation, seals its import
namespace, and runs the archived worker to import the installed candidate's
package, host launcher, local launcher and plugin module. It checks their origins
and package version without calling a launcher, registering a plugin, constructing
providers or activating capture. Linux purposes refuse this Windows recipe.

Live runtime access ends when its work owner begins closing. Only verified work
cleanup can mint `CompletedRuntimeEnvironmentV1`; its final-input bindings and
actual invocation facts remain verifiable without reopening removed execution
trees. The final input root must have a separate owner that remains live through
acceptance. These installed import facts do not prove Hermes PluginManager
execution, provider model construction, Linux installation or physical behavior.

The [runtime tests](../tests/test_qualification_runtime_environment.py) exercise
purpose and source binding, unavailable prerequisites and installed candidate
selection. Worker tests use synthetic modules to test import isolation and
refusals; those fixtures do not establish a complete installed runtime closure.

## Bind and exercise upstream Hermes source

The [Hermes source admission](../scripts/qualification_hermes_source.py)
authenticates the reviewed official release archive before parsing it. It checks
the decompressed archive separately, preserves every source file and mode, and
normalizes archive metadata under the harness's expected source prefix. The
complete publisher source is retained in an owned immutable namespace. Only the
dedicated source-archive role uses the larger archive bound; ordinary source
files keep their existing limit.

Final transfer compares that normalized source and the candidate's PluginManager
harness with their final sealed artifact roles while both authorities are live.
The [runtime qualifier](../scripts/qualification_hermes_pluginmanager_runtime.py)
uses the matching installed Hermes-purpose runtime. A candidate-archived stdlib
worker selects the harness's three fixed literal stages: discovery while
disabled, CLI enablement, and enabled PluginManager registration. Each stage runs
in an isolated child under the existing process owner. Imports use the immutable
official source and installed packages; the qualifier does not create another
venv or install missing dependencies.

The candidate's `qualification-hermes` dependency group supplies the additional
packages required by the reviewed upstream import path. The dependency verifier
checks that group's exact pins in the genuine source archive, then checks its
wheel bytes against the same archived lockfile. These roots apply only to the
Hermes PluginManager purpose. Ordinary runtime, local-provider, Linux and build
dependency selections retain their own policies; callers cannot add roots or
select another group.

The Hermes and development groups are
[explicitly mutually exclusive](https://docs.astral.sh/uv/concepts/projects/config/#conflicting-dependencies)
in the shared lockfile: the pinned upstream requires a different `cryptography`
version from development. Export the Hermes recipe with default groups disabled
and `qualification-hermes` selected. This preserves the ordinary development
selection while keeping the upstream compatibility environment reproducible.

The parent checks the stage results against actual installed module and
distribution identities, the exact permitted configuration change, and its
disposable profile roots. Preparation creates the pinned upstream's fixed profile
skeleton, empty plugin directory and default document before the tested
stages, without creating a configuration file. Those prepared entries form the
unchanged baseline. The child's home and default-profile lookups resolve inside a
separate owned temporary
directory; that entire directory must remain unchanged.

The configuration comparison includes the pinned upstream CLI's version and
disabled-plugin fields. The verifier checks the complete profile tree with
bounded traversal and file reads, rejecting unrelated changes. CLI output has a
bounded private channel separate from the harness's one canonical JSON result;
neither profile contents nor output text enters the completed receipt.
The isolated flow admits no log files and makes no logging-initialization claim.

Its completed receipt requires worker/child exit and
cleanup of the qualifier's own workspace. The caller's installed runtime may
remain operational afterward; final input acceptance separately requires the
matching completed runtime-cleanup receipt. Missing upstream dependencies refuse
execution and require a reviewed dependency recipe. Synthetic stage tests do not
establish compatibility with the complete official upstream source.

## Accept the complete input authority

The [acceptance boundary](../scripts/qualification_input_acceptance.py) consumes
the existing file, build, dependency, publisher, benchmark, Linux and completed
Windows runtime capabilities. It requires all three Windows runtime purposes,
both Windows provider bindings, the matching Hermes PluginManager result, and
completed cleanup of the original preparation and tool owners. It checks exact
shared authority objects as well as their source and artifact identities.

The accepted capability retains those receipts and revalidates them whenever it
is consumed. Its metadata is the existing input-closure record; it adds no
report fields. Missing purposes, mismatched lineage, failed cleanup or expired
final seals refuse acceptance. Operational file bindings remain usable while
their owners are working; completed input acceptance has the stricter cleanup
requirement. Removed execution trees are never reopened to validate historical
facts.

This boundary does not launch scenarios or produce a full qualification report.
The complete runner must preserve the three-stage preparation order, retain the
final input seals through its consumers, and compose actual scenario, operator
and cleanup evidence under the execution protocol. Passing input-acceptance tests
does not establish a completed desktop qualification.

## Evidence and privacy

The [file-seal tests](../tests/test_qualification_file_seals.py),
[input-graph tests](../tests/test_retained_qualification_inputs.py),
[source-binding tests](../tests/test_qualification_candidate_files.py),
[wheelhouse tests](../tests/test_qualification_wheelhouse.py), and
[snapshot tests](../tests/test_qualification_execution_files.py) exercise real
Windows handles and permissions, genuine source archives, source-derived package
bytes, and adversarial refusal cases. Synthetic package fixtures provide no
publisher, build, installation, physical or service-authentication evidence.

The [distribution tests](../tests/test_qualification_tool_distributions.py) use
explicit synthetic trust substitutions only to test archive inspection. The
[tool-owner tests](../tests/test_qualification_tool_environment.py) exercise real
Windows namespace protection and cleanup; they do not execute their synthetic
images. Actual artifact admission records must identify the governed release
digests separately from these parser and ownership tests.
The [owned-source tests](../tests/test_qualification_tool_source.py) require a live
tool owner before Git dispatch and preserve the ordinary pin's location boundary.

Live bindings expire when their retained input owner closes. Completed build
facts remain readable as historical observations. Metadata values cannot
be submitted in place of those capabilities. Private inventories contain relative
member names; retained owner state also holds file identities. Keep both outside
public reports and comments.
Public evidence uses only the permitted source/artifact digests, aggregate counts,
closed outcomes and exact CI links. Full qualification remains unavailable until
all separately required authorities are established.
