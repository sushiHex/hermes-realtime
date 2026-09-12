# Qualification input file bindings

[Execution protocol](qualification-execution.md) · [Evidence contract](evidence-capture.md)

The input-file layer verifies selected bytes and preserves their identities for
later consumers. It does not by itself authenticate a tool publisher, establish
an independent build, prove installation, or accept a full qualification report.
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

## Own immutable execution files

File handles alone do not prevent new import files from appearing in a directory.
`owned_execution_files` copies a bounded selection into a newly owned private
snapshot, independently compares the written bytes, applies protected directory
permissions that prevent new entries, and retains all file and ancestor handles.
It never adopts or changes an existing caller-selected directory. Its context
removes only its own verified snapshot after the lease ends.

This owner supplies filesystem evidence. It neither launches nor owns processes,
authenticates the selected contents, nor turns a copied directory into an installed
environment. The trusted process owner must verify that every consumer has stopped
before leaving the context. Full execution additionally requires the protocol's
durable ownership journal, authenticated tool environments, installation and
import-origin evidence, and independent cleanup observations.

`qualification_tool_process` invokes an admitted tool through the existing
Windows Job owner. Root exit and zero active Job accounting share one bounded
wait; a nonzero exit or incomplete finalization produces no completed invocation.
Its receipt retains the actual command, closed environment, root identity and
cleanup observation privately. That proves an invocation, not a governed build
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
