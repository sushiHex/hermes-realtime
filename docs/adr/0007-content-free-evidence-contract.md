# ADR 0007: One contract for content-free evidence

Status: Proposed. It awaits the ten owner decisions listed below. No work item exists yet; each
migration step gets an issue before work starts ([ADR 0001](0001-github-work-tracking.md)).
It applies [ADR 0006](0006-architecture-direction.md) Move 1 ("one definition") to evidence
markers and receipts, and proposes one amendment to that move's wording (owner decision 1).
Reviewed source: `be07b5f`. Every `file:line` was re-read at that commit and is relative to
`src/hermes_realtime/` unless it starts with `scripts/`, `tests/`, `web/`, `docs/` or
`.github/`. Nothing below is implemented. This record changes no marker, schema or gate.

## Context

The development guide requires that a refusal leave one bounded JSON line, prefixed
`[marker-name]`, carrying counts, kinds and categories only. Those lines, and the private
receipt files written beside them, are the project's main privacy boundary and its main
diagnostic channel. The channel has no owner: each module defines its own name, its own emit
code, its own vocabulary and, in tests and scripts, its own parser.

Measured at `25cabec`; `src/`, `scripts/` and `web/` are unchanged from there to `be07b5f`.
**The counts are syntactic**: a scan of string and bytes constants holding `[name]` with at
least one hyphen. A prefix assembled from parts, or a one-word name, is not counted. The
catalogue test of migration step 1 becomes the owner of the per-marker list; no copy of it is
kept in `docs/`.

| Measure | Value |
| --- | --- |
| Marker names with an emit site | 151 in Python (99 only in the package, 51 only in `scripts/`, 1 in both), 12 in the browser |
| Names read or declared but never written | 2: `lookup-presentation` (`scripts/lookup_backend_gateway.py:143`), `voice-archive-path-step` (`scripts/qualify_voice_archive_path.py:65`) |
| Python emit sites | 164 in 78 files: stdout 138, stderr 20, logging 6; plus 25 `console.info` sites in 7 browser files |
| Position of the write | inside a `finally` 118; inside an `except` only 4; neither 42 |
| Guard on the write | none 138; `suppress` 16; `try/except` 10 |
| Byte layouts | hand-typed JSON literal 33; compact with insertion order 63; compact and sorted 37; default separators 31; 33 sites do not flush; 2 names have no space after the prefix |
| Names with no `version` member | 59 of 151 |
| Discriminating key | `category` 86 names, `refusal` 27, `failure` or `failure_kind` 9, `cause` 8, `outcome` 7 (overlapping) |
| Emit helpers | 38 functions, 10 of them a module-private `_marker` (for example `companion/archive.py:162`); 4 emitting context managers |
| Label-shaped regexes | 44 sites, 16 spellings; 6 admit an evidence value (for example `scripts/preview_cutover.py:35`, `scripts/rehearse_desktop_mvp.py:99`) |
| Exception identity | 47 uses of the raw class name; 6 separate exact-type tables (for example `providers/kokoro_worker.py:166`) |
| Hand-written readers | 2 generic regexes (`scripts/rehearse_desktop_mvp.py:81`, `scripts/real_hermes_api_gate.py:98`); 11 dedicated readers in `scripts/`; in `tests/`, 16 local helpers and about 200 inline idioms |
| Names never spelled in a test | 63 of the 165 names above (literal search) |
| Names spelled in `docs/` | 44; in a workflow file or `scripts/release_gate.py`: 0 |
| File writes | 8 temporary-then-publish helpers (1 in the package, 7 in `scripts/`); elsewhere in `scripts/` 40 exclusive creates of the final name, 54 `write_text`, 21 `write_bytes` |

Privacy rests on shape, not vocabulary. Where a value is checked at all it is checked against a
label regex. `scripts/rehearse_desktop_mvp.py:99` admits any string of up to 64 letters,
digits, dots, colons, underscores and hyphens: a host name with a port, or a model tag, is a
valid "category".

### What the arrangement allows

Each row is one example on main, not a hypothetical.

| Class | Example |
| --- | --- |
| Lost category | `scripts/preview_cutover.py:1336-1344` keeps one of the tokens `parse_preflight_result` (`:1122`) raises and reports the rest as `preparation_failed`. Behind it, `_is_category_refusal` (`:2591`) is a hand-kept tuple of three exception types, so a fourth is converted by hand (`:2231`). Also: `integration/native_web_evidence.py:387` prints one `refused` for six call sites; `scripts/preview_cutover.py:2537-2551` writes `failure_category` to the receipt and omits it from the line |
| Prose-only refusal | `scripts/local_livekit.py:170`: a missing server, a checksum mismatch and a refused archive leave one sentence on stderr and no marker. In the package: `host_launcher.py:3749`, `:3767` |
| Content leak, live | A wire-supplied category, typed as an open bounded string (`protocol/events.py:364`, `:412`), is printed unchanged at `integration/voice_memory.py:188`, `voice_review.py:205`, `voice_archive.py:234`, `:240`, `voice_forget.py:177`. A model name from the command line reaches `scripts/probe_confirmed_natural_work.py:505` through `:460`. A directory name is printed at `scripts/rehearse_desktop_mvp.py:2996` |
| Content leak, latent | `integration/codex_access_credentials.py:185` prints `str(error)` as the category. Every raise site passes a fixed token today |
| Reader and writer drift | `scripts/lookup_backend_logs.py:23` restates the schema and the seven kinds that `conversation/streaming.py:2618-2623` writes; the same file restates three more vocabularies (`:38`, `:40`). `scripts/lookup_backend_gateway.py:143` reads a marker nothing in the tree writes |
| A test parses differently from the writer | `scripts/storage_process.py:154` writes no space after the prefix. Both generic regexes require one. `tests/test_synthetic_fault_matrix.py:402` strips the prefix and parses, which accepts either form |
| The `finally` does not do the work | It runs and says nothing: `integration/tool_catalog.py:75-79` sets the category only for `Exception`. It is skipped: `providers/kokoro_worker.py:1001-1002` call `os._exit`, so the emit at `:1177` never runs, and 70 and 71 are absent from the table at `:36`. It replaces the refusal: 138 unguarded writes, for example `conversation/task_status.py:40`; the guarded ones drop the line uncounted (`conversation/result_projection.py:190`) |

The six classes share one cause: a vocabulary, a line shape, a refusal's category and "one line
at exit" are each restated at every site that needs them.

## Decision

**One declaration owns each evidence line and each content-free record. It is the emitter, the
parser and the test oracle. Its field types cannot express a free string.**

### The value grammar has no string kind

A field is exactly one of:

| Kind | Rule |
| --- | --- |
| Label | A member of a `Literal[...]` set declared on the field. Membership, not shape, is the safety property |
| Flag | Exact `bool` |
| Count | Exact `int`, never `bool`, never negative; saturates at a maximum that every count must declare |
| Signed | Exact `int` in a declared range (exit status, `errno`) |
| Digest | Lowercase hex of length 16, 32 or 64, with a declared label for what was hashed |
| Optional | Any of the above, or null |
| Labels, Tally | A bounded list of labels; an object whose keys are labels and whose values are counts |

- The runtime vocabulary is derived from the same `Literal` annotation the type checker reads.
  One definition refuses a non-literal at the call site, a non-member at encoding and a
  non-member at parsing. `scripts/preview_readiness.py:8` and `providers/kokoro_worker.py:193`
  already declare vocabularies this way.
- A declaration whose field is not one of these kinds raises when its module is imported.
  `category: str` cannot be declared.
- Exception identity is a label read from one shared exact-type table. An unlisted type reads
  `other`. A caller cannot supply the string.

### One declaration is the emitter, the parser and the oracle

- The constructor is the only way to make a line, so no emitter can write an undeclared field.
- `parse` requires the exact prefix and separator, a JSON object, no duplicate key, the exact
  key set, a valid value per field and the declared version. It refuses the whole line
  otherwise. An unknown name is counted, never retained.
- Tests read captured output through the same `parse`, and one golden file pins the bytes of
  each frozen line.
- A declared layout (key order, separators, the space after the prefix, the presence of
  `version`, the flush) reproduces today's bytes, so adopting a declaration changes nothing an
  operator or a reader sees.
- Names are unique by construction: a second declaration of a name fails at import.

### A refusal carries its category

A refusal type declares its categories as a `Literal` and takes one as its only argument; its
text is that category and nothing else. A line admits refusal types by naming them, and its
vocabulary is derived from theirs. The emit site therefore holds no allowlist to go stale, and a
refusal the line does not admit is a fault (below), never a generic label.
`scripts/preview_readiness.py:14` is already this shape; `scripts/preview_cutover.py:47` and
`scripts/native_work_profile.py:40` carry the category by convention only.

### Exactly one line at exit

An outcome line has no free-standing emit. It is written in one of two ways:

- at a guard, by a call that writes the line and returns the exception to raise, unchanged;
- around a region, by a scope whose exit writes the line.

The scope's exit never swallows the exception in flight and runs once. It takes the category
from a refusal, else from what the body recorded, else from the label the line declares for an
unclassified failure. A line must declare what cancellation and interruption record; there is
no default. The write resolves its stream when called and sends the line and its newline in
one call.

### A fault never changes control flow

- A value that fails its kind is never written. The field is emitted as null and the line gains
  a reserved `fault` count. Under tests the fault raises at the faulting statement, and a
  captured line with `fault` fails its gate.
- A write to a missing or broken stream is dropped and counted. It is not retried elsewhere and
  it cannot replace the refusal. `scripts/storage_process.py:132` and
  `providers/moonshine.py:158` already state this rule for their own sites.
- An interrupt during the write propagates only when no exception is already in flight.

### Versions and unknown fields

Every new line and record carries `version`. A line has one version, because its reader is the
same checkout; adding a label does not change it. A record may declare older versions it still
reads, each with its own declaration. An unknown field, a missing field, an unknown version or a
duplicate key refuses the whole line or file.

### Records use the same grammar and one write

- A content-free record is a declaration in the same field kinds, with named groups and bounded
  rows, a canonical encoding and a declared maximum size. One function writes it and one reads
  it; the reader is the declaration.
- A private file (a plan holding paths such as `scripts/preview_cutover.py:1368`, a file holding
  a process identifier such as `:2137`, a capture of model output, the saved conversation) is
  bytes with an owner. It shares the write and none of the grammar, and no evidence field can
  hold one.
- The write: a temporary in the destination directory, written, flushed and synced, then
  published either by replacement or by an exclusive link that refuses an existing name, with
  the temporary removed on every path. `integration/run_record.py:41` and
  `scripts/preview_cutover.py:1984` are the two existing models.
- The read: bounded, refusing a link or reparse point and a duplicate key, as
  `scripts/preview_cutover.py:1388` does today.

### Placement, amending ADR 0006 Move 1

ADR 0006 says evidence-marker names are defined once in `protocol`. This record proposes a new
**standard-library-only unit in the lowest tier** instead, holding the mechanism and every
package declaration.

- It must be the lowest tier: `conversation` holds 17 `print` calls, all of them markers.
- It cannot be inside `protocol`: `protocol/__init__.py` imports its events module, and so
  Pydantic, on import. `providers/kokoro_worker.py` and
  `integration/codex_access_credentials.py` are emitters that import only the standard library,
  and they run as an isolated worker and as a credential helper.
- Declarations sit in one directory so that one review, and `.github/CODEOWNERS`, covers the
  whole privacy vocabulary. Script-only lines are declared in `scripts/` with the same mechanism.
- File reading and writing sit in a shared-facts unit, which `conversation` may not import.
- A browser helper mirrors the grammar with string-literal types. No Python code parses browser
  console lines, so the two are bound by test, not generated.

Placing a unit is a design decision under `tests/test_import_direction.py`; this is decision 1.

## What this does not prevent

- A reviewer approving a label that is really content. One directory makes the review
  possible; nothing makes it certain.
- A count that reveals something: the length of private text, or an identifier stored as a
  number. A field-name lint and mandatory maxima narrow this.
- A digest of low-entropy input. Entropy cannot be checked.
- A plain `print`, or a prefix assembled from parts, outside the mechanism. A ratchet over
  output calls in the package narrows it; the static scan has the limit ADR 0006 states for
  computed imports.
- Catching a refusal and raising a vaguer one, or declaring one coarse label for several
  guards. The catalogue reports and ratchets the second; the first is review.
- A line from a process that is killed or calls `os._exit`. Its owner is the parent, which
  declares the exit-status vocabulary and emits for the child.
- Schema drift in files that carry content. The run-record restatement named in ADR 0006 is
  closed by a public reader beside its writer (Moves 3 and 4), not by this record.
- Tracebacks and exception messages printed as prose outside the channel. Steps 8 and 10
  address the known sites; an uncaught exception at top level remains.

## Migration

Each step ships alone. "Identical" means marker names and line bytes do not change.

| # | Step | Effect | Removes |
| --- | --- | --- | --- |
| 1 | Catalogue ratchet test: a scan bound by equality to committed tables of marker names and emit sites, non-literal strings reaching a payload, hand-written readers, and names read but never written | Identical; test only | Nothing; every later step removes entries |
| 2 | The mechanism, with its own tests, used by no product code; the unit is placed in a tier | Identical | Nothing |
| 3 | Declare the lines another process parses and move only their readers to `parse` | Identical for writers; readers become stricter (an undeclared or invalid line is no longer folded into a gate's output) | Four restated vocabularies, both generic regexes, the rehearsal's name list |
| 4 | Writers, one change per package unit, then `scripts/` by family | Identical per golden; a broken stream no longer replaces a refusal | The `_marker` helpers, the emitting context managers, the `try/finally/print` blocks |
| 5 | Tests read through one helper; marker literals in tests are ratcheted | Identical | 16 helpers and the inline idioms |
| 6 | Refusal types adopt the base | **Labelled change**: categories that were collapsed now appear | `_is_category_refusal`, the emit-site allowlists |
| 7 | One atomic write and one bounded read; content-free preview receipts become records | Stricter only | Seven write helpers and the field-by-field receipt checks |
| 8 | Remove free strings: class names, the wire category, model, version and directory fields | **Labelled change**, one line at a time | The entries of step 1's second table |
| 9 | Normalize lines that are not frozen: `version`, one layout, one discriminating key | **Labelled change** | The layout exceptions |
| 10 | Entry points end with one status line; markers for the prose-only refusals | **Labelled change** | `parser.error(str(exc))` and the prose in the LiveKit installer |
| 11 | Browser helper and its catalogue test | Identical | 25 hand-written templates |

Steps 1 to 3 are test-only or stricter-only and fit before desktop MVP acceptance. Steps 6 and
8 to 10 change what an operator sees and belong after it.

## Frozen contracts

Name, key set, key order and separators stay byte-identical until a labelled change updates the
reader or the document in the same commit.

- **Parsed by another process or script**: `voice-preview`, `voice-preview-host-failure`,
  `foreground-failure`, `kokoro-mcp-refused`, `native-credential-refused`,
  `presentation-credential-refused`, `lookup-presentation`, `ollama-prompt`, `real-hermes-gate`,
  `voice-tail`, `voice-archive`, `voice-archive-open`, `voice-archive-lease`, the five `-step`
  lines of the voice qualifiers and `hermes-continuity-ready`.
- **Recorded by the rehearsal**: the 36 names at `scripts/rehearse_desktop_mvp.py:84`. Their
  values are re-rendered only when they match the category regex, so a change of value shape
  drops the line from the record sheet without an error.
- **Named in operator documents**: 44 names, most of them in
  [the diagnostic runbook](../desktop-mvp-diagnostic.md) and
  [the bridge reference](../hermes-bridge.md).
- **Parsed by CI**: none directly. CI depends on markers only through the test suite.

About 100 names are in none of these sets and are held only by tests.

## Owner decisions

| # | Decision | Recommendation |
| --- | --- | --- |
| 1 | Place the mechanism and the declarations in a new standard-library-only lowest-tier unit, amending ADR 0006 Move 1, which names `protocol`? | Yes. The alternative makes `protocol` import lazily, which changes how the strictest contract is loaded |
| 2 | Declare every package line in that one directory, gated by code owners, and not beside each emitter? | Yes |
| 3 | No free strings at all: an unlisted exception type reads `other`, and a category a newer companion sends is a fault and is not printed? | Yes. The wire type stays open; only the marker stops echoing it |
| 4 | A value outside its vocabulary is written as null with a `fault` count in production and raises under tests? | Yes |
| 5 | Accept the frozen set above, and normalize the remaining lines after acceptance? | Yes |
| 6 | Are model tags and third-party version strings content for this purpose? | Yes: private files only, or a label from a pinned set |
| 7 | Keep `ports` in `voice-preview` as a named exception, and keep process identifiers out of everything except private files? | Yes to both |
| 8 | Every product entry point ends with exactly one status line, and argument-parser prose is for usage errors only? | Yes, after acceptance |
| 9 | Treat the step 1 ratchet as a consequential guard, proved by targeted mutations? | Yes. It guards a privacy boundary |
| 10 | Bind browser lines to the grammar by a test and a helper, not by generation? | Yes, at low priority |

## Not verified

- Nothing was built or run. The import-time check of annotations and the behaviour of strict
  type checking at 164 call sites are reasoned, not tried.
- Every count is syntactic and was taken at `25cabec`. The field lists behind the layout and
  key rows were extracted from syntax; of the 113 names with a computed field, about 40 were
  read in source.
- "Never spelled in a test" misses a test that reaches a marker through a constant.
- Who can read each marker's output. The record states where a line is written, not where that
  stream is captured or posted.
- What the messages behind `host_launcher.py:3749` and `scripts/local_livekit.py:170` can
  contain, and the objects the browser prints in its connection-recovery lines.
- Where `lookup-presentation` is written. Its only writer in the tree is a test fixture.
- Whether ten scripts that print markers and never import the package can import it where they
  run. One that cannot stays hand-written as a permanent, named ratchet entry.
- That one write per line removes the fragmented lines earlier fixes worked around.
- Windows behaviour of replacement while a reader holds the destination, and the absence of a
  directory sync there.
- Whether code-owner review is enforced on the default branch.
- Every "about" figure, and what each step removes, are estimates.
